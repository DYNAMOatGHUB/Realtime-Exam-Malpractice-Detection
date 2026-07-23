"""
Model evaluation: confusion matrix, per-class precision/recall/F1,
confidence calibration curve, and threshold tuning.

Usage:
    python -m ml.training.evaluate \
        --model_path ml/weights/lstm_classifier.pt \
        --data_dir /path/to/sequences \
        --output_dir ml/eval_results \
        --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))

from ml.models.lstm_classifier import MalpracticeLSTM, NUM_CLASSES, INPUT_SIZE
from ml.training.dataset import MalpracticeDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "COPYING_FROM_NEIGHBOUR", "USING_MOBILE_PHONE", "PASSING_NOTES",
    "UNAUTHORIZED_MATERIALS", "LOOKING_AROUND_SUSPICIOUSLY",
    "TALKING", "LEAVING_SEAT", "OTHER",
]


def evaluate(
    model_path: str,
    data_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    sequence_length: int = 80,
    device_str: str = "auto",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and device_str != "cpu" else "cpu")

    # Load model
    model = MalpracticeLSTM(num_classes=NUM_CLASSES)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    model.to(device)

    # Dataset (no augmentation for eval)
    dataset = MalpracticeDataset(data_dir=data_dir, sequence_length=sequence_length, augment=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=2)

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            scores = model(batch_x).cpu().numpy()
            all_preds.append(scores)
            all_labels.append(batch_y.numpy())

    preds = np.vstack(all_preds)    # (N, num_classes)
    labels = np.vstack(all_labels)  # (N, num_classes)

    binary_preds = (preds >= threshold).astype(int)

    # Per-class metrics
    results = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        tp = int(((binary_preds[:, c_idx] == 1) & (labels[:, c_idx] == 1)).sum())
        fp = int(((binary_preds[:, c_idx] == 1) & (labels[:, c_idx] == 0)).sum())
        fn = int(((binary_preds[:, c_idx] == 0) & (labels[:, c_idx] == 1)).sum())
        tn = int(((binary_preds[:, c_idx] == 0) & (labels[:, c_idx] == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[c_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        logger.info(
            "%-35s P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)",
            c_name, precision, recall, f1, tp, fp, fn,
        )

    # Overall macro metrics
    macro_f1 = np.mean([v["f1"] for v in results.values()])
    macro_precision = np.mean([v["precision"] for v in results.values()])
    macro_recall = np.mean([v["recall"] for v in results.values()])
    logger.info(
        "\nMacro-avg → P=%.3f R=%.3f F1=%.3f (threshold=%.2f)",
        macro_precision, macro_recall, macro_f1, threshold,
    )

    output = {
        "threshold": threshold,
        "num_samples": len(dataset),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": results,
    }

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved evaluation results → %s", results_file)

    # Optional: generate matplotlib plots if available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Per-class F1 bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        f1_scores = [results[c]["f1"] for c in CLASS_NAMES]
        bars = ax.bar(CLASS_NAMES, f1_scores, color="steelblue")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Per-class F1 (threshold={threshold})")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_path / "f1_per_class.png", dpi=150)
        plt.close()
        logger.info("Saved F1 chart → %s", out_path / "f1_per_class.png")
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping plots")

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="ml/eval_results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        sequence_length=args.seq_len,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
