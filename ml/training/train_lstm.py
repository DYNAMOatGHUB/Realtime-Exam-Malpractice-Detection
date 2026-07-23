"""
Training script for the MalpracticeLSTM/GRU classifier.

Usage:
    python -m ml.training.train_lstm \
        --data_dir /path/to/sequences \
        --output_dir /app/ml/weights \
        --epochs 50 \
        --batch_size 32 \
        --lr 1e-3

Data format:
    Each sample is a .npy file of shape (seq_len, 51) containing
    normalised keypoint vectors (x, y, conf) × 17 keypoints.
    Labels are stored as a parallel .json file or in a manifest CSV:
        {
          "labels": [0, 0, 1, 0, 0, 0, 0, 0],   // one-hot multi-label for NUM_CLASSES
          "source": "camera_01_2024_01_15_session_2"
        }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).parents[2]))

from ml.models.lstm_classifier import MalpracticeLSTM, INPUT_SIZE, NUM_CLASSES
from ml.training.dataset import MalpracticeDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.15,
    sequence_length: int = 80,
    device_str: str = "auto",
    resume_from: str | None = None,
) -> None:
    # ── Device ────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Training on device: %s", device)

    # ── Dataset ───────────────────────────────────────────────────
    dataset = MalpracticeDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
    )
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    logger.info("Dataset: %d train / %d val samples", n_train, n_val)

    # ── Model ─────────────────────────────────────────────────────
    model = MalpracticeLSTM(num_classes=NUM_CLASSES).to(device)

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Resumed from %s at epoch %d", resume_from, start_epoch)

    # ── Optimiser / Loss ──────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs - start_epoch,
    )
    # Binary cross-entropy: independent per-class confidence
    criterion = nn.BCELoss()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item()
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            "Epoch %3d/%3d | train_loss=%.4f | val_loss=%.4f | lr=%.6f",
            epoch + 1, epochs,
            avg_train_loss, avg_val_loss,
            scheduler.get_last_lr()[0],
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = output_path / "lstm_classifier.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {
                        "num_classes": NUM_CLASSES,
                        "input_size": INPUT_SIZE,
                        "sequence_length": sequence_length,
                    },
                },
                ckpt_path,
            )
            logger.info("  ✓ Saved best checkpoint → %s", ckpt_path)

    logger.info("Training complete. Best val_loss: %.4f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MalpracticeLSTM classifier")
    parser.add_argument("--data_dir", required=True, help="Path to sequence data directory")
    parser.add_argument("--output_dir", default="ml/weights", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seq_len", type=int, default=80, help="Frames per sequence (fps × window_s)")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        sequence_length=args.seq_len,
        device_str=args.device,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
