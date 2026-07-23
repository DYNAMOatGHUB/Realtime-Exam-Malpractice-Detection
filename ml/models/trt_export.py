"""
TensorRT export script for YOLO and LSTM models.
Run this once on the target GPU to generate .engine files for maximum inference speed.

Usage:
    # Export YOLO detection model
    python -m ml.models.trt_export --model yolo --input ml/weights/yolov11n.pt

    # Export pose estimation model
    python -m ml.models.trt_export --model pose --input ml/weights/yolov11n-pose.pt

    # Export LSTM classifier (via ONNX → TRT)
    python -m ml.models.trt_export --model lstm --input ml/weights/lstm_classifier.pt

Requirements:
    - CUDA toolkit matching TensorRT version
    - tensorrt, pycuda packages installed
    - Sufficient GPU VRAM (RTX 5070 12GB is more than adequate)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def export_yolo_to_trt(model_path: str, imgsz: int = 640, batch_size: int = 1) -> str:
    """Export a YOLO model to TensorRT using Ultralytics built-in export."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    logger.info("Loading YOLO model from %s", model_path)
    model = YOLO(model_path)

    logger.info("Exporting to TensorRT (imgsz=%d, batch=%d)...", imgsz, batch_size)
    export_path = model.export(
        format="engine",
        imgsz=imgsz,
        batch=batch_size,
        half=True,          # FP16 for RTX 5070
        device=0,
        simplify=True,
    )
    logger.info("TensorRT engine exported → %s", export_path)
    return str(export_path)


def export_lstm_to_onnx_and_trt(
    model_path: str,
    output_dir: str,
    sequence_length: int = 80,
    batch_size: int = 1,
) -> str:
    """Export LSTM to ONNX, then convert to TensorRT engine."""
    import torch
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from ml.models.lstm_classifier import MalpracticeLSTM, INPUT_SIZE, NUM_CLASSES

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_path = output_path / "lstm_classifier.onnx"
    engine_path = output_path / "lstm_classifier.engine"

    # Load model
    model = MalpracticeLSTM(num_classes=NUM_CLASSES)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    # Dummy input: (batch, seq_len, input_size)
    dummy = torch.randn(batch_size, sequence_length, INPUT_SIZE)

    # Export to ONNX
    logger.info("Exporting LSTM to ONNX → %s", onnx_path)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["keypoints"],
        output_names=["confidence_scores"],
        dynamic_axes={
            "keypoints": {0: "batch_size"},
            "confidence_scores": {0: "batch_size"},
        },
    )
    logger.info("ONNX export complete")

    # Convert ONNX → TensorRT
    try:
        import subprocess
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            "--best",
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("trtexec failed:\n%s", result.stderr)
        else:
            logger.info("TensorRT engine saved → %s", engine_path)
    except FileNotFoundError:
        logger.warning(
            "trtexec not found in PATH. Install TensorRT and add trtexec to PATH. "
            "ONNX model saved at %s — convert manually.", onnx_path
        )

    return str(engine_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to TensorRT")
    parser.add_argument(
        "--model", choices=["yolo", "pose", "lstm"], required=True,
        help="Which model to export"
    )
    parser.add_argument("--input", required=True, help="Path to .pt weights file")
    parser.add_argument("--output_dir", default="ml/weights", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=80, help="LSTM sequence length")
    args = parser.parse_args()

    if args.model in ("yolo", "pose"):
        export_yolo_to_trt(args.input, imgsz=args.imgsz, batch_size=args.batch)
    elif args.model == "lstm":
        export_lstm_to_onnx_and_trt(
            args.input, args.output_dir,
            sequence_length=args.seq_len,
            batch_size=args.batch,
        )


if __name__ == "__main__":
    main()
