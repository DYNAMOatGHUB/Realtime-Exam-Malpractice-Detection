"""
Train a Normal Pose Profile using only "normal act" images.

Pipeline:
  1. Load all images from ml/datasets/ExamCheatingDataset/.../train/normal act/
  2. Run yolov8s-pose.pt to extract 17 COCO keypoints per person
  3. Normalize keypoints to [0,1] relative to bounding box
  4. Fit PCA on all collected keypoint vectors
  5. Compute 95th-percentile reconstruction error as anomaly threshold
  6. Save the fitted model to ml/weights/normal_pose_profile.pkl

Usage:
    cd /home/techpark-11/DYNAMO/exam-anomally-detection
    source backend/venv/bin/activate
    python ml/training/train_pose_profile.py
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────
ML_DIR      = Path(__file__).resolve().parent.parent
DATASET_DIR = ML_DIR / "datasets" / "ExamCheatingDataset" / "ExamCheatingDataset" / "train" / "normal act"
WEIGHTS_DIR = ML_DIR / "weights"
OUTPUT_FILE = WEIGHTS_DIR / "normal_pose_profile.pkl"

# ── Config ───────────────────────────────────────────────────────
PERCENTILE      = 95        # anomaly threshold percentile
PCA_COMPONENTS  = 16        # number of PCA components
POSE_MODEL_NAME = "yolov8s-pose.pt"
CONF_THRESHOLD  = 0.25      # minimum keypoint confidence to include
IMGSZ           = 640

# 17 COCO keypoints, each with (x, y, conf) → 51 values per person
# We use only x and y (34 values) after normalisation
N_KEYPOINTS = 17


def _normalise_keypoints(kps: np.ndarray, box: list[float]) -> np.ndarray | None:
    """
    Flatten 17 COCO keypoints and normalise x/y relative to bounding box.
    Returns a 34-element vector or None if not enough visible keypoints.

    kps  : (17, 3) array of [x, y, conf]
    box  : [x1, y1, x2, y2] bounding box in pixels
    """
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    vec = []
    visible = 0
    for kp in kps:
        kx, ky, kc = float(kp[0]), float(kp[1]), float(kp[2])
        if kc >= CONF_THRESHOLD:
            # Normalise to box: range roughly [0, 1]
            nx = (kx - x1) / bw
            ny = (ky - y1) / bh
            visible += 1
        else:
            # Use centre of box as placeholder for invisible keypoints
            nx = 0.5
            ny = 0.5
        vec.extend([nx, ny])

    # Require at least 5 visible keypoints for a usable sample
    if visible < 5:
        return None
    return np.array(vec, dtype=np.float32)


def extract_keypoints(image_paths: list[Path]) -> np.ndarray:
    """Run pose model on all images in batches, return matrix of normalised keypoint vectors."""
    import torch
    import cv2
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("Using device: %s (%s)", "GPU" if device == 0 else "CPU", device_name)
    logger.info("Loading pose model: %s", POSE_MODEL_NAME)

    model = YOLO(POSE_MODEL_NAME)

    # Load and convert all images to BGR first
    logger.info("Loading %d images …", len(image_paths))
    frames = []
    valid_paths = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        frames.append(img)
        valid_paths.append(img_path)

    logger.info("Loaded %d valid images. Running batch inference …", len(frames))

    # Run all images as one batch — maximises GPU throughput
    BATCH = 32  # RTX 5070 can handle large batches at 640px
    vectors = []

    for batch_start in range(0, len(frames), BATCH):
        batch = frames[batch_start : batch_start + BATCH]
        batch_num = batch_start // BATCH + 1
        total_batches = (len(frames) + BATCH - 1) // BATCH
        logger.info("  Batch %d / %d …", batch_num, total_batches)

        results = model.predict(
            source=batch,
            imgsz=IMGSZ,
            conf=0.10,
            device=device,
            verbose=False,
            stream=False,
        )

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            boxes   = result.boxes.xyxy.cpu().numpy()
            kps_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

            for j in range(len(boxes)):
                box = boxes[j].tolist()
                kps = kps_all[j]
                if len(kps) < N_KEYPOINTS:
                    continue
                vec = _normalise_keypoints(kps, box)
                if vec is not None:
                    vectors.append(vec)

    logger.info("Extracted %d pose vectors from %d images", len(vectors), len(frames))
    if not vectors:
        raise RuntimeError("No pose vectors extracted. Check the dataset path and images.")
    return np.stack(vectors)


def fit_pca_model(X: np.ndarray) -> dict:
    """Fit PCA and compute anomaly threshold."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    logger.info("Fitting StandardScaler …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Fitting PCA with %d components on %d samples …", PCA_COMPONENTS, len(X))
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    pca.fit(X_scaled)

    # Compute reconstruction errors on training set
    X_reduced   = pca.transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)
    errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    threshold = float(np.percentile(errors, PERCENTILE))
    mean_err  = float(np.mean(errors))
    logger.info(
        "Reconstruction error — mean: %.4f  |  %dth-percentile threshold: %.4f",
        mean_err, PERCENTILE, threshold,
    )

    explained = float(np.sum(pca.explained_variance_ratio_) * 100)
    logger.info("PCA explains %.1f%% of variance with %d components", explained, PCA_COMPONENTS)

    return {
        "scaler":    scaler,
        "pca":       pca,
        "threshold": threshold,
        "percentile": PERCENTILE,
        "n_train":   len(X),
        "explained_variance_pct": explained,
    }


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        logger.error("Dataset directory not found: %s", DATASET_DIR)
        sys.exit(1)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in sorted(DATASET_DIR.iterdir()) if p.suffix.lower() in image_exts]
    logger.info("Found %d images in: %s", len(image_paths), DATASET_DIR)

    if not image_paths:
        logger.error("No images found! Check the dataset path.")
        sys.exit(1)

    # Step 1: Extract pose keypoints
    X = extract_keypoints(image_paths)

    # Step 2: Fit PCA model
    model_data = fit_pca_model(X)

    # Step 3: Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(model_data, f)
    logger.info("✅ Pose profile saved to: %s", OUTPUT_FILE)
    logger.info("   Trained on %d pose samples", model_data["n_train"])
    logger.info("   Anomaly threshold (%.0fth pct): %.4f", model_data["percentile"], model_data["threshold"])


if __name__ == "__main__":
    main()
