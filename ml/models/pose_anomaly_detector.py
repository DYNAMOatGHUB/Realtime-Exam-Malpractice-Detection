"""
PoseAnomalyDetector — Singleton that loads the PCA normal-pose profile
and scores any person's pose keypoints against it.

Usage:
    from ml.models.pose_anomaly_detector import PoseAnomalyDetector
    detector = PoseAnomalyDetector.get()
    state, score = detector.classify(keypoints_34_values)
    # state → "NORMAL" or "CHEATING"
    # score → float 0.0–1.0 (anomaly confidence)
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ML_DIR      = Path(__file__).resolve().parent.parent
PROFILE_PATH = ML_DIR / "weights" / "normal_pose_profile.pkl"

STATE_NORMAL   = "NORMAL"
STATE_CHEATING = "CHEATING"

COLOR_NORMAL   = (30, 200, 30)    # green  (R, G, B)
COLOR_CHEATING = (200, 10, 10)    # red

# Minimum visible keypoints needed to make a decision
MIN_VISIBLE_KP = 5
CONF_THRESHOLD  = 0.25


class PoseAnomalyDetector:
    """
    Singleton PCA-based normal-pose anomaly detector.
    Call PoseAnomalyDetector.get() to obtain the shared instance.
    """
    _instance: Optional["PoseAnomalyDetector"] = None

    def __init__(self) -> None:
        self._loaded   = False
        self._scaler   = None
        self._pca      = None
        self._threshold = 1.0      # fallback — everything is normal
        self._load()

    @classmethod
    def get(cls) -> "PoseAnomalyDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Force reload — useful after re-training during development."""
        cls._instance = None

    @property
    def ready(self) -> bool:
        return self._loaded

    # ── Internal ────────────────────────────────────────────────

    def _load(self) -> None:
        profile_path = Path(os.environ.get("POSE_PROFILE_PATH", str(PROFILE_PATH)))
        if not profile_path.exists():
            logger.warning(
                "⚠️  Normal pose profile not found at %s. "
                "Run ml/training/train_pose_profile.py first.",
                profile_path,
            )
            return

        try:
            with open(profile_path, "rb") as f:
                data = pickle.load(f)
            self._scaler    = data["scaler"]
            self._pca       = data["pca"]
            self._threshold = data["threshold"]
            self._loaded    = True
            logger.info(
                "✅ Normal pose profile loaded | threshold=%.4f | trained_on=%d samples",
                self._threshold, data.get("n_train", 0),
            )
        except Exception as exc:
            logger.error("❌ Failed to load pose profile: %s", exc)

    # ── Public API ───────────────────────────────────────────────

    def _normalise_kp(self, keypoints: List[List[float]], box: Optional[list] = None) -> Optional[np.ndarray]:
        """
        Convert raw keypoints (17 × [x, y, conf]) to a normalised 34-element vector.
        If box is provided, normalise relative to the bounding box.
        Otherwise normalise to [0, 1] relative to the max extent.
        """
        if len(keypoints) < 17:
            return None

        xs = [kp[0] for kp in keypoints]
        ys = [kp[1] for kp in keypoints]

        if box:
            x1, y1, x2, y2 = box
        else:
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)

        vec     = []
        visible = 0
        for kp in keypoints:
            kx, ky = float(kp[0]), float(kp[1])
            kc     = float(kp[2]) if len(kp) > 2 else 1.0
            if kc >= CONF_THRESHOLD:
                visible += 1
                nx = (kx - x1) / bw
                ny = (ky - y1) / bh
            else:
                nx, ny = 0.5, 0.5   # placeholder for invisible joints
            vec.extend([nx, ny])

        if visible < MIN_VISIBLE_KP:
            return None

        return np.array(vec, dtype=np.float32)

    def score(self, keypoints: List[List[float]], box: Optional[list] = None) -> float:
        """
        Return a raw reconstruction error for the given pose.
        Higher = more anomalous.
        Returns -1.0 if the profile is not loaded or keypoints are insufficient.
        """
        if not self._loaded:
            return -1.0
        vec = self._normalise_kp(keypoints, box)
        if vec is None:
            return -1.0
        try:
            vec_scaled  = self._scaler.transform(vec.reshape(1, -1))
            vec_reduced = self._pca.transform(vec_scaled)
            vec_recon   = self._pca.inverse_transform(vec_reduced)
            error = float(np.mean((vec_scaled - vec_recon) ** 2))
            return error
        except Exception as exc:
            logger.debug("score() error: %s", exc)
            return -1.0

    def classify(self, keypoints: List[List[float]], box: Optional[list] = None) -> Tuple[str, float]:
        """
        Classify a person's pose as NORMAL or CHEATING.

        Returns:
            (state, confidence)
            state      → "NORMAL" or "CHEATING"
            confidence → float 0.0–1.0 proportional to anomaly severity
        """
        if not self._loaded:
            # No profile loaded → treat everyone as normal
            return STATE_NORMAL, 0.0

        error = self.score(keypoints, box)
        if error < 0:
            return STATE_NORMAL, 0.0

        if error <= self._threshold:
            # How normal (0 = perfectly normal, approaching threshold)
            conf = round(error / self._threshold, 3)
            return STATE_NORMAL, conf
        else:
            # Anomaly: confidence scales with how far above threshold
            # Cap at 1.0
            conf = round(min(1.0, (error - self._threshold) / self._threshold + 0.5), 3)
            return STATE_CHEATING, conf

    def color_for(self, state: str) -> tuple:
        """Return the RGB draw color for a given state."""
        return COLOR_NORMAL if state == STATE_NORMAL else COLOR_CHEATING
