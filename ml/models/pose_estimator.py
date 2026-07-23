"""
YOLOv11-Pose skeletal keypoint extractor.
Extracts 17 COCO keypoints per detected person bounding box.
Key keypoints used for malpractice detection:
  - 0: Nose (head orientation)
  - 5,6: Shoulders (body twist)
  - 7,8: Elbows
  - 9,10: Wrists (hand position — critical for note-passing / phone use)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO 17-keypoint indices
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Keypoints most relevant to malpractice detection
CRITICAL_KEYPOINTS = {
    0: "nose",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
}

NUM_KEYPOINTS = 17
KEYPOINT_DIM = 3  # x, y, confidence


@dataclass
class Keypoint:
    x: float           # normalised [0, 1]
    y: float           # normalised [0, 1]
    confidence: float  # visibility score [0, 1]
    name: str = ""

    @property
    def visible(self) -> bool:
        return self.confidence > 0.5


@dataclass
class PoseResult:
    """Keypoints for a single detected person."""
    track_id: int | None
    keypoints: List[Keypoint] = field(default_factory=list)
    raw_array: np.ndarray | None = None  # shape (17, 3) — x, y, conf

    def get_critical_vector(self) -> np.ndarray:
        """
        Return a flattened feature vector of only critical keypoints.
        Shape: (len(CRITICAL_KEYPOINTS) * 3,) = (7*3,) = 21 features per frame.
        """
        indices = sorted(CRITICAL_KEYPOINTS.keys())
        vectors = []
        for idx in indices:
            if idx < len(self.keypoints):
                kp = self.keypoints[idx]
                vectors.extend([kp.x, kp.y, kp.confidence])
            else:
                vectors.extend([0.0, 0.0, 0.0])
        return np.array(vectors, dtype=np.float32)

    def get_full_vector(self) -> np.ndarray:
        """Return flattened full keypoint vector. Shape: (51,) = 17 * 3."""
        if self.raw_array is not None:
            return self.raw_array.flatten()
        vectors = []
        for kp in self.keypoints:
            vectors.extend([kp.x, kp.y, kp.confidence])
        return np.array(vectors, dtype=np.float32)


class PoseEstimator:
    """
    YOLOv11-Pose wrapper for multi-person skeletal keypoint extraction.
    Pairs each pose with a track ID from the YOLODetector for temporal
    consistency across the LSTM sliding window.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        self.device = device
        self._model = None

        logger.info("Initialising PoseEstimator model=%s device=%s", model_path, device)
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            logger.info("Pose model loaded on %s", self.device)
        except Exception as exc:
            logger.error("Failed to load pose model: %s", exc)
            raise

    def estimate(
        self,
        frame: np.ndarray,
        person_boxes: list | None = None,
    ) -> List[PoseResult]:
        """
        Run pose estimation on a full frame.

        Args:
            frame: H×W×3 BGR numpy array.
            person_boxes: Optional list of BoundingBox (not used directly;
                          YOLO-Pose runs end-to-end). Provided for logging.

        Returns:
            List of PoseResult, one per detected person.
        """
        if self._model is None:
            raise RuntimeError("Pose model not loaded")

        results = self._model.predict(
            frame,
            conf=self.conf_thresh,
            verbose=False,
        )

        pose_results: list[PoseResult] = []

        if not results or len(results) == 0:
            return pose_results

        r = results[0]
        h, w = frame.shape[:2]

        if r.keypoints is None:
            return pose_results

        kp_data = r.keypoints.data  # shape: (N_persons, 17, 3)
        if kp_data is None:
            return pose_results

        # Track IDs — may be None if tracking wasn't run on this frame
        track_ids: list[int | None] = []
        if r.boxes is not None and r.boxes.id is not None:
            track_ids = [int(tid) for tid in r.boxes.id.cpu().numpy()]
        else:
            track_ids = [None] * len(kp_data)

        for person_idx, (kp_array, tid) in enumerate(zip(kp_data.cpu().numpy(), track_ids)):
            # kp_array: (17, 3) — pixel x, pixel y, confidence
            keypoints = []
            for idx, (px, py, conf) in enumerate(kp_array):
                keypoints.append(
                    Keypoint(
                        x=float(px) / w,
                        y=float(py) / h,
                        confidence=float(conf),
                        name=KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kp_{idx}",
                    )
                )

            # Build normalised raw array
            raw = np.array(
                [[kp.x, kp.y, kp.confidence] for kp in keypoints],
                dtype=np.float32,
            )

            pose_results.append(PoseResult(track_id=tid, keypoints=keypoints, raw_array=raw))

        return pose_results

    def warmup(self, img_size: tuple[int, int] = (640, 640)) -> None:
        dummy = np.zeros((*img_size, 3), dtype=np.uint8)
        self.estimate(dummy)
        logger.info("PoseEstimator GPU warmup complete")
