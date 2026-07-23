"""
YOLOv11 / YOLOv8 person detector wrapper.
Handles model loading (PyTorch .pt or TensorRT .engine),
frame preprocessing, and bounding-box extraction for students in exam halls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Normalised bounding box [0, 1] for a detected person."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: int | None = None

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_pixel(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalised coords to pixel coords."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )


@dataclass
class DetectionResult:
    boxes: List[BoundingBox] = field(default_factory=list)
    frame_id: int = 0
    camera_id: str = ""
    inference_ms: float = 0.0


class YOLODetector:
    """
    Wraps Ultralytics YOLO for person detection.
    Supports:
      - Standard PyTorch .pt weights (CPU / CUDA)
      - TensorRT .engine weights (CUDA only, faster inference)
      - ByteTrack multi-object tracking for consistent track IDs
    """

    # COCO class index for "person"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str,
        use_tensorrt: bool = False,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        enable_tracking: bool = True,
    ) -> None:
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.device = device
        self.enable_tracking = enable_tracking
        self._model = None

        logger.info(
            "Initialising YOLODetector model=%s trt=%s device=%s",
            model_path, use_tensorrt, device,
        )
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            logger.info("YOLO model loaded successfully on %s", self.device)
        except ImportError:
            logger.error("ultralytics package not installed. Run: pip install ultralytics")
            raise
        except Exception as exc:
            logger.error("Failed to load YOLO model from %s: %s", self.model_path, exc)
            raise

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        camera_id: str = "",
    ) -> DetectionResult:
        """
        Run inference on a single BGR frame (OpenCV format).

        Args:
            frame: H×W×3 BGR numpy array.
            frame_id: Sequential frame counter.
            camera_id: Source camera identifier.

        Returns:
            DetectionResult with normalised bounding boxes.
        """
        import time

        if self._model is None:
            raise RuntimeError("YOLO model not loaded")

        t0 = time.perf_counter()

        if self.enable_tracking:
            results = self._model.track(
                frame,
                persist=True,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                classes=[self.PERSON_CLASS_ID],
                verbose=False,
            )
        else:
            results = self._model.predict(
                frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                classes=[self.PERSON_CLASS_ID],
                verbose=False,
            )

        inference_ms = (time.perf_counter() - t0) * 1000.0

        boxes: list[BoundingBox] = []
        if results and len(results) > 0:
            r = results[0]
            h, w = frame.shape[:2]

            for i, box in enumerate(r.boxes):
                xyxyn = box.xyxyn[0].cpu().numpy()  # normalised [x1,y1,x2,y2]
                conf = float(box.conf[0].cpu())
                track_id = None
                if self.enable_tracking and box.id is not None:
                    track_id = int(box.id[0].cpu())

                boxes.append(
                    BoundingBox(
                        x1=float(xyxyn[0]),
                        y1=float(xyxyn[1]),
                        x2=float(xyxyn[2]),
                        y2=float(xyxyn[3]),
                        confidence=conf,
                        track_id=track_id,
                    )
                )

        return DetectionResult(
            boxes=boxes,
            frame_id=frame_id,
            camera_id=camera_id,
            inference_ms=inference_ms,
        )

    def warmup(self, img_size: tuple[int, int] = (640, 640)) -> None:
        """Run a dummy inference pass to warm up the GPU kernel."""
        dummy = np.zeros((*img_size, 3), dtype=np.uint8)
        self.detect(dummy, frame_id=-1, camera_id="warmup")
        logger.info("YOLODetector GPU warmup complete")
