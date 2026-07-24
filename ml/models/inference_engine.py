"""
ExamGuard AI — Inference Engine
================================
Supports two model types:
  1. YOLOv8 Classifier (.pt with -cls suffix trained)
     → Labels the entire frame with a class (CHEATING, GIVING_CODE, etc.)
     → Used when dataset has class folders (your ExamCheatingDataset)
     → Output: frame-level class + confidence (no bounding boxes)

  2. YOLOv8 Detector (.pt trained on annotated bounding boxes)
     → Detects specific people and localises the anomaly
     → Requires Roboflow bounding box annotations
     → Output: boxes + class + confidence

The engine auto-detects the model type on load.

Place your trained weights at:
    ml/weights/exam_anomaly_classifier.pt   ← classification model
    ml/weights/exam_anomaly_best.pt         ← detection model (optional upgrade)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Your ExamCheatingDataset classes ──────────────────────────────────────────
# Maps the class names from your dataset to display names used in the UI.
# Adjust this if your folder names differ.
EXAM_CLASS_MAP: Dict[str, str] = {
    "normal act":     "NORMAL",
    "looking friend": "LOOKING_AT_FRIEND",
    "giving object":  "GIVING_OBJECT",
    "giving code":    "GIVING_CODE",
    "cheating":       "CHEATING",
    "NORMAL":            "NORMAL",
    "LOOKING_AT_FRIEND": "LOOKING_AT_FRIEND",
    "GIVING_OBJECT":     "GIVING_OBJECT",
    "GIVING_CODE":       "GIVING_CODE",
    "CHEATING":          "CHEATING",
    
    # YOLO COCO fallback mappings for the base model
    "cell phone":        "USING_MOBILE_PHONE",
    "book":              "UNAUTHORIZED_MATERIALS",
    "laptop":            "UNAUTHORIZED_MATERIALS",
}

# Classes that count as an anomaly (everything except NORMAL)
ANOMALY_CLASSES = {
    "LOOKING_AT_FRIEND", "GIVING_OBJECT", "GIVING_CODE", "CHEATING",
    "USING_MOBILE_PHONE", "UNAUTHORIZED_MATERIALS"
}

# Visual colour per class (R, G, B)
CLASS_COLORS: Dict[str, tuple] = {
    "NORMAL":                 (30, 200, 30),    # green (should never show a box)
    "LOOKING_AT_FRIEND":      (255, 60,  30),   # red-orange
    "GIVING_OBJECT":          (255, 130, 20),   # orange
    "GIVING_CODE":            (220, 30,  30),   # red
    "CHEATING":               (200, 10,  10),   # deep red
    "USING_MOBILE_PHONE":     (255, 0,   255),  # magenta
    "UNAUTHORIZED_MATERIALS": (255, 165, 0),    # orange

    # Legacy / detection model names
    "COPYING":           (255, 30,  30),
    "MOBILE_PHONE":      (220, 60,  30),
    "PASSING_NOTES":     (255, 110, 20),
    "UNAUTH_MATERIAL":   (200, 20,  20),
    "SUSPICIOUS":        (255, 40,  40),
    "TALKING":           (240, 120, 10),
    "LEAVING_SEAT":      (255, 170, 20),
    "OTHER":             (180, 10,  10),
}

DEFAULT_COLOR = (239, 68, 68)

# ── Weight file search order ───────────────────────────────────────────────────
ML_DIR  = Path(__file__).resolve().parent.parent  # ml/
WEIGHTS = ML_DIR / "weights"

WEIGHT_CANDIDATES = [
    "exam_anomaly_classifier.pt",   # classification model (your dataset)
    "exam_anomaly_best.pt",         # detection model (if upgraded)
    "exam_anomaly.pt",
    "best.pt",
    "yolov8n.pt",                   # Default base model
]


def _normalise_class(raw_name: str) -> str:
    """Convert any raw class name to the system canonical name."""
    return EXAM_CLASS_MAP.get(raw_name, raw_name.upper().replace(" ", "_"))


class ExamAnomalyEngine:
    """
    YOLOv8 inference engine for exam anomaly detection.

    Auto-detects model type (classifier vs detector) from the loaded weights.
    Call ExamAnomalyEngine.get() to obtain the singleton instance.
    """

    _instance: Optional["ExamAnomalyEngine"] = None

    def __init__(self) -> None:
        self.model       = None
        self.model_path  = None
        self.model_type  = "unknown"   # "classifier" | "detector" | "onnx"
        self.class_names: List[str] = []
        self._loaded     = False
        self._load()

    @classmethod
    def get(cls) -> "ExamAnomalyEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_weights(self) -> Optional[Path]:
        # Try both ML_MODEL_PATH and YOLO_MODEL_PATH
        env = os.getenv("ML_MODEL_PATH", "") or os.getenv("YOLO_MODEL_PATH", "")
        if env and Path(env).exists():
            return Path(env)
        for name in WEIGHT_CANDIDATES:
            p = WEIGHTS / name
            if p.exists():
                return p

        # Also check for a classes JSON to determine type
        return None

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        path = self._find_weights()
        if path is None:
            logger.warning(
                "⚠️  No model found in %s. Candidates: %s. "
                "System will use motion-proxy detection until model is present.",
                WEIGHTS, WEIGHT_CANDIDATES,
            )
            return

        self.model_path = path
        suffix = path.suffix.lower()

        try:
            if suffix == ".onnx":
                self._load_onnx(path)
            else:
                self._load_pytorch(path)

            # Load companion JSON if available
            json_path = path.with_suffix(".json").parent / "exam_anomaly_classes.json"
            if json_path.exists():
                with open(json_path) as f:
                    info = json.load(f)
                if "names" in info:
                    self.class_names = [_normalise_class(n) for n in info["names"]]

            self._loaded = True
            logger.info(
                "✅ Model loaded: %s | type=%s | classes=%s",
                path.name, self.model_type, self.class_names,
            )
        except Exception as exc:
            logger.error("❌ Model load failed (%s): %s", path, exc)

    def _load_pytorch(self, path: Path) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(path))

        # Detect model type from task field
        task = getattr(self.model, "task", "") or ""
        if "classify" in task or "cls" in path.name.lower():
            self.model_type = "classifier"
            # ── Two-Stage Pipeline: Load Base Detector ──
            try:
                self.person_detector = YOLO("yolov8n.pt")
                logger.info("✅ Base person detector loaded for two-stage inference.")
            except Exception as e:
                logger.error("Failed to load base person detector: %s", e)
                self.person_detector = None
        else:
            self.model_type = "detector"

        # Class names
        if hasattr(self.model, "names") and self.model.names:
            names = self.model.names
            raw   = [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else list(names)
            self.class_names = [_normalise_class(n) for n in raw]

    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model      = ort.InferenceSession(str(path), sess_options=opts)
        self.model_type = "onnx"

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._loaded and self.model is not None

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.40) -> List[dict]:
        """
        Run inference on a BGR numpy frame.

        Returns a list of detection dicts. Each dict:
        {
            "class":      "CHEATING",           ← normalised class name
            "confidence": 0.87,
            "bbox":       [x1, y1, x2, y2],     ← pixel coords
                          For classifiers: approximated full-frame region
            "color":      (220, 30, 30),
            "is_anomaly": True,
            "model_type": "classifier",
        }
        """
        if not self.ready:
            return []
        try:
            if self.model_type == "classifier":
                return self._infer_classifier(frame, conf_threshold)
            elif self.model_type == "detector":
                return self._infer_detector(frame, conf_threshold)
            else:
                return []
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            return []

    def infer_jpeg(self, jpeg_bytes: bytes, conf_threshold: float = 0.40) -> List[dict]:
        """Run inference directly on JPEG bytes."""
        if not self.ready:
            return []
        try:
            import cv2
            arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return []
            return self.infer(frame, conf_threshold)
        except Exception as exc:
            logger.error("infer_jpeg error: %s", exc)
            return []

    # ── Classifier inference ──────────────────────────────────────────────────

    def _infer_classifier(self, frame: np.ndarray, conf_threshold: float) -> List[dict]:
        """
        Two-stage inference: 
        1. Detect persons using yolov8n.pt.
        2. Crop each person and classify using the custom model.
        """
        detections = []
        
        # 1. Fallback to full frame if no detector
        if getattr(self, "person_detector", None) is None:
            results = self.model.predict(source=frame, verbose=False, stream=False)
            for result in results:
                if not hasattr(result, "probs") or result.probs is None: continue
                probs = result.probs
                cls_id = int(probs.top1)
                conf = float(probs.top1conf)
                if conf < conf_threshold: continue
                
                raw_name = self.model.names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                cls_name = _normalise_class(raw_name)
                if cls_name == "NORMAL": continue
                
                h, w = frame.shape[:2]
                pad_x, pad_y = int(w * 0.15), int(h * 0.10)
                detections.append({
                    "class": cls_name, "confidence": round(conf, 3),
                    "bbox": [pad_x, pad_y, w - pad_x, h - int(h * 0.05)],
                    "color": CLASS_COLORS.get(cls_name, DEFAULT_COLOR),
                    "is_anomaly": cls_name in ANOMALY_CLASSES,
                    "model_type": "classifier",
                })
            return detections
            
        # 2. Stage 1: Detection (Detect people in the frame)
        det_results = self.person_detector.predict(source=frame, classes=[0], verbose=False, stream=False)
        if not det_results:
            return detections
            
        h, w = frame.shape[:2]
        
        for det_res in det_results:
            if not getattr(det_res, "boxes", None): continue
            
            for box in det_res.boxes:
                # bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # ensure within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # skip invalid or tiny boxes
                if x2 - x1 < 20 or y2 - y1 < 20: continue
                
                # crop person
                crop = frame[y1:y2, x1:x2]
                
                # Stage 2: Classification
                cls_results = self.model.predict(source=crop, verbose=False, stream=False)
                for res in cls_results:
                    if not hasattr(res, "probs") or res.probs is None: continue
                    probs = res.probs
                    cls_id = int(probs.top1)
                    conf = float(probs.top1conf)
                    
                    if conf < conf_threshold: continue
                    
                    raw_name = self.model.names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                    cls_name = _normalise_class(raw_name)
                    
                    is_anomaly = cls_name in ANOMALY_CLASSES
                    color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                    
                    detections.append({
                        "class":      cls_name,
                        "confidence": round(conf, 3),
                        "bbox":       [x1, y1, x2, y2],
                        "color":      color,
                        "is_anomaly": is_anomaly,
                        "model_type": "two_stage",
                    })
                    
        return detections

        return detections

    # ── Detector inference ────────────────────────────────────────────────────

    def _infer_detector(self, frame: np.ndarray, conf_threshold: float) -> List[dict]:
        """Object detection model: returns precise per-person bounding boxes."""
        results = self.model.predict(
            source=frame, conf=conf_threshold, iou=0.45,
            verbose=False, stream=False,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                raw_name = (
                    self.model.names[cls_id]
                    if cls_id < len(self.class_names) else str(cls_id)
                )
                cls_name = _normalise_class(raw_name)
                detections.append({
                    "class":      cls_name,
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2, y2],
                    "color":      CLASS_COLORS.get(cls_name, DEFAULT_COLOR),
                    "is_anomaly": cls_name in ANOMALY_CLASSES,
                    "model_type": "detector",
                })
        return detections


# ── Singleton access ──────────────────────────────────────────────────────────
# from models.inference_engine import ExamAnomalyEngine
# engine = ExamAnomalyEngine.get()
# dets   = engine.infer(frame)
