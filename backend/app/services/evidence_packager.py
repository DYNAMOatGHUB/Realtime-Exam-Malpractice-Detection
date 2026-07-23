"""
Evidence packaging service — Layer 7.
Extracts flagged frames, draws annotated bounding boxes and keypoint overlays,
packages metadata, and uploads to MinIO object storage.
"""
from __future__ import annotations

import io
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

# cv2 and numpy are imported locally in functions to avoid FastAPI startup crash

from app.core.minio_client import upload_evidence_frame, get_presigned_url
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Overlay style constants
BOX_COLOUR = (0, 0, 255)          # Red bounding box
KEYPOINT_COLOUR = (0, 255, 255)   # Yellow keypoints
TEXT_COLOUR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2


def annotate_frame(
    frame: np.ndarray,
    bounding_box: list[float] | None,
    malpractice_class: str,
    confidence: float,
    track_id: int | None = None,
    keypoints: list | None = None,
) -> Any:
    """
    Draw detection overlay on frame:
    - Red bounding box around flagged student
    - Confidence score + class label badge
    - Keypoint dots (if provided)
    - Timestamp watermark
    """
    import cv2
    import numpy as np
    
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Bounding box
    if bounding_box:
        x1 = int(bounding_box[0] * w)
        y1 = int(bounding_box[1] * h)
        x2 = int(bounding_box[2] * w)
        y2 = int(bounding_box[3] * h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOUR, THICKNESS + 1)

        # Label badge (dark background + white text)
        label = f"{malpractice_class.replace('_', ' ')} ({confidence:.0%})"
        if track_id is not None:
            label = f"[Track {track_id}] {label}"

        (lw, lh), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        badge_y1 = max(y1 - lh - baseline - 8, 0)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + lw + 8, y1), BOX_COLOUR, -1)
        cv2.putText(
            annotated, label,
            (x1 + 4, y1 - baseline - 2),
            FONT, FONT_SCALE, TEXT_COLOUR, THICKNESS - 1, cv2.LINE_AA,
        )

    # Keypoint overlay
    if keypoints:
        for idx, kp in enumerate(keypoints):
            if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                kx, ky, kconf = kp[0], kp[1], kp[2]
                if kconf > 0.4:
                    px, py = int(kx * w), int(ky * h)
                    cv2.circle(annotated, (px, py), 4, KEYPOINT_COLOUR, -1)

    # Timestamp watermark
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cv2.putText(annotated, ts, (8, h - 10), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


def package_evidence(
    frame_b64: str,
    camera_id: str,
    lh_mapping_id: str | None,
    malpractice_class: str,
    confidence_score: float,
    track_id: int | None,
    bounding_box: list[float] | None,
    timestamp: str,
    event_id: str | None = None,
) -> dict[str, Any]:
    """
    Full evidence packaging pipeline:
    1. Decode frame
    2. Annotate with detection overlay
    3. Upload annotated frame to MinIO
    4. Return evidence metadata dict

    Returns:
        dict with 'frame_path', 'presigned_url', 'metadata_path', 'metadata'
    """
    import base64
    import cv2
    import numpy as np

    if event_id is None:
        event_id = str(uuid.uuid4())

    # Decode
    frame_bytes = base64.b64decode(frame_b64)
    frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Failed to decode frame from base64")

    # Annotate
    annotated = annotate_frame(
        frame,
        bounding_box=bounding_box,
        malpractice_class=malpractice_class,
        confidence=confidence_score,
        track_id=track_id,
    )

    # Build object path: evidence-frames/YYYY/MM/DD/<camera_id>/<event_id>.jpg
    ts_dt = datetime.now(timezone.utc)
    date_prefix = ts_dt.strftime("%Y/%m/%d")
    frame_object_name = f"{date_prefix}/{camera_id}/{event_id}.jpg"
    metadata_object_name = f"{date_prefix}/{camera_id}/{event_id}.json"

    # Encode annotated frame as JPEG
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    frame_bytes_out = buf.tobytes()

    # Upload frame
    upload_evidence_frame(
        object_name=frame_object_name,
        data=frame_bytes_out,
        content_type="image/jpeg",
    )

    # Build and upload metadata JSON
    metadata = {
        "event_id": event_id,
        "camera_id": camera_id,
        "lh_mapping_id": lh_mapping_id,
        "malpractice_class": malpractice_class,
        "confidence_score": confidence_score,
        "track_id": track_id,
        "bounding_box": bounding_box,
        "timestamp": timestamp,
        "packaged_at": ts_dt.isoformat(),
        "frame_path": frame_object_name,
    }
    meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
    upload_evidence_frame(
        object_name=metadata_object_name,
        data=meta_bytes,
        content_type="application/json",
        length=len(meta_bytes),
    )

    # Generate 24h presigned URL for email attachment
    presigned_url = get_presigned_url(frame_object_name)

    logger.info(
        "Evidence packaged: event=%s camera=%s class=%s conf=%.2f",
        event_id, camera_id, malpractice_class, confidence_score,
    )

    return {
        "event_id": event_id,
        "frame_path": frame_object_name,
        "metadata_path": metadata_object_name,
        "presigned_url": presigned_url,
        "metadata": metadata,
        "frame_bytes": frame_bytes_out,  # passed directly to email attachment
    }
