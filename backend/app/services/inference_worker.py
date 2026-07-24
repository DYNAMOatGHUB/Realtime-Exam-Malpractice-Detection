"""
GPU Inference worker - YOLO Pose + Behavior Analyzer
"""
from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from app.core.config import get_settings
import queue

logger = logging.getLogger(__name__)
settings = get_settings()

_model = None
_analyzer = None
_models_loaded = False

def _load_models_once() -> None:
    global _model, _analyzer, _models_loaded
    if _models_loaded:
        return

    current_path = Path(__file__).resolve()
    repo_root = str(current_path.parents[3]) if len(current_path.parents) > 3 else "/"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from ultralytics import YOLO
    from ml.models.behavior_analyzer import BehaviorAnalyzer
    
    device = "cuda" if _gpu_available() else "cpu"
    logger.info("Loading YOLO Pose model on device=%s", device)
    
    _model = YOLO('yolov8n-pose.pt')
    _model.to(device)
    
    _analyzer = BehaviorAnalyzer(fps=10.0, sustained_seconds=5.0, required_episodes=5)
    
    _models_loaded = True
    logger.info("All ML models loaded")

def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

COLORS = {
    "NORMAL": (0, 255, 0),       # Green
    "WARNING": (0, 255, 255),    # Yellow
    "SUSPICIOUS": (0, 0, 255)    # Red
}

def draw_skeleton(frame, keypoints, state):
    color = COLORS.get(state, (0, 255, 0))
    for kp in keypoints:
        x, y, conf = kp
        if conf > 0.4:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

def process_single_frame(frame_payload: dict[str, Any]) -> None:
    _load_models_once()

    try:
        from ml.models.behavior_analyzer import STATE_SUSPICIOUS

        frame_b64: str = frame_payload["frame_b64"]
        camera_id: str = frame_payload["camera_id"]
        frame_bytes = base64.b64decode(frame_b64)
        frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        timestamp: str = frame_payload.get("timestamp", "")
        lh_mapping_id: str | None = frame_payload.get("lh_mapping_id")
        h, w = frame.shape[:2]

        results = _model.track(frame, persist=True, classes=[0], verbose=False)
        
        if not results or len(results) == 0:
            return
            
        r = results[0]
        
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            return
            
        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.int().cpu().tolist()
        keypoints = r.keypoints.data.cpu().numpy()

        has_suspicious = False
        
        for i, track_id in enumerate(track_ids):
            kp_raw = keypoints[i]
            kp_norm = [[float(k[0])/w, float(k[1])/h, float(k[2])] for k in kp_raw]
                
            state = _analyzer.update_person(track_id, kp_norm)
            
            x1, y1, x2, y2 = map(int, boxes[i])
            color = COLORS.get(state, (0, 255, 0))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id} {state}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            draw_skeleton(frame, kp_raw, state)
            
            if state == STATE_SUSPICIOUS:
                has_suspicious = True
                from app.services.alert_router import route_detection_event
                import threading
                
                _, buffer = cv2.imencode('.jpg', frame)
                out_b64 = base64.b64encode(buffer).decode('utf-8')
                
                threading.Thread(target=route_detection_event, kwargs={
                    "camera_id": camera_id,
                    "lh_mapping_id": lh_mapping_id,
                    "track_id": track_id,
                    "malpractice_class": "BEHAVIOR_ANOMALY",
                    "confidence_score": 0.99,
                    "all_scores": [],
                    "bounding_box": [float(x1), float(y1), float(x2), float(y2)],
                    "frame_b64": out_b64,
                    "timestamp": timestamp,
                }, daemon=True).start()

    except Exception as exc:
        logger.error("Inference error on frame: %s", exc, exc_info=True)


def check_retraining_needed() -> dict[str, Any]:
    return {"status": "Not needed for YOLO Pose"}

def process_video_task(job_id: str, video_path: str, mapping_id: str | None) -> dict[str, Any]:
    _load_models_once()
    
    logger.info("Starting batch processing for job %s: %s", job_id, video_path)
    
    if not os.path.exists(video_path):
        _update_job_status(job_id, "failed")
        return {"error": "File not found"}
        
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    from ml.models.behavior_analyzer import BehaviorAnalyzer, STATE_SUSPICIOUS
    local_analyzer = BehaviorAnalyzer(fps=original_fps)
    
    current_path = Path(__file__).resolve()
    repo_root = current_path.parents[3]
    out_dir = repo_root / "dashboard" / "media" / "processed_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"annotated_{job_id}.mp4"
    out_path = out_dir / out_filename
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, original_fps, (width, height))
    
    anomalies_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = _model.track(frame, persist=True, classes=[0], verbose=False)
        
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().tolist()
                keypoints = r.keypoints.data.cpu().numpy()
                
                for i, track_id in enumerate(track_ids):
                    kp_raw = keypoints[i]
                    kp_norm = [[float(k[0])/width, float(k[1])/height, float(k[2])] for k in kp_raw]
                    
                    state = local_analyzer.update_person(track_id, kp_norm)
                    
                    if state == STATE_SUSPICIOUS:
                        anomalies_detected += 1
                        
                    x1, y1, x2, y2 = map(int, boxes[i])
                    color = COLORS.get(state, (0, 255, 0))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {track_id} {state}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    draw_skeleton(frame, kp_raw, state)
                    
        out.write(frame)
        
    cap.release()
    out.release()
    
    rel_out_path = f"processed_videos/{out_filename}"
    _update_job_status(job_id, "completed", anomalies_detected, rel_out_path)
    return {"status": "completed", "anomalies": anomalies_detected}


def _update_job_status(job_id: str, status: str, anomalies: int = 0, out_path: str = None) -> None:
    import asyncio
    from sqlalchemy import text
    from app.core.database import get_db_context
    
    async def _update():
        async with get_db_context() as db:
            if out_path:
                await db.execute(
                    text("UPDATE exam_control_videoanalysisjob SET status = :status, total_anomalies = :anomalies, output_video_file = :out_path WHERE id = :id"),
                    {"status": status, "anomalies": anomalies, "out_path": out_path, "id": job_id}
                )
            else:
                await db.execute(
                    text("UPDATE exam_control_videoanalysisjob SET status = :status WHERE id = :id"),
                    {"status": status, "id": job_id}
                )
            await db.commit()
            
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_update())
    loop.close()
