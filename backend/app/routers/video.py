"""
Video analysis router — real-time SSE streaming.

Detection pipeline:
  1. PoseAnomalyDetector (PCA on normal-act keypoints) — per-person NORMAL/CHEATING
  2. Motion-proxy fallback (no model) — raw frame stream

Evidence rules:
  - A person must hold a "CHEATING" state for ≥ 5 consecutive frames (~0.6 s at 8 FPS)
    before the frame is clipped as evidence.
  - Evidence frames contain ONLY red-boxed (anomalous) persons — green boxes are hidden.

Frame delivery:
  - ffmpeg MJPEG pipe → Python → annotate → base64 JPEG → SSE → browser
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import time
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add root project folder to sys.path so we can import `ml` module
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from ml.models.pose_anomaly_detector import PoseAnomalyDetector
from ultralytics import YOLO

# ── Shared pose model (lazy-loaded) ───────────────────────────
_pose_model = None
def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        _pose_model = YOLO('yolov8s-pose.pt')
    return _pose_model

# ── Frames a person must hold CHEATING state before evidence ──
EVIDENCE_MIN_FRAMES = 5

logger = logging.getLogger(__name__)
router = APIRouter()

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")

# ── Class metadata (matches ExamCheatingDataset folder names) ────
# These are the exact classes trained on your dataset:
#   normal act / looking friend / giving object / giving code / cheating
CLASS_COLORS = {
    # ExamCheatingDataset classes (primary)
    "NORMAL":            (30,  200,  30),
    "LOOKING_AT_FRIEND": (255,  60,  30),
    "GIVING_OBJECT":     (255, 130,  20),
    "GIVING_CODE":       (220,  30,  30),
    "CHEATING":          (200,  10,  10),
    # Legacy fallback names (kept for compatibility)
    "COPYING":           (255,  30,  30),
    "SUSPICIOUS":        (255,  40,  40),
    "TALKING":           (240, 120,  10),
    "LEAVING_SEAT":      (255, 170,  20),
}
# Classes that are actual anomalies (NORMAL is excluded)
ANOMALY_CLASS_NAMES = [
    "LOOKING_AT_FRIEND", "GIVING_OBJECT", "GIVING_CODE", "CHEATING"
]
DEFAULT_COLOR = (239, 68, 68)

# ── Inference engine (lazy-loaded) ────────────────────────────
_engine = None
_engine_checked = False


def _get_engine():
    """Return the inference engine, or None if model not available."""
    global _engine, _engine_checked
    if _engine_checked:
        return _engine
    _engine_checked = True
    try:
        from models.inference_engine import ExamAnomalyEngine
        eng = ExamAnomalyEngine.get()
        if eng.ready:
            logger.info("✅ Inference engine ready: %s", eng.model_path)
            _engine = eng
        else:
            logger.warning("⚠️  No trained model found — using motion-proxy detection")
    except Exception as exc:
        logger.warning("Could not load inference engine: %s", exc)
    return _engine


# ── Path resolver ──────────────────────────────────────────────

def _resolve(video_rel: str) -> str:
    import urllib.parse
    p = urllib.parse.unquote(video_rel)
    media = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dashboard/media"))
    for pfx in ("/app/media/", "/media/", "\\media\\"):
        if pfx in p:
            p = p.split(pfx, 1)[-1]
            break
    if not p.startswith(media):
        p = p.lstrip("\\/")
        return os.path.join(media, p)
    return p


# ── ffprobe metadata ───────────────────────────────────────────

def _probe(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            capture_output=True, timeout=15,
        )
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                num, den = (s.get("r_frame_rate", "25/1").split("/") + ["1"])[:2]
                fps = float(num) / max(float(den), 1)
                return {
                    "width":    int(s.get("width", 640)),
                    "height":   int(s.get("height", 360)),
                    "fps":      fps,
                    "nb_frames": int(s.get("nb_frames", 0) or 0),
                    "duration": float(s.get("duration", 0) or 0),
                }
    except Exception as exc:
        logger.warning("ffprobe failed: %s", exc)
    return {}


# ── Motion proxy (packet-size analysis) ───────────────────────
# Used when no trained model is available.

def _motion_proxy_timestamps(video_path: str, top_n: int = 3) -> List[float]:
    """
    Find the top-N most motion-heavy timestamps using H.264 packet sizes.
    Larger encoded packets = more inter-frame change = more activity.
    """
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "quiet", "-select_streams", "v:0",
             "-show_packets", "-of", "compact", video_path],
            capture_output=True, text=True, timeout=20,
        )
        packets = []
        for line in r.stdout.split("\n"):
            if not line.startswith("packet|"):
                continue
            parts = {kv.split("=")[0]: kv.split("=")[1]
                     for kv in line.split("|")[1:] if "=" in kv}
            try:
                ts   = float(parts.get("pts_time", -1))
                size = int(parts.get("size", 0))
                if ts >= 0 and size > 0:
                    packets.append((ts, size))
            except (ValueError, IndexError):
                continue

        if not packets:
            return []

        packets.sort(key=lambda x: x[1], reverse=True)
        selected: List[float] = []
        for ts, _ in packets:
            if all(abs(ts - s) > 1.0 for s in selected):
                selected.append(ts)
            if len(selected) >= top_n:
                break
        return sorted(selected)
    except Exception as exc:
        logger.warning("Motion proxy failed: %s", exc)
        return []


# ── Annotate JPEG with bounding boxes ─────────────────────────

def _annotate_jpeg(jpeg_bytes: bytes, detections: List[dict],
                   width: int, height: int) -> bytes:
    """
    Decode JPEG → draw coloured bounding boxes → re-encode JPEG.
    All via ffmpeg (no cv2 required). Falls back to original if error.
    """
    if not detections:
        return jpeg_bytes
    try:
        dec = subprocess.run(
            [FFMPEG, "-i", "pipe:0", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-vframes", "1",
             "-s", f"{width}x{height}", "pipe:1"],
            input=jpeg_bytes, capture_output=True, timeout=8,
        )
        if dec.returncode != 0 or len(dec.stdout) < width * height * 3:
            return jpeg_bytes

        raw    = bytearray(dec.stdout)
        stride = width * 3

        def clamp(v: int) -> int:
            return max(0, min(255, v))

        def fill_alpha(x1: int, y1: int, x2: int, y2: int,
                       r: int, g: int, b: int, a: float = 0.18) -> None:
            for y in range(max(0, y1), min(height, y2)):
                for x in range(max(0, x1), min(width, x2)):
                    o = y * stride + x * 3
                    raw[o]   = clamp(int(raw[o]   * (1-a) + r * a))
                    raw[o+1] = clamp(int(raw[o+1] * (1-a) + g * a))
                    raw[o+2] = clamp(int(raw[o+2] * (1-a) + b * a))

        def hline(y: int, x1: int, x2: int,
                  r: int, g: int, b: int, t: int = 4) -> None:
            for dy in range(-t//2, t//2 + 1):
                ry = y + dy
                if 0 <= ry < height:
                    for x in range(max(0, x1), min(width, x2)):
                        o = ry * stride + x * 3
                        raw[o] = r; raw[o+1] = g; raw[o+2] = b

        def vline(x: int, y1: int, y2: int,
                  r: int, g: int, b: int, t: int = 4) -> None:
            for dx in range(-t//2, t//2 + 1):
                rx = x + dx
                if 0 <= rx < width:
                    for y in range(max(0, y1), min(height, y2)):
                        o = y * stride + rx * 3
                        raw[o] = r; raw[o+1] = g; raw[o+2] = b

        for i, det in enumerate(detections):
            bbox  = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            r, g, b = det.get("color", CLASS_COLORS.get(det.get("class", ""), DEFAULT_COLOR))

            fill_alpha(x1, y1, x2, y2, r, g, b, 0.18)

            # Bold border
            hline(y1, x1, x2, r, g, b, 4)
            hline(y2, x1, x2, r, g, b, 4)
            vline(x1, y1, y2, r, g, b, 4)
            vline(x2, y1, y2, r, g, b, 4)

            # Corner brackets
            bl = max(20, min(50, (x2-x1)//4))
            hline(y1, x1, x1+bl, r, g, b, 5); vline(x1, y1, y1+bl, r, g, b, 5)
            hline(y1, x2-bl, x2, r, g, b, 5); vline(x2, y1, y1+bl, r, g, b, 5)
            hline(y2, x1, x1+bl, r, g, b, 5); vline(x1, y2-bl, y2, r, g, b, 5)
            hline(y2, x2-bl, x2, r, g, b, 5); vline(x2, y2-bl, y2, r, g, b, 5)

            # Label bar above box
            fill_alpha(x1, max(0, y1-24), x2, y1, r, g, b, 0.85)

        # HUD strip at top
        for y in range(min(40, height)):
            for x in range(width):
                o = y * stride + x * 3
                raw[o]   = clamp(int(raw[o]   * 0.30))
                raw[o+1] = clamp(int(raw[o+1] * 0.30))
                raw[o+2] = clamp(int(raw[o+2] * 0.30))

        enc = subprocess.run(
            [FFMPEG, "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-i", "pipe:0",
             "-vframes", "1", "-q:v", "3", "-f", "mjpeg", "pipe:1"],
            input=bytes(raw), capture_output=True, timeout=8,
        )
        if enc.returncode == 0 and enc.stdout:
            return enc.stdout
    except Exception as exc:
        logger.warning("annotate_jpeg error: %s", exc)
    return jpeg_bytes


# ── cv2-based annotation (used when cv2 is available) ─────────

def _annotate_with_cv2(jpeg_bytes: bytes, detections: List[dict]) -> bytes:
    """Fast annotation using cv2 when available."""
    try:
        import cv2
        import numpy as np
        arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jpeg_bytes

        h, w = frame.shape[:2]
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            r, g, b = det.get("color", DEFAULT_COLOR)
            color_bgr = (b, g, r)
            label     = f"{det.get('class','?')} {det.get('confidence',0):.0%}"

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

            # Bold border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)

            # Corner brackets
            bl = max(20, min(50, (x2-x1)//4))
            for (px, py, ex, ey) in [
                ((x1,y1), (x1+bl,y1)), ((x1,y1), (x1,y1+bl)),
                ((x2,y1), (x2-bl,y1)), ((x2,y1), (x2,y1+bl)),
                ((x1,y2), (x1+bl,y2)), ((x1,y2), (x1,y2-bl)),
                ((x2,y2), (x2-bl,y2)), ((x2,y2), (x2,y2-bl)),
            ]:
                cv2.line(frame, px, py, color_bgr, 5)

            # Label bar
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+6, y1), color_bgr, -1)
            cv2.putText(frame, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # Dark HUD strip
        frame[:40] = (frame[:40] * 0.30).astype(frame.dtype)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes()
    except Exception as exc:
        logger.debug("cv2 annotation failed (%s), using ffmpeg path", exc)
        return jpeg_bytes


# ── MJPEG pipe reader ──────────────────────────────────────────

async def _iter_mjpeg(proc) -> AsyncGenerator[bytes, None]:
    loop = asyncio.get_event_loop()
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"
    buf = b""
    in_frame = False

    while True:
        chunk = await loop.run_in_executor(None, proc.stdout.read, 131072)
        if not chunk:
            break
        buf += chunk

        while True:
            if not in_frame:
                idx = buf.find(SOI)
                if idx == -1:
                    buf = b""
                    break
                buf = buf[idx:]
                in_frame = True

            eoi_idx = buf.find(EOI, 2)
            if eoi_idx == -1:
                break

            end = eoi_idx + 2
            yield buf[:end]
            buf = buf[end:]
            in_frame = False


# ── Extract single evidence frame ─────────────────────────────

async def _extract_jpeg(video_path: str, ts: float,
                         width: int, height: int) -> Optional[bytes]:
    try:
        proc = subprocess.Popen(
            [
                FFMPEG,
                "-ss", f"{ts:.4f}", "-i", video_path,
                "-vframes", "1",
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                "-q:v", "2", "-f", "mjpeg", "pipe:1"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdout, _ = proc.communicate(timeout=10)
        if stdout and len(stdout) > 200:
            return stdout
    except Exception as exc:
        logger.debug("evidence frame extract failed @%.2fs: %s", ts, exc)
    return None


# ── Determine detection mode for this session ─────────────────

def _detection_mode(engine) -> str:
    if engine and engine.ready:
        return "yolo"
    try:
        import cv2  # noqa: F401
        return "motion_proxy"
    except ImportError:
        return "motion_proxy"


# ── Main SSE generator ─────────────────────────────────────────

async def _stream_video(job_id: str, video_rel: str) -> AsyncGenerator[str, None]:
    sep        = "\n\n"
    video_path = _resolve(video_rel)
    loop       = asyncio.get_event_loop()

    # ── Probe ────────────────────────────────────────────────────
    meta = _probe(video_path)
    if not meta:
        yield f'data: {json.dumps({"type":"error","message":f"Cannot read video: {video_path}"})}{sep}'
        return

    width, height = meta["width"], meta["height"]
    fps      = meta["fps"]
    nb_frames = meta["nb_frames"] or int(meta["duration"] * fps)
    duration  = meta["duration"]

    # ── Load inference engine ────────────────────────────────────
    engine = _get_engine()
    mode   = _detection_mode(engine)

    # Check cv2 availability for annotation
    try:
        import cv2 as _cv2  # noqa: F401
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    yield f'data: {json.dumps({"type":"meta","total_frames":nb_frames,"fps":fps,"width":width,"height":height,"duration":duration,"detection_mode":mode})}{sep}'

    # ── Pre-scan ─────────────────────────────────────────────────
    anomaly_timestamps: List[float] = []

    if mode == "yolo":
        # YOLO mode: no pre-scan needed — detect on every frame
        yield f'data: {json.dumps({"type":"scanning","message":"Model loaded — starting real-time inference…"})}{sep}'
        yield f'data: {json.dumps({"type":"scan_complete","count":0,"mode":"yolo"})}{sep}'
    else:
        # Motion proxy: pre-scan to find most-active frames
        yield f'data: {json.dumps({"type":"scanning","message":"No model loaded — analysing motion to locate anomaly moments…"})}{sep}'
        anomaly_timestamps = await loop.run_in_executor(
            None, _motion_proxy_timestamps, video_path, 3
        )
        yield f'data: {json.dumps({"type":"scan_complete","anomaly_timestamps":anomaly_timestamps,"count":len(anomaly_timestamps),"mode":"motion_proxy"})}{sep}'
        logger.info("Motion proxy timestamps: %s", anomaly_timestamps)

    # ── Stream frames ─────────────────────────────────────────────
    OUT_FPS  = 8
    SCALE_W  = min(width,  1280)
    SCALE_H  = min(height, 720)
    ANOM_WIN = 0.8   # seconds window around anomaly timestamps

    proc = subprocess.Popen(
        [FFMPEG, "-i", video_path, "-r", str(OUT_FPS),
        "-vf", f"scale={SCALE_W}:{SCALE_H}:force_original_aspect_ratio=decrease,pad={SCALE_W}:{SCALE_H}:(ow-iw)/2:(oh-ih)/2:black",
        "-f", "mjpeg", "-q:v", "4", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    frame_count        = 0
    total_anomalies    = 0
    anomaly_log        = []
    # track_id → consecutive frames classified as CHEATING
    red_frame_counter: dict = {}
    # track_ids for which we have already emitted evidence this "episode"
    evidence_emitted: set   = set()

    pose_model = _get_pose_model()
    detector   = PoseAnomalyDetector.get()

    try:
        async for jpeg_bytes in _iter_mjpeg(proc):
            frame_count += 1
            timestamp = frame_count / OUT_FPS
            progress  = round(min(100.0, timestamp / max(duration, 0.001) * 100), 1)

            # ── Per-frame detection results ──────────────────────────
            # List of dicts for anomalous persons only (used for SSE payload)
            det_payload  = []   # anomalies this frame
            all_persons  = []   # all persons (for live stream annotation)

            annotated = jpeg_bytes  # default: raw frame

            if has_cv2:
                import cv2
                import numpy as np

                arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Run pose model with tracking
                    results = await loop.run_in_executor(
                        None,
                        lambda: pose_model.track(
                            frame, persist=True, verbose=False,
                            imgsz=1280, conf=0.15
                        )
                    )

                    for r in results:
                        if r.boxes is None:
                            continue
                        boxes      = r.boxes.xyxy.cpu().numpy()
                        confs      = r.boxes.conf.cpu().numpy()
                        track_ids  = (
                            r.boxes.id.int().cpu().numpy()
                            if r.boxes.id is not None else []
                        )
                        kps_all    = (
                            r.keypoints.data.cpu().numpy()
                            if r.keypoints is not None and r.keypoints.data is not None
                            else []
                        )

                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = map(int, boxes[i])
                            det_conf  = float(confs[i])
                            track_id  = int(track_ids[i]) if i < len(track_ids) else -1
                            kp_raw    = kps_all[i].tolist() if i < len(kps_all) else []

                            # ── Classify pose ──────────────────────
                            state, anom_conf = detector.classify(
                                kp_raw, box=[x1, y1, x2, y2]
                            )
                            color = detector.color_for(state)
                            color_bgr = (color[2], color[1], color[0])  # RGB→BGR for cv2

                            # ── Update red-frame counter ────────────
                            if state == "CHEATING":
                                red_frame_counter[track_id] = red_frame_counter.get(track_id, 0) + 1
                            else:
                                # Reset counter when person goes back to normal
                                if track_id in red_frame_counter:
                                    red_frame_counter[track_id] = 0
                                    evidence_emitted.discard(track_id)

                            person_info = {
                                "track_id":   track_id,
                                "state":       state,
                                "confidence":  round(anom_conf, 3),
                                "bbox":        [float(x1), float(y1), float(x2), float(y2)],
                                "color":       color,
                                "kp_raw":      kp_raw,
                            }
                            all_persons.append(person_info)

                            if state == "CHEATING":
                                det_payload.append({
                                    "track_id":   track_id,
                                    "class":      "CHEATING",
                                    "confidence": round(anom_conf, 3),
                                    "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                                })

                    # ── Draw annotations on live stream frame ────────
                    # Draw ALL persons (green for normal, red for cheating)
                    for p in all_persons:
                        x1, y1, x2, y2 = map(int, p["bbox"])
                        tid    = p["track_id"]
                        state  = p["state"]
                        conf   = p["confidence"]
                        bgr    = (p["color"][2], p["color"][1], p["color"][0])
                        label  = f"ID:{tid} {state} {conf:.2f}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                        cv2.putText(
                            frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2
                        )
                        # Draw keypoints
                        for kpt in p["kp_raw"]:
                            if len(kpt) >= 3 and float(kpt[2]) > 0.4:
                                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, bgr, -1)

                    ret, buffer = cv2.imencode(".jpg", frame)
                    annotated   = buffer.tobytes() if ret else jpeg_bytes

            # ── 5-frame evidence gate ────────────────────────────────
            # For each track_id that has been red for >= EVIDENCE_MIN_FRAMES,
            # build an evidence frame (red boxes only) and emit once per episode.
            if has_cv2 and det_payload:
                for p in all_persons:
                    tid = p["track_id"]
                    if (
                        p["state"] == "CHEATING"
                        and red_frame_counter.get(tid, 0) >= EVIDENCE_MIN_FRAMES
                        and tid not in evidence_emitted
                    ):
                        evidence_emitted.add(tid)
                        total_anomalies += 1

                        # ── Build evidence frame: only red-boxed persons ──
                        import cv2 as _cv2
                        arr_ev   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                        ev_frame = _cv2.imdecode(arr_ev, _cv2.IMREAD_COLOR)
                        if ev_frame is not None:
                            for ep in all_persons:
                                if ep["state"] != "CHEATING":
                                    continue  # skip green boxes in evidence
                                ex1, ey1, ex2, ey2 = map(int, ep["bbox"])
                                red_bgr = (0, 0, 200)
                                _cv2.rectangle(ev_frame, (ex1, ey1), (ex2, ey2), red_bgr, 3)
                                elabel = f"ID:{ep['track_id']} CHEATING {ep['confidence']:.2f}"
                                _cv2.putText(
                                    ev_frame, elabel,
                                    (ex1, max(10, ey1 - 10)),
                                    _cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_bgr, 2
                                )
                                for kpt in ep["kp_raw"]:
                                    if len(kpt) >= 3 and float(kpt[2]) > 0.4:
                                        _cv2.circle(ev_frame, (int(kpt[0]), int(kpt[1])), 4, red_bgr, -1)

                            _, ev_buf = _cv2.imencode(".jpg", ev_frame)
                            ev_bytes  = ev_buf.tobytes()
                        else:
                            ev_bytes = jpeg_bytes

                        anomaly_log.append({
                            "frame":      frame_count,
                            "timestamp":  round(timestamp, 2),
                            "track_id":   tid,
                            "class":      "CHEATING",
                            "confidence": p["confidence"],
                            "bbox":       p["bbox"],
                        })
                        if len(anomaly_log) > 200:
                            anomaly_log.pop(0)

                        yield f'data: {json.dumps({"type":"evidence","timestamp":round(timestamp,2),"track_id":tid,"class":"CHEATING","confidence":p["confidence"],"bbox":p["bbox"],"image":base64.b64encode(ev_bytes).decode(),"mime_type":"image/jpeg"})}{sep}'

            yield f'data: {json.dumps({"type":"frame","frame":frame_count,"progress":progress,"anomalies":total_anomalies,"detections":det_payload,"image":base64.b64encode(annotated).decode(),"mime_type":"image/jpeg"})}{sep}'
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("Stream error job %s: %s", job_id, exc)
        yield f'data: {json.dumps({"type":"error","message":str(exc)})}{sep}'
    finally:
        try:
            proc.kill()
        except Exception:
            pass

    yield f'data: {json.dumps({"type":"complete","total_frames":frame_count,"total_anomalies":total_anomalies,"anomaly_log":anomaly_log[-50:],"detection_mode":mode})}{sep}'


# ── Request model ──────────────────────────────────────────────

class VideoAnalysisRequest(BaseModel):
    job_id:     str
    video_path: str
    mapping_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_video(req: VideoAnalysisRequest):
    logger.info("Analyze: job=%s", req.job_id)
    try:
        from app.services.inference_worker import process_video_task
        process_video_task.delay(req.job_id, _resolve(req.video_path), req.mapping_id)
        return {"status": "ok", "message": "Task dispatched"}
    except Exception as exc:
        logger.warning("Celery skip: %s", exc)
        return {"status": "ok", "message": "Stream-only mode"}


@router.get("/stream/{job_id}")
async def stream_video_analysis(job_id: str, video_path: str):
    """SSE endpoint: streams annotated frames with real YOLO detections."""
    return StreamingResponse(
        _stream_video(job_id, video_path),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@router.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    from app.core.database import get_db_context
    from sqlalchemy import text
    try:
        async with get_db_context() as db:
            result = await db.execute(
                text("SELECT status, total_anomalies, output_video_file "
                     "FROM exam_control_videoanalysisjob WHERE id = :id"),
                {"id": job_id},
            )
            row = result.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": row[0], "total_anomalies": row[1], "output_video_file": row[2]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/model/status")
async def model_status():
    """Check if the trained model is loaded."""
    engine = _get_engine()
    return {
        "model_loaded": engine is not None and engine.ready,
        "model_path":   str(engine.model_path) if engine and engine.model_path else None,
        "classes":      engine.class_names if engine else [],
        "mode":         _detection_mode(engine),
    }
