"""
RTSP capture service — Layer 3.
Reads live CCTV feeds via OpenCV, samples at configured fps,
and pushes serialised frame metadata to Redis for inference workers.

Design:
  - Each camera gets its own background thread (CaptureWorker).
  - Frames are JPEG-compressed before Redis insertion (reduces memory by ~10x).
  - Sampler drops frames to hit target fps rather than blocking GPU workers.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict

# cv2 and numpy are imported lazily inside CaptureWorker.run()
# so that FastAPI starts even when they are not installed.

from app.core.config import get_settings
from app.core.redis_client import push_frame_sync, get_sync_redis, ACTIVE_STREAMS_KEY

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class StreamConfig:
    camera_id: str
    rtsp_url: str
    lh_mapping_id: str | None = None
    target_fps: int = field(default_factory=lambda: settings.frame_sample_rate)


@dataclass
class CaptureWorker:
    config: StreamConfig
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _frame_count: int = field(default=0, init=False)
    _last_push_time: float = field(default=0.0, init=False)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.config.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started capture worker for camera: %s", self.config.camera_id)

        # Register in Redis active streams set
        r = get_sync_redis()
        r.sadd(ACTIVE_STREAMS_KEY, self.config.camera_id)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        r = get_sync_redis()
        r.srem(ACTIVE_STREAMS_KEY, self.config.camera_id)
        logger.info("Stopped capture worker for camera: %s", self.config.camera_id)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _capture_loop(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.config.rtsp_url)

        if not cap.isOpened():
            logger.error(
                "Cannot open RTSP stream: %s (camera: %s)",
                self.config.rtsp_url, self.config.camera_id,
            )
            return

        # Minimal buffer to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_interval = 1.0 / self.config.target_fps
        logger.info(
            "Camera %s native=%.1ffps, sampling at %dfps",
            self.config.camera_id, native_fps, self.config.target_fps,
        )

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed for camera %s — retrying", self.config.camera_id)
                time.sleep(0.5)
                # Attempt reconnect
                cap.release()
                cap = cv2.VideoCapture(self.config.rtsp_url)
                continue

            now = time.monotonic()
            # Rate limiting: only push at target fps
            if now - self._last_push_time < sample_interval:
                continue
            self._last_push_time = now

            self._push_frame(frame)
            self._frame_count += 1

        cap.release()

    def _push_frame(self, frame) -> None:
        """Compress frame to JPEG and push to Redis queue."""
        import cv2
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            payload = {
                "camera_id": self.config.camera_id,
                "lh_mapping_id": self.config.lh_mapping_id,
                "frame_b64": frame_b64,
                "frame_idx": self._frame_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "height": frame.shape[0],
                "width": frame.shape[1],
            }
            push_frame_sync(self.config.camera_id, payload)
        except Exception as exc:
            logger.error("Frame push error for camera %s: %s", self.config.camera_id, exc)


# ── Registry of active workers ─────────────────────────────────

_workers: Dict[str, CaptureWorker] = {}
_registry_lock = threading.Lock()


def start_stream(config: StreamConfig) -> bool:
    """Start a capture worker for a camera. Returns False if already running or limit reached."""
    with _registry_lock:
        if config.camera_id in _workers and _workers[config.camera_id].is_running:
            logger.warning("Camera %s already capturing", config.camera_id)
            return False

        if len(_workers) >= settings.max_cameras:
            logger.error(
                "Max camera limit reached (%d). Cannot add camera %s",
                settings.max_cameras, config.camera_id,
            )
            return False

        worker = CaptureWorker(config=config)
        worker.start()
        _workers[config.camera_id] = worker
        return True


def stop_stream(camera_id: str) -> bool:
    """Stop and remove a capture worker by camera ID."""
    with _registry_lock:
        worker = _workers.pop(camera_id, None)
        if worker is None:
            return False
        worker.stop()
        return True


def get_active_cameras() -> list[str]:
    with _registry_lock:
        return [cid for cid, w in _workers.items() if w.is_running]


def stop_all_streams() -> None:
    with _registry_lock:
        for worker in _workers.values():
            worker.stop()
        _workers.clear()
