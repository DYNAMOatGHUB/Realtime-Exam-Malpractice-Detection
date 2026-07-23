"""
Celery application factory.
Defines task queues for:
  - 'capture'   : RTSP frame ingestion (CPU-bound, I/O)
  - 'inference' : YOLO + Pose + LSTM GPU inference (GPU-bound)
  - 'alerts'    : Evidence packaging + email sending (I/O)
"""
from __future__ import annotations

from celery import Celery
from celery.utils.log import get_task_logger

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "exam_vigilance",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.services.rtsp_capture",
        "app.services.inference_worker",
        "app.services.alert_router",
    ],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Queue routing
    task_queues={
        "capture": {"exchange": "capture", "routing_key": "capture"},
        "inference": {"exchange": "inference", "routing_key": "inference"},
        "alerts": {"exchange": "alerts", "routing_key": "alerts"},
    },
    task_routes={
        "app.services.rtsp_capture.*": {"queue": "capture"},
        "app.services.inference_worker.*": {"queue": "inference"},
        "app.services.alert_router.*": {"queue": "alerts"},
    },
    task_default_queue="inference",

    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,   # one task at a time per GPU worker
    task_soft_time_limit=30,        # 30s soft limit per inference task
    task_time_limit=60,             # 60s hard limit

    # Result expiry
    result_expires=3600,

    # Beat schedule (Layer 12: retraining trigger)
    beat_schedule={
        "check-retraining-trigger": {
            "task": "app.services.inference_worker.check_retraining_needed",
            "schedule": 86400.0,  # every 24h
        },
    },
)
