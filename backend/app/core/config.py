"""
Core application configuration.
All settings are loaded from environment variables (or .env file).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import AnyUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_env: str = "development"
    secret_key: str = "change-me"
    debug: bool = True

    # ── Database ─────────────────────────────────────────────────
    database_url: str = Field(
        "postgresql+asyncpg://vigilance_user:password@localhost:5432/exam_vigilance"
    )
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "exam_vigilance"
    postgres_user: str = "vigilance_user"
    postgres_password: str = "password"

    # ── Redis / Celery ────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── MinIO ─────────────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_evidence: str = "evidence-frames"
    minio_bucket_models: str = "model-weights"
    minio_secure: bool = False

    # ── Gmail SMTP ────────────────────────────────────────────────
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = "your-email@gmail.com"
    smtp_password: str = "your-app-password"
    smtp_from_name: str = "Exam Vigilance System"
    daily_email_limit: int = 100

    # ── ML Model Paths ────────────────────────────────────────────
    yolo_model_path: str = "/ml/weights/yolov8n.pt"
    pose_model_path: str = "/ml/weights/yolov8n-pose.pt"
    lstm_model_path: str = "/ml/weights/lstm_classifier.pt"
    use_tensorrt: bool = False
    tensorrt_yolo_path: str = "/ml/weights/yolov8n.engine"
    tensorrt_lstm_path: str = "/ml/weights/lstm_classifier.engine"

    # ── Inference Parameters ──────────────────────────────────────
    frame_sample_rate: int = 8          # fps sampled from RTSP
    sequence_window_seconds: int = 10   # temporal window for LSTM
    max_cameras: int = 5
    celery_workers: int = 2

    # ── Confidence Thresholds ─────────────────────────────────────
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.40

    # ── Django Integration ────────────────────────────────────────
    django_secret_key: str = "change-me-django"
    django_allowed_hosts: str = "localhost,127.0.0.1"
    fastapi_base_url: str = "http://localhost:8000"

    @property
    def allowed_hosts_list(self) -> List[str]:
        return [h.strip() for h in self.django_allowed_hosts.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
