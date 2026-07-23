"""
Alert log model — records every automated email notification sent to invigilators.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class AlertStatus(str, enum.Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    SUPPRESSED = "SUPPRESSED"   # daily limit reached


class AlertLog(Base):
    """
    Audit trail for every automated invigilator alert email dispatched by Layer 7.
    Tracks delivery status, retry count, and error details.
    """
    __tablename__ = "alert_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    detection_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("detection_events.id"),
        nullable=False,
        index=True,
    )
    invigilator_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("invigilators.id"),
        nullable=False,
    )

    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, name="alert_status_enum"),
        nullable=False,
        default=AlertStatus.PENDING,
        index=True,
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Email delivery metadata
    recipient_email: Mapped[str | None] = mapped_column(Text, nullable=True)
    email_subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ─────────────────────────────────────────────
    detection_event: Mapped["DetectionEvent"] = relationship(  # type: ignore[name-defined]
        "DetectionEvent", back_populates="alert"
    )
    invigilator: Mapped["Invigilator"] = relationship("Invigilator")  # type: ignore[name-defined]

    def __repr__(self) -> str:
        return (
            f"<AlertLog id={self.id} event={self.detection_event_id} "
            f"status={self.status}>"
        )
