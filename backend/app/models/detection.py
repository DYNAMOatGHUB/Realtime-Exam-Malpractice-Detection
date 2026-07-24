"""
Detection event model — stores every classified malpractice event from the ML pipeline.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.types import Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class MalpracticeClass(str, enum.Enum):
    """
    Malpractice behaviour categories detected by the LSTM/GRU classifier.
    """
    COPYING_FROM_NEIGHBOUR = "COPYING_FROM_NEIGHBOUR"
    USING_MOBILE_PHONE = "USING_MOBILE_PHONE"
    PASSING_NOTES = "PASSING_NOTES"
    UNAUTHORIZED_MATERIALS = "UNAUTHORIZED_MATERIALS"
    LOOKING_AROUND_SUSPICIOUSLY = "LOOKING_AROUND_SUSPICIOUSLY"
    TALKING = "TALKING"
    LEAVING_SEAT = "LEAVING_SEAT"
    OTHER = "OTHER"


class TriageStatus(str, enum.Enum):
    """
    Confidence-based routing outcome (Layer 6).
    """
    AUTO_ALERTED = "AUTO_ALERTED"       # high confidence — email sent automatically
    REVIEW_PENDING = "REVIEW_PENDING"   # mid confidence — awaiting HEC review
    CONFIRMED = "CONFIRMED"             # HEC manually confirmed as malpractice
    DISMISSED = "DISMISSED"             # HEC dismissed as false positive
    DISCARDED = "DISCARDED"             # low confidence — dropped


class DetectionEvent(Base):
    """
    Represents a single malpractice detection event from the inference pipeline.
    Stores confidence score, classification, frame evidence reference, and triage status.
    """
    __tablename__ = "detection_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ── Source identification ─────────────────────────────────────
    camera_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    lh_mapping_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("lh_mappings.id"), nullable=True
    )

    # ── Detection data ────────────────────────────────────────────
    malpractice_class: Mapped[MalpracticeClass] = mapped_column(
        Enum(MalpracticeClass, name="malpractice_class_enum"), nullable=False
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    bounding_box: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="JSON-serialised [x1,y1,x2,y2] in frame coords"
    )
    keypoints_snapshot: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON-serialised 17-keypoint array at peak confidence frame"
    )
    frame_count_in_window: Mapped[int] = mapped_column(Integer, default=0)

    # ── Evidence storage (MinIO object path) ──────────────────────
    evidence_frame_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    evidence_clip_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ── Triage ────────────────────────────────────────────────────
    triage_status: Mapped[TriageStatus] = mapped_column(
        Enum(TriageStatus, name="triage_status_enum"),
        nullable=False,
        default=TriageStatus.REVIEW_PENDING,
        index=True,
    )
    hec_reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    hec_review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ── Timestamps ────────────────────────────────────────────────
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # ── Relationships ─────────────────────────────────────────────
    lh_mapping: Mapped["LHMapping"] = relationship("LHMapping")  # type: ignore[name-defined]
    hec_reviewer: Mapped["User"] = relationship("User")  # type: ignore[name-defined]
    alert: Mapped["AlertLog | None"] = relationship("AlertLog", back_populates="detection_event", uselist=False)

    def __repr__(self) -> str:
        return (
            f"<DetectionEvent id={self.id} cam={self.camera_id} "
            f"class={self.malpractice_class} conf={self.confidence_score:.2f} "
            f"status={self.triage_status}>"
        )
