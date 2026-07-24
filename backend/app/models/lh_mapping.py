"""
Lecture Hall (LH) to Invigilator mapping model.
The Head of Exam Cell creates these before each exam session.
Each mapping links a camera to the responsible invigilator.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.types import Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Invigilator(Base):
    """
    Represents a faculty member who can receive malpractice alerts.
    Stored independently so the same person can be mapped to multiple halls.
    """
    __tablename__ = "invigilators"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Reverse relation
    lh_mappings: Mapped[list["LHMapping"]] = relationship("LHMapping", back_populates="invigilator")

    def __repr__(self) -> str:
        return f"<Invigilator {self.full_name} <{self.email}>>"


class LectureHall(Base):
    """
    Represents a physical exam hall/room.
    """
    __tablename__ = "lecture_halls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lh_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    building: Mapped[str | None] = mapped_column(String(255), nullable=True)
    capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Reverse relation
    lh_mappings: Mapped[list["LHMapping"]] = relationship("LHMapping", back_populates="lecture_hall")

    def __repr__(self) -> str:
        return f"<LectureHall {self.lh_code}: {self.name}>"


class LHMapping(Base):
    """
    Session-scoped mapping: links a LectureHall to an Invigilator for a specific exam session,
    and ties the hall to a specific camera ID (RTSP stream).
    This is what Layer 2 stores, and Layers 5-7 query to route alerts.
    """
    __tablename__ = "lh_mappings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lh_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("lecture_halls.id"), nullable=False
    )
    invigilator_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("invigilators.id"), nullable=False
    )
    camera_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    rtsp_url: Mapped[str] = mapped_column(String(512), nullable=False)
    exam_session_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    lecture_hall: Mapped["LectureHall"] = relationship("LectureHall", back_populates="lh_mappings")
    invigilator: Mapped["Invigilator"] = relationship("Invigilator", back_populates="lh_mappings")

    def __repr__(self) -> str:
        return f"<LHMapping cam={self.camera_id} lh={self.lh_id}>"
