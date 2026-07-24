"""
Confidence-Based Triage and Alert Router — Layer 6 & 7.
Celery task that receives a classified detection event and:
  - HIGH confidence → package evidence + email invigilator + log AUTO_ALERTED
  - MID confidence  → push to HEC review queue + log REVIEW_PENDING
  - LOW confidence  → discard (log DISCARDED)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.celery_app import celery_app
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _run_async(coro):
    """Helper to run an async coroutine from within a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _get_invigilator_for_camera(camera_id: str) -> dict[str, Any] | None:
    """Look up the active LH mapping for a camera and return invigilator info."""
    from app.core.database import get_db_context
    from app.models.lh_mapping import LHMapping
    from sqlalchemy import select

    async with get_db_context() as db:
        result = await db.execute(
            select(LHMapping)
            .where(LHMapping.camera_id == camera_id, LHMapping.is_active == True)
            .limit(1)
        )
        mapping = result.scalar_one_or_none()
        if mapping is None:
            return None

        await db.refresh(mapping, ["invigilator", "lecture_hall"])
        return {
            "mapping_id": str(mapping.id),
            "invigilator_id": str(mapping.invigilator_id),
            "invigilator_name": mapping.invigilator.full_name,
            "invigilator_email": mapping.invigilator.email,
            "lecture_hall_name": mapping.lecture_hall.name,
            "lh_code": mapping.lecture_hall.lh_code,
        }


async def _persist_detection_event(
    camera_id: str,
    lh_mapping_id: str | None,
    malpractice_class: str,
    confidence_score: float,
    bounding_box: list | None,
    triage_status: str,
    evidence_frame_path: str | None = None,
    event_id: str | None = None,
) -> str:
    """Persist a DetectionEvent to PostgreSQL and return its ID."""
    import uuid
    from app.core.database import get_db_context
    from app.models.detection import DetectionEvent, MalpracticeClass, TriageStatus

    async with get_db_context() as db:
        event = DetectionEvent(
            id=uuid.UUID(event_id) if event_id else uuid.uuid4(),
            camera_id=camera_id,
            lh_mapping_id=uuid.UUID(lh_mapping_id) if lh_mapping_id else None,
            malpractice_class=MalpracticeClass(malpractice_class),
            confidence_score=confidence_score,
            bounding_box=str(bounding_box) if bounding_box else None,
            triage_status=TriageStatus(triage_status),
            evidence_frame_path=evidence_frame_path,
        )
        db.add(event)
        await db.flush()
        return str(event.id)


async def _persist_alert_log(
    event_id: str,
    invigilator_id: str,
    status: str,
    recipient_email: str,
    email_subject: str,
    error_detail: str | None = None,
) -> None:
    from app.core.database import get_db_context
    from app.models.alert import AlertLog, AlertStatus
    import uuid

    async with get_db_context() as db:
        log = AlertLog(
            detection_event_id=uuid.UUID(event_id),
            invigilator_id=uuid.UUID(invigilator_id),
            status=AlertStatus(status),
            recipient_email=recipient_email,
            email_subject=email_subject,
            error_detail=error_detail,
            sent_at=datetime.now(timezone.utc) if status == "SENT" else None,
        )
        db.add(log)


@celery_app.task(name="app.services.alert_router.route_detection_event", bind=True, max_retries=3)
def route_detection_event(
    self,
    camera_id: str,
    lh_mapping_id: str | None,
    track_id: int | None,
    malpractice_class: str,
    confidence_score: float,
    all_scores: list[float],
    bounding_box: list[float] | None,
    frame_b64: str,
    timestamp: str,
) -> dict[str, Any]:
    """
    Layer 6: Confidence-based triage.
    Routes a detection event to AUTO_ALERTED, REVIEW_PENDING, or DISCARDED.
    """
    high_thresh = settings.high_confidence_threshold
    low_thresh = settings.low_confidence_threshold

    logger.info(
        "Routing event: camera=%s class=%s conf=%.3f",
        camera_id, malpractice_class, confidence_score,
    )

    # ── DISCARD ───────────────────────────────────────────────────
    if confidence_score < low_thresh:
        logger.debug("DISCARDED low-confidence event (%.3f < %.3f)", confidence_score, low_thresh)
        return {"status": "DISCARDED", "confidence": confidence_score}

    # ── PERSIST base event ────────────────────────────────────────
    triage_status = "AUTO_ALERTED" if confidence_score >= high_thresh else "REVIEW_PENDING"
    event_id = _run_async(
        _persist_detection_event(
            camera_id=camera_id,
            lh_mapping_id=lh_mapping_id,
            malpractice_class=malpractice_class,
            confidence_score=confidence_score,
            bounding_box=bounding_box,
            triage_status=triage_status,
        )
    )

    # ── MID CONFIDENCE → Review queue ────────────────────────────
    if confidence_score < high_thresh:
        review_payload = {
            "event_id": event_id,
            "camera_id": camera_id,
            "malpractice_class": malpractice_class,
            "confidence_score": confidence_score,
            "bounding_box": bounding_box,
            "timestamp": timestamp,
            "frame_b64": frame_b64[:500],  # truncated preview for queue
        }
        _run_async(push_to_review_queue(review_payload))
        logger.info("MID-CONFIDENCE event %s → review queue (%.3f)", event_id, confidence_score)
        return {"status": "REVIEW_PENDING", "event_id": event_id, "confidence": confidence_score}

    # ── HIGH CONFIDENCE → Auto-alert ─────────────────────────────
    # 1. Look up invigilator
    invigilator_info = _run_async(_get_invigilator_for_camera(camera_id))
    if invigilator_info is None:
        logger.warning(
            "No active LH mapping found for camera %s — event %s logged but no email sent",
            camera_id, event_id,
        )
        return {"status": "AUTO_ALERTED_NO_MAPPING", "event_id": event_id}

    # 2. Package evidence
    from app.services.evidence_packager import package_evidence
    evidence = package_evidence(
        frame_b64=frame_b64,
        camera_id=camera_id,
        lh_mapping_id=lh_mapping_id,
        malpractice_class=malpractice_class,
        confidence_score=confidence_score,
        track_id=track_id,
        bounding_box=bounding_box,
        timestamp=timestamp,
        event_id=event_id,
    )

    # 3. Send email
    from app.services.email_service import send_malpractice_alert
    email_result = send_malpractice_alert(
        recipient_email=invigilator_info["invigilator_email"],
        recipient_name=invigilator_info["invigilator_name"],
        lecture_hall_name=invigilator_info["lecture_hall_name"],
        malpractice_class=malpractice_class,
        confidence_score=confidence_score,
        timestamp=timestamp,
        camera_id=camera_id,
        presigned_url=evidence["presigned_url"],
        event_id=event_id,
        frame_bytes=evidence.get("frame_bytes"),
    )

    # 4. Log alert
    alert_status = "SENT" if email_result["success"] else (
        "SUPPRESSED" if email_result.get("suppressed") else "FAILED"
    )
    email_subject = (
        f"[ALERT] Malpractice Detected — {invigilator_info['lecture_hall_name']} — "
        f"{malpractice_class.replace('_', ' ').title()}"
    )
    _run_async(
        _persist_alert_log(
            event_id=event_id,
            invigilator_id=invigilator_info["invigilator_id"],
            status=alert_status,
            recipient_email=invigilator_info["invigilator_email"],
            email_subject=email_subject,
            error_detail=email_result.get("error"),
        )
    )

    logger.info(
        "HIGH-CONFIDENCE event %s → email %s to %s",
        event_id, alert_status, invigilator_info["invigilator_email"],
    )
    return {
        "status": "AUTO_ALERTED",
        "event_id": event_id,
        "email_status": alert_status,
        "invigilator": invigilator_info["invigilator_email"],
    }
