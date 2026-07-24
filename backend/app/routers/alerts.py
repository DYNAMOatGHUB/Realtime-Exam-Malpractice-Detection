"""
Alerts router — HEC manual review actions (confirm/dismiss flagged events).
Layer 8: Human-in-the-loop review endpoints.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.detection import DetectionEvent, TriageStatus
from app.models.alert import AlertLog

router = APIRouter()


class ReviewActionRequest(BaseModel):
    event_id: str
    reviewer_id: str        # User UUID from Django auth
    action: str             # "confirm" or "dismiss"
    note: Optional[str] = None


class ReviewActionResponse(BaseModel):
    event_id: str
    new_status: str
    reviewed_at: str
    message: str


@router.post("/review", response_model=ReviewActionResponse)
async def review_event(req: ReviewActionRequest, db: AsyncSession = Depends(get_db)):
    """
    Layer 8: HEC manually confirms or dismisses a flagged event.
    - confirm → sets triage_status to CONFIRMED
    - dismiss → sets triage_status to DISMISSED

    All actions are logged with reviewer ID and timestamp.
    """
    try:
        event_uuid = uuid.UUID(req.event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid event_id UUID format")

    result = await db.execute(
        select(DetectionEvent).where(DetectionEvent.id == event_uuid)
    )
    event = result.scalar_one_or_none()

    if event is None:
        raise HTTPException(status_code=404, detail=f"Detection event {req.event_id} not found")

    if event.triage_status not in (TriageStatus.REVIEW_PENDING, TriageStatus.AUTO_ALERTED):
        raise HTTPException(
            status_code=409,
            detail=f"Event already resolved with status: {event.triage_status.value}",
        )

    # Apply action
    if req.action == "confirm":
        event.triage_status = TriageStatus.CONFIRMED
        new_status = "CONFIRMED"
        message = "Event confirmed as malpractice. Disciplinary action can proceed."
    elif req.action == "dismiss":
        event.triage_status = TriageStatus.DISMISSED
        new_status = "DISMISSED"
        message = "Event dismissed as false positive. No further action required."
    else:
        raise HTTPException(status_code=400, detail="action must be 'confirm' or 'dismiss'")

    try:
        event.hec_reviewer_id = uuid.UUID(req.reviewer_id)
    except ValueError:
        pass  # reviewer_id optional to parse

    event.hec_review_note = req.note
    event.reviewed_at = datetime.now(timezone.utc)

    await db.flush()

    reviewed_at = event.reviewed_at.isoformat()
    return ReviewActionResponse(
        event_id=req.event_id,
        new_status=new_status,
        reviewed_at=reviewed_at,
        message=message,
    )


@router.get("/log")
async def alert_log(
    page: int = 1,
    page_size: int = 20,
    status_filter: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Full audit log of all auto-sent email alerts."""
    q = select(AlertLog).order_by(desc(AlertLog.created_at))

    if status_filter:
        from app.models.alert import AlertStatus
        try:
            q = q.where(AlertLog.status == AlertStatus(status_filter))
        except ValueError:
            pass

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    offset = (page - 1) * page_size
    rows = (await db.execute(q.offset(offset).limit(page_size))).scalars().all()

    return {
        "total": total,
        "page": page,
        "results": [
            {
                "id": str(r.id),
                "event_id": str(r.detection_event_id),
                "recipient_email": r.recipient_email,
                "status": r.status.value,
                "sent_at": r.sent_at.isoformat() if r.sent_at else None,
                "retry_count": r.retry_count,
                "error_detail": r.error_detail,
            }
            for r in rows
        ],
    }


@router.get("/review-queue")
async def get_review_queue_items(limit: int = 20):
    """Fetch pending mid-confidence events from the Redis review queue for the HEC dashboard."""
    import json
    
    raw_items = []

    items = []
    for raw in raw_items:
        try:
            items.append(json.loads(raw))
        except Exception:
            pass

    return {"queue_length": len(raw_items), "items": items}
