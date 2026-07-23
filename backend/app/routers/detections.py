"""
Detection events router — paginated log + WebSocket live feed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis_client import get_async_redis, REVIEW_QUEUE_KEY
from app.models.detection import DetectionEvent, TriageStatus

router = APIRouter()
logger = logging.getLogger(__name__)


class DetectionEventSchema(BaseModel):
    id: str
    camera_id: str
    malpractice_class: str
    confidence_score: float
    triage_status: str
    evidence_frame_path: Optional[str]
    detected_at: datetime
    lh_mapping_id: Optional[str]

    model_config = {"from_attributes": True}


@router.get("/", response_model=dict)
async def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    triage_status: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Paginated list of detection events.
    Supports filtering by triage_status and camera_id.
    """
    q = select(DetectionEvent).order_by(desc(DetectionEvent.detected_at))

    if triage_status:
        try:
            status_filter = TriageStatus(triage_status)
            q = q.where(DetectionEvent.triage_status == status_filter)
        except ValueError:
            pass

    if camera_id:
        q = q.where(DetectionEvent.camera_id == camera_id)

    # Count
    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar_one()

    # Paginate
    offset = (page - 1) * page_size
    result = await db.execute(q.offset(offset).limit(page_size))
    events = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
        "results": [
            {
                "id": str(e.id),
                "camera_id": e.camera_id,
                "malpractice_class": e.malpractice_class.value,
                "confidence_score": e.confidence_score,
                "triage_status": e.triage_status.value,
                "evidence_frame_path": e.evidence_frame_path,
                "detected_at": e.detected_at.isoformat(),
                "lh_mapping_id": str(e.lh_mapping_id) if e.lh_mapping_id else None,
            }
            for e in events
        ],
    }


@router.get("/stats")
async def detection_stats(db: AsyncSession = Depends(get_db)):
    """Aggregated detection statistics for dashboard widgets."""
    total_q = await db.execute(select(func.count(DetectionEvent.id)))
    total = total_q.scalar_one()

    auto_q = await db.execute(
        select(func.count(DetectionEvent.id)).where(
            DetectionEvent.triage_status == TriageStatus.AUTO_ALERTED
        )
    )
    auto_alerted = auto_q.scalar_one()

    pending_q = await db.execute(
        select(func.count(DetectionEvent.id)).where(
            DetectionEvent.triage_status == TriageStatus.REVIEW_PENDING
        )
    )
    pending = pending_q.scalar_one()

    r = await get_async_redis()
    queue_depth = await r.llen(REVIEW_QUEUE_KEY)

    return {
        "total_events": total,
        "auto_alerted": auto_alerted,
        "pending_review": pending,
        "redis_queue_depth": queue_depth,
    }


# ── WebSocket live feed ────────────────────────────────────────

class LiveFeedManager:
    """Manages active WebSocket connections per camera."""
    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, camera_id: str, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(camera_id, []).append(ws)

    def disconnect(self, camera_id: str, ws: WebSocket):
        if camera_id in self.connections:
            self.connections[camera_id].remove(ws)

    async def broadcast(self, camera_id: str, data: dict):
        dead = []
        for ws in self.connections.get(camera_id, []):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(camera_id, ws)


feed_manager = LiveFeedManager()


@router.websocket("/ws/live/{camera_id}")
async def websocket_live_feed(camera_id: str, websocket: WebSocket):
    """
    WebSocket endpoint: subscribe to real-time detection events for a camera.
    The dashboard connects here to display live detection overlays.
    """
    await feed_manager.connect(camera_id, websocket)
    r = await get_async_redis()
    pubsub = r.pubsub()
    await pubsub.subscribe(f"eids:events:{camera_id}")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await websocket.send_json(data)
    except WebSocketDisconnect:
        feed_manager.disconnect(camera_id, websocket)
        await pubsub.unsubscribe(f"eids:events:{camera_id}")
