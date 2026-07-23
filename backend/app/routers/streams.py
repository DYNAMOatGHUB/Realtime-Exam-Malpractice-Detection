"""
Stream management router — start/stop RTSP capture workers.
Only accessible by authenticated HEC or Admin users.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator

from app.core.config import get_settings
from app.services.rtsp_capture import (
    StreamConfig,
    start_stream,
    stop_stream,
    get_active_cameras,
)

router = APIRouter()
settings = get_settings()


class StartStreamRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    lh_mapping_id: str | None = None
    target_fps: int = 8

    @field_validator("target_fps")
    @classmethod
    def validate_fps(cls, v):
        if not 1 <= v <= 30:
            raise ValueError("target_fps must be between 1 and 30")
        return v

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id(cls, v):
        if not v.strip():
            raise ValueError("camera_id cannot be empty")
        return v.strip()


class StreamResponse(BaseModel):
    camera_id: str
    status: str
    message: str


@router.post("/start", response_model=StreamResponse, status_code=status.HTTP_201_CREATED)
async def start_stream_endpoint(req: StartStreamRequest):
    """
    Start an RTSP capture worker for a camera.
    Spawns a background thread that continuously reads the RTSP feed
    and pushes sampled frames to the Redis inference queue.
    """
    config = StreamConfig(
        camera_id=req.camera_id,
        rtsp_url=req.rtsp_url,
        lh_mapping_id=req.lh_mapping_id,
        target_fps=req.target_fps,
    )

    success = start_stream(config)
    if not success:
        active = get_active_cameras()
        if req.camera_id in active:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Camera '{req.camera_id}' is already capturing",
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Max camera limit ({settings.max_cameras}) reached",
        )

    # Trigger the first inference batch task immediately
    from app.services.inference_worker import process_frame_batch
    process_frame_batch.apply_async(args=[req.camera_id], queue="inference")

    return StreamResponse(
        camera_id=req.camera_id,
        status="started",
        message=f"Capture started at {req.target_fps}fps from {req.rtsp_url}",
    )


@router.delete("/stop/{camera_id}", response_model=StreamResponse)
async def stop_stream_endpoint(camera_id: str):
    """Stop an active RTSP capture worker."""
    success = stop_stream(camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active capture worker found for camera '{camera_id}'",
        )
    return StreamResponse(
        camera_id=camera_id,
        status="stopped",
        message="Capture worker stopped",
    )


@router.get("/active")
async def list_active_streams():
    """List all currently active camera streams."""
    cameras = get_active_cameras()
    return {"active_streams": cameras, "count": len(cameras)}
