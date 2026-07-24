"""
FastAPI application entry point.
"""
from __future__ import annotations

import sys
import asyncio
import logging

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.database import create_all_tables
from app.routers import streams, detections, alerts, auth, video

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Lifespan ───────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Exam Vigilance FastAPI backend...")

    # Ensure DB tables exist
    await create_all_tables()
    logger.info("Database tables verified")

    yield  # ← application is running

    # Shutdown
    from app.services.rtsp_capture import stop_all_streams
    stop_all_streams()
    logger.info("Exam Vigilance backend shut down cleanly")


# ── Application ────────────────────────────────────────────────

app = FastAPI(
    title="Exam Vigilance API",
    description=(
        "Real-time CCTV malpractice detection backend. "
        "Manages RTSP streams, inference results, and invigilator alerts."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── CORS (Django dashboard + local dev) ───────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        settings.fastapi_base_url,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(streams.router, prefix="/api/streams", tags=["Stream Management"])
app.include_router(video.router, prefix="/api/video", tags=["Video Analysis"])
app.include_router(detections.router, prefix="/api/detections", tags=["Detections"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])


# ── Health check ──────────────────────────────────────────────

@app.get("/api/health", tags=["Health"])
async def health_check():
    from app.services.rtsp_capture import get_active_cameras
    import datetime

    active_cameras = get_active_cameras()

    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "services": {
            "active_streams": len(active_cameras),
            "camera_ids": active_cameras,
        },
    })
