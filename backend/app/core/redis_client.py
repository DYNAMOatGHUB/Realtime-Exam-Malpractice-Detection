"""
Redis connection pool shared across the application.
Used by Celery workers and FastAPI routes for frame queuing and caching.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Async Redis pool (for FastAPI async routes)
_async_pool: Redis | None = None

# Sync Redis client (for Celery workers which run synchronously)
_sync_client = None


def get_sync_redis():
    """Return a synchronous Redis client (for Celery tasks)."""
    import redis as sync_redis
    global _sync_client
    if _sync_client is None:
        _sync_client = sync_redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
    return _sync_client


async def get_async_redis() -> Redis:
    """Return a shared async Redis client (for FastAPI routes)."""
    global _async_pool
    if _async_pool is None:
        _async_pool = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
    return _async_pool


async def close_async_redis() -> None:
    """Gracefully close the async Redis pool on shutdown."""
    global _async_pool
    if _async_pool is not None:
        await _async_pool.aclose()
        _async_pool = None


# ── Queue helpers ──────────────────────────────────────────────

FRAME_QUEUE_KEY = "eids:frames:{camera_id}"
ACTIVE_STREAMS_KEY = "eids:active_streams"
REVIEW_QUEUE_KEY = "eids:review_queue"


def frame_queue_key(camera_id: str) -> str:
    return FRAME_QUEUE_KEY.format(camera_id=camera_id)


def push_frame_sync(camera_id: str, frame_data: dict[str, Any]) -> None:
    """Push serialised frame metadata to Redis queue (sync, for capture threads)."""
    r = get_sync_redis()
    r.lpush(frame_queue_key(camera_id), json.dumps(frame_data))
    # Cap queue length to avoid memory growth
    r.ltrim(frame_queue_key(camera_id), 0, 299)


def pop_frame_sync(camera_id: str, timeout: int = 2) -> dict[str, Any] | None:
    """Blocking pop from frame queue (sync, for Celery workers)."""
    r = get_sync_redis()
    result = r.brpop(frame_queue_key(camera_id), timeout=timeout)
    if result:
        _, raw = result
        return json.loads(raw)
    return None


async def push_to_review_queue(event_payload: dict[str, Any]) -> None:
    """Push a mid-confidence detection event to the HEC review queue (async)."""
    r = await get_async_redis()
    await r.lpush(REVIEW_QUEUE_KEY, json.dumps(event_payload))


async def get_review_queue_length() -> int:
    r = await get_async_redis()
    return await r.llen(REVIEW_QUEUE_KEY)
