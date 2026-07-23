"""
MinIO client for self-hosted S3-compatible object storage.
Handles evidence frame uploads and pre-signed URL generation.
"""
from __future__ import annotations

import io
import logging
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_minio_client: Minio | None = None


def get_minio_client() -> Minio:
    """Return a shared MinIO client instance."""
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        _ensure_buckets(_minio_client)
    return _minio_client


def _ensure_buckets(client: Minio) -> None:
    """Create required buckets if they don't exist."""
    for bucket in [settings.minio_bucket_evidence, settings.minio_bucket_models]:
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info("Created MinIO bucket: %s", bucket)
        except S3Error as exc:
            logger.error("MinIO bucket setup error for %s: %s", bucket, exc)


def upload_evidence_frame(
    object_name: str,
    data: bytes | BinaryIO,
    content_type: str = "image/jpeg",
    length: int = -1,
) -> str:
    """
    Upload an evidence frame to MinIO.

    Args:
        object_name: Path within the bucket, e.g. "2024/01/lh01/event_001.jpg"
        data: Raw bytes or file-like object.
        content_type: MIME type.
        length: Data length in bytes (-1 for unknown when streaming).

    Returns:
        The object name (used to generate pre-signed URLs).
    """
    client = get_minio_client()
    if isinstance(data, bytes):
        stream = io.BytesIO(data)
        length = len(data)
    else:
        stream = data

    client.put_object(
        bucket_name=settings.minio_bucket_evidence,
        object_name=object_name,
        data=stream,
        length=length,
        content_type=content_type,
    )
    logger.debug("Uploaded evidence frame: %s", object_name)
    return object_name


def get_presigned_url(
    object_name: str,
    expires: timedelta = timedelta(hours=24),
    bucket: str | None = None,
) -> str:
    """Generate a pre-signed URL for temporary access to an object."""
    client = get_minio_client()
    bucket = bucket or settings.minio_bucket_evidence
    return client.presigned_get_object(
        bucket_name=bucket,
        object_name=object_name,
        expires=expires,
    )


def download_object(object_name: str, bucket: str | None = None) -> bytes:
    """Download an object and return its raw bytes."""
    client = get_minio_client()
    bucket = bucket or settings.minio_bucket_evidence
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def delete_object(object_name: str, bucket: str | None = None) -> None:
    """Delete an object from MinIO."""
    client = get_minio_client()
    bucket = bucket or settings.minio_bucket_evidence
    client.remove_object(bucket, object_name)
