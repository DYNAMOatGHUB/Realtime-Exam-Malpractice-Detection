#!/usr/bin/env bash
# ============================================================
# ExamGuard AI — MinIO bucket setup
# Called once after the MinIO container is healthy
# ============================================================
set -euo pipefail

MINIO_HOST="${MINIO_HOST:-minio:9000}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
ALIAS="local"

echo "[MinIO Setup] Waiting for MinIO to be ready..."
until mc alias set "$ALIAS" "http://${MINIO_HOST}" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" 2>/dev/null; do
  sleep 2
done
echo "[MinIO Setup] Connected."

# Create buckets (ignore error if they already exist)
for BUCKET in evidence-frames model-weights; do
  if mc ls "${ALIAS}/${BUCKET}" &>/dev/null; then
    echo "[MinIO Setup] Bucket '${BUCKET}' already exists."
  else
    mc mb "${ALIAS}/${BUCKET}"
    echo "[MinIO Setup] Created bucket '${BUCKET}'."
  fi
done

# Set evidence-frames bucket policy to private (default)
mc policy set private "${ALIAS}/evidence-frames"
mc policy set private "${ALIAS}/model-weights"

echo "[MinIO Setup] Done."
