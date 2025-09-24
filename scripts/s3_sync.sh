#!/usr/bin/env bash
set -euo pipefail

if ! command -v aws >/dev/null 2>&1; then
  echo "[s3_sync] AWS CLI is not installed. Please install awscli first." >&2
  exit 1
fi

BUCKET=${S3_BUCKET:-}
REGION=${AWS_DEFAULT_REGION:-ap-northeast-1}
if [ -z "$BUCKET" ]; then
  echo "[s3_sync] S3_BUCKET env var is empty" >&2
  exit 1
fi

echo "[s3_sync] Using bucket: s3://$BUCKET (region: $REGION)"

# Create bucket if not exists (ignore error if exists)
set +e
aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "[s3_sync] Creating bucket..."
  aws s3api create-bucket \
    --bucket "$BUCKET" \
    --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION" || true
fi
set -e

mkdir -p checkpoints weights

echo "[s3_sync] Syncing checkpoints/ -> s3://$BUCKET/checkpoints/"
aws s3 sync checkpoints/ "s3://$BUCKET/checkpoints/" --exclude "*.tmp"

echo "[s3_sync] Syncing weights/ -> s3://$BUCKET/weights/"
aws s3 sync weights/ "s3://$BUCKET/weights/" --exclude "*.tmp"

echo "[s3_sync] Done."

