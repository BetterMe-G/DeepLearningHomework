#!/usr/bin/env bash
# Re-compute FID-50k_full and IS-50k on the latest snapshot.
# The metric JSONL is auto-appended to the snapshot's run dir.

set -euo pipefail
cd "$(dirname "$0")"

SNAP=$(ls -1t runs/*/network-snapshot-*.pkl 2>/dev/null | head -1 || true)
if [[ -z "${SNAP:-}" ]]; then
    echo "[ERROR] No snapshot found in runs/. Run train.sh first." >&2
    exit 1
fi
echo "[INFO] Using snapshot: $SNAP"

DATA="data/celeba64.zip"
if [[ ! -f "$DATA" ]]; then
    echo "[ERROR] $DATA not found." >&2
    exit 1
fi

mkdir -p samples/metrics

# calc_metrics.py prints one JSON line per metric; capture to a file for the report.
python3 ../stylegan2-ada-pytorch/calc_metrics.py \
    --metrics=fid50k_full,is50k \
    --data="$DATA" \
    --mirror=1 \
    --network="$SNAP" \
    2>&1 | tee samples/metrics/final_metrics.txt

echo "[INFO] Final metrics saved to samples/metrics/final_metrics.txt"
