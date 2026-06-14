#!/usr/bin/env bash
# Generate an 8x8 sample grid using the latest snapshot.

set -euo pipefail
cd "$(dirname "$0")"

SNAP=$(ls -1t runs/*/network-snapshot-*.pkl 2>/dev/null | head -1 || true)
if [[ -z "${SNAP:-}" ]]; then
    echo "[ERROR] No snapshot found in runs/. Run train.sh first." >&2
    exit 1
fi
echo "[INFO] Using snapshot: $SNAP"

mkdir -p samples/grid
python3 ../stylegan2-ada-pytorch/generate.py \
    --outdir=samples/grid \
    --trunc=1.0 \
    --seeds=0-63 \
    --network="$SNAP"

# Concatenate the 64 individual PNGs into a single 8x8 grid for easy viewing.
python3 - <<'PY'
import glob
from pathlib import Path
import numpy as np
from PIL import Image

paths = sorted(glob.glob("samples/grid/seed*.png"))
if not paths:
    raise SystemExit("no seed*.png found in samples/grid/")
imgs = [np.array(Image.open(p)) for p in paths]
# Stack 8x8
rows = [np.concatenate(imgs[r*8:(r+1)*8], axis=1) for r in range(8)]
grid = np.concatenate(rows, axis=0)
out = Path("samples/grid/grid_8x8.png")
Image.fromarray(grid).save(out)
print(f"[INFO] Saved 8x8 grid to {out} ({grid.shape[1]}x{grid.shape[0]})")
PY
