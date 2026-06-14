#!/usr/bin/env bash
# Server-side one-shot launcher. Run ONCE on the A800 host after rsyncing
# stylegan2_ada/ into the repo root.
#
# What it does:
#   1. Verifies python, torch, CUDA, NCCL.
#   2. Resolves the CelebA source directory.
#   3. Runs prepare_data.sh (skips if celeba64.zip already exists).
#   4. Launches train.sh in background with nohup, captures PID + log path.
#   5. Prints monitoring commands.
#
# Override CelebA path with:  CELEBA_SRC=/path/to/img_align_celeba bash launch_remote.sh
# Override training config:   KIMG=5000 GPUS=4 bash launch_remote.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo " stylegan2_ada remote launcher"
echo " host:    $(hostname)"
echo " cwd:     $(pwd)"
echo " date:    $(date -Iseconds)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Environment check
# ---------------------------------------------------------------------------
echo
echo "[1/4] Environment check"
python3 - <<'PY'
import sys
print(f"  python : {sys.version.split()[0]}")
try:
    import torch
    print(f"  torch  : {torch.__version__}")
    print(f"  cuda   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  gpu {i}  : {torch.cuda.get_device_name(i)}")
        print(f"  nccl   : {'OK' if torch.distributed.is_nccl_available() else 'MISSING'}")
    else:
        print("  [FATAL] no CUDA — training impossible")
        sys.exit(2)
except ImportError:
    print("  [FATAL] torch not installed — pip install torch torchvision")
    sys.exit(2)
try:
    import ninja; print("  ninja  : OK")
except ImportError:
    print("  ninja  : MISSING (some custom ops may fail to JIT compile)")
PY
[[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "[FATAL] env check failed"; exit 1; }

# ---------------------------------------------------------------------------
# 2. CelebA source resolution
# ---------------------------------------------------------------------------
echo
echo "[2/4] CelebA source"
DEFAULT_SRC="../data/celeba/img_align_celeba"
SRC="${CELEBA_SRC:-$DEFAULT_SRC}"
if [[ ! -d "$SRC" ]]; then
    echo "  [FATAL] $SRC not found."
    echo "          Set CELEBA_SRC=/path/to/img_align_celeba (with .jpg or .png images inside)"
    exit 1
fi
N_IMG=$(find "$SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l | tr -d ' ')
echo "  source : $SRC"
echo "  images : $N_IMG"
if (( N_IMG < 100000 )); then
    echo "  [WARN] $N_IMG images — expected ~200k for full CelebA. Continuing anyway."
fi

# ---------------------------------------------------------------------------
# 3. Prepare dataset
# ---------------------------------------------------------------------------
echo
echo "[3/4] Prepare CelebA 64x64 ZIP"
mkdir -p data
if [[ -f data/celeba64.zip ]]; then
    echo "  [SKIP] data/celeba64.zip already exists ($(du -h data/celeba64.zip | cut -f1))"
else
    echo "  Running prepare_data.sh (CELEBA_SRC=$SRC) ..."
    CELEBA_SRC="$SRC" bash prepare_data.sh
fi

# ---------------------------------------------------------------------------
# 4. Launch training in background
# ---------------------------------------------------------------------------
echo
echo "[4/4] Launch training"
mkdir -p runs samples
LOG=runs/train_$(date +%Y%m%d_%H%M%S).log
GPUS="${GPUS:-1}"
BATCH="${BATCH:-32}"
KIMG="${KIMG:-25000}"
RUN_NAME="${RUN_NAME:-celeba64}"

# Use a small wrapper to propagate env vars cleanly into the torchrun process
nohup env GPUS="$GPUS" BATCH="$BATCH" KIMG="$KIMG" \
    bash train.sh > "$LOG" 2>&1 &
PID=$!
disown $PID 2>/dev/null || true

# Give it a moment so the .pid and log file actually appear
sleep 2
if ! kill -0 $PID 2>/dev/null; then
    echo "  [FATAL] training process exited immediately. Tail of $LOG:"
    tail -40 "$LOG" || true
    exit 1
fi

echo "  PID    : $PID"
echo "  log    : $(pwd)/$LOG"
echo "  outdir : $(pwd)/runs/"
echo "  cfg    : GPUS=$GPUS BATCH=$BATCH KIMG=$KIMG"

echo
echo "=========================================="
echo " Training launched in background."
echo "=========================================="
cat <<'EOF'

Monitor:
  tail -f $LOG                                     # launcher log (first 30s, then defer to runs/)
  tail -f runs/*/log.txt                           # per-tick training log
  tail -f runs/*/metric-fid50k_full.jsonl          # FID curve (should decrease)
  nvidia-smi                                       # GPU utilization

When training reaches a snapshot you like (~25k kimg, ~10-15h on 1 A800):
  bash generate_samples.sh                         # 8x8 face grid
  bash eval_metrics.sh                             # final FID/IS (re-runs on latest snapshot)
  bash interpolate.sh                              # lerp + slerp grids + GIFs

To pull results back to your local machine (run on local Mac):
  rsync -avP <server>:/hpc_stor03/sjtu_home/siru.ge/DeepLearningHomework/stylegan2_ada/samples/ ./stylegan2_ada_samples/
  rsync -avP <server>:/hpc_stor03/sjtu_home/siru.ge/DeepLearningHomework/stylegan2_ada/runs/ ./stylegan2_ada_runs/
EOF
