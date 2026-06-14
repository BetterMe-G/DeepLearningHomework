#!/usr/bin/env bash
# Train all 10 social-avatar StyleCLIP mappers.
# By default uses one process per visible GPU (parallel), one mapper per GPU.
#
# Resumable: if checkpoints/mappers/<name>.pt already exists, that prompt is
# skipped. Per-prompt logs go to logs/mappers/<name>.log; per-prompt sample
# images (before/after every save_interval) go to checkpoints/mappers/<name>_samples/.
#
# Total wall time: ~10h on a single RTX 3090 / A100 (50k iters × 10 prompts).
#
# Usage:
#   bash scripts/train_all_mappers.sh                        # train all missing
#   bash scripts/train_all_mappers.sh holographic            # train only one
#   ITERATIONS=20000 bash scripts/train_all_mappers.sh       # shorter
#   GPU_IDS=0,1 bash scripts/train_all_mappers.sh            # pin to GPU 0/1

set -euo pipefail

cd "$(dirname "$0")/.."           # project root
mkdir -p checkpoints/mappers logs/mappers

ITERATIONS="${ITERATIONS:-50000}"
GPU_IDS="${GPU_IDS:-}"

# name => prompt
declare -a NAMES=(
  holographic
  cyber_tattoo
  golden_freckles
  pastel_split
  dark_academia
  egirl_heart
  baroque_oil
  kabuki
  crystal_skin
  neon_noir
)
declare -A PROMPTS=(
  [holographic]="a person with iridescent holographic hair"
  [cyber_tattoo]="a person with cyberpunk neon face tattoos"
  [golden_freckles]="a person with golden freckles constellation across the face"
  [pastel_split]="a person with pastel pink and platinum split dyed hair"
  [dark_academia]="a person with dark academia aesthetic and vintage round glasses"
  [egirl_heart]="a person with e-girl aesthetic, heart shaped cheek blush"
  [baroque_oil]="a baroque oil painting portrait of a person"
  [kabuki]="a person with kabuki theater white face makeup and red accent lines"
  [crystal_skin]="a person with crystals growing from the skin"
  [neon_noir]="a person in dramatic neon noir lighting, pink and blue rim light"
)

# Optional filter: train only the names passed on the command line.
if [ $# -gt 0 ]; then
  SELECTED=("$@")
else
  SELECTED=("${NAMES[@]}")
fi

declare -a JOBS=()
for name in "${SELECTED[@]}"; do
  prompt="${PROMPTS[$name]:-}"
  if [ -z "$prompt" ]; then
    echo "[skip] unknown mapper name: $name"
    continue
  fi
  out="checkpoints/mappers/${name}.pt"
  if [ -f "$out" ]; then
    echo "[skip] $name (checkpoint already exists at $out)"
    continue
  fi
  JOBS+=("$name")
done

if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "[$(date '+%F %T')] nothing to train."
  exit 0
fi

if [ -n "$GPU_IDS" ]; then
  IFS=',' read -r -a GPUS <<< "$GPU_IDS"
else
  mapfile -t GPUS < <(python - <<'PY'
import torch
n = torch.cuda.device_count()
if n <= 0:
    raise SystemExit("no cuda device found; set GPU_IDS manually or fix CUDA")
print("\n".join(str(i) for i in range(n)))
PY
)
fi

if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "[error] no GPU ids available."
  exit 1
fi

echo "[$(date '+%F %T')] queue size=${#JOBS[@]}, gpus=${GPUS[*]}"

declare -A PID_TO_GPU
declare -A PID_TO_NAME
declare -i NEXT_IDX=0

launch_job() {
  local gpu="$1"
  local name="$2"
  local prompt="${PROMPTS[$name]}"
  local out="checkpoints/mappers/${name}.pt"
  local log="logs/mappers/${name}.log"
  echo "[$(date '+%F %T')] [gpu $gpu] training '$name' :: \"$prompt\" -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_mapper.py \
    --description "$prompt" \
    --output "$out" \
    --iterations "$ITERATIONS" \
    --sample_dir "checkpoints/mappers/${name}_samples" \
    >"$log" 2>&1 &
  local pid=$!
  PID_TO_GPU[$pid]="$gpu"
  PID_TO_NAME[$pid]="$name"
}

# Fill GPU slots once.
for gpu in "${GPUS[@]}"; do
  if [ "$NEXT_IDX" -ge "${#JOBS[@]}" ]; then
    break
  fi
  launch_job "$gpu" "${JOBS[$NEXT_IDX]}"
  NEXT_IDX=$((NEXT_IDX + 1))
done

# Keep scheduling until all jobs are done.
while [ "${#PID_TO_GPU[@]}" -gt 0 ]; do
  finished_pid=""
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      finished_pid="$pid"
      break
    fi
  done

  if [ -z "$finished_pid" ]; then
    sleep 2
    continue
  fi

  gpu="${PID_TO_GPU[$finished_pid]}"
  name="${PID_TO_NAME[$finished_pid]}"
  if wait "$finished_pid"; then
    echo "[$(date '+%F %T')] [gpu $gpu] done '$name'"
  else
    echo "[$(date '+%F %T')] [gpu $gpu] failed '$name' (see logs/mappers/${name}.log)"
  fi
  unset 'PID_TO_GPU[$finished_pid]'
  unset 'PID_TO_NAME[$finished_pid]'

  if [ "$NEXT_IDX" -lt "${#JOBS[@]}" ]; then
    launch_job "$gpu" "${JOBS[$NEXT_IDX]}"
    NEXT_IDX=$((NEXT_IDX + 1))
  fi
done

echo "[$(date '+%F %T')] all done."
