#!/bin/bash
# Sequentially re-evaluate finished checkpoints with the NaN-safe accumulator.
# Runs are mapped to wandb run ids; their checkpoint dirs follow models/model-<id>/.
set -e
cd "$(dirname "$0")/.."
GPU="${1:-6}"

declare -A RUNS=(
  [F-wide-W192]=8hjzo90j
  [baseline-default]=7pkeajkh
  [A-ema]=g5riv0jt
  [B-sw30]=xa1gcvq3
  [C-sn128]=7se8k7u2
  [D-cheap-combo]=moazbb42
  [E-deep-L8]=qoxkjjwk
  [G-scale-L8W192]=jdh48aci
  [H-W256]=hswqk8re
  [K-F-sn128]=uk8jdd77
  [L-F-lr2e4]=e7922ai6
  [N-F-pw2]=jl0l1wma
  [P-F-long]=vcvzpvhj
)

mkdir -p research/reeval

for name in "${!RUNS[@]}"; do
  rid="${RUNS[$name]}"
  ckpt="models/model-$rid/checkpoint.pt"
  out="research/reeval/${name}.json"
  if [ ! -f "$ckpt" ]; then
    echo "[SKIP] $name (no $ckpt)"; continue
  fi
  if [ -f "$out" ]; then
    echo "[CACHED] $name (already $out)"; continue
  fi
  echo "=== Re-eval $name ($rid) ==="
  CUDA_VISIBLE_DEVICES=$GPU python scripts/reeval_test.py \
    --checkpoint "$ckpt" \
    --also_val \
    --out_json "$out" 2>&1 | tail -15
  echo
done
echo "=== DONE ==="
ls -la research/reeval/