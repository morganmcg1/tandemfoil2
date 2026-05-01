#!/usr/bin/env bash
# Run test evaluation on top candidate checkpoints. Outputs per-checkpoint
# JSON + a combined results file.
set -u

cd "$(dirname "$0")/.."
mkdir -p logs/test_eval research/eval_outputs

GPU="${1:-0}"

# Each entry: <run_id>:<short_name>:<expected_val>
CANDIDATES=(
  "ns6s7av2:r3-onecycle-ema-w192-150:30.8866"
  "x13i7anx:r3-onecycle-ema-w192-seed1-150:30.8793"
  "ds51n23u:r4-onecycle-ema-200:30.9431"
  "xkovi6vb:r3-onecycle-ema-seed2:31.8952"
  "v5866ues:r3-onecycle-ema-w192-noise-150:32.2284"
  "jy3qdhhm:r2-l1-onecycle-ema:39.3342"
)

for entry in "${CANDIDATES[@]}"; do
  run_id="${entry%%:*}"
  rest="${entry#*:}"
  short="${rest%%:*}"
  expected="${rest##*:}"

  ckpt="models/model-${run_id}/checkpoint.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[SKIP] $short ($run_id) — checkpoint missing"
    continue
  fi

  outfile="research/eval_outputs/${short}.json"
  logfile="logs/test_eval/${short}.log"
  echo "Eval $short (val=$expected) → $outfile"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_checkpoint.py \
    --checkpoint "$ckpt" \
    --bf16 \
    --out_json "$outfile" \
    > "$logfile" 2>&1
  printf "  test avg/mae_surf_p = "
  python -c "import json; d=json.load(open('$outfile')); print(d['test_avg']['avg/mae_surf_p'])" 2>/dev/null || echo "?"
done
