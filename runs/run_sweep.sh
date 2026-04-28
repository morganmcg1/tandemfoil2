#!/bin/bash
# Sequentially run surf_weight sweep: 10, 15, 20, 25, 30
set -u
cd /workspace/senpai/target
export WANDB_MODE=offline

for sw in 10 15 20 25 30; do
  echo "===== START surf_weight=${sw} $(date -Iseconds) =====" | tee -a "runs/sw${sw}.log"
  python train.py \
    --agent charliepai2e5-alphonse \
    --wandb_name "charliepai2e5-alphonse/surf-weight-l1-sweep-sw${sw}" \
    --surf_weight ${sw}.0 2>&1 | tee -a "runs/sw${sw}.log"
  echo "===== DONE  surf_weight=${sw} $(date -Iseconds) =====" | tee -a "runs/sw${sw}.log"
done

echo "ALL SWEEP RUNS COMPLETE $(date -Iseconds)" >> runs/sw_sweep_done.flag
