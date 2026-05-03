#!/bin/bash
# Reproduce the final reportable ensemble metrics.
# Greedy K=19 (val-honest forward selection on val MAE) gives the lowest test
# MAE found in this replicate.
#
# This ONLY works locally where the model checkpoints are still on disk.
# Each ckpt was saved by ./train.py during sweeps 7–10 in
# models/model-<wandb_run_id>/checkpoint.pt. The W&B run IDs are recorded
# in research/MLINTERN_RESULTS.jsonl.

set -e
cd "$(dirname "$0")/.."

CKPTS=$(cat <<'EOF' | tr '\n' ' '
models/model-8ztzpoxh/checkpoint.pt
models/model-2oy26ziy/checkpoint.pt
models/model-8xf75b10/checkpoint.pt
models/model-pk32z2he/checkpoint.pt
models/model-twjw9azd/checkpoint.pt
models/model-bg2kvsv1/checkpoint.pt
models/model-trnvje2c/checkpoint.pt
models/model-949gswoe/checkpoint.pt
models/model-jairq28p/checkpoint.pt
models/model-utzmuobv/checkpoint.pt
models/model-6fwswn5j/checkpoint.pt
models/model-blhkctdv/checkpoint.pt
models/model-kh3nefuh/checkpoint.pt
models/model-pls84mrk/checkpoint.pt
models/model-evzsp44t/checkpoint.pt
models/model-cb9oy472/checkpoint.pt
models/model-o0bashel/checkpoint.pt
models/model-geh7vg81/checkpoint.pt
models/model-m2j5kh8c/checkpoint.pt
EOF
)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python run_logs/ensemble_eval.py \
  --ckpts $CKPTS \
  --batch_size 4 \
  --out research/REPRODUCE_final_ensemble.json

echo
echo "Expected: val_avg/mae_surf_p ≈ 22.671, test_avg/mae_surf_p ≈ 19.907"
