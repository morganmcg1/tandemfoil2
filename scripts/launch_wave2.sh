#!/bin/bash
# Wave 2 launcher. F (n_hidden=192, n_head=6, EMA=0.999, bf16) is wave 1 winner.
# Usage: bash scripts/launch_wave2.sh [<gpu_id> <variant>]
# Without args, launches all 7 variants on GPUs 1-7.

cd "$(dirname "$0")/.."

launch() {
  local gpu=$1; local name=$2; shift 2
  local args="$@"
  echo "GPU $gpu: $name ($args)"
  nohup env CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --agent ml-intern-r1 \
    --wandb_group mlintern-pai2-r1 \
    --wandb_name "mlintern-pai2-r1/$name" \
    $args \
    > "logs/$name.log" 2>&1 &
  echo $! > "logs/$name.pid"
  sleep 3
}

# Wave 2 variants (all use bf16 + EMA + grad_clip=1.0 — proven from wave 1)
# H: F + n_hidden=256 (push width)        — ~200s/ep, 40 ep = 133min
# I: F + n_layers=8 (push depth in F)     — ~230s/ep, 35 ep = 134min
# J: F exact + ema=0.9999 (more stable)   — ~148s/ep, 60 ep = 148min
# K: F + slice_num=128                    — ~165s/ep, 50 ep = 138min
# L: F + lr=2e-4 (lower LR for tuning)    — ~148s/ep, 60 ep = 148min
# M: F + warmup=0.05 + lr=8e-4 (test higher LR with warmup) — ~148s/ep, 60 ep = 148min
# N: F + p_loss_weight=2.0 (extra pressure weight) — ~148s/ep, 60 ep = 148min
# Final: best of wave2 + long training

case "${1:-all}" in
  H|h) launch 1 "H-W256-bf16-ema" --n_hidden 256 --n_head 8 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 40 ;;
  I|i) launch 2 "I-L8-W192-bf16-ema" --n_layers 8 --n_hidden 192 --n_head 6 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 35 ;;
  J|j) launch 3 "J-F-ema9999" --n_hidden 192 --n_head 6 --ema_decay 0.9999 --amp_dtype bf16 --grad_clip 1.0 --epochs 60 ;;
  K|k) launch 4 "K-F-sn128" --n_hidden 192 --n_head 6 --slice_num 128 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 50 ;;
  L|l) launch 5 "L-F-lr2e4" --n_hidden 192 --n_head 6 --lr 0.0002 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 60 ;;
  M|m) launch 6 "M-F-warmup-lr8e4" --n_hidden 192 --n_head 6 --lr 0.0008 --warmup_frac 0.05 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 60 ;;
  P|p) launch 0 "P-F-long" --n_hidden 192 --n_head 6 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 80 ;;
  Q|q) launch 0 "Q-W256-L8" --n_hidden 256 --n_head 8 --n_layers 8 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 25 ;;
  N|n) launch 7 "N-F-pw2" --n_hidden 192 --n_head 6 --p_loss_weight 2.0 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 60 ;;
  all)
    launch 1 "H-W256-bf16-ema" --n_hidden 256 --n_head 8 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 40
    launch 2 "I-L8-W192-bf16-ema" --n_layers 8 --n_hidden 192 --n_head 6 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 35
    launch 3 "J-F-ema9999" --n_hidden 192 --n_head 6 --ema_decay 0.9999 --amp_dtype bf16 --grad_clip 1.0 --epochs 60
    launch 4 "K-F-sn128" --n_hidden 192 --n_head 6 --slice_num 128 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 50
    launch 5 "L-F-lr2e4" --n_hidden 192 --n_head 6 --lr 0.0002 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 60
    launch 6 "M-F-warmup-lr8e4" --n_hidden 192 --n_head 6 --lr 0.0008 --warmup_frac 0.05 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 60
    launch 7 "N-F-pw2" --n_hidden 192 --n_head 6 --p_loss_weight 2.0 --ema_decay 0.999 --amp_dtype bf16 --grad_clip 1.0 --epochs 60
    ;;
  *)
    echo "Usage: $0 [H|I|J|K|L|M|N|all]"
    exit 1
    ;;
esac