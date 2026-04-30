#!/bin/bash
# Phase 5: 8 parallel ~8-hour final candidate runs
# Phase 4 winner: p4F (slice_num=16 + EMA decay=0.999 + lr=2e-3 + gcond + gradclip 0.1 + OneCycle pct_start=0.05)
# test_avg/mae_surf_p = 29.58
# Phase 5 explores variations and longer training on top of this winner.

cd /workspace/ml-intern-benchmark/target
GROUP="mlintern-pai2-24h-v3-r5"
AGENT="ml-intern-r5"

# A: BEST p4F config, 8h, schedule_epochs=280 (slice_num=16 ~80s/ep => 360 epochs in 8h, head-room)
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --epochs 999 --max_minutes 480 --lr_schedule_epochs 280 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 16 --use_ema --ema_decay 0.999 \
    --seed 0 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p5A-slice16-ema-8h-s0" \
    > logs/p5A-slice16-ema-8h-s0.log 2>&1 &
echo "p5A PID=$!"

# B: BEST seed 1
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --epochs 999 --max_minutes 480 --lr_schedule_epochs 280 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 16 --use_ema --ema_decay 0.999 \
    --seed 1 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p5B-slice16-ema-8h-s1" \
    > logs/p5B-slice16-ema-8h-s1.log 2>&1 &
echo "p5B PID=$!"

# C: BEST seed 2
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --epochs 999 --max_minutes 480 --lr_schedule_epochs 280 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 16 --use_ema --ema_decay 0.999 \
    --seed 2 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p5C-slice16-ema-8h-s2" \
    > logs/p5C-slice16-ema-8h-s2.log 2>&1 &
echo "p5C PID=$!"

# D: p4B-style (slice_num=32 + EMA) for ensemble diversity
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --epochs 999 --max_minutes 480 --lr_schedule_epochs 240 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 32 --use_ema --ema_decay 0.999 \
    --seed 0 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p5D-slice32-ema-8h" \
    > logs/p5D-slice32-ema-8h.log 2>&1 &
echo "p5D PID=$!"

# E: BEST + n_hidden=160 (larger capacity for OOD)
CUDA_VISIBLE_DEVICES=4 nohup python train.py \
    --epochs 999 --max_minutes 480 --lr_schedule_epochs 200 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 16 --use_ema --ema_decay 0.999 \
    --n_hidden 160 --n_head 8 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p5E-h160-slice16-ema" \
    > logs/p5E-h160-slice16-ema.log 2>&1 &
echo "p5E PID=$!"

# F: BEST + n_layers=6 (deeper with small slice)
CUDA_VISIBLE_DEVICES=5 nohup python train.py \
    --epochs 999 --max_minutes 480 --lr_schedule_epochs 220 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 16 --use_ema --ema_decay 0.999 \
    --n_layers 6 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p5F-l6-slice16-ema" \
    > logs/p5F-l6-slice16-ema.log 2>&1 &
echo "p5F PID=$!"

# G: BEST + slower EMA decay 0.9995 (smoother averaging for long runs)
CUDA_VISIBLE_DEVICES=6 nohup python train.py \
    --epochs 999 --max_minutes 480 --lr_schedule_epochs 280 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 16 --use_ema --ema_decay 0.9995 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p5G-slice16-ema9995" \
    > logs/p5G-slice16-ema9995.log 2>&1 &
echo "p5G PID=$!"

# H: BEST + dropout 0.05 (light regularization for OOD splits)
CUDA_VISIBLE_DEVICES=7 nohup python train.py \
    --epochs 999 --max_minutes 480 --lr_schedule_epochs 280 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 16 --use_ema --ema_decay 0.999 \
    --dropout 0.05 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p5H-slice16-ema-drop05" \
    > logs/p5H-slice16-ema-drop05.log 2>&1 &
echo "p5H PID=$!"

date +"%H:%M:%S"
