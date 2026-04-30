#!/bin/bash
# Phase 3: 8 parallel 3-hour runs based on Phase 2 winner (p2C noeidetic)
# Winner config: gcond + max_grad_norm=0.1 + OneCycle + lr=1e-3 (no eidetic)

cd /workspace/ml-intern-benchmark/target
GROUP="mlintern-pai2-24h-v3-r5"
AGENT="ml-intern-r5"

# A: p2C control 3h (seed 0)
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --epochs 999 --max_minutes 180 --lr_schedule_epochs 70 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --seed 0 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p3A-noeid-3h-s0" \
    > logs/p3A-noeid-3h-s0.log 2>&1 &
echo "p3A PID=$!"

# B: p2C control 3h seed 1 (variance check)
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --epochs 999 --max_minutes 180 --lr_schedule_epochs 70 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --seed 1 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p3B-noeid-3h-s1" \
    > logs/p3B-noeid-3h-s1.log 2>&1 &
echo "p3B PID=$!"

# C: p2C + EMA decay=0.999 (could smooth final convergence)
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --epochs 999 --max_minutes 180 --lr_schedule_epochs 70 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --use_ema --ema_decay 0.999 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p3C-noeid-ema" \
    > logs/p3C-noeid-ema.log 2>&1 &
echo "p3C PID=$!"

# D: p2C + n_hidden=192 (medium bigger model, more capacity)
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --epochs 999 --max_minutes 180 --lr_schedule_epochs 60 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --n_hidden 192 --n_head 6 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p3D-noeid-h192" \
    > logs/p3D-noeid-h192.log 2>&1 &
echo "p3D PID=$!"

# E: p2C + slice_num=32 (T++ industrial recommended)
CUDA_VISIBLE_DEVICES=4 nohup python train.py \
    --epochs 999 --max_minutes 180 --lr_schedule_epochs 70 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --slice_num 32 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p3E-noeid-slice32" \
    > logs/p3E-noeid-slice32.log 2>&1 &
echo "p3E PID=$!"

# F: p2C + n_layers=6 (deeper model)
CUDA_VISIBLE_DEVICES=5 nohup python train.py \
    --epochs 999 --max_minutes 180 --lr_schedule_epochs 60 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --n_layers 6 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p3F-noeid-l6" \
    > logs/p3F-noeid-l6.log 2>&1 &
echo "p3F PID=$!"

# G: p2A with eidetic (Gumbel-Softmax) for direct comparison at 3h
CUDA_VISIBLE_DEVICES=6 nohup python train.py \
    --epochs 999 --max_minutes 180 --lr_schedule_epochs 60 \
    --use_eidetic --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p3G-eid-3h" \
    > logs/p3G-eid-3h.log 2>&1 &
echo "p3G PID=$!"

# H: p2C + lr=2e-3 (higher peak LR, more aggressive)
CUDA_VISIBLE_DEVICES=7 nohup python train.py \
    --epochs 999 --max_minutes 180 --lr_schedule_epochs 70 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.1 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p3H-noeid-lr2e3" \
    > logs/p3H-noeid-lr2e3.log 2>&1 &
echo "p3H PID=$!"

date +"%H:%M:%S"
