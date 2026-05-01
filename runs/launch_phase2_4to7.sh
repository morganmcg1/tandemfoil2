#!/bin/bash
# Phase 2 second batch: launches on GPUs 4-7 once Phase 1 E-H finish.
# All 60 min runs with --lr_schedule_epochs 22

cd /workspace/ml-intern-benchmark/target
GROUP="mlintern-pai2-24h-v3-r5"
AGENT="ml-intern-r5"

# C: p1D - eidetic = does eidetic actually contribute?
CUDA_VISIBLE_DEVICES=4 nohup python train.py \
    --epochs 999 --max_minutes 60 --lr_schedule_epochs 22 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p2C-noeidetic" \
    > logs/p2C-noeidetic.log 2>&1 &
PC=$!
echo "p2C-noeidetic PID=$PC"

# D: p1D - global_cond = does global_cond actually contribute?
CUDA_VISIBLE_DEVICES=5 nohup python train.py \
    --epochs 999 --max_minutes 60 --lr_schedule_epochs 22 \
    --use_eidetic --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p2D-nogcond" \
    > logs/p2D-nogcond.log 2>&1 &
PD=$!
echo "p2D-nogcond PID=$PD"

# E: p1D + bigger model (n_hidden=192, n_head=6)
CUDA_VISIBLE_DEVICES=6 nohup python train.py \
    --epochs 999 --max_minutes 60 --lr_schedule_epochs 22 \
    --use_eidetic --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --n_hidden 192 --n_head 6 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p2E-h192" \
    > logs/p2E-h192.log 2>&1 &
PE=$!
echo "p2E-h192 PID=$PE"

# F: p1D + surf_weight=20 (loss reweighting)
CUDA_VISIBLE_DEVICES=7 nohup python train.py \
    --epochs 999 --max_minutes 60 --lr_schedule_epochs 22 \
    --use_eidetic --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --surf_weight 20 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p2F-surfw20" \
    > logs/p2F-surfw20.log 2>&1 &
PF=$!
echo "p2F-surfw20 PID=$PF"

echo "$PC $PD $PE $PF" >> runs/p2_pids.txt
