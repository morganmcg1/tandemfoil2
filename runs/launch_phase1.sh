#!/bin/bash
# Phase 1: 4 parallel diagnostic runs, 30 min each, GPUs 0-3
# All experiments use 30 min cap with --max_minutes 30

cd /workspace/ml-intern-benchmark/target
export WANDB_PROJECT WANDB_ENTITY WANDB_API_KEY

GROUP="mlintern-pai2-24h-v3-r5"
AGENT="ml-intern-r5"

# A) Baseline (current default config)
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --epochs 999 --max_minutes 30 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p1-baseline" \
    > logs/p1-baseline.log 2>&1 &
echo "p1-baseline PID=$!"

# B) +Gumbel-Softmax eidetic attention (Transolver++)
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --epochs 999 --max_minutes 30 \
    --use_eidetic \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p1-eidetic" \
    > logs/p1-eidetic.log 2>&1 &
echo "p1-eidetic PID=$!"

# C) +Global conditioning embedding
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --epochs 999 --max_minutes 30 \
    --use_global_cond \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p1-globalcond" \
    > logs/p1-globalcond.log 2>&1 &
echo "p1-globalcond PID=$!"

# D) Full Transolver++ recipe: eidetic + global_cond + grad_clip + onecycle + lr=1e-3
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --epochs 999 --max_minutes 30 \
    --use_eidetic --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 1e-3 --pct_start 0.1 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p1-tplus-recipe" \
    > logs/p1-tplus-recipe.log 2>&1 &
echo "p1-tplus-recipe PID=$!"

