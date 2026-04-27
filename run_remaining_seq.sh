#!/usr/bin/env bash
# Sequential sweep for PR #255 remaining variants.
# Waits for any running train.py (no-film-s0), then runs each variant in sequence on GPU 0.
# Priority order: merge-criterion runs first, then ablations.
cd "$(dirname "$0")"
LOGS="$(pwd)/sweep_logs"
mkdir -p "$LOGS"
SWEEP_LOG="$LOGS/sweep_seq.log"

WANDB_PROJECT="${WANDB_PROJECT:-senpai-charlie-wilson-charlie-r5}"
WANDB_ENTITY="${WANDB_ENTITY:-wandb-applied-ai-team}"

# Wait for any existing no-film-s0 run to finish (PID 3862 if still alive).
echo "[$(date +%H:%M:%S)] Waiting for any running train.py to finish..." | tee -a "$SWEEP_LOG"
while pgrep -f "experiment_name charliepai2c5-nezuko/no-film-s0" >/dev/null 2>&1; do
    sleep 10
done
echo "[$(date +%H:%M:%S)] no-film-s0 finished, starting sequential sweep." | tee -a "$SWEEP_LOG"

run() {
    local name=$1; shift
    echo "[$(date +%H:%M:%S)] START $name args=$*" | tee -a "$SWEEP_LOG"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --agent charliepai2c5-nezuko \
        --experiment_name "charliepai2c5-nezuko/$name" \
        --wandb_group nezuko \
        --wandb_name "nezuko/$name" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        "$@" >"$LOGS/$name.log" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] END $name (exit $rc)" | tee -a "$SWEEP_LOG"
    return 0
}

# Priority 1: merge-criterion variants
run "no-film-s1"   --nofilm --seed 1
run "film-d32-s0"  --film true --film_cond_dim 32 --seed 0
run "film-d32-s1"  --film true --film_cond_dim 32 --seed 1

# Priority 2: ablations (run if time allows)
run "film-d64-s0"  --film true --film_cond_dim 64 --seed 0
run "film-d64-s1"  --film true --film_cond_dim 64 --seed 1
run "film-d32-s2"  --film true --film_cond_dim 32 --seed 2
run "film-d16-s0"  --film true --film_cond_dim 16 --seed 0

echo "[$(date +%H:%M:%S)] SWEEP COMPLETE" | tee -a "$SWEEP_LOG"
