#!/bin/bash
# Print best val_avg/mae_surf_p observed so far in each Phase log

cd /workspace/ml-intern-benchmark/target

show_run() {
    local f="$1"
    [ -f "$f" ] || return
    echo "=== $(basename "$f" .log) ==="
    # Extract epoch summary lines
    local lines=$(tr '\r' '\n' < "$f" | grep -E "^Epoch.*\(.*s\)")
    if [ -z "$lines" ]; then
        echo "  no epoch lines yet"
        return
    fi
    echo "$lines" | tail -5
    # Best val_avg/mae_surf_p so far:
    local best=$(echo "$lines" | grep -oE "val_avg_surf_p=[0-9.]+" | sed 's/val_avg_surf_p=//' | sort -g | head -1)
    local nepochs=$(echo "$lines" | wc -l)
    echo "  → epochs=$nepochs  best_val_avg_surf_p=$best"
    # Test result if present
    local testline=$(grep "TEST" "$f" | head -1)
    [ -n "$testline" ] && echo "  → $testline"
}

for f in logs/p*.log; do
    show_run "$f"
done

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | head -10
