#!/usr/bin/env bash
# Show current status of all phase1 training jobs.
cd /workspace/ml-intern-benchmark/target

echo "=== $(date -u) ==="
echo ""
printf "%-25s %-25s %-8s %-9s %-15s %s\n" "Run" "BestVal" "BestEp" "CurEp" "AvgEpSec" "GPU%"
printf "%-25s %-25s %-8s %-9s %-15s %s\n" "---" "------" "------" "------" "--------" "----"

for f in logs/phase1/*.log; do
  name=$(basename $f .log)

  # Best val so far (with `*` mark)
  best=$(grep -oE "val_avg_surf_p=[0-9.]+ \*" "$f" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  # Best epoch (the one with `*` on its line)
  bestep=$(grep -B0 -E "val_avg_surf_p=[0-9.]+ \*" "$f" 2>/dev/null | grep -oE "^Epoch +[0-9]+" | grep -oE "[0-9]+" | tail -1)
  # Current epoch (highest seen)
  curep=$(grep -oE "Epoch +[0-9]+/" $f 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  # Avg epoch time (simple: last reported)
  avgep=$(grep -oE "^Epoch +[0-9]+ \([0-9]+s\)" $f 2>/dev/null | tail -1 | grep -oE "[0-9]+s")

  pid=$(grep -m1 " $name " logs/phase1.pids 2>/dev/null | awk '{print $3}')
  if [[ -n $pid ]] && kill -0 $pid 2>/dev/null; then
    alive=run
  else
    alive=DONE
  fi

  printf "%-25s %-25s %-8s %-9s %-15s %s\n" "$name" "${best:-...}" "${bestep:-..}" "${curep:-..}" "${avgep:-..}" "$alive"
done

echo ""
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -8
