#!/bin/bash
# Quick status across all running training runs.
LOG_DIR="/workspace/ml-intern-benchmark/target/launch_logs"

printf "%-40s  %-7s  %-12s  %-13s  %s\n" "RUN" "STATUS" "EPOCH" "VAL_SURF_P" "BEST"
printf -- '-%.0s' {1..100}; echo
for f in $LOG_DIR/*.log; do
  name=$(basename "$f" .log)
  # Find PID for this run
  pid=$(awk -v n="$name" '$0 ~ n {print $3}' $LOG_DIR/runs.tsv | tr -d 'pid=' | tail -1)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then status="RUN"; else status="STOPPED"; fi

  # Latest epoch summary line:  Epoch  N (Ts)  ...  val_avg_surf_p=X
  last=$(grep -aE "^Epoch +[0-9]+ +\([0-9]+s\)" "$f" | tail -1)
  if [ -n "$last" ]; then
    ep=$(echo "$last" | sed -E 's/^Epoch +([0-9]+).*/\1/')
    val=$(echo "$last" | grep -oE 'val_avg_surf_p=[0-9.]+' | sed 's/val_avg_surf_p=//')
    star=$(echo "$last" | grep -oE '\*$' || echo "")
  else
    ep=""; val=""; star=""
  fi

  # Best val so far (lowest)
  best=$(grep -aoE 'val_avg_surf_p=[0-9.]+ \*' "$f" | sed 's/val_avg_surf_p=//;s/ \*//' | sort -n | head -1)

  printf "%-40s  %-7s  %-12s  %-13s  %s\n" \
    "$name" "$status" "${ep:-?}" "${val:-?}${star}" "${best:-?}"
done
echo ""
echo "GPU memory:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -10
