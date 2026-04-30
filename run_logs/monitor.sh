#!/bin/bash
cd /workspace/ml-intern-benchmark/target
for f in run_logs/*.log; do
  name=$(basename "$f" .log)
  status="RUN"
  if grep -aq "Training done" "$f" 2>/dev/null; then status="DONE"; fi
  if grep -aq "TEST  avg_surf_p=" "$f" 2>/dev/null; then status="TESTED"; fi
  if grep -aq "Traceback\|OutOfMemoryError\|Error: " "$f" 2>/dev/null; then status="ERROR"; fi
  # Convert \r to \n then grep summary lines
  best=$(tr '\r' '\n' < "$f" | grep -aE "^Epoch +[0-9]+ +\(.*val_avg_surf_p" | tail -3)
  echo "=== $name [$status] ==="
  echo "$best"
  if [ "$status" = "TESTED" ] || [ "$status" = "DONE" ]; then
    grep -aE "Best val:|TEST  avg_surf_p=" "$f" 2>/dev/null
  fi
done
