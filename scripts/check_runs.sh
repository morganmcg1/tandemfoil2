#!/bin/bash
# Status of all sweep runs based on log files
cd "$(dirname "$0")/.."
printf "%-20s %-9s %-22s %-12s %-8s\n" "RUN" "STATUS" "LAST_EPOCH" "BEST_VAL" "TIME"
printf -- "-%.0s" {1..90}
echo

for f in baseline A-ema B-sw30 C-sn128 D-cheap-combo E-deep-L8 F-wide-W192 G-scale-L8W192 \
         "$@"; do
  [ -z "$f" ] && continue
  log=logs/$f.log
  [ ! -f "$log" ] && continue
  # search for the run's wandb_name in process list (python train.py only)
  pid=$(pgrep -f "python train.py.*mlintern-pai2-r1/${f}([^a-zA-Z]|$)" 2>/dev/null | head -1)
  if [ -z "$pid" ]; then
    pid=$(pgrep -f "python train.py.*mlintern-pai2-r1/${f}" 2>/dev/null | head -1)
  fi
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    status="LIVE"
    elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  else
    status="DONE"
    elapsed=""
  fi
  last_epoch=$(grep -oE "Epoch +[0-9]+ \([0-9]+s\)" "$log" 2>/dev/null | tail -1)
  best=$(grep -oE "val_avg_surf_p=[0-9.]+ \*" "$log" 2>/dev/null | grep -oE "[0-9.]+" | sort -n | head -1)
  [ -z "$best" ] && best=$(grep -oE "val_avg_surf_p=[0-9.]+" "$log" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
  printf "%-20s %-9s %-22s %-12s %-8s\n" "$f" "$status" "${last_epoch:-—}" "${best:-—}" "${elapsed:-—}"
done