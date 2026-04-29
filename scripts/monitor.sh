#!/bin/bash
cd "$(dirname "$0")/.."
runs="baseline A-ema B-sw30 C-sn128 D-cheap-combo E-deep-L8 F-wide-W192 G-scale-L8W192"
for extra in "$@"; do runs="$runs $extra"; done

printf "%-22s %-9s %-22s %-12s %-8s %-22s\n" "RUN" "STATUS" "LAST_EPOCH" "BEST_VAL" "TIME" "ALL_VAL_HISTORY"
printf -- "-%.0s" {1..130}
echo

for f in $runs; do
  log=logs/$f.log
  [ ! -f "$log" ] && continue
  # match python train.py with this wandb_name
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
  hist=$(grep -oE "val_avg_surf_p=[0-9.]+" "$log" 2>/dev/null | grep -oE "[0-9.]+" | tail -8 | tr '\n' ',' | sed 's/,$//')
  printf "%-22s %-9s %-22s %-12s %-8s %s\n" "$f" "$status" "${last_epoch:-—}" "${best:-—}" "${elapsed:-—}" "${hist:-—}"
done