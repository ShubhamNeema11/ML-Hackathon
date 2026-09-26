#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTHONUTF8=1
for i in 1 2 3 4 5; do
  echo "== supervisor3 attempt $i $(date +%H:%M:%S)"
  python -u autopilot3.py && exit 0
  [ -f final_overnight_v3/READY_v3.txt ] && exit 1
  sleep 30
done
