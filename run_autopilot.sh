#!/usr/bin/env bash
# Supervisor: restart autopilot.py (resumable) if it dies before writing READY.txt; at most 5 attempts.
cd "$(dirname "$0")"
export PYTHONUTF8=1
for i in 1 2 3 4 5; do
  echo "== supervisor attempt $i $(date +%H:%M:%S)"
  python -u autopilot.py && exit 0
  [ -f final_overnight/READY.txt ] && exit 1     # autopilot reported a definite failure: do not loop on it
  sleep 30
done
