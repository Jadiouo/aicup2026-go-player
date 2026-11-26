#!/bin/bash
# E112 step 3a: X matrices on the pseudo-query dumps (real, and two shuffled nulls).  GPU KDE.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 9
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for args in "" "--shuffle 1" "--shuffle 2"; do
  /usr/bin/python3 scripts/experiments/e112/e112_stage2.py compute $args > .scratch/e112/logs/s2_compute${args// /_}.log 2>&1 || { echo FAILED "$args"; tail -5 .scratch/e112/logs/s2_compute${args// /_}.log; exit 5; }
  tail -n 2 .scratch/e112/logs/s2_compute${args// /_}.log | cut -c1-200
done
