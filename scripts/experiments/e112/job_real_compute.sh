#!/bin/bash
# E112 amendment 1: X on the real test2 queries (GPU KDE, ~minutes)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 9
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/usr/bin/python3 scripts/experiments/e112/e112_real.py compute > .scratch/e112/logs/a1_real_compute.log 2>&1 || { echo FAILED; tail -8 .scratch/e112/logs/a1_real_compute.log; exit 5; }
tail -n 3 .scratch/e112/logs/a1_real_compute.log | cut -c1-200
