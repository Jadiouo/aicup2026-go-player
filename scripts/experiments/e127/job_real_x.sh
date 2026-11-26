#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 9
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/usr/bin/python3 scripts/experiments/e127/real_x.py "$1" 2>&1 | tee -a .scratch/e127/final/logs/real_x_$1.log
