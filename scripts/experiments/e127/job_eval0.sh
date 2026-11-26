#!/bin/bash
# usage (through gpujob + systemd-run 6G): bash job_eval0.sh TAG "name=path{net}/last.pt,..." [M]
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 9
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/usr/bin/python3 scripts/experiments/e127/e127_eval0.py --tag $1 --models "$2" --M ${3:-24} 2>&1 | tee .scratch/e127/logs/eval0_$1.log
