#!/bin/bash
# usage (gpujob + systemd-run 6G):  bash job_x.sh NAME "ckpt-template-with-{net}" M [extra args e.g. --shuffle 1 --reps 10-14]
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 9
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NAME=$1; CK=$2; M=$3; shift 3
/usr/bin/python3 scripts/experiments/e127/e127_x.py --name $NAME --models "$CK" --M $M "$@" 2>&1 | tee -a .scratch/e127/logs/x_$NAME.log
