#!/bin/bash
# usage (through gpujob):  gpujob env -u TMPDIR systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 bash scripts/experiments/e127/job_train.sh NAME KIND "EXTRA ARGS"   (trains both nets sequentially; resumes from last.pt)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 9
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NAME=$1; KIND=$2; EXTRA=$3
for net in finh_h0 finh_h1; do
  /usr/bin/python3 scripts/experiments/e127/e127_train.py --net $net --kind $KIND --out .scratch/e127/${NAME}_$net $EXTRA >> .scratch/e127/logs/train_${NAME}_$net.log 2>&1 || { echo FAIL $net; tail -5 .scratch/e127/logs/train_${NAME}_$net.log; exit 5; }
  tail -n 2 .scratch/e127/logs/train_${NAME}_$net.log | cut -c1-300
done
