#!/bin/bash
# E110: v7q pseudo-query dumps, structures a, c, b, strictly one after another (6G cap, 2 threads).  Stops at the first failure (a first: REPRO check).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
W="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 env PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="
L=.scratch/e110/logs; mkdir -p $L
for s in a c b; do
  a=$(free -g | awk '/^Mem:/{print $7}'); echo "$(date +%T) struct $s start, available ${a}G" >> $L/chain.log
  $W .venv/bin/python scripts/experiments/e110/e110_t2dump.py --struct $s > $L/t2dump_$s.log 2>&1; rc=$?; echo "$(date +%T) struct $s rc=$rc" >> $L/chain.log
  [ $rc -ne 0 ] && exit $rc
done
touch .scratch/e110/dumps/all.done
