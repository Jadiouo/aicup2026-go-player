#!/bin/bash
# usage: run_cpu.sh <script> [args]   (CPU job, memory capped at 4G, .venv python)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" && CUDA_VISIBLE_DEVICES= exec systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_CACHE_DIR=.scratch/e127/numba_cache CUDA_VISIBLE_DEVICES= .venv/bin/python "$@"
