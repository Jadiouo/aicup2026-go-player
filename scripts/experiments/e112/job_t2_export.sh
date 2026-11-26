#!/bin/bash
# E112 step 2 export: test2 14,200 player games, selected position set (P12), the three delivered S networks (final_model.pt of finh_h0 / finh_h1 / finh_full).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 9
[ "$(df -Pk . | awk 'NR==2{print $4}')" -ge 3000000 ] || { echo "disk low"; exit 8; }
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for n in finh_full finh_h0 finh_h1; do
  /usr/bin/python3 scripts/experiments/e112/e112_export.py --ckpt .scratch/e90/colab/$n/final_model.pt --games .scratch/e90/data/t_games.npz --rows .scratch/e90/data/export_spec_h.npz:test_player_t --pset P12 --out .scratch/e112/t2_pos_$n.npz > .scratch/e112/logs/t2_export_$n.log 2>&1 || { echo "FAILED $n"; tail -5 .scratch/e112/logs/t2_export_$n.log; exit 5; }
  tail -n 1 .scratch/e112/logs/t2_export_$n.log | cut -c1-300
done
