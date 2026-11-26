#!/bin/bash
# E127 step B export (GPU).  Run through: gpujob env -u TMPDIR systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 bash scripts/experiments/e127/job_export.sh
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 9
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
E=scripts/experiments/e127/e127_export.py; O=.scratch/e127
for net in finh_h0 finh_h1; do
  CK=.scratch/e90/colab/$net/final_model.pt
  [ -f $O/pos_${net}_test.npz ] || /usr/bin/python3 $E --ckpt $CK --games .scratch/e90/data/t_games.npz --rows .scratch/e90/data/export_spec_h.npz:test_player_t --pset P2_59 --out $O/pos_${net}_test.npz > $O/logs/exp_${net}_test.log 2>&1 || { echo FAIL $net test; exit 5; }
  for part in val trA trB; do
    [ -f $O/pos_${net}_$part.npz ] || /usr/bin/python3 $E --ckpt $CK --games .scratch/e90/data/uh_games.npz --rows $O/rows_${net}_$part.npy --pset P2_59 --out $O/pos_${net}_$part.npz > $O/logs/exp_${net}_$part.log 2>&1 || { echo FAIL $net $part; tail -5 $O/logs/exp_${net}_$part.log; exit 5; }
    tail -n 1 $O/logs/exp_${net}_$part.log | cut -c1-250
  done
done
echo ALL DONE
