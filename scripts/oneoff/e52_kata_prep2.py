"""E52 step 5 scale-up prep: KataGo input planes for the sim rows capped at 24 games per player (train + eval players of the p1111 sim; no H), K own-move positions per game
(evenly spaced in own moves [2,60)), in blocks of 10k games -> .scratch/e52/kata2/blk{i}_{bits,glob,loc,rows}.npy.  CPU, 6 workers (nice).  Also writes cap_rows.npy (sim row ids kept).
Usage: e52_kata_prep2.py [--k 6] [--cap 24]"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "oneoff")]
import json, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
from scripts.experiments.e13 import encode as E, lib
from gogame.sgf import parse_sgf
import e52_kata_prep as P1
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=6); ap.add_argument("--cap", type=int, default=24); a = ap.parse_args()
P1.K = a.k
OUT = ROOT / ".scratch" / "e52" / "kata2"; OUT.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    t0 = time.perf_counter()
    m = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet"); rng = np.random.default_rng(521)
    keep = []
    for p, idx in m.groupby("player_id").groups.items():
        idx = np.asarray(idx)
        if len(idx) >= 28: keep += list(rng.permutation(idx)[: a.cap])
    keep = np.sort(np.array(keep)); np.save(OUT / "cap_rows.npy", keep); sel = m.loc[keep]; print("rows", len(sel), sel.groupby("grp").player_id.nunique().to_dict(), flush=True)
    sg = E.read_sgf_for_uids(lib.TRAIN_CSV_DIR, set(sel.uid)); print("sgf read", len(sg), f"{time.perf_counter() - t0:.0f}s", flush=True)
    sgm = dict(zip(sg.uid, sg.sgf)); items = [(r.uid, int(r.color), sgm[r.uid], int(r.Index)) for r in sel.itertuples()]; del sg, sgm
    BLK = 10000
    with Pool(6) as pool:
        for bi, s in enumerate(range(0, len(items), BLK)):
            chunks = [items[i:i + 100] for i in range(s, min(s + BLK, len(items)), 100)]
            res = [r for r in pool.map(P1.work, chunks, chunksize=1) if r is not None]
            np.save(OUT / f"blk{bi}_bits.npy", np.concatenate([r[0] for r in res])); np.save(OUT / f"blk{bi}_glob.npy", np.concatenate([r[1] for r in res]))
            rw = np.concatenate([r[3] for r in res]); np.save(OUT / f"blk{bi}_loc.npy", np.concatenate([r[2] for r in res])); np.save(OUT / f"blk{bi}_rows.npy", rw); np.save(OUT / f"blk{bi}_color.npy", m["color"].to_numpy()[rw[:, 0]].astype(np.int8))
            print(f"block {bi} done {time.perf_counter() - t0:.0f}s", flush=True)
    (OUT / "PREP_DONE").write_text("ok")
