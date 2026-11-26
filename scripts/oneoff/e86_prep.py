"""E86 CPU prep (local, <= 2 workers): ALL own-move positions (own idx in [2, min(120, n_own))) of the games in an items pickle [(uid, color, sgf, rid)] -> .scratch/e86/<stage>/blocks/{name}_{bits,glob,loc,rows,oidx,rep,color}.npy
(bits written last; 1,000 games per block).  rep = how many times the position occurs in the E80 24-position set (work24: window0 own_move_idx_eval(n_own,12) + late_idx(n_own,12), duplicates kept) -> the K=24 control is rebuilt from the same tokens.
Maths = e79_prep_local.work24 (imported helpers, not copied).  Backpressure: <= --maxpend blocks of bits waiting for the GPU half.  usage: e86_prep.py STAGE ITEMS.pkl [--workers 2] [--maxpend 6]"""
import os, sys, time, argparse, pickle
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "oneoff")]
import numpy as np
from multiprocessing import Pool
import e79_prep_local as P
from scripts.experiments.e13 import encode as E, lib
from gogame.sgf import parse_sgf
HI = 120; TEST_OFF = 10_000_000; _CHK = [None]
def _chk_set():
    if _CHK[0] is None:                       # full stage: bundle games reuse the E79 24 tokens (skip those positions) except the 200 same-model-check games
        f = os.environ.get("E86_CHK"); _CHK[0] = set(int(x) for x in np.load(f)["rid"]) if f else None
    return _CHK[0]
def work_all(chunk):
    if P._enc[0] is None: P._enc[0] = E.KataEncoder(P.KS_, P.CFG_)
    bl, gl, pl, rows, oi, rp = [], [], [], [], [], []
    for uid, color, sgf, rid in chunk:
        pg = parse_sgf(sgf)
        if E._skip_reason(pg): continue
        moves = pg.moves; n_own = int((moves[:, 0] == color).sum()); idx = np.arange(2, min(HI, n_own), dtype=np.int64)
        if len(idx) == 0: continue
        i24 = np.concatenate([lib.own_move_idx_eval(n_own, k=12), P.late_idx(n_own, 12)]); assert np.isin(i24, idx).all(); chk = _chk_set()
        if chk is not None and rid < TEST_OFF and rid not in chk:       # E79 already holds these 24 (same model/PCA); extract only the others
            idx = idx[~np.isin(idx, i24)]
            if len(idx) == 0: continue
        b, g, plies = P._enc[0].encode_game(moves, int(color), idx); assert len(plies) == len(idx)
        loc = np.array([(-1 if moves[p][1] < 0 else int(moves[p][1]) * 19 + int(moves[p][2])) for p in plies], np.int16)
        bl.append(np.packbits(b.reshape(len(b), -1), axis=1)); gl.append(g); pl.append(loc); oi.append(idx.astype(np.int16)); rp.append(np.array([(i24 == j).sum() for j in idx], np.int8))
        rows += [(rid, t) for t in range(len(idx))]
    return (np.concatenate(bl), np.concatenate(gl), np.concatenate(pl), np.array(rows), np.concatenate(oi), np.concatenate(rp)) if bl else None
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("items"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--maxpend", type=int, default=6); ap.add_argument("--blk", type=int, default=1000); a = ap.parse_args()
    OUT = ROOT / ".scratch/e86" / a.stage / "blocks"; OUT.mkdir(parents=True, exist_ok=True); items = pickle.load(open(a.items, "rb")); t0 = time.time(); print("items", len(items), flush=True)
    with Pool(a.workers) as pool:
        for bi, s in enumerate(range(0, len(items), a.blk)):
            nm = f"b{bi}"
            if (OUT / f"{nm}_bits.npy").exists() or (OUT / f"{nm}_pc.npy").exists(): continue            # resumable
            while len(list(OUT.glob("*_bits.npy"))) >= a.maxpend: time.sleep(15)
            its = items[s:s + a.blk]; colmap = {it[3]: it[1] for it in its}; chunks = [its[i:i + 25] for i in range(0, len(its), 25)]
            res = [x for x in pool.map(work_all, chunks, chunksize=1) if x is not None]; rw = np.concatenate([x[3] for x in res])
            for k, j in (("glob", 1), ("loc", 2), ("rows", 3), ("oidx", 4), ("rep", 5)): np.save(OUT / f"{nm}_{k}.npy", np.concatenate([x[j] for x in res]) if k != "rows" else rw)
            np.save(OUT / f"{nm}_color.npy", np.array([colmap[int(x)] for x in rw[:, 0]], np.int8)); np.save(OUT / f"{nm}_bits_tmp.npy", np.concatenate([x[0] for x in res])); os.replace(OUT / f"{nm}_bits_tmp.npy", OUT / f"{nm}_bits.npy")
            print(time.strftime("%H:%M:%S"), nm, "games", len(its), "pos", len(rw), f"{time.time() - t0:.0f}s", flush=True)
    (OUT.parent / "PREP_DONE").write_text("ok"); print("prep done", flush=True)
