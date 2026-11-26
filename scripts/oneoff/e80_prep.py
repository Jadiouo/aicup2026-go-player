"""E80 step 1 (CPU, local): per-position extraction inputs for the Player test2 games (7,100 pub + 7,100 priv candidates/queries, row id = 10M + split*100k + position in Z_{s}.npz uids, same as e77_prep_full) and 200 bundle games (row id = 20M + bundle row; same-model check against E79's tok_full).
Maths = e79_prep_local.work24 (K=12, two windows; imported, not copied).  Output blocks .scratch/e80/blocks/{t0,t1,...}_{bits,glob,loc,rows,color,oidx,win}.npy (bits last), 10,000 games per block.  <= 2 workers.
usage: e80_prep.py [--workers 2]"""
import os, sys, time, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "oneoff")]
import numpy as np, pandas as pd
from multiprocessing import Pool
import e79_prep_local as P
TEST_OFF, CHK_OFF = 10_000_000, 20_000_000
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args(); OUT = ROOT / ".scratch/e80/blocks"; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    t2 = ROOT / ".scratch/p1111/t2"; items = []
    for si, s in enumerate(("pub", "priv")):
        z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); zu = z["uids"].astype(str); pos = {u: i for i, u in enumerate(zu)}; got = set()
        c = pd.read_csv(t2 / f"player_identification_candidates_{'public' if s == 'pub' else 'private'}.csv", dtype=str)
        for r in c.itertuples(): u = f"cand_{s}:{r.game_id}"; items.append((u, 0 if r.color == "B" else 1, r.sgf_content, TEST_OFF + si * 100_000 + pos[u])); got.add(u)
        q = pd.read_csv(t2 / f"player_identification_test_{'public' if s == 'pub' else 'private'}.csv", dtype=str)
        for r in q.itertuples(index=False):
            d = dict(zip(q.columns, r))
            for k in range(1, int(d["num_games"]) + 1):
                u = f"{d['question_id']}#{k}"; items.append((u, 0 if d[f"color_{k}"] == "B" else 1, d[f"sgf_{k}"], TEST_OFF + si * 100_000 + pos[u])); got.add(u)
        assert got == set(zu), (s, len(got), len(zu))
    print("test items", len(items), flush=True)
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); uid = aux["uid"].astype(str); gcol = aux["game_color"].astype(np.int8); row_of = pd.Series(np.arange(len(uid)), index=uid)
    ch = next(pd.read_csv(ROOT / "data/training/train_1D.csv", usecols=["game_id", "rank", "sgf_content"], chunksize=10_000)); ch["uid"] = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str); ch = ch[ch["uid"].isin(row_of.index)].iloc[:200]
    for u, s in zip(ch["uid"], ch["sgf_content"]): r = int(row_of[u]); items.append((u, int(gcol[r]), s, CHK_OFF + r))
    print("total items", len(items), flush=True); BLK = 10000
    with Pool(a.workers) as pool:
        for bi, s in enumerate(range(0, len(items), BLK)):
            nm = f"t{bi}"; its = items[s:s + BLK]; colmap = {it[3]: it[1] for it in its}; chunks = [its[i:i + 100] for i in range(0, len(its), 100)]
            res = [x for x in pool.map(P.work24, chunks, chunksize=1) if x is not None]; rw = np.concatenate([x[3] for x in res])
            np.save(OUT / f"{nm}_glob.npy", np.concatenate([x[1] for x in res])); np.save(OUT / f"{nm}_loc.npy", np.concatenate([x[2] for x in res])); np.save(OUT / f"{nm}_rows.npy", rw)
            np.save(OUT / f"{nm}_oidx.npy", np.concatenate([x[4] for x in res])); np.save(OUT / f"{nm}_win.npy", np.concatenate([x[5] for x in res])); np.save(OUT / f"{nm}_color.npy", np.array([colmap[int(x)] for x in rw[:, 0]], np.int8))
            np.save(OUT / f"{nm}_bits_tmp.npy", np.concatenate([x[0] for x in res])); os.replace(OUT / f"{nm}_bits_tmp.npy", OUT / f"{nm}_bits.npy")
            print(time.strftime("%H:%M:%S"), nm, "games", len(its), "pos", len(rw), f"{time.time() - t0:.0f}s", flush=True)
    (ROOT / ".scratch/e80/PREP_DONE").write_text("ok"); print("prep done", flush=True)
