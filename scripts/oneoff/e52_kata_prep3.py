"""E52 step 5 FINAL-pipeline prep: KataGo input planes for (A) 24 games per eligible player of the final training space (same row selection as e38_final.build_final_space: non-H players with >=30 games,
<=24 games per colour, rng 20261111; eligible = >=28 rows) and (B) all 14,200 real test2 games (cand_*/question games of .scratch/p1111/t2, uid = Z_*.npz uids).  K=6 own-move positions per game.
Output .scratch/e52/kata3/blk*_{bits,glob,loc,rows,color}.npy (+ sel.npz: D-row indices of (A), test uid order of (B)).  H players only read to be EXCLUDED."""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "oneoff")]
import numpy as np, pandas as pd
from multiprocessing import Pool
from scripts.experiments.e13 import encode as E, lib
import e52_kata_prep as P1
P1.K = 6
OUT = ROOT / ".scratch" / "e52" / "kata3"; OUT.mkdir(parents=True, exist_ok=True)
TEST_OFF = 10_000_000
if __name__ == "__main__":
    t0 = time.perf_counter()
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); Hset = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"])
    tot = meta.groupby("player_id").size(); keep = set(tot.index[tot >= 30]) - Hset; rng = np.random.default_rng(20261111)
    m = meta.assign(i=np.arange(len(meta))); m = m[m.player_id.isin(keep)]; rows = []
    for (p, c), g in m.groupby(["player_id", "color"]):
        idx = g["i"].to_numpy(); rows.append(idx[rng.permutation(len(idx))[:24]])
    rows = np.sort(np.concatenate(rows)); pid = meta["player_id"].to_numpy()[rows]; assert not set(pid) & Hset
    elig = sorted(p for p in set(pid) if (pid == p).sum() >= 28); print("D rows", len(rows), "eligible players", len(elig), flush=True)
    rng2 = np.random.default_rng(522); drow = []
    for p in elig:
        idx = np.flatnonzero(pid == p); drow += list(idx[rng2.permutation(len(idx))[:24]])
    drow = np.sort(np.array(drow)); uids = meta["uid"].to_numpy()[rows[drow]]; colors = meta["color"].to_numpy()[rows[drow]].astype(int)
    print("trunk rows", len(drow), flush=True)
    sg = E.read_sgf_for_uids(lib.TRAIN_CSV_DIR, set(uids)); sgm = dict(zip(sg.uid, sg.sgf)); del sg
    items = [(u, int(c), sgm[u], int(j)) for u, c, j in zip(uids, colors, drow)]; del sgm; print("sgf read", len(items), f"{time.perf_counter() - t0:.0f}s", flush=True)
    t2 = ROOT / ".scratch" / "p1111" / "t2"; test_uids = {}
    for si, s in enumerate(("pub", "priv")):
        z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); zu = z["uids"].astype(str); test_uids[s] = zu; pos = {u: i for i, u in enumerate(zu)}; got = set()
        c = pd.read_csv(t2 / f"player_identification_candidates_{'public' if s == 'pub' else 'private'}.csv", dtype=str)
        for r in c.itertuples(): u = f"cand_{s}:{r.game_id}"; items.append((u, 0 if r.color == "B" else 1, r.sgf_content, TEST_OFF + si * 100_000 + pos[u])); got.add(u)
        q = pd.read_csv(t2 / f"player_identification_test_{'public' if s == 'pub' else 'private'}.csv", dtype=str)
        for r in q.itertuples(index=False):
            d = r._asdict() if hasattr(r, "_asdict") else dict(zip(q.columns, r))
            for k in range(1, int(d["num_games"]) + 1):
                u = f"{d['question_id']}#{k}"; items.append((u, 0 if d[f"color_{k}"] == "B" else 1, d[f"sgf_{k}"], TEST_OFF + si * 100_000 + pos[u])); got.add(u)
        assert got == set(zu), (s, len(got), len(zu))
    np.savez(OUT / "sel.npz", drow=drow, uids=uids, test_pub=test_uids["pub"], test_priv=test_uids["priv"]); print("items", len(items), flush=True)
    colmap = {it[3]: it[1] for it in items}; BLK = 10000
    with Pool(6) as pool:
        for bi, s in enumerate(range(0, len(items), BLK)):
            chunks = [items[i:i + 100] for i in range(s, min(s + BLK, len(items)), 100)]
            res = [r for r in pool.map(P1.work, chunks, chunksize=1) if r is not None]
            rw = np.concatenate([r[3] for r in res])
            np.save(OUT / f"blk{bi}_bits.npy", np.concatenate([r[0] for r in res])); np.save(OUT / f"blk{bi}_glob.npy", np.concatenate([r[1] for r in res])); np.save(OUT / f"blk{bi}_loc.npy", np.concatenate([r[2] for r in res]))
            np.save(OUT / f"blk{bi}_rows.npy", rw); np.save(OUT / f"blk{bi}_color.npy", np.array([colmap[int(r)] for r in rw[:, 0]], np.int8))
            print(f"block {bi} done {time.perf_counter() - t0:.0f}s", flush=True)
    (OUT / "PREP_DONE").write_text("ok")
