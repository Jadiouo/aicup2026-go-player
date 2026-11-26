"""E56 prep: KataGo input planes (K own-move positions per game, default 12) for
  --set sim   : (A) the 110,448 sim rows of E52's cap_rows.npy (24 games / sim player, same rows as kata2) + (B) EXTRAS = non-H players that are not in the sim at all (all their games, <30 each)
  --set final : (A) 24 games per eligible player of the final training space (same selection as e52_kata_prep3 / e38_final.build_final_space) + test2's 14,200 games + (B) EXTRAS = non-H players
                outside the 4,565 eligible final players (all their games; capped 40 per player)
Row ids: sim rows = sim meta index; final rows = D-row index; test rows = 10_000_000 + split*100_000 + position in Z_{pub,priv}.npz uids; extras = 30_000_000 + games_meta index.
Output .scratch/e56/<set>/blk*_{bits,glob,loc,rows,color}.npy (blocks of 10k games); throttled so at most --maxpend blocks of bits are waiting for the GPU (disk!).  H players are only read to be EXCLUDED."""
import os, sys, time, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "oneoff")]
import numpy as np, pandas as pd
from multiprocessing import Pool
from scripts.experiments.e13 import encode as E, lib
import e52_kata_prep as P1
EXTRA_OFF, TEST_OFF = 30_000_000, 10_000_000
ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["sim", "final"]); ap.add_argument("--k", type=int, default=12); ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--maxpend", type=int, default=5); ap.add_argument("--max-extra-games", type=int, default=40); a = ap.parse_args()
P1.K = a.k
OUT = ROOT / ".scratch" / "e56" / a.set; OUT.mkdir(parents=True, exist_ok=True)


def extras_items(meta, extra_players, rng, cap):
    m = meta.assign(i=np.arange(len(meta))); m = m[m.player_id.isin(extra_players)]
    sel = []
    for p, g in m.groupby("player_id"):
        idx = g["i"].to_numpy(); sel.append(idx[rng.permutation(len(idx))[:cap]] if len(idx) > cap else idx)
    return np.sort(np.concatenate(sel))


if __name__ == "__main__":
    t0 = time.perf_counter()
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); Hset = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"]); nonH = set(sp.player_id) - Hset
    items = []; sgf_uids = set(); spec = []
    if a.set == "sim":
        m = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet"); keep = np.load(ROOT / ".scratch/e52/kata2/cap_rows.npy"); sel = m.loc[keep]
        assert not set(sel.player_id) & Hset
        for r in sel.itertuples(): spec.append((r.uid, int(r.color), int(r.Index)))
        ex_players = nonH - set(m.player_id)
    else:
        tot = meta.groupby("player_id").size(); keep_p = set(tot.index[tot >= 30]) - Hset; rng = np.random.default_rng(20261111)
        mm = meta.assign(i=np.arange(len(meta))); mm = mm[mm.player_id.isin(keep_p)]; rows = []
        for (p, c), g in mm.groupby(["player_id", "color"]):
            idx = g["i"].to_numpy(); rows.append(idx[rng.permutation(len(idx))[:24]])
        rows = np.sort(np.concatenate(rows)); pid = meta["player_id"].to_numpy()[rows]; assert not set(pid) & Hset
        elig = sorted(p for p in set(pid) if (pid == p).sum() >= 28); print("D rows", len(rows), "eligible players", len(elig), flush=True)
        rng2 = np.random.default_rng(522); drow = []
        for p in elig:
            idx = np.flatnonzero(pid == p); drow += list(idx[rng2.permutation(len(idx))[:24]])
        drow = np.sort(np.array(drow)); uids = meta["uid"].to_numpy()[rows[drow]]; colors = meta["color"].to_numpy()[rows[drow]].astype(int)
        for u, c, j in zip(uids, colors, drow): spec.append((u, int(c), int(j)))
        ex_players = nonH - set(elig)
        np.savez(OUT / "sel.npz", drow=drow, uids=uids, elig=np.array(elig))
    xr = extras_items(meta, ex_players, np.random.default_rng(560), a.max_extra_games); print("extras: players", len(ex_players), "games", len(xr), flush=True)
    _u, _c = meta["uid"].to_numpy(), meta["color"].to_numpy()
    for i in xr: spec.append((_u[i], int(_c[i]), EXTRA_OFF + int(i)))
    np.save(OUT / "extra_rows.npy", xr)
    sg = E.read_sgf_for_uids(lib.TRAIN_CSV_DIR, set(s[0] for s in spec)); sgm = dict(zip(sg.uid, sg.sgf)); del sg
    items = [(u, c, sgm[u], rid) for u, c, rid in spec]; del sgm, spec; print("sgf read", len(items), f"{time.perf_counter() - t0:.0f}s", flush=True)
    if a.set == "final":
        t2 = ROOT / ".scratch" / "p1111" / "t2"; test_uids = {}
        for si, s in enumerate(("pub", "priv")):
            z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); zu = z["uids"].astype(str); test_uids[s] = zu; pos = {u: i for i, u in enumerate(zu)}; got = set()
            c = pd.read_csv(t2 / f"player_identification_candidates_{'public' if s == 'pub' else 'private'}.csv", dtype=str)
            for r in c.itertuples(): u = f"cand_{s}:{r.game_id}"; items.append((u, 0 if r.color == "B" else 1, r.sgf_content, TEST_OFF + si * 100_000 + pos[u])); got.add(u)
            q = pd.read_csv(t2 / f"player_identification_test_{'public' if s == 'pub' else 'private'}.csv", dtype=str)
            for r in q.itertuples(index=False):
                d = dict(zip(q.columns, r))
                for k in range(1, int(d["num_games"]) + 1):
                    u = f"{d['question_id']}#{k}"; items.append((u, 0 if d[f"color_{k}"] == "B" else 1, d[f"sgf_{k}"], TEST_OFF + si * 100_000 + pos[u])); got.add(u)
            assert got == set(zu), (s, len(got), len(zu))
        np.savez(OUT / "test_uids.npz", test_pub=test_uids["pub"], test_priv=test_uids["priv"])
    print("items", len(items), flush=True)
    colmap = {it[3]: it[1] for it in items}; BLK = 10000
    with Pool(a.workers) as pool:
        for bi, s in enumerate(range(0, len(items), BLK)):
            while len(list(OUT.glob("blk*_bits.npy"))) >= a.maxpend: time.sleep(20)
            chunks = [items[i:i + 100] for i in range(s, min(s + BLK, len(items)), 100)]
            res = [r for r in pool.map(P1.work, chunks, chunksize=1) if r is not None]
            rw = np.concatenate([r[3] for r in res])
            np.save(OUT / f"blk{bi}_glob.npy", np.concatenate([r[1] for r in res])); np.save(OUT / f"blk{bi}_loc.npy", np.concatenate([r[2] for r in res]))
            np.save(OUT / f"blk{bi}_rows.npy", rw); np.save(OUT / f"blk{bi}_color.npy", np.array([colmap[int(r)] for r in rw[:, 0]], np.int8))
            np.save(OUT / f"blk{bi}_bits_tmp.npy", np.concatenate([r[0] for r in res])); os.replace(OUT / f"blk{bi}_bits_tmp.npy", OUT / f"blk{bi}_bits.npy")   # bits last (atomic): the GPU side keys on it
            print(f"block {bi} done {time.perf_counter() - t0:.0f}s", flush=True)
    (OUT / "PREP_DONE").write_text("ok")
