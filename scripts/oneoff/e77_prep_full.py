"""E77 prep, full scale (CPU).  Copy of e69_prep.py (same sim / final / test2 games, same row ids, same 10k-game blocks) with TWO windows per game: early = own_move_idx_eval(n_own, k=12) (own moves 2-59, = E69/E56 positions) then late = own moves 60-119 (12 evenly spaced; < 61 own moves -> last 58 own moves, = E62) -> <= 24 positions per game, and the own-move index (oidx) + window flag (win: 0 early, 1 late) of every position saved.
  --set pca   : 1,000 random sim TRAIN players (seed 77) x 4 games (rows of cap_rows) -> .scratch/e77/pca/  (PCA fitting set; PLAN section 1)
  --set sim   : the 110,448 sim rows of cap_rows.npy -> .scratch/e77/sim/
  --set final : 24 games per eligible final player (as e69_prep) + test2's 14,200 games -> .scratch/e77/final/
Output blk{i}_{bits,glob,loc,rows,color,oidx,win}.npy; throttled by --maxpend blocks of bits waiting for the GPU.  H players are only read to be EXCLUDED."""
import os, sys, time, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "oneoff")]
import numpy as np, pandas as pd
from multiprocessing import Pool
from scripts.experiments.e13 import encode as E, lib
import e52_kata_prep as P1
from gogame.sgf import parse_sgf
EXTRA_OFF, TEST_OFF = 30_000_000, 10_000_000
ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["pca", "sim", "final"]); ap.add_argument("--k", type=int, default=12); ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--maxpend", type=int, default=6); ap.add_argument("--start-block", type=int, default=0); ap.add_argument("--max-extra-games", type=int, default=40); ap.add_argument("--dump-items", default=""); a = ap.parse_args()
P1.K = a.k
KS_, CFG_ = P1.KS, P1.CFG
_enc = [None]
def late_idx(n_own, k):
    idx = lib.own_move_idx_eval(n_own, k=k, lo=60, hi=120)
    if len(idx) == 0: idx = lib.own_move_idx_eval(n_own, k=k, lo=max(2, n_own - 58), hi=n_own)
    return idx
def work24(chunk):
    if _enc[0] is None: _enc[0] = E.KataEncoder(KS_, CFG_)
    bl, gl, pl, rows, oi, wi = [], [], [], [], [], []
    for uid, color, sgf, rid in chunk:
        pg = parse_sgf(sgf)
        if E._skip_reason(pg): continue
        moves = pg.moves; n_own = int((moves[:, 0] == color).sum()); j = 0; got = False
        for w, idx in ((0, lib.own_move_idx_eval(n_own, k=P1.K)), (1, late_idx(n_own, P1.K))):
            if len(idx) == 0: continue
            b, g, plies = _enc[0].encode_game(moves, int(color), idx)
            loc = np.array([(-1 if moves[p][1] < 0 else int(moves[p][1]) * 19 + int(moves[p][2])) for p in plies], np.int16)
            bl.append(np.packbits(b.reshape(len(b), -1), axis=1)); gl.append(g); pl.append(loc); oi.append(np.asarray(idx, np.int16)); wi.append(np.full(len(idx), w, np.int8))
            rows += [(rid, j + t) for t in range(len(plies))]; j += len(plies); got = True
    return (np.concatenate(bl), np.concatenate(gl), np.concatenate(pl), np.array(rows), np.concatenate(oi), np.concatenate(wi)) if bl else None
OUT = ROOT / ".scratch" / "e77" / a.set; OUT.mkdir(parents=True, exist_ok=True)


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
    if a.set in ("sim", "pca"):
        m = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet"); keep = np.load(ROOT / ".scratch/e52/kata2/cap_rows.npy"); sel = m.loc[keep]
        assert not set(sel.player_id) & Hset
        if a.set == "pca":
            rng = np.random.default_rng(77); trp = np.array(sorted(set(sel.loc[sel.grp == "train", "player_id"]))); pk = set(rng.choice(trp, 1000, replace=False)); sel = sel[sel.player_id.isin(pk)]
            sel = sel.groupby("player_id", group_keys=False).apply(lambda g: g.iloc[np.random.default_rng(77).permutation(len(g))[:4]]); assert (sel.grp == "train").all() and len(sel) == 4000, len(sel)
        for r in sel.itertuples(): spec.append((r.uid, int(r.color), int(r.Index)))
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
        np.savez(OUT / "sel.npz", drow=drow, uids=uids, elig=np.array(elig))
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
    if a.dump_items:
        import pickle; pickle.dump(items, open(a.dump_items, "wb")); print("dumped", len(items), "items ->", a.dump_items); sys.exit(0)
    colmap = {it[3]: it[1] for it in items}; BLK = 10000
    with Pool(a.workers) as pool:
        for bi, s in enumerate(range(0, len(items), BLK)):
            if bi < a.start_block: continue
            while len(list(OUT.glob("blk*_bits.npy"))) >= a.maxpend: time.sleep(20)
            chunks = [items[i:i + 100] for i in range(s, min(s + BLK, len(items)), 100)]
            res = [r for r in pool.map(work24, chunks, chunksize=1) if r is not None]
            rw = np.concatenate([r[3] for r in res])
            np.save(OUT / f"blk{bi}_glob.npy", np.concatenate([r[1] for r in res])); np.save(OUT / f"blk{bi}_loc.npy", np.concatenate([r[2] for r in res]))
            np.save(OUT / f"blk{bi}_rows.npy", rw); np.save(OUT / f"blk{bi}_oidx.npy", np.concatenate([r[4] for r in res])); np.save(OUT / f"blk{bi}_win.npy", np.concatenate([r[5] for r in res])); np.save(OUT / f"blk{bi}_color.npy", np.array([colmap[int(r)] for r in rw[:, 0]], np.int8))
            np.save(OUT / f"blk{bi}_bits_tmp.npy", np.concatenate([r[0] for r in res])); os.replace(OUT / f"blk{bi}_bits_tmp.npy", OUT / f"blk{bi}_bits.npy")   # bits last (atomic): the GPU side keys on it
            print(f"block {bi} done {time.perf_counter() - t0:.0f}s", flush=True)
    (OUT / "PREP_DONE").write_text("ok")
