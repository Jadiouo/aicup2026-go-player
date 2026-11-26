"""E86 full stage step 1 (CPU, local): items [(uid, color, sgf, rid)] for every game the E86 pipeline embeds: bundle rows = sim_rows U fin_rows of .scratch/e80/meta.npz (163,500 games, rid = bundle row)
+ the Player test2 games (14,200; rid = 10M + split*100k + position in Z_{s}.npz uids, as e80_prep).  -> .scratch/e86/items_full.pkl (sorted: bundle rows ascending, then test).
Also .scratch/e86/chk_e79.npz: E79 tok_full rows (24 positions, mask) of 200 random bundle games of the union (same-model check on the VM).  H never appears (bundle has none; asserted)."""
import sys, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np, pandas as pd
z = np.load(ROOT / ".scratch/e80/meta.npz"); rows = np.array(sorted(set(z["sim_rows"].tolist()) | set(z["fin_rows"].tolist())), np.int64); print("bundle union games", len(rows))
aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); uid = aux["uid"].astype(str); gcol = aux["game_color"].astype(np.int8)
sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"]); meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet"); u2p = dict(zip(meta.uid, meta.player_id))
assert not any(u2p[uid[r]] in H for r in rows[::50]) and not (set(meta.loc[meta.uid.isin(set(uid[rows])), "player_id"]) & H)
want = {uid[r]: int(r) for r in rows}; got = {}
for f in sorted((ROOT / "data" / "training").glob("train_*.csv")):
    for ch in pd.read_csv(f, usecols=["game_id", "rank", "sgf_content"], chunksize=10_000):
        ch["uid"] = ch["rank"].astype(str) + "_" + ch["game_id"].astype(str); ch = ch[ch["uid"].isin(want)]
        for u, s in zip(ch["uid"], ch["sgf_content"]): got[u] = s
    print(f.name, len(got), flush=True)
assert len(got) == len(want)
items = [(uid[r], int(gcol[r]), got[uid[r]], int(r)) for r in rows]; del got
TEST_OFF = 10_000_000; t2 = ROOT / ".scratch/p1111/t2"; n0 = len(items)
for si, s in enumerate(("pub", "priv")):
    zz = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); zu = zz["uids"].astype(str); pos = {u: i for i, u in enumerate(zu)}; seen = set()
    c = pd.read_csv(t2 / f"player_identification_candidates_{'public' if s == 'pub' else 'private'}.csv", dtype=str)
    for r in c.itertuples(): u = f"cand_{s}:{r.game_id}"; items.append((u, 0 if r.color == "B" else 1, r.sgf_content, TEST_OFF + si * 100_000 + pos[u])); seen.add(u)
    q = pd.read_csv(t2 / f"player_identification_test_{'public' if s == 'pub' else 'private'}.csv", dtype=str)
    for r in q.itertuples(index=False):
        d = dict(zip(q.columns, r))
        for k in range(1, int(d["num_games"]) + 1): u = f"{d['question_id']}#{k}"; items.append((u, 0 if d[f"color_{k}"] == "B" else 1, d[f"sgf_{k}"], TEST_OFF + si * 100_000 + pos[u])); seen.add(u)
    assert seen == set(zu), s
print("test items", len(items) - n0, "total", len(items)); pickle.dump(items, open(ROOT / ".scratch/e86/items_full.pkl", "wb"), protocol=4)
rng = np.random.default_rng(86); pick = np.sort(rng.choice(rows, 200, replace=False)); T = np.load(ROOT / ".scratch/e79/tok_full.npy", mmap_mode="r"); Tm = np.load(ROOT / ".scratch/e79/tok_full_mask.npy", mmap_mode="r")
np.savez(ROOT / ".scratch/e86/chk_e79.npz", rid=pick, X=np.stack([np.asarray(T[r]) for r in pick]), mask=np.stack([np.asarray(Tm[r]) for r in pick])); print("done")
