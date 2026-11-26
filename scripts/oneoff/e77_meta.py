"""E77: gid-aligned meta for the full-scale encoder training.  usage: e77_meta.py sim|final -> .scratch/e77/<set>_meta.npz (gid, half (0/1 training halves, -1 = not a training player), player code (-1 none), [sim: eval mask]).
sim   : halves = the E52/E58 rule used by e77_rerank (factorise player ids of sim meta, TRAIN players only, rng(99) permutation, first half = 0); sim EVAL players get -1.
final : TA / TB from .scratch/e39/final_groups.json (= e39_final.groups_final, the halves of the 4,565 eligible non-H players); test2 games (gid >= 10M) get -1; H never appears."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; K = ROOT / ".scratch" / "e77"; name = sys.argv[1]; z = np.load(K / f"{name}_tok.npz"); gid = z["gid"]
if name == "sim":
    m = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet"); codes, uni = pd.factorize(m.player_id); tc = np.unique(codes[(m.grp == "train").to_numpy()]); perm = np.random.default_rng(99).permutation(len(tc))
    h0 = set(uni[tc[np.sort(perm[: len(perm) // 2])]]); h1 = set(uni[tc[np.sort(perm[len(perm) // 2:])]]); pid = m.player_id.to_numpy()[gid]; grp = m.grp.to_numpy()[gid]
    half = np.where((grp == "train") & np.isin(pid, list(h0)), 0, np.where((grp == "train") & np.isin(pid, list(h1)), 1, -1)).astype(np.int8)
    pl = pd.factorize(pid)[0].astype(np.int32); np.savez(K / "sim_meta.npz", gid=gid, half=half, player=pl, eval=(grp == "eval"))
    print("sim games", len(gid), "half0/half1/-1", [(half == v).sum() for v in (0, 1, -1)], "train players", len(set(pid[half >= 0])), "eval players", len(set(pid[grp == "eval"])))
else:
    G = {k: set(v) for k, v in json.load(open(ROOT / ".scratch/e39/final_groups.json")).items()}; TA, TB = G["TA"], G["TB"]; meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet", columns=["uid", "player_id"]); u2p = dict(zip(meta.uid, meta.player_id))
    s = np.load(K / "final" / "sel.npz", allow_pickle=True); d2u = dict(zip(s["drow"].tolist(), s["uids"].tolist()))
    pid = np.array([u2p[d2u[int(g)]] if g < 10_000_000 else "" for g in gid], dtype=object); tr = gid < 10_000_000
    half = np.where(tr & np.isin(pid, list(TA)), 0, np.where(tr & np.isin(pid, list(TB)), 1, -1)).astype(np.int8)
    pl = np.full(len(gid), -1, np.int32); pl[tr] = pd.factorize(pid[tr])[0]; np.savez(K / "final_meta.npz", gid=gid, half=half, player=pl)
    print("final games", len(gid), "train rows", int(tr.sum()), "half0/half1/-1", [(half == v).sum() for v in (0, 1, -1)], "players TA/TB", len(TA), len(TB), "players seen", len(set(pid[half >= 0])), "(must be", len(TA | TB), ")")
