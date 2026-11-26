"""E80 CPU (.venv): game/player bookkeeping for the full-data encoders -> .scratch/e80/meta.npz.  PLAN section 1.
Rows = E79 bundle rows (aux.npz uid order = all 712,945 non-H games).
  sim   h0/h1/full : training games = games of all non-H players that are NOT sim EVAL players (1,481) and have >= 12 games.  sim TRAIN players (grp=='train' of the full sim meta, 3,199) keep the E77 halves (e77_meta.py: rng(99) permutation of the train codes);
                     the other non-H players (not in the sim meta) are split by rng(80) permutation of sorted player_id (first half -> half 0).  full = h0 + h1.
  final h0/h1/full : TA/TB (final_groups.json, = E77) + X group split by rng(80); full = all non-H players with >= 12 games.
  embed sets: sim_rows (bundle rows of cap_rows, gid = sim meta index), fin_rows (bundle rows of the final D rows, gid = D row).
Asserts: no H player, no sim EVAL player in any sim training set, sim train/eval disjoint."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; K = ROOT / ".scratch" / "e80"
aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); uid = aux["uid"].astype(str); row_of = pd.Series(np.arange(len(uid)), index=uid)
meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet"); sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"])
u2p = dict(zip(meta.uid, meta.player_id)); pid = np.array([u2p[u] for u in uid], dtype=object); assert not (set(pid) & H)
cnt = pd.Series(pid).value_counts(); elig_pl = set(cnt.index[cnt >= 12]); in_elig = np.isin(pid, list(elig_pl)); log = {"players_nonH": int(cnt.size), "games_nonH": int(len(uid)), "players_ge12": len(elig_pl), "games_ge12": int(in_elig.sum())}
sm = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet"); keep = np.load(ROOT / ".scratch/e52/kata2/cap_rows.npy")
codes, uni = pd.factorize(sm.player_id); trm = (sm.grp == "train").to_numpy(); tc = np.unique(codes[trm]); perm = np.random.default_rng(99).permutation(len(tc))
h0 = set(uni[tc[np.sort(perm[: len(perm) // 2])]]); h1 = set(uni[tc[np.sort(perm[len(perm) // 2:])]]); ev = set(sm.loc[sm.grp == "eval", "player_id"]); trp = h0 | h1
assert not (ev & trp) and not ((ev | trp) & H) and len(h0 & h1) == 0
others = sorted(set(pid) - ev - trp); po = np.random.default_rng(80).permutation(len(others)); o0 = {others[i] for i in np.sort(po[: len(po) // 2])}; o1 = set(others) - o0
S0, S1 = h0 | o0, h1 | o1; assert not ((S0 | S1) & ev) and not (S0 & S1)
def mask(S): return np.isin(pid, list(S)) & in_elig
m = {"sim_h0": mask(S0), "sim_h1": mask(S1)}; m["sim_full"] = m["sim_h0"] | m["sim_h1"]
for k in ("sim_h0", "sim_h1", "sim_full"): log[k] = {"players": int(len(set(pid[m[k]]))), "games": int(m[k].sum())}
assert not (set(pid[m["sim_full"]]) & ev), "sim eval player in sim encoder training"
log["sim_eval_players"] = len(ev); log["sim_eval_games_excluded"] = int(np.isin(pid, list(ev)).sum()); log["sim_train_players_E77rule"] = len(trp); log["sim_other_players_ge12_h0h1"] = [len(o0 & elig_pl), len(o1 & elig_pl)]
G = {k: set(v) for k, v in json.load(open(ROOT / ".scratch/e39/final_groups.json")).items()}; TA, TB, X = G["TA"], G["TB"], sorted(G["X"]); assert not ((TA | TB | set(X)) & H) and len(TA | TB | set(X)) == len(set(pid)) == 7367, (len(TA | TB | set(X)), len(set(pid)))
px = np.random.default_rng(80).permutation(len(X)); x0 = {X[i] for i in np.sort(px[: len(px) // 2])}; x1 = set(X) - x0
F0, F1 = TA | x0, TB | x1
m["fin_h0"] = mask(F0); m["fin_h1"] = mask(F1); m["fin_full"] = m["fin_h0"] | m["fin_h1"]
for k in ("fin_h0", "fin_h1", "fin_full"): log[k] = {"players": int(len(set(pid[m[k]]))), "games": int(m[k].sum())}
# embed sets
sel = sm.iloc[keep]; sim_rows = row_of.reindex(sel.uid.to_numpy()).to_numpy(); assert not np.isnan(sim_rows).any(); sim_rows = sim_rows.astype(np.int64)
s = np.load(ROOT / ".scratch/e77/final/sel.npz", allow_pickle=True); fin_rows = row_of.reindex(s["uids"].astype(str)).to_numpy(); assert not np.isnan(fin_rows).any(); fin_rows = fin_rows.astype(np.int64); fin_gid = s["drow"].astype(np.int64)
log["sim_embed_games"] = int(len(sim_rows)); log["final_embed_games"] = int(len(fin_rows)); log["sim_eval_embed_games"] = int((sel.grp == "eval").sum())
# colour agreement between games_meta and bundle (already checked in E79, re-assert for the embed rows)
gcol = aux["game_color"].astype(np.int8); mc = dict(zip(meta.uid, meta.color)); assert all(int(mc[u]) == int(gcol[r]) for u, r in zip(sel.uid.to_numpy()[:5000], sim_rows[:5000]))
tot = meta.groupby("player_id").size(); kp = set(tot.index[tot >= 30]) - H; mm = meta[meta.player_id.isin(kp)]; NROWS = int(sum(min(n, 24) for n in mm.groupby(["player_id", "color"]).size()))
np.savez(K / "meta.npz", player=pd.factorize(pid)[0].astype(np.int32), sim_rows=sim_rows, sim_gid=keep.astype(np.int64), sim_eval=(sel.grp == "eval").to_numpy(), fin_rows=fin_rows, fin_gid=fin_gid,
         fin_nrows=np.int64(NROWS), **{k: v for k, v in m.items()})
(K / "meta_log.json").write_text(json.dumps(log, indent=1)); print(json.dumps(log, indent=1))
