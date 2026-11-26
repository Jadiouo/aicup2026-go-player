"""E86 full stage (CPU, local): game/player bookkeeping for the E77-size encoders over the E86 union token array -> .scratch/e86/meta86.npz, .scratch/e86/e79_union.npz.
union = sorted bundle rows of sim_rows U fin_rows (163,500 games; = the first 163,500 items of items_full.pkl).  Encoder training sets (PLAN section 1, E77 sizes = the embed games themselves):
  sim h0/h1/full : games of sim_rows that are NOT sim-eval games, whose player is a sim TRAIN player (E77 halves, rng(99) permutation of sim-train codes; = e77_meta/e80_meta h0/h1), players with >= 12 such games.
  final h0/h1/full : games of fin_rows whose player is in TA (h0) / TB (h1) of .scratch/e39/final_groups.json (E77 groups); full = both.
Embed sets: sim_rows (gid = sim meta index; pe_sim arrays have 223,693 rows), fin_rows (gid = D row; pe_final arrays have fin_nrows rows).  Asserts: no H player, no sim EVAL player in any sim training set, sim eval/train disjoint.
Also dumps the E79 tok_full tokens (24 positions, mask, bucket, colour) of the union games for the VM (E86 reuses them for the positions Colab did not re-extract)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; K = ROOT / ".scratch/e86"
z = np.load(ROOT / ".scratch/e80/meta.npz"); sim_rows, fin_rows = z["sim_rows"], z["fin_rows"]; union = np.array(sorted(set(sim_rows.tolist()) | set(fin_rows.tolist())), np.int64); pos = pd.Series(np.arange(len(union)), index=union)
aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=False); uid = aux["uid"].astype(str)
meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet"); sp = pd.read_parquet(ROOT / "data/splits/split.parquet"); H = set(sp.loc[sp["is_holdout"].astype(bool), "player_id"])
u2p = dict(zip(meta.uid, meta.player_id)); pid_u = np.array([u2p[u] for u in uid[union]], dtype=object); assert not (set(pid_u) & H)
sm = pd.read_parquet(ROOT / ".scratch/p1111/sim/meta.parquet"); keep = np.load(ROOT / ".scratch/e52/kata2/cap_rows.npy"); sel = sm.iloc[keep]; assert (sel.uid.to_numpy() == uid[sim_rows]).all()
sim_eval = (sel.grp == "eval").to_numpy(); ev = set(sm.loc[sm.grp == "eval", "player_id"])
codes, uni = pd.factorize(sm.player_id); trm = (sm.grp == "train").to_numpy(); tc = np.unique(codes[trm]); perm = np.random.default_rng(99).permutation(len(tc))
h0 = set(uni[tc[np.sort(perm[: len(perm) // 2])]]); h1 = set(uni[tc[np.sort(perm[len(perm) // 2:])]]); assert not (ev & (h0 | h1)) and not (h0 & h1) and not ((h0 | h1) & H)
G = {k: set(v) for k, v in json.load(open(ROOT / ".scratch/e39/final_groups.json")).items()}; TA, TB, X = G["TA"], G["TB"], G["X"]
pid_sim = np.array([u2p[u] for u in uid[sim_rows]], dtype=object); pid_fin = np.array([u2p[u] for u in uid[fin_rows]], dtype=object); log = {"union_games": len(union), "fin_players_in_X": len(set(pid_fin) & X)}
def elig(rows_idx_in_union, pids):         # keep players with >= 12 games in the candidate list
    vc = pd.Series(pids).value_counts(); ok = set(vc.index[vc >= 12]); return np.array([p in ok for p in pids])
m = {}
base = ~sim_eval
for name, S in (("sim_h0", h0), ("sim_h1", h1), ("sim_full", h0 | h1)):
    cand = base & np.array([p in S for p in pid_sim]); cand &= np.where(cand, elig(None, np.where(cand, pid_sim, "_")), False); mk = np.zeros(len(union), bool); mk[pos[sim_rows[cand]].to_numpy()] = True; m[name] = mk
    assert not (set(pid_u[mk]) & ev); log[name] = {"players": len(set(pid_u[mk])), "games": int(mk.sum())}
for name, S in (("fin_h0", TA), ("fin_h1", TB), ("fin_full", TA | TB)):
    cand = np.array([p in S for p in pid_fin]); mk = np.zeros(len(union), bool); mk[pos[fin_rows[cand]].to_numpy()] = True; m[name] = mk; log[name] = {"players": len(set(pid_u[mk])), "games": int(mk.sum())}
pcode = pd.factorize(pid_u)[0].astype(np.int32)
np.savez(K / "meta86.npz", union=union, player=pcode, sim_pos=pos[sim_rows].to_numpy(), sim_gid=z["sim_gid"], sim_eval=sim_eval, fin_pos=pos[fin_rows].to_numpy(), fin_gid=z["fin_gid"], fin_nrows=z["fin_nrows"], **m)
T = np.load(ROOT / ".scratch/e79/tok_full.npy", mmap_mode="r"); Tm = np.load(ROOT / ".scratch/e79/tok_full_mask.npy", mmap_mode="r"); To = np.load(ROOT / ".scratch/e79/tok_full_oidx.npy", mmap_mode="r"); Tg = np.load(ROOT / ".scratch/e79/tok_full_gcol.npy")
X79 = np.empty((len(union), 24, 64), np.float16)
for s in range(0, len(union), 20000): X79[s:s + 20000] = T[union[s:s + 20000]]
np.savez(K / "e79_union.npz", X=X79, mask=np.asarray(Tm[union]), oidx=np.asarray(To[union]), gcol=Tg[union]); (K / "meta86_log.json").write_text(json.dumps(log, indent=1)); print(json.dumps(log, indent=1))
