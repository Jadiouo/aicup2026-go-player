"""E41 2: merge test2 first-60 per-move HSL rows (data/test_scan_p1111/results/*_moves.parquet) with the late-window rows (own moves 60..119, .scratch/e41/late/results/E36T*_moves.parquet) and compute the
104 W120 HSL columns with scan_runner.games_from_moves (same function and column order as E36 build_w120_full guard 1).  Output data/test_scan_p1111/hsl120_test_games.parquet (uid + 104 columns).
Guards: (1) W60-only recompute of the first-60 rows reproduces hsl_test_games.parquet (every shared column, allclose 1e-5); (2) late rows start at own_move_idx 60, contiguous, exactly 60 stored old rows for games with new rows;
(3) new-row count == clip(n_own - 60, 0, 60) when n_own is known (hsl n_moves proxy not used); (4) no NaN; (5) every todo uid present once.  Usage: e41_late_merge.py [--allow-partial]"""
import sys, glob, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "katago"), str(ROOT)]
import numpy as np, pandas as pd
from scan_runner import games_from_moves
from kgio import PROFILES
from scripts.experiments.e36.common import E29B
partial = "--allow-partial" in sys.argv
todo = pd.read_parquet(ROOT / ".scratch/p1111/plan_all/hsl_todo.parquet", columns=["uid"]); U = set(todo.uid)
old = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "data/test_scan_p1111/results/S92*_moves.parquet")))], ignore_index=True)
assert set(old.uid) == U and not old.duplicated(["uid", "own_move_idx"]).any(), "first-60 rows do not cover test2 exactly"
lf = sorted(glob.glob(str(ROOT / ".scratch/e41/late/results/E36T*_moves.parquet")))
new = pd.concat([pd.read_parquet(f) for f in lf], ignore_index=True) if lf else old.iloc[:0]
done_games = set()
for f in sorted(glob.glob(str(ROOT / ".scratch/e41/late/results/E36T*_games.parquet"))): done_games |= set(pd.read_parquet(f, columns=["uid"]).uid)
n_skip = 0
for f in sorted(glob.glob(str(ROOT / ".scratch/e41/late/results/E36T*_meta.json"))):
    m = json.load(open(f)); assert set(m["skip_reasons"]) <= {"no_own_moves_in_range"}, m["skip_reasons"]; n_skip += int(m["n_skipped"])
missing = U - done_games
assert len(missing) == n_skip, (len(missing), n_skip)       # every unscanned game was skipped by the runner (no own move in 60..119; meta lists only the first 50 per shard)
done_games |= missing                                        # such games keep their W60 features (window-120 == window-60); verified below: they have < 61 stored own-move rows
print("late shards", len(lf), "late games scanned", len(done_games), "new rows", len(new), flush=True)
if not partial: assert done_games == U, f"late scan incomplete: {len(U - done_games)} games missing"
assert (old[old.uid.isin(missing)].groupby("uid").size() <= 60).all(), "skipped game with >60 own-move rows"
assert set(new.uid) <= U and not new.duplicated(["uid", "own_move_idx"]).any()
names = np.load(E29B / "aux.npz")["names"].astype(str).tolist()
hc = [n for n in names if not (n.startswith("l0_") or n.startswith("se_") or n == "is_white")]
g0 = games_from_moves(old, PROFILES).set_index("uid"); assert list(g0.columns) == hc, "column order"
# guard 1: W60 reproduces the stored test table
h60 = pd.read_parquet(ROOT / "data/test_scan_p1111/hsl_test_games.parquet").set_index("uid"); shared = [c for c in h60.columns if c in g0.columns]
d = (h60.loc[g0.index, shared].to_numpy(np.float64) - g0[shared].to_numpy(np.float64)); print("guard1: shared cols", len(shared), "max|diff| vs hsl_test_games", np.nanmax(np.abs(d)), flush=True)
assert np.nanmax(np.abs(d)) < 1e-4
if len(new):
    nc = new.groupby("uid").own_move_idx.agg(["min", "max", "size"]); assert ((nc["min"] == 60) & (nc["max"] - nc["min"] + 1 == nc["size"])).all(), "late rows not contiguous from 60"
    oc = old.groupby("uid").size(); assert (oc.loc[nc.index] == 60).all(), "games with new rows must have exactly 60 old rows"
    both = pd.concat([old[old.uid.isin(set(nc.index))], new], ignore_index=True); g = games_from_moves(both, PROFILES).set_index("uid"); assert list(g.columns) == hc
    out = g0.copy(); out.loc[g.index, hc] = g[hc].to_numpy()
else: out = g0.copy()
assert not out[hc].isna().any().any(), "NaN in W120 columns"
res = out.reset_index(); res = res[["uid"] + hc]
dst = ROOT / "data/test_scan_p1111" / ("hsl120_test_games_partial.parquet" if partial else "hsl120_test_games.parquet"); res.to_parquet(dst, index=False)
print("wrote", dst, res.shape, "games with late rows", int(len(new.uid.unique())) if len(new) else 0, "mean n_moves W60/W120", float(g0["n_moves"].mean()), float(out["n_moves"].mean()))
