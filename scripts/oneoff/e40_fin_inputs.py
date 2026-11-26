"""E40 production inputs (final Z space): capped booster-pool row index (same rule as e38_final.build_final_space) and test2 input matrices [Z, HSL60-standardised] for the torch embed step.
Requires .scratch/e40/fin/{Z.npy,meta.parquet,hsl60_std.npz} (e40_finalprep.py).  Writes fin/cap_idx.npy, fin/test_{pub,priv}.npy (+ _uids.npy)."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np, pandas as pd
import e38_final as EF
F, ROOT = EF.F, EF.ROOT
OUT = ROOT / ".scratch/e40/fin"
mm = pd.read_parquet(OUT / "meta.parquet"); meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
tot = meta.groupby("player_id").size(); Hset = set(EF.S3.R.sets()["H"]); keep = set(tot.index[tot >= 30]) - Hset
rng = np.random.default_rng(20261111)
m = meta.assign(i=np.arange(len(meta))); m = m[m.player_id.isin(keep)]
rows = []
for (p, c), g in m.groupby(["player_id", "color"]):
    idx = g["i"].to_numpy(); rows.append(idx[rng.permutation(len(idx))[:F.CAP]])
rows = np.sort(np.concatenate(rows)); uid_cap = meta["uid"].to_numpy()[rows]
pos = {u: i for i, u in enumerate(mm.uid)}; cap_idx = np.array([pos[u] for u in uid_cap]); np.save(OUT / "cap_idx.npy", cap_idx); print("cap rows", len(cap_idx))
std = np.load(OUT / "hsl60_std.npz"); hp = pd.read_parquet(EF.HSL_PARQUET)
cols = [c for c in hp.columns if c != "n_moves" and c != "uid"] if "uid" in hp.columns else [c for c in hp.columns if c != "n_moves"]
hp = pd.read_parquet(ROOT / "data/test_scan_p1111/hsl_test_games.parquet").set_index("uid")
cols_tr = list(np.load(ROOT / ".scratch/e38/final/v7e.npz", allow_pickle=False)["hsl_cols"]); print("hsl cols", len(cols_tr))
for sp in ("pub", "priv"):
    z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{sp}.npz", allow_pickle=False); uids, Zt = z["uids"].astype(str), z["Z"]
    Xh = hp.reindex(uids)[cols_tr].to_numpy(np.float64); assert np.isfinite(Xh).all()
    X = np.hstack([Zt.astype(np.float32), ((Xh - std["mu"]) / std["sd"]).astype(np.float32)]).astype(np.float16)
    np.save(OUT / f"test_{sp}.npy", X); np.save(OUT / f"test_{sp}_uids.npy", uids); print(sp, X.shape)
