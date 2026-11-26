"""E40: FINAL-space (data/models_final standardiser) Z for ALL non-H games (712,945; H excluded by id only), fp16, + HSL W60/W120 standardised (mu/sd saved).  -> .scratch/e40/fin/"""
import sys, os
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np, pandas as pd
import e38_final as EF
F, S3, ROOT = EF.F, EF.S3, EF.ROOT
sys.path[:0] = [str(ROOT)]
from scripts.experiments.e36.common import E29B
OUT = ROOT / ".scratch/e40/fin"; OUT.mkdir(parents=True, exist_ok=True)
P, pm = F.load_pm()
from gogame.aggregate import L0Store
from gogame.gametable import load_table
Hset = set(S3.R.sets()["H"]); meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
rows = np.flatnonzero(~meta.player_id.isin(Hset).to_numpy()); mm = meta.iloc[rows].reset_index(drop=True); print("rows", len(rows), flush=True)
table = load_table(ROOT / "data/cache/l0"); assert list(table.uids) == meta["uid"].tolist(); store = L0Store.from_table(table)
if not (OUT / "meta.parquet").exists():
  Z = np.lib.format.open_memmap(OUT / "Z.npy", mode="w+", dtype=np.float16, shape=(len(rows), len(pm.s["mu"]) + len(pm.s["cols"])))
  for s in range(0, len(rows), 20000): Z[s:s + 20000] = pm._z(store, rows[s:s + 20000]).astype(np.float16)
  Z.flush(); del Z; mm.to_parquet(OUT / "meta.parquet")
del store, table
a = np.load(E29B / "aux.npz"); names = a["names"].astype(str); uid = a["uid"].astype(str); pos = {u: i for i, u in enumerate(uid)}
X = np.load(ROOT / ".scratch/e36/bundle/X120_full.npy", mmap_mode="r"); hc = [i for i, n in enumerate(names) if not (n.startswith("l0_") or n.startswith("se_") or n == "is_white")]
H120 = np.asarray(X[:, hc])[np.array([pos[u] for u in mm.uid])].astype(np.float64)
hdf = pd.read_parquet(EF.HSL_PARQUET).set_index("uid"); cols60 = [c for c in hdf.columns if c != "n_moves"]
H60 = hdf.loc[mm.uid.to_numpy(), cols60].to_numpy(np.float64); del hdf
for nm, H in (("hsl60", H60), ("hsl120", H120)):
    mu, sd = H.mean(0), H.std(0) + 1e-9; np.save(OUT / f"{nm}.npy", ((H - mu) / sd).astype(np.float16)); np.savez(OUT / f"{nm}_std.npz", mu=mu, sd=sd); print(nm, H.shape, flush=True)
print("done")
