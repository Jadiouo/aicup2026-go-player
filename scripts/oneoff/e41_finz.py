"""E41: regenerate .scratch/e40/fin/Z.npy (final-space Z of all 712,945 non-H games, fp16; deleted earlier for disk) -- identical code path to e40_finalprep.py (Z block only).  Run before e40_deep_mlp.py --data fin."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np, pandas as pd
import e38_final as EF
F, S3, ROOT = EF.F, EF.S3, EF.ROOT
OUT = ROOT / ".scratch/e40/fin"
P, pm = F.load_pm()
from gogame.aggregate import L0Store
from gogame.gametable import load_table
Hset = set(S3.R.sets()["H"]); meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
rows = np.flatnonzero(~meta.player_id.isin(Hset).to_numpy()); mm = meta.iloc[rows].reset_index(drop=True)
old = pd.read_parquet(OUT / "meta.parquet"); assert (old.uid.to_numpy() == mm.uid.to_numpy()).all(), "row order differs from the stored fin/meta.parquet"
table = load_table(ROOT / "data/cache/l0"); assert list(table.uids) == meta["uid"].tolist(); store = L0Store.from_table(table)
Z = np.lib.format.open_memmap(OUT / "Z.npy", mode="w+", dtype=np.float16, shape=(len(rows), len(pm.s["mu"]) + len(pm.s["cols"])))
for s in range(0, len(rows), 20000): Z[s:s + 20000] = pm._z(store, rows[s:s + 20000]).astype(np.float16)
Z.flush(); print("Z", Z.shape, "max|Z|", float(np.abs(Z[:5000].astype(np.float32)).max()))
