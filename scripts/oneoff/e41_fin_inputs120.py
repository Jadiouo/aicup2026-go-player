"""E41: test2 input matrices [Z, HSL120-standardised] for the W120 deep encoders (.scratch/e40/fin/test_{pub,priv}_w120.npy, fp16), same layout as e40_fin_inputs.py (W60) but with
data/test_scan_p1111/hsl120_test_games.parquet and fin/hsl120_std.npz (mu/sd of the training rows)."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / ".scratch/e40/fin"
std = np.load(OUT / "hsl120_std.npz"); hp = pd.read_parquet(ROOT / "data/test_scan_p1111/hsl120_test_games.parquet").set_index("uid")
assert hp.shape[1] == len(std["mu"]) == 104
for sp in ("pub", "priv"):
    z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{sp}.npz", allow_pickle=False); uids, Zt = z["uids"].astype(str), z["Z"]
    assert (np.load(OUT / f"test_{sp}_uids.npy") == uids).all()
    Xh = hp.reindex(uids).to_numpy(np.float64); assert np.isfinite(Xh).all()
    X = np.hstack([Zt.astype(np.float32), ((Xh - std["mu"]) / std["sd"]).astype(np.float32)]).astype(np.float16)
    np.save(OUT / f"test_{sp}_w120.npy", X); print(sp, X.shape, "max|std hsl|", float(np.abs((Xh - std["mu"]) / std["sd"]).max()))
