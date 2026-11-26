"""E50: build the ensemble embeddings emb_{OUT}_*  =  concat(W60 encoder emb_{A}_*, W120 encoder emb_fin120_* (E41, non-H, unchanged))/sqrt(2), fp16 -- same formula as the (unsaved) v7g `finens`
(re-derived and verified bit-identical on the stored v7g files).  Usage: e50_ens.py --w60-dir DIR --w60-tag TAG --out TAG2 [--verify]"""
import argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--w60-dir", required=True); ap.add_argument("--w60-tag", required=True); ap.add_argument("--out", required=True); ap.add_argument("--out-dir", default=str(ROOT / ".scratch/e50/emb")); ap.add_argument("--verify", action="store_true")
a = ap.parse_args(); D120 = ROOT / ".scratch/e41"
ens = lambda A, B: (np.concatenate([A.astype(np.float32), B.astype(np.float32)], 1) / np.sqrt(2)).astype(np.float16)   # noqa: E731
if a.verify:   # the stored v7g files must be reproduced from the stored parts
    for nm in ("h0_cap", "h1_cap", "full_test_pub", "full_test_priv"):
        A = np.load(ROOT / f".scratch/e40/emb_fin_{nm}.npy"); B = np.load(D120 / f"emb_fin120_{nm}.npy"); assert np.array_equal(ens(A, B), np.load(D120 / f"emb_finens_{nm}.npy")), nm
    print("formula verified on v7g files")
for nm in ("h0_cap", "h1_cap", "full_test_pub", "full_test_priv"):
    A = np.load(Path(a.w60_dir) / f"emb_{a.w60_tag}_{nm}.npy"); B = np.load(D120 / f"emb_fin120_{nm}.npy"); assert A.shape[0] == B.shape[0], (nm, A.shape, B.shape)
    np.save(Path(a.out_dir) / f"emb_{a.out}_{nm}.npy", ens(A, B)); print("wrote", a.out, nm, A.shape, B.shape)
