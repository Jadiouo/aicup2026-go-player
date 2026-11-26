"""E99 unit test for variant (b) (pe86 + small S + big S = THREE channels), synthetic data, a few seconds, < 1 GB.  Reuses make() of the e90 test (read-only import).
Run: CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q scripts/experiments/e99/test_feats3.py
 (1) three channels give 6 more booster columns than two (18 channel columns in total);
 (2) the last 18 columns are channel0's six, channel1's six, channel2's six, each equal to an independent recomputation (id()-keyed ColorCtx cache must not alias channels);
 (3) the third channel is not an alias of the second and also enters the first-stage shortlist F."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "scripts/experiments/e90"), str(ROOT / "scripts/oneoff")]
import numpy as np
import e77_common as C
import e90_feats2 as C2
from test_feats2 import make
from gogame import player_rerank as prr


def test_three_channels_columns_and_cache():
    kw, A, B = make(2); rng = np.random.default_rng(5); D = rng.standard_normal(A.shape).astype(np.float32); D /= np.linalg.norm(D, axis=1, keepdims=True)
    two = C2.feats(**kw, Ype=[A, B]); three = C2.feats(**kw, Ype=[A, B, D])
    assert three["X"].shape[1] == two["X"].shape[1] + 6
    top = three["top"]; lib, q, color = kw["pool"]["lib"], kw["pool"]["q"], kw["color"]
    cc = C.ColorCtx(lib, q, color, 0.5); cols = []
    for M in (A, B, D):
        Cp = np.stack([M[r].astype(np.float64).mean(0) for r in lib]); Qp = np.stack([M[r].astype(np.float64).mean(0) for r in q]); pcos = prr.unit(Qp) @ prr.unit(Cp).T
        pcosc = cc.cos(M.astype(np.float64).copy())
        for Mx in (pcos, pcosc):
            v = np.take_along_axis(Mx, top, 1); c = v - v.mean(1, keepdims=True); cols += [v, c, c / (v.std(1, keepdims=True) + 1e-9)]
    nq, kk = top.shape; ref = np.stack(cols, 2).reshape(nq * kk, -1).astype(np.float32); got = three["X"][:, -18:]
    assert np.allclose(got, ref, atol=1e-6), np.abs(got - ref).max()
    assert np.abs(got[:, 6:12] - got[:, 12:]).max() > 1e-3 and np.abs(got[:, :6] - got[:, 12:]).max() > 1e-3
    assert not (two["top"] == three["top"]).all()
