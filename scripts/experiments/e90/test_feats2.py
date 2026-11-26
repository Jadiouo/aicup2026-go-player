"""E90 stage 2 unit test (light: synthetic data, a few seconds, < 1 GB).  Run: CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q scripts/experiments/e90/test_feats2.py
 (1) e90_feats2.feats with ONE channel gives bit-identical X / top / F as the original e77_common.feats (so the check run can reproduce E86);
 (2) with TWO channels the last 12 columns are channel0's six features then channel1's six, each equal to an independent recomputation (catches the id()-keyed ColorCtx cache trap);
 (3) the two channels are not aliased (channel-1 columns differ from channel-0 columns) and the shortlist F really contains both terms (top differs from the one-channel top)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(HERE)]
import numpy as np
import e77_common as C
import e90_feats2 as C2
from gogame import player_rerank as prr


def make(seed=0):
    rng = np.random.default_rng(seed); P, per, nq = 60, 6, 80; R = P * per + nq * 2 + 20
    pid = np.concatenate([np.repeat(np.arange(P), per), rng.integers(0, P, R - P * per)])
    Z = rng.standard_normal((R, 90)).astype(np.float32) + 0.3 * pid[:, None] % 7 * 0.1
    def Y(d, s=0.5): return (rng.standard_normal((R, d)) * 0.7 + (np.sin(pid[:, None] * (1 + np.arange(d)[None, :] * 0.01)) * s)).astype(np.float32)
    lib = [np.flatnonzero(pid[: P * per] == p) for p in range(P)]
    q, truth = [], []
    for i in range(nq): p = int(rng.integers(0, P)); n = int(rng.integers(1, 4)); q.append(P * per + rng.integers(0, R - P * per, n)); truth.append(p)
    emb = rng.standard_normal((R, 48)).astype(np.float32)
    def unit(M): return (M / np.linalg.norm(M, axis=1, keepdims=True)).astype(np.float32)
    A, B = unit(Y(32)), unit(Y(32, 0.9))
    rows = np.arange(P * per); Ytrk = Y(100)
    Yth = [Y(100), Y(100)]; Pth = [C.est_plda(y[rows], pid[rows]) for y in Yth]
    kw = dict(Z=Z, sel=np.arange(31, 80), pool=dict(lib=lib, q=q), k=40, Ylda=Y(150), Yhsl=Y(40), use_h=True, emb=emb, wdeep=1.0, Ytrk=Ytrk, wk=1.0,
              opt=dict(color=1, rho=0.5, late=1, wk2=1.0, tplda=1, blk=1, ft=0, wf=1.0, pe=1), color=(rng.random(R) < 0.5).astype(np.int8), Ylate=Y(60), Yth=Yth, Pth=Pth, Yblk=[Y(60) for _ in C.BLK_IDS], Yft=None)
    return kw, A, B


def test_one_channel_is_identical_to_original():
    kw, A, B = make()
    o = C.feats(**kw, Ype=A); n1 = C2.feats(**kw, Ype=[A]); n2 = C2.feats(**kw, Ype=A)
    for n in (n1, n2):
        assert (o["top"] == n["top"]).all() and np.array_equal(o["X"], n["X"]) and np.array_equal(o["F"], n["F"]) and o["k"] == n["k"]


def test_two_channels_columns_and_cache():
    kw, A, B = make(1)
    one = C2.feats(**kw, Ype=[A]); two = C2.feats(**kw, Ype=[A, B])
    assert two["X"].shape[1] == one["X"].shape[1] + 6 and two["X"].shape[0] == one["X"].shape[0]
    top = two["top"]; lib, q, color = kw["pool"]["lib"], kw["pool"]["q"], kw["color"]
    cc = C.ColorCtx(lib, q, color, 0.5); cols = []
    for M in (A, B):
        Cp = np.stack([M[r].astype(np.float64).mean(0) for r in lib]); Qp = np.stack([M[r].astype(np.float64).mean(0) for r in q]); pcos = prr.unit(Qp) @ prr.unit(Cp).T
        pcosc = cc.cos(M.astype(np.float64).copy())               # fresh array each time: cannot hit a stale id() entry
        for Mx in (pcos, pcosc):
            v = np.take_along_axis(Mx, top, 1); c = v - v.mean(1, keepdims=True); cols += [v, c, c / (v.std(1, keepdims=True) + 1e-9)]
    nq, kk = top.shape; ref = np.stack(cols, 2).reshape(nq * kk, -1).astype(np.float32)
    got = two["X"][:, -12:]
    assert np.allclose(got, ref, atol=1e-6), np.abs(got - ref).max()
    assert np.abs(got[:, :6] - got[:, 6:]).max() > 1e-3          # channels not aliased
    assert not (one["top"] == two["top"]).all()                   # second channel also changes the first-stage shortlist F (same wiring as pe)


if __name__ == "__main__":
    test_one_channel_is_identical_to_original(); test_two_channels_columns_and_cache(); print("OK")
