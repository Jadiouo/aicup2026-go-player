"""E90 stage 2: COPY of scripts/oneoff/e77_common.py::feats (lines 124-198, read-only original) with ONE change: `Ype` may be a LIST of position-encoder-style channels
(pe86, then S).  Everything else is character-for-character the original.
 - one channel (a list of length 1, or a bare array) -> the same features in the same order as the original (verified by `stage2_eval.py check`: per-question scores identical to E86's res_a86.json);
 - several channels -> for every channel i in order: wf*z(colour-aware cosine_i) is added to F (as the original does for the single pe channel) and its 6 booster features
   (plain, colour-aware) x (value, centred, z) are appended, in channel order, after the previous channel's six.
 - id()-keyed cache trap: ColorCtx._sums/_qm cache by id(Y).  The original builds `Ype.astype(np.float64)` as a temporary; with several channels a freed temporary's id can be reused by the next
   channel's temporary -> stale cache (silent wrong features).  Here every float64 copy is kept alive in a list until the function returns, so ids are distinct.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
import numpy as np
import e77_common as C
from e77_common import OPT0, BLK_IDS, ColorCtx, tplda_feats, EF, prr, S3, zn


def feats(Z, sel, pool, k, Ylda, Yhsl, use_h, emb, wdeep, Ytrk, wk, opt, color=None, Ylate=None, Yth=None, Pth=None, Yblk=None, Yft=None, Ype=None):
    """Same maths as e56_final.feats3d.  Ytrk: trunk LDA coords (None = no trunk channel); Ylate: late-window trunk coords; Yth/Pth: half-metric trunk coords + PLDA params for tplda."""
    opt = {**OPT0, **(opt or {})}
    lib, q = pool["lib"], pool["q"]
    Zl, Zq = [Z[r] for r in lib], [Z[r] for r in q]
    nq = len(q); f64 = lambda Y, rr: [Y[r].astype(np.float64) for r in rr]  # noqa: E731
    d = {f"l{m}": S3.dist_matrix(f64(Ylda, lib), f64(Ylda, q), m) for m in (10, 30, 80)}
    if use_h:
        for m in (5, 15, 40): d[f"h{m}"] = S3.dist_matrix(f64(Yhsl, lib), f64(Yhsl, q), m)
    if Ytrk is not None:
        for m_ in (10, 30, 60): d[f"t{m_}"] = S3.dist_matrix(f64(Ytrk, lib), f64(Ytrk, q), m_)
    if opt["late"]:
        for m_ in (10, 30, 60): d[f"u{m_}"] = S3.dist_matrix(f64(Ylate, lib), f64(Ylate, q), m_)
    Cd = np.stack([emb[r].mean(0) for r in lib]); Qd = np.stack([emb[r].mean(0) for r in q]); cosd = prr.unit(Qd) @ prr.unit(Cd).T
    cc = None; cosdc = None
    if opt["color"]:
        cc = ColorCtx(lib, q, color, opt["rho"])
        for m in (10, 30, 80): d[f"l{m}c"] = -cc.euc(Ylda, m)
        if Ytrk is not None:
            for m_ in (10, 30, 60): d[f"t{m_}c"] = -cc.euc(Ytrk, m_)
        if opt["late"]:
            for m_ in (10, 30, 60): d[f"u{m_}c"] = -cc.euc(Ylate, m_)
        cosdc = cc.cos(emb)
    if opt["blk"]:
        for j, Yb in zip(BLK_IDS, Yblk):
            for m_ in (10, 30, 60):
                d[f"b{j}_{m_}"] = S3.dist_matrix(f64(Yb, lib), f64(Yb, q), m_)
                if opt["color"]: d[f"b{j}_{m_}c"] = -cc.euc(Yb, m_)
    dft = {}
    if opt["ft"]:
        for m_ in (10, 30, 60):
            dft[f"f{m_}"] = S3.dist_matrix(f64(Yft, lib), f64(Yft, q), m_)
            if opt["color"]: dft[f"f{m_}c"] = -cc.euc(Yft, m_)
    if opt["pe"]:
        Yl_ = list(Ype) if isinstance(Ype, (list, tuple)) else [Ype]
        Y64 = [Y_.astype(np.float64) for Y_ in Yl_]            # kept alive: distinct id() per channel (see module docstring)
        assert len({id(y) for y in Y64}) == len(Y64)
        PC = []                                                # per channel: (pcos, pcosc)
        for Ype_i in Yl_:
            Cp = np.stack([Ype_i[r].astype(np.float64).mean(0) for r in lib]); Qp = np.stack([Ype_i[r].astype(np.float64).mean(0) for r in q]); pcos_i = prr.unit(Qp) @ prr.unit(Cp).T
            PC.append([pcos_i, None])
        if opt["color"]:
            for pc_, y64 in zip(PC, Y64): pc_[1] = cc.cos(y64)
    if opt["color"]:
        Fm = zn(-d["l30c"]) + (zn(-d["h15"]) if use_h else 0.0) + (wk * zn(-d["t30c"]) if Ytrk is not None else 0.0) + wdeep * zn(cosdc)
        if opt["late"]: Fm = Fm + opt["wk2"] * zn(-d["u30c"])
    else:
        Fm = zn(-d["l30"]) + (zn(-d["h15"]) if use_h else 0.0) + (wk * zn(-d["t30"]) if Ytrk is not None else 0.0) + wdeep * zn(cosd)
        if opt["late"]: Fm = Fm + opt["wk2"] * zn(-d["u30"])
    if opt["ft"]: Fm = Fm + opt["wf"] * zn(-dft["f30c"] if opt["color"] else -dft["f30"])
    if opt["pe"]:
        for pcos, pcosc in PC: Fm = Fm + opt["wf"] * zn(pcosc if opt["color"] else pcos)
    kk = min(k, Fm.shape[1]); top = np.argsort(-Fm, axis=1, kind="stable")[:, :kk]
    scal, diff, prod = EF.cand_feats(Zl, Zq, sel, top, Fm, kk)
    ext = []
    for v in (np.take_along_axis(M, top, 1) for M in d.values()):
        c = v - v.mean(1, keepdims=True); ext += [v, c, c / (v.std(1, keepdims=True) + 1e-9)]
    v = np.take_along_axis(cosd, top, 1); c = v - v.mean(1, keepdims=True); ext += [v, c, c / (v.std(1, keepdims=True) + 1e-9)]
    Ld = np.stack([emb[r] for r in lib]); kmd = np.empty((nq, kk)); kxd = np.empty((nq, kk))
    for i in range(nq):
        sm_ = np.einsum("nd,kmd->knm", emb[q[i]], Ld[top[i]], optimize=True); kmd[i] = sm_.mean((1, 2)); kxd[i] = sm_.max(2).mean(1)
    ext += [kmd, kxd]
    blocks = [scal, np.stack(ext, 2), diff, prod]
    if opt["color"]:
        v = np.take_along_axis(cosdc, top, 1); c = v - v.mean(1, keepdims=True)
        blocks.append(np.stack([v, c, c / (v.std(1, keepdims=True) + 1e-9)], 2)); blocks.append(cc.cols(top))
    if opt["tplda"]:
        e2 = tplda_feats(Yth, Pth, lib, q, top); blocks.append(np.stack(e2, 2))
        Qb = np.stack([Ytrk[r, :40].astype(np.float64).mean(0) for r in q]); Cb = np.stack([Ytrk[r, :40].astype(np.float64).mean(0) for r in lib])
        blocks.append(np.abs(Qb[:, None, :] - Cb[top]))
    if opt["ft"]:
        e3 = []
        for v in (np.take_along_axis(M, top, 1) for M in dft.values()):
            c = v - v.mean(1, keepdims=True); e3 += [v, c, c / (v.std(1, keepdims=True) + 1e-9)]
        blocks.append(np.stack(e3, 2))
    if opt["pe"]:
        e4 = []
        for pcos, pcosc in PC:
            for M_ in ((pcos, pcosc) if opt["color"] else (pcos,)):
                v = np.take_along_axis(M_, top, 1); c = v - v.mean(1, keepdims=True); e4 += [v, c, c / (v.std(1, keepdims=True) + 1e-9)]
        blocks.append(np.stack(e4, 2))
    X = np.concatenate(blocks, axis=2).astype(np.float32); assert np.isfinite(X).all()
    return dict(X=X.reshape(nq * kk, -1), top=top, F=Fm, k=kk)
