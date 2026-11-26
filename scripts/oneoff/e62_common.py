"""E62 shared feature code (used by e62_rerank.py = simulation and e62_final.py / e62_t2rr.py = production).
With every option off, `feats` reproduces e56_final.feats3d (and e56_rerank.feats3) bit for bit (new columns are appended at the end / new d entries after the old ones).
Options (dict `opt`): color (0/1), rho, late (0/1), wk2, tplda (0/1).
  color : colour-aware profiles (E61 diag2.py colsim) for L0 LDA10/30/80, trunk LDA10/30/60, deep cosine (+ late trunk LDA10/30/60 when late); F uses the colour-aware versions; 3 colour columns.
  late  : second trunk LDA channel (own moves 60-119): 3 distances (+ colour versions) and a term wk2 * z(-u30) in F.
  tplda : trunk diagonal-PLDA LLR (60 and 100 dims; mean over the half metrics) + |bag-mean difference| of the first 40 trunk LDA dims.
No H data anywhere."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
import numpy as np
import e38_final as EF
from gogame import player_rerank as prr
S3, zn = EF.S3, EF.zn
OPT0 = dict(color=0, rho=0.5, late=0, wk2=1.0, tplda=0)


def d2m(a, b): return np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T, 0)


class ColorCtx:
    """Colour-aware similarities for one pool.  lib/q: lists of row-index arrays, color: array (0/1) indexed by row.  Returned matrices are 'higher = more similar'."""

    def __init__(self, lib, q, color, rho):
        self.lib, self.q, self.col, self.rho = lib, q, color, rho
        P = len(lib); n_ = np.zeros((2, P))
        for j, r in enumerate(lib):
            c = color[r]; n_[0, j] = (c == 0).sum(); n_[1, j] = (c == 1).sum()
        self.n_ = n_; self.P = P
        self.qcol = [np.array([(color[r] == c).sum() for r in q], float) for c in (0, 1)]       # per query, number of games of colour c
        self.qn = np.array([len(r) for r in q], float)
        self.den, self.neff = [], []
        for c in (0, 1):
            den = n_[c] + rho * n_[1 - c]; neff = den ** 2 / np.maximum(n_[c] + rho ** 2 * n_[1 - c], 1e-9)
            self.den.append(np.maximum(den, 1e-9)); self.neff.append(np.maximum(neff, 1e-9))
        self.qi = [np.flatnonzero(self.qcol[c] > 0) for c in (0, 1)]
        self._S = {}; self._Q = {}

    def _sums(self, Y):
        k = id(Y)
        if k not in self._S:
            S = np.zeros((2, self.P, Y.shape[1]))
            for j, r in enumerate(self.lib):
                for c in (0, 1):
                    m = self.col[r] == c
                    if m.any(): S[c, j] = Y[r[m]].astype(np.float64).sum(0)
            self._S[k] = S
        return self._S[k]

    def _qm(self, Y, c):
        k = (id(Y), c)
        if k not in self._Q:
            self._Q[k] = np.stack([Y[self.q[i][self.col[self.q[i]] == c]].astype(np.float64).mean(0) for i in self.qi[c]]) if len(self.qi[c]) else np.zeros((0, Y.shape[1]))
        return self._Q[k]

    def euc(self, Y, m):
        S = self._sums(Y); out = np.zeros((len(self.q), self.P))
        for c in (0, 1):
            if not len(self.qi[c]): continue
            prof = (S[c][:, :m] + self.rho * S[1 - c][:, :m]) / self.den[c][:, None]
            qm = self._qm(Y, c)[:, :m]; nq = self.qcol[c][self.qi[c]]
            out[self.qi[c]] += -d2m(qm, prof) / (1 / nq[:, None] + 1 / self.neff[c][None, :])
        return out

    def cos(self, Y):
        S = self._sums(Y); out = np.zeros((len(self.q), self.P))
        for c in (0, 1):
            if not len(self.qi[c]): continue
            prof = (S[c] + self.rho * S[1 - c]) / self.den[c][:, None]
            qm = self._qm(Y, c); a = qm / (np.linalg.norm(qm, axis=1, keepdims=True) + 1e-12); b = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12)
            out[self.qi[c]] += (a @ b.T) * (self.qcol[c][self.qi[c]] / self.qn[self.qi[c]])[:, None]
        return out

    def cols(self, top):
        """(nq,k,3): query black fraction; expected same-colour library games; library black fraction minus query black fraction."""
        qb = self.qcol[0] / self.qn; qw = self.qcol[1] / self.qn
        same = qb[:, None] * self.n_[0][None, :] + qw[:, None] * self.n_[1][None, :]
        libb = self.n_[0] / np.maximum(self.n_[0] + self.n_[1], 1)
        k = top.shape[1]
        return np.stack([np.broadcast_to(qb[:, None], top.shape), np.take_along_axis(same, top, 1), np.take_along_axis(np.broadcast_to(libb[None, :], same.shape), top, 1) - qb[:, None]], 2)


# ------------------------------------------------------------------ diagonal PLDA (E61 diag2.py), per-candidate gallery size
def est_plda(Y, pid):
    """within-game variance W (d,), between-player variance Psi (d,), grand mean gm (d,) of the rows Y (n,d) with player labels pid (E61 `est`)."""
    c_, inv = np.unique(pid, return_inverse=True); cnt = np.bincount(inv).astype(float); Yd = Y.astype(np.float64)
    mu = np.zeros((len(c_), Y.shape[1]))
    for a in range(0, len(Yd), 20000): np.add.at(mu, inv[a:a + 20000], Yd[a:a + 20000])
    mu /= cnt[:, None]; W = np.zeros(Y.shape[1])
    for a in range(0, len(Yd), 20000): W += ((Yd[a:a + 20000] - mu[inv[a:a + 20000]]) ** 2).sum(0)
    W /= (len(Yd) - len(c_)); psi = np.maximum(mu.var(0, ddof=1) - W * np.mean(1 / cnt), 1e-4)
    return dict(gm=mu.mean(0), W=W, Psi=psi)


def plda_llr(Y, P, lib, q, m, chunk=25):
    g = np.stack([Y[r, :m].astype(np.float64).mean(0) for r in lib]) - P["gm"][:m]; qq = np.stack([Y[r, :m].astype(np.float64).mean(0) for r in q]) - P["gm"][:m]
    nq = np.array([len(r) for r in q], float)[:, None, None]; nG = np.array([len(r) for r in lib], float)[None, :, None]; w, ps = P["W"][:m], P["Psi"][:m]
    out = np.empty((len(q), len(lib)))
    for s in range(0, len(q), chunk):
        Q = qq[s:s + chunk, None, :]; Gm = g[None]; a = ps + w / nq[s:s + chunk]; b = ps + w / nG; c = ps; det = a * b - c * c
        same = -0.5 * ((b * Q * Q - 2 * c * Q * Gm + a * Gm * Gm) / det + np.log(det)); diff = -0.5 * (Q * Q / a + Gm * Gm / b + np.log(a * b))
        out[s:s + chunk] = (same - diff).sum(2)
    return out


def tplda_feats(Yh, Ph, lib, q, top):
    """Yh/Ph: lists (one per half metric) of trunk LDA coords (>=100 dims) and PLDA params; mean LLR over them.  -> list of (nq,k) blocks (value, centred, z) for 60 and 100 dims."""
    ext = []
    for m in (60, 100):
        L = np.mean([plda_llr(Y, P, lib, q, m) for Y, P in zip(Yh, Ph)], axis=0); v = np.take_along_axis(L, top, 1); c = v - v.mean(1, keepdims=True)
        ext += [v, c, c / (v.std(1, keepdims=True) + 1e-9)]
    return ext


# ------------------------------------------------------------------ the feature function
def feats(Z, sel, pool, k, Ylda, Yhsl, use_h, emb, wdeep, Ytrk, wk, opt, color=None, Ylate=None, Yth=None, Pth=None):
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
    if opt["color"]:
        Fm = zn(-d["l30c"]) + (zn(-d["h15"]) if use_h else 0.0) + (wk * zn(-d["t30c"]) if Ytrk is not None else 0.0) + wdeep * zn(cosdc)
        if opt["late"]: Fm = Fm + opt["wk2"] * zn(-d["u30c"])
    else:
        Fm = zn(-d["l30"]) + (zn(-d["h15"]) if use_h else 0.0) + (wk * zn(-d["t30"]) if Ytrk is not None else 0.0) + wdeep * zn(cosd)
        if opt["late"]: Fm = Fm + opt["wk2"] * zn(-d["u30"])
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
    X = np.concatenate(blocks, axis=2).astype(np.float32); assert np.isfinite(X).all()
    return dict(X=X.reshape(nq * kk, -1), top=top, F=Fm, k=kk)
