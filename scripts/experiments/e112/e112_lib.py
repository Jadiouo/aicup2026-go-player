"""E112 shared helpers: position-set selection, ragged position-vector files, KDE likelihood-ratio channel (torch).  Written 2026-11-22.
Position sets (PLAN: P12 / P60 / PALL), all on the game's own-move ordinals `vo` (fastboard.valid_ordinals, passes counted, non-pass own moves):
  P12  = cand = vo[(vo>=2)&(vo<60)] (all vo if empty); if len(cand)>12 take unique(round(linspace(0,len-1,12))) of cand      (= consult k5)
  P60  = vo[vo<60] (all vo if empty)
  PALL = vo
Kernel (fixed by PLAN): whitened (divide every dim by the std over the pool's candidate-library positions) then L2-normalised vectors;
  score(x, c) = log mean_{y in lib_c} exp(20 (cos(x,y)-1)) - log mean_c' ( that mean ); query score = mean over ALL of the query's positions (pooled over its games).
"""
import numpy as np
BETA = 20.0


def select_idx(ords, pset):
    ords = np.asarray(ords)
    if pset == "PALL": return np.arange(len(ords))
    if pset == "P60":
        i = np.flatnonzero(ords < 60); return i if len(i) else np.arange(len(ords))
    if pset == "P12":
        c = np.flatnonzero((ords >= 2) & (ords < 60))
        if len(c) == 0: c = np.arange(len(ords))
        return c if len(c) <= 12 else c[np.unique(np.round(np.linspace(0, len(c) - 1, 12)).astype(int))]
    raise ValueError(pset)


class Pos:
    """Ragged per-game position vectors: vec [N,128] float16, off [n+1], ords [N] int16, rid [n]."""
    def __init__(self, path, pset=None):
        d = np.load(path); self.rid = d["rid"]; vec, off, ords = d["vec"], d["off"], d["ords"]; self.n = len(off) - 1
        if pset is None: self.vec, self.off, self.ords = vec, off, ords
        else:
            keep = [off[g] + select_idx(ords[off[g]:off[g + 1]], pset) for g in range(self.n)]
            cnt = np.array([len(k) for k in keep]); self.off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
            ix = np.concatenate(keep); self.vec, self.ords = vec[ix], ords[ix]
        self.cnt = np.diff(self.off)

    def games(self, gl):
        """concatenated vectors of games gl (list/array of game indices) as float32 + per-position game slot (index into gl)."""
        parts = [self.vec[self.off[g]:self.off[g + 1]] for g in gl]; slot = np.concatenate([np.full(len(p), j, np.int64) for j, p in enumerate(parts)])
        return np.concatenate(parts).astype(np.float32), slot


def nz(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def kde_game_sums(Qv, Qslot, nslot, Lv, Llens, beta=BETA, budget=6e7, dev="cuda"):
    """Qv [m,128] whitened+normalised query positions (float32 numpy), Qslot [m] game slot in 0..nslot-1;
    Lv [Nl,128] library positions sorted by candidate, Llens [nC] positions per candidate (all > 0).
    Returns (sums [nslot,nC] float64 = sum over the slot's positions of the log-likelihood-ratio, cnt [nslot])."""
    import torch
    L = torch.from_numpy(np.ascontiguousarray(Lv)).to(dev); lens = torch.from_numpy(Llens.astype(np.int64)).to(dev); nC = len(Llens); invn = (1.0 / lens.double())
    cand = torch.repeat_interleave(torch.arange(nC, device=dev), lens)
    out = torch.zeros(nslot, nC, dtype=torch.float64, device=dev); cnt = np.bincount(Qslot, minlength=nslot)
    mb = max(16, int(budget // max(1, len(Lv)))); slot_t = torch.from_numpy(Qslot).to(dev)
    for a in range(0, len(Qv), mb):
        q = torch.from_numpy(np.ascontiguousarray(Qv[a:a + mb])).to(dev); M = torch.exp(beta * (q @ L.T - 1.0))              # [m,Nl] float32
        e = torch.zeros(len(q), nC, dtype=torch.float32, device=dev).index_add_(1, cand, M).double() * invn[None]          # mean over the candidate's library positions
        lr = torch.log(e + 1e-30) - torch.log(e.mean(1, keepdim=True) + 1e-30)
        out.index_add_(0, slot_t[a:a + mb], lr); del M, e, lr
    return out.cpu().numpy(), cnt
