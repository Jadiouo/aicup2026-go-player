"""E127 data + episode sampler (GPU tensors).  Positions = P2_59 (own-move ordinals 2..59 of every game, <=58 per game), 128-d vectors of ONE S network
(finh_h0 for the h0 matcher, finh_h1 for the h1 matcher).  Vectors are whitened by a per-net std (from the training positions) and L2-normalised, stored fp16, zero padded to PMAX per game.
Pool = games grouped by player (every player has 16..24 games).  Episode = (query N games of a positive player, K candidates = positive + hard negatives, each with LS library games, M positions per game)."""
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; OUT = ROOT / ".scratch/e127"; PMAX = 60


def _read(net, part):
    d = np.load(OUT / f"pos_{net}_{part}.npz"); m = np.load(OUT / f"meta_{net}_{part}.npz") if part != "test" else None
    return d, m


def fit_sigma(net, parts=("trA", "trB")):
    s = np.zeros(128); s2 = np.zeros(128); n = 0
    for p in parts:
        d, _ = _read(net, p); v = d["vec"].astype(np.float64); s += v.sum(0); s2 += (v ** 2).sum(0); n += len(v)
    return np.sqrt(s2 / n - (s / n) ** 2).astype(np.float32)


def prep(vec, sigma, dev="cpu"):
    v = torch.from_numpy(np.ascontiguousarray(vec)).to(dev).float() / torch.from_numpy(sigma).to(dev); return F.normalize(v, dim=-1)


class Pool:
    """games of one or more parts, padded.  attrs on `dev`: GV [n,PMAX,128] fp16 (whitened+normalised), GO [n,PMAX] long ordinal, GN [n] count, EFF [n] long, GM [n,128] float (L2-normalised mean over ALL own positions = S game vector, from export),
    player index per game, PG [nP,maxg] games of each player (-1 pad), PCNT [nP], PC [nP,128] player centroid of GM."""
    def __init__(self, net, parts, sigma, dev="cuda", with_players=True):
        self.net, self.dev, self.sigma = net, dev, sigma; GV, GO, GN, EFF, GM, PL, ROWS = [], [], [], [], [], [], []
        for part in parts:
            d, m = _read(net, part); vec, off, ords = d["vec"], d["off"], d["ords"]; n = len(off) - 1; cnt = np.diff(off); assert cnt.max() <= PMAX and cnt.min() >= 1
            gi = np.repeat(np.arange(n), cnt); pi = np.arange(len(vec)) - off[gi]
            g = torch.zeros(n, PMAX, 128, dtype=torch.float16, device=dev); o = torch.zeros(n, PMAX, dtype=torch.long, device=dev)
            for a in range(0, len(vec), 1 << 20):
                sl = slice(a, a + (1 << 20)); gii = torch.from_numpy(gi[sl]).to(dev); pii = torch.from_numpy(pi[sl]).to(dev)
                g[gii, pii] = prep(vec[sl], sigma, dev).half(); o[gii, pii] = torch.from_numpy(ords[sl].astype(np.int64)).to(dev)
            GV.append(g); GO.append(o); GN.append(torch.from_numpy(cnt).to(dev)); GM.append(torch.from_numpy(d["gv_all"]).to(dev)); ROWS.append(d["rows"])
            if m is not None: assert (m["rows"] == d["rows"]).all(); EFF.append(torch.from_numpy(m["eff"].astype(np.int64)).to(dev)); PL.append(m["player"])
            if m is None: EFF.append(torch.from_numpy(np.load(ROOT / '.scratch/e90/data/t_games.npz')['eff'][d['rows']].astype(np.int64)).to(dev))   # test part: eff of t_games rows
            del d
        self.GV, self.GO, self.GN, self.GM = torch.cat(GV), torch.cat(GO), torch.cat(GN), torch.cat(GM); self.rows = np.concatenate(ROWS)
        if EFF: self.EFF = torch.cat(EFF)
        if with_players and PL:
            pl = np.concatenate(PL); ups, inv = np.unique(pl, return_inverse=True); self.players = ups; nP = len(ups); mx = np.bincount(inv).max(); PG = np.full((nP, mx), -1, np.int64); cn = np.zeros(nP, np.int64)
            for g, p in enumerate(inv): PG[p, cn[p]] = g; cn[p] += 1
            self.pidx = inv; self.PG = torch.from_numpy(PG).to(dev); self.PCNT = torch.from_numpy(cn).to(dev)
            self.PC = F.normalize(torch.stack([self.GM[self.PG[p, :cn[p]]].mean(0) for p in range(nP)]), dim=-1)
        self.n = len(self.GN)

    def sample_pos(self, games, M, gen=None):
        """games: long tensor [...]; returns vec [...,M,128] fp16, ord [...,M] long, mask [...,M] bool (M random distinct valid positions per game; all valid positions if M >= count)."""
        cnt = self.GN[games]; r = torch.rand(*games.shape, PMAX, device=self.dev, generator=gen); ar = torch.arange(PMAX, device=self.dev)
        r = torch.where(ar < cnt[..., None], r, torch.full_like(r, 2.0)); idx = r.topk(min(M, PMAX), dim=-1, largest=False).indices; idx, _ = idx.sort(-1)
        v = torch.gather(self.GV[games], -2, idx[..., None].expand(*idx.shape, 128)); o = torch.gather(self.GO[games], -1, idx); mk = idx < cnt[..., None]; return v, o, mk

    def all_pos(self, games):
        return self.GV[games], self.GO[games], torch.arange(PMAX, device=self.dev) < self.GN[games][..., None]


def episode(pool, B, N, LS, M, K=40, topn=60, gen=None, MQ=None, fixed_players=None):
    """one batch of B episodes with shared (N query games, LS library games, M positions).  Returns dict: qv [B,N*MQ,128], qo, qm, qc (colour per position); yv [B,K,LS*M,128], yo, ym, yc; label [B] (slot of the positive)."""
    dev = pool.dev; MQ = MQ or M; nP = len(pool.players)
    pos = (torch.randint(0, nP, (B,), device=dev, generator=gen) if fixed_players is None else fixed_players)
    PGp = pool.PG[pos]; cn = pool.PCNT[pos]; r = torch.rand(B, PGp.shape[1], device=dev, generator=gen); r = torch.where(PGp >= 0, r, torch.full_like(r, 2.0)); perm = r.argsort(1)
    PGs = torch.gather(PGp, 1, perm); assert int(cn.min()) >= N + LS
    qg = PGs[:, :N]; lg = PGs[:, N:N + LS]                                                       # query games / positive library games
    qm_ = F.normalize(pool.GM[qg].mean(1), dim=-1); sims = qm_ @ pool.PC.T; sims[torch.arange(B), pos] = -9.0
    top = sims.topk(topn, dim=1).indices; pick = torch.rand(B, topn, device=dev, generator=gen).argsort(1)[:, :K - 1]; neg = torch.gather(top, 1, pick)       # [B,K-1] hard negatives (S-cosine neighbours)
    label = torch.randint(0, K, (B,), device=dev, generator=gen); cand = torch.empty(B, K, dtype=torch.long, device=dev); ar = torch.arange(K, device=dev)
    other = torch.where(ar[None] < label[:, None], ar[None], ar[None] - 1).clamp(0, K - 2); cand = torch.gather(neg, 1, other); cand[torch.arange(B), label] = pos
    CG = pool.PG[cand]; CC = pool.PCNT[cand]; r = torch.rand(B, K, CG.shape[2], device=dev, generator=gen); r = torch.where(CG >= 0, r, torch.full_like(r, 2.0)); CGs = torch.gather(CG, 2, r.argsort(2))[:, :, :LS]
    CGs[torch.arange(B), label] = lg                                                              # positive's library games exclude the query games
    scos = torch.einsum("bd,bkd->bk", qm_, F.normalize(pool.GM[CGs].mean(2), dim=-1))        # S cosine: query game-vector mean vs candidate library mean (the baseline the matcher is trained to add to)
    qv, qo, qk = pool.sample_pos(qg, MQ, gen); yv, yo, yk = pool.sample_pos(CGs, M, gen)
    qc = pool.EFF[qg][..., None].expand(B, N, MQ).reshape(B, -1); yc = pool.EFF[CGs][..., None].expand(B, K, LS, M).reshape(B, K, -1)
    return dict(qv=qv.reshape(B, N * MQ, 128), qo=qo.reshape(B, -1), qm=qk.reshape(B, -1), qc=qc, yv=yv.reshape(B, K, LS * M, 128), yo=yo.reshape(B, K, -1), ym=yk.reshape(B, K, -1), yc=yc, label=label, pos=pos, cand=cand, cg=CGs, qg=qg, scos=scos)
