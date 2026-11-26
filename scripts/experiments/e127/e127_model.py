"""E127 matchers.
 kind 'kde'  (P-2, guard arm): s(x_i, Y_c) = log mean_j exp(beta * cos(P x_i, P y_j)); logit_c = scale * mean_i s.  P [128->d] initialised to the identity (d=128), beta init 20 = the E112 fixed kernel (listwise softmax removes the per-query constant of E112's LR).
 kind 'cross' (P-1): the same kde term + a cross-attention residual (zero-initialised output layer, so it starts as P-2): optional self-attention encoder over the candidate's positions (and the query's), then per head
                    a_ij = beta_h cos(Wq h_i, Wk k_j) over the candidate's library positions, ctx_i = softmax_j(a) V k_j, features [h_i*ctx_i, |h_i-ctx_i|, lse_h] -> MLP -> per-position score.
Inputs per batch: qv [B,Nq,128] qo qm qc ; yv [B,K,Ny,128] yo ym yc (vectors already whitened + L2 normalised)."""
import math, torch, torch.nn as nn, torch.nn.functional as F


def lme(a, m):
    """log mean exp over last dim with boolean mask m (broadcastable)."""
    a = a.masked_fill(~m, -1e4); n = m.float().sum(-1).clamp(min=1); return torch.logsumexp(a, -1) - torch.log(n)


class SA(nn.Module):
    def __init__(s, d, H, drop):
        super().__init__(); s.H = H; s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d); s.ln2 = nn.LayerNorm(d); s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)); s.drop = drop
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias); nn.init.zeros_(s.ff[2].weight); nn.init.zeros_(s.ff[2].bias)

    def forward(s, x, m):          # x [G,T,d], m [G,T] bool
        G, T, d = x.shape; q, k, v = s.qkv(s.ln1(x)).view(G, T, 3, s.H, d // s.H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m[:, None, None, :], dropout_p=s.drop if s.training else 0.0).transpose(1, 2).reshape(G, T, d)
        x = x + s.o(a); return x + s.ff(s.ln2(x))


class Matcher(nn.Module):
    def __init__(s, kind="kde", d=128, H=4, hid=128, drop=0.1, cand_sa=1, qry_sa=0, beta0=20.0, knorm=True, resid=False):
        super().__init__(); s.kind, s.d, s.H, s.knorm, s.resid = kind, d, H, knorm, resid; 
        if resid: s.a_s = nn.Parameter(torch.tensor(20.0))
        s.P = nn.Linear(128, d, bias=False)
        with torch.no_grad(): s.P.weight.copy_(torch.eye(d, 128) if d <= 128 else torch.randn(d, 128) * 0.1)
        s.logbeta = nn.Parameter(torch.tensor(math.log(beta0))); s.scale = nn.Parameter(torch.tensor(1.0))
        if kind == "cross":
            s.ord = nn.Embedding(PMAXORD, d); s.col = nn.Embedding(2, d); s.inp = nn.Linear(128, d); nn.init.normal_(s.ord.weight, std=0.02); nn.init.normal_(s.col.weight, std=0.02); s.lni = nn.LayerNorm(d)
            s.csa = nn.ModuleList([SA(d, H, drop) for _ in range(cand_sa)]); s.qsa = nn.ModuleList([SA(d, H, drop) for _ in range(qry_sa)])
            s.wq = nn.Linear(d, d); s.wk = nn.Linear(d, d); s.wv = nn.Linear(d, d); s.logb = nn.Parameter(torch.full((H,), math.log(10.0))); s.drop = nn.Dropout(drop)
            s.mlp = nn.Sequential(nn.Linear(3 * d + H, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, 1)); nn.init.zeros_(s.mlp[3].weight); nn.init.zeros_(s.mlp[3].bias)

    def embed(s, v, o, c):
        return s.lni(s.inp(v.float()) + s.ord(o.clamp(max=PMAXORD - 1)) + s.col(c))

    def enc_cands(s, b):
        """candidate-side encoding (independent of the query): returns dict(py [B,K,Ny,d], ym [B,K,Ny], and for 'cross': k [B,K,Ny,H,dh], v)."""
        yv, ym = b["yv"].float(), b["ym"]; B, K, Ny, _ = yv.shape; e = dict(py=F.normalize(s.P(yv), dim=-1), ym=ym)
        if s.kind == "cross":
            hy = s.embed(yv, b["yo"], b["yc"]).reshape(B * K, Ny, -1); my = ym.reshape(B * K, Ny)
            for l in s.csa: hy = l(hy, my)
            H, dh = s.H, s.d // s.H; e["k"] = F.normalize(s.wk(hy).view(B, K, Ny, H, dh), dim=-1); e["v"] = s.wv(hy).view(B, K, Ny, H, dh)
        return e

    def score(s, b, e):
        """b: query side (qv, qo, qm, qc) with batch dim B matching e's B (e may be expanded over B by the caller); returns logits [B,K]."""
        qv, qm, py, ym = b["qv"].float(), b["qm"], e["py"], e["ym"]; B, K, Ny, _ = py.shape; Nq = qv.shape[1]; beta = s.logbeta.exp()
        pq = F.normalize(s.P(qv), dim=-1); sim = torch.einsum("bqd,bkyd->bkqy", pq, py); kde = lme(beta * sim, ym[:, :, None, :]); score = kde            # [B,K,Nq]
        if s.kind == "cross":
            hq = s.embed(qv, b["qo"], b["qc"])
            for l in s.qsa: hq = l(hq, qm)
            H, dh = s.H, s.d // s.H; q = F.normalize(s.wq(hq).view(B, Nq, H, dh), dim=-1)
            a = torch.einsum("bqhe,bkyhe->bkhqy", q, e["k"]) * s.logb.exp()[None, None, :, None, None]; mk = ym[:, :, None, None, :]
            lse = lme(a, mk).permute(0, 1, 3, 2); al = torch.softmax(a.masked_fill(~mk, -1e4), -1); ctx = torch.einsum("bkhqy,bkyhe->bkqhe", al, e["v"]).reshape(B, K, Nq, -1)
            h = hq[:, None].expand(B, K, Nq, -1); f = torch.cat([h * ctx, (h - ctx).abs(), ctx, lse], -1); score = kde + s.mlp(s.drop(f)).squeeze(-1)
        if s.knorm and K > 1: score = score - torch.logsumexp(score, 1, keepdim=True) + math.log(K)       # E112's likelihood ratio: per query position, minus log mean over the K candidates
        w = qm[:, None, :].float(); out = s.scale * (score * w).sum(-1) / w.sum(-1).clamp(min=1)
        if s.resid and b.get("scos") is not None: out = out + s.a_s * b["scos"].float()          # training only: matcher learns what S cosine does not already explain; X = matcher part (no scos)
        return out

    def forward(s, b):
        return s.score(b, s.enc_cands(b))


PMAXORD = 64
