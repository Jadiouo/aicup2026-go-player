"""E81 shared analysis helpers (pure numpy/pandas; reads only the E81 dumps)."""
import numpy as np, pandas as pd

def sc(rank): return np.where(rank <= 5, np.exp(-(np.minimum(rank, 6) - 1.0)), 0.0)

def strat(df, key, label=None, score="score"):
    """per-stratum: n share, mean score, share of total loss (loss = 1 - score)."""
    g = df.groupby(key, observed=True)[score].agg(["size", "mean"]); loss = df.assign(l=1 - df[score]).groupby(key, observed=True).l.sum()
    g["q_share"] = g["size"] / len(df); g["loss_share"] = loss / loss.sum(); g = g.rename(columns={"mean": "score"})[["size", "q_share", "score", "loss_share"]]
    print(f"-- by {label or key}"); print(g.round(4).to_string()); return g

def rank_bucket(rank):
    return pd.cut(rank, [0, 1, 2, 3, 5, 10, 40, 10 ** 6], labels=["1", "2", "3", "4-5", "6-10", "11-40", "out"])

def rank_after(e, top, truth):
    order = np.argsort(-e, axis=1, kind="stable"); ts = np.take_along_axis(top, order, 1); hit = ts == truth[:, None]
    return np.where(hit.any(1), hit.argmax(1) + 1, 999), ts

def softmax(e):
    x = e - e.max(1, keepdims=True); p = np.exp(x); return p / p.sum(1, keepdims=True)

def csls_qq(e, top, P, k, alpha, mode="z"):
    """CSLS on the booster output (query-side hubness, transductive over the queries of ONE pool).
    e (nq,K) booster scores of the shortlist `top` (nq,K candidate ids); P candidates.
    mode z : s = per-query z-score of e inside the shortlist (out of shortlist = missing);  mode p : s = log softmax prob.
    r_c = mean of the k largest s(q',c) over the queries q' whose shortlist contains c (fewer than k -> padded with the floor value).
    returns adjusted scores s - alpha * r_c."""
    if mode == "z": s = (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-9); floor = s.min()
    else: s = np.log(softmax(e) + 1e-12); floor = s.min()
    nq, K = e.shape; M = np.full((nq, P), floor, np.float64); np.put_along_axis(M, top.astype(np.int64), s, 1)
    r = np.sort(M, axis=0)[-k:].mean(0)                                    # (P,)
    return s - alpha * r[top]

def csls_self_excl(e, top, P, k, alpha, mode="z"):
    """same, but r_c for query q excludes q's own score (leave-one-out), so a candidate is not penalised by the query it is being scored for."""
    if mode == "z": s = (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-9); floor = s.min()
    else: s = np.log(softmax(e) + 1e-12); floor = s.min()
    nq, K = e.shape; M = np.full((nq, P), floor, np.float64); np.put_along_axis(M, top.astype(np.int64), s, 1)
    srt = np.sort(M, axis=0)[-(k + 1):]                                    # k+1 largest per candidate, ascending
    tot_k = srt[1:].sum(0); kth1 = srt[0]                                  # sum of the k largest; the (k+1)-th largest
    sc_ = s; cand = top.astype(np.int64)
    in_topk = sc_ >= srt[1][cand]                                          # own score is among the k largest of that candidate
    r = np.where(in_topk, (tot_k[cand] - sc_ + kth1[cand]) / k, tot_k[cand] / k)
    return s - alpha * r

def boot_ci(d, groups, B=2000, seed=0):
    """cluster bootstrap over groups (e.g. reps or pools) of a per-query difference d."""
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = [np.flatnonzero(groups == g) for g in ug]; o = np.empty(B)
    for b in range(B):
        gi = rng.integers(0, len(ug), len(ug)); o[b] = np.mean([d[idx[i]][rng.integers(0, len(idx[i]), len(idx[i]))].mean() for i in gi])
    return np.percentile(o, [2.5, 97.5])

def hub_stats(top1, truth, P):
    """top1: predicted candidate per query; -> k-occurrence counts and how the wrong answers concentrate."""
    wrong = top1 != truth; cnt = np.bincount(top1, minlength=P); wcnt = np.bincount(top1[wrong], minlength=P)
    return cnt, wcnt
