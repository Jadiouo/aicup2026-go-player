"""E38 shared helpers on top of p1111_sim (same pools, same Z, same HSL Y).  CPU only.  Never loads H (p1111_sim prep excluded them)."""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
import p1111_sim as S
import numpy as np
import pandas as pd
import lightgbm as lgb
from gogame import player_rerank as prr
S3 = S.S3
SC = ROOT / ".scratch" / "e38"; SC.mkdir(parents=True, exist_ok=True)


def pools(D, seeds=(1, 2), npool=4, structs=("a", "b"), **kw):
    out = []
    for st in structs:
        for sd in seeds:
            for pi in range(npool):
                rng = np.random.default_rng(1000 * sd + 10 * pi + (1 if st == "a" else 2))
                p = S.make_pool(D, "eval", rng, st, **kw); p["tag"] = f"{st}|{sd}|{pi}"; out.append(p)
    return out


def fit_lda(D, cols, ridge=1e-2, mmax=100, colour=None):
    """LDA metric on Z[:, cols] from TRAIN players (both colours unless `colour` given)."""
    tr = np.flatnonzero((D.meta.grp == "train").to_numpy())
    if colour is not None:
        tr = tr[D.color[tr] == colour]
    X = D.Z[tr][:, cols].astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    gm, V, ev = S3.fit_metric((X - mu) / sd, D.meta["player_id"].to_numpy()[tr], ridge=ridge, mmax=mmax)
    return dict(cols=cols, mu=mu, sd=sd, gm=gm, V=V, ev=ev)


def apply_lda(D, M, rows=None):
    Z = D.Z if rows is None else D.Z[rows]
    out = np.empty((len(Z), M["V"].shape[1]), np.float32)
    for a in range(0, len(Z), 20000):
        out[a:a + 20000] = ((Z[a:a + 20000][:, M["cols"]].astype(np.float64) - M["mu"]) / M["sd"] - M["gm"]) @ M["V"]
    return out


def rank_stats(sim, truth):
    """sim (nq,P) higher=better -> rank of truth (1-based)."""
    t = sim[np.arange(len(truth)), truth]
    return 1 + (sim > t[:, None]).sum(1)


def score_from_rank(r):
    return np.where(r <= 5, np.exp(-(r - 1.0)), 0.0)
