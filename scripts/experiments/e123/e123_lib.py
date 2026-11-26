"""E123 library: channel X_lr = row-z of the mean log-probability of a 600-class logistic-regression head trained on the pool's candidate-library games
(PLAN experiments/E123_player_pool_head/PLAN.md, 'Zuo fa').  Written 2026-11-23.
Vectors (F3): per group row-L2 -> concat (384) -> row-L2 -> *8.   Head: sklearn LogisticRegression(C=1.0) (L2, multinomial), max_iter large, ConvergenceWarning is an ERROR.
"""
import sys, warnings, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.dont_write_bytecode = True; sys.path[:0] = [str(HERE), str(ROOT / "scripts/experiments/e112"), str(ROOT / "scripts/lib")]
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
OUT = ROOT / ".scratch/e123"; DUMPS = ROOT / ".scratch/e110/dumps"
GROUPS = ("e90/r_h", "e99/r", "e86/r")          # small S, big S, pe86
SCALE = 8.0; NC = 600; MAXITER = 5000


def nz(a): return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def load_vecs(s, groups=GROUPS):
    """[7100, 128*len(groups)] float32, row order = .scratch/p1111/z_t2/Z_{s}.npz uids (candidates 0..5999 first, query games after)."""
    parts = []
    for g in groups:
        a = np.load(ROOT / f".scratch/{g}/pe_test_{s}.npy").astype(np.float64); assert a.shape == (7100, 128) and np.isfinite(a).all(); n = np.linalg.norm(a, axis=1); assert n.min() > .5, (g, n.min()); parts.append(nz(a))
    v = nz(np.concatenate(parts, 1)) * SCALE; return v.astype(np.float32)


def zrow(S): return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def fit_head(X, y, C=1.0):
    """returns (log-softmax function, n_iter).  Any ConvergenceWarning raises."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning); m = LogisticRegression(C=C, max_iter=MAXITER); m.fit(X, y)
    assert (m.classes_ == np.arange(NC)).all(); n_iter = int(np.max(m.n_iter_)); W = m.coef_.astype(np.float64); b = m.intercept_.astype(np.float64)
    def logp(Q):
        z = Q.astype(np.float64) @ W.T + b; z -= z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
    return logp, n_iter


def centroid_cos(Xlib, ylib, Q):
    """control channel: cosine of each query game to each candidate's centroid (of the library vectors)."""
    cen = np.stack([Xlib[ylib == c].mean(0) for c in range(NC)]); return nz(Q.astype(np.float64)) @ nz(cen).T


def query_channel(game_scores, qrows):
    """game_scores [n_games_total_in_rows..., 600] indexed by candidate-table row (callable via dict) -> per-query mean over its games, then row-z.
    here game_scores: dict row -> [600]"""
    out = np.zeros((len(qrows), NC))
    for i, q in enumerate(qrows): out[i] = np.mean([game_scores[g] for g in q if g >= 0], 0)
    return zrow(out)


# ------------------------------------------------------------------------------------------------ pseudo-query channel (per rep, per structure/pool)
TAGS = ("", "c", "b"); STRUCT = {"": "a", "c": "c", "b": "b"}; POOLS = (("pub", "public"), ("priv", "private"))
import e112_stage2 as S


def check_rows(s, sp):
    """independent check of the candidate-table-row -> export-row map; returns row [6000]."""
    row = S.cand_rows(s, sp); uids = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False)["uids"].astype(str)
    with open(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", newline="") as f: rs = list(csv.DictReader(f))
    assert len(rs) == 6000 and len(uids) == 7100 and len(row) == 6000 and len(set(row)) == 6000 and row.max() < 6000 and row.min() >= 0, "candidate rows must be the first 6000 export rows"
    for i, r in enumerate(rs):
        g = r["game_id"].split("_g_")[1]; assert uids[row[i]] == f"cand_{s}:{s}_g_{g}", (i, uids[row[i]], r["game_id"])
    assert all(u.startswith(f"{s}_q_") and "#" in u for u in uids[6000:]), "rows 6000.. are the query games"
    return row


def check_players(z, sp):
    """dump's players / cand_player agree with the candidate csv (class id = index in the sorted player list)."""
    with open(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", newline="") as f: rs = list(csv.DictReader(f))
    pl = sorted({r["player_id"] for r in rs}); assert len(pl) == NC and list(z["players"]) == pl and [pl.index(r["player_id"]) for r in rs] == list(z["cand_player"]) and (np.bincount(z["cand_player"]) == 10).all()


def assert_rep(z, r, row_unused=None):
    """per-rep assertions for the dump z (npz): library 6 games per candidate with the right candidate id; query games never in the library; query games belong to the truth player."""
    lib = z["lib"][r]; cp = z["cand_player"]; rep, qrows, truth = z["rep"], z["qrows"], z["truth"]; m = rep == r
    assert lib.shape == (NC, 6) and (cp[lib.reshape(-1)] == np.repeat(np.arange(NC), 6)).all(), "library row -> candidate id mismatch"
    assert len(set(lib.reshape(-1).tolist())) == NC * 6, "duplicate library rows"
    qr = qrows[m]; ug = qr[qr >= 0]; assert len(ug) and not (set(ug.tolist()) & set(lib.reshape(-1).tolist())), "query game inside the library"
    for q, t in zip(qr, truth[m]):
        g = q[q >= 0]; assert len(g) >= 1 and (cp[g] == t).all(), "query game does not belong to the truth player"
    assert m.sum() == 400
    return lib


def build_X(tag, s, sp, v, row, mode="head", reps=None, seed=123, log=None):
    """X [8000,600] float32 (rows of other reps stay 0 when `reps` is given).  mode: head | shuf | cent."""
    z = np.load(DUMPS / f"t2{tag}_{s}.npz"); check_players(z, sp); rep, qrows = z["rep"], z["qrows"]; X = np.zeros((len(rep), NC), np.float32); iters = []
    rng = np.random.default_rng([seed, TAGS.index(tag), [p[0] for p in POOLS].index(s)])
    for r in (range(int(rep.max()) + 1) if reps is None else reps):
        lib = assert_rep(z, r); m = np.flatnonzero(rep == r); qr = qrows[m]; ug = np.unique(qr[qr >= 0]); slot = {g: j for j, g in enumerate(ug)}
        Xtr = v[row[lib.reshape(-1)]]; ytr = np.repeat(np.arange(NC), 6); Q = v[row[ug]]
        if mode == "shuf": Xtr = Xtr[rng.permutation(len(Xtr))]            # vectors permuted relative to the candidate labels (null control)
        if mode in ("head", "shuf"): logp, it = fit_head(Xtr, ytr); iters.append(it); G = logp(Q)
        elif mode == "cent": G = centroid_cos(Xtr, ytr, Q)
        else: raise ValueError(mode)
        A = np.zeros((len(m), NC))
        for i, q in enumerate(qr): A[i] = G[[slot[g] for g in q if g >= 0]].mean(0)
        X[m] = zrow(A).astype(np.float32)
        if log: log(f"    {tag or 'a'} {s} {mode} rep {r} iters {iters[-1] if iters else '-'}")
    return X, iters


def per_query2(tag, s, Xe_file, Xl=None, w_lr=1.0, w_e=.5, reps=None):
    """current (v7u: z_row(e)+.5*X_e112 -> csls_qq(3,.5,z), literally the arithmetic of e112_stage2.per_query) and, if Xl is given, the same plus w_lr*Xl[short list] before the SAME csls_qq.
    returns dict(cur, new, top1_cur, top1_new, truth, n, rep)."""
    sc, csls_qq, rank_after = S.scorer(); z = np.load(DUMPS / f"t2{tag}_{s}.npz"); rep, e, top, truth = z["rep"], z["e"], z["top"].astype(np.int64), z["truth"]; Xe = np.load(Xe_file)["X"]
    cur = np.zeros(len(rep)); new = np.zeros(len(rep)); t1c = np.zeros(len(rep), np.int64); t1n = np.zeros(len(rep), np.int64)
    for r in (sorted(set(rep)) if reps is None else reps):
        m = rep == r; zs = (e[m] - e[m].mean(1, keepdims=True)) / (e[m].std(1, keepdims=True) + 1e-9); Xs = np.take_along_axis(Xe[m], top[m], 1); c = zs + w_e * Xs
        rk, ts = rank_after(csls_qq(c, top[m], NC, S.K, S.ALPHA, "z"), top[m], truth[m]); cur[m] = sc(rk); t1c[m] = ts[:, 0]
        if Xl is not None:
            Ls = np.take_along_axis(Xl[m], top[m], 1); c2 = c + w_lr * Ls
            rk, ts = rank_after(csls_qq(c2, top[m], NC, S.K, S.ALPHA, "z"), top[m], truth[m]); new[m] = sc(rk); t1n[m] = ts[:, 0]
    return dict(cur=cur, new=new, top1_cur=t1c, top1_new=t1n, truth=truth, n=z["n"], rep=rep)
