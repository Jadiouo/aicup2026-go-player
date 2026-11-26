"""E39: does fitting the Player metrics (L0 LDA + HSL LDA) on MORE players / games help?  Same sim as E38 (16 pools of 1,481 EVAL players, LHx recipe), only the metric-fit population changes.
Metric fit sets come from streaming statistics (e39_stats.py): every non-H, non-EVAL player; EVAL players are never in any fit set (asserted).  Cross-fit rule kept: pools of train half h are scored with metrics
that exclude every player of half h (they use the other half + the extra low-activity players).
Scales (--scale):  S0 = sim train players only, cap 24/colour (E38 baseline) | Smid = + half of the extra players | S1 = + all extra players (cap 24) | S2 = S1 with cap 60 | S3 = S1 with all games | S0all = S0 players, all games
Commands: stage1 --scales ...            first-stage only (no booster), 16 pools, prints score/R@40
          run --scale S1 --seed 7 --out NAME   full LHx train (60 pools x 600 rounds, K=40) + eval -> .scratch/e39/res_NAME.json
"""
import argparse, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
import e38_final as EF
from e38_core import pools, score_from_rank, rank_stats
S, S3 = EF.S, EF.S3
import numpy as np, pandas as pd, scipy.linalg, lightgbm as lgb
from gogame import player_rerank as prr
OUT = ROOT / ".scratch" / "e39"
FAM = {  # scale -> (cap, groups at test time, extra groups added to the OTHER half for cross-fitting)
    "S0": (24, ["TA", "TB", "X78"], []), "Smid": (24, ["TA", "TB", "X78", "E1"], ["E1"]), "S1": (24, ["TA", "TB", "X78", "E1", "E2"], ["X78", "E1", "E2"]),
    "S2": (60, ["TA", "TB", "X78", "E1", "E2"], ["X78", "E1", "E2"]), "S3": (10 ** 9, ["TA", "TB", "X78", "E1", "E2"], ["X78", "E1", "E2"]),
    "S0all": (10 ** 9, ["TA", "TB", "X78"], []),
}
FAM["S0"] = (24, ["TA", "TB", "X78"], [])
FAM["S0c60"] = (60, ["TA", "TB", "X78"], [])
FAM["Smid3"] = (10 ** 9, ["TA", "TB", "X78", "E1"], ["E1"])
RIDGE_L, RIDGE_H, M_L, M_H = 3e-2, 1e-2, 150, 40


def load_stats():
    z = np.load(OUT / "stats2.npz")
    return {k: z[k] for k in z.files}


def lda_from_stats(ST, groups, cap, ridge=RIDGE_L, mmax=M_L, kappa=None):
    """kappa=None: between-player scatter = unweighted covariance of player means (as fit_metric); kappa=k: player weight n_p/(n_p+k) (precision-style; k=0 -> equal weights)."""
    g = lambda k: sum(ST[f"{x}|{cap}|{k}"] for x in groups)  # noqa: E731
    N, C = float(g("n")), float(g("C"))
    mu = g("sx") / N; sd = np.sqrt(np.maximum(g("sx2") / N - mu ** 2, 0)) + 1e-9
    d = len(mu)
    Sw = g("Sw") / (N - C) / np.outer(sd, sd)
    Sw += ridge * np.trace(Sw) / d * np.eye(d)
    if kappa is None:
        mb = g("sm") / C
        Sb = (g("smm") - C * np.outer(mb, mb)) / (C - 1) / np.outer(sd, sd)
    else:
        PM = np.concatenate([ST[f"{x}|{cap}|pm"] for x in groups]); pn = np.concatenate([ST[f"{x}|{cap}|pn"] for x in groups])
        w = pn / (pn + kappa); mb = (w[:, None] * PM).sum(0) / w.sum()
        Xc = PM - mb
        Sb = (Xc.T * w) @ Xc / (w.sum() - (w ** 2).sum() / w.sum()) / np.outer(sd, sd)
    ev, V = scipy.linalg.eigh(Sb, Sw)
    o = np.argsort(-ev); V = V[:, o][:, :mmax]
    return dict(mu=mu, sd=sd, gm=(mb - mu) / sd, V=V.astype(np.float32), ev=ev[o], N=int(N), C=int(C))


def selected_rows(meta, rank, groups_json, groups, cap):
    P = set().union(*[set(groups_json[x]) for x in groups])
    pid = meta["player_id"].to_numpy()
    return np.flatnonzero(np.isin(pid, list(P)) & (rank < cap))


def hsl_metric(Xh, pid, rows):
    H = Xh[rows]; hmu, hsd = H.mean(0), H.std(0) + 1e-9
    g2, V2, _ = S3.fit_metric((H - hmu) / hsd, pid[rows], ridge=RIDGE_H, mmax=M_H)
    return dict(hmu=hmu, hsd=hsd, hgm=g2, hV=V2)


class Ctx:
    def __init__(self, kappa=None):
        self.kappa = kappa
        self.meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
        pid, col = self.meta["player_id"].to_numpy(), self.meta["color"].to_numpy()
        key = np.random.default_rng(39).random(len(self.meta))
        self.rank = pd.Series(key).groupby([pid, col]).rank(method="first").to_numpy() - 1
        self.pid = pid
        self.G = {k: set(v) for k, v in json.load(open(OUT / "groups.json")).items()}
        self.ST = load_stats()
        hh = pd.read_parquet(EF.HSL_PARQUET).set_index("uid").loc[self.meta["uid"].to_numpy()]
        self.Xh = hh.drop(columns=["n_moves"]).to_numpy(np.float64) if "n_moves" in hh.columns else hh.to_numpy(np.float64); del hh
        self.uidpos = {u: i for i, u in enumerate(self.meta["uid"])}
        self.D0 = S.Data()
        simrows = np.array([self.uidpos[u] for u in self.D0.meta["uid"].to_numpy()])
        self.sim_pos = simrows                                   # row in games_meta of each sim row
        evp = set(self.D0.meta.loc[self.D0.meta.grp == "eval", "player_id"])
        assert not (evp & set().union(*self.G.values())), "EVAL player in a fit group"
        self.eval_players = evp

    def metrics(self, groups, cap):
        L = lda_from_stats(self.ST, groups, cap, kappa=self.kappa)
        rows = selected_rows(self.meta, self.rank, {k: list(v) for k, v in self.G.items()}, groups, cap)
        assert not (set(self.pid[rows]) & self.eval_players)
        M = dict(L); M.update(hsl_metric(self.Xh, self.pid, rows)); M["n_rows"] = len(rows); M["n_players"] = len(set(self.pid[rows]))
        return M

    def project_sim(self, M):
        Xh = self.Xh[self.sim_pos]
        return EF.project(self.D0.Z, Xh, M, True)


def stage1_score(ctx, M):
    Yl, Yh = ctx.project_sim(M)
    sc, r40, rk = [], [], []
    for pool in pools(ctx.D0):
        lib, q = pool["lib"], pool["q"]
        f64 = lambda Y, rr: [Y[r].astype(np.float64) for r in rr]  # noqa: E731
        dl = S3.dist_matrix(f64(Yl, lib), f64(Yl, q), 30); dh = S3.dist_matrix(f64(Yh, lib), f64(Yh, q), 15)
        for nm, sim in (("L", -dl), ("H", -dh), ("F", EF.zn(-dl) + EF.zn(-dh))):
            r = rank_stats(sim, pool["truth"])
            sc.append((nm, score_from_rank(r).mean(), (r <= 40).mean(), (r <= 5).mean()))
    df = pd.DataFrame(sc, columns=["m", "score", "R40", "R5"]).groupby("m").mean()
    return df


def cmd_stage1(a):
    ctx = Ctx()
    for kap, sc in [(k_, s_) for k_ in a.kappas for s_ in a.scales]:
        ctx.kappa = None if kap < 0 else kap
        a.kappa = ctx.kappa
        cap, tg, _ = FAM[sc]
        t0 = time.perf_counter(); M = ctx.metrics(tg, cap)
        df = stage1_score(ctx, M)
        print(f"{sc} kappa={a.kappa}: players {M['n_players']} rows {M['n_rows']}  L {df.loc['L','score']:.4f}  H {df.loc['H','score']:.4f}  F {df.loc['F','score']:.4f} R40(F) {df.loc['F','R40']:.3f}  R5(F) {df.loc['F','R5']:.3f}  {time.perf_counter() - t0:.0f}s", flush=True)


def cmd_selftest(a):
    """stats-based LDA == in-memory fit_metric on the same rows (X78 group, cap 24)."""
    ctx = Ctx()
    rows = selected_rows(ctx.meta, ctx.rank, {k: list(v) for k, v in ctx.G.items()}, ["X78"], 24)
    L = lda_from_stats(ctx.ST, ["X78"], 24, ridge=3e-2, mmax=30)
    rec = np.load(OUT / "recipe.npz")
    store = __import__("gogame.aggregate", fromlist=["L0Store"]).L0Store.from_table(__import__("gogame.gametable", fromlist=["load_table"]).load_table(ROOT / "data/cache/l0"))
    Zd = np.nan_to_num((store.dense[rows] - rec["mu"]) / rec["sd"], nan=0.0); Zs = (store.sparse[rows][:, rec["cols"]].toarray() - rec["smu"]) / rec["ssd"]
    X = np.hstack([Zd, Zs]).astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    gm, V, ev = S3.fit_metric((X - mu) / sd, ctx.pid[rows], ridge=3e-2, mmax=30)
    Ya = ((X - mu) / sd - gm) @ V; Yb = ((X - L["mu"]) / L["sd"] - L["gm"]) @ L["V"].astype(np.float64)
    # eigenvector sign ambiguity: compare absolute values of distances instead
    from scipy.spatial.distance import pdist
    ia = np.random.default_rng(0).choice(len(X), 400, replace=False)
    da, db = pdist(Ya[ia]), pdist(Yb[ia])
    print("rows", len(rows), "ev rel err", np.abs(ev[:30] - L["ev"][:30]).max() / ev[0], "mu err", np.abs(mu - L["mu"]).max(), "pairwise dist corr", np.corrcoef(da, db)[0, 1], "max rel", np.abs(da - db).max() / da.max())


def cmd_run(a):
    ctx = Ctx(a.kappa)
    cap, tg, xg = FAM[a.scale]
    t0 = time.perf_counter()
    D0 = ctx.D0
    tr_players = sorted(D0.eligible["train"])
    perm = np.random.default_rng(99).permutation(len(tr_players))
    half = {0: set(np.array(tr_players)[perm[: len(perm) // 2]]), 1: set(np.array(tr_players)[perm[len(perm) // 2:]])}
    assert half[0] == ctx.G["TA"] and half[1] == ctx.G["TB"], "half split differs from stats groups"
    Mfull = ctx.metrics(tg, cap)
    print(f"{a.scale}: test-time metric players {Mfull['n_players']} rows {Mfull['n_rows']} ({time.perf_counter() - t0:.0f}s)", flush=True)
    Yx = {}
    for h in (0, 1):
        other = ["TB" if h == 0 else "TA"] + xg                     # metric for pools from half h: other half + extras
        Mh = ctx.metrics(other, cap)
        assert not (set(ctx.pid[selected_rows(ctx.meta, ctx.rank, {k: list(v) for k, v in ctx.G.items()}, other, cap)]) & half[h])
        Yx[h] = ctx.project_sim(Mh); print(f"half-metric {h}: players {Mh['n_players']} rows {Mh['n_rows']} {time.perf_counter() - t0:.0f}s", flush=True)
        del Mh
    rng = np.random.default_rng(a.seed)
    Xs_, ys_ = [], []
    for dr in range(a.draws):
        st = ("a", "b")[dr % 2]; h = dr % 4 // 2
        D0.eligible["train"] = [p for p in tr_players if p in half[h]]
        pool = S.make_pool(D0, "train", rng, st)
        Ylda, Yhsl = Yx[h]          # Yx[h] = metric fitted WITHOUT the players of half h
        f = EF.feats3(D0.Z, D0.sel, pool, a.k, Ylda, Yhsl, True)
        Xs_.append(f["X"]); ys_.append((f["top"] == pool["truth"][:, None]).astype(np.int8).reshape(-1))
        if dr % 10 == 0:
            print(f"  train pool {dr}: recall@{f['k']} {(f['top'] == pool['truth'][:, None]).any(1).mean():.3f}  {time.perf_counter() - t0:.0f}s", flush=True)
    D0.eligible["train"] = tr_players
    del Yx
    X, y = np.concatenate(Xs_), np.concatenate(ys_); del Xs_
    kk = f["k"]
    bst = lgb.train(prr.lgb_params(a.seed, 4), lgb.Dataset(X, y, group=np.full(len(y) // kk, kk), free_raw_data=False), num_boost_round=a.rounds)
    print(f"booster trained {time.perf_counter() - t0:.0f}s, X {X.shape}", flush=True); del X
    Yl, Yh = ctx.project_sim(Mfull)
    res = {}
    for pool in pools(D0):
        f = EF.feats3(D0.Z, D0.sel, pool, a.k, Yl, Yh, True)
        e = bst.predict(f["X"], num_threads=4).reshape(len(pool["q"]), -1)
        ts = np.take_along_axis(f["top"], np.argsort(-e, axis=1, kind="stable"), 1)
        hit = ts == pool["truth"][:, None]
        rank = np.where(hit.any(1), hit.argmax(1) + 1, 999)
        res[pool["tag"]] = dict(score=score_from_rank(rank).tolist(), rank=rank.tolist(), n=pool["n"].tolist())
        print(pool["tag"], round(float(np.mean(res[pool["tag"]]["score"])), 3), flush=True)
    res["_info"] = dict(scale=a.scale, kappa=a.kappa, seed=a.seed, test_players=Mfull["n_players"], test_rows=Mfull["n_rows"], draws=a.draws, rounds=a.rounds, k=a.k)
    (OUT / f"res_{a.out}.json").write_text(json.dumps(res))
    print("MEAN", np.mean([np.mean(v["score"]) for k, v in res.items() if k != "_info"]).round(4), res["_info"], flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stage1"); p.add_argument("--scales", nargs="+", default=list(FAM)); p.add_argument("--kappas", type=float, nargs="+", default=[-1])
    sub.add_parser("selftest")
    p = sub.add_parser("run"); p.add_argument("--scale", required=True); p.add_argument("--kappa", type=float, default=None); p.add_argument("--seed", type=int, default=8); p.add_argument("--draws", type=int, default=60)
    p.add_argument("--rounds", type=int, default=600); p.add_argument("--k", type=int, default=40); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"stage1": cmd_stage1, "selftest": cmd_selftest, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
