"""E38 production port of the LHx Player reranker (v7d): learned-metric shortlist + cross-fitted metric features.  CPU only, never loads H labels/ledger.

Pipeline (same code path for the sim check and the real test2):
  metrics  : L0 LDA (all 2067 Z columns, both colours, 150 dims, ridge 3e-2) and HSL LDA (104 HSL columns, 40 dims, ridge 1e-2), fitted on NON-H train players only.
             Training pools use metrics cross-fitted on the OTHER half of the train players (LHx); test time uses the metric fitted on all of them.
  shortlist: top-K (40) by z(-D_lda30) + z(-D_hsl15)  (Mahalanobis^2 between bag means, divided by 1/nq+1/nc).
  reranker : LightGBM lambdarank (60 pools x 600 rounds, E05 params) on the E05 feature block + cos(Z) + LDA/HSL distance blocks.  Independent per-question ranking (decoder A).
Commands
  sim      e38_final.py sim --out NAME [--draws 60 --rounds 600 --seed 8]    port check on the 16 p1111 sim pools (train players only) -> .scratch/e38/res_NAME.json
  train    e38_final.py train --name v7d [--draws 60 --rounds 600 --seed 8]  final Z space (data/models_final standardiser), non-H players -> .scratch/e38/final/NAME.{txt,npz,json}
  predict  e38_final.py predict --model NAME --data-dir .scratch/p1111/t2 --out FILE --hsl-parquet PATH [--z-cache-dir .scratch/p1111/z_t2]
Then scripts/oneoff/p1111_assemble.py --new2 FILE --out data/submissions/...csv
"""
import argparse, json, os, sys, time
os.environ.setdefault("GOGAME_COLOR_SEMANTICS", "actual")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
import p1111_final as F          # sets sys.path; imports p1111_sim (S) etc.
S = F.S
S3 = S.S3
import numpy as np
import pandas as pd
import lightgbm as lgb
from gogame import player_rerank as prr

OUT = ROOT / ".scratch" / "e38"
FIN = OUT / "final"
K_DEFAULT, M_LDA, M_HSL_D = 40, 30, 15
HSL_PARQUET = F.HSL_PARQUET


def zn(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


# ----------------------------------------------------------------------------- features
def cand_feats(Zl, Zq, sel, top, Fm, k):
    """E05 feature block for an externally given shortlist `top` (nq,k) ordered by the similarity matrix Fm (nq,P); plus cos(Z) of the shortlist."""
    nq = len(Zq)
    Craw = np.stack([z.mean(0) for z in Zl]); Qraw = np.stack([z.mean(0) for z in Zq]); ns = np.array([len(z) for z in Zq])
    cosz = prr.unit(Qraw) @ prr.unit(Craw).T
    ct = np.take_along_axis(Fm, top, 1)
    mt, st = ct.mean(1, keepdims=True), ct.std(1, keepdims=True) + 1e-9
    ma, sa = Fm.mean(1, keepdims=True), Fm.std(1, keepdims=True) + 1e-9
    gap1 = ct[:, :1] - ct
    gapp = np.concatenate([np.zeros((nq, 1)), ct[:, :-1] - ct[:, 1:]], axis=1)
    rank = np.broadcast_to(np.arange(1, k + 1), (nq, k))
    cosD = prr.unit(Qraw[:, :prr.N_DENSE]) @ prr.unit(Craw[:, :prr.N_DENSE]).T
    sp = sel[prr.N_DENSE:]
    cosS = prr.unit(Qraw[:, sp]) @ prr.unit(Craw[:, sp]).T
    cD, cS, cZ = (np.take_along_axis(x, top, 1) for x in (cosD, cosS, cosz))
    cZr = (cZ - cZ.mean(1, keepdims=True)) / (cZ.std(1, keepdims=True) + 1e-9)
    LIB = np.stack([prr.unit(z[:, sel]) for z in Zl])
    kmax = np.empty((nq, k)); kmean = np.empty((nq, k)); ktop3 = np.empty((nq, k))
    for i in range(nq):
        Gq = prr.unit(Zq[i][:, sel])
        s = np.einsum("nd,kmd->knm", Gq, LIB[top[i]], optimize=True)
        kmax[i] = s.max(2).mean(1); kmean[i] = s.mean((1, 2)); ktop3[i] = np.sort(s, 2)[:, :, -3:].mean((1, 2))
    q_sel, c_sel = Qraw[:, sel], Craw[:, sel]
    pair = c_sel[top]
    diff = np.abs(q_sel[:, None, :] - pair); prod = q_sel[:, None, :] * pair
    scal = np.stack([ct, rank, gap1, gapp, (ct - mt) / st, (ct - ma) / sa, np.broadcast_to(ns[:, None], (nq, k)), cD, cS, kmax, kmean, ktop3, cZ, cZr], axis=2)
    return scal, diff, prod


def feats3(Z, sel, pool, k, Ylda, Yhsl, use_h):
    lib, q = pool["lib"], pool["q"]
    Zl, Zq = [Z[r] for r in lib], [Z[r] for r in q]
    nq = len(q)
    f64 = lambda Y, rr: [Y[r].astype(np.float64) for r in rr]  # noqa: E731
    d = {f"l{m}": S3.dist_matrix(f64(Ylda, lib), f64(Ylda, q), m) for m in (10, 30, 80)}
    if use_h:
        for m in (5, 15, 40):
            d[f"h{m}"] = S3.dist_matrix(f64(Yhsl, lib), f64(Yhsl, q), m)
    Fm = zn(-d["l30"]) + (zn(-d["h15"]) if use_h else 0.0)
    kk = min(k, Fm.shape[1])
    top = np.argsort(-Fm, axis=1, kind="stable")[:, :kk]
    scal, diff, prod = cand_feats(Zl, Zq, sel, top, Fm, kk)
    ext = []
    for v in (np.take_along_axis(M, top, 1) for M in d.values()):
        c = v - v.mean(1, keepdims=True)
        ext += [v, c, c / (v.std(1, keepdims=True) + 1e-9)]
    X = np.concatenate([scal, np.stack(ext, 2), diff, prod], axis=2).astype(np.float32)
    assert np.isfinite(X).all()
    return dict(X=X.reshape(nq * kk, -1), top=top, F=Fm, k=kk)


# ----------------------------------------------------------------------------- metrics
def fit_metrics(Z, Xh, pid, rows, use_h):
    X = Z[rows].astype(np.float64); mu, sd = X.mean(0), X.std(0) + 1e-9
    X -= mu; X /= sd                                                  # in place: the full fit (200k rows) would otherwise need ~3 float64 copies
    gm, V, _ = S3.fit_metric(X, pid[rows], ridge=3e-2, mmax=150); del X
    M = dict(mu=mu, sd=sd, gm=gm, V=V.astype(np.float32))
    if use_h:
        H = Xh[rows]; hmu, hsd = H.mean(0), H.std(0) + 1e-9
        g2, V2, _ = S3.fit_metric((H - hmu) / hsd, pid[rows], ridge=1e-2, mmax=40)
        M.update(hmu=hmu, hsd=hsd, hgm=g2, hV=V2)
    return M


def project(Z, Xh, M, use_h):
    Yl = np.empty((len(Z), M["V"].shape[1]), np.float32)
    for s in range(0, len(Z), 20000):
        Yl[s:s + 20000] = ((Z[s:s + 20000].astype(np.float64) - M["mu"]) / M["sd"] - M["gm"]) @ M["V"]
    Yh = None
    if use_h:
        Yh = (((Xh - M["hmu"]) / M["hsd"] - M["hgm"]) @ M["hV"]).astype(np.float32)      # NaN rows stay NaN (dist_matrix drops them)
    return Yl, Yh


# ----------------------------------------------------------------------------- training
def train_model(D, a, log=print):
    """D: .Z (n,2067 f32), .Xh (n,104 f64 or None), .pid (n,), .sel, .fit_rows (rows of all train players), .rows {"train": {player: rows}}, .eligible {"train": [players]}."""
    t0 = time.perf_counter()
    use_h = D.Xh is not None
    tr_players = sorted(D.eligible["train"])
    perm = np.random.default_rng(99).permutation(len(tr_players))
    half = {0: set(np.array(tr_players)[perm[: len(perm) // 2]]), 1: set(np.array(tr_players)[perm[len(perm) // 2:]])}
    Yx = {}
    for h in (0, 1):
        rows = D.fit_rows[np.isin(D.pid[D.fit_rows], list(half[h]))]
        Yx[h] = project(D.Z, D.Xh, fit_metrics(D.Z, D.Xh, D.pid, rows, use_h), use_h)
        log(f"half-metric {h}: {len(rows)} rows {time.perf_counter() - t0:.0f}s")
    Mfull = fit_metrics(D.Z, D.Xh, D.pid, D.fit_rows, use_h)
    rng = np.random.default_rng(a.seed)
    Xs_, ys_ = [], []
    for dr in range(a.draws):
        st = ("a", "b")[dr % 2]; h = dr % 4 // 2          # pool from half h players, metric from the other half
        D.eligible["train"] = [p for p in tr_players if p in half[h]]
        pool = S.make_pool(D, "train", rng, st)
        Ylda, Yhsl = Yx[1 - h]
        f = feats3(D.Z, D.sel, pool, a.k, Ylda, Yhsl, use_h)
        Xs_.append(f["X"]); ys_.append((f["top"] == pool["truth"][:, None]).astype(np.int8).reshape(-1))
        if dr % 10 == 0:
            log(f"  train pool {dr}: recall@{f['k']} {(f['top'] == pool['truth'][:, None]).any(1).mean():.3f}  {time.perf_counter() - t0:.0f}s")
    D.eligible["train"] = tr_players
    del Yx
    X, y = np.concatenate(Xs_), np.concatenate(ys_); del Xs_
    kk = f["k"]
    bst = lgb.train(prr.lgb_params(a.seed, 4), lgb.Dataset(X, y, group=np.full(len(y) // kk, kk), free_raw_data=False), num_boost_round=a.rounds)
    log(f"booster trained {time.perf_counter() - t0:.0f}s, X {X.shape}")
    return bst, Mfull, dict(k=kk, hsl=use_h, draws=a.draws, rounds=a.rounds, seed=a.seed, n_players=len(tr_players), n_fit_rows=int(len(D.fit_rows)), n_feat=int(X.shape[1]))


class Obj:
    pass


def stage_sim(a):
    sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
    from e38_core import pools, score_from_rank
    D0 = S.Data()
    h = pd.read_parquet(HSL_PARQUET).set_index("uid").loc[D0.meta["uid"].to_numpy()]
    D = Obj(); D.Z, D.sel = D0.Z, D0.sel
    D.Xh = h.drop(columns=["n_moves"]).to_numpy(np.float64) if "n_moves" in h.columns else h.to_numpy(np.float64); del h
    D.pid = D0.meta["player_id"].to_numpy()
    D.fit_rows = np.flatnonzero((D0.meta.grp == "train").to_numpy())          # non-H train players only (sim prep never loaded H)
    D.rows, D.eligible = D0.rows, D0.eligible
    bst, M, info = train_model(D, a)
    Yl, Yh = project(D.Z, D.Xh, M, True)
    res = {}
    for pool in pools(D0):
        f = feats3(D.Z, D.sel, pool, a.k, Yl, Yh, True)
        e = bst.predict(f["X"], num_threads=4).reshape(len(pool["q"]), -1)
        ts = np.take_along_axis(f["top"], np.argsort(-e, axis=1, kind="stable"), 1)
        hit = ts == pool["truth"][:, None]
        rank = np.where(hit.any(1), hit.argmax(1) + 1, 999)
        res[pool["tag"]] = dict(score=score_from_rank(rank).tolist(), rank=rank.tolist(), n=pool["n"].tolist())
        print(pool["tag"], round(float(np.mean(res[pool["tag"]]["score"])), 3), flush=True)
    (OUT / f"res_{a.out}.json").write_text(json.dumps(res))
    print("MEAN", np.mean([np.mean(v["score"]) for v in res.values()]).round(4), info)


def build_final_space(pm, min_total=30):
    """non-H players with >=30 games, <=24 games per colour, Z in the FINAL (data/models_final) standardiser, HSL raw columns from the E10 scan."""
    from gogame.aggregate import L0Store
    from gogame.gametable import load_table
    Hset = set(S3.R.sets()["H"])                       # player ids only, to EXCLUDE them (no labels / ledger are read)
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    tot = meta.groupby("player_id").size()
    keep = set(tot.index[(tot >= min_total)]) - Hset
    rng = np.random.default_rng(20261111)
    m = meta.assign(i=np.arange(len(meta)))
    m = m[m.player_id.isin(keep)]
    rows = []
    for (p, c), g in m.groupby(["player_id", "color"]):
        idx = g["i"].to_numpy(); rows.append(idx[rng.permutation(len(idx))[:F.CAP]])
    rows = np.sort(np.concatenate(rows))
    table = load_table(ROOT / "data/cache/l0"); assert list(table.uids) == meta["uid"].tolist()
    store = L0Store.from_table(table)
    Z = np.empty((len(rows), len(pm.s["mu"]) + len(pm.s["cols"])), np.float32)
    for s in range(0, len(rows), 20000):
        Z[s:s + 20000] = pm._z(store, rows[s:s + 20000])
    del store, table
    mm = meta.iloc[rows].reset_index(drop=True)
    assert not set(mm.player_id) & Hset
    h = pd.read_parquet(HSL_PARQUET).set_index("uid").loc[mm["uid"].to_numpy()]
    cols = [c for c in h.columns if c != "n_moves"]
    D = Obj(); D.Z, D.sel = Z, pm.s["sel"]; D.Xh = h[cols].to_numpy(np.float64); D.hsl_cols = cols
    D.pid = mm["player_id"].to_numpy(); D.fit_rows = np.arange(len(mm))
    D.rows = {"train": {p: np.flatnonzero(D.pid == p) for p in sorted(set(D.pid))}}
    D.eligible = {"train": [p for p, r in D.rows["train"].items() if len(r) >= F.MIN_G]}
    return D


def stage_train(a):
    FIN.mkdir(parents=True, exist_ok=True)
    P, pm = F.load_pm()
    t0 = time.perf_counter()
    D = build_final_space(pm)
    print(f"train rows {len(D.Z)} eligible players {len(D.eligible['train'])} {time.perf_counter() - t0:.0f}s", flush=True)
    bst, M, info = train_model(D, a)
    bst.save_model(str(FIN / f"{a.name}.txt"))
    np.savez(FIN / f"{a.name}.npz", hsl_cols=np.array(D.hsl_cols), **M)
    info.update(seconds=round(time.perf_counter() - t0), n_rows=int(len(D.Z)))
    (FIN / f"{a.name}.json").write_text(json.dumps(info))
    print("saved", a.name, info, flush=True)


def stage_predict(a):
    import hashlib
    P, pm = F.load_pm()
    from p1111_decode import load_questions_any
    info = json.loads((FIN / f"{a.model}.json").read_text()); k, use_h = info["k"], info["hsl"]
    bst = lgb.Booster(model_file=str(FIN / f"{a.model}.txt"))
    Mz = np.load(FIN / f"{a.model}.npz", allow_pickle=False); M = {x: Mz[x] for x in Mz.files if x != "hsl_cols"}
    data = Path(a.data_dir)
    hp = pd.read_parquet(a.hsl_parquet).set_index("uid") if use_h else None
    qids, tops = [], []
    for sp in ("pub", "priv"):
        pq = load_questions_any(P, data / P.FILES[f"player_{sp}"], sp)
        cands = P.load_candidates(data / P.FILES[f"cand_{sp}"], sp); pool = sorted(cands)
        z = np.load(Path(a.z_cache_dir) / f"Z_{sp}.npz", allow_pickle=False); uids, Zt = z["uids"].astype(str), z["Z"]
        rowof = {u: i for i, u in enumerate(uids)}
        Xh = None
        if use_h:
            hh = hp.reindex(uids)
            Xh = hh[list(Mz["hsl_cols"])].to_numpy(np.float64)
            print(f"[{sp}] HSL rows missing: {int(np.isnan(Xh[:, 0]).sum())} of {len(uids)}", flush=True)
        Yl, Yh = project(Zt, Xh, M, use_h)
        rows = lambda us: np.array([rowof[u] for u in us])   # noqa: E731
        pl = dict(lib=[rows([g.uid for g in cands[p]]) for p in pool], q=[rows([g.uid for g in q.games]) for q in pq])
        f = feats3(Zt, pm.s["sel"], pl, k, Yl, Yh, use_h)
        e = bst.predict(f["X"], num_threads=4).reshape(len(pq), -1)
        t5 = S.E30.top5_orig(f["top"], e)
        qids += [q.qid for q in pq]; tops.append(np.array(pool, dtype=object)[t5])
        print(f"[{sp}] scored {len(pq)} questions, pool {len(pool)}", flush=True)
    dst = Path(a.out); dst.parent.mkdir(parents=True, exist_ok=True)
    P.write_player_submission(qids, np.vstack(tops), dst)
    import check_submission
    rc = check_submission.main([str(dst), "--task", "player", "--data-dir", str(data)])
    print("check_rc", rc, "sha256_12", hashlib.sha256(dst.read_bytes()).hexdigest()[:12])
    return rc


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("sim", "train"):
        p = sub.add_parser(nm); p.add_argument("--draws", type=int, default=60); p.add_argument("--rounds", type=int, default=600)
        p.add_argument("--k", type=int, default=K_DEFAULT); p.add_argument("--seed", type=int, default=8)
        p.add_argument("--out" if nm == "sim" else "--name", required=True)
    p = sub.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--hsl-parquet"); p.add_argument("--z-cache-dir", default=str(ROOT / ".scratch/p1111/z_t2"))
    a = ap.parse_args()
    return {"sim": stage_sim, "train": stage_train, "predict": stage_predict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
