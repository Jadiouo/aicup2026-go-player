"""E40 production port (v7f): LHx reranker (e39_final/e38_final path) + DEEP embedding channel (episodic-contrastive MLP on [Z, HSL60], e40_deep_mlp.py --data fin).
Booster training pools: identical to v7e (4,565 non-H players, <=24 games/colour, 60 pools x 600 rounds, K=40, seed 8), metrics cross-fitted as in e39_final; the deep encoder for pools of half h is the one
trained WITHOUT half h (emb_fin_h{1-h}); test time uses the encoder trained on all 7,367 non-H players (emb_fin_full).  Same feature code as e40_rerank.feats3 (deep block appended).  CPU only, no H.
  train    e40_final.py train --name v7f [--wdeep 1.0]    -> .scratch/e40/final/NAME.{txt,json}  (+ copies the v7e metric npz)
  predict  e40_final.py predict --model v7f --out FILE
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
import e38_final as EF
import e39_final as E39
import e39_run as ER
F, S, S3 = EF.F, EF.S, EF.S3
import numpy as np, pandas as pd, lightgbm as lgb
from gogame import player_rerank as prr
E4 = ROOT / ".scratch" / "e40"; FINO = ROOT / ".scratch" / "e41" / "final"
zn = EF.zn


def load_x120_raw(uids):
    """raw (unstandardised) W120 HSL matrix (len(uids), 104) float64 from E36's X120_full.npy for the given game uids, plus the 104 column names (as in E29 aux names[:104] = hc)."""
    from scripts.experiments.e36.common import E29B
    a = np.load(E29B / "aux.npz"); names = a["names"].astype(str); uid = a["uid"].astype(str); pos = pd.Series(np.arange(len(uid)), index=uid)
    X = np.load(ROOT / ".scratch/e36/bundle/X120_full.npy", mmap_mode="r")
    hc = [i for i, n in enumerate(names) if not (n.startswith("l0_") or n.startswith("se_") or n == "is_white")]
    idx = pos.reindex(uids).to_numpy()
    out = np.full((len(uids), len(hc)), np.nan)
    ok = ~np.isnan(idx)
    ii = idx[ok].astype(int); order = np.argsort(ii)
    tmp = np.asarray(X[:, hc])[ii].astype(np.float64)            # fancy index on the (361k x 104 | rows are the full 712,945 non-H table) columns subset
    out[np.flatnonzero(ok)] = tmp
    return out, [names[i] for i in hc]


def feats3d(Z, sel, pool, k, Ylda, Yhsl, use_h, emb, wdeep):
    lib, q = pool["lib"], pool["q"]
    Zl, Zq = [Z[r] for r in lib], [Z[r] for r in q]
    nq = len(q); f64 = lambda Y, rr: [Y[r].astype(np.float64) for r in rr]  # noqa: E731
    d = {f"l{m}": S3.dist_matrix(f64(Ylda, lib), f64(Ylda, q), m) for m in (10, 30, 80)}
    if use_h:
        for m in (5, 15, 40): d[f"h{m}"] = S3.dist_matrix(f64(Yhsl, lib), f64(Yhsl, q), m)
    Cd = np.stack([emb[r].mean(0) for r in lib]); Qd = np.stack([emb[r].mean(0) for r in q]); cosd = prr.unit(Qd) @ prr.unit(Cd).T
    Fm = zn(-d["l30"]) + (zn(-d["h15"]) if use_h else 0.0) + wdeep * zn(cosd)
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
    X = np.concatenate([scal, np.stack(ext, 2), diff, prod], axis=2).astype(np.float32); assert np.isfinite(X).all()
    return dict(X=X.reshape(nq * kk, -1), top=top, F=Fm, k=kk)


def stage_train(a):
    t0 = time.perf_counter()
    P, pm = F.load_pm()
    D = EF.build_final_space(pm)
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet"); Hset = set(S3.R.sets()["H"])
    # row uids of D (same deterministic selection as EF.build_final_space)
    _tot = meta.groupby("player_id").size(); _keep = set(_tot.index[_tot >= 30]) - Hset; _rng = np.random.default_rng(20261111)
    _m = meta.assign(i=np.arange(len(meta))); _m = _m[_m.player_id.isin(_keep)]; _r = []
    for (_p, _c), _g in _m.groupby(["player_id", "color"]):
        _idx = _g["i"].to_numpy(); _r.append(_idx[_rng.permutation(len(_idx))[:F.CAP]])
    _r = np.sort(np.concatenate(_r)); D.uid = meta["uid"].to_numpy()[_r]; assert (meta["player_id"].to_numpy()[_r] == D.pid).all()
    zz = np.load(ROOT / ".scratch/e39/final_stats.npz"); ST = {k: zz[k] for k in zz.files}
    G = {k: set(v) for k, v in json.load(open(ROOT / ".scratch/e39/final_groups.json")).items()}
    TA, TB, X = E39.groups_final(D, meta, Hset); assert G["TA"] == TA and G["TB"] == TB
    pid_all = meta["player_id"].to_numpy()
    if a.w == 120:
        Xh_all, cols = load_x120_raw(meta["uid"].to_numpy()); print("W120 matrix", Xh_all.shape, "NaN rows (H / uncovered)", int(np.isnan(Xh_all[:, 0]).sum()), flush=True)
        D.Xh = load_x120_raw(D.uid)[0]; assert not np.isnan(D.Xh).any(); D.hsl_cols = cols
    else:
        hh = pd.read_parquet(EF.HSL_PARQUET).set_index("uid").loc[meta["uid"].to_numpy()]; cols = [c for c in hh.columns if c != "n_moves"]
        Xh_all = hh[cols].to_numpy(np.float64); del hh

    def metric(groups):
        L = ER.lda_from_stats(ST, groups, "all", kappa=300.0); Mh, nrows = E39.hsl_fit(Xh_all, pid_all, set().union(*[G[x] for x in groups])); L.update(Mh); return L
    tr_players = sorted(D.eligible["train"]); half = {0: TA, 1: TB}
    EM = {h: np.load(Path(a.emb_dir) / f"emb_{a.embname}_h{h}_cap.npy").astype(np.float32) for h in (0, 1)}
    assert len(EM[0]) == len(D.Z), (len(EM[0]), len(D.Z))
    Yx = {}
    for h in (0, 1):
        Mh = metric(["TB" if h == 0 else "TA", "X"]); Yx[h] = EF.project(D.Z, D.Xh, Mh, True); del Mh
    rng = np.random.default_rng(a.seed); Xs_, ys_ = [], []
    for dr in range(a.draws):
        st = ("a", "b")[dr % 2]; h = dr % 4 // 2
        D.eligible["train"] = [p for p in tr_players if p in half[h]]
        pool = S.make_pool(D, "train", rng, st)
        Ylda, Yhsl = Yx[h]
        f = feats3d(D.Z, D.sel, pool, a.k, Ylda, Yhsl, True, EM[1 - h], a.wdeep)       # encoder trained on the OTHER half (+extras)
        Xs_.append(f["X"]); ys_.append((f["top"] == pool["truth"][:, None]).astype(np.int8).reshape(-1))
        if dr % 10 == 0: print(f"  train pool {dr}: recall@{f['k']} {(f['top'] == pool['truth'][:, None]).any(1).mean():.3f}  {time.perf_counter() - t0:.0f}s", flush=True)
    D.eligible["train"] = tr_players
    FINO.mkdir(parents=True, exist_ok=True)
    v7e = np.load(EF.FIN / "v7e.npz", allow_pickle=False); Mt = {x: v7e[x] for x in v7e.files if x != "hsl_cols"}; hcols_t = np.array(v7e["hsl_cols"])
    Mfull = metric(["TA", "TB", "X"])                       # test-time metric (L0 LDA from stats, kappa 300; HSL LDA on ALL games of the final players), as e39_final
    for kx in ("mu", "sd", "gm", "V"): assert np.allclose(Mfull[kx], Mt[kx], rtol=1e-5, atol=1e-6), f"L0 metric part {kx} differs from v7e"
    if a.w == 120: Mt.update({kx: Mfull[kx] for kx in ("hmu", "hsd", "hgm", "hV")}); hcols_t = np.array(cols)
    else:
        for kx in ("hmu", "hsd", "hgm", "hV"): assert np.allclose(Mfull[kx], Mt[kx], rtol=1e-5, atol=1e-6), f"W60 port check: {kx} differs from v7e"
        print("W60 port check: test-time metric == v7e", flush=True)
    np.savez(FINO / f"{a.name}_metric.npz", hsl_cols=hcols_t, **Mt)
    import gc
    Xm, y = np.concatenate(Xs_), np.concatenate(ys_); kk = f["k"]; del Xs_, ys_, D, Xh_all, Yx, EM, ST, pool, f; gc.collect()
    bst = lgb.train(prr.lgb_params(a.seed, 4), lgb.Dataset(Xm, y, group=np.full(len(y) // kk, kk), free_raw_data=False), num_boost_round=a.rounds)
    FINO.mkdir(parents=True, exist_ok=True); bst.save_model(str(FINO / f"{a.name}.txt"))

    (FINO / f"{a.name}.json").write_text(json.dumps(dict(k=kk, wdeep=a.wdeep, draws=a.draws, rounds=a.rounds, seed=a.seed, n_feat=int(Xm.shape[1]), metric_model=a.name + "_metric", w=a.w, seconds=round(time.perf_counter() - t0))))
    print("saved", a.name, Xm.shape, f"{time.perf_counter() - t0:.0f}s", flush=True)


def stage_predict(a):
    import hashlib
    P, pm = F.load_pm()
    from p1111_decode import load_questions_any
    info = json.loads((FINO / f"{a.model}.json").read_text()); k = info["k"]
    bst = lgb.Booster(model_file=str(FINO / f"{a.model}.txt"))
    Mz = np.load(FINO / f"{info['metric_model']}.npz", allow_pickle=False); M = {x: Mz[x] for x in Mz.files if x != "hsl_cols"}
    data = ROOT / ".scratch/p1111/t2"; hp = pd.read_parquet(a.hsl_parquet).set_index("uid")
    qids, tops = [], []
    for sp in ("pub", "priv"):
        pq = load_questions_any(P, data / P.FILES[f"player_{sp}"], sp); cands = P.load_candidates(data / P.FILES[f"cand_{sp}"], sp); pool = sorted(cands)
        z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{sp}.npz", allow_pickle=False); uids, Zt = z["uids"].astype(str), z["Z"]
        assert (np.load(E4 / "fin" / f"test_{sp}_uids.npy") == uids).all()
        emb = np.load(Path(a.emb_dir) / f"emb_{a.emb}_test_{sp}.npy").astype(np.float32)
        rowof = {u: i for i, u in enumerate(uids)}; Xh = hp.reindex(uids)[list(Mz["hsl_cols"])].to_numpy(np.float64)
        Yl, Yh = EF.project(Zt, Xh, M, True); rows = lambda us: np.array([rowof[u] for u in us])  # noqa: E731
        pl = dict(lib=[rows([g.uid for g in cands[p]]) for p in pool], q=[rows([g.uid for g in q.games]) for q in pq])
        f = feats3d(Zt, pm.s["sel"], pl, k, Yl, Yh, True, emb, info["wdeep"])
        e = bst.predict(f["X"], num_threads=4).reshape(len(pq), -1); t5 = S.E30.top5_orig(f["top"], e)
        qids += [q.qid for q in pq]; tops.append(np.array(pool, dtype=object)[t5])
        np.savez_compressed(FINO / f"real_{a.model}_{sp}.npz", e=e, top=f["top"], F=f["F"], pool=np.array(pool), qids=np.array([q.qid for q in pq]))
        print(f"[{sp}] scored {len(pq)} questions", flush=True)
    dst = Path(a.out); dst.parent.mkdir(parents=True, exist_ok=True); P.write_player_submission(qids, np.vstack(tops), dst)
    import check_submission
    rc = check_submission.main([str(dst), "--task", "player", "--data-dir", str(data)])
    print("check_rc", rc, "sha256_12", hashlib.sha256(dst.read_bytes()).hexdigest()[:12]); return rc


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("--name", required=True); p.add_argument("--wdeep", type=float, default=1.0); p.add_argument("--draws", type=int, default=60); p.add_argument("--w", type=int, default=120); p.add_argument("--embname", default="fin"); p.add_argument("--emb-dir", default=str(E4))
    p.add_argument("--rounds", type=int, default=600); p.add_argument("--k", type=int, default=40); p.add_argument("--seed", type=int, default=8)
    p = sub.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--emb", default="fin_full"); p.add_argument("--emb-dir", default=str(E4)); p.add_argument("--hsl-parquet", default=str(ROOT / "data/test_scan_p1111/hsl_test_games.parquet"))
    a = ap.parse_args(); return {"train": stage_train, "predict": stage_predict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
