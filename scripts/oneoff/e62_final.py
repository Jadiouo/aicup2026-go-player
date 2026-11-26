"""E62 production port (copy of e56_final.py; options color/rho/late/wk2/tplda via e62_common.feats, stored in the model json; all off = v7j bit for bit).  E56 docstring: E56 production port (copy of e52_final.py): trunk game vectors K=12 / column groups / EXTRAS (all non-H players outside the final training space) from .scratch/e56/final; --old-gv 1 = E52 kata3 (K=6, 768 dims, no extras) = must reproduce v7i bit for bit.
E52 docstring: E52 production port: e41_final.py (v7g recipe) + optional KataGo-trunk LDA channel (--kata 1; trunk game vectors from .scratch/e52/kata3, e52_kata_prep3.py / e52_kata_trunk2.py).  With --kata 0 it must reproduce v7g bit for bit.
Original docstring: E40 production port (v7f): LHx reranker (e39_final/e38_final path) + DEEP embedding channel (episodic-contrastive MLP on [Z, HSL60], e40_deep_mlp.py --data fin).
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
import e62_common as C
F, S, S3 = EF.F, EF.S, EF.S3
import numpy as np, pandas as pd, lightgbm as lgb
from gogame import player_rerank as prr
E4 = ROOT / ".scratch" / "e40"; FINO = ROOT / ".scratch" / "e62" / "final"; KD = ROOT / ".scratch" / "e52" / "kata3"; KD56 = ROOT / ".scratch" / "e56" / "final"; KD62 = ROOT / ".scratch" / "e62" / "final"; TEST_OFF = 10_000_000; EXTRA_OFF = 30_000_000
CG = {"at": (0, 384), "mean": (384, 768), "max": (768, 1152), "midat": (1152, 1536), "midmean": (1536, 1920), "std": (1920, 2304)}
CFG = dict(old=0, cols="at,mean", extras=0, nc=60)       # set from the CLI (trunk vector source, column groups, extras)
GVX = [None, None]       # (extras vectors, extras player ids) when --extras 1
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


def load_gv(nrows_train=None, test=False, late=False):
    """trunk game vectors: train rows -> (n_rows,d) fp16 NaN-filled, or test -> {sp: (7100,d)}.  CFG: old (E52 kata3, 768) / K=12 blocks with column groups; extras -> GVX."""
    import glob
    if late: d, cidx, KDX = 1920, np.arange(1920), KD62
    elif CFG["old"]: d, cidx, KDX = 768, np.arange(768), KD
    else: cidx = np.concatenate([np.arange(*CG[c_]) for c_ in CFG["cols"].split(",")]); d = len(cidx); KDX = KD56
    out = np.full((nrows_train, d), np.nan, np.float16) if not test else {sp: np.full((7100, d), np.nan, np.float16) for sp in ("pub", "priv")}
    gx, vx = [], []
    for f in sorted(glob.glob(str(KDX / "blk*_gv.npy"))):
        gid = np.load(f.replace("_gv", "_gid")); gv = np.load(f)[:, cidx]
        if test:
            m = (gid >= TEST_OFF) & (gid < EXTRA_OFF)
            for g_, v_ in zip(gid[m], gv[m]): out[("pub", "priv")[(int(g_) - TEST_OFF) // 100_000]][(int(g_) - TEST_OFF) % 100_000] = v_
        else:
            m = gid < TEST_OFF; out[gid[m]] = gv[m]
        mx = gid >= EXTRA_OFF
        if mx.any(): gx.append(gid[mx] - EXTRA_OFF); vx.append(gv[mx])
    if CFG["extras"] and not test and gx and not late:
        gm_ = pd.read_parquet(ROOT / "data/cache/games_meta.parquet", columns=["player_id"]); xi = np.concatenate(gx); GVX[0] = np.concatenate(vx); GVX[1] = gm_["player_id"].to_numpy()[xi]
        print("extras", GVX[0].shape, "players", len(set(GVX[1])), flush=True)
    return out


def fit_trk(GV, rows, pid, nc=None, extras=True):
    nc = nc or CFG["nc"]
    X = GV[rows].astype(np.float64); pp = pid[rows]
    if CFG["extras"] and extras:
        assert GVX[0] is not None and not (set(GVX[1]) & set(pp)), "extras missing or overlapping"
        X = np.concatenate([X, GVX[0].astype(np.float64)]); pp = np.concatenate([pp, GVX[1]])
    if CFG["old"]: mu, sd = X.mean(0), X.std(0) + 1e-6
    else:      # chunked (no full-size temporaries): the 1,900-dim variants otherwise exceed the memory cap while the booster pools are in RAM
        mu = X.mean(0); v = np.zeros(X.shape[1])
        for s_ in range(0, len(X), 20000): v += ((X[s_:s_ + 20000] - mu) ** 2).sum(0)
        sd = np.sqrt(v / len(X)) + 1e-6
    X -= mu; X /= sd
    if X.shape[1] > 800:
        import e40_lean; gm, V, _ = e40_lean.fit_metric_lean(X, pp, ridge=3e-2, mmax=nc)
    else: gm, V, _ = S3.fit_metric(X, pp, ridge=3e-2, mmax=nc)
    return dict(mu=mu, sd=sd, gm=gm, V=V)


def apply_trk(P, GV):
    Y = np.full((len(GV), P["V"].shape[1]), np.nan, np.float32); ok = np.flatnonzero(np.isfinite(GV[:, 0].astype(np.float32)))
    if CFG["old"]: Y[ok] = (((GV[ok].astype(np.float64) - P["mu"]) / P["sd"] - P["gm"]) @ P["V"]).astype(np.float32); return Y      # E52 code path, unchunked (bit-identical repro)
    for s_ in range(0, len(ok), 20000):
        o_ = ok[s_:s_ + 20000]; Y[o_] = (((GV[o_].astype(np.float64) - P["mu"]) / P["sd"] - P["gm"]) @ P["V"]).astype(np.float32)
    return Y


def feats3d(Z, sel, pool, k, Ylda, Yhsl, use_h, emb, wdeep, Ytrk=None, wk=1.0, opt=None, color=None, Ylate=None, Yth=None, Pth=None):
    return C.feats(Z, sel, pool, k, Ylda, Yhsl, use_h, emb, wdeep, Ytrk, wk, opt, color, Ylate, Yth, Pth)


def load_half(path):
    z = np.load(path); return {m_: (dict(mu=z[f"mu{m_}"], sd=z[f"sd{m_}"], gm=z[f"gm{m_}"], V=z[f"V{m_}"]), dict(gm=z[f"pgm{m_}"], W=z[f"pW{m_}"], Psi=z[f"pPsi{m_}"])) for m_ in (0, 1)}


def half_Y(Ph, GVsp):
    """half-metric trunk coords of a test GV matrix and the PLDA params of each half metric (for e62_common.feats Yth/Pth)."""
    return [apply_trk(Ph[m_][0], GVsp) for m_ in (0, 1)], [Ph[m_][1] for m_ in (0, 1)]


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
    _r = np.sort(np.concatenate(_r)); D.uid = meta["uid"].to_numpy()[_r]; D.color = meta["color"].to_numpy()[_r].astype(int); assert (meta["player_id"].to_numpy()[_r] == D.pid).all()
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
    GV = None
    if a.kata:
        GV = load_gv(len(D.Z)); have = np.isfinite(GV[:, 0].astype(np.float32)); print("kata rows with vectors", int(have.sum()), flush=True)
        D.rows["train"] = {p_: r_[have[r_]] for p_, r_ in D.rows["train"].items()}; D.eligible["train"] = [p_ for p_, r_ in D.rows["train"].items() if len(r_) >= 24]
        assert set(D.eligible["train"]) == (TA | TB), "eligible players differ after the trunk restriction"
    tr_players = sorted(D.eligible["train"]); half = {0: TA, 1: TB}
    EM = {h: np.load(Path(a.emb_dir) / f"emb_{a.embname}_h{h}_cap.npy").astype(np.float32) for h in (0, 1)}
    assert len(EM[0]) == len(D.Z), (len(EM[0]), len(D.Z))
    Yx = {}
    for h in (0, 1):
        Mh = metric(["TB" if h == 0 else "TA", "X"]); Yx[h] = EF.project(D.Z, D.Xh, Mh, True); del Mh
    YTm = None; YUm = None; Phalf = {}; PLMp = {}
    _rows_of = lambda ps: np.concatenate([D.rows["train"][p_] for p_ in sorted(ps)])
    if a.kata:
        Phalf = {m_: fit_trk(GV, _rows_of(half[m_]), D.pid) for m_ in (0, 1)}; YTm = {m_: apply_trk(Phalf[m_], GV) for m_ in (0, 1)}; print("kata LDA halves fitted", flush=True)      # YTm[m]: metric fitted on half m; pools of half h use YTm[1-h]
        PkFull = fit_trk(GV, np.concatenate([D.rows["train"][p_] for p_ in sorted(D.eligible["train"])]), D.pid); print("kata LDA full fitted (before the booster pools: memory)", flush=True)
        if a.tplda:
            for m_ in (0, 1): r_ = _rows_of(half[1 - m_]); PLMp[m_] = C.est_plda(YTm[m_][r_], D.pid[r_])
    if a.late:
        GVL = load_gv(len(D.Z), late=True); _hv = np.isfinite(GVL[:, 0].astype(np.float32)); assert _hv[have].all(), "late vectors missing for rows that have early vectors"
        PUh = {m_: fit_trk(GVL, _rows_of(half[m_]), D.pid, nc=60, extras=False) for m_ in (0, 1)}; YUm = {m_: apply_trk(PUh[m_], GVL) for m_ in (0, 1)}
        PUfull = fit_trk(GVL, np.concatenate([D.rows["train"][p_] for p_ in sorted(D.eligible["train"])]), D.pid, nc=60, extras=False); GVL = None; print("late LDA fitted (halves + full)", flush=True)
    rng = np.random.default_rng(a.seed); OPT = dict(color=a.color, rho=a.rho, late=a.late, wk2=a.wk2, tplda=a.tplda)
    for dr in range(a.draws):
        st = ("a", "b")[dr % 2]; h = dr % 4 // 2
        D.eligible["train"] = [p for p in tr_players if p in half[h]]
        pool = S.make_pool(D, "train", rng, st)
        Ylda, Yhsl = Yx[h]
        Yth_ = Pth_ = None
        if a.tplda: _ex = np.concatenate([D.rows["train"][p_] for p_ in sorted(half[h] - set(pool["pl"])) if p_ in D.rows["train"]]); Yth_ = [YTm[1 - h]]; Pth_ = [C.est_plda(YTm[1 - h][_ex], D.pid[_ex])]
        f = feats3d(D.Z, D.sel, pool, a.k, Ylda, Yhsl, True, EM[1 - h], a.wdeep, YTm[1 - h] if a.kata else None, a.wk, OPT, D.color, YUm[1 - h] if a.late else None, Yth_, Pth_)       # encoder trained on the OTHER half (+extras)
        if dr == 0: nrow = f["X"].shape[0]; Xm = np.empty((a.draws * nrow, f["X"].shape[1]), np.float32); y = np.empty(a.draws * nrow, np.int8)
        Xm[dr * nrow:(dr + 1) * nrow] = f["X"]; y[dr * nrow:(dr + 1) * nrow] = (f["top"] == pool["truth"][:, None]).astype(np.int8).reshape(-1)
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
    if a.kata:
        np.savez(FINO / f"{a.name}_trk.npz", **PkFull); print("kata LDA full fitted/saved", flush=True)
        if a.tplda: np.savez(FINO / f"{a.name}_trkhalf.npz", **{f"{k_}{m_}": v_ for m_ in (0, 1) for k_, v_ in {**Phalf[m_], **{"pgm": PLMp[m_]["gm"], "pW": PLMp[m_]["W"], "pPsi": PLMp[m_]["Psi"]}}.items()})
    if a.late: np.savez(FINO / f"{a.name}_late.npz", **PUfull)
    import gc
    GV = None; YTm = None; YUm = None; GVX[0] = None; gc.collect()          # free trunk vectors before the booster (peak memory)
    kk = f["k"]; del D, Xh_all, Yx, EM, ST, pool, f; gc.collect()
    bst = lgb.train(prr.lgb_params(a.seed, 4), lgb.Dataset(Xm, y, group=np.full(len(y) // kk, kk), free_raw_data=False), num_boost_round=a.rounds)
    FINO.mkdir(parents=True, exist_ok=True); bst.save_model(str(FINO / f"{a.name}.txt"))

    (FINO / f"{a.name}.json").write_text(json.dumps(dict(k=kk, wdeep=a.wdeep, kata=a.kata, wk=a.wk, opt=OPT, nc=CFG["nc"], old_gv=a.old_gv, cols=a.cols, extras=a.extras, draws=a.draws, rounds=a.rounds, seed=a.seed, n_feat=int(Xm.shape[1]), metric_model=a.name + "_metric", w=a.w, seconds=round(time.perf_counter() - t0))))
    print("saved", a.name, Xm.shape, f"{time.perf_counter() - t0:.0f}s", flush=True)


def stage_predict(a):
    import hashlib
    P, pm = F.load_pm()
    from p1111_decode import load_questions_any
    info = json.loads((FINO / f"{a.model}.json").read_text()); k = info["k"]
    bst = lgb.Booster(model_file=str(FINO / f"{a.model}.txt"))
    Mz = np.load(FINO / f"{info['metric_model']}.npz", allow_pickle=False); M = {x: Mz[x] for x in Mz.files if x != "hsl_cols"}
    OPT = info.get("opt", {}); GVt = load_gv(test=True) if info.get("kata") else None
    Ptrk = dict(np.load(FINO / f"{a.model}_trk.npz")) if info.get("kata") else None
    GVu = load_gv(test=True, late=True) if OPT.get("late") else None; Pu = dict(np.load(FINO / f"{a.model}_late.npz")) if OPT.get("late") else None
    Ph = load_half(FINO / f"{a.model}_trkhalf.npz") if OPT.get("tplda") else None
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
        Ytk = None
        if info.get("kata"):
            assert np.isfinite(GVt[sp][:, 0].astype(np.float32)).all(), "missing test trunk vectors"; Ytk = apply_trk(Ptrk, GVt[sp])
        col = np.zeros(len(uids), int)
        for p in pool:
            for g in cands[p]: col[rowof[g.uid]] = int(g.color)
        for q in pq:
            for g in q.games: col[rowof[g.uid]] = int(g.color)
        Yu = None; Yth = Pth = None
        if OPT.get("late"): assert np.isfinite(GVu[sp][:, 0].astype(np.float32)).all(); Yu = apply_trk(Pu, GVu[sp])
        if OPT.get("tplda"): Yth, Pth = half_Y(Ph, GVt[sp])
        f = feats3d(Zt, pm.s["sel"], pl, k, Yl, Yh, True, emb, info["wdeep"], Ytk, info.get("wk", 1.0), OPT, col, Yu, Yth, Pth)
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
    p.add_argument("--rounds", type=int, default=600); p.add_argument("--k", type=int, default=40); p.add_argument("--seed", type=int, default=8); p.add_argument("--kata", type=int, default=0); p.add_argument("--wk", type=float, default=1.0); p.add_argument("--old-gv", type=int, default=0); p.add_argument("--cols", default="at,mean"); p.add_argument("--extras", type=int, default=0)
    p.add_argument("--color", type=int, default=0); p.add_argument("--rho", type=float, default=0.5); p.add_argument("--late", type=int, default=0); p.add_argument("--wk2", type=float, default=1.0); p.add_argument("--tplda", type=int, default=0)
    p = sub.add_parser("predict"); p.add_argument("--old-gv", type=int, default=0); p.add_argument("--cols", default="at,mean"); p.add_argument("--tplda", type=int, default=0); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--emb", default="fin_full"); p.add_argument("--emb-dir", default=str(E4)); p.add_argument("--hsl-parquet", default=str(ROOT / "data/test_scan_p1111/hsl_test_games.parquet"))
    a = ap.parse_args(); CFG.update(old=a.old_gv, cols=a.cols, extras=getattr(a, "extras", 0), nc=100 if getattr(a, "tplda", 0) else 60); return {"train": stage_train, "predict": stage_predict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
