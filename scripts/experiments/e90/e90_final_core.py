"""E90 final (delivery) version, core: COPY of scripts/oneoff/e86_final.py stage_train / stage_predict (read-only original; its helper functions are imported, not copied).
Differences from the original (everything else is character-for-character the same maths):
 1. `Ype` is a LIST of position-encoder-style channels, read from the directories in PE_DIRS (each holds pe_final_h{0,1}.npy and pe_test_{pub,priv}.npy); features come from e90_feats2.feats
    (one channel -> bit-identical to e77_common.feats, so PE_DIRS=[.scratch/e86/r] reproduces E86's booster; two channels = pe86 + S).
 2. All outputs go to OUT_DIR (models/, real_<model>_<sp>.npz) instead of .scratch/e86/r (nothing of E86 is overwritten).
 3. Channel guards before training: every channel file has finite unit-length vectors on exactly the rows that carry KataGo vectors (= the booster's training rows), zeros elsewhere, and the rows
    of every channel are identical in coverage (so a channel cannot silently feed zero vectors to the booster).
Training pool of half h reads channel[h_{1-h}] (the network that did not see half h) - same line as E86 (`PEh[1-h]`); test time uses pe_test_* (full network).
"""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(HERE)]
import e86_final as FN                                   # read-only original: helpers + module state (CFG, GVX)
from e86_final import (EF, E39, ER, C, F, S, S3, CFG, GVX, E4, load_x120_raw, load_gv, load_ft, fit_trk, apply_trk, load_half, load_blk, apply_blk, half_Y)
import e90_feats2 as C2
import numpy as np, pandas as pd, lightgbm as lgb
from gogame import player_rerank as prr

FINO = None            # model directory (set by the caller)
PE_DIRS = []             # channel directories, in channel order (set by the caller)
OUT_DIR = [None]         # output directory (models/ and real_*.npz go here)


def feats3d(Z, sel, pool, k, Ylda, Yhsl, use_h, emb, wdeep, Ytrk=None, wk=1.0, opt=None, color=None, Ylate=None, Yth=None, Pth=None, Yblk=None, Yft=None, Ype=None):
    return C2.feats(Z, sel, pool, k, Ylda, Yhsl, use_h, emb, wdeep, Ytrk, wk, opt, color, Ylate, Yth, Pth, Yblk, Yft, Ype)


def load_pe_final(have):
    """[{0: (206536,128) float32, 1: ...}, ...] per channel, with the coverage guards (3)."""
    chans = []
    for d in PE_DIRS:
        PEh = {h_: np.load(Path(d) / f"pe_final_h{h_}.npy").astype(np.float32) for h_ in (0, 1)}
        for h_, A in PEh.items():
            assert A.shape == (len(have), 128) and np.isfinite(A).all(), (d, h_, A.shape)
            nz = np.abs(A).sum(1) > 0; assert (nz == have).all(), f"{d}/pe_final_h{h_}: covered rows != rows with KataGo vectors ({int(nz.sum())} vs {int(have.sum())})"
            assert np.abs(np.linalg.norm(A[nz], axis=1) - 1).max() < 1e-3, f"{d}/pe_final_h{h_}: not unit length"
        chans.append(PEh)
    return chans


def load_pe_test(sp, n):
    out = []
    for d in PE_DIRS:
        Y = np.load(Path(d) / f"pe_test_{sp}.npy").astype(np.float32)
        assert Y.shape == (n, 128) and np.isfinite(Y).all() and (np.abs(np.linalg.norm(Y, axis=1) - 1) < 1e-3).all(), f"bad test pe {d} {sp}"; out.append(Y)
    return out


def rss_gb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS"): return round(int(ln.split()[1]) / 1e6, 2)


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
    YBm = None; PBfull = None
    if a.blk:
        GVB = load_gv(len(D.Z), blk=True); _hb = np.isfinite(GVB[:, 0].astype(np.float32)); assert _hb[have].all(), "block vectors missing for rows that have early vectors"
        sl = lambda j: slice(1152 * j, 1152 * (j + 1)); nb = len(C.BLK_IDS)
        PBh = {m_: [fit_trk(GVB[:, sl(j)], _rows_of(half[m_]), D.pid, nc=60, extras=False) for j in range(nb)] for m_ in (0, 1)}; YBm = {m_: [apply_trk(PBh[m_][j], GVB[:, sl(j)]) for j in range(nb)] for m_ in (0, 1)}
        PBfull = [fit_trk(GVB[:, sl(j)], np.concatenate([D.rows["train"][p_] for p_ in sorted(D.eligible["train"])]), D.pid, nc=60, extras=False) for j in range(nb)]; GVB = None; print("block LDA fitted (halves + full)", flush=True)
    YFm = None; PFfull = None
    if a.ft:
        GVF = load_ft(len(D.Z)); _hf = np.isfinite(GVF[:, 0]); assert _hf[have].all(), "ft vectors missing for rows that have early vectors"
        PFh = {m_: fit_trk(GVF, _rows_of(half[m_]), D.pid, nc=60, extras=False) for m_ in (0, 1)}; YFm = {m_: apply_trk(PFh[m_], GVF) for m_ in (0, 1)}
        PFfull = fit_trk(GVF, np.concatenate([D.rows["train"][p_] for p_ in sorted(D.eligible["train"])]), D.pid, nc=60, extras=False); GVF = None; print("ft LDA fitted (halves + full)", flush=True)
    PEh = None
    if a.pe:
        assert a.kata and len(PE_DIRS) >= 1, "pe channels need the kata rows (coverage guard) and at least one channel directory"
        PEh = load_pe_final(have); print("pe channels", [str(d) for d in PE_DIRS], flush=True)
    rng = np.random.default_rng(a.seed); OPT = dict(color=a.color, rho=a.rho, late=a.late, wk2=a.wk2, tplda=a.tplda, blk=a.blk, ft=a.ft, wf=a.wf, pe=a.pe)
    for dr in range(a.draws):
        st = ("a", "b")[dr % 2]; h = dr % 4 // 2
        D.eligible["train"] = [p for p in tr_players if p in half[h]]
        pool = S.make_pool(D, "train", rng, st)
        Ylda, Yhsl = Yx[h]
        Yth_ = Pth_ = None
        if a.tplda: _ex = np.concatenate([D.rows["train"][p_] for p_ in sorted(half[h] - set(pool["pl"])) if p_ in D.rows["train"]]); Yth_ = [YTm[1 - h]]; Pth_ = [C.est_plda(YTm[1 - h][_ex], D.pid[_ex])]
        f = feats3d(D.Z, D.sel, pool, a.k, Ylda, Yhsl, True, EM[1 - h], a.wdeep, YTm[1 - h] if a.kata else None, a.wk, OPT, D.color, YUm[1 - h] if a.late else None, Yth_, Pth_, YBm[1 - h] if a.blk else None, YFm[1 - h] if a.ft else None, [c_[1 - h] for c_ in PEh] if a.pe else None)       # encoder trained on the OTHER half (+extras)
        if dr == 0: nrow = f["X"].shape[0]; Xm = np.empty((a.draws * nrow, f["X"].shape[1]), np.float32); y = np.empty(a.draws * nrow, np.int8)
        Xm[dr * nrow:(dr + 1) * nrow] = f["X"]; y[dr * nrow:(dr + 1) * nrow] = (f["top"] == pool["truth"][:, None]).astype(np.int8).reshape(-1)
        if dr % 10 == 0: print(f"  train pool {dr}: recall@{f['k']} {(f['top'] == pool['truth'][:, None]).any(1).mean():.3f}  {time.perf_counter() - t0:.0f}s", flush=True)
    D.eligible["train"] = tr_players
    import gc
    PEh = None; EM = None; Yx = None; YBm = None; YFm = None; gc.collect(); print("rss GB after pools (pe/emb/metric coords freed; MEMORY CHANGE vs E86: numerics untouched):", rss_gb(), flush=True)      # 2-channel run hit the 12G cap at the test-time metric fit
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
    if a.ft: np.savez(FINO / f"{a.name}_ft.npz", **PFfull)
    if a.blk: np.savez(FINO / f"{a.name}_blk.npz", **{f"{k_}{j}": v_ for j in range(len(PBfull)) for k_, v_ in PBfull[j].items()})
    import gc
    GV = None; YTm = None; YUm = None; GVX[0] = None; gc.collect()          # free trunk vectors before the booster (peak memory)
    kk = f["k"]; del D, Xh_all, Yx, EM, ST, pool, f; gc.collect()
    print("rss GB before booster:", rss_gb(), flush=True)
    bst = lgb.train(prr.lgb_params(a.seed, 4), lgb.Dataset(Xm, y, group=np.full(len(y) // kk, kk), free_raw_data=False), num_boost_round=a.rounds)
    FINO.mkdir(parents=True, exist_ok=True); bst.save_model(str(FINO / f"{a.name}.txt"))

    (FINO / f"{a.name}.json").write_text(json.dumps(dict(k=kk, wdeep=a.wdeep, kata=a.kata, wk=a.wk, opt=OPT, nc=CFG["nc"], old_gv=a.old_gv, cols=a.cols, extras=a.extras, power=a.power, ft=a.ft, wf=a.wf, pe=a.pe, draws=a.draws, rounds=a.rounds, seed=a.seed, n_feat=int(Xm.shape[1]), pe_dirs=[str(d) for d in PE_DIRS], metric_model=a.name + "_metric", w=a.w, seconds=round(time.perf_counter() - t0))))
    print("saved", a.name, Xm.shape, f"{time.perf_counter() - t0:.0f}s", flush=True)


def stage_predict(a):
    import hashlib
    global FINO
    if a.model_dir: FINO = Path(a.model_dir)
    P, pm = F.load_pm()
    from p1111_decode import load_questions_any
    info = json.loads((FINO / f"{a.model}.json").read_text()); k = info["k"]
    bst = lgb.Booster(model_file=str(FINO / f"{a.model}.txt"))
    Mz = np.load(FINO / f"{info['metric_model']}.npz", allow_pickle=False); M = {x: Mz[x] for x in Mz.files if x != "hsl_cols"}
    OPT = info.get("opt", {}); CFG["power"] = info.get("power", 0); GVt = load_gv(test=True) if info.get("kata") else None
    Ptrk = dict(np.load(FINO / f"{a.model}_trk.npz")) if info.get("kata") else None
    GVu = load_gv(test=True, late=True) if OPT.get("late") else None; Pu = dict(np.load(FINO / f"{a.model}_late.npz")) if OPT.get("late") else None
    Ph = load_half(FINO / f"{a.model}_trkhalf.npz") if OPT.get("tplda") else None
    GVf = load_ft(test=True) if OPT.get("ft") else None; Pf = dict(np.load(FINO / f"{a.model}_ft.npz")) if OPT.get("ft") else None
    GVb = load_gv(test=True, blk=True) if OPT.get("blk") else None; Pb = load_blk(FINO / f"{a.model}_blk.npz") if OPT.get("blk") else None
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
        Yb = None
        if OPT.get("blk"): assert np.isfinite(GVb[sp][:, 0].astype(np.float32)).all(); Yb = apply_blk(Pb, GVb[sp])
        Yf = None
        if OPT.get("ft"): assert np.isfinite(GVf[sp][:, 0]).all(), "missing test ft vectors"; Yf = apply_trk(Pf, GVf[sp])
        Ype = None
        if OPT.get("pe"): Ype = load_pe_test(sp, len(uids))
        f = feats3d(Zt, pm.s["sel"], pl, k, Yl, Yh, True, emb, info["wdeep"], Ytk, info.get("wk", 1.0), OPT, col, Yu, Yth, Pth, Yb, Yf, Ype)
        e = bst.predict(f["X"], num_threads=4).reshape(len(pq), -1); t5 = S.E30.top5_orig(f["top"], e)
        qids += [q.qid for q in pq]; tops.append(np.array(pool, dtype=object)[t5])
        np.savez_compressed(Path(OUT_DIR[0]) / f"real_{a.model}_{sp}.npz", e=e, top=f["top"], F=f["F"], pool=np.array(pool), qids=np.array([q.qid for q in pq]))
        print(f"[{sp}] scored {len(pq)} questions", flush=True)
    dst = Path(a.out); dst.parent.mkdir(parents=True, exist_ok=True); P.write_player_submission(qids, np.vstack(tops), dst)
    import check_submission
    rc = check_submission.main([str(dst), "--task", "player", "--data-dir", str(data)])
    print("check_rc", rc, "sha256_12", hashlib.sha256(dst.read_bytes()).hexdigest()[:12]); return rc

