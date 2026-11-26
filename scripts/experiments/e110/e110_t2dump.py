"""E110: copy of experiments/E90_player_csls_recheck/e90_t2dump.py with the model switched from v7p's k7p_pe to v7q's final booster v7q_S (pe86 + S channels).
Per-query dump on the REAL test2 candidate libraries with pseudo queries (rng 40, 6/4 split, N=1-4, 400 queries, 20 reps, Public/Private separately), structures a / c / b exactly as in e90_t2dump.
Changes against e90_t2dump.py (everything else is the same code, same rng order):
 1. model files come from .scratch/e90/final/make/models (v7q_S.*), channels = [.scratch/e86/r (pe86), .scratch/e90/r_h (S)]; loading follows scripts/experiments/e90/e90_final.py::cmd_t2rr (`load`).
 2. features through e110_feats2.feats (verbatim copy of e90_feats2.feats; Ype is a LIST of two channels; test vectors from <dir>/pe_test_<sp>.npy).
 3. ROOT = parents[3] (this file lives in scripts/experiments/e110); output .scratch/e110/dumps/t2{,c,b}_{pub,priv}.npz (np.savez_compressed refuses nothing, but the dir is new).
 4. struct a: REPRO check at the end against .scratch/e90/final/make/t2rr_v7q_S_perq.parquet column `new` (v7q per-question scores) and .scratch/e90/final/result.json t2rr (no CSLS).
Usage: systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 env PYTHONPATH=src:scripts:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/experiments/e110/e110_t2dump.py --struct a|b|c"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(HERE)]
import numpy as np, pandas as pd, lightgbm as lgb
import e38_final as EF
import e86_final as FN
import e110_feats2 as C2
OUT = ROOT / ".scratch" / "e110" / "dumps"; OUT.mkdir(parents=True, exist_ok=True)
MODEL, REPS, NQ = "v7q_S", 20, 400
MDIR = ROOT / ".scratch/e90/final/make/models"; PES = [ROOT / ".scratch/e86/r", ROOT / ".scratch/e90/r_h"]
STRUCT = sys.argv[sys.argv.index("--struct") + 1] if "--struct" in sys.argv else "a"; TAG = "" if STRUCT == "a" else STRUCT
FN.CFG.update(old=0, cols="at,mean,midat,midmean,std", extras=0)
P, pm = EF.F.load_pm(); sel = pm.s["sel"]
hp2 = pd.read_parquet(ROOT / "data/test_scan_p1111/hsl120_test_games.parquet").set_index("uid")
info = json.loads((MDIR / f"{MODEL}.json").read_text()); Mz = np.load(MDIR / f"{info['metric_model']}.npz", allow_pickle=False); opt = info.get("opt", {})
assert opt.get("pe") and info.get("kata") and opt.get("late") and opt.get("tplda") and opt.get("blk") and not opt.get("ft"), opt
M = {x: Mz[x] for x in Mz.files if x != "hsl_cols"}; cols = list(Mz["hsl_cols"])
bst = lgb.Booster(model_file=str(MDIR / f"{MODEL}.txt"))
Pk = dict(np.load(MDIR / f"{MODEL}_trk.npz")); Pu = dict(np.load(MDIR / f"{MODEL}_late.npz")); Ph = FN.load_half(MDIR / f"{MODEL}_trkhalf.npz"); Pb = FN.load_blk(MDIR / f"{MODEL}_blk.npz")
FN.CFG["power"] = info.get("power", 0)
GVt = FN.load_gv(test=True); GVu = FN.load_gv(test=True, late=True); GVb = FN.load_gv(test=True, blk=True)
rng = np.random.default_rng(40); chk = []
for sp, s in (("public", "pub"), ("private", "priv")):
    z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); uids = z["uids"].astype(str); Zt = z["Z"]; pos = {u: i for i, u in enumerate(uids)}
    embE = np.load(ROOT / f".scratch/e41/emb_finens_full_test_{s}.npy").astype(np.float32)
    Yl, Yh = EF.project(Zt, hp2.reindex(uids)[cols].to_numpy(np.float64), M, True)
    c = pd.read_csv(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", dtype=str)
    gid = c.game_id.str.replace("pub2_", "pub_").str.replace("priv2_", "priv_"); row = np.array([pos[f"cand_{s}:{g}"] for g in gid]); col = (c.color == "W").to_numpy().astype(int)
    Zr, Yl, Yh, embE = Zt[row], Yl[row], Yh[row], embE[row]
    Y = FN.apply_trk(Pk, GVt[s])[row]; Yu = FN.apply_trk(Pu, GVu[s])[row]
    hy = FN.half_Y(Ph, GVt[s]); Yth, Pth = [y_[row] for y_ in hy[0]], hy[1]
    Yb = [y_[row] for y_ in FN.apply_blk(Pb, GVb[s])]
    Ype = []
    for d in PES:
        Yp = np.load(d / f"pe_test_{s}.npy").astype(np.float32); assert Yp.shape == (len(uids), 128) and np.isfinite(Yp).all(); Ype.append(Yp[row])
    pl = sorted(c.player_id.unique()); Pn = len(pl); byp = {p: np.flatnonzero((c.player_id == p).to_numpy()) for p in pl}; print(sp, "players", Pn, "games", len(row), flush=True)
    D = dict(rep=[], n=[], truth=[], rank=[], rankF=[], top=[], e=[], F=[], qrows=[], lib=[])
    for rep in range(REPS):
        lib, qp = [], []
        for p in pl: r = byp[p][rng.permutation(10)]; lib.append(r[:6]); qp.append(r[6:])
        qs, truth, ns = [], [], []
        if STRUCT == "a":
            for t in rng.permutation(Pn)[:NQ]: n = int(rng.choice([1, 2, 3, 4])); qs.append(qp[t][:n]); truth.append(t); ns.append(n)
        else:
            owners = rng.choice(Pn, NQ, replace=True) if STRUCT == "c" else rng.choice(rng.permutation(Pn)[:300], NQ, replace=True)
            for t in owners: n = int(rng.choice([1, 2, 3, 4])); qs.append(qp[t][rng.permutation(4)[:n]]); truth.append(int(t)); ns.append(n)
        truth = np.array(truth); pool = dict(lib=lib, q=qs)
        f = C2.feats(Zr, sel, pool, 40, Yl, Yh, True, embE, info["wdeep"], Y, info.get("wk", 1.0), opt, col, Yu, Yth, Pth, Yb, None, Ype)
        e = bst.predict(f["X"], num_threads=2).reshape(len(qs), -1); order = np.argsort(-e, axis=1, kind="stable"); ts = np.take_along_axis(f["top"], order, 1)
        hit = ts == truth[:, None]; rank = np.where(hit.any(1), hit.argmax(1) + 1, 999)
        tF = f["F"][np.arange(len(truth)), truth]; rF = 1 + (f["F"] > tF[:, None]).sum(1)
        sc = np.where(rank <= 5, np.exp(-(rank - 1.0)), 0.0); chk += [(sp, rep, ns[q], sc[q]) for q in range(len(qs))]
        qr = np.full((len(qs), 4), -1, np.int32)
        for i, r in enumerate(qs): qr[i, :len(r)] = r
        D["rep"].append(np.full(len(qs), rep)); D["n"].append(np.array(ns)); D["truth"].append(truth); D["rank"].append(rank); D["rankF"].append(rF)
        D["top"].append(f["top"].astype(np.int16)); D["e"].append(e.astype(np.float32)); D["F"].append(f["F"].astype(np.float32)); D["qrows"].append(qr); D["lib"].append(np.stack(lib).astype(np.int32))
        print(sp, rep, round(float(sc.mean()), 4), flush=True)
    np.savez_compressed(OUT / f"t2{TAG}_{s}.npz", **{k: (np.stack(v) if k == "lib" else np.concatenate(v)) for k, v in D.items()}, players=np.array(pl), cand_player=np.array([pl.index(p) for p in c.player_id]), cand_color=col)
if STRUCT != "a": print("means", pd.DataFrame(chk, columns=["sp", "rep", "n", "score"]).groupby("sp").score.mean().round(4).to_dict()); sys.exit(0)
df = pd.DataFrame(chk, columns=["sp", "rep", "n", "score"]); ref = pd.read_parquet(ROOT / ".scratch/e90/final/make/t2rr_v7q_S_perq.parquet")
assert len(df) == len(ref) and (df.sp.values == ref.sp.values).all() and (df.n.values == ref.n.values).all()
print("REPRO max|score - v7q t2rr perq new| =", float(np.abs(df.score.values - ref.new.values).max()), " means", df.groupby("sp").score.mean().round(6).to_dict(), flush=True)
rj = json.load(open(ROOT / ".scratch/e90/final/result.json"))["t2rr"]; print("result.json t2rr new: public", rj["public"]["new"], "private", rj["private"]["new"], flush=True)
