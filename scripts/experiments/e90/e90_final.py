"""E90 final (delivery version): put the S channel next to pe86 in the v7p production path and write the Player submission (v7q).
Corresponds step by step to how E86 made v7p (all originals are read-only; copies are named in e90_final_core.py / e90_feats2.py):
  train   (E86 .scratch/e86/ft.sh k7p_pe --pe 1)   final booster: 60 pools x 600 rounds, K=40, seed 8, 786 features (pe86) or 792 (pe86 + S)
  predict (.scratch/e86/pred.sh)                  test2 booster scores of Public / Private (pools never mixed)
  csls    (experiments/E86_player_all_positions/e86_make.py csls)   CSLS-qq k=10 alpha=.5 z, Public only among Public queries, Private among Private
  assemble(scripts/oneoff/p1111_assemble.py --old .scratch/e62/old_v6_800.csv)   original 800 rows kept from v7o/v7p, test2 800 rows replaced
  t2rr    (.scratch/e86/t2.sh = scripts/oneoff/e86_t2rr.py)  real test2 pseudo-queries, booster level (E86 PLAN criterion 4)

Sub-commands (all outputs under .scratch/e90/final/, nothing under data/submissions/ except `make`'s one new file):
  check                : NO S.  Re-trains the pe86-only booster with the copy and requires that every artefact equals E86's v7p artefacts bit for bit (model text, plain t2 csv sha 745a6050..., CSLS t2 csv,
                         assembled 1,600-row csv sha 4d031029... = data/submissions/player_v7p_全部己方手.csv), and that the pseudo-query copy reproduces E86's stored per-question scores (Public .5850 / Private .6036).
  dryrun               : whole path with a FAKE S (E80's pe_final_h*/pe_test_* linked as S: a real, different, unit-length channel).  Output names carry `dryrun`; the numbers are NOT an S result.
  make --s-dir DIR     : the real thing.  DIR holds pe_final_h0.npy, pe_final_h1.npy (206,536 x 128, rows = D rows), pe_test_pub.npy, pe_test_priv.npy (7,100 x 128) written by assemble_pe_h.py from finh_h0 / finh_h1 / finh_full.
                         Writes data/submissions/player_v7q_S通道.csv (refuses to overwrite) and .scratch/e90/final/result.json (refuses to overwrite).
  preflight --s-dir D  : only the light guards (roster / cross-fit lists, S files), no heavy job.
Heavy steps run under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`, 4 BLAS/OMP threads, no GPU, one at a time; they refuse to start when MemAvailable < --min-avail GB (default 16).
"""
import sys, os, json, time, argparse, hashlib, shutil, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(HERE)]
import numpy as np

FIN = ROOT / ".scratch/e90/final"
E86R = ROOT / ".scratch/e86/r"; E80 = ROOT / ".scratch/e80"
PY = str(ROOT / ".venv/bin/python"); PY_SELF = str(HERE / "e90_final.py")
V7P = ROOT / "data/submissions/player_v7p_全部己方手.csv"; V7P_SHA = "4d031029861e215a539c5f27278032a9cc730f9e17f425ea049f6e56b16759eb"
OLD800 = ROOT / ".scratch/e62/old_v6_800.csv"
OUT_NAME = "player_v7q_S通道.csv"
COLS = "at,mean,midat,midmean,std"
HSL_PQ = "data/test_scan_p1111/hsl120_test_games.parquet"
DATA90 = ROOT / ".scratch/e90/data"
N_H_CODE0 = 7367                      # player codes >= 7,367 in uh_games.npz are H players (make_data_h.py)
TRAIN_FLAGS = ["--w", "120", "--embname", "finens", "--emb-dir", ".scratch/e41", "--kata", "1", "--wk", "1.0", "--cols", COLS, "--extras", "1", "--late", "1", "--color", "1", "--rho", "0.5", "--tplda", "1", "--blk", "1", "--pe", "1"]   # .scratch/e86/ft.sh + --pe 1
PRED_FLAGS = ["--cols", COLS, "--tplda", "1", "--emb", "finens_full", "--emb-dir", ".scratch/e41", "--hsl-parquet", HSL_PQ]       # E80 RESULTS section 7 predict flags (used for v7n/v7p)
MEM = {"max": "12G", "min_avail": 16.0}


def sha(p, n=64):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()[:n]


def mem_available_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"): return int(ln.split()[1]) / 1e6


def guard_mem():
    """MemAvailable >= min_avail GB before every heavy step (reserve rule).  Looks at most 6 times, 5 minutes apart (MEM['wait'] = False: one look), then refuses."""
    for i in range(6 if MEM.get("wait", True) else 1):
        a = mem_available_gb()
        if a >= MEM["min_avail"]: return
        print(f"MemAvailable {a:.1f} GB < {MEM['min_avail']:.0f} GB (look {i + 1}); waiting 5 min", flush=True)
        if i < 5 and MEM.get("wait", True): time.sleep(300)
    raise AssertionError(f"MemAvailable {a:.1f} GB < {MEM['min_avail']:.0f} GB after 6 looks (reserve rule): not starting a heavy job")


def heavy(args, log, label):
    """Run `e90_final.py <args>` under the memory cap; blocks; raises on non-zero exit (137/-9 = killed by the cap).  Logs to `log`."""
    guard_mem(); Path(log).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={MEM['max']}", "-p", "MemorySwapMax=0", "env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4", "MKL_NUM_THREADS=4", PY, PY_SELF, *args]
    t0 = time.time()
    with open(log, "w") as f: f.write("CMD " + " ".join(cmd) + "\n"); f.flush(); rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    with open(log, "a") as f: f.write(f"exit {rc} wall {dt:.0f}s\n")
    assert rc == 0, f"{label}: exit {rc} (137/-9 = killed by the memory cap); see {log}"
    print(f"[{label}] done in {dt:.0f}s", flush=True); return dt


# ---------------------------------------------------------------- light guards
def roster_check():
    """Training-list guards of the delivery networks (finh_*), non-H part only (finh_h0 and finh_h1 both contain ALL of H by design, so 'mutually exclusive' is checked on non-H players):
    finh_full = fin_full + all H; finh_hK = fin_hK + all H; the non-H players of finh_h0 / finh_h1 are disjoint and contain TA / TB respectively (TA, TB = the E77 final groups .scratch/e39/final_groups.json = the booster pools' halves: 2,282 / 2,283 players with >= 30 games; the networks' lists cover 6,725 non-H players with >= 12 games);
    H players are exactly the codes >= 7,367 and are in every finh list."""
    R = np.load(DATA90 / "rosters_h.npz"); U = np.load(DATA90 / "uh_games.npz"); player = U["player"].astype(np.int64); G = len(player)
    aux = np.load(ROOT / ".scratch/e29/bundle/aux.npz", allow_pickle=True); pids = aux["player_ids"].astype(str)
    nonH = player < N_H_CODE0; assert len(pids) == N_H_CODE0 and (player[nonH] >= 0).all() and nonH.sum() == 707_820 and (~nonH).sum() == 287_055, (len(pids), int(nonH.sum()))
    g = json.load(open(ROOT / ".scratch/e39/final_groups.json")); TA, TB = set(g["TA"]), set(g["TB"]); assert not (TA & TB)
    m = {k: R[k] for k in ("fin_full", "fin_h0", "fin_h1", "finh_full", "finh_h0", "finh_h1")}
    for k, v in m.items(): assert v.shape == (G,), (k, v.shape)
    assert m["finh_full"].all() and (m["fin_full"] == nonH).all(), "finh_full / fin_full are not (everything) / (non-H part)"
    for k in ("h0", "h1"):
        assert (m[f"finh_{k}"] & nonH == m[f"fin_{k}"]).all(), f"finh_{k} non-H part != fin_{k}"
        assert m[f"finh_{k}"][~nonH].all(), f"finh_{k} does not contain all H games"
    assert not (m["finh_h0"] & m["finh_h1"] & nonH).any(), "non-H games in both finh_h0 and finh_h1"
    s0, s1 = set(pids[player[m["finh_h0"] & nonH]]), set(pids[player[m["finh_h1"] & nonH]])
    assert not (s0 & s1) and TA <= s0 and TB <= s1, f"cross-fit lists broken: non-H players of finh_h0/h1 overlap ({len(s0 & s1)}), or TA not inside finh_h0 ({len(TA - s0)} missing), or TB not inside finh_h1 ({len(TB - s1)} missing)"
    assert not (TA & s1) and not (TB & s0), "a booster-pool player of one half is in the other half's network list"
    hset = set(player[~nonH].tolist()); assert len(hset) == 640 and min(hset) == N_H_CODE0
    return dict(games=G, nonH=int(nonH.sum()), H_games=int((~nonH).sum()), H_players=len(hset), finh_h0_games=int(m["finh_h0"].sum()), finh_h1_games=int(m["finh_h1"].sum()), TA_players=len(TA), TB_players=len(TB),
                nonH_players_h0=len(s0), nonH_players_h1=len(s1), TA_in_h0=True, TB_in_h1=True, nonH_disjoint=True, H_in_both=True)


_D = {}


def d_rows():
    """(pid per D row (206,536), TA, TB): same deterministic selection as e86_final.stage_train (`_r`), without building the final space.  Asserts no H player in D."""
    if _D: return _D["pid"], _D["TA"], _D["TB"]
    import pandas as pd
    import e86_final as FN
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet", columns=["player_id", "color"]); Hset = set(FN.S3.R.sets()["H"])
    tot = meta.groupby("player_id").size(); keep = set(tot.index[tot >= 30]) - Hset; rng = np.random.default_rng(20261111)
    m = meta.assign(i=np.arange(len(meta))); m = m[m.player_id.isin(keep)]; r = []
    for (_p, _c), g_ in m.groupby(["player_id", "color"]):
        idx = g_["i"].to_numpy(); r.append(idx[rng.permutation(len(idx))[:FN.F.CAP]])
    r = np.sort(np.concatenate(r)); pid = meta["player_id"].to_numpy()[r].astype(str); assert len(pid) == 206_536, len(pid); assert not (set(pid) & Hset), "H player in D"
    g = json.load(open(ROOT / ".scratch/e39/final_groups.json")); _D.update(pid=pid, TA=set(g["TA"]), TB=set(g["TB"])); assert (_D["TA"] | _D["TB"]) <= set(pid) and not (_D["TA"] & _D["TB"])
    return pid, _D["TA"], _D["TB"]


def crossfit_signature(sdir, reps=5, nmax=2000):
    """Orientation of pe_final_h0 / h1 (cheap, numpy only).  Pool half h is fed network h{1-h}, so network hk must have been trained on half k (TA for h0, TB for h1) and NOT on the other half.
    Score = bag retrieval top-1 of up to nmax (2,000) players per half (>= 12 rows; each player's rows split at random into library / query bags, cosine of bag means, `reps` reps).  A network must retrieve the players it was
    trained on (own half) better than the other half's: score(net hk on half k) > score(net hk on half 1-k).  Reference numbers for pe86 are printed by `check`."""
    pid, TA, TB = d_rows(); half = {0: TA, 1: TB}; res = {}
    for k in (0, 1):
        A = np.load(Path(sdir) / f"pe_final_h{k}.npy", mmap_mode="r"); sc = {}; nzk = np.abs(np.asarray(A)).sum(1) > 0; by = {}
        for i in np.flatnonzero(nzk): by.setdefault(pid[i], []).append(int(i))
        for hh in (0, 1):
            ps = sorted(p for p in half[hh] if len(by.get(p, ())) >= 12); rg = np.random.default_rng(1000 + hh); ps = [ps[i] for i in rg.permutation(len(ps))[:nmax]]; acc = []
            for rep in range(reps):
                rg = np.random.default_rng(7 * rep + hh); Lb, Qb = [], []
                for p in ps:
                    ix = np.array(by[p])[rg.permutation(len(by[p]))]; h2 = len(ix) // 2
                    v = np.asarray(A[ix]); Lb.append(v[:h2].mean(0)); Qb.append(v[h2:].mean(0))
                Lb, Qb = np.array(Lb), np.array(Qb); Lb /= np.linalg.norm(Lb, axis=1, keepdims=True); Qb /= np.linalg.norm(Qb, axis=1, keepdims=True)
                acc.append(float((np.argmax(Qb @ Lb.T, 1) == np.arange(len(ps))).mean()))
            sc[hh] = round(float(np.mean(acc)), 4)
        res[f"h{k}"] = dict(own_half=sc[k], other_half=sc[1 - k], ok=bool(sc[k] > sc[1 - k]))
    return res


def validate_pe_dir(sdir, fake=False, strict_orientation=True, label="S"):
    """The four files of one channel directory (see module docstring).  fake=True allows links into E80 (dryrun).  Orientation failure raises only when strict_orientation."""
    sdir = Path(sdir); info = {}
    ref = {h: np.load(E86R / f"pe_final_h{h}.npy", mmap_mode="r") for h in (0, 1)}; mask = np.abs(np.asarray(ref[0])).sum(1) > 0
    for h in (0, 1):
        p = sdir / f"pe_final_h{h}.npy"; assert p.exists(), f"missing {p}"
        if not fake and sdir.resolve() not in (E86R.resolve(),):
            assert not p.is_symlink() and "e80" not in str(p.resolve()) and "e86" not in str(p.resolve()), f"{p}: S file must be a real file in its own directory"
        A = np.load(p); assert A.shape == (206_536, 128) and A.dtype == np.float32 and np.isfinite(A).all(), (p, A.shape, A.dtype)
        nz = np.abs(A).sum(1) > 0; assert (nz == mask).all(), f"{p}: covered rows ({int(nz.sum())}) != rows that carry KataGo vectors / pe86 ({int(mask.sum())})"
        nr = np.linalg.norm(A[nz], axis=1); assert np.abs(nr - 1).max() < 1e-3, f"{p}: covered rows not unit length"
        if sdir.resolve() != E86R.resolve(): assert np.abs(A[nz] - np.asarray(ref[h])[nz]).max() > 1e-3, f"{p} equals pe86 final h{h}"
        info[f"pe_final_h{h}"] = dict(sha16=sha(p, 16), bytes=p.stat().st_size, rows_covered=int(nz.sum()), max_unit_dev=float(np.abs(nr - 1).max()))
    assert info["pe_final_h0"]["sha16"] != info["pe_final_h1"]["sha16"], "pe_final_h0 and h1 are identical files"
    for sp in ("pub", "priv"):
        p = sdir / f"pe_test_{sp}.npy"; assert p.exists(), f"missing {p}"
        A = np.load(p); assert A.shape == (7100, 128) and A.dtype == np.float32 and np.isfinite(A).all() and np.abs(np.linalg.norm(A, axis=1) - 1).max() < 1e-3, (p, A.shape)
        if sdir.resolve() != E86R.resolve(): assert np.abs(A - np.load(E86R / f"pe_test_{sp}.npy")).max() > 1e-3, f"{p} equals pe86 test"
        info[f"pe_test_{sp}"] = dict(sha16=sha(p, 16), bytes=p.stat().st_size)
    sig = crossfit_signature(sdir); info["crossfit_signature"] = sig
    if strict_orientation: assert all(v["ok"] for v in sig.values()), f"{label}: cross-fit orientation check failed {sig}: pe_final_h0/h1 swapped, or a network saw the other half's players"
    return info


# ---------------------------------------------------------------- internal heavy steps (run in a sub-process)
def core_args(ap_args):
    import e90_final_core as K
    K.PE_DIRS[:] = [str(Path(d)) for d in ap_args.pe_dirs.split(",")]; K.OUT_DIR[0] = Path(ap_args.outdir); K.FINO = Path(ap_args.outdir) / "models"
    return K


def cmd_train(a):
    K = core_args(a); K.CFG.update(old=a.old_gv, cols=a.cols, extras=a.extras, nc=100 if a.tplda else 60, power=a.power)
    return K.stage_train(a)


def cmd_predict(a):
    K = core_args(a); K.CFG.update(old=a.old_gv, cols=a.cols, extras=0, nc=100 if a.tplda else 60, power=0)
    return K.stage_predict(a)


def add_train_args(p):
    p.add_argument("--name", required=True); p.add_argument("--wdeep", type=float, default=1.0); p.add_argument("--draws", type=int, default=60); p.add_argument("--w", type=int, default=120); p.add_argument("--embname", default="fin"); p.add_argument("--emb-dir", default=str(ROOT / ".scratch/e40"))
    p.add_argument("--rounds", type=int, default=600); p.add_argument("--k", type=int, default=40); p.add_argument("--seed", type=int, default=8); p.add_argument("--kata", type=int, default=0); p.add_argument("--wk", type=float, default=1.0); p.add_argument("--old-gv", type=int, default=0)
    p.add_argument("--cols", default="at,mean"); p.add_argument("--extras", type=int, default=0); p.add_argument("--color", type=int, default=0); p.add_argument("--rho", type=float, default=0.5); p.add_argument("--late", type=int, default=0); p.add_argument("--wk2", type=float, default=1.0)
    p.add_argument("--tplda", type=int, default=0); p.add_argument("--power", type=int, default=0); p.add_argument("--blk", type=int, default=0); p.add_argument("--ft", type=int, default=0); p.add_argument("--wf", type=float, default=1.0); p.add_argument("--pe", type=int, default=0)
    p.add_argument("--pe-dirs", required=True); p.add_argument("--outdir", required=True)


def add_pred_args(p):
    p.add_argument("--model-dir", default=""); p.add_argument("--old-gv", type=int, default=0); p.add_argument("--cols", default="at,mean"); p.add_argument("--tplda", type=int, default=0); p.add_argument("--model", required=True); p.add_argument("--out", required=True)
    p.add_argument("--emb", default="fin_full"); p.add_argument("--emb-dir", default=str(ROOT / ".scratch/e40")); p.add_argument("--hsl-parquet", default=str(ROOT / "data/test_scan_p1111/hsl_test_games.parquet")); p.add_argument("--pe-dirs", required=True); p.add_argument("--outdir", required=True)


def cmd_t2rr(a):
    """COPY of scripts/oneoff/e86_t2rr.py (real test2 pseudo-queries, booster level, E86 PLAN criterion 4): same protocol / rng (default_rng(40), 20 reps, 400 queries, N in 1..4, 6-game library / 4-game query pool per player),
    same bootstrap (default_rng(0), 2,000, reps then queries).  Changes: models come from --base-dir/--new-dir (not .scratch/e86/r/models), every model has a LIST of channel directories (test vectors from <dir>/pe_test_<sp>.npy),
    features through e90_feats2 (one channel = original maths), the model-independent trunk vectors are loaded once; `--new-name ''` = base only (used by `check` to reproduce E86's stored scores)."""
    sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
    import pandas as pd, lightgbm as lgb
    import e38_final as EF
    import e86_final as FN
    import e90_feats2 as C2
    S3, zn = EF.S3, EF.zn
    FN.CFG.update(old=0, cols=COLS, extras=0)
    P, pm = EF.F.load_pm(); sel = pm.s["sel"]
    hp2 = pd.read_parquet(ROOT / HSL_PQ).set_index("uid")

    def load(model, fd, pes):
        fd = Path(fd); info = json.loads((fd / f"{model}.json").read_text()); Mz = np.load(fd / f"{info['metric_model']}.npz", allow_pickle=False); opt = info.get("opt", {})
        assert opt.get("pe") and info.get("kata") and opt.get("late") and opt.get("tplda") and opt.get("blk") and not opt.get("ft"), opt
        return dict(info=info, opt=opt, bst=lgb.Booster(model_file=str(fd / f"{model}.txt")), M={x: Mz[x] for x in Mz.files if x != "hsl_cols"}, cols=list(Mz["hsl_cols"]),
                    Pk=dict(np.load(fd / f"{model}_trk.npz")), Pu=dict(np.load(fd / f"{model}_late.npz")), Ph=FN.load_half(fd / f"{model}_trkhalf.npz"), Pb=FN.load_blk(fd / f"{model}_blk.npz"), pes=[Path(x) for x in pes.split(",")])
    mods = {"base": load(a.base_name, a.base_dir, a.base_pes)}
    if a.new_name: mods["new"] = load(a.new_name, a.new_dir, a.new_pes)
    B = mods["base"]
    for MD in mods.values():
        for kx in B["M"]: assert np.allclose(B["M"][kx], MD["M"][kx]), f"metric part {kx} differs between the two models"
        assert B["cols"] == MD["cols"] and B["info"]["opt"] == {**MD["info"]["opt"]} and B["info"]["wdeep"] == MD["info"]["wdeep"] and B["info"].get("wk") == MD["info"].get("wk")
    M, cols = B["M"], B["cols"]; FN.CFG["power"] = B["info"].get("power", 0)
    GVt = FN.load_gv(test=True); GVu = FN.load_gv(test=True, late=True); GVb = FN.load_gv(test=True, blk=True)      # model independent (same KataGo files)
    rng = np.random.default_rng(40); rows = []; perq = []
    for sp, s in (("public", "pub"), ("private", "priv")):
        z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); uids = z["uids"].astype(str); Zt = z["Z"]; pos = {u: i for i, u in enumerate(uids)}
        embE = np.load(ROOT / f".scratch/e41/emb_finens_full_test_{s}.npy").astype(np.float32)
        Yl, Yh = EF.project(Zt, hp2.reindex(uids)[cols].to_numpy(np.float64), M, True)
        assert np.isfinite(GVt[s][:, 0].astype(np.float32)).all()
        c = pd.read_csv(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", dtype=str)
        gid = c.game_id.str.replace("pub2_", "pub_").str.replace("priv2_", "priv_"); row = np.array([pos[f"cand_{s}:{g}"] for g in gid]); col = (c.color == "W").to_numpy().astype(int)
        Zr, Yl, Yh, embE = Zt[row], Yl[row], Yh[row], embE[row]
        for MD in mods.values():
            MD["Y"] = FN.apply_trk(MD["Pk"], GVt[s])[row]; MD["Yu"] = FN.apply_trk(MD["Pu"], GVu[s])[row]
            ys, ps_ = FN.half_Y(MD["Ph"], GVt[s]); MD["Yth"], MD["Pth"] = [y_[row] for y_ in ys], ps_
            MD["Yb"] = [y_[row] for y_ in FN.apply_blk(MD["Pb"], GVb[s])]
            MD["Ype"] = []
            for d in MD["pes"]:
                Y = np.load(d / f"pe_test_{s}.npy").astype(np.float32); assert Y.shape == (len(uids), 128) and np.isfinite(Y).all(); MD["Ype"].append(Y[row])
        pl = sorted(c.player_id.unique()); Pn = len(pl); byp = {p: np.flatnonzero((c.player_id == p).to_numpy()) for p in pl}; print(sp, "players", Pn, "games", len(row), flush=True)
        for rep in range(a.reps):
            lib, qp = [], []
            for p in pl: r = byp[p][rng.permutation(10)]; lib.append(r[:6]); qp.append(r[6:])
            qs, truth, ns = [], [], []
            for t in rng.permutation(Pn)[:a.nq]: n = int(rng.choice([1, 2, 3, 4])); qs.append(qp[t][:n]); truth.append(t); ns.append(n)
            truth = np.array(truth); pool = dict(lib=lib, q=qs); out = {}
            for nm, MD in mods.items():
                f = C2.feats(Zr, sel, pool, 40, Yl, Yh, True, embE, MD["info"]["wdeep"], MD["Y"], MD["info"].get("wk", 1.0), MD["opt"], col, MD["Yu"], MD["Yth"], MD["Pth"], MD["Yb"], None, MD["Ype"])
                e = MD["bst"].predict(f["X"], num_threads=4).reshape(len(qs), -1); order = np.argsort(-e, axis=1, kind="stable"); ts = np.take_along_axis(f["top"], order, 1)
                hit = ts == truth[:, None]; rank = np.where(hit.any(1), hit.argmax(1) + 1, 999)
                tF = f["F"][np.arange(len(truth)), truth]; rF = 1 + (f["F"] > tF[:, None]).sum(1)
                out[nm] = dict(score=np.where(rank <= 5, np.exp(-(rank - 1.0)), 0.0), scoreF=np.where(rF <= 5, np.exp(-(rF - 1.0)), 0.0), r40=(f["top"] == truth[:, None]).any(1).astype(float))
            nan = np.full(len(qs), np.nan)
            for q in range(len(qs)):
                o = out.get("new"); perq.append((sp, rep, ns[q], out["base"]["score"][q], o["score"][q] if o else np.nan, out["base"]["scoreF"][q], o["scoreF"][q] if o else np.nan, out["base"]["r40"][q], o["r40"][q] if o else np.nan))
            rows.append(dict(sp=sp, rep=rep, base=out["base"]["score"].mean(), new=out["new"]["score"].mean() if "new" in out else float("nan")))
            print(sp, rep, {k: round(v, 4) for k, v in rows[-1].items() if k not in ("sp", "rep")}, flush=True)
    df = pd.DataFrame(perq, columns=["sp", "rep", "n", "base", "new", "baseF", "newF", "r40b", "r40n"]); outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True); df.to_parquet(outdir / f"{a.out}_perq.parquet")
    brng = np.random.default_rng(0); res = {}
    for sp in ("public", "private"):
        d = df[df.sp == sp]; reps = sorted(d.rep.unique()); byrep = {r: d[d.rep == r] for r in reps}
        res[sp] = dict(base=float(d.base.mean()), baseF=float(d.baseF.mean()), byN_base={n: float(d[d.n == n].base.mean()) for n in (1, 2, 3, 4)})
        if not a.new_name: continue

        def stat(cn, cb):
            x = np.array([(byrep[r][cn] - byrep[r][cb]).mean() for r in reps]); return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

        def boot(cn, cb, B_=2000):
            o = np.empty(B_)
            for b in range(B_):
                ri = brng.integers(0, len(reps), len(reps)); o[b] = np.mean([(lambda g: (g[cn] - g[cb]).to_numpy()[brng.integers(0, len(g), len(g))].mean())(byrep[reps[i]]) for i in ri])
            return np.percentile(o, [2.5, 97.5])
        m, se = stat("new", "base"); lo, hi = boot("new", "base"); mF, seF = stat("newF", "baseF"); m40, se40 = stat("r40n", "r40b")
        res[sp].update(new=float(d.new.mean()), delta=float(m), se_rep=float(se), ci=[float(lo), float(hi)], deltaF=float(mF), seF=float(seF), dR40=float(m40), newF=float(d.newF.mean()),
                       byN={n: float((d[d.n == n].new - d[d.n == n].base).mean()) for n in (1, 2, 3, 4)})
        print(f"{sp}: base {d.base.mean():.4f} new {d.new.mean():.4f}  booster delta {m:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}]) | first-stage F delta {mF:+.4f} | R@40 delta {m40:+.4f}")
    json.dump(res, open(outdir / f"{a.out}.json", "w"), indent=1)


# ---------------------------------------------------------------- csls / assemble / comparisons
def t2_csv(real_dir, model, out, csls):
    """COPY of experiments/E86_player_all_positions/e86_make.py with the dump directory and model as arguments (CSLS-qq k=10, alpha=.5, z; Public only among Public queries, Private among Private)."""
    import csv
    sys.path[:0] = [str(ROOT / "scripts/lib")]
    from e81_lib import csls_qq
    rows = []
    for sp in ("pub", "priv"):
        z = np.load(Path(real_dir) / f"real_{model}_{sp}.npz", allow_pickle=True); e, top, pool, qids = z["e"], z["top"].astype(np.int64), z["pool"], z["qids"]
        if csls: e = csls_qq(e, top, len(pool), 10, 0.5, "z")
        t5 = np.take_along_axis(top, np.argsort(-e, axis=1, kind="stable"), 1)[:, :5]
        rows += [[str(q)] + [str(x) for x in pool[t5[i]]] for i, q in enumerate(qids)]
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]); w.writerows(rows)


def assemble(t2, out):
    r = subprocess.run([PY, str(ROOT / "scripts/oneoff/p1111_assemble.py"), "--old", str(OLD800), "--new2", str(t2), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def read_csv_rows(p):
    import csv
    with open(p, newline="") as f: return list(csv.reader(f))


def check_submission_out(p, data_dir, task="player"):
    r = subprocess.run([PY, str(ROOT / "scripts/check_submission.py"), str(p), "--task", task, "--data-dir", str(data_dir)], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def compare_to_v7p(final_csv):
    """Label-free checks of a final 1,600-row file against v7p: same question order, original 800 rows identical, test2 top-1 agreement (all / Public / Private), top-5 set identity."""
    a, b = read_csv_rows(final_csv), read_csv_rows(V7P); assert a[0] == b[0] and len(a) == len(b) == 1601
    out = dict(rows=len(a) - 1, same_question_order=[r[0] for r in a] == [r[0] for r in b], first800_identical=a[1:801] == b[1:801])
    t = list(zip(a[801:], b[801:])); out["test2_rows"] = len(t)
    out["test2_top1_same"] = sum(x[1] == y[1] for x, y in t); out["test2_top1_same_frac"] = round(out["test2_top1_same"] / len(t), 4)
    for nm, sl in (("public", slice(0, 400)), ("private", slice(400, 800))):
        tt = t[sl]; out[f"test2_top1_same_{nm}"] = sum(x[1] == y[1] for x, y in tt); out[f"top5_set_identical_{nm}"] = sum(set(x[1:]) == set(y[1:]) for x, y in tt)
    out["identical_file"] = sha(final_csv) == V7P_SHA
    return out


# ---------------------------------------------------------------- the path
def produce(tag, name, pe_dirs, outdir, t2rr_base, final_csv, t2rr_new=True, resume=False):
    """train -> predict -> CSLS -> assemble -> (pseudo-query).  Returns a dict of artefacts (paths, shas, timings)."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True); res = dict(tag=tag, name=name, pe_dirs=[str(d) for d in pe_dirs], timing={}); pe = ",".join(str(d) for d in pe_dirs)
    mdl = outdir / "models"
    done = lambda *ps: all(Path(p).exists() for p in ps)           # noqa: E731
    if done(mdl / f"{name}.txt") and not (resume and done(mdl / f"{name}.json")): raise SystemExit(f"{mdl / (name + '.txt')} exists: refusing to overwrite (delete the directory to redo, or --resume after an interrupted run)")
    plain = outdir / f"t2_plain_{name}.csv"
    if resume and done(mdl / f"{name}.json"): print("[train] resume: model exists, skipped", flush=True); res["timing"]["train"] = None
    else: res["timing"]["train"] = heavy(["_train", "--name", name, *TRAIN_FLAGS, "--pe-dirs", pe, "--outdir", str(outdir)], outdir / f"train_{name}.log", "train")
    if resume and done(plain, outdir / f"real_{name}_pub.npz", outdir / f"real_{name}_priv.npz"): print("[predict] resume: outputs exist, skipped", flush=True); res["timing"]["predict"] = None
    else: res["timing"]["predict"] = heavy(["_predict", "--model", name, "--model-dir", str(mdl), *PRED_FLAGS, "--out", str(plain), "--pe-dirs", pe, "--outdir", str(outdir)], outdir / f"pred_{name}.log", "predict")
    res["plain_t2_sha"] = sha(plain); rc, txt = check_submission_out(plain, ROOT / ".scratch/p1111/t2"); res["check_submission_plain_t2"] = dict(rc=rc, text=txt[-300:])
    t2 = outdir / f"t2_csls_{name}.csv"; t2_csv(outdir, name, t2, True); res["csls_t2_sha"] = sha(t2)
    rc, txt = assemble(t2, final_csv); res["assemble"] = dict(rc=rc, text=txt); assert rc == 0, txt
    res["final_csv"] = str(final_csv); res["final_sha"] = sha(final_csv)
    res["model_files"] = {p.name: sha(p, 16) for p in sorted(mdl.glob(f"{name}*"))}
    if resume and done(outdir / f"t2rr_{name}.json"): print("[t2rr] resume: output exists, skipped", flush=True); res["timing"]["t2rr"] = None
    else: res["timing"]["t2rr"] = heavy(["_t2rr", "--base-name", "k7p_pe", "--base-dir", t2rr_base[0], "--base-pes", t2rr_base[1], *( ["--new-name", name, "--new-dir", str(mdl), "--new-pes", pe] if t2rr_new else [] ), "--out", f"t2rr_{name}", "--outdir", str(outdir)], outdir / f"t2rr_{name}.log", "t2rr")
    res["t2rr"] = json.load(open(outdir / f"t2rr_{name}.json"))
    return res


def test_order_check():
    """The 14,200 test vectors of a channel are in export order: rows of export_spec_h.npz[test_player_t] -> t_games_uids; the first 7,100 must be Z_pub's game uids, the last 7,100 Z_priv's (the order pe_test_<sp>.npy is read in)."""
    tu = np.load(DATA90 / "t_games_uids.npz")["uids"].astype(str); tp = np.load(DATA90 / "export_spec_h.npz")["test_player_t"]
    zp = np.load(ROOT / ".scratch/p1111/z_t2/Z_pub.npz")["uids"].astype(str); zr = np.load(ROOT / ".scratch/p1111/z_t2/Z_priv.npz")["uids"].astype(str)
    assert len(tp) == 14_200 and (tu[tp[:7100]] == zp).all() and (tu[tp[7100:]] == zr).all(), "export order of the test vectors != Z_pub / Z_priv uid order"
    return True


def preflight(s_dir, fake=False, label="S"):
    out = dict(roster=roster_check(), test_vectors_order_matches_Z_uids=test_order_check(), pe86_orientation_reference=crossfit_signature(E86R))
    if s_dir: out["s_dir"] = validate_pe_dir(s_dir, fake=fake, strict_orientation=not fake, label=label)
    return out


def cmd_check(a):
    resume = getattr(a, "resume", False)
    FIN.mkdir(parents=True, exist_ok=True); d = FIN / "check"; res_path = FIN / "check.json"
    if res_path.exists(): raise SystemExit(f"{res_path} exists (delete to redo)")
    out = dict(note="NO S. Re-trains the pe86-only booster with the delivery copy and compares with E86's v7p artefacts.")
    ut = subprocess.run([PY, "-m", "pytest", "-q", str(HERE / "test_feats2.py")], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}); out["test_feats2"] = ut.stdout.strip().splitlines()[-1]
    out["preflight"] = preflight(None); print(json.dumps(out["preflight"], indent=1), flush=True)
    r = produce("check", "k7p_pe", [E86R], d, (str(E86R / "models"), str(E86R)), d / "player_v7p_recheck.csv", t2rr_new=False, resume=resume)
    cmpf = lambda x, y: (x.read_bytes() == y.read_bytes())          # noqa: E731
    mdl, old = d / "models", E86R / "models"
    out["bitwise"] = {"model_text_k7p_pe.txt": cmpf(mdl / "k7p_pe.txt", old / "k7p_pe.txt"), "plain_t2_csv_vs_v7p_plain_t2.csv": cmpf(d / "t2_plain_k7p_pe.csv", ROOT / ".scratch/e86/v7p_plain_t2.csv"),
                      "csls_t2_csv_vs_v7p_t2.csv": cmpf(d / "t2_csls_k7p_pe.csv", ROOT / ".scratch/e86/v7p_t2.csv"), "assembled_csv_vs_data/submissions/v7p": r["final_sha"] == V7P_SHA}
    for f in ("k7p_pe_metric.npz", "k7p_pe_trk.npz", "k7p_pe_late.npz", "k7p_pe_trkhalf.npz", "k7p_pe_blk.npz"):
        a_, b_ = np.load(mdl / f), np.load(old / f); out["bitwise"][f] = all(np.array_equal(a_[k], b_[k]) for k in b_.files) and set(a_.files) == set(b_.files)
    ja, jb = json.load(open(mdl / "k7p_pe.json")), json.load(open(old / "k7p_pe.json")); [x.pop(k, None) for x in (ja, jb) for k in ("seconds", "pe_dirs")]; out["bitwise"]["model_json_except_seconds"] = ja == jb
    for sp in ("pub", "priv"):
        x, y = np.load(d / f"real_k7p_pe_{sp}.npz", allow_pickle=True), np.load(E86R / f"real_k7p_pe_{sp}.npz", allow_pickle=True); out["bitwise"][f"real_dump_{sp}"] = all(np.array_equal(x[k], y[k]) for k in y.files)
    import pandas as pd
    q_new, q_old = pd.read_parquet(d / "t2rr_k7p_pe_perq.parquet"), pd.read_parquet(E86R / "t2rr_k7p_pe_vs_k7n_perq.parquet")
    out["t2rr_copy_reproduces_E86"] = dict(per_question_scores_identical=bool(np.array_equal(q_new["base"].to_numpy(), q_old["new"].to_numpy())), n=len(q_new), public_mean=r["t2rr"]["public"]["base"], private_mean=r["t2rr"]["private"]["base"],
                                           e86_public_mean=.5850, e86_private_mean=.6036)
    out["compare_to_v7p"] = compare_to_v7p(d / "player_v7p_recheck.csv"); out["run"] = r
    out["PASS"] = bool(all(out["bitwise"].values()) and out["t2rr_copy_reproduces_E86"]["per_question_scores_identical"] and out["compare_to_v7p"]["identical_file"])
    res_path.write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps({k: v for k, v in out.items() if k != "run"}, indent=1, ensure_ascii=False)); print("CHECK", "PASS" if out["PASS"] else "FAIL")
    return out["PASS"]


def summarize_s(r, final_csv, kind):
    out = dict(kind=kind, compare_to_v7p=compare_to_v7p(final_csv), t2rr=r["t2rr"], timing=r["timing"], final_sha=r["final_sha"], plain_t2_sha=r["plain_t2_sha"], model_files=r["model_files"])
    rc, txt = check_submission_out(final_csv, ROOT / ".scratch/p1111/t2"); rc0, txt0 = check_submission_out(V7P, ROOT / ".scratch/p1111/t2")
    out["check_submission_1600"] = dict(rc=rc, same_failure_text_as_v7p=(txt == txt0), v7p_rc=rc0, last_line=txt.splitlines()[-1] if txt else "", note="check_submission reports E_QPREFIX / E_ROWS on the official 1,600-row layout for v7p too; the authoritative validation is p1111_assemble (errors 0)")
    out["check_submission_plain_t2"] = r["check_submission_plain_t2"]; out["assemble_validation"] = r["assemble"]
    return out


def cmd_dryrun(a):
    FIN.mkdir(parents=True, exist_ok=True); d = FIN / "dryrun"; res_path = FIN / "result_dryrun.json"
    if res_path.exists(): raise SystemExit(f"{res_path} exists (delete to redo)")
    sd = d / "fake_s"; sd.mkdir(parents=True, exist_ok=True)
    for f in ("pe_final_h0.npy", "pe_final_h1.npy", "pe_test_pub.npy", "pe_test_priv.npy"):
        l = sd / f
        if not l.exists(): l.symlink_to(E80 / f)
    pre = preflight(sd, fake=True, label="fake S"); print(json.dumps(pre, indent=1), flush=True)
    final_csv = FIN / "player_v7q_dryrun_S通道.csv"
    if final_csv.exists() and not getattr(a, "resume", False): raise SystemExit(f"{final_csv} exists")
    r = produce("dryrun", "v7q_dryrun", [E86R, sd], d, (str(E86R / "models"), str(E86R)), final_csv, resume=getattr(a, "resume", False))
    out = dict(BANNER="DRYRUN with a FAKE S (E80's pe vectors linked as S). NOT an S result; the numbers only show the path runs.", preflight=pre, **summarize_s(r, final_csv, "dryrun"))
    res_path.write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out, indent=1, ensure_ascii=False)); return True


def cmd_make(a):
    FIN.mkdir(parents=True, exist_ok=True); res_path = FIN / "result.json"; final_dst = ROOT / "data/submissions" / OUT_NAME
    if res_path.exists() or final_dst.exists(): raise SystemExit(f"{res_path} or {final_dst} exists: refusing to overwrite")
    sd = Path(a.s_dir).resolve(); assert sd != E86R.resolve() and sd != E80.resolve() and "fake_s" not in str(sd), f"{sd} is not an S directory"; pre = preflight(sd, fake=False, label="S"); print(json.dumps(pre, indent=1), flush=True)
    d = FIN / "make"; final_tmp = d / OUT_NAME; d.mkdir(parents=True, exist_ok=True)
    r = produce("make", "v7q_S", [E86R, sd], d, (str(E86R / "models"), str(E86R)), final_tmp, resume=getattr(a, "resume", False))
    out = dict(preflight=pre, s_dir=str(sd), **summarize_s(r, final_tmp, "make"))
    with open(final_dst, "xb") as f: f.write(final_tmp.read_bytes())          # never overwrites
    out["submission_path"] = str(final_dst); out["submission_sha256"] = sha(final_dst)
    with open(res_path, "x") as f: f.write(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)); print("written", final_dst, res_path); return True


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("check", "dryrun", "make", "preflight"):
        p = sub.add_parser(n); p.add_argument("--mem-max", default="12G"); p.add_argument("--min-avail", type=float, default=16.0); p.add_argument("--resume", action="store_true", help="reuse finished train/predict/t2rr outputs after an interrupted run (same directory)")
        if n in ("make", "preflight"): p.add_argument("--s-dir", required=(n == "make"), default=None)
    add_train_args(sub.add_parser("_train")); add_pred_args(sub.add_parser("_predict"))
    t = sub.add_parser("_t2rr"); t.add_argument("--base-name", required=True); t.add_argument("--base-dir", required=True); t.add_argument("--base-pes", required=True); t.add_argument("--new-name", default=""); t.add_argument("--new-dir", default=""); t.add_argument("--new-pes", default="")
    t.add_argument("--reps", type=int, default=20); t.add_argument("--nq", type=int, default=400); t.add_argument("--out", required=True); t.add_argument("--outdir", required=True)
    a = ap.parse_args()
    if a.cmd in ("check", "dryrun", "make", "preflight"): MEM["max"], MEM["min_avail"] = a.mem_max, a.min_avail
    if a.cmd == "check": sys.exit(0 if cmd_check(a) else 1)
    if a.cmd == "dryrun": cmd_dryrun(a)
    elif a.cmd == "make": cmd_make(a)
    elif a.cmd == "preflight":
        pre = preflight(a.s_dir, fake=False); print(json.dumps(pre, indent=1))
    elif a.cmd == "_train": return cmd_train(a)
    elif a.cmd == "_predict": return cmd_predict(a)
    elif a.cmd == "_t2rr": cmd_t2rr(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
