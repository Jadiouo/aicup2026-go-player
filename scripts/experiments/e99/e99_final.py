"""E99 final: produce the Player candidate files v7r from the BIG-network S channel.  Thin wrapper around scripts/experiments/e90/e90_final.py (imported, NEVER modified): the whole path
train -> predict -> CSLS -> assemble -> pseudo-query is e90_final.produce(), which is exactly how v7q was made.  What this file changes: where the S channel directories come from, the
output names/folders, the overwrite guards, and the variants.

Variants (channel list handed to the booster = e90_final.produce(pe_dirs=...), pe86 is always channel 0):
  repro : [pe86, SMALL]            = v7q.  Only a consistency check (output: .scratch/e99/final_check/).  With --reuse-models DIR it skips the 13-minute booster training and re-runs everything after it
                                   with the models v7q was trained with (DIR default .scratch/e90/final/make/models).
  a     : [pe86, BIG]              big-network S replaces the small-network S.            -> .scratch/e99/final/player_v7r_a_*.csv
  b     : [pe86, SMALL, BIG]       both S channels, as two channels (6 more booster features each, one more first-stage cosine term).  -> .scratch/e99/final/player_v7r_b_*.csv
--dry : the BIG directory is a COPY of the small one (fake big network); outputs go to .scratch/e99/final_dry/ and carry `dry` in every name.  The numbers are NOT an S result.

Sub-commands:
  preflight --variant V [--dry] [--small-dir D] [--big-dir D]      light guards only (rosters, test order, shapes, unit length, covered rows, cross-fit orientation, big != small)
  make      --variant V [--dry] [--small-dir D] [--big-dir D] [--mem-max 12G] [--min-avail 16] [--threads 4] [--reuse-models DIR] [--quick-train (dry only)] [--resume]
  compare   RESULT.json [RESULT.json ...]                          one table: pseudo-query Public / Private score with 95% intervals, delta against v7p, against v7q, file identity

Guards: output files only under .scratch/e99/ (never data/submissions/, never an existing name there); result.json is written with mode "x" (never overwritten); produce() refuses to retrain over an existing model.
Heavy steps run under systemd-run (MemoryMax=--mem-max, swap 0), CUDA_VISIBLE_DEVICES empty, --threads BLAS/OMP threads, and refuse to start when MemAvailable < --min-avail GB (one look, no waiting).
Nothing here uses a GPU.  The booster training peaks at ~9.9 GB RSS with two channels (v7q), so it needs --mem-max 12G; predict / pseudo-query / CSLS / assemble are small.
"""
import sys, os, json, time, argparse, shutil, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "experiments" / "e90")]
import e90_final as E                                   # read-only original; its sub-commands _train/_predict/_t2rr are what the heavy steps call

SCR = ROOT / ".scratch/e99"
ROOTS = {"real": SCR / "final", "dry": SCR / "final_dry", "check": SCR / "final_check"}
SUB = ROOT / "data/submissions"
SMALL = ROOT / ".scratch/e90/r_h"; BIG = SCR / "r"; V7Q = SUB / "player_v7q_S通道.csv"; V7Q_RESULT = ROOT / ".scratch/e90/final/result.json"; V7Q_MODELS = ROOT / ".scratch/e90/final/make/models"
V7Q_SHA = "93ef784b8f9c00e38dc12e96738a785f976bf22c0a720590a53d650a78a44e58"
FILE_NAME = {"repro": "player_v7q_repro_check.csv", "a": "player_v7r_a_大網路S取代.csv", "b": "player_v7r_b_大小網路S並用.csv"}
MODEL_NAME = {"repro": "v7q_S", "a": "v7r_a", "b": "v7r_b"}              # repro must keep v7q's model name (the reused model files carry it)


def my_heavy_factory(threads):
    def heavy(args, log, label):
        """Same as e90_final.heavy (guard, systemd-run cap, no GPU) but with a configurable thread count and a single memory look (no 5-minute waiting)."""
        E.guard_mem(); Path(log).parent.mkdir(parents=True, exist_ok=True); t = str(threads)
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={E.MEM['max']}", "-p", "MemorySwapMax=0", "env", "CUDA_VISIBLE_DEVICES=", f"OMP_NUM_THREADS={t}", f"OPENBLAS_NUM_THREADS={t}", f"MKL_NUM_THREADS={t}",
               "PYTHONDONTWRITEBYTECODE=1", E.PY, E.PY_SELF, *args]
        t0 = time.time()
        with open(log, "w") as f: f.write("CMD " + " ".join(cmd) + "\n"); f.flush(); rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        with open(log, "a") as f: f.write(f"exit {rc} wall {dt:.0f}s\n")
        assert rc == 0, f"{label}: exit {rc} (137/-9 = killed by the memory cap); see {log}"
        print(f"[{label}] done in {dt:.0f}s", flush=True); return dt
    return heavy


def layout(variant, dry):
    """(output root, model name, final csv path).  Every output lives under .scratch/e99/."""
    root = ROOTS["check"] if variant == "repro" else ROOTS["dry" if dry else "real"]
    fname = FILE_NAME[variant]
    if dry: fname = fname.replace(".csv", "").replace("player_v7r_", "player_v7r_dry_") + ".csv"
    name = MODEL_NAME[variant] + ("_dry" if dry and variant != "repro" else "")
    final = root / fname
    rp = final.resolve()
    assert (SCR.resolve() in rp.parents) and SUB.resolve() not in rp.parents, f"output {rp} must be under .scratch/e99/ and never under data/submissions/"
    assert not (SUB / final.name).exists(), f"a file named {final.name} already exists in data/submissions/ (name must differ from every existing submission)"
    return root, name, final


def channels(variant, small, big):
    return {"repro": [small], "a": [big], "b": [small, big]}[variant]


def preflight(variant, dry, small, big):
    out = dict(roster=E.roster_check(), test_vectors_order_matches_Z_uids=E.test_order_check(), pe86_orientation_reference=E.crossfit_signature(E.E86R), s_dirs={})
    dirs = channels(variant, small, big)
    for d in dirs:
        out["s_dirs"][str(d)] = E.validate_pe_dir(d, fake=False, strict_orientation=True, label=str(d))
    if variant != "repro" and not dry:                      # a real BIG channel must be a different network from the small one (and from pe86: validate_pe_dir)
        assert E.sha(Path(big) / "pe_final_h0.npy") != E.sha(Path(small) / "pe_final_h0.npy") and E.sha(Path(big) / "pe_test_pub.npy") != E.sha(Path(small) / "pe_test_pub.npy"), "BIG channel equals the SMALL channel (use --dry for a fake copy)"
        assert Path(big).resolve() != Path(small).resolve()
    return out


def pq_summary(t2rr):
    return {sp: dict(v7p=round(t2rr[sp]["base"], 4), new=round(t2rr[sp]["new"], 4), delta_vs_v7p=round(t2rr[sp]["delta"], 4), ci95=[round(x, 4) for x in t2rr[sp]["ci"]]) for sp in ("public", "private")}


def make_tiny_booster(mdl, name, n_feat):
    """DRY ONLY (b): the 3-channel booster cannot be trained here (RAM), so take v7q's channel-independent model files (metric / trunk LDA / late / block / half LDA, fitted on KataGo vectors only) and put a
    random 5-round LightGBM ranker with the right feature count (798) in place of the booster.  Everything after training (predict, CSLS, assemble, pseudo-query, 3-channel feature building and loading) then runs
    for real; the scores are meaningless."""
    import numpy as np, lightgbm as lgb
    sys.path[:0] = [str(ROOT / "src")]
    from gogame import player_rerank as prr
    old = MODEL_NAME["repro"]; mdl.mkdir(parents=True, exist_ok=True); assert not (mdl / f"{name}.txt").exists()
    for f in sorted(V7Q_MODELS.glob(f"{old}*")):
        if f.suffix == ".txt": continue
        dst = mdl / (name + f.name[len(old):]); shutil.copyfile(f, dst)
        if f.suffix == ".json":
            j = json.loads(f.read_text()); j.update(metric_model=name + "_metric", n_feat=n_feat, tiny_booster_DRY_ONLY=True); dst.write_text(json.dumps(j))
    k = json.loads((mdl / f"{name}.json").read_text())["k"]; rng = np.random.default_rng(0); n = 100 * k
    X = rng.standard_normal((n, n_feat)).astype(np.float32); y = np.zeros(n, np.int8); y[::k] = 1
    bst = lgb.train(prr.lgb_params(0, 2), lgb.Dataset(X, y, group=np.full(n // k, k)), num_boost_round=5); bst.save_model(str(mdl / f"{name}.txt"))


def cmd_make(a):
    small, big = Path(a.small_dir).resolve(), Path(a.big_dir).resolve(); v = a.variant
    if a.reuse_models and not (v == "repro" or (v == "a" and a.dry)): raise SystemExit("--reuse-models is only for repro, and for `a --dry` (whose fake big channel is a byte copy of the small one, so v7q's booster is the right booster); a real S channel needs a newly trained booster")
    if a.dry and v == "repro": raise SystemExit("repro has no --dry")
    if a.quick_train and not a.dry: raise SystemExit("--quick-train (6 pools x 40 rounds, a path test with a useless booster) is only allowed with --dry")
    if a.quick_train: E.TRAIN_FLAGS = E.TRAIN_FLAGS + ["--draws", "6", "--rounds", "40"]
    if a.tiny_booster and not (a.dry and v == "b"): raise SystemExit("--tiny-booster is only for `b --dry`")
    root, name, final = layout(v, a.dry); root.mkdir(parents=True, exist_ok=True); res_path = root / f"result_{v}.json"; d = root / v
    if res_path.exists(): raise SystemExit(f"{res_path} exists: refusing to overwrite (use a new folder or delete it yourself)")
    if final.exists() and not a.resume: raise SystemExit(f"{final} exists: refusing to overwrite")
    E.MEM["max"], E.MEM["min_avail"], E.MEM["wait"] = a.mem_max, a.min_avail, False; E.heavy = my_heavy_factory(a.threads)
    if v != "repro" and not a.dry: assert str(root.resolve()).endswith(".scratch/e99/final")
    t0 = time.time(); pre = preflight(v, a.dry, small, big); print(json.dumps(pre["s_dirs"], indent=1), "\nroster / order / orientation guards passed", flush=True)
    dirs = channels(v, small, big); pe_dirs = [E.E86R] + dirs; d.mkdir(parents=True, exist_ok=True); resume = a.resume
    if a.tiny_booster: resume = True
    if a.reuse_models:
        src = Path(a.reuse_models); mdl = d / "models"; assert not (mdl / f"{name}.txt").exists(), f"{mdl} already holds a model"; mdl.mkdir(parents=True, exist_ok=True)
        old = MODEL_NAME["repro"]; files = sorted(src.glob(f"{old}*")); assert len(files) == 7, [f.name for f in files]
        if name != old:                                        # a --dry: only valid when the fake big directory is byte-identical to the small one
            for fn in ("pe_final_h0.npy", "pe_final_h1.npy", "pe_test_pub.npy", "pe_test_priv.npy"): assert E.sha(big / fn) == E.sha(small / fn), f"{big / fn} is not a byte copy of the small channel: v7q's booster does not apply"
        for f in files:
            dst = mdl / (name + f.name[len(old):]); shutil.copyfile(f, dst)
            if f.suffix == ".json": dst.write_text(dst.read_text().replace(old + "_metric", name + "_metric"))
        resume = True; print(f"REUSING booster models of {src} ({len(files)} files copied); only predict / CSLS / assemble / pseudo-query are re-run", flush=True)
    if a.tiny_booster: make_tiny_booster(d / "models", name, n_feat=792 + 6 * (len(dirs) - 1))
    r = E.produce(f"e99_{v}", name, pe_dirs, d, (str(E.E86R / "models"), str(E.E86R)), final, resume=resume)
    out = dict(variant=v, dry=a.dry, BANNER=("DRY RUN with a FAKE big network (copy of the small one); NOT an S result" if a.dry else ("CONSISTENCY CHECK of the v7q path" if v == "repro" else "candidate")),
               channels=[str(x) for x in pe_dirs], models_reused_from=(str(a.reuse_models) if a.reuse_models else None), mem_max=a.mem_max, threads=a.threads, quick_train=a.quick_train, tiny_booster=a.tiny_booster, wall_seconds=round(time.time() - t0),
               preflight=pre, **E.summarize_s(r, final, "e99_" + v))
    out["pseudo_query"] = pq_summary(out["t2rr"]); out["file"] = str(final); out["sha256"] = E.sha(final); out["identical_to_v7q"] = (out["sha256"] == V7Q_SHA and V7Q.read_bytes() == final.read_bytes())
    out["n_booster_features"] = json.loads((d / "models" / f"{name}.json").read_text())["n_feat"]
    with open(res_path, "x") as f: f.write(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("variant", "dry", "pseudo_query", "sha256", "identical_to_v7q", "n_booster_features", "wall_seconds", "check_submission_plain_t2", "assemble_validation")}, indent=1, ensure_ascii=False))
    print("written", final, res_path, "(nothing under data/submissions/)"); return out


def cmd_compare(a):
    v7q = json.loads(V7Q_RESULT.read_text())["t2rr"]; rows = [("v7p (baseline)", v7q["public"]["base"], None, None, v7q["private"]["base"], None, None, "-")]
    rows.append(("v7q (small S, uploaded)", v7q["public"]["new"], v7q["public"]["delta"], v7q["public"]["ci"], v7q["private"]["new"], v7q["private"]["delta"], v7q["private"]["ci"], V7Q_SHA[:8]))
    for p in a.results:
        r = json.loads(Path(p).read_text()); t = r["t2rr"]; tag = r["variant"] + (" DRY" if r["dry"] else "")
        rows.append((tag, t["public"]["new"], t["public"]["delta"], t["public"]["ci"], t["private"]["new"], t["private"]["delta"], t["private"]["ci"], r["sha256"][:8]))
    f = lambda x: "-" if x is None else f"{x:.4f}"; ci = lambda c: "-" if c is None else f"[{c[0]:+.4f}, {c[1]:+.4f}]"; dl = lambda x: "-" if x is None else f"{x:+.4f}"   # noqa: E731
    print("| 版本 | Public | Δ vs v7p | 95% CI | Private | Δ vs v7p | 95% CI | sha256 前 8 |\n|---|---|---|---|---|---|---|---|")
    for n, pu, pd_, pc, pr, prd, prc, s in rows: print(f"| {n} | {f(pu)} | {dl(pd_)} | {ci(pc)} | {f(pr)} | {dl(prd)} | {ci(prc)} | {s} |")
    print("\nPaired difference against v7q (same 20 reps x 400 pseudo-queries per split: same rng, same questions; bootstrap = reps then queries, 2,000, default_rng(0), as in t2rr):")
    import numpy as np, pandas as pd
    q0 = pd.read_parquet(ROOT / ".scratch/e90/final/make/t2rr_v7q_S_perq.parquet")
    for p in a.results:
        pp = Path(p).resolve(); r = json.loads(pp.read_text()); fs = sorted((pp.parent / r["variant"]).glob("t2rr_*_perq.parquet")); assert len(fs) == 1, fs
        q1 = pd.read_parquet(fs[0]); assert len(q1) == len(q0) and (q1[["sp", "rep", "n"]].to_numpy() == q0[["sp", "rep", "n"]].to_numpy()).all() and np.allclose(q1["base"], q0["base"]), "pseudo-query sets differ"
        for sp in ("public", "private"):
            m0, m1 = q0[q0.sp == sp].reset_index(drop=True), q1[q1.sp == sp].reset_index(drop=True); dd = (m1["new"] - m0["new"]).to_numpy(); rep = m1["rep"].to_numpy(); reps = sorted(set(rep)); by = {x: dd[rep == x] for x in reps}
            br = np.random.default_rng(0); o = np.empty(2000)
            for b in range(2000): ri = br.integers(0, len(reps), len(reps)); o[b] = np.mean([by[reps[i]][br.integers(0, len(by[reps[i]]), len(by[reps[i]]))].mean() for i in ri])
            lo, hi = np.percentile(o, [2.5, 97.5]); print(f"{Path(p).name} {r['variant']}{' DRY' if r['dry'] else ''} {sp}: new - v7q = {dd.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("preflight", "make"):
        p = sub.add_parser(n); p.add_argument("--variant", required=True, choices=["repro", "a", "b"]); p.add_argument("--dry", action="store_true"); p.add_argument("--small-dir", default=str(SMALL)); p.add_argument("--big-dir", default=str(BIG))
        if n == "make":
            p.add_argument("--mem-max", default="12G"); p.add_argument("--min-avail", type=float, default=16.0); p.add_argument("--threads", type=int, default=4)
            p.add_argument("--reuse-models", default=None, help="repro only: skip booster training, reuse the models of this directory (v7q's)"); p.add_argument("--resume", action="store_true")
            p.add_argument("--tiny-booster", action="store_true", help="b --dry only: random 5-round booster with the 3-channel feature count (everything after training runs for real, scores meaningless)")
            p.add_argument("--quick-train", action="store_true", help="--dry only: train the booster with 6 pools x 40 rounds (path test, fits in 8G, scores meaningless)")
    c = sub.add_parser("compare"); c.add_argument("results", nargs="+")
    a = ap.parse_args()
    if a.cmd == "preflight":
        layout(a.variant, a.dry); pre = preflight(a.variant, a.dry, Path(a.small_dir).resolve(), Path(a.big_dir).resolve()); print(json.dumps(pre, indent=1)); print("PREFLIGHT OK")
    elif a.cmd == "make": cmd_make(a)
    else: cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
