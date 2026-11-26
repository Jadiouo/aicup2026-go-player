"""E123 validation.  Subcommands
  dry     dry-run checks (1)-(3); prints no score that includes X_lr.  Writes .scratch/e123/dry_check.json
  formal  needs E123_FORMAL=1; one time only (result.json created with open(...,'x')).
Controls: zero-new-information (centroid cosine of the same vectors, w=1), shuffled library (vectors permuted against labels), per-N, top-1 changes.
"""
import sys, os, json, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE)]
import numpy as np
import e123_lib as L
from e123_lib import OUT, DUMPS, TAGS, STRUCT, POOLS, NC
E112 = L.ROOT / ".scratch/e112"
EXPECT = {("a", "pub"): .646954, ("a", "priv"): .669544, ("c", "pub"): .633376, ("c", "priv"): .647162}
LOG = OUT / "logs"


def xe_file(tag, s): return E112 / f"t2_X_{tag}_{s}.npz"


def ci(d): return L.S.ci(np.asarray(d, np.float64))


def player_means(delta, truth):
    ups = np.unique(truth); pi = np.searchsorted(ups, truth); return ups, np.bincount(pi, delta) / np.bincount(pi)


def dry(a):
    R = dict(); L_ = lambda *x: print(*x, flush=True); sc_ok = True
    # (1) reproduction of the current v7u scores and equality with e112_stage2.per_query
    r1 = {}
    for tag in TAGS:
        for s, sp in POOLS:
            mine = L.per_query2(tag, s, xe_file(tag, s)); _, ref, truth = L.S.per_query(tag, s, .5, xe_file(tag, s))
            diff = float(np.abs(mine["cur"] - ref).max()); mean = float(mine["cur"].mean()); k = f"{STRUCT[tag]}/{s}"
            r1[k] = dict(mean=mean, mean_6dp=round(mean, 6), expected=EXPECT.get((STRUCT[tag], s)), max_abs_diff_vs_e112_per_query=diff, equal_array=bool(np.array_equal(mine["cur"], ref)))
            L_(f"  (1) {k}: current mean {mean:.6f} expected {EXPECT.get((STRUCT[tag], s))} max|diff vs e112 per_query| {diff}")
            if (STRUCT[tag], s) in EXPECT: sc_ok &= round(mean, 6) == EXPECT[(STRUCT[tag], s)]
            sc_ok &= diff == 0.0
    R["check1_repro_current"] = dict(cells=r1, ok=bool(sc_ok))
    # (2) per-rep assertions on all reps (no training) + row-map check + one-rep smoke test
    r2 = {}
    for s, sp in POOLS:
        row = L.check_rows(s, sp)
        for tag in TAGS:
            z = np.load(DUMPS / f"t2{tag}_{s}.npz"); L.check_players(z, sp)
            for r in range(int(z["rep"].max()) + 1): L.assert_rep(z, r)
            r2[f"{STRUCT[tag]}/{s}"] = dict(reps=int(z["rep"].max()) + 1, ok=True)
    L_("  (2) per-rep assertions (library <-> candidate id, query rows not in library, query games belong to the truth player), all reps, all structures/pools: OK")
    s, sp = "pub", "public"; v = L.load_vecs(s); row = L.check_rows(s, sp); t0 = time.time()
    X, iters = L.build_X("", s, sp, v, row, "head", reps=[0]); m = np.load(DUMPS / f"t2_{s}.npz")["rep"] == 0
    assert X[m].shape == (400, NC) and np.isfinite(X[m]).all() and abs(float(X[m].mean())) < 1e-5 and (X[~m] == 0).all()
    _ = L.per_query2("", s, xe_file("", s), X, 1.0, reps=[0]); assert np.isfinite(_["new"][m]).all() and _["new"][m].shape == (400,)        # runs, shape ok; NOT printed
    R["check2_assert_all_reps"] = r2; R["smoke_one_rep"] = dict(pool="pub", struct="a", rep=0, X_shape=[400, NC], finite=True, seconds=round(time.time() - t0, 1))
    L_("  (2) smoke: one rep, one pool (pub, a, rep 0): shapes ok, runs end to end (score not printed)")
    R["check3_convergence_smoke"] = dict(n_iter=iters, convergence_warning=False, max_iter=L.MAXITER, solver="lbfgs", tol_default=1e-4, note="ConvergenceWarning raises in fit_head; formal run records every fit's n_iter")
    L_(f"  (3) logistic regression converged (no ConvergenceWarning), n_iter {iters}")
    with open(OUT / "dry_check.json", "w") as f: f.write(json.dumps(R, indent=1))
    L_("DRY_1_2_3", "OK" if sc_ok else "MISMATCH")


def formal(a):
    assert os.environ.get("E123_FORMAL") == "1", "needs E123_FORMAL=1"; outp = OUT / "result.json"; assert not outp.exists(), "formal scoring runs once"
    logf = open(LOG / "formal.log", "a"); P = lambda *x: (print(*x, flush=True), print(*x, file=logf, flush=True)); t0 = time.time(); R = {}; iters_all = {}; perq = {}
    for s, sp in POOLS:
        v = L.load_vecs(s); row = L.check_rows(s, sp)
        for tag in TAGS:
            k = f"{STRUCT[tag]}/{s}"; xf = xe_file(tag, s); Xh, ih = L.build_X(tag, s, sp, v, row, "head"); Xc, _ = L.build_X(tag, s, sp, v, row, "cent"); iters_all[k] = ih
            H = L.per_query2(tag, s, xf, Xh, 1.0); C = L.per_query2(tag, s, xf, Xc, 1.0); cur = H["cur"]; truth = H["truth"]; n = H["n"]
            assert np.array_equal(cur, C["cur"])
            dH, dC = H["new"] - cur, C["new"] - cur; ups, pH = player_means(dH, truth); _, pC = player_means(dC, truth)
            cell = dict(current_mean=float(cur.mean()), with_head_mean=float(H["new"].mean()), delta_head=ci(pH), delta_centroid=ci(pC), head_minus_centroid=ci(pH - pC), n_iter=dict(min=min(ih), max=max(ih), mean=float(np.mean(ih)), n_fits=len(ih)))
            cell["by_N"] = {}
            for N in sorted(set(n)):
                mm = n == N; u2 = np.unique(truth[mm]); pi = np.searchsorted(u2, truth[mm]); d = np.bincount(pi, dH[mm]) / np.bincount(pi); cell["by_N"][str(int(N))] = dict(ci(d), n_queries=int(mm.sum()), cur_mean=float(cur[mm].mean()))
            ch = H["top1_new"] != H["top1_cur"]; cell["top1_changed_per_400"] = float(ch.sum() / 20.0); cell["top1_changed_total"] = int(ch.sum())
            if tag in ("", "c"):
                Xs, _ = L.build_X(tag, s, sp, v, row, "shuf"); Sh = L.per_query2(tag, s, xf, Xs, 1.0); _, pS = player_means(Sh["new"] - cur, truth); cell["delta_shuffled"] = ci(pS); perq[f"{k}/shuf_new"] = Sh["new"]
            R[k] = cell; perq[f"{k}/cur"] = cur; perq[f"{k}/head_new"] = H["new"]; perq[f"{k}/cent_new"] = C["new"]
            P(f"  {k:9s} cur {cell['current_mean']:.6f} head {cell['with_head_mean']:.6f} | d_head {cell['delta_head']['point']:+.4f} [{cell['delta_head']['lo']:+.4f}, {cell['delta_head']['hi']:+.4f}] "
              f"d_cent {cell['delta_centroid']['point']:+.4f} [{cell['delta_centroid']['lo']:+.4f}, {cell['delta_centroid']['hi']:+.4f}] head-cent {cell['head_minus_centroid']['point']:+.4f} [{cell['head_minus_centroid']['lo']:+.4f}, {cell['head_minus_centroid']['hi']:+.4f}]"
              + (f" shuf {cell['delta_shuffled']['point']:+.4f}" if "delta_shuffled" in cell else "") + f" | iters {min(ih)}-{max(ih)} | top1 changed/400 {cell['top1_changed_per_400']:.1f}  [{time.time() - t0:.0f}s]")
            P("     by N: " + " ".join(f"N={nn}: {c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" for nn, c in cell["by_N"].items()))
    cond = {}
    for s in ("pub", "priv"):
        c = R[f"a/{s}"]["delta_head"]; cond[f"1_a_{s}_point>=+.005"] = bool(c["point"] >= .005); cond[f"1_a_{s}_lo>0"] = bool(c["lo"] > 0)
    for s in ("pub", "priv"): cond[f"2_c_{s}_point>=-.005"] = bool(R[f"c/{s}"]["delta_head"]["point"] >= -.005)
    for s in ("pub", "priv"): cond[f"3_a_{s}_head_minus_centroid_lo>0"] = bool(R[f"a/{s}"]["head_minus_centroid"]["lo"] > 0)
    for st in ("a", "c"):
        for s in ("pub", "priv"): cond[f"4_shuffle_{st}_{s}_point<=+.001"] = bool(R[f"{st}/{s}"]["delta_shuffled"]["point"] <= .001)
    fails = [k for k, ok in cond.items() if not ok]; verdict = "E123 PASS" if not fails else "E123 STOP " + ", ".join(fails)
    res = dict(cells=R, conditions=cond, verdict=verdict, n_iter_all={k: v for k, v in iters_all.items()}, F5=dict(status="abandoned", reason="see RESULTS.md: finh_h0/finh_h1 vec files hold training games only (rid<1e7, 109,560 rows), no test2 games"),
               method="F3 vectors (small S, big S, pe86; per group L2 -> concat -> L2 -> x8), LogisticRegression(C=1) per rep on the rep's 6 library games/candidate, query = mean log-prob over its games then row-z; combined = z_row(e)+.5*X_e112+1.0*X_lr on the 40 shortlist -> csls_qq(3,.5,z); player-level bootstrap B=2000 seed 0")
    with open(outp, "x") as f: f.write(json.dumps(res, indent=1))
    np.savez_compressed(OUT / "formal_perquery.npz", **perq); P("conditions:", json.dumps(cond)); P(verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("dry", "formal"): sp.add_parser(n)
    a = ap.parse_args(); {"dry": dry, "formal": formal}[a.cmd](a)
