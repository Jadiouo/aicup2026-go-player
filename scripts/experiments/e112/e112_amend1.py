"""E112 amendment 1 (PLAN 'Revision 1', written AFTER the first formal result was seen): same second gate as e112_stage2.formal, only the w grid is {.1,.25,.5,1}.
Everything else (kernel/sharpness 20, P12, X files, merge position, CSLS(3,.5), player-level 2-fold CV, player bootstrap B=2000 seed 0, conditions) is imported from / identical to e112_stage2.
Subcommands
  dry2     CPU, prints no real-channel number for w in {.5, 1}.  (a) grid {.1,.25}: evaluate2 on the real X must equal .scratch/e112/result.json['structs'] exactly (json round trip).
           (b) SHUFFLED channel files (shuf1, shuf2), grid {.1,.25,.5,1}: per-w delta for every structure/pool; also runs the information-only parts (conservative CI, per-N) on them so bugs surface before the one formal run.
  formal2  CPU, needs E112_FORMAL2=1.  Real X, grid {.1,.25,.5,1}.  Result .scratch/e112/result_amend1.json created with open(...,"x").  Last line: E112A1 ADOPT or E112A1 STOP <failed conditions>.
"""
import sys, os, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE)]
import numpy as np
import e112_stage2 as S
from e112_stage2 import OUT, DUMPS, TAGS, STRUCT, POOLS, per_query, ci
WS4 = (.1, .25, .5, 1.0); WS2 = (.1, .25)


def player_level2(dl, truth, ws):
    """as e112_stage2.player_level but with the grid as argument (identical arithmetic).  Also returns the per-player selected w for the CV stitching."""
    ups = np.unique(truth); P = {w: np.array([d[truth == u].mean() for u in ups]) for w, d in dl.items()}; n = len(ups)
    half = np.random.default_rng(0).permutation(n); h1, h2 = half[: n // 2], half[n // 2:]
    w1 = max(ws, key=lambda w: P[w][h1].mean()); w2 = max(ws, key=lambda w: P[w][h2].mean()); d = np.zeros(n); d[h2] = P[w1][h2]; d[h1] = P[w2][h1]
    wsel = np.zeros(n); wsel[h2] = w1; wsel[h1] = w2
    wall = max(ws, key=lambda w: P[w].mean()); return ups, P, d, (w1, w2), wall, wsel


def twoway_ci(dq, rep, truth, ups, B=2000, seed=0):
    """INFORMATION ONLY.  rep x player two-way bootstrap of the (CV-selected-w, held fixed) per-query delta dq; estimate = player-mean of per-player mean delta, as the formal point estimate.
    Reps and players are resampled independently with replacement; a player with no query in the resampled reps is dropped from that draw."""
    R = int(rep.max()) + 1; n = len(ups); pi = np.searchsorted(ups, truth); M = np.zeros((R, n)); N = np.zeros((R, n)); np.add.at(M, (rep, pi), dq); np.add.at(N, (rep, pi), 1.0)
    def est(a, b):
        Ma, Na = a @ M, a @ N; v = (Na > 0) & (b > 0); return float((b[v] * Ma[v] / Na[v]).sum() / b[v].sum())
    rng = np.random.default_rng(seed); out = np.empty(B)
    for i in range(B):
        a = np.bincount(rng.integers(0, R, R), minlength=R).astype(float); b = np.bincount(rng.integers(0, n, n), minlength=n).astype(float); out[i] = est(a, b)
    p = est(np.ones(R), np.ones(n)); return dict(point=p, lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)))


def evaluate2(xprefix, ws, info=True):
    R = {}
    for tag in TAGS:
        for s, sp, _ in POOLS:
            xf = OUT / f"t2_X{xprefix}_{tag}_{s}.npz"; dl, cur = {}, None
            for w in ws: cur, new, truth = per_query(tag, s, w, xf); dl[w] = new - cur
            ups, P, d, wsel, wall, wsel_p = player_level2(dl, truth, ws)
            r = dict(current_mean=float(cur.mean()), delta_cv=ci(d), w_sel_folds=list(wsel), w_all_players=wall, per_w={str(w): ci(P[w]) for w in ws}, mean_with_x_by_w={str(w): float((cur + dl[w]).mean()) for w in ws})
            if info:
                z = np.load(DUMPS / f"t2{tag}_{s}.npz"); rep, nq = z["rep"], z["n"]; pi = np.searchsorted(ups, truth); dq = np.array([dl[wsel_p[pi[i]]][i] for i in range(len(truth))])
                assert abs(np.array([dq[truth == u].mean() for u in ups]).mean() - d.mean()) < 1e-12          # query-level stitched delta == player-level stitched delta
                r["info_twoway_cv"] = twoway_ci(dq, rep, truth, ups)
                r["info_twoway_per_w"] = {str(w): twoway_ci(dl[w], rep, truth, ups) for w in ws}
                r["info_by_N_cv"] = {}
                for N in sorted(set(nq)):
                    m = nq == N; pu = np.unique(truth[m]); dn = np.array([dq[m & (truth == u)].mean() for u in pu]); r["info_by_N_cv"][str(int(N))] = dict(ci(dn), n_queries=int(m.sum()), cur_mean=float(cur[m].mean()))
            R[f"{STRUCT[tag]}/{sp}"] = r
    return R


def fmt2(R, info=True):
    out = [S.fmt(R)]
    if info:
        for k, v in R.items():
            tw = v["info_twoway_cv"]; out.append(f"  [info] {k:16s} two-way (rep x player) CV delta {tw['point']:+.4f} [{tw['lo']:+.4f}, {tw['hi']:+.4f}]  per w " + " ".join(f"{w}: {t['point']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}]" for w, t in v["info_twoway_per_w"].items()))
            out.append(f"  [info] {k:16s} by N (CV w, player CI) " + " ".join(f"N={n}: {t['point']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}] (q {t['n_queries']})" for n, t in v["info_by_N_cv"].items()))
    return "\n".join(out)


def dry2(a):
    ref = json.loads((OUT / "result.json").read_text())["structs"]; R = S_eval_ws2()
    print("(a) grid {.1,.25}, real X, evaluate2 vs .scratch/e112/result.json['structs']:")
    ok = True
    for k in ref:
        same = json.loads(json.dumps({kk: R[k][kk] for kk in ref[k]})) == ref[k]; ok &= same; c = R[k]["delta_cv"]; print(f"  {k:16s} identical to result.json: {same}   (delta_cv {c['point']:+.6f} lo {c['lo']:+.6f} hi {c['hi']:+.6f})")
    print("DRY_A", "OK" if ok else "MISMATCH")
    allok = True
    for sh in (1, 2):
        Rs = evaluate2(f"_shuf{sh}", WS4); print(f"(b) shuffled channel shuf{sh}, grid {WS4} (per-w delta [95% player CI]; CV-selected delta):"); print(fmt2(Rs))
        pos = [(k, w, p["point"]) for k, v in Rs.items() for w, p in v["per_w"].items() if p["point"] > 0] + [(k, "cv", v["delta_cv"]["point"]) for k, v in Rs.items() if v["delta_cv"]["point"] > 0]
        print(f"  shuf{sh} entries with a positive point estimate: {pos}"); allok &= not pos
    print("DRY_B", "NO POSITIVE DELTA" if allok else "SOME POSITIVE DELTA (see lists)")


def S_eval_ws2(): return evaluate2("", WS2, info=False)


def formal2(a):
    assert os.environ.get("E112_FORMAL2") == "1", "needs E112_FORMAL2=1"
    outp = OUT / "result_amend1.json"; assert not outp.exists(), f"{outp} exists - amendment-1 scoring runs once"
    R = evaluate2("", WS4); pa, pv = R["a/public"]["delta_cv"], R["a/private"]["delta_cv"]; cc, cv = R["c/public"]["delta_cv"], R["c/private"]["delta_cv"]
    cond = {"1_a_public_point>=+.005": pa["point"] >= .005, "1_a_public_lo>0": pa["lo"] > 0, "1_a_private_point>=+.005": pv["point"] >= .005, "1_a_private_lo>0": pv["lo"] > 0,
            "2_c_public_point>=-.005": cc["point"] >= -.005, "2_c_private_point>=-.005": cv["point"] >= -.005}
    fails = [k for k, v in cond.items() if not v]; verdict = "E112A1 ADOPT" if not fails else "E112A1 STOP " + ", ".join(fails)
    res = dict(structs=R, conditions=cond, w_for_candidate=dict(public=R["a/public"]["w_all_players"], private=R["a/private"]["w_all_players"]), verdict=verdict, grid=list(WS4),
               method="same as e112_stage2.formal but w grid {.1,.25,.5,1}; info_* entries are informational only (not in the verdict)")
    with open(outp, "x") as f: f.write(json.dumps(res, indent=1))
    print(fmt2(R)); print("conditions:", json.dumps(cond)); print("w for candidate file (struct a, all players):", res["w_for_candidate"]); print(verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("dry2", "formal2"): sp.add_parser(n)
    a = ap.parse_args(); {"dry2": dry2, "formal2": formal2}[a.cmd](a)
