"""E112 second gate (adoption basis): real test2 candidate libraries, E110 pseudo-query dumps (structures a / c / b, Public / Private separately), channel X put after the booster output.
Subcommands
  compute [--shuffle SEED]   GPU (system python3 + torch).  X = mean over the three delivered S networks (finh_full / finh_h0 / finh_h1) of the row-z-scored (over all 600 candidates) KDE likelihood-ratio matrix
                             (P12 positions, whitened with the std over all positions of the pool's 6,000 candidate games per network, beta 20, library = the dump's 6 games per candidate, query score = mean over all positions of its games).
                             Writes .scratch/e112/t2_X{_shufSEED}_{tag}_{pub,priv}.npz  [8000,600] float32 (tag '' = structure a, 'c', 'b').  Reads no truths, scores nothing.
  repro                      CPU.  Recomputes the E110 (3,.5) pseudo-query scores from the dumps and compares with .scratch/e110/result.json.  Prints only.
  dry                        CPU.  w = 0 identity check on the real X, and the full pipeline on the SHUFFLED channel files (needs `compute --shuffle SEED` first).  Never prints a real-channel delta.  Writes nothing.
  formal                     CPU.  Needs E112_FORMAL=1.  Combination: z_row(booster e over its shortlist) + w * X[query, shortlist]  (only candidates the booster scored), then csls_qq(k=3, alpha=.5, mode z) per rep (= one pool);
                             w in {.1,.25} chosen per pool and structure by player-level 2-fold CV (players = true candidates; halves default_rng(0).permutation); delta = [with X] - [current (3,.5)] per query -> player means,
                             player-level bootstrap B=2000 default_rng(0).  Result .scratch/e112/result.json is created with open(...,"x").  Last line: E112 ADOPT or E112 STOP <failed conditions>.
"""
import sys, os, json, csv, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; OUT = ROOT / ".scratch/e112"; DUMPS = ROOT / ".scratch/e110/dumps"
sys.dont_write_bytecode = True; sys.path[:0] = [str(HERE), str(ROOT / "scripts/lib")]
import numpy as np
from e112_lib import Pos, nz, kde_game_sums, BETA
NETS = ("finh_full", "finh_h0", "finh_h1"); TAGS = ("", "c", "b"); STRUCT = {"": "a", "c": "c", "b": "b"}; POOLS = (("pub", "public", 0), ("priv", "private", 7100)); WS = (.1, .25); K, ALPHA = 3, .5; NC = 600


def zrow(S): return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def cand_rows(s, sp):
    """candidate-table row -> position inside the pool's 7,100 export rows (same mapping as scripts/experiments/e110/e110_t2dump.py)."""
    uids = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False)["uids"].astype(str); pos = {u: i for i, u in enumerate(uids)}
    with open(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", newline="") as f: rows = list(csv.DictReader(f))
    gid = [r["game_id"].replace("pub2_", "pub_").replace("priv2_", "priv_") for r in rows]; return np.array([pos[f"cand_{s}:{g}"] for g in gid])


def compute(a):
    import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True); t0 = time.time()
    perm = np.random.default_rng(a.shuffle).permutation(7100) if a.shuffle is not None else None; sfx = f"_shuf{a.shuffle}" if a.shuffle is not None else ""
    res = {(tag, s): [] for tag in TAGS for s, _, _ in POOLS}
    for net in NETS:
        P = Pos(OUT / f"t2_pos_{net}.npz"); assert P.n == 14200
        for s, sp, base in POOLS:
            row = cand_rows(s, sp); assert len(row) == 6000 and row.max() < 7100
            gm = (lambda g: perm[g]) if perm is not None else (lambda g: g)                 # pool-local game index -> pool-local export game (null control: random other game)
            cg = np.unique(row); allc = np.concatenate([P.vec[P.off[base + gm(g)]:P.off[base + gm(g) + 1]] for g in cg]).astype(np.float32); sd = allc.astype(np.float64).std(0).astype(np.float32)   # std over the 6,000 candidate games' positions
            W = {}
            def wpos(g):
                if g not in W: W[g] = nz(P.vec[P.off[base + gm(g)]:P.off[base + gm(g) + 1]].astype(np.float32) / sd)
                return W[g]
            for tag in TAGS:
                z = np.load(DUMPS / f"t2{tag}_{s}.npz"); lib, qrows, rep = z["lib"], z["qrows"], z["rep"]; Xn = np.zeros((len(rep), NC), np.float32)
                for r in range(lib.shape[0]):
                    segs = [np.concatenate([wpos(row[x]) for x in lib[r, c]]) for c in range(NC)]; Lv = np.concatenate(segs); lens = np.array([len(q) for q in segs])
                    m = np.flatnonzero(rep == r); qr = qrows[m]; ug = np.unique(qr[qr >= 0]); slot_of = {g: j for j, g in enumerate(ug)}
                    parts = [wpos(row[g]) for g in ug]; Qv = np.concatenate(parts); Qslot = np.concatenate([np.full(len(p), j, np.int64) for j, p in enumerate(parts)])
                    sums, cnt = kde_game_sums(Qv, Qslot, len(ug), Lv, lens, BETA, dev=dev)
                    for i, q in zip(m, qr):
                        js = [slot_of[g] for g in q if g >= 0]; Xn[i] = zrow((sums[js].sum(0) / cnt[js].sum())[None])[0]
                res[(tag, s)].append(Xn); print(f"  {net} {s} struct {STRUCT[tag]} done {time.time() - t0:.0f}s", flush=True)
    for (tag, s), L in res.items():
        X = np.mean(L, 0).astype(np.float32); np.savez(OUT / f"t2_X{sfx}_{tag}_{s}.npz", X=X, nets=np.array(NETS)); print("wrote", f"t2_X{sfx}_{tag}_{s}.npz", X.shape, "row-mean/std", float(X.mean()), float(X.std()))


# ---------------------------------------------------------------------------------------------------- scoring (CPU)
def scorer():
    from e81_lib import sc, csls_qq, rank_after
    return sc, csls_qq, rank_after


def per_query(tag, s, w, Xfile=None):
    """returns (current per-query score [8000], combined per-query score [8000] or None, truth [8000]).  Reps are the transduction units for CSLS (as E110)."""
    sc, csls_qq, rank_after = scorer(); z = np.load(DUMPS / f"t2{tag}_{s}.npz"); rep, e, top, truth = z["rep"], z["e"], z["top"].astype(np.int64), z["truth"]; X = np.load(Xfile)["X"] if Xfile else None
    cur = np.zeros(len(rep)); new = np.zeros(len(rep)) if w is not None else None
    for r in sorted(set(rep)):
        m = rep == r; rk, _ = rank_after(csls_qq(e[m], top[m], NC, K, ALPHA, "z"), top[m], truth[m]); cur[m] = sc(rk)
        if w is not None:
            zs = (e[m] - e[m].mean(1, keepdims=True)) / (e[m].std(1, keepdims=True) + 1e-9); Xs = np.take_along_axis(X[m], top[m], 1); c = zs + w * Xs
            rk, _ = rank_after(csls_qq(c, top[m], NC, K, ALPHA, "z"), top[m], truth[m]); new[m] = sc(rk)
    return cur, new, truth


def player_level(delta_by_w, truth):
    """delta_by_w {w: per-query delta}; -> players, per-player mean delta per w, cv-stitched per-player delta, selected w per fold, all-player selected w."""
    ups = np.unique(truth); P = {w: np.array([d[truth == u].mean() for u in ups]) for w, d in delta_by_w.items()}; n = len(ups)
    half = np.random.default_rng(0).permutation(n); h1, h2 = half[: n // 2], half[n // 2:]
    w1 = max(WS, key=lambda w: P[w][h1].mean()); w2 = max(WS, key=lambda w: P[w][h2].mean()); d = np.zeros(n); d[h2] = P[w1][h2]; d[h1] = P[w2][h1]
    wall = max(WS, key=lambda w: P[w].mean()); return ups, P, d, (w1, w2), wall


def ci(d):
    n = len(d); Bt = np.random.default_rng(0).integers(0, n, (2000, n)); bs = d[Bt].mean(1); return dict(point=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n_players=int(n))


def evaluate(xprefix):
    """full pipeline on X files named t2_X{xprefix}_{tag}_{pool}.npz."""
    R = {}
    for tag in TAGS:
        for s, sp, _ in POOLS:
            xf = OUT / f"t2_X{xprefix}_{tag}_{s}.npz"; dl, cur = {}, None
            for w in WS: cur, new, truth = per_query(tag, s, w, xf); dl[w] = new - cur
            ups, P, d, wsel, wall = player_level(dl, truth)
            R[f"{STRUCT[tag]}/{sp}"] = dict(current_mean=float(cur.mean()), delta_cv=ci(d), w_sel_folds=list(wsel), w_all_players=wall, per_w={str(w): ci(P[w]) for w in WS},
                                              mean_with_x_by_w={str(w): float((cur + dl[w]).mean()) for w in WS})
    return R


def fmt(R):
    out = []
    for k, v in R.items():
        c = v["delta_cv"]; out.append(f"  {k:16s} current(3,.5) {v['current_mean']:.4f}  delta_cv {c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] (players {c['n_players']})  w folds {v['w_sel_folds']} all {v['w_all_players']}  per w " +
                   " ".join(f"{w}: {p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]" for w, p in v["per_w"].items()))
    return "\n".join(out)


def repro(a):
    ref = json.loads((ROOT / ".scratch/e110/result.json").read_text())["t2"]; ok = True
    for tag in TAGS:
        for s, sp, _ in POOLS:
            cur, _, _ = per_query(tag, s, None); want = ref[STRUCT[tag]][sp.capitalize()]["k3"]; d = abs(cur.mean() - want); ok &= d < 1e-9
            print(f"  struct {STRUCT[tag]} {sp:8s} (3,.5) mean {cur.mean():.6f}  E110 result.json {want:.6f}  |diff| {d:.2e}")
    print("REPRO", "OK" if ok else "MISMATCH"); sys.exit(0 if ok else 1)


def dry(a):
    sc, csls_qq, rank_after = scorer()
    print("w = 0 identity (real X files; combined score with w=0 vs the current (3,.5) path, per query, all structures / pools):")
    for tag in TAGS:
        for s, sp, _ in POOLS:
            cur, new, _ = per_query(tag, s, 0.0, OUT / f"t2_X_{tag}_{s}.npz"); z = np.load(DUMPS / f"t2{tag}_{s}.npz"); rep, e, top = z["rep"], z["e"], z["top"].astype(np.int64); same_rank = True; maxd = 0.0
            for r in sorted(set(rep)):
                m = rep == r; zs = (e[m] - e[m].mean(1, keepdims=True)) / (e[m].std(1, keepdims=True) + 1e-9); A = csls_qq(e[m], top[m], NC, K, ALPHA, "z"); Bm = csls_qq(zs, top[m], NC, K, ALPHA, "z")
                same_rank &= bool((np.argsort(-A, 1, kind="stable") == np.argsort(-Bm, 1, kind="stable")).all()); maxd = max(maxd, float(np.abs(A - Bm).max()))
            print(f"  struct {STRUCT[tag]} {sp:8s} per-query scores identical: {bool((cur == new).all())}  shortlist order identical: {same_rank}  max|csls diff| {maxd:.2e}")
    sh = sorted(OUT.glob("t2_X_shuf*__pub.npz"))
    for f in sh:
        pre = f.name.split("_")[2]; print(f"shuffled channel ({pre}: pool-local random game permutation inside compute) through the full pipeline:"); print(fmt(evaluate(f"_{pre}")))


def formal(a):
    assert os.environ.get("E112_FORMAL") == "1", "formal scoring needs E112_FORMAL=1"
    outp = OUT / "result.json"; assert not outp.exists(), f"{outp} exists - formal scoring runs once"
    R = evaluate(""); pa, pv = R["a/public"]["delta_cv"], R["a/private"]["delta_cv"]; cc, cv = R["c/public"]["delta_cv"], R["c/private"]["delta_cv"]
    cond = {"1_a_public_point>=+.005": pa["point"] >= .005, "1_a_public_lo>0": pa["lo"] > 0, "1_a_private_point>=+.005": pv["point"] >= .005, "1_a_private_lo>0": pv["lo"] > 0,
            "2_c_public_point>=-.005": cc["point"] >= -.005, "2_c_private_point>=-.005": cv["point"] >= -.005}
    fails = [k for k, v in cond.items() if not v]; verdict = "E112 ADOPT" if not fails else "E112 STOP " + ", ".join(fails)
    res = dict(structs=R, conditions=cond, w_for_candidate=dict(public=R["a/public"]["w_all_players"], private=R["a/private"]["w_all_players"]), verdict=verdict,
               method="combined = z_row(e over shortlist) + w*X[shortlist]; csls_qq(3,.5,z) per rep; delta vs current (3,.5); player-level 2-fold CV of w in {.1,.25}; player bootstrap B=2000 seed 0")
    with open(outp, "x") as f: f.write(json.dumps(res, indent=1))
    print(fmt(R)); print("conditions:", json.dumps(cond)); print("w for candidate file (struct a, all players):", res["w_for_candidate"]); print(verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True); c = sp.add_parser("compute"); c.add_argument("--shuffle", type=int, default=None)
    for n in ("repro", "dry", "formal"): sp.add_parser(n)
    a = ap.parse_args(); {"compute": compute, "repro": repro, "dry": dry, "formal": formal}[a.cmd](a)
