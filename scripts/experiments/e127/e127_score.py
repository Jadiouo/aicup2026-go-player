"""E127 step 1 scoring (CPU, .venv python).  v7w = z_row(booster e)[shortlist] + .5*X_e112 + 1*X_lr -> csls_qq(3,.5,z)   (arithmetic of e125_score.Ctx.combo); + w*X_new before the same csls_qq.
Pools Public / Private never mixed.  Selection (w, checkpoint/variant) uses ONLY Public structure-a reps 0-9; judgement uses reps 10-14 (Public and Private each with their own pool), player-level bootstrap (players = true candidates), B=2000 seed 0.
subcommands:
  xlr     build + cache X_lr (E123 code) for reps 0-14, both pools, tags '' and c     -> .scratch/e127/Xlr_{tag}_{pool}.npz
  select  --names a,b,..  --ws .25,.5,...     Public a reps 0-9: mean delta vs v7w per (name,w) -> prints; writes .scratch/e127/select_{tag}.json
  judge   --name N --w W [--shuf name_shuf1,name_shuf2]   reps 10-14, all pool x struct cells, delta + CI, shuffle control
  repro   v7w reproduction for reps 0-14 (a/pub vs E123 formal_perquery on the same reps)"""
import sys, os, json, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "scripts/experiments/e123")]
import numpy as np
import e123_lib as L
from e123_lib import NC, DUMPS, S
OUT = ROOT / ".scratch/e127"; POOLS = (("pub", "public"), ("priv", "private")); TAGS = ("", "c"); STRUCT = {"": "a", "c": "c", "b": "b"}; ALLREPS = list(range(15))
sc, csls_qq, rank_after = S.scorer()


def zrow(S_): return (S_ - S_.mean(1, keepdims=True)) / (S_.std(1, keepdims=True) + 1e-9)


class Cell:
    def __init__(self, tag, s, sp):
        self.tag, self.s, self.sp = tag, s, sp; self.z = np.load(DUMPS / f"t2{tag}_{s}.npz"); self.Xe = np.load(ROOT / f".scratch/e112/t2_X_{tag}_{s}.npz")["X"]; f = OUT / f"Xlr_{tag}_{s}.npz"
        self.Xlr = np.load(f)["X"] if f.exists() else None; self.m = {r: np.flatnonzero(self.z["rep"] == r) for r in range(20)}; self.truth = self.z["truth"]

    def combo(self, r, terms):
        m = self.m[r]; z = self.z; e = z["e"][m]; top = z["top"][m].astype(np.int64); zs = zrow(e); c = zs + .5 * np.take_along_axis(self.Xe[m], top, 1) + 1.0 * np.take_along_axis(self.Xlr[m], top, 1)
        for w, X in terms: c = c + w * np.take_along_axis(X[m], top, 1)
        rk, ts = rank_after(csls_qq(c, top, NC, S.K, S.ALPHA, "z"), top, self.truth[m]); return sc(rk)


def ci_players(delta, truth, B=2000):
    ups = np.unique(truth); pi = np.searchsorted(ups, truth); pm = np.bincount(pi, delta) / np.bincount(pi); n = len(pm); bs = pm[np.random.default_rng(0).integers(0, n, (B, n))].mean(1)
    return dict(point=float(pm.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n_players=int(n), n_queries=int(len(delta)))


def cmd_xlr(a):
    for s, sp in POOLS:
        v = L.load_vecs(s); row = L.check_rows(s, sp)
        for tag in TAGS:
            f = OUT / f"Xlr_{tag}_{s}.npz"
            if f.exists(): continue
            t0 = time.time(); X, it = L.build_X(tag, s, sp, v, row, "head", reps=ALLREPS); np.savez(f, X=X, iters=np.array(it)); print("wrote", f.name, f"{time.time() - t0:.0f}s iters {min(it)}-{max(it)}", flush=True)


def cmd_repro(a):
    ref = np.load(ROOT / ".scratch/e123/formal_perquery.npz"); c = Cell("", "pub", "public"); print("keys", [k for k in ref.files][:12])
    for r in ALLREPS:
        v7w = c.combo(r, []); m = c.m[r]; print(f"rep {r}: v7w {v7w.mean():.6f}  E123 head_new {ref['a/pub/head_new'][m].mean():.6f}  maxdiff {np.abs(v7w - ref['a/pub/head_new'][m]).max():.2e}  | pub a v7u-on-rep E123 cur {ref['a/pub/cur'][m].mean():.6f}")


def load_X(name, tag, s):
    name, _, sfx = name.partition("|"); f = OUT / "X" / f"X_{name}_{tag}_{s}{'_' + sfx if sfx else ''}.npz"; return np.load(f)["X"]      # "name|shuf1" = shuffled-label control file


def cmd_select(a):
    names = a.names.split(","); ws = [float(x) for x in a.ws.split(",")]; tg = a.struct; c = Cell(tg, "pub", "public"); res = {}
    base = {r: c.combo(r, []) for r in range(10)}
    print("v7w Public a reps 0-9 mean", np.mean([base[r].mean() for r in range(10)]))
    for nm in names:
        X = load_X(nm, tg, "pub")
        for w in ws:
            d = np.concatenate([c.combo(r, [(w, X)]) - base[r] for r in range(10)]); tr = np.concatenate([c.truth[c.m[r]] for r in range(10)]); ci = ci_players(d, tr); res[f"{nm}|{w}"] = ci; print(f"  {nm:24s} w {w:5.2f}  delta {ci['point']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]", flush=True)
    (OUT / f"select_{a.tag}.json").write_text(json.dumps(res, indent=1))


def cmd_judge(a):
    out = {}; w = a.w; shuf = [x for x in a.shuf.split(",") if x]; reps = list(range(10, 15))
    for s, sp in POOLS:
        for tag in TAGS:
            c = Cell(tag, s, sp); L.check_players(c.z, sp); [L.assert_rep(c.z, r) for r in reps]; X = load_X(a.name, tag, s); base = {r: c.combo(r, []) for r in reps}; d = np.concatenate([c.combo(r, [(w, X)]) - base[r] for r in reps]); tr = np.concatenate([c.truth[c.m[r]] for r in reps])
            n = np.concatenate([c.z["n"][c.m[r]] for r in reps]); k = f"{STRUCT[tag]}/{sp}"; out[k] = dict(v7w_mean=float(np.concatenate([base[r] for r in reps]).mean()), delta=ci_players(d, tr), by_n={str(i): float(d[n == i].mean()) for i in (1, 2, 3, 4) if (n == i).any()}, per_rep=[float((c.combo(r, [(w, X)]) - base[r]).mean()) for r in reps])
            for sn in shuf:
                Xs = load_X(sn, tag, s); ds = np.concatenate([c.combo(r, [(w, Xs)]) - base[r] for r in reps]); out[k].setdefault("shuffle", {})[sn] = ci_players(ds, tr)
            o = out[k]; print(f"{k:16s} v7w {o['v7w_mean']:.4f}  delta {o['delta']['point']:+.4f} [{o['delta']['lo']:+.4f},{o['delta']['hi']:+.4f}]  by_n {o['by_n']}  per_rep {[round(x, 4) for x in o['per_rep']]}  shuffle " + " ".join(f"{sn}: {v['point']:+.4f}" for sn, v in o.get("shuffle", {}).items()), flush=True)
    (OUT / f"judge_{a.name}_w{w}.json").write_text(json.dumps(out, indent=1))


def cmd_ens(a):
    names = a.names.split(","); tags = TAGS
    for s_, sp in POOLS:
        for tag in tags:
            for sfx in ("",) + tuple(f"_shuf{k}" for k in a.shuf_ids.split(",") if k):
                fs = [OUT / "X" / f"X_{n}_{tag}_{s_}{sfx}.npz" for n in names]
                if not all(f.exists() for f in fs): continue
                X = np.mean([np.load(f)["X"] for f in fs], 0).astype(np.float32); np.savez(OUT / "X" / f"X_{a.out}_{tag}_{s_}{sfx}.npz", X=X); print("wrote", a.out, tag, s_, sfx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True); sp.add_parser("xlr"); sp.add_parser("repro")
    s = sp.add_parser("select"); s.add_argument("--names", required=True); s.add_argument("--ws", default=".25,.5,1"); s.add_argument("--tag", default="x"); s.add_argument("--struct", default="")
    e = sp.add_parser("ens"); e.add_argument("--names", required=True); e.add_argument("--out", required=True); e.add_argument("--shuf_ids", default="1,2")
    j = sp.add_parser("judge"); j.add_argument("--name", required=True); j.add_argument("--w", type=float, required=True); j.add_argument("--shuf", default="")
    a = ap.parse_args(); {"xlr": cmd_xlr, "repro": cmd_repro, "select": cmd_select, "judge": cmd_judge, "ens": cmd_ens}[a.cmd](a)
