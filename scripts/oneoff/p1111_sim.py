"""P1111 offline simulation of the 11/11 test2 (Player) structure.  CPU only, no H players (prep builds eval = dev+tune, train = fit+other; H never loaded).
Structure (from data/contest_1111 inspect): pool 600 players x 10 games (colours iid B/W), 400 questions with N in {1,2,3,5} x100, query games mixed colours,
query games disjoint from the library.  Owners: 'a' = 400 distinct owners + 200 distractor candidates; 'b' = owners drawn with replacement (siblings).
Methods: M01 cosine; E05 booster (frozen, black/lib95/N5-20) with decoders A (independent) / B (v3 transductive) / D (E30 soft exclusion); colour-centred Z;
booster retrained on test2-like pools (train players only).  Usage:  p1111_sim.py run --seeds 1 2 --pools 4 --struct a b
"""
import argparse, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts" / "oneoff"), str(ROOT / "scripts" / "experiments" / "e23"), str(ROOT / "src"), str(ROOT)]
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import numpy as np
import pandas as pd
import lightgbm as lgb
import player_sim3 as S3
from gogame import player_rerank as prr
from scripts.experiments.e30 import common as E30
sys.argv = _argv

SIM = ROOT / ".scratch" / "p1111" / "sim"
NS = (1, 2, 3, 5)
GAMMA = json.loads((ROOT / "experiments/E30_player_robustness/tuned.json").read_text())["gamma_star"]


class Data:
    def __init__(self):
        self.Z = np.load(SIM / "Z.npy")
        self.meta = pd.read_parquet(SIM / "meta.parquet")
        self.sel = np.load(SIM / "sel.npy")
        m = self.meta
        self.color = m["color"].to_numpy()
        self.rows = {g: {p: np.flatnonzero((m.player_id == p).to_numpy()) for p in sorted(m.loc[m.grp == g, "player_id"].unique())} for g in ("eval", "train")}
        self.eligible = {g: [p for p, r in d.items() if len(r) >= 28] for g, d in self.rows.items()}
        tr = np.flatnonzero((m.grp == "train").to_numpy())
        self.muc = np.stack([self.Z[tr[self.color[tr] == c]].mean(0) for c in (0, 1)])
        self._Zc = None
        self._Y = None
        self.hsl_black_only = False

    @property
    def Y(self):
        """HSL LDA coordinates (40 dims) of every row, metric fitted on TRAIN players (both colours, 24/colour/player)."""
        if self._Y is None:
            f = SIM / "Y_hsl.npy"
            if f.exists():
                self._Y = np.load(f)
            else:
                h = pd.read_parquet(ROOT / "data/cache/e10/hsl_games.parquet").set_index("uid").loc[self.meta["uid"].to_numpy()]
                X = h.drop(columns=["n_moves"]).to_numpy(np.float64)
                tr = np.flatnonzero((self.meta.grp == "train").to_numpy())
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                Xs = (X - mu) / sd
                gm, V, ev = S3.fit_metric(Xs[tr], self.meta["player_id"].to_numpy()[tr], ridge=1e-2, mmax=40)
                self._Y = ((Xs - gm) @ V).astype(np.float32)
                np.save(f, self._Y); print("HSL metric top LDA ratios", np.round(ev[:6], 3), flush=True)
        return self._Y

    def Zc(self):
        if self._Zc is None:
            self._Zc = self.Z - self.muc[self.color]
        return self._Zc


def make_pool(D, grp, rng, struct, npool=600, nq=400, libn=10, ns=NS):
    pl = sorted(rng.choice(D.eligible[grp], size=npool, replace=False).tolist())
    if struct == "a":
        owners = list(rng.permutation(npool)[:nq])
    else:                                    # 'b': 400 queries, owners drawn with replacement from 300 active players (siblings, 200+ distractors)
        act = rng.permutation(npool)[:300]
        owners = list(rng.choice(act, size=nq, replace=True))
    Ns = np.array([ns[r % len(ns)] for r in rng.permutation(nq)])
    lib_rows, rest = [], []
    for p in pl:
        r = D.rows[grp][p][rng.permutation(len(D.rows[grp][p]))]
        lib_rows.append(np.sort(r[:libn])); rest.append(list(r[libn:]))
    q_rows, truth = [], []
    for o, n in zip(owners, Ns):
        if len(rest[o]) >= n:
            q_rows.append(np.array(rest[o][:n])); rest[o] = rest[o][n:]
        else:                                # exhausted: reuse non-library games
            allr = D.rows[grp][pl[o]]; pool_ = np.setdiff1d(allr, lib_rows[o])
            q_rows.append(rng.choice(pool_, size=n, replace=False))
        truth.append(o)
    return dict(pl=pl, lib=lib_rows, q=q_rows, truth=np.array(truth), n=Ns)


def zs(D, pool, centred):
    Z = D.Zc() if centred else D.Z
    return [Z[r] for r in pool["lib"]], [Z[r] for r in pool["q"]]


def feats(D, pool, cent, k, hsl):
    lib_z, q_z = zs(D, pool, cent)
    f = prr.candidate_features(lib_z, q_z, D.sel, k, prr.N_DENSE)
    if hsl:
        def getY(r):
            y = D.Y[r].astype(np.float64)
            if D.hsl_black_only:
                y[D.color[r] == 1] = np.nan
            return y
        Yl = [getY(r) for r in pool["lib"]]; Yq = [getY(r) for r in pool["q"]]
        ext = []
        for m in (5, 15, 40):
            d = np.take_along_axis(S3.dist_matrix(Yl, Yq, m), f["top"], 1)
            c = d - d.mean(1, keepdims=True)
            ext += [d, c, c / (d.std(1, keepdims=True) + 1e-9)]
        f["X"] = np.hstack([f["X"], np.stack(ext, 2).reshape(len(q_z) * k, -1).astype(np.float32)])
    return f


def score(t5, truth):
    s = np.zeros(len(truth))
    for k in range(5):
        s += np.where(t5[:, k] == truth, np.exp(-k), 0.0)
    return s


def run_methods(D, pool, boosters, centred_flags):
    """-> {method: per-query score}"""
    out = {}
    truth = pool["truth"]
    class DS: pass
    ds = DS(); ds.sel = D.sel
    for cent in centred_flags:
        lib_z, q_z = zs(D, pool, cent)
        tag = "ctr" if cent else "raw"
        fk = {}
        f = fk.setdefault((prr.K, False), feats(D, pool, cent, prr.K, False))
        out[f"M01_{tag}"] = score(f["top"][:, :5], truth)
        for bn, bst in boosters.items():
            if bn.endswith("_ctr") != cent:
                continue
            bst, kk, hs = bst if isinstance(bst, tuple) else (bst, prr.K, False)
            f = fk.setdefault((kk, hs), feats(D, pool, cent, kk, hs)); top = f["top"]
            base = bn[:-4] if bn.endswith("_ctr") else bn
            e = bst.predict(f["X"], num_threads=4).reshape(len(q_z), -1)
            out[f"{base}_{tag}_A"] = score(E30.top5_orig(top, e), truth)
            out[f"{base}_{tag}_D"] = score(E30.decode_D(top, e, len(lib_z), GAMMA), truth)
            if kk == prr.K and not hs:
                t5 = S3.trans_t5(bst, ds, lib_z, q_z, None, tau=0.9)[0]
                out[f"{base}_{tag}_B"] = score(t5, truth)
            # c_all statistic for calibration
            order = np.argsort(-e, axis=1, kind="stable")
            best = np.take_along_axis(top, order[:, :1], 1)[:, 0]
            coll = (np.bincount(best, minlength=len(lib_z))[best] > 1).mean()
            sm = lambda x: (lambda z: z / z.sum(1, keepdims=True))(np.exp(x - x.max(1, keepdims=True)))  # noqa: E731
            out.setdefault("_stat", {})[f"{base}_{tag}"] = (float(coll), float(sm(e).max(1).mean()))
    return out


def train_booster(D, centred, struct_mix=("a", "b"), draws=12, seed=7, rounds=400, k=prr.K, hsl=False, log=print):
    Xs, ys = [], []
    rng = np.random.default_rng(seed)
    for d in range(draws):
        pool = make_pool(D, "train", rng, struct_mix[d % len(struct_mix)])
        f = feats(D, pool, centred, k, hsl)
        Xs.append(f["X"]); ys.append((f["top"] == pool["truth"][:, None]).astype(np.int8).reshape(-1))
        log(f"  train pool {d}: recall@{k} {(f['top'] == pool['truth'][:, None]).any(1).mean():.3f}")
    X, y = np.concatenate(Xs), np.concatenate(ys)
    ds = lgb.Dataset(X, y, group=np.full(len(y) // k, k), free_raw_data=False)
    return lgb.train(prr.lgb_params(seed, 4), ds, num_boost_round=rounds)


def boot(diff_by_pool, rng, B=2000):
    """two-level bootstrap: resample pools, then queries inside; diff_by_pool: list of arrays."""
    P = len(diff_by_pool); out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, P, P)
        out[b] = np.mean([diff_by_pool[i][rng.integers(0, len(diff_by_pool[i]), len(diff_by_pool[i]))].mean() for i in idx])
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2]); ap.add_argument("--pools", type=int, default=4)
    ap.add_argument("--struct", nargs="+", default=["a", "b"]); ap.add_argument("--out", default=str(ROOT / ".scratch/p1111/sim/results.json"))
    ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--ks", type=int, nargs="+", default=[20]); ap.add_argument("--no-ctr", action="store_true"); ap.add_argument("--hsl", action="store_true"); ap.add_argument("--hsl-black-only", action="store_true"); ap.add_argument("--no-e05", action="store_true")
    a = ap.parse_args()
    D = Data(); D.hsl_black_only = a.hsl_black_only
    t0 = time.perf_counter()
    e05 = lgb.Booster(model_file=str(ROOT / ".scratch/e05/model.txt"))
    boosters = {} if a.no_e05 else {"E05": e05, "E05_ctr": e05}
    for k in a.ks:
        nm = "RT" if k == prr.K else f"RT{k}"
        boosters[nm] = (train_booster(D, False, draws=a.draws, rounds=a.rounds, k=k), k, False)
        if a.hsl:
            boosters["RTH" + nm[2:]] = (train_booster(D, False, draws=a.draws, rounds=a.rounds, k=k, hsl=True), k, True)
        if not a.no_ctr:
            boosters[nm + "_ctr"] = (train_booster(D, True, draws=a.draws, rounds=a.rounds, k=k), k, False)
    print(f"boosters ready {time.perf_counter() - t0:.0f}s", flush=True)
    res = {}
    for st in a.struct:
        for sd in a.seeds:
            for pi in range(a.pools):
                rng = np.random.default_rng(1000 * sd + 10 * pi + (1 if st == "a" else 2))
                pool = make_pool(D, "eval", rng, st)
                r = run_methods(D, pool, boosters, (False,) if a.no_ctr else (False, True))
                stat = r.pop("_stat")
                res[f"{st}|{sd}|{pi}"] = dict(scores={k: v.tolist() for k, v in r.items()}, n=pool["n"].tolist(), stat=stat)
                print(st, sd, pi, f"{time.perf_counter() - t0:.0f}s", {k: round(float(np.mean(v)), 3) for k, v in r.items() if k.endswith(("_A", "_ctr", "_raw"))}, flush=True)
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
