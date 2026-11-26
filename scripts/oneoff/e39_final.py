"""E39 production port (v7e): LHx Player reranker whose metrics (L0 LDA + HSL LDA) are fitted on ALL games of ALL non-H players (kappa-weighted between-player scatter), in the FINAL Z space.
Booster training pools: same non-H players as v7d (>=30 total games, <=24 games/colour, >=28 rows), two cross-fit halves (same split as e38_final); metric for pools of half h = fitted on the other half
(all games) + every non-eligible non-H player; test-time metric = all non-H players.  H players: ids used only to EXCLUDE them (no labels, no ledger).  CPU only.
  stats  e39_final.py stats                      -> .scratch/e39/final_stats.npz (streaming Sw / player means per group)
  train  e39_final.py train --name v7e [--kappa 300] [--draws 60 --rounds 600 --seed 8]   -> .scratch/e38/final/NAME.{txt,npz,json}
Predict with e38_final.py predict --model v7e (same artifact format).
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
import e38_final as EF
import e39_run as ER
F, S, S3 = EF.F, EF.S, EF.S3
import numpy as np, pandas as pd, lightgbm as lgb
from gogame import player_rerank as prr
OUT = ROOT / ".scratch" / "e39"


def groups_final(D, meta, Hset):
    """TA/TB = halves of D.eligible (as e38_final), X = every other non-H player."""
    tr = sorted(D.eligible["train"])
    perm = np.random.default_rng(99).permutation(len(tr))
    TA, TB = set(np.array(tr)[perm[: len(perm) // 2]]), set(np.array(tr)[perm[len(perm) // 2:]])
    X = set(meta.player_id) - Hset - TA - TB
    return TA, TB, X


def stage_stats(a):
    from gogame.aggregate import L0Store
    from gogame.gametable import load_table
    t0 = time.perf_counter()
    P, pm = F.load_pm()
    D = EF.build_final_space(pm)
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    Hset = set(S3.R.sets()["H"])
    TA, TB, X = groups_final(D, meta, Hset)
    pid = meta["player_id"].to_numpy()
    assert not ((TA | TB | X) & Hset) and not (TA & TB) and not (TA & X) and not (TB & X)
    table = load_table(ROOT / "data/cache/l0"); assert list(table.uids) == meta["uid"].tolist()
    store = L0Store.from_table(table)
    d = len(pm.s["mu"]) + len(pm.s["cols"])
    res = {}
    for g, Sg in (("TA", TA), ("TB", TB), ("X", X)):
        rows_g = np.flatnonzero(np.isin(pid, list(Sg)))
        rows_g = rows_g[np.argsort(pid[rows_g], kind="stable")]
        pg = pid[rows_g]; bounds = np.flatnonzero(np.r_[True, pg[1:] != pg[:-1], True])
        chunks, cs = [], bounds[0]
        for a_, b_ in zip(bounds[:-1], bounds[1:]):
            if b_ - cs > 12000 and a_ > cs:
                chunks.append((cs, a_)); cs = a_
        chunks.append((cs, bounds[-1]))
        acc = dict(n=0, sx=np.zeros(d), sx2=np.zeros(d), C=0, Sw=np.zeros((d, d))); pm_l, pn_l = [], []
        for (a_, b_) in chunks:
            r = rows_g[a_:b_]; Z = pm._z(store, r).astype(np.float64); pk = pg[a_:b_]
            st_ = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]]); en = np.r_[st_[1:], len(pk)]; n_p = (en - st_).astype(np.float64)
            m = np.add.reduceat(Z, st_, axis=0) / n_p[:, None]
            Xc = Z - np.repeat(m, en - st_, axis=0)
            acc["n"] += len(Z); acc["sx"] += Z.sum(0); acc["sx2"] += (Z ** 2).sum(0); acc["C"] += len(m); acc["Sw"] += Xc.T @ Xc
            pm_l.append(m); pn_l.append(n_p)
        for k_, v in acc.items():
            res[f"{g}|all|{k_}"] = np.asarray(v)
        res[f"{g}|all|pm"] = np.concatenate(pm_l); res[f"{g}|all|pn"] = np.concatenate(pn_l)
        print(f"group {g}: players {len(Sg)} rows {acc['n']} {time.perf_counter() - t0:.0f}s", flush=True)
    np.savez(OUT / "final_stats.npz", **res)
    json.dump({"TA": sorted(map(str, TA)), "TB": sorted(map(str, TB)), "X": sorted(map(str, X))}, open(OUT / "final_groups.json", "w"))
    print("saved", len(res), flush=True)


def hsl_fit(Xh_all, pid_all, players, cap_rows=None):
    rows = np.flatnonzero(np.isin(pid_all, list(players)))
    return ER.hsl_metric(Xh_all, pid_all, rows), len(rows)


def stage_train(a):
    t0 = time.perf_counter()
    P, pm = F.load_pm()
    D = EF.build_final_space(pm)                       # capped (24/colour) rows of non-H players with >=30 games = booster-pool material (identical to v7d)
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    Hset = set(S3.R.sets()["H"])
    zz = np.load(OUT / "final_stats.npz"); ST = {k: zz[k] for k in zz.files}
    gj = json.load(open(OUT / "final_groups.json")); G = {k: set(v) for k, v in gj.items()}
    TA, TB, X = groups_final(D, meta, Hset)
    assert G["TA"] == TA and G["TB"] == TB, "group split differs from stats"
    pid_all = meta["player_id"].to_numpy()
    hh = pd.read_parquet(EF.HSL_PARQUET).set_index("uid").loc[meta["uid"].to_numpy()]
    cols = [c for c in hh.columns if c != "n_moves"]
    Xh_all = hh[cols].to_numpy(np.float64); del hh
    assert not (set(pid_all[np.isin(pid_all, list(TA | TB | X))]) & Hset)

    def metric(groups):
        L = ER.lda_from_stats(ST, groups, "all", kappa=a.kappa)
        Mh, nrows = hsl_fit(Xh_all, pid_all, set().union(*[G[x] for x in groups]))
        L.update(Mh); L["n_rows"] = nrows
        return L
    Xh_D = D.Xh                                        # raw HSL of the same capped rows as D.Z (built by EF.build_final_space)
    Mfull = metric(["TA", "TB", "X"])
    print(f"test-time metric: players {Mfull['C']} games {Mfull['N']} {time.perf_counter() - t0:.0f}s", flush=True)
    tr_players = sorted(D.eligible["train"]); half = {0: TA, 1: TB}
    Yx = {}
    for h in (0, 1):
        Mh = metric(["TB" if h == 0 else "TA", "X"])
        Yx[h] = EF.project(D.Z, Xh_D, Mh, True); print(f"half-metric {h}: players {Mh['C']} games {Mh['N']} {time.perf_counter() - t0:.0f}s", flush=True); del Mh
    rng = np.random.default_rng(a.seed)
    Xs_, ys_ = [], []
    for dr in range(a.draws):
        st = ("a", "b")[dr % 2]; h = dr % 4 // 2
        D.eligible["train"] = [p for p in tr_players if p in half[h]]
        pool = S.make_pool(D, "train", rng, st)
        Ylda, Yhsl = Yx[h]          # Yx[h] = metric fitted WITHOUT the players of half h
        f = EF.feats3(D.Z, D.sel, pool, a.k, Ylda, Yhsl, True)
        Xs_.append(f["X"]); ys_.append((f["top"] == pool["truth"][:, None]).astype(np.int8).reshape(-1))
        if dr % 10 == 0:
            print(f"  train pool {dr}: recall@{f['k']} {(f['top'] == pool['truth'][:, None]).any(1).mean():.3f}  {time.perf_counter() - t0:.0f}s", flush=True)
    D.eligible["train"] = tr_players
    del Yx
    Xm, y = np.concatenate(Xs_), np.concatenate(ys_); del Xs_
    kk = f["k"]
    bst = lgb.train(prr.lgb_params(a.seed, 4), lgb.Dataset(Xm, y, group=np.full(len(y) // kk, kk), free_raw_data=False), num_boost_round=a.rounds)
    print(f"booster trained {time.perf_counter() - t0:.0f}s, X {Xm.shape}", flush=True)
    EF.FIN.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(EF.FIN / f"{a.name}.txt"))
    Mo = dict(mu=Mfull["mu"], sd=Mfull["sd"], gm=Mfull["gm"], V=Mfull["V"], hmu=Mfull["hmu"], hsd=Mfull["hsd"], hgm=Mfull["hgm"], hV=Mfull["hV"])
    np.savez(EF.FIN / f"{a.name}.npz", hsl_cols=np.array(cols), **Mo)
    info = dict(k=kk, hsl=True, draws=a.draws, rounds=a.rounds, seed=a.seed, kappa=a.kappa, n_players=int(Mfull["C"]), n_fit_rows=int(Mfull["N"]), n_feat=int(Xm.shape[1]), n_booster_pool_players=len(tr_players), seconds=round(time.perf_counter() - t0))
    (EF.FIN / f"{a.name}.json").write_text(json.dumps(info)); print("saved", a.name, info, flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    p = sub.add_parser("train"); p.add_argument("--name", required=True); p.add_argument("--kappa", type=float, default=300.0); p.add_argument("--draws", type=int, default=60)
    p.add_argument("--rounds", type=int, default=600); p.add_argument("--k", type=int, default=40); p.add_argument("--seed", type=int, default=8)
    a = ap.parse_args()
    {"stats": stage_stats, "train": stage_train}[a.cmd](a)


if __name__ == "__main__":
    main()
