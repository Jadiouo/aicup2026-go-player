"""P1111 production: retrain the Player reranker on test2-like pools (library 10 games, colours iid B/W, N in 1/2/3/5, 600-pool, K shortlist) with the FINAL
standardiser (data/models_final, include_holdout) and score the real test2.  Recipe = scripts/oneoff/p1111_sim.py (validated offline, see docs/experiments/p1111_test2_sim.md).
Training players: every player with >=28 games in the cache INCLUDING H (final model, no evaluation here; the sim used non-H only).  Not used: any H label, h_eval_ledger.
  train    p1111_final.py train --k 40 [--hsl] --draws 60 --rounds 600     -> .scratch/p1111/final/<name>.txt (+ hsl_metric.npz)
  predict  p1111_final.py predict --model NAME --data-dir .scratch/p1111/t2 --out FILE [--hsl-parquet PATH]   (ids already stripped of the 2 in pub2_/priv2_, see p1111_assemble.py)
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "oneoff")]
import p1111_sim as S           # sets sys.path, imports player_sim3 / E30 helpers
import numpy as np
import pandas as pd
import lightgbm as lgb
from gogame import player_rerank as prr

OUT = ROOT / ".scratch" / "p1111" / "final"
MD = ROOT / "data" / "models_final"
CAP, MIN_G = 24, 28
HSL_PARQUET = ROOT / "data" / "cache" / "e10" / "hsl_games.parquet"


class Obj: pass


def load_pm():
    sys.path[:0] = [str(ROOT / "scripts")]
    import predict as P
    return P, P.PLAYER_MODELS["e04v2"].load(MD, 4)


def build_train(pm, hsl):
    from gogame.aggregate import L0Store
    from gogame.gametable import load_table
    meta = pd.read_parquet(ROOT / "data/cache/games_meta.parquet")
    tot = meta.groupby("player_id").size()
    keep = set(tot.index[tot >= MIN_G])
    rng = np.random.default_rng(20261111)
    m = meta.assign(i=np.arange(len(meta)))
    m = m[m.player_id.isin(keep)]
    rows = []
    for (p, c), g in m.groupby(["player_id", "color"]):
        idx = g["i"].to_numpy(); rows.append(idx[rng.permutation(len(idx))[:CAP]])
    rows = np.sort(np.concatenate(rows))
    table = load_table(ROOT / "data/cache/l0"); assert list(table.uids) == meta["uid"].tolist()
    store = L0Store.from_table(table)
    Z = np.empty((len(rows), len(pm.s["mu"]) + len(pm.s["cols"])), np.float32)
    for a in range(0, len(rows), 20000):
        Z[a:a + 20000] = pm._z(store, rows[a:a + 20000])
    mm = meta.iloc[rows].reset_index(drop=True); mm["grp"] = "train"
    del store, table
    D = Obj(); D.Z, D.meta, D.color, D.sel = Z, mm, mm["color"].to_numpy(), pm.s["sel"]
    D.rows = {"train": {p: np.flatnonzero((mm.player_id == p).to_numpy()) for p in sorted(mm.player_id.unique())}}
    D.eligible = {"train": [p for p, r in D.rows["train"].items() if len(r) >= MIN_G]}
    D.hsl_black_only = False; D.muc = None
    if hsl:
        h = pd.read_parquet(HSL_PARQUET).set_index("uid").loc[mm["uid"].to_numpy()]
        cols = [c for c in h.columns if c != "n_moves"]
        X = h[cols].to_numpy(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        gm, V, ev = S.S3.fit_metric(Xs, mm["player_id"].to_numpy(), ridge=1e-2, mmax=40)
        D.Y = ((Xs - gm) @ V).astype(np.float32)
        np.savez(OUT / "hsl_metric.npz", cols=np.array(cols), mu=mu, sd=sd, gm=gm, V=V)
        print("HSL LDA ratios", np.round(ev[:5], 3), flush=True)
    return D


def stage_train(a):
    OUT.mkdir(parents=True, exist_ok=True)
    P, pm = load_pm()
    t0 = time.perf_counter()
    D = build_train(pm, a.hsl)
    print(f"train rows {len(D.Z)} players {len(D.eligible['train'])} {time.perf_counter() - t0:.0f}s", flush=True)
    bst = S.train_booster(D, False, draws=a.draws, rounds=a.rounds, k=a.k, hsl=a.hsl, log=lambda *_: None)
    name = f"rt{'h' if a.hsl else ''}_k{a.k}"
    bst.save_model(str(OUT / f"{name}.txt"))
    (OUT / f"{name}.json").write_text(json.dumps(dict(k=a.k, hsl=a.hsl, draws=a.draws, rounds=a.rounds, n_rows=int(len(D.Z)), seconds=round(time.perf_counter() - t0))))
    print("saved", name, f"{time.perf_counter() - t0:.0f}s")


def stage_predict(a):
    P, pm = load_pm()
    import hashlib
    from p1111_decode import load_questions_any
    meta = json.loads((OUT / f"{a.model}.json").read_text()); k, hsl = meta["k"], meta["hsl"]
    bst = lgb.Booster(model_file=str(OUT / f"{a.model}.txt"))
    data = Path(a.data_dir)
    if hsl:
        mt = np.load(OUT / "hsl_metric.npz", allow_pickle=False)
        hp = pd.read_parquet(a.hsl_parquet).set_index("uid")
    qids, tops = [], []
    for sp in ("pub", "priv"):
        pq = load_questions_any(P, data / P.FILES[f"player_{sp}"], sp)
        cands = P.load_candidates(data / P.FILES[f"cand_{sp}"], sp); pool = sorted(cands)
        z = np.load(Path(a.z_cache_dir) / f"Z_{sp}.npz", allow_pickle=False); uids, Zt = z["uids"].astype(str), z["Z"]
        rowof = {u: i for i, u in enumerate(uids)}
        gl = {u: g for p in pool for g in cands[p] for u in [g.uid]}
        gq = {g.uid: g for q in pq for g in q.games}
        D = Obj(); D.Z, D.sel, D.hsl_black_only = Zt, pm.s["sel"], False
        col = np.zeros(len(uids), int)
        for u, g in {**gl, **gq}.items():
            col[rowof[u]] = g.color
        D.color = col
        if hsl:
            hh = hp.reindex(uids)
            miss = hh[mt["cols"][0]].isna().to_numpy()
            print(f"[{sp}] HSL rows missing: {int(miss.sum())} of {len(uids)}", flush=True)
            Xs = (np.nan_to_num(hh[list(mt["cols"])].to_numpy(np.float64)) - mt["mu"]) / mt["sd"]
            Y = ((Xs - mt["gm"]) @ mt["V"]).astype(np.float32); Y[miss] = np.nan
            D.Y = Y
        rows = lambda us: np.array([rowof[u] for u in us])   # noqa: E731
        pl = dict(lib=[rows([g.uid for g in cands[p]]) for p in pool], q=[rows([g.uid for g in q.games]) for q in pq])
        f = S.feats(D, pl, False, k, hsl)
        e = bst.predict(f["X"], num_threads=4).reshape(len(pq), -1)
        t5 = S.E30.top5_orig(f["top"], e)
        qids += [q.qid for q in pq]; tops.append(np.array(pool, dtype=object)[t5])
        print(f"[{sp}] scored {len(pq)} questions, pool {len(pool)}", flush=True)
    dst = Path(a.out); dst.parent.mkdir(parents=True, exist_ok=True)
    P.write_player_submission(qids, np.vstack(tops), dst)
    import check_submission
    rc = check_submission.main([str(dst), "--task", "player", "--data-dir", str(data)])
    print("check_rc", rc, "sha256_12", hashlib.sha256(dst.read_bytes()).hexdigest()[:12])
    return rc


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--k", type=int, default=40); t.add_argument("--hsl", action="store_true")
    t.add_argument("--draws", type=int, default=60); t.add_argument("--rounds", type=int, default=600)
    p = sub.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--hsl-parquet"); p.add_argument("--z-cache-dir", default=str(ROOT / ".scratch/p1111/z_t2"))
    a = ap.parse_args()
    return stage_train(a) if a.cmd == "train" else stage_predict(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
