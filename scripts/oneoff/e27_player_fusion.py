"""E27: Player test-set candidate = v3 (E04v2 + transductive tau .9) with the SE+HSL fusion bonus validated in player_sim3.

Stages (argv[1]):
  v3      reproduce data/submissions/player_v3_*.csv row by row from the model alone (no new features) -> .scratch/e27/v3_repro.csv
  games   write the games that still need features: .scratch/e27/se_games.parquet (uid,color,sgf: all candidate+query games) and
          .scratch/e27/hsl_games_in.parquet (uid,color,sgf: sim rule 20 games/candidate (default_rng(20), per split, pool order) + all query games)
  fuse    needs .scratch/e27/se/se_test_games.parquet and .scratch/e27/hsl/hsl_test_games.parquet -> .scratch/player_sim3/player_v6_SE_HSL融合.csv
Fusion (identical to player_sim3.trans_t5/pool_D): e' = e_E04v2 + sum_s -0.5*lam_s*D_s with se1 lam .5 and hsl5(subsample 20) lam .25,
D = whitened-LDA Mahalanobis^2 / (1/Nq+1/nlib); metrics fitted on FIT players only (.scratch/player_sim3/metric_{se,hsl}.npz).
lambda source: argmax of .scratch/player_sim3/tune_combo_table.csv (tune pools, never test).
"""
import os, sys, json, hashlib
os.environ.setdefault("GOGAME_COLOR_SEMANTICS", "actual")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts" / "oneoff"), str(ROOT / "scripts" / "experiments" / "e23"), str(ROOT / "src")]
STAGE = sys.argv[1] if len(sys.argv) > 1 else ""
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import numpy as np, pandas as pd
import predict as P
from gogame import player_rerank as prr
from gogame.sgf import get_color_semantics
import player_sim3 as S3          # trans_t5, pool_D, SRC, dist_matrix (sys.argv reset => STAGE "" there)
sys.argv = _argv

DATA = ROOT / "data" / "contest_1104"
MD = ROOT / "data" / "models_final"
OUT = ROOT / ".scratch" / "e27"
SIM = ROOT / ".scratch" / "player_sim3"
V3 = Path(os.environ.get("V3_REF", ROOT / "data" / "submissions" / "player_v3.csv"))   # optional: the v3 submission (sha256 b51ae167...) to compare with
LAM = (("se1", 0.5), ("hsl5s20", 0.25))      # = eval_sub "se1@0.5+hsl5s20@0.25"
HSL_SUB, HSL_SEED = 20, 20                    # SRC["hsl5s20"] = ("hsl", 5, 20); rg = default_rng(20)
TAU = 0.9


def load_split(sp):
    pq = P.load_questions(DATA / P.FILES[f"player_{sp}"], sp, "player")
    cands = P.load_candidates(DATA / P.FILES[f"cand_{sp}"], sp)
    return pq, cands, sorted(cands)


def zmats(pm, sp, pq, cands, pool):
    cache = OUT / f"Z_{sp}.npz"
    games = [g for p in pool for g in cands[p]] + [g for q in pq for g in q.games]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        uids, Z = z["uids"].astype(str), z["Z"]
        assert list(uids) == [g.uid for g in games] or set(uids) == {g.uid for g in games}
    else:
        store = P.l0_store(games, 4)
        Z = np.empty((len(store.uids), len(pm.s["mu"]) + len(pm.s["cols"])), np.float32)
        for a in range(0, len(Z), 20000):
            Z[a:a + 20000] = pm._z(store, np.arange(a, min(a + 20000, len(Z))))
        uids = np.array(store.uids).astype(str)
        np.savez(cache, uids=uids, Z=Z)
    rowof = {u: i for i, u in enumerate(uids)}
    rows = lambda us: np.fromiter((rowof[u] for u in us), np.int64, len(us))   # noqa: E731
    lib_u = [np.array([g.uid for g in cands[p]]) for p in pool]
    q_u = [np.array([g.uid for g in q.games]) for q in pq]
    return [Z[rows(u)] for u in lib_u], [Z[rows(u)] for u in q_u], lib_u, q_u


class TestFeat:
    """same interface as player_sim3.Feat but backed by the test-game tables."""
    def __init__(self, se_path, hsl_path):
        se = pd.read_parquet(se_path)
        self.se_ix = pd.Index(se.uid); self.se_arr = se[["se_mean", "se_std"]].to_numpy(np.float64)
        m = np.load(SIM / "metric_se.npz"); self.m = {k: m[k] for k in m.files}
        h = pd.read_parquet(hsl_path)
        z = np.load(SIM / "metric_hsl.npz", allow_pickle=True)
        cols = [str(c) for c in z["cols"]]
        assert [c for c in h.columns if c not in ("uid", "n_moves")] == cols
        Xs = (h[cols].to_numpy(np.float64) - z["mu"]) / z["sd"]
        self.h_ix = pd.Index(h.uid); self.h_Y = (Xs - z["gm"]) @ z["V"]

    def se(self, uids, kind):
        ix = self.se_ix.get_indexer(uids)
        a = np.where((ix >= 0)[:, None], self.se_arr[np.maximum(ix, 0)], np.nan)
        return (a[:, :1] - self.m["g1"]) @ self.m["V1"] if kind == "se1" else (a - self.m["g2"]) @ self.m["V2"]

    def hsl(self, uids):
        ix = self.h_ix.get_indexer(uids)
        assert (ix >= 0).all(), "HSL missing for requested games"
        return self.h_Y[ix]


def decode(pm, sp, F=None):
    pq, cands, pool = load_split(sp)
    lib_z, q_z, lib_u, q_u = zmats(pm, sp, pq, cands, pool)
    B = None
    if F is not None:
        D = S3.pool_D(F, lib_u, q_u, [s for s, _ in LAM])
        B = sum(-0.5 * lam * D[s] for s, lam in LAM)
    class DS: pass
    ds = DS(); ds.sel = pm.s["sel"]
    assert prr.N_DENSE == len(pm.s["mu"])
    t5, top0, e0 = S3.trans_t5(pm.booster, ds, lib_z, q_z, B, tau=TAU)
    return [q.qid for q in pq], np.array(pool, dtype=object)[t5], (B, lib_u, q_u)


def write_and_check(qids, ids, path):
    P.write_player_submission(qids, ids, path)
    import check_submission
    return check_submission.main([str(path), "--task", "player", "--data-dir", str(DATA)])


def stage_v3():
    OUT.mkdir(parents=True, exist_ok=True)
    print("color_semantics:", get_color_semantics(), flush=True)
    pm = P.PLAYER_MODELS["e04v2"].load(MD, 4)
    print("model include_holdout:", pm.meta.get("include_holdout"), flush=True)
    qids, tops = [], []
    for sp in ("pub", "priv"):
        q, t, _ = decode(pm, sp, None); qids += q; tops.append(t)
    pp = OUT / "v3_repro.csv"
    rc = write_and_check(qids, np.vstack(tops), pp)
    a = pd.read_csv(pp)
    print("sha256 repro", hashlib.sha256(pp.read_bytes()).hexdigest()[:12], "(expected b51ae1671927 = v3)")
    if not V3.exists():
        print("no V3_REF file; compare the sha256 above with the expected one"); return
    b = pd.read_csv(V3)
    print("check rc", rc, "cols", list(a.columns), "== v3 cols", list(a.columns) == list(b.columns))
    print("sha256 repro", hashlib.sha256(pp.read_bytes()).hexdigest()[:12], "v3", hashlib.sha256(V3.read_bytes()).hexdigest()[:12])
    m = a.merge(b, on=a.columns[0], suffixes=("_n", "_v"))
    nd = int((m.filter(like="_n").to_numpy() != m.filter(like="_v").to_numpy()).any(1).sum())
    print("rows", len(a), len(b), "merged", len(m), "rows differing in any of top1..5:", nd)
    print("IDENTICAL" if nd == 0 and len(a) == len(b) == len(m) else "DIFFERENT")


def stage_games():
    OUT.mkdir(parents=True, exist_ok=True)
    se_rows, hsl_rows, stat = [], [], {}
    for sp in ("pub", "priv"):
        pq, cands, pool = load_split(sp)
        rg = np.random.default_rng(HSL_SEED)
        nlib = 0
        for p in pool:
            gs = cands[p]
            keep = np.sort(rg.permutation(len(gs))[:HSL_SUB])
            hsl_rows += [(gs[i].uid, gs[i].color, gs[i].sgf) for i in keep]
            se_rows += [(g.uid, g.color, g.sgf) for g in gs]
            nlib += len(gs)
        nq = 0
        for q in pq:
            for g in q.games:
                se_rows.append((g.uid, g.color, g.sgf)); hsl_rows.append((g.uid, g.color, g.sgf)); nq += 1
        stat[sp] = dict(cands=len(pool), lib_games=nlib, lib_per_cand=nlib / len(pool), query_games=nq, nq=len(pq))
    se = pd.DataFrame(se_rows, columns=["uid", "color", "sgf"]); hs = pd.DataFrame(hsl_rows, columns=["uid", "color", "sgf"])
    for d in (se, hs):
        d["color"] = d["color"].astype(np.int8)
        assert not d.uid.duplicated().any()
    print(stat); print("SE games", len(se), "HSL games", len(hs), "colors", se.color.value_counts().to_dict())
    se.to_parquet(OUT / "se_games.parquet", index=False); hs.to_parquet(OUT / "hsl_games_in.parquet", index=False)
    (OUT / "games_stat.json").write_text(json.dumps(stat, indent=1))


def stage_fuse():
    pm = P.PLAYER_MODELS["e04v2"].load(MD, 4)
    tdir = Path(_argv[2]) if len(_argv) > 2 else OUT          # tables dir: <dir>/se/se_test_games.parquet, <dir>/hsl/hsl_test_games.parquet
    F = TestFeat(tdir / "se" / "se_test_games.parquet", tdir / "hsl" / "hsl_test_games.parquet")
    qids, tops, Bs = [], [], {}
    for sp in ("pub", "priv"):
        q, t, (B, _, _) = decode(pm, sp, F); qids += q; tops.append(t)
    new = np.vstack(tops)
    dst = Path(_argv[3]) if len(_argv) > 3 else SIM / "player_v6_SE_HSL融合.csv"
    rc = write_and_check(qids, new, dst)
    print("sha256", hashlib.sha256(dst.read_bytes()).hexdigest()[:12], "(expected b4eed172f033 = old_v6_800.csv)")
    if not V3.exists():
        print("no V3_REF file; skipping the comparison with v3"); return
    a, b = pd.read_csv(dst), pd.read_csv(V3)
    m = a.merge(b, on=a.columns[0], suffixes=("_n", "_v"))
    c1n, c1v = a.columns[1], b.columns[1]
    nn = pd.read_csv(DATA / "player_identification_test_public.csv", usecols=["question_id", "num_games"])
    nn2 = pd.read_csv(DATA / "player_identification_test_private.csv", usecols=["question_id", "num_games"])
    nmap = dict(zip(pd.concat([nn, nn2]).question_id.map(lambda s: s), pd.concat([nn, nn2]).num_games))
    # question ids in the submission are normalised (pub_q_0001); the raw test file uses the same
    m["N"] = m[a.columns[0]].map(nmap)
    top1d = m[c1n + "_n"] != m[c1v + "_v"]
    anyd = (m.filter(like="_n").to_numpy() != m.filter(like="_v").to_numpy()).any(1)
    res = {"check_rc": rc, "n": len(m), "top1_agree": float(1 - top1d.mean()), "top1_changed": int(top1d.sum()), "any_top5_changed": int(anyd.sum()),
           "by_N": {int(k): {"n": int((m.N == k).sum()), "top1_changed": int(top1d[m.N == k].sum()), "any_changed": int(anyd[m.N == k].sum())} for k in sorted(m.N.dropna().unique())}}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    (OUT / ("fuse_summary.json" if len(_argv) <= 3 else "fuse_summary_dry.json")).write_text(json.dumps(res, indent=1))
    print("sha256", hashlib.sha256(dst.read_bytes()).hexdigest()[:12])


if __name__ == "__main__":
    {"v3": stage_v3, "games": stage_games, "fuse": stage_fuse}[STAGE]()
