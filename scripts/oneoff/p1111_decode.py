#!/usr/bin/env python
"""P1111 steps 4-5: merge SE/HSL features (reused 11/04 rows + newly computed rows) and write a Player submission.

  merge   .venv/bin/python scripts/oneoff/p1111_decode.py merge  [--se-out DIR_with_c*/] [--hsl-new data/test_scan_p1111/hsl_test_games.parquet]
          -> .scratch/p1111/feat/{se_test_games,hsl_test_games}.parquet keyed by the NEW uids (needs .scratch/p1111/plan from p1111_inspect.py); validates coverage.
  decode  .venv/bin/python scripts/oneoff/p1111_decode.py decode --data-dir D --out FILE [--decoder B|D|auto] [--features none|se|se+hsl] [--compare REF.csv]
          D = canonical-named data dir (the original one, or .scratch/p1111/normalized written by p1111_inspect.py).

Decoders: A = independent per-question ranking of the first pass (added 11/11 for test2, where one player can match several questions).
Decoders (per split = per candidate file, E30 section 5):  B = v3 transductive tau .9 (player_sim3.trans_t5, as in E27);  D = E30 soft exclusion (gamma 2) on first-pass scores;
auto = exc statistic of the first pass WITHOUT fusion vs theta of experiments/E30_player_robustness/tuned.json: exc > theta -> D, else B.
Features: none = v3/E04v2 only;  se = + SE bonus (lambda se1 .5);  se+hsl = + SE and HSL bonus (E27: se1 .5, hsl5s20 .25).  Bonus B[q,c] = -0.5 * sum lam * D (E27/player_sim3).
White questions: blocked by io.py Q_NOT_BLACK unless --allow-white (local loader, src untouched; docs/runbook_1111_player.md section 7): then SE/HSL fusion is applied to black questions only.  Never touches src/, data/submissions, E27/E30 files.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

os.environ.setdefault("GOGAME_COLOR_SEMANTICS", "actual")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts" / "oneoff"), str(ROOT / "scripts" / "experiments" / "e23"), str(ROOT / "src"), str(ROOT)]
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]               # player_sim / player_sim3 read argv at import
import numpy as np
import pandas as pd

OUT = ROOT / ".scratch" / "p1111"
PLAN, FEAT = OUT / "plan", OUT / "feat"
OLD_SE = ROOT / ".scratch" / "e27" / "se" / "se_test_games.parquet"
OLD_HSL = ROOT / ".scratch" / "e27" / "hsl" / "hsl_test_games.parquet"
SIM = ROOT / ".scratch" / "player_sim3"
TUNED = ROOT / "experiments" / "E30_player_robustness" / "tuned.json"
MD = ROOT / "data" / "models_final"
SE_SHA = "95fc295137b1fd09a0ec4d5cc79ed888f8c12d0ba6a77fea58b24efc3f418126"
TAU = 0.9


# ----------------------------------------------------------------------------- merge
def stage_merge(a):
    FEAT.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(PLAN / "games_new.parquet")
    # ---- SE
    rs = pd.read_parquet(PLAN / "reuse_se.parquet")
    parts = []
    if len(rs):
        old = pd.read_parquet(OLD_SE).set_index("uid")
        o = old.loc[rs.old_uid.to_numpy()].reset_index(drop=True)
        o.insert(0, "uid", rs.new_uid.to_numpy()); parts.append(o)
    if a.se_out:
        dirs = sorted(Path(a.se_out).glob("c*"))
        assert dirs, f"no c*/ chunk directories in {a.se_out}"
        mans = []
        for d in dirs:
            m = json.loads((d / "se_manifest.json").read_text()); mans.append(m)
            assert m["color_semantics"] == "actual" and m["device"] == "cuda" and m["half"] and m["n_nan_not_hd_ge2"] == 0 and m["n_se_n_lt32"] == 0, (d, m)
            assert m["model_sha256"] == SE_SHA, (d, m["model_sha256"])
            parts.append(pd.read_parquet(d / "se_test_games.parquet"))
        assert len({m["torch"] for m in mans}) == 1
    se = pd.concat(parts, ignore_index=True)
    assert not se.uid.duplicated().any(), "duplicate SE uids"
    need = set(games[games.color == "B"].uid)          # features are computed for black games only (runbook section 7)
    miss = need - set(se.uid)
    assert not miss, f"SE missing for {len(miss)} games (e.g. {sorted(miss)[:3]}): the todo chunks are not all done"
    se = se[se.uid.isin(need)]
    assert not se[["se_mean", "se_std"]].isna().any().any() and (se.se_n == 32).all()
    se.to_parquet(FEAT / "se_test_games.parquet", index=False)
    print(f"SE merged: {len(se)} games (reused {len(rs)}, new {len(se) - len(rs)})")
    # ---- HSL
    rh = pd.read_parquet(PLAN / "reuse_hsl.parquet")
    hp = []
    if len(rh):
        oh = pd.read_parquet(OLD_HSL).set_index("uid")
        o = oh.loc[rh.old_uid.to_numpy()].reset_index(drop=True)
        o.insert(0, "uid", rh.new_uid.to_numpy()); hp.append(o)
    todo = pd.read_parquet(PLAN / "hsl_todo.parquet", columns=["uid"])
    if len(todo):
        assert a.hsl_new, "hsl_todo is not empty: pass --hsl-new data/test_scan_p1111/hsl_test_games.parquet (output of predict_hsl.py features)"
        man = json.loads((Path(a.hsl_new).parent / "features_manifest.json").read_text())
        assert man["color_semantics"] == "actual" and man["complete"] and man["config_hash"].startswith("d6811b41b13b5d88"), man
        n = pd.read_parquet(a.hsl_new)
        hp.append(n)
    if hp:
        cols = None
        for p in hp:
            cols = cols or list(p.columns)
            assert list(p.columns) == cols, "HSL column sets differ between reused and new tables"
        h = pd.concat(hp, ignore_index=True)
        assert not h.uid.duplicated().any()
        selset = set(u for v in json.loads((PLAN / "hsl_lib_sel.json").read_text()).values() for u in v)
        q_uids = set(games[(games.kind == "q") & (games.color == "B")].uid)
        miss = (q_uids | selset) - set(h.uid)
        assert not miss, f"HSL missing for {len(miss)} games (e.g. {sorted(miss)[:3]})"
        h = h[h.uid.isin(q_uids | selset)]
        h.to_parquet(FEAT / "hsl_test_games.parquet", index=False)
        print(f"HSL merged: {len(h)} games (reused {len(rh)}, new {len(h) - len(rh)}), library games with HSL {len(selset)}")


# ----------------------------------------------------------------------------- decode
class Feats:
    """E27 TestFeat, but the HSL library subset is 'the library games that have HSL' (= the plan's selection), not the in-code rng draw."""
    def __init__(self, se_path, hsl_path):
        self.se_ix = self.h_ix = None
        if se_path:
            se = pd.read_parquet(se_path)
            self.se_ix = pd.Index(se.uid); self.se_arr = se[["se_mean", "se_std"]].to_numpy(np.float64)
            m = np.load(SIM / "metric_se.npz"); self.m = {k: m[k] for k in m.files}
        if hsl_path:
            h = pd.read_parquet(hsl_path)
            z = np.load(SIM / "metric_hsl.npz", allow_pickle=True)
            cols = [str(c) for c in z["cols"]]
            assert [c for c in h.columns if c not in ("uid", "n_moves")] == cols
            Xs = (h[cols].to_numpy(np.float64) - z["mu"]) / z["sd"]
            self.h_ix = pd.Index(h.uid); self.h_Y = (Xs - z["gm"]) @ z["V"]

    def se1(self, uids):
        ix = self.se_ix.get_indexer(uids)
        assert (ix >= 0).all(), "SE missing for requested games"
        return (self.se_arr[ix][:, :1] - self.m["g1"]) @ self.m["V1"]

    def hsl(self, uids):
        ix = self.h_ix.get_indexer(uids)
        assert (ix >= 0).all(), "HSL missing for requested games"
        return self.h_Y[ix]


def bonus(F, S3, lam, lib_u, q_u, lib_black, q_black):
    """(nq, nc) additive bonus.  Library side uses black games only and white questions get 0 (no fusion): the metrics were fitted on black games.
    All-black data: lib_black/q_black are all True -> identical to E27."""
    nq, nc = len(q_u), len(lib_u)
    lb = [u[m] for u, m in zip(lib_u, lib_black)]
    qi = np.flatnonzero(q_black)
    B = np.zeros((nq, nc))
    if "se1" in lam:
        B[qi] += -0.5 * lam["se1"] * S3.dist_matrix([F.se1(u) for u in lb], [F.se1(q_u[i]) for i in qi], 1)
    if "hsl5s20" in lam:
        lu = [u[F.h_ix.get_indexer(u) >= 0] for u in lb]
        assert all(len(u) > 0 for u in lu), "a candidate has no HSL library game"
        B[qi] += -0.5 * lam["hsl5s20"] * S3.dist_matrix([F.hsl(u) for u in lu], [F.hsl(q_u[i]) for i in qi], 5)
    return B


def det_stats(top0, e0, nc):
    sm = lambda x: (lambda z: z / z.sum(1, keepdims=True))(np.exp(x - x.max(1, keepdims=True)))   # noqa: E731
    order = np.argsort(-e0, axis=1, kind="stable")
    best = np.take_along_axis(top0, order[:, :1], 1)[:, 0]
    p1 = sm(e0).max(1)
    coll = np.bincount(best, minlength=nc)[best] > 1
    return dict(c_all=float(coll.mean()), exc=float(coll.mean()) - (1.0 - float(p1.mean())), p1_mean=float(p1.mean()))


def load_questions_any(P, path, split):
    """P.load_questions without the Q_NOT_BLACK block (the src change planned in runbook section 7, done locally so src/ stays untouched)."""
    from gogame.errors import DataFormatError
    from gogame.io import explode_question, normalize_qid, read_questions_csv, validate_questions_frame
    df = read_questions_csv(path, split)
    bad = [i for i in validate_questions_frame(df, "player") if not i.code.startswith("W_") and i.code not in ("Q_NOT_BLACK", "Q_MIXED_COLOR")]   # mixed colours inside a question are fine for Player (colour is per game)
    if bad:
        raise DataFormatError(f"{path}: {len(bad)} format issue(s), first: {bad[0].code} {bad[0].detail}")
    out = []
    for _, row in df.iterrows():
        gs = explode_question(row, split)
        out.append(P.Question(normalize_qid(str(row["question_id"]), split), split, gs[0].color, gs))
    return out


def stage_decode(a):
    import predict as P
    from gogame import player_rerank as prr
    from gogame.errors import DataFormatError
    import player_sim3 as S3
    from scripts.experiments.e30 import common as E30
    sys.argv = _argv
    data = Path(a.data_dir)
    lam = {}
    if a.features in ("se", "se+hsl"):
        lam["se1"] = a.lam_se
    if a.features == "se+hsl":
        lam["hsl5s20"] = a.lam_hsl
    F = Feats(FEAT / "se_test_games.parquet" if lam else None, FEAT / "hsl_test_games.parquet" if "hsl5s20" in lam else None) if lam else None
    pm = P.PLAYER_MODELS["e04v2"].load(MD, 4)
    print(f"model include_holdout={pm.meta.get('include_holdout')} decoder={a.decoder} features={a.features} lam={lam} data={data}", flush=True)
    tj = json.loads(TUNED.read_text()); ch, gamma = tj["chosen"], tj["gamma_star"]
    zdir = Path(a.z_cache_dir); zdir.mkdir(parents=True, exist_ok=True)

    class DS: pass
    ds = DS(); ds.sel = pm.s["sel"]
    qids, tops, rep = [], [], {}
    for sp in ("pub", "priv"):
        qf, cf = data / P.FILES[f"player_{sp}"], data / P.FILES[f"cand_{sp}"]
        if not (qf.exists() and cf.exists()):
            continue
        try:
            pq = load_questions_any(P, qf, sp) if a.allow_white else P.load_questions(qf, sp, "player")
        except DataFormatError as e:
            print(f"STOP: {e}\n(white questions: rerun with --allow-white, see docs/runbook_1111_player.md section 7; no guessing)"); return 3
        cands = P.load_candidates(cf, sp); pool = sorted(cands)
        games = [g for p in pool for g in cands[p]] + [g for q in pq for g in q.games]
        zp = zdir / f"Z_{sp}.npz"
        if zp.exists():
            z = np.load(zp, allow_pickle=False); uids, Z = z["uids"].astype(str), z["Z"]
            assert set(uids) == {g.uid for g in games}, f"{zp} was made from different games; use another --z-cache-dir"
        else:
            t0 = time.perf_counter(); store = P.l0_store(games, 4)
            Z = np.empty((len(store.uids), len(pm.s["mu"]) + len(pm.s["cols"])), np.float32)
            for i in range(0, len(Z), 20000):
                Z[i:i + 20000] = pm._z(store, np.arange(i, min(i + 20000, len(Z))))
            uids = np.array(store.uids).astype(str); np.savez(zp, uids=uids, Z=Z)
            print(f"[{sp}] L0 features {len(uids)} games {time.perf_counter() - t0:.0f}s", flush=True)
        rowof = {u: i for i, u in enumerate(uids)}
        rows = lambda us: np.fromiter((rowof[u] for u in us), np.int64, len(us))   # noqa: E731
        lib_u = [np.array([g.uid for g in cands[p]]) for p in pool]; q_u = [np.array([g.uid for g in q.games]) for q in pq]
        lib_z = [Z[rows(u)] for u in lib_u]; q_z = [Z[rows(u)] for u in q_u]
        nc, nq = len(pool), len(pq)
        t5_v3, top0, e0 = S3.trans_t5(pm.booster, ds, lib_z, q_z, None, tau=TAU)          # first pass without bonus: detector input; also the v3 answer
        st = det_stats(top0, e0, nc)
        dec = a.decoder if a.decoder != "auto" else (ch["F"] if st[ch["stat"]] > ch["theta"] else "B")
        if a.decoder == "auto" and nq != nc:
            print(f"[{sp}] NOTE nq {nq} != nc {nc} (reported only; E30 decision uses exc alone)")
        print(f"[{sp}] nq {nq} nc {nc} exc {st['exc']:.4f} (theta {ch['theta']:.4f}) -> decoder {dec}", flush=True)
        lib_black = [np.array([g.color == 0 for g in cands[p]]) for p in pool]
        q_black = np.array([all(g.color == 0 for g in q.games) for q in pq])
        if not (q_black.all() and all(m.all() for m in lib_black)):
            print(f"[{sp}] WHITE present: questions with W {int((~q_black).sum())}, candidates with W games {sum(int((~m).any()) for m in lib_black)}: "
                  f"fusion bonus only for black questions with black library games", flush=True)
        Bm = bonus(F, S3, lam, lib_u, q_u, lib_black, q_black) if lam else None
        if dec == "A":      # independent per-question ranking (no one-to-one / transductive step): the first-pass order
            t5 = S3.np.take_along_axis(top0, np.argsort(-(e0 if Bm is None else e0 + Bm[np.arange(nq)[:, None], top0]), axis=1, kind="stable"), 1)[:, :5]
        elif dec == "B":
            t5 = t5_v3 if Bm is None else S3.trans_t5(pm.booster, ds, lib_z, q_z, Bm, tau=TAU)[0]
        else:
            e = e0 if Bm is None else e0 + Bm[np.arange(nq)[:, None], top0]
            t5 = E30.decode_D(top0, e, nc, gamma)
        qids += [q.qid for q in pq]; tops.append(np.array(pool, dtype=object)[t5])
        rep[sp] = dict(nq=nq, nc=nc, exc=st["exc"], c_all=st["c_all"], decoder=dec)
    dst = Path(a.out); dst.parent.mkdir(parents=True, exist_ok=True)
    P.write_player_submission(qids, np.vstack(tops), dst)
    import check_submission
    rc = check_submission.main([str(dst), "--task", "player", "--data-dir", str(data)])
    rep["check_rc"] = rc; rep["sha256_12"] = hashlib.sha256(dst.read_bytes()).hexdigest()[:12]
    if a.compare:
        x, y = pd.read_csv(dst), pd.read_csv(a.compare)
        m = x.merge(y, on=x.columns[0], suffixes=("_n", "_r"))
        rep["compare"] = dict(ref=str(a.compare), rows=int(len(m)), top1_agree=float((m[x.columns[1] + "_n"] == m[x.columns[1] + "_r"]).mean()),
                              rows_any_top5_differs=int((m.filter(like="_n").to_numpy() != m.filter(like="_r").to_numpy()).any(1).sum()),
                              sha_ref12=hashlib.sha256(Path(a.compare).read_bytes()).hexdigest()[:12])
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    (dst.with_suffix(".json")).write_text(json.dumps(rep, indent=1))
    return 0 if rc == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge"); m.add_argument("--se-out"); m.add_argument("--hsl-new")
    d = sub.add_parser("decode")
    d.add_argument("--data-dir", required=True); d.add_argument("--out", required=True)
    d.add_argument("--decoder", choices=["A", "B", "D", "auto"], default="auto"); d.add_argument("--features", choices=["none", "se", "se+hsl"], default="none")
    d.add_argument("--lam-se", type=float, default=0.5); d.add_argument("--lam-hsl", type=float, default=0.25)
    d.add_argument("--z-cache-dir", default=str(OUT / "z")); d.add_argument("--compare")
    d.add_argument("--allow-white", action="store_true", help="accept white questions (see runbook section 7); no fusion for them")
    a = ap.parse_args(_argv[1:])
    return stage_merge(a) if a.cmd == "merge" else stage_decode(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
