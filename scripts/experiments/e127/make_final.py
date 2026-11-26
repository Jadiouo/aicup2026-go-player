"""E127 alternate candidate (v7x): v7w formula + w_new * X_p1r[shortlist], E123 production path (e123_make.t2_csv arithmetic, one extra term).  CPU, .venv python.  Writes ONLY .scratch/e127/final/.  Reads no truths.
subcommands: path (w=0 byte identity vs v7w, candidate w=.35, check 1-4), stats (check 5 + descriptive scale comparison)"""
import sys, os, json, csv, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; FIN = ROOT / ".scratch/e127/final"
sys.dont_write_bytecode = True; sys.path[:0] = [str(ROOT / "scripts/experiments/e123"), str(ROOT / "scripts/experiments/e127")]
import numpy as np
import e123_make as EM                       # REAL, E112, FIN(e123), assemble, check, rows_of, sha
import e123_lib as L
import e112_stage2 as S
from e81_lib import csls_qq
V7W = ROOT / "data/submissions/player_v7w_候選池分類頭w100.csv"; W_NEW = 0.35; NAME = "player_v7x_成對比對w35.csv"
sha = EM.sha


def t2_csv_new(out, w_e, w_lr, w_new):
    """e123_make.t2_csv line by line + `c = c + w_new * Ns` right after the w_lr term (= e127_score.Cell.combo: zs + .5*Xe + 1.0*Xlr + w*X_new, then the same csls_qq)."""
    rows = []
    for sp in ("pub", "priv"):
        z = np.load(EM.REAL / f"real_v7q_S_{sp}.npz", allow_pickle=True); e, top, pool, qids = z["e"], z["top"].astype(np.int64), z["pool"], z["qids"]
        zs = (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-9); X = np.load(EM.E112 / f"real_X_{sp}.npz")["X"].astype(np.float64); Xs = np.take_along_axis(X, top, 1)
        Xl = np.load(EM.FIN / f"real_Xlr_{sp}.npz")["X"].astype(np.float64); Ls = np.take_along_axis(Xl, top, 1)
        Xn = np.load(FIN / f"real_Xp1r_{sp}.npz")["X"].astype(np.float64); Ns = np.take_along_axis(Xn, top, 1)
        c = zs + w_e * Xs; c = c + w_lr * Ls; c = c + w_new * Ns; c = csls_qq(c, top, len(pool), 3, 0.5, "z")
        t5 = np.take_along_axis(top, np.argsort(-c, axis=1, kind="stable"), 1)[:, :5]
        rows += [[str(q)] + [str(x) for x in pool[t5[i]]] for i, q in enumerate(qids)]
    with open(out, "w", newline="") as f:
        w_ = csv.writer(f, lineterminator="\n"); w_.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]); w_.writerows(rows)


def path():
    R = {}
    ref_t2 = FIN / "t2_ref_e123path_w1.csv"; t2_0 = FIN / "t2_w0.csv"; t2_c = FIN / "t2_cand.csv"; fin0 = FIN / "player_w0_repro.csv"; fin = FIN / NAME
    for p in (ref_t2, t2_0, t2_c, fin0, fin): assert not p.exists(), f"refuse to overwrite {p}"
    EM.t2_csv(ref_t2, .5, 1.0)                                      # the ORIGINAL e123 function (no new term), only the output location differs
    t2_csv_new(t2_0, .5, 1.0, 0.0); t2_csv_new(t2_c, .5, 1.0, W_NEW)
    R["t2_w0_bytes_equal_original_e123_t2_csv"] = t2_0.read_bytes() == ref_t2.read_bytes()
    rc, txt = EM.assemble(t2_0, fin0); R["assemble_w0"] = dict(rc=rc, out=txt); R["w0_final_sha256"] = sha(fin0); R["v7w_sha256"] = sha(V7W); R["w0_bytes_equal_v7w"] = fin0.read_bytes() == V7W.read_bytes()
    rc, txt = EM.assemble(t2_c, fin); R["assemble_cand"] = dict(rc=rc, out=txt); R["cand_sha256"] = sha(fin); R["cand_sha8"] = sha(fin)[:8]
    R["check_t2_w0"] = EM.check(t2_0); R["check_t2_cand"] = EM.check(t2_c); R["check_1600_cand"] = EM.check(fin); R["check_1600_v7w"] = EM.check(V7W)
    a, b = EM.rows_of(V7W), EM.rows_of(fin); R["rows_total"] = dict(v7w=len(a), cand=len(b)); R["header_equal"] = a[0] == b[0]; R["qid_order_equal"] = [r[0] for r in a] == [r[0] for r in b]; R["first800_identical"] = a[1:801] == b[1:801]
    R["old800_question_prefixes"] = sorted({r[0].split("_")[0] for r in b[1:801]}); R["new800_question_prefixes"] = sorted({r[0].split("_")[0] for r in b[801:]})
    R["last800_n_differing_rows"] = sum(x != y for x, y in zip(a[801:], b[801:]))
    (FIN / "path_result.json").write_text(json.dumps(R, indent=1, ensure_ascii=False)); print(json.dumps(R, indent=1, ensure_ascii=False))
    assert R["t2_w0_bytes_equal_original_e123_t2_csv"] and R["w0_bytes_equal_v7w"]


def stats():
    R = {}; a, b = EM.rows_of(V7W), EM.rows_of(FIN / NAME); R["pool_membership"] = {}
    ng = {}; pools = {}
    for sp, pref in (("public", "pub2_"), ("private", "priv2_")):
        for r in csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_test2_{sp}.csv", newline="")): ng[r["question_id"]] = int(r["num_games"])
        pools[pref] = {r["player_id"] for r in csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", newline=""))}; assert len(pools[pref]) == 600
    bad_d = [r[0] for r in b[1:] if len(set(r[1:])) != 5 or len(r) != 6]; R["rows_with_non_5_distinct_all1600"] = len(bad_d)
    bad_p = [r[0] for r in b[801:] if not all(x in pools[r[0].split("_")[0] + "_"] for x in r[1:])]; R["new800_rows_with_candidate_outside_own_pool"] = len(bad_p)
    R["new800_pool_prefix_ok"] = all(x.startswith(r[0].split("_")[0] + "_p_") for r in b[801:] for x in r[1:])
    for nm, pref in (("Public", "pub2_"), ("Private", "priv2_")):
        ia = [i for i in range(801, 1601) if a[i][0].startswith(pref)]; assert len(ia) == 400; ch = [i for i in ia if a[i][1] != b[i][1]]
        old_rank = [a[i][1:].index(b[i][1]) + 1 if b[i][1] in a[i][1:] else 6 for i in ch]; byN = {}
        for N in (1, 2, 3, 4, 5):
            ii = [i for i in ia if ng[a[i][0]] == N]
            if ii: byN[str(N)] = dict(queries=len(ii), top1_changed=sum(a[i][1] != b[i][1] for i in ii), top5_set_changed=sum(set(a[i][1:]) != set(b[i][1:]) for i in ii), top5_ordered_changed=sum(a[i][1:] != b[i][1:] for i in ii))
        R[nm] = dict(queries=len(ia), top1_changed=len(ch), top5_set_changed=sum(set(a[i][1:]) != set(b[i][1:]) for i in ia), top5_ordered_changed=sum(a[i][1:] != b[i][1:] for i in ia), new_top1_old_rank_dist={str(k): int(sum(1 for r in old_rank if r == k)) for k in range(2, 7)},
                     new_top1_in_old_top5=int(sum(1 for r in old_rank if r <= 5)), by_N=byN)
    # ---- descriptive scale comparison: real X vs pseudo X (shortlist only; no scores, no truths)
    D = {}
    def cur_top1(c_, top_):                       # current (v7w) first place after the same csls_qq
        c2 = csls_qq(c_, top_, 600, S.K, S.ALPHA, "z"); return np.take_along_axis(top_, np.argsort(-c2, axis=1, kind="stable"), 1)[:, 0]
    def metrics(Xs, c, top, n):
        mx = Xs.max(1); sd = Xs.std(1); ch = top[np.arange(len(top)), Xs.argmax(1)] == cur_top1(c, top); o = {}
        for N in sorted(set(n.tolist())): k = n == N; o[str(N)] = dict(q=int(k.sum()), mean_max_z=float(mx[k].mean()), mean_row_std=float(sd[k].mean()), chan_top1_eq_cur_top1=float(ch[k].mean()))
        o["all"] = dict(q=int(len(n)), mean_max_z=float(mx.mean()), mean_row_std=float(sd.mean()), chan_top1_eq_cur_top1=float(ch.mean())); return o
    for s, sp in (("pub", "public"), ("priv", "private")):
        z = np.load(EM.REAL / f"real_v7q_S_{s}.npz", allow_pickle=True); e, top = z["e"], z["top"].astype(np.int64); zs = (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-9)
        c = zs + .5 * np.take_along_axis(np.load(EM.E112 / f"real_X_{s}.npz")["X"].astype(np.float64), top, 1) + 1.0 * np.take_along_axis(np.load(EM.FIN / f"real_Xlr_{s}.npz")["X"].astype(np.float64), top, 1)
        d = np.load(FIN / f"real_Xp1r_{s}.npz"); n = d["ng"]; D[f"real/{s}"] = metrics(np.take_along_axis(d["X"].astype(np.float64), top, 1), c, top, n)
        for tag in ("", "c"):
            zz = np.load(ROOT / f".scratch/e110/dumps/t2{tag}_{s}.npz"); Xe = np.load(ROOT / f".scratch/e112/t2_X_{tag}_{s}.npz")["X"]; Xlr = np.load(ROOT / f".scratch/e127/Xlr_{tag}_{s}.npz")["X"]; Xp = np.load(ROOT / f".scratch/e127/X/X_p1r_M58_{tag}_{s}.npz")["X"]
            acc = {}; ag = {}
            for r in range(15):
                m = np.flatnonzero(zz["rep"] == r); tp = zz["top"][m].astype(np.int64); ee = zz["e"][m]; zr = (ee - ee.mean(1, keepdims=True)) / (ee.std(1, keepdims=True) + 1e-9)
                cc = zr + .5 * np.take_along_axis(Xe[m], tp, 1) + 1.0 * np.take_along_axis(Xlr[m], tp, 1); Xs = np.take_along_axis(Xp[m].astype(np.float64), tp, 1); nn = zz["n"][m]
                mx = Xs.max(1); sd = Xs.std(1); ch = tp[np.arange(len(tp)), Xs.argmax(1)] == cur_top1(cc, tp)
                for N in sorted(set(nn.tolist())) + ["all"]:
                    k = np.ones(len(nn), bool) if N == "all" else nn == N; acc.setdefault(str(N), []).append((mx[k], sd[k], ch[k]))
            D[f"pseudo/{s}/{tag or 'a'}"] = {N: dict(q=int(sum(len(x[0]) for x in v)), mean_max_z=float(np.concatenate([x[0] for x in v]).mean()), mean_row_std=float(np.concatenate([x[1] for x in v]).mean()), chan_top1_eq_cur_top1=float(np.concatenate([x[2] for x in v]).mean())) for N, v in acc.items()}
    R["descriptive"] = D; (FIN / "stats_result.json").write_text(json.dumps(R, indent=1, ensure_ascii=False)); print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"path": path, "stats": stats}[sys.argv[1]]()
