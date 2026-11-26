"""E123 real-query path (adapted from scripts/experiments/e112/e112_amend1.py make/t2_csv).  Subcommands
  real   X_lr for the REAL test2 queries (library = all 10 games of each of the 600 candidates, query = its num_games games), Public/Private separately; then the w_lr = 0 byte-identity check against
         data/submissions/player_v7u_局面比對w50.csv (check 4).  Writes .scratch/e123/final/ only.  Reads no truths.
  cand   only if .scratch/e123/result.json says `E123 PASS`: candidate file with w_lr = 1.0 (+ guards).
"""
import sys, os, json, csv, hashlib, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE)]
import numpy as np
import e123_lib as L
from e123_lib import ROOT, OUT, NC, POOLS
FIN = OUT / "final"; REAL = ROOT / ".scratch/e90/final/make"; V7U = ROOT / "data/submissions/player_v7u_局面比對w50.csv"; OLD800 = ROOT / ".scratch/e62/old_v6_800.csv"; E112 = ROOT / ".scratch/e112/final"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def real_Xlr(s, sp):
    v = L.load_vecs(s); row = L.check_rows(s, sp)
    c = list(csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", newline=""))); pid = np.array([r["player_id"] for r in c]); pl = sorted(set(pid)); assert len(pl) == NC
    y = np.array([pl.index(p) for p in pid]); assert (np.bincount(y) == 10).all()
    tq = list(csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_test2_{sp}.csv", newline=""))); ng = np.array([int(r["num_games"]) for r in tq]); assert len(tq) == 400 and ng.sum() == 1100
    z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); uids = z["uids"].astype(str); qid = np.load(REAL / f"real_v7q_S_{s}.npz", allow_pickle=True)["qids"]; start = np.concatenate([[0], np.cumsum(ng)])[:-1]
    for i in range(400):                                                    # query games = export rows 6000 + start .. + ng, uid 'pub_q_0001#1'
        assert qid[i] == tq[i]["question_id"].replace("pub2_", "pub_").replace("priv2_", "priv_")
        for j in range(ng[i]): assert uids[6000 + start[i] + j] == f"{qid[i]}#{j + 1}", (i, j, uids[6000 + start[i] + j])
    pool = np.load(REAL / f"real_v7q_S_{s}.npz", allow_pickle=True)["pool"]; assert [x.split("_", 1)[1] for x in pool] == [p.split("_", 1)[1] for p in pl], "pool order != sorted candidate ids"
    logp, it = L.fit_head(v[row], y); G = logp(v[6000:7100]); A = np.zeros((400, NC))
    for i in range(400): A[i] = G[start[i]:start[i] + ng[i]].mean(0)
    X = L.zrow(A).astype(np.float32); assert np.isfinite(X).all(); np.savez(FIN / f"real_Xlr_{s}.npz", X=X); print(f"  real X_lr {s}: shape {X.shape} n_iter {it} mean/std {float(X.mean()):.4f}/{float(X.std()):.4f}", flush=True); return it


def t2_csv(out, w_e, w_lr):
    """E112 production path + w_lr * X_lr[shortlist] after the E112 channel (same CSLS call)."""
    from e81_lib import csls_qq
    rows = []
    for sp in ("pub", "priv"):
        z = np.load(REAL / f"real_v7q_S_{sp}.npz", allow_pickle=True); e, top, pool, qids = z["e"], z["top"].astype(np.int64), z["pool"], z["qids"]
        zs = (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-9); X = np.load(E112 / f"real_X_{sp}.npz")["X"].astype(np.float64); Xs = np.take_along_axis(X, top, 1)
        Xl = np.load(FIN / f"real_Xlr_{sp}.npz")["X"].astype(np.float64); Ls = np.take_along_axis(Xl, top, 1)
        c = zs + w_e * Xs; c = c + w_lr * Ls; c = csls_qq(c, top, len(pool), 3, 0.5, "z")
        t5 = np.take_along_axis(top, np.argsort(-c, axis=1, kind="stable"), 1)[:, :5]
        rows += [[str(q)] + [str(x) for x in pool[t5[i]]] for i, q in enumerate(qids)]
    with open(out, "w", newline="") as f:
        w_ = csv.writer(f, lineterminator="\n"); w_.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]); w_.writerows(rows)


def assemble(t2, out):
    r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/oneoff/p1111_assemble.py"), "--old", str(OLD800), "--new2", str(t2), "--out", str(out)], cwd=ROOT, capture_output=True, text=True); return r.returncode, r.stdout.strip()


def check(p):
    r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/check_submission.py"), str(p), "--task", "player", "--data-dir", str(ROOT / ".scratch/p1111/t2")], cwd=ROOT, capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr).strip()


def rows_of(p):
    with open(p, newline="") as f: return list(csv.reader(f))


def real():
    its = {s: real_Xlr(s, sp) for s, sp in POOLS}
    t2 = FIN / "t2_w0.csv"; fin = FIN / "player_w0_repro.csv"; assert not t2.exists() and not fin.exists(); t2_csv(t2, .5, 0.0); rc, txt = assemble(t2, fin); assert rc == 0, txt
    eq = fin.read_bytes() == V7U.read_bytes(); R = dict(n_iter=its, w0_final_sha=sha(fin), v7u_sha=sha(V7U), w0_bytes_equal_v7u=eq, assemble=txt, check_t2_rc=check(t2)[0])
    with open(OUT / "check4_w0.json", "x") as f: f.write(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1)); assert eq


def cand():
    res = json.loads((OUT / "result.json").read_text()); assert res["verdict"] == "E123 PASS", "candidate only after E123 PASS"
    t2 = FIN / "t2_cand.csv"; fin = FIN / "player_v7u_poolhead.csv"; assert not t2.exists() and not fin.exists(); t2_csv(t2, .5, 1.0); rc, txt = assemble(t2, fin); assert rc == 0, txt
    a, b = rows_of(V7U), rows_of(fin); assert a[0] == b[0] and len(a) == len(b) == 1601 and [r[0] for r in a] == [r[0] for r in b]
    R = dict(final_sha=sha(fin), final_sha8=sha(fin)[:8], assemble=txt, check_t2_rc=check(t2)[0], check_1600=check(fin), first800_identical_to_v7u=a[1:801] == b[1:801], w0_bytes_equal_v7u=json.loads((OUT / "check4_w0.json").read_text())["w0_bytes_equal_v7u"])
    ng = {}
    for sp in ("public", "private"):
        for r in csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_test2_{sp}.csv", newline="")): ng[r["question_id"]] = int(r["num_games"])
    for nm, pref in (("Public", "pub2_"), ("Private", "priv2_")):
        ia = [i for i in range(801, 1601) if a[i][0].startswith(pref)]; ch = [i for i in ia if a[i][1] != b[i][1]]
        old_rank = [a[i][1:].index(b[i][1]) + 1 if b[i][1] in a[i][1:] else 6 for i in ch]; byN = {}
        for N in (1, 2, 3, 4, 5):
            ii = [i for i in ia if ng[a[i][0]] == N]; byN[str(N)] = dict(queries=len(ii), top1_changed=sum(a[i][1] != b[i][1] for i in ii))
        R[nm] = dict(rows=len(ia), top1_changed=len(ch), top5_ordered_changed=sum(a[i][1:] != b[i][1:] for i in ia), top5_set_changed=sum(set(a[i][1:]) != set(b[i][1:]) for i in ia),
                     new_top1_old_rank_dist={str(k): int(sum(1 for r in old_rank if r == k)) for k in range(2, 7)}, new_top1_in_old_top5_share=(sum(1 for r in old_rank if r <= 5) / len(ch)) if ch else None, by_N=byN)
    with open(FIN / "make_result.json", "x") as f: f.write(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"real": real, "cand": cand}[sys.argv[1]]()
