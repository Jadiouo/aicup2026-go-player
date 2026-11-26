"""Player final step: channel score matrices -> 1,600-row submission (v7x).  CPU only, seconds.

New-question half (pub2_/priv2_, 800 rows), per pool (Public / Private, 400 questions each), exactly the arithmetic of the uploaded v7x:
    z   = rowz(e)                                   e = stage-2 booster scores on the 40-candidate shortlist (real_v7q_S_*.npz)
    c   = z + W_E112 * X_e112[shortlist] + W_LR * X_lr[shortlist] + W_NEW * X_p1r[shortlist]
    c'  = csls_qq(c, shortlist, P=600, k=3, alpha=.5, mode="z")      soft one-to-one prior (transductive inside the pool)
    top5 = shortlist sorted by c' (stable), first five
Old-question half (pub_/priv_, 800 rows) is taken unchanged from --old800 (v6 submission rows).  Both halves are joined by scripts/oneoff/p1111_assemble.py,
which also validates the file against the four official test files (needs the official data in $DATA_DIR/contest_1111).
Setting --w-new 0 reproduces the previous submission (v7w).

Inputs under --work-dir (default $WORK_DIR or <repo>/.scratch):
  e90/final/make/real_v7q_S_{pub,priv}.npz   e, top, pool, qids         (booster scores, shortlist, candidate ids, question ids)
  e112/final/real_X_{pub,priv}.npz           X (400,600)                (E112 kernel-density position matching)
  e123/final/real_Xlr_{pub,priv}.npz         X (400,600)                (E123 per-pool logistic-regression head)
  e127/final/real_Xp1r_{pub,priv}.npz        X (400,600), ng            (E127 pairwise matcher)
  e62/old_v6_800.csv                         the 800 old-question rows (default --old800)

  python final/make_player_final.py --out /tmp/player_v7x.csv [--reference path/to/uploaded.csv]
"""
import os, sys, csv, argparse, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "scripts/lib"), str(ROOT / "scripts/oneoff"), str(ROOT / "src")]
import numpy as np
from e81_lib import csls_qq
import p1111_assemble as ASM

W_E112, W_LR, W_NEW = 0.5, 1.0, 0.35
sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def new_question_rows(W, w_e, w_lr, w_new):
    """e123_make.t2_csv + one extra term (= scripts/experiments/e127/make_final.py: t2_csv_new)."""
    rows = []
    for sp in ("pub", "priv"):
        z = np.load(W / f"e90/final/make/real_v7q_S_{sp}.npz", allow_pickle=True); e, top, pool, qids = z["e"], z["top"].astype(np.int64), z["pool"], z["qids"]
        zs = (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-9)
        Xs = np.take_along_axis(np.load(W / f"e112/final/real_X_{sp}.npz")["X"].astype(np.float64), top, 1)
        Ls = np.take_along_axis(np.load(W / f"e123/final/real_Xlr_{sp}.npz")["X"].astype(np.float64), top, 1)
        Ns = np.take_along_axis(np.load(W / f"e127/final/real_Xp1r_{sp}.npz")["X"].astype(np.float64), top, 1)
        c = zs + w_e * Xs; c = c + w_lr * Ls; c = c + w_new * Ns; c = csls_qq(c, top, len(pool), 3, 0.5, "z")
        t5 = np.take_along_axis(top, np.argsort(-c, axis=1, kind="stable"), 1)[:, :5]
        rows += [[str(q)] + [str(x) for x in pool[t5[i]]] for i, q in enumerate(qids)]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR", str(ROOT / ".scratch"))); ap.add_argument("--old800"); ap.add_argument("--out", required=True)
    ap.add_argument("--w-e112", type=float, default=W_E112); ap.add_argument("--w-lr", type=float, default=W_LR); ap.add_argument("--w-new", type=float, default=W_NEW)
    ap.add_argument("--reference", help="uploaded CSV to compare with (optional)")
    a = ap.parse_args(); W = Path(a.work_dir); out = Path(a.out); old = a.old800 or str(W / "e62/old_v6_800.csv")
    if out.exists(): raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True); t2 = out.with_name(out.stem + "_new2_part.csv")
    rows = new_question_rows(W, a.w_e112, a.w_lr, a.w_new)
    with open(t2, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(ASM.HDR); w.writerows(rows)
    sys.argv = ["p1111_assemble", "--old", old, "--new2", str(t2), "--out", str(out)]; rc = ASM.main()      # joins old + new, validates against the official files
    t2.unlink()
    print(f"sha256 {sha256(out)}")
    if a.reference:
        eq = Path(a.reference).read_bytes() == out.read_bytes(); print(f"reference {a.reference}\nreference sha256 {sha256(a.reference)}\nBYTE-IDENTICAL: {eq}"); return rc or (0 if eq else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
