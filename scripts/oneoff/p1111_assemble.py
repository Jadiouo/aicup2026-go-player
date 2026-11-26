#!/usr/bin/env python
"""P1111: assemble the 1,600-row Player submission (11/11 test set) and validate it against the four official question files.

  assemble: pub_/priv_ rows come from --old (required; the four original files are byte-identical to 11/04),
            pub2_/priv2_ rows come from --new2 (decoded on the id-stripped copy in .scratch/p1111/t2, where pub2_->pub_, priv2_->priv_; mapped back here).
  validate: 1,600 rows, header, question-id set == the four test files, id prefix matches, 5 distinct, all in the matching candidate file, all-filled.
Usage: p1111_assemble.py --old OLD800.csv --new2 .scratch/p1111/out/t2_xxx.csv --out submission.csv   (DATA_DIR env: directory holding contest_1111/)
       p1111_assemble.py --validate-only FILE
"""
import argparse, csv, hashlib, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = Path(os.environ.get("DATA_DIR", ROOT / "data")) / "contest_1111"      # official 11/11 test files; override with $DATA_DIR
HDR = ["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]
TESTS = {"pub_": "player_identification_test_public.csv", "priv_": "player_identification_test_private.csv",
         "pub2_": "player_identification_test2_public.csv", "priv2_": "player_identification_test2_private.csv"}
CANDS = {"pub_": "player_identification_candidates_public.csv", "priv_": "player_identification_candidates_private.csv",
         "pub2_": "player_identification_candidates2_public.csv", "priv2_": "player_identification_candidates2_private.csv"}
csv.field_size_limit(sys.maxsize)


def prefix(x):
    m = re.match(r"^(pub2_|priv2_|pub_|priv_)", x)
    return m.group(1) if m else None


def read_col(path, col):
    with open(path, newline="") as f:
        return [r[col] for r in csv.DictReader(f)]


def validate(path):
    errs = []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != HDR:
        errs.append(f"header {rows[0]}")
    body = rows[1:]
    if len(body) != 1600:
        errs.append(f"rows {len(body)} != 1600")
    qs = [r[0] for r in body]
    if len(set(qs)) != len(qs):
        errs.append("duplicate question ids")
    expect = set()
    for k, fn in TESTS.items():
        ids = read_col(D / fn, "question_id")
        assert all(prefix(i) == k for i in ids), k
        expect |= set(ids)
    if set(qs) != expect:
        errs.append(f"question set differs: missing {len(expect - set(qs))}, extra {len(set(qs) - expect)}")
    pools = {k: set(read_col(D / fn, "player_id")) for k, fn in CANDS.items()}
    for r in body:
        k = prefix(r[0])
        tops = r[1:]
        if len(r) != 6 or any(t == "" for t in tops):
            errs.append(f"{r[0]}: not 5 filled"); continue
        if len(set(tops)) != 5:
            errs.append(f"{r[0]}: duplicates")
        for t in tops:
            if prefix(t) != k:
                errs.append(f"{r[0]}: prefix mismatch {t}"); break
            if t not in pools[k]:
                errs.append(f"{r[0]}: {t} not in candidate pool"); break
    return errs, len(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new2"); ap.add_argument("--out"); ap.add_argument("--validate-only")
    a = ap.parse_args()
    if a.validate_only:
        path = Path(a.validate_only)
    else:
        with open(a.old, newline="") as f:
            old = list(csv.reader(f))[1:]
        with open(a.new2, newline="") as f:
            new = list(csv.reader(f))[1:]
        fix = lambda x: re.sub(r"^(pub|priv)_", r"\g<1>2_", x)   # noqa: E731
        new = [[fix(c) for c in r] for r in new]
        assert all(prefix(r[0]) in ("pub_", "priv_") for r in old)
        path = Path(a.out)
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(HDR); w.writerows(old + new)
    if not D.exists():     # no official data (e.g. a Colab run without the competition files): the CSV is written, validation is skipped
        print(f"{path}  sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:8]}  (official files not found in {D}: validation SKIPPED)")
        return 0
    errs, n = validate(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path}  rows {n}  sha256 {sha[:8]}  errors {len(errs)}")
    for e in errs[:20]:
        print("  ", e)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
