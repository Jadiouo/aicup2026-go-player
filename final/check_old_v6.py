"""Sanity-check the saved old-question half (v6, 800 rows) of the Player submission.

The 800 rows were produced once by scripts/oneoff/e27_player_fusion.py (stage `fuse`, 2026-11-06) and then reused unchanged by every later submission.
This script does NOT regenerate them; it checks that the file you hold is the one that was uploaded:
  * sha256 == b4eed172f033... (first 12 hex digits)        * 800 rows, header question_id,top_1..top_5, ids pub_q_*/priv_q_*, 5 distinct ids per row
  * optional --v3 FILE (the earlier v3 submission, sha256 b51ae1671927...): prints how many rows differ in top-1 / in any of top-1..5
    (recorded at fusion time: 10 top-1 changes, 795 rows with any change; by N = 5/10/15/20: top-1 6/2/1/1)

  python final/check_old_v6.py .scratch/e62/old_v6_800.csv [--v3 player_v3.csv]
"""
import sys, csv, hashlib, argparse
from pathlib import Path

EXPECT = "b4eed172f033"
HDR = ["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--v3")
    a = ap.parse_args(); p = Path(a.csv)
    sha = hashlib.sha256(p.read_bytes()).hexdigest(); ok = sha.startswith(EXPECT)
    print(f"sha256 {sha}\n  expected prefix {EXPECT}: {'OK' if ok else 'MISMATCH'}")
    rows = list(csv.reader(open(p, newline="")))
    errs = []
    if rows[0] != HDR: errs.append(f"header {rows[0]}")
    body = rows[1:]
    if len(body) != 800: errs.append(f"rows {len(body)} != 800")
    if len({r[0] for r in body}) != len(body): errs.append("duplicate question ids")
    for r in body:
        if not (r[0].startswith("pub_q_") or r[0].startswith("priv_q_")): errs.append(f"unexpected id {r[0]}")
        if len(r) != 6 or len(set(r[1:])) != 5 or "" in r[1:]: errs.append(f"{r[0]}: not 5 distinct filled ids")
    print(f"format: {len(body)} rows, {len(errs)} errors"); [print("  ", e) for e in errs[:10]]
    if a.v3:
        v3 = {r[0]: r[1:] for r in list(csv.reader(open(a.v3, newline="")))[1:]}
        t1 = sum(v3[r[0]][0] != r[1] for r in body); anyd = sum(v3[r[0]] != r[1:] for r in body)
        print(f"vs v3: top-1 changed {t1} (recorded 10), any of top-1..5 changed {anyd} (recorded 795)")
    return 0 if ok and not errs else 1


if __name__ == "__main__":
    sys.exit(main())
