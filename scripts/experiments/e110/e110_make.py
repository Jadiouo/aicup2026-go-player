"""E110 candidate file (only after `E110 ADOPT`): same production path as scripts/experiments/e90/e90_final.py (t2_csv -> p1111_assemble), copied with k as an argument.
Input = v7q's own test2 booster dump (.scratch/e90/final/make/real_v7q_S_{pub,priv}.npz, read-only).  Step 1 shows that k=10 reproduces the stored v7q t2 csv and the submitted
data/submissions/player_v7q_S通道.csv byte for byte; step 2 changes k to 3 (alpha .5, mode z, Public among Public queries, Private among Private - unchanged).  Writes only .scratch/e110/final/.
Usage: PYTHONPATH=src:scripts:. .venv/bin/python scripts/experiments/e110/e110_make.py"""
import sys, csv, json, hashlib, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/E81_player_error_analysis")]
import numpy as np
from e81_lib import csls_qq
PY = str(ROOT / ".venv/bin/python"); REAL = ROOT / ".scratch/e90/final/make"; MODEL = "v7q_S"; OLD800 = ROOT / ".scratch/e62/old_v6_800.csv"
V7Q = ROOT / "data/submissions/player_v7q_S通道.csv"; OUT = ROOT / ".scratch/e110/final"; OUT.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def t2_csv(out, k):
    """COPY of e90_final.t2_csv with k as argument (alpha .5, mode z)."""
    rows = []
    for sp in ("pub", "priv"):
        z = np.load(REAL / f"real_{MODEL}_{sp}.npz", allow_pickle=True); e, top, pool, qids = z["e"], z["top"].astype(np.int64), z["pool"], z["qids"]
        e = csls_qq(e, top, len(pool), k, 0.5, "z")
        t5 = np.take_along_axis(top, np.argsort(-e, axis=1, kind="stable"), 1)[:, :5]
        rows += [[str(q)] + [str(x) for x in pool[t5[i]]] for i, q in enumerate(qids)]
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]); w.writerows(rows)

def assemble(t2, out):
    r = subprocess.run([PY, str(ROOT / "scripts/oneoff/p1111_assemble.py"), "--old", str(OLD800), "--new2", str(t2), "--out", str(out)], cwd=ROOT, capture_output=True, text=True); return r.returncode, r.stdout.strip()

def check(p, data_dir):
    r = subprocess.run([PY, str(ROOT / "scripts/check_submission.py"), str(p), "--task", "player", "--data-dir", str(data_dir)], cwd=ROOT, capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr).strip()

def rows(p):
    with open(p, newline="") as f: return list(csv.reader(f))

R = {}
for k in (10, 3):
    t2 = OUT / f"t2_csls_k{k}.csv"; fin = OUT / (f"player_v7q_k10_repro.csv" if k == 10 else "player_v7q_csls_k3.csv")
    assert not t2.exists() and not fin.exists(), "refusing to overwrite"
    t2_csv(t2, k); rc, txt = assemble(t2, fin); assert rc == 0, txt
    R[f"k{k}"] = dict(t2_sha=sha(t2), final_sha=sha(fin), assemble=txt, check_t2_rc=check(t2, ROOT / ".scratch/p1111/t2")[0], check_1600=check(fin, ROOT / ".scratch/p1111/t2"))
R["k10_t2_bytes_equal_stored_v7q_t2csv"] = (OUT / "t2_csls_k10.csv").read_bytes() == (REAL / "t2_csls_v7q_S.csv").read_bytes()
R["k10_final_bytes_equal_submitted_v7q"] = (OUT / "player_v7q_k10_repro.csv").read_bytes() == V7Q.read_bytes()
R["v7q_check_1600"] = check(V7Q, ROOT / ".scratch/p1111/t2")
a, b = rows(V7Q), rows(OUT / "player_v7q_csls_k3.csv"); assert a[0] == b[0] and len(a) == len(b) == 1601 and [r[0] for r in a] == [r[0] for r in b]
R["first800_identical_to_v7q"] = a[1:801] == b[1:801]
for nm, pref in (("Public", "pub2_"), ("Private", "priv2_")):
    ia = [i for i in range(801, 1601) if a[i][0].startswith(pref)]
    R[nm] = dict(rows=len(ia), top1_changed=sum(a[i][1] != b[i][1] for i in ia), top5_ordered_list_changed=sum(a[i][1:] != b[i][1:] for i in ia), top5_set_changed=sum(set(a[i][1:]) != set(b[i][1:]) for i in ia))
R["any_old_row_prefix_in_new_block"] = any(not a[i][0].startswith(("pub2_", "priv2_")) for i in range(801, 1601))
with open(OUT / "make_result.json", "x") as f: f.write(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1, ensure_ascii=False))
