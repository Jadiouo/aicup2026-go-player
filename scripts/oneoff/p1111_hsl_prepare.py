#!/usr/bin/env python
"""P1111: `predict_hsl.py prepare` for the games that still need HSL (.scratch/p1111/plan/hsl_todo.parquet from p1111_inspect.py).

Same idea as scripts/oneoff/e27_hsl_prepare.py (reuses predict_hsl.prepare unchanged, patching only collect_test_games), but the games come from the plan,
no rank files are needed, and the shard ids are renamed S9001.. -> S9201.. so they cannot clash with the S9001-S9003 (Rank) and S9101-S9108 (E27 Player)
results that still sit in the school dir ~/gogame_scan_1104/out.
Usage:  GOGAME_COLOR_SEMANTICS=actual .venv/bin/python scripts/oneoff/p1111_hsl_prepare.py ROOT      (ROOT = data/test_scan_p1111, must be NEW, never data/test_scan)
Then:   GOGAME_COLOR_SEMANTICS=actual SCAN_SCHOOL_LIVE=1 nohup .venv/bin/python scripts/predict_hsl.py --root ROOT run --backend school --share-gpu --hours 10 --poll-seconds 120 > ROOT/run.log 2>&1 &
"""
import json, os, sys
os.environ.setdefault("GOGAME_COLOR_SEMANTICS", "actual")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
import pandas as pd
import predict_hsl as H

PLAN = Path(os.environ["P1111_PLAN"]) if os.environ.get("P1111_PLAN") else ROOT / ".scratch" / "p1111" / "plan"
root = Path(sys.argv[1]).resolve()
assert root != (ROOT / "data" / "test_scan").resolve() and not (root / "ledger.parquet").exists(), "root must be new (and never data/test_scan)"
todo = pd.read_parquet(PLAN / "hsl_todo.parquet")
if not len(todo):
    print("hsl_todo is empty: every needed HSL row is reused from 11/04; nothing to prepare"); sys.exit(0)
if (PLAN / "games_new.parquet").exists():
    gm = pd.read_parquet(PLAN / "games_new.parquet").drop_duplicates("uid").set_index("uid")
else:   # plan_all (11/11 test2, uids already in the stripped Z-cache scheme, e.g. pub_q_0001#1 / cand_pub:pub_g_0004082): qid/split/slot are manifest metadata only
    u = todo["uid"].astype(str)
    sp = np.where(u.str.contains("priv"), "priv", "pub")
    pid = u.str.split("#").str[0]
    slot = pd.to_numeric(u.str.split("#").str[1], errors="coerce").fillna(0).astype(int)
    gm = pd.DataFrame(dict(uid=u, pid=pid, split=sp, slot=slot)).drop_duplicates("uid").set_index("uid")


def collect(_data_dir):
    d = todo.copy()
    g = gm.loc[d.uid]
    d["qid"] = g.pid.to_numpy(); d["split"] = g.split.to_numpy(); d["slot"] = g.slot.to_numpy().astype(int)
    d["color"] = d["color"].astype(np.int8)
    return d[["uid", "color", "sgf", "qid", "split", "slot"]]


H.collect_test_games = collect
H.RANK_FILES = ()
meta = H.prepare(Path("."), root)
ren = {f"S{9000 + k:04d}": f"S{9200 + k:04d}" for k in range(1, 60)}
for name in ("shards.parquet", "manifest.parquet", "ledger.parquet"):
    d = pd.read_parquet(root / name)
    d["shard_id"] = d["shard_id"].map(lambda s: ren.get(s, s))
    d.to_parquet(root / name, index=False)
for f in sorted((root / "sgf").glob("S9*.parquet")):
    f.rename(f.with_name(ren[f.stem] + ".parquet"))
pm = json.loads((root / "prepare_meta.json").read_text())
pm["note"] = "P1111 Player-test games still needing HSL; shard ids renamed S92xx; no rank inputs"
pm["inputs"] = {"hsl_todo.parquet": H.sha256_file(PLAN / "hsl_todo.parquet")}
(root / "prepare_meta.json").write_text(json.dumps(pm, indent=1, default=str))
print(pd.read_parquet(root / "ledger.parquet")[["shard_id", "n_games", "status"]].to_string())
n = int(pm["n_games"]); print(f"-> {n} games; ETA on school ~{n / 1.7 / 60:.0f} min at 1.7 games/s (E27: 1.37-1.76)")
