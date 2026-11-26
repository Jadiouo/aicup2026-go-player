"""E27: `predict_hsl.py prepare` for the Player-test games chosen by e27_player_fusion.py games (.scratch/e27/hsl_games_in.parquet).
Reuses predict_hsl.prepare unchanged (patching only collect_test_games), then renames the shard ids S9001.. -> S9101.. so they cannot clash
with the Rank test shards S9001-S9003 whose results still sit in the school dir ~/gogame_scan_1104/out.
Usage: GOGAME_COLOR_SEMANTICS=actual e27_hsl_prepare.py ROOT      (ROOT must be a NEW directory, never data/test_scan)"""
import os, sys, json
os.environ.setdefault("GOGAME_COLOR_SEMANTICS", "actual")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np, pandas as pd
import predict as P
import predict_hsl as H

root = Path(sys.argv[1]).resolve()
assert root != (ROOT / "data" / "test_scan").resolve() and not (root / "ledger.parquet").exists(), "root must be new"
DATA = ROOT / "data" / "contest_1104"
sel = pd.read_parquet(ROOT / ".scratch" / "e27" / "hsl_games_in.parquet")
info = {}
for sp in ("pub", "priv"):
    for p, gs in P.load_candidates(DATA / P.FILES[f"cand_{sp}"], sp).items():
        for i, g in enumerate(gs, 1):
            info[g.uid] = (p, sp, i)
    for q in P.load_questions(DATA / P.FILES[f"player_{sp}"], sp, "player"):
        for i, g in enumerate(q.games, 1):
            info[g.uid] = (q.qid, sp, i)


def collect(_data_dir):
    d = sel.copy()
    d["qid"] = [info[u][0] for u in d.uid]; d["split"] = [info[u][1] for u in d.uid]; d["slot"] = [info[u][2] for u in d.uid]
    d["color"] = d["color"].astype(np.int8)
    return d[["uid", "color", "sgf", "qid", "split", "slot"]]


H.collect_test_games = collect
H.RANK_FILES = ("rank_pub", "rank_priv")     # prepare() only hashes these input files into prepare_meta (unused here)
meta = H.prepare(DATA, root)
# rename shard ids S9001.. -> S9101..
ren = {f"S{9000 + k:04d}": f"S{9100 + k:04d}" for k in range(1, 40)}
for name in ("shards.parquet", "manifest.parquet", "ledger.parquet"):
    d = pd.read_parquet(root / name)
    d["shard_id"] = d["shard_id"].map(lambda s: ren.get(s, s))
    d.to_parquet(root / name, index=False)
for f in sorted((root / "sgf").glob("S9*.parquet")):
    f.rename(f.with_name(ren[f.stem] + ".parquet"))
pm = json.loads((root / "prepare_meta.json").read_text())
pm["note"] = "E27 Player-test games; shard ids renamed S91xx; rank_pub/rank_priv input hashes above are not inputs of this scan"
pm["inputs"] = {"hsl_games_in.parquet": H.sha256_file(ROOT / ".scratch" / "e27" / "hsl_games_in.parquet")}
(root / "prepare_meta.json").write_text(json.dumps(pm, indent=1, default=str))
print(pd.read_parquet(root / "ledger.parquet")[["shard_id", "n_games", "status"]].to_string())
