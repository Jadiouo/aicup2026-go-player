"""E27: merge the 9 school SE chunks (.scratch/e27/se_out/c*/) into .scratch/e27/se/se_test_games.parquet after validating every manifest."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[2] / ".scratch" / "e27"
want = pd.read_parquet(R / "se_games.parquet", columns=["uid"]).uid
parts, mans = [], []
for d in sorted((R / "se_out").glob("c*")):
    m = json.loads((d / "se_manifest.json").read_text()); mans.append(m)
    assert m["color_semantics"] == "actual" and m["device"] == "cuda" and m["half"] and m["n_nan_not_hd_ge2"] == 0 and m["n_se_n_lt32"] == 0, (d, m)
    assert m["model_sha256"] == "95fc295137b1fd09a0ec4d5cc79ed888f8c12d0ba6a77fea58b24efc3f418126"
    parts.append(pd.read_parquet(d / "se_test_games.parquet"))
assert len({m["torch"] for m in mans}) == 1
se = pd.concat(parts, ignore_index=True)
assert len(parts) == 9 and len(se) == len(want) == 90000 and set(se.uid) == set(want) and not se.uid.duplicated().any()
assert not se[["se_mean", "se_std"]].isna().any().any() and (se.se_n == 32).all()
(R / "se").mkdir(exist_ok=True)
se.to_parquet(R / "se" / "se_test_games.parquet", index=False)
print("merged", len(se), "chunks", len(parts), "seconds", [m["seconds"] for m in mans])
print("test  se_mean mean %.3f sd %.3f | se_std mean %.3f" % (se.se_mean.mean(), se.se_mean.std(), se.se_std.mean()))
tp = R.parents[0] / "e21" / "se_uid.parquet"      # optional sanity vs the training SE table (black games): location/scale of se_mean
if tp.exists():
    tr = pd.read_parquet(tp)
    print("train se_mean mean %.3f sd %.3f | se_std mean %.3f" % (tr.se_mean.mean(), tr.se_mean.std(), tr.se_std.mean()))
