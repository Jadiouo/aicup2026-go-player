"""E90 delivery version: COPY of assemble_pe.py for the finh_* networks (trained on the original fin lists + all H players).  assemble_pe.py itself is untouched and its build() is imported.
Exported file names do not change with the roster (train_e90h.py keeps the export-spec names: vec_fin_h0.npz, vec_fin_h1.npz, vec_fin_full_test.npz), but they live in one directory per network:
    COLAB_DIR/finh_h0/vec_fin_h0.npz     -> pe_final_h0.npy   (206,536 x 128; row = D row; 109,560 rows covered, rest 0)
    COLAB_DIR/finh_h1/vec_fin_h1.npz     -> pe_final_h1.npy
    COLAB_DIR/finh_full/vec_fin_full_test.npz -> pe_test_pub.npy / pe_test_priv.npy   (7,100 x 128 each)
Guards (refuse otherwise): the vec file's `net` field must equal the finh_* roster name (a fin_* file cannot slip in); results.json (if present) must say roster == name, steps == expected steps, n_classes and n_train as in
.scratch/e90/data/data_stats_h.json; export_spec_h.npz indices are used (byte-identical to export_spec.npz).
usage: assemble_pe_h.py [COLAB_DIR (default .scratch/e90/colab)] [OUT_DIR (default .scratch/e90/r_h)] [--check-only]
Output default is .scratch/e90/r_h (NOT .scratch/e90/r, which belongs to the simulated-stage networks).  About 106 MB per pe_final file, 3.6 MB per pe_test file.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; DATA = ROOT / ".scratch/e90/data"
sys.path[:0] = [str(HERE)]
import numpy as np
from assemble_pe import build

EXPECT = {"finh_full": dict(n_train=994_875, n_classes=7_365, steps=77_740), "finh_h0": dict(n_train=631_584, n_classes=3_992, steps=49_360), "finh_h1": dict(n_train=650_346, n_classes=4_013, steps=50_820)}
JOBS = [("finh_h0", "fin_h0"), ("finh_h1", "fin_h1"), ("finh_full", "fin_full_test")]        # (network dir, export entry name)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]; check_only = "--check-only" in sys.argv
    src = Path(args[0]) if args else ROOT / ".scratch/e90/colab"; out = Path(args[1]) if len(args) > 1 else ROOT / ".scratch/e90/r_h"
    assert out.resolve() != (ROOT / ".scratch/e90/r").resolve(), "OUT_DIR must not be .scratch/e90/r (simulated-stage files)"
    spec = json.loads((DATA / "export_spec_h.json").read_text()); arr = np.load(DATA / "export_spec_h.npz"); made = []; missing = []
    for net, ename in JOBS:
        entry = next(e for e in spec["exports"][net] if e["name"] == ename); f = src / net / f"vec_{ename}.npz"
        if not f.exists(): missing.append(str(f)); print("MISSING", f); continue
        vec = np.load(f); assert str(vec["net"]) == net, f"{f}: net field {vec['net']} != {net} (a file of another network?)"
        rj = src / net / "results.json"
        if rj.exists():
            r = json.loads(rj.read_text()); ex = EXPECT[net]
            assert r["roster"] == net and r["steps"] == ex["steps"] == r["expected_steps"] and r["n_classes"] == ex["n_classes"] and r["n_train"] == ex["n_train"], (net, {k: r.get(k) for k in ("roster", "steps", "expected_steps", "n_classes", "n_train")}, ex)
            print("results.json OK", net, "steps", r["steps"], "classes", r["n_classes"], "n_train", r["n_train"])
        else: print("WARNING: no results.json next to", f, "(training success marker not checked)")
        for fn, A in build(vec, entry, arr).items():
            if not check_only: out.mkdir(parents=True, exist_ok=True); np.save(out / fn, A)
            made.append((fn, A.shape)); print("ok", fn, A.shape, "(not written)" if check_only else f"-> {out / fn}")
    if missing: print("INCOMPLETE: missing", missing)
    return made


if __name__ == "__main__":
    main()
