"""E99: COPY of scripts/experiments/e90/assemble_pe_h.py for the BIG finh_* networks (16 x 224, 12 epochs, accum 8).  assemble_pe_h.py / assemble_pe.py stay untouched (build() is imported).
Why a copy: assemble_pe_h.py asserts results.json steps == the 20-epoch values (77,740 / 49,360 / 50,820), which can never hold for the E99 networks (steps = steps_per_epoch x epochs).
Same layout and file names as assemble_pe_h.py:
    COLAB_DIR/finh_h0/vec_fin_h0.npz           -> pe_final_h0.npy   (206,536 x 128)
    COLAB_DIR/finh_h1/vec_fin_h1.npz           -> pe_final_h1.npy
    COLAB_DIR/finh_full/vec_fin_full_test.npz  -> pe_test_pub.npy / pe_test_priv.npy   (7,100 x 128 each)
Guards (refuse otherwise): vec `net` field == network name; results.json (if present) roster == name, blocks 16, ch 224, accum 8, n_train / n_classes as in data_stats_h.json,
steps == steps_per_epoch x epochs == expected_steps, and `epochs` is the SAME for every network found (cosine schedule); 8 <= epochs <= 20.
usage: assemble_pe_e99.py [NETS_DIR (default .scratch/e99/nets)] [OUT_DIR (default .scratch/e99/r)] [--check-only]
Default OUT_DIR is .scratch/e99/r (NOT .scratch/e90/r_h, which holds the small-network files); ~106 MB per pe_final file, 3.6 MB per pe_test file.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; DATA = ROOT / ".scratch/e90/data"
sys.path[:0] = [str(ROOT / "scripts/experiments/e90")]
import numpy as np
from assemble_pe import build

EXPECT = {"finh_full": dict(n_train=994_875, n_classes=7_365, steps_per_epoch=3_887), "finh_h0": dict(n_train=631_584, n_classes=3_992, steps_per_epoch=2_468), "finh_h1": dict(n_train=650_346, n_classes=4_013, steps_per_epoch=2_541)}
JOBS = [("finh_h0", "fin_h0"), ("finh_h1", "fin_h1"), ("finh_full", "fin_full_test")]        # (network dir, export entry name)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]; check_only = "--check-only" in sys.argv
    src = Path(args[0]) if args else ROOT / ".scratch/e99/nets"; out = Path(args[1]) if len(args) > 1 else ROOT / ".scratch/e99/r"
    assert out.resolve() not in ((ROOT / ".scratch/e90/r").resolve(), (ROOT / ".scratch/e90/r_h").resolve()), "OUT_DIR must not be a small-network output folder"
    spec = json.loads((DATA / "export_spec_h.json").read_text()); arr = np.load(DATA / "export_spec_h.npz"); made = []; missing = []; epochs_seen = {}
    for net, ename in JOBS:
        entry = next(e for e in spec["exports"][net] if e["name"] == ename); f = src / net / f"vec_{ename}.npz"
        if not f.exists(): missing.append(str(f)); print("MISSING", f); continue
        vec = np.load(f); assert str(vec["net"]) == net, f"{f}: net field {vec['net']} != {net} (a file of another network?)"
        rj = src / net / "results.json"
        if rj.exists():
            r = json.loads(rj.read_text()); ex = EXPECT[net]
            assert r["roster"] == net and (r["blocks"], r["ch"], r["accum"]) == (16, 224, 8) and 8 <= r["epochs"] <= 20, (net, {k: r.get(k) for k in ("roster", "blocks", "ch", "accum", "epochs")})
            assert r["steps"] == ex["steps_per_epoch"] * r["epochs"] == r["expected_steps"] and r["n_classes"] == ex["n_classes"] and r["n_train"] == ex["n_train"], (net, {k: r.get(k) for k in ("roster", "steps", "expected_steps", "n_classes", "n_train", "epochs")}, ex)
            assert int(vec["epoch"]) == r["epochs"], f"{f}: vec epoch {int(vec['epoch'])} != results epochs {r['epochs']}"
            epochs_seen[net] = r["epochs"]; print("results.json OK", net, "16x224 accum 8 epochs", r["epochs"], "steps", r["steps"], "classes", r["n_classes"], "n_train", r["n_train"])
        else: print("WARNING: no results.json next to", f, "(training success marker not checked)")
        for fn, A in build(vec, entry, arr).items():
            if not check_only: out.mkdir(parents=True, exist_ok=True); np.save(out / fn, A)
            made.append((fn, A.shape)); print("ok", fn, A.shape, "(not written)" if check_only else f"-> {out / fn}")
    assert len(set(epochs_seen.values())) <= 1, f"the three networks must have the SAME epochs: {epochs_seen}"
    if missing: print("INCOMPLETE: missing", missing)
    return made


if __name__ == "__main__":
    main()
