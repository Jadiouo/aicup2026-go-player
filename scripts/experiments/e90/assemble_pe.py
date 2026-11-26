"""E90: turn the per-game vector files exported by train_e90.py (vec_<name>.npz, compact: one row per embedded game) into the files the E80/E77 Player pipeline reads
(interface notes .scratch/e90/interface_notes.md Q1.2, E80 scripts/oneoff/e80_enc.py):
  sim_h0 / sim_h1 / sim_full  vec_<net>.npz  -> pe_sim_<h0|h1|full>.npy     (223,693, 128) float32, row = sim meta row; rows outside sim_rows are 0; sim_rows rows unit length
  fin_h0 / fin_h1             vec_<net>.npz  -> pe_final_<h0|h1>.npy        (206,536, 128) float32, row = D row; rows outside fin_rows are 0
  fin_full (test)             vec_fin_full_test.npz -> pe_test_pub.npy / pe_test_priv.npy  (7,100, 128) each, order = Z_{pub,priv}.npz uids
  usage: assemble_pe.py IN_DIR [OUT_DIR] [--check-only]     IN_DIR holds vec_*.npz (local copy from the remote run dir); OUT_DIR default .scratch/e90/r
build() is a pure function (used by the V-tests); main() writes the files (about 114 MB each for pe_sim_*, 106 MB for pe_final_*, 3.6 MB for pe_test_*).
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; DATA = ROOT / ".scratch/e90/data"
import numpy as np


def build(vec, entry, arr):
    """vec: loaded vec_<name>.npz (dict-like); entry: export_spec.json entry; arr: export_spec.npz.  Returns {out_filename: array}."""
    rows = arr[entry["rows"]].astype(np.int64); gv = np.asarray(vec["gv_all"], np.float32)
    assert (np.asarray(vec["ids"]) == rows).all() and gv.shape == (len(rows), 128), (entry["name"], gv.shape, len(rows))
    assert np.isfinite(gv).all() and np.abs(np.linalg.norm(gv, axis=1) - 1).max() < 1e-3, f"{entry['name']}: vectors not unit length"
    if "scatter" in entry:
        sc = arr[entry["scatter"]].astype(np.int64); assert len(sc) == len(rows) and len(np.unique(sc)) == len(sc) and sc.max() < entry["n_out"]
        A = np.zeros((entry["n_out"], 128), np.float32); A[sc] = gv; return {entry["out"][0]: A}
    if entry["name"] == "fin_full_test":
        assert len(gv) == 14_200; return {"pe_test_pub.npy": gv[:7100].copy(), "pe_test_priv.npy": gv[7100:].copy()}
    return {}                                                                       # optional Rank-leg exports: used directly from vec_<name>.npz


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]; check_only = "--check-only" in sys.argv
    src = Path(args[0]); out = Path(args[1]) if len(args) > 1 else ROOT / ".scratch/e90/r"
    spec = json.loads((DATA / "export_spec.json").read_text()); arr = np.load(DATA / "export_spec.npz"); made = []
    for net, entries in spec["exports"].items():
        for e in entries:
            f = src / f"vec_{e['name']}.npz"
            if not f.exists():
                if e["required"]: print("MISSING (required)", f)
                continue
            for fn, A in build(np.load(f), e, arr).items():
                if not check_only: out.mkdir(parents=True, exist_ok=True); np.save(out / fn, A)
                made.append((fn, A.shape)); print("ok", fn, A.shape, "(not written)" if check_only else f"-> {out / fn}")
    return made


if __name__ == "__main__":
    main()
