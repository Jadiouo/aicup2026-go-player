"""E80 GPU half 2 (system python3 + torch; submit with gpujob, or run on remote-vm with E77_DEV=cuda): trains the E77 position encoder (e77_enc.train; attn, p_in 0.3, 6000 steps, seed 0 = .scratch/e77/selected.json, NO retuning) on the FULL non-H data (PLAN section 1):
  sim   : h0 / h1 / full  (sim EVAL players never enter; meta.npz masks sim_*)       -> .scratch/e80/pe_sim_{h0,h1,full}.npy   (223,693,128) rows = sim meta index (rows outside cap_rows 0)
  final : h0 / h1 / full  (all non-H; H never)                                       -> .scratch/e80/pe_final_{h0,h1}.npy (nrows,128) rows = D rows ; pe_test_{pub,priv}.npy (7100,128) from the final 'full' encoder
Tokens: E79 tok_full (712,945,24,64) fp16 PCA vectors + oidx bucket (x5 -> Enc does //5) + colour.  Test tokens from e80_extract.py (tok_pub/tok_priv.npz).  Per-encoder standardisation = mean/std over the training games' tokens (chunked).
Each encoder is skipped if its outputs exist (resumable).  usage: e80_enc.py sim|final [--only h0,h1,full]"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).resolve().parent)); K = ROOT / ".scratch" / "e80"
import e77_enc as M
dev = M.dev; log = M.log
class FD: pass
def mk(X, mask, oidx_bucket, col):
    D = FD(); D.X = torch.from_numpy(X).to(dev); D.mask = torch.from_numpy(mask).to(dev); D.oidx = torch.from_numpy(oidx_bucket.astype(np.int64) * 5).to(dev); D.col = torch.from_numpy(col.astype(np.int64)).to(dev); return D
def stats(D, idx, chunk=50000):
    s = torch.zeros(D.X.shape[2], device=dev, dtype=torch.float64); n = 0.0
    for a in range(0, len(idx), chunk):
        g = torch.from_numpy(idx[a:a + chunk]).to(dev); x = D.X[g].double(); m = D.mask[g][:, :, None].double(); s += (x * m).sum((0, 1)); n += float(m.sum())
    mu = s / n; v = torch.zeros_like(s)
    for a in range(0, len(idx), chunk):
        g = torch.from_numpy(idx[a:a + chunk]).to(dev); x = D.X[g].double(); m = D.mask[g][:, :, None].double(); v += (((x - mu) ** 2) * m).sum((0, 1))
    return mu.float(), (v / n).sqrt().float() + 1e-6
def main(name, only):
    sel = json.load(open(ROOT / ".scratch/e77/selected.json")); arch, p_in, steps = sel["arch"], sel["p_in"], sel["steps"]; log("config", arch, p_in, steps, "(E77 selected, not retuned)")
    z = np.load(K / "meta.npz"); player = z["player"]
    X = np.load(ROOT / ".scratch/e79/tok_full.npy", mmap_mode="r"); X = np.ascontiguousarray(X); D = mk(X, np.load(ROOT / ".scratch/e79/tok_full_mask.npy"), np.load(ROOT / ".scratch/e79/tok_full_oidx.npy"), np.load(ROOT / ".scratch/e79/tok_full_gcol.npy")); del X
    log("tokens on", dev, tuple(D.X.shape), f"maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    if name == "sim": rows, n_out, gid = z["sim_rows"], 223_693, z["sim_gid"]
    else: rows, n_out, gid = z["fin_rows"], int(z["fin_nrows"]), z["fin_gid"]
    cfgs = [c for c in ("h0", "h1", "full") if c in only]
    for cfg in cfgs:
        f = K / (f"pe_{name}_{cfg}.npy" if not (name == "final" and cfg == "full") else "pe_test_priv.npy")
        if f.exists(): log("skip", name, cfg); continue
        ok = z[f"{name[:3] if name == 'sim' else 'fin'}_{cfg}"]; gof = {}
        for g in np.flatnonzero(ok): gof.setdefault(int(player[g]), []).append(g)
        elig = [p for p, v in gof.items() if len(v) >= 12]; games_of = {p: np.array(v) for p, v in gof.items()}; tr_idx = np.flatnonzero(ok)
        mu, sd = stats(D, tr_idx); log(name, cfg, "players", len(elig), "train games", len(tr_idx), "max games/player", max(len(v) for v in gof.values()), f"maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
        t0 = time.time(); model = M.train(D, arch, p_in, steps, 0, mu, sd, games_of, elig, tag=f"{name}_{cfg}"); log("trained", f"{time.time() - t0:.0f}s")
        E = M.embed_np(model, D, rows, mu, sd).astype(np.float32); nrm = np.linalg.norm(E, axis=1); assert np.isfinite(E).all() and np.abs(nrm - 1).max() < 1e-3
        torch.save({"sd": model.state_dict(), "mu": mu.cpu(), "std": sd.cpu()}, K / f"enc_{name}_{cfg}.pt")
        if name == "sim":
            A = np.zeros((n_out, 128), np.float32); A[gid] = E; np.save(K / f"pe_sim_{cfg}.npy", A)
            if cfg == "full":
                ev = np.flatnonzero(z["sim_eval"]); log("INFO eval-player retrieval (not used for any selection; E77 .327):", M.val_score(E[ev], player[rows[ev]]))
        else:
            if cfg != "full": A = np.zeros((n_out, 128), np.float32); A[gid] = E; np.save(K / f"pe_final_{cfg}.npy", A)
            else:
                for si, s in enumerate(("pub", "priv")):
                    t = np.load(K / f"tok_{s}.npz"); Dt = mk(t["X"], t["mask"], t["oidx"], t["gcol"]); Et = M.embed_np(model, Dt, np.arange(len(t["X"])), mu, sd).astype(np.float32)
                    assert np.isfinite(Et).all() and np.abs(np.linalg.norm(Et, axis=1) - 1).max() < 1e-3 and len(Et) == 7100; np.save(K / f"pe_test_{s}.npy", Et); del Dt
        log("saved", name, cfg, E.shape); del model; torch.cuda.empty_cache()
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--only", default="h0,h1,full"); a = ap.parse_args(); log("args", vars(a)); main(a.name, set(a.only.split(","))); log("done")
