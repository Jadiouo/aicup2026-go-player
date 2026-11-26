"""E77 GPU half 1, full scale (system python3 + torch; submit with gpujob).  Streams the blocks written by e77_prep_full.py: per position u=[at|mean] (768, act_trunkfinal, e70_gpu.trunk_u) -> standardise + PCA(96) -> fp16 blk{i}_pc.npy (bits deleted after use).
Step 'pca': .scratch/e77/pca/ (1,000 sim-train players x 4 games, both windows) -> PCA fitted ONLY on these positions -> .scratch/e77/pca_full.npz.  Then 'sim', 'final' in the same job (blocks as they appear; exits a set when PREP_DONE and nothing is left, or after --idle-exit seconds without a ready block)."""
import sys, os, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts" / "oneoff"))
import numpy as np, torch
import e70_gpu as G
K = ROOT / ".scratch" / "e77"; dev = G.dev; log = G.log
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=str(K / "ckpt.ckpt")); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--idle-exit", type=int, default=3600); ap.add_argument("--sets", default="pca,sim,final"); a = ap.parse_args()
model = G.load(a.ckpt)
@torch.no_grad()
def u_block(d, bi):
    bits = np.load(d / f"blk{bi}_bits.npy", mmap_mode="r"); glob = np.load(d / f"blk{bi}_glob.npy"); loc = np.load(d / f"blk{bi}_loc.npy").astype(np.int64); color = np.load(d / f"blk{bi}_color.npy"); N = len(bits); out = np.zeros((N, 768), np.float16)
    for s in range(0, N, a.bs):
        e = min(s + a.bs, N); b = np.unpackbits(np.asarray(bits[s:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
        bx = torch.from_numpy(b).to(dev).float(); gx = torch.from_numpy(glob[s:e]).to(dev); mx = torch.stack([G.MROWS[int(c)] for c in color[s:e]]).to(dev); l = torch.from_numpy(loc[s:e]).to(dev)
        out[s:e] = G.trunk_u(model, bx, gx, mx, l).cpu().numpy().astype(np.float16)
    return out
def wait_block(d, bi, t_last):
    while True:
        if (d / f"blk{bi}_bits.npy").exists() and (d / f"blk{bi}_oidx.npy").exists(): time.sleep(1); return True
        if (d / "PREP_DONE").exists() and not (d / f"blk{bi}_bits.npy").exists(): return False
        if time.time() - t_last > a.idle_exit: log("idle exit", d); return False
        time.sleep(10)
def do_pca():
    d = K / "pca"; U = []; bi = 0
    while wait_block(d, bi, time.time()): U.append(u_block(d, bi)); os.remove(d / f"blk{bi}_bits.npy"); bi += 1
    X = np.concatenate(U).astype(np.float64); mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; w, V = np.linalg.eigh(Z.T @ Z / len(Z)); o = np.argsort(w)[::-1][:96]; comps = V[:, o].T.astype(np.float32); evr = w[o] / w.sum()
    log(f"PCA(full) fitted on {len(X)} positions (1,000 sim-train players x 4 games, both windows); explained variance of 96 comps {evr.sum():.4f}; first 10 {np.round(evr[:10], 4).tolist()}")
    np.savez(K / "pca_full.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32), comps=comps, evr=evr)
def do_set(name):
    d = K / name; z = np.load(K / "pca_full.npz"); mu, sd, comps = z["mu"], z["sd"], z["comps"]; bi = 0; tot = 0; t0 = time.time()
    while True:
        if (d / f"blk{bi}_pc.npy").exists(): bi += 1; continue
        if not wait_block(d, bi, time.time()): break
        U = u_block(d, bi); P = (((U.astype(np.float32) - mu) / sd) @ comps.T).astype(np.float16); assert np.isfinite(P).all()
        np.save(d / f"blk{bi}_pc_tmp.npy", P); os.replace(d / f"blk{bi}_pc_tmp.npy", d / f"blk{bi}_pc.npy"); os.remove(d / f"blk{bi}_bits.npy"); tot += len(U); log(name, "block", bi, "positions so far", tot, f"{time.time() - t0:.0f}s maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"); bi += 1
for s in a.sets.split(","): log("== set", s); do_pca() if s == "pca" else do_set(s)
log("done")
