"""E86 GPU half (system python3 + torch; submit with gpujob): consumes .scratch/e86/<stage>/blocks/{name}_bits... as written by e86_prep.py: u=[at|mean] (act_trunkfinal, e70_gpu.trunk_u) -> standardise with E77 pca_full.npz -> first 64 PCs (= e79_gpu/e80_extract maths)
-> {name}_pc.npy (fp16); bits/glob/loc deleted after use.  Exits when PREP_DONE exists and nothing is left, or after --idle-exit s without a ready block.  usage: e86_gpu.py STAGE [--ckpt F] [--bs 512]"""
import sys, os, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(ROOT / "src"), str(ROOT)]
import numpy as np, torch
import e70_gpu as G
dev = G.dev; log = G.log
ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--ckpt", default=str(ROOT / ".scratch/e86/ckpt.ckpt")); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--idle-exit", type=int, default=5400); a = ap.parse_args()
B = ROOT / ".scratch/e86" / a.stage / "blocks"; B.mkdir(parents=True, exist_ok=True)
model = G.load(a.ckpt); z = np.load(ROOT / ".scratch/e77/pca_full.npz"); MU, SD, COMPS = z["mu"], z["sd"], z["comps"][:64]; assert COMPS.shape == (64, 768)
@torch.no_grad()
def u_block(bits, glob, loc, color):
    N = len(bits); out = np.zeros((N, 768), np.float16); loc = loc.astype(np.int64)
    for s in range(0, N, a.bs):
        e = min(s + a.bs, N); b = np.unpackbits(np.asarray(bits[s:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
        bx = torch.from_numpy(b).to(dev).float(); gx = torch.from_numpy(glob[s:e]).to(dev); mx = torch.stack([G.MROWS[int(c)] for c in color[s:e]]).to(dev); l = torch.from_numpy(loc[s:e]).to(dev)
        out[s:e] = G.trunk_u(model, bx, gx, mx, l).cpu().numpy().astype(np.float16)
    return out
t0 = time.time(); tot = 0; last = time.time()
while True:
    ready = sorted((p.name[:-len("_bits.npy")] for p in B.glob("*_bits.npy") if not (B / (p.name[:-len("_bits.npy")] + "_pc.npy")).exists()), key=lambda s: int(s[1:]))
    if ready:
        nm = ready[0]; bits = np.load(B / f"{nm}_bits.npy", mmap_mode="r"); U = u_block(bits, np.load(B / f"{nm}_glob.npy"), np.load(B / f"{nm}_loc.npy"), np.load(B / f"{nm}_color.npy"))
        P = (((U.astype(np.float32) - MU) / SD) @ COMPS.T).astype(np.float16); assert np.isfinite(P).all() and len(P) == len(bits); np.save(B / f"{nm}_pc_tmp.npy", P); os.replace(B / f"{nm}_pc_tmp.npy", B / f"{nm}_pc.npy"); del bits
        for s in ("bits", "glob", "loc"): os.remove(B / f"{nm}_{s}.npy")
        tot += len(P); last = time.time(); log(nm, "positions so far", tot, f"{time.time() - t0:.0f}s maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"); continue
    if (B.parent / "PREP_DONE").exists(): break
    if time.time() - last > a.idle_exit: log("idle exit"); break
    time.sleep(10)
log("done", tot)
