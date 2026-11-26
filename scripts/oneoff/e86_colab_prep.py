"""E86 full stage step 2 (runs ON THE COLAB VM, /content/Gogame): all own-move positions (idx [2,min(120,n_own))) of every item in /content/items_full.pkl:
CPU prep (e86_prep.work_all, N worker processes) overlapped with the GPU trunk (e70_gpu.trunk_u -> E77 PCA first 64 comps, fp16) -> /content/out/b{i}_{pc,rows,oidx,rep,color}.npy (1,000 games per block).
Resumable (skips blocks whose pc exists).  Heartbeat /content/out/heartbeat.log (one line per block), /content/out/PREP_DONE at the end.  usage: e86_colab_prep.py [--workers 11] [--blk 1000]"""
import os, sys, time, argparse, pickle, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
os.environ.setdefault("E86_CHK", "/content/chk_e79.npz")
from pathlib import Path
ROOT = Path("/content/Gogame"); sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts" / "oneoff")]
import numpy as np
from multiprocessing import Pool
import e86_prep as P86
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--blk", type=int, default=1000); ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--items", default="/content/items_full.pkl"); ap.add_argument("--ckpt", default="/content/ckpt.ckpt"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
OUT = Path("/content/out"); OUT.mkdir(exist_ok=True); HB = OUT / "heartbeat.log"
def hb(*x):
    with open(HB, "a") as f: f.write(time.strftime("%H:%M:%S ") + " ".join(str(i) for i in x) + "\n")
items = pickle.load(open(a.items, "rb")); items = items[:a.limit] if a.limit else items; hb("items", len(items))
pool = Pool(a.workers)                      # fork BEFORE CUDA is initialised
import torch, collections, e70_gpu as G
import functools; _tl = torch.load; torch.load = lambda *a_, **k_: _tl(*a_, **{**k_, 'weights_only': False})   # torch>=2.6 default weights_only=True rejects the official ckpt (sha256-verified release file)
dev = G.dev; model = G.load(a.ckpt); z = np.load(ROOT / ".scratch/e77/pca_full.npz"); MU, SD, COMPS = z["mu"], z["sd"], z["comps"][:64]; assert COMPS.shape == (64, 768)
@torch.no_grad()
def u_block(bits, glob, loc, color):
    N = len(bits); out = np.zeros((N, 768), np.float16); loc = loc.astype(np.int64)
    for s in range(0, N, a.bs):
        e = min(s + a.bs, N); b = np.unpackbits(np.asarray(bits[s:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
        bx = torch.from_numpy(b).to(dev).float(); gx = torch.from_numpy(glob[s:e]).to(dev); mx = torch.stack([G.MROWS[int(c)] for c in color[s:e]]).to(dev); l = torch.from_numpy(loc[s:e]).to(dev)
        out[s:e] = G.trunk_u(model, bx, gx, mx, l).cpu().numpy().astype(np.float16)
    return out
nb = (len(items) + a.blk - 1) // a.blk; t0 = time.time(); tot = 0
def submit(bi):
    its = items[bi * a.blk:(bi + 1) * a.blk]; return its, pool.map_async(P86.work_all, [its[i:i + 25] for i in range(0, len(its), 25)], chunksize=1)
nxt = None; todo = [bi for bi in range(nb) if not (OUT / f"b{bi}_pc.npy").exists()]; hb("blocks", nb, "todo", len(todo))
if todo: nxt = (todo[0],) + submit(todo[0])
for k, bi in enumerate(todo):
    cur_bi, its, fut = nxt; assert cur_bi == bi
    if k + 1 < len(todo): nxt = (todo[k + 1],) + submit(todo[k + 1])
    res = [x for x in fut.get() if x is not None]; t1 = time.time(); rw = np.concatenate([x[3] for x in res]); colmap = {it[3]: it[1] for it in its}
    bits, glob, loc = np.concatenate([x[0] for x in res]), np.concatenate([x[1] for x in res]), np.concatenate([x[2] for x in res]); color = np.array([colmap[int(x)] for x in rw[:, 0]], np.int8)
    U = u_block(bits, glob, loc, color); Pc = (((U.astype(np.float32) - MU) / SD) @ COMPS.T).astype(np.float16); assert np.isfinite(Pc).all() and len(Pc) == len(rw)
    nm = f"b{bi}"; np.save(OUT / f"{nm}_rows.npy", rw); np.save(OUT / f"{nm}_oidx.npy", np.concatenate([x[4] for x in res])); np.save(OUT / f"{nm}_rep.npy", np.concatenate([x[5] for x in res])); np.save(OUT / f"{nm}_color.npy", color)
    np.save(OUT / f"{nm}_pc_tmp.npy", Pc); os.replace(OUT / f"{nm}_pc_tmp.npy", OUT / f"{nm}_pc.npy"); tot += len(Pc)
    hb(nm, "games", len(its), "pos", len(Pc), "gpu_s", round(time.time() - t1), "total_pos", tot, "elapsed_s", round(time.time() - t0), "pos_per_s", round(tot / (time.time() - t0)))
(OUT / "PREP_DONE").write_text("ok"); hb("prep done", tot); pool.close()
