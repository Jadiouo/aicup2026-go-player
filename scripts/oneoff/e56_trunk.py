"""E56 GPU half: KataGo b18c384nbt-humanv0 per-GAME vectors (fp16, 2304 dims) = mean over the game's positions of
   [at-final 0:384 | mean-final 384:768 | max-final 768:1152 | at-mid 1152:1536 | mean-mid 1536:1920]  +  [std over positions of mean-final 1920:2304]
(final = act_trunkfinal; mid = trunk state after 9 of the 18 blocks; at = value at the played point (board mean on passes); mean / max = over the 19x19 board).  The first 768 dims are E52's kata2 vector.
Reads .scratch/e56/<set>/blk{i}_{bits,glob,loc,rows,color}.npy (e56_prep.py), writes blk{i}_{gv,gid,cnt}.npy, deletes the block's bits.  System python3 + torch; submit with gpujob.
Exits when PREP_DONE and nothing is left, or after --idle-exit seconds without a ready block.  Env/args: --set sim|final, --ckpt (default .scratch/e56/b18c384nbt-humanv0.ckpt)."""
import os, sys, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--idle-exit", type=int, default=1800); ap.add_argument("--ckpt", default=str(ROOT / ".scratch/e56/b18c384nbt-humanv0.ckpt")); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--dir", default=""); a = ap.parse_args()
K = Path(a.dir) if a.dir else ROOT / ".scratch" / "e56" / a.set; ks = str(ROOT / ".scratch" / "e52" / "katago_src")
mr = np.load(ROOT / ".scratch" / "e52" / "kata" / "meta_rows.npy"); mrows = {0: torch.tensor(mr[0]), 1: torch.tensor(mr[1])}
MID = 9
def load_katago_trunk(ckpt, katago_src, device):
    sys.path.insert(0, str(Path(katago_src) / "python"))
    from katago.train.load_model import load_model
    model, _, _ = load_model(ckpt, use_swa=False, device=device, pos_len=19)
    class StopFwd(Exception):
        def __init__(self, x): self.x = x
    store = {}
    model.act_trunkfinal.register_forward_hook(lambda m, i, o: (_ for _ in ()).throw(StopFwd(o)))
    model.blocks[MID].register_forward_pre_hook(lambda m, args, kwargs: store.__setitem__("mid", args[0]), with_kwargs=True)    # input of block MID = trunk after MID blocks
    def trunk(bx, gx, mx):
        try: model(bx, gx, input_meta=mx)
        except StopFwd as e: return e.x, store.pop("mid")
        raise RuntimeError("trunk hook did not fire")
    return model, trunk
DEV = "cuda"
model, trunk = load_katago_trunk(a.ckpt, ks, DEV)
def pos_vec(f, mid, loc):
    N = f.shape[0]; l = loc.clamp(min=0)[:, None, None].expand(-1, 384, 1); pas = loc < 0
    out = []
    for t in (f, mid):
        bm = t.mean((2, 3)); at = t.flatten(2).gather(2, l)[:, :, 0]; at[pas] = bm[pas]; out.append((at, bm, t.amax((2, 3))))
    (atf, bmf, mxf), (atm, bmm, _) = out
    return torch.cat([atf, bmf, mxf, atm, bmm], 1)
def run_block(bi):
    bits = np.load(K / f"blk{bi}_bits.npy", mmap_mode="r"); glob = np.load(K / f"blk{bi}_glob.npy"); loc = np.load(K / f"blk{bi}_loc.npy"); rows = np.load(K / f"blk{bi}_rows.npy")
    color = np.load(K / f"blk{bi}_color.npy")
    N = len(bits); vec = np.zeros((N, 1920), np.float16); BS = a.bs
    with torch.no_grad():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            b = np.unpackbits(np.asarray(bits[s:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
            bx = torch.from_numpy(b).to(DEV).float(); gx = torch.from_numpy(glob[s:e]).to(DEV); mx = torch.stack([mrows[int(c)] for c in color[s:e]]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16): f, mid = trunk(bx, gx, mx)
            vec[s:e] = pos_vec(f.float(), mid.float(), torch.from_numpy(loc[s:e].astype(np.int64)).to(DEV)).cpu().numpy()
    gid, inv = np.unique(rows[:, 0], return_inverse=True); cnt = np.bincount(inv).astype(np.float32)
    v32 = vec.astype(np.float32); gv = np.zeros((len(gid), 1920), np.float32); np.add.at(gv, inv, v32); gv /= cnt[:, None]
    sq = np.zeros((len(gid), 384), np.float32); np.add.at(sq, inv, v32[:, 384:768] ** 2); sd = np.sqrt(np.maximum(sq / cnt[:, None] - gv[:, 384:768] ** 2, 0))
    np.save(K / f"blk{bi}_gv.npy", np.concatenate([gv, sd], 1).astype(np.float16)); np.save(K / f"blk{bi}_gid.npy", gid); np.save(K / f"blk{bi}_cnt.npy", cnt.astype(np.int16))
    os.remove(K / f"blk{bi}_bits.npy")
    return N
if __name__ == "__main__":
    t0 = time.perf_counter(); bi = 0; tot = 0; last = time.perf_counter()
    while True:
        if (K / f"blk{bi}_gv.npy").exists(): bi += 1; continue
        if (K / f"blk{bi}_bits.npy").exists() and (K / f"blk{bi}_rows.npy").exists():
            time.sleep(1); tot += run_block(bi); last = time.perf_counter(); print(f"block {bi} done, positions {tot}, {time.perf_counter() - t0:.0f}s", flush=True); bi += 1; continue
        if (K / "PREP_DONE").exists() and not (K / f"blk{bi}_bits.npy").exists(): break
        if time.perf_counter() - last > a.idle_exit: print("idle exit", flush=True); break
        time.sleep(10)
    print("all done", f"{time.perf_counter() - t0:.0f}s")
