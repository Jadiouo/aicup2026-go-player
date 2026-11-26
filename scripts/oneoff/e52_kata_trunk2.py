"""E52 step 5 scale-up: KataGo b18c384nbt-humanv0 trunk, per-GAME vectors = mean over the game's positions of [trunk at the played move (384) | trunk board mean (384)] -> fp16 (n_games, 768).
Reads .scratch/e52/kata2/blk{i}_*.npy (from e52_kata_prep2.py) as they appear; writes blk{i}_gv.npy / blk{i}_gid.npy and deletes the block's bits.  System python3 + torch (GPU); submit with gpujob.
Env: E52_KDIR (default kata2)."""
import os, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, torch
K = ROOT / ".scratch" / "e52" / os.environ.get("E52_KDIR", "kata2"); ks = str(ROOT / ".scratch" / "e52" / "katago_src")
mr = np.load(ROOT / ".scratch" / "e52" / "kata" / "meta_rows.npy"); mrows = {0: torch.tensor(mr[0]), 1: torch.tensor(mr[1])}
def load_katago_trunk(ckpt, katago_src, device):          # same as scripts/experiments/e13/train.py (system python lacks pyarrow)
    sys.path.insert(0, str(Path(katago_src) / "python"))
    from katago.train.load_model import load_model
    model, _, _ = load_model(ckpt, use_swa=False, device=device, pos_len=19)
    class StopFwd(Exception):
        def __init__(self, x): self.x = x
    def hook(m, i, o): raise StopFwd(o)
    model.act_trunkfinal.register_forward_hook(hook)
    def trunk(bx, gx, mx):
        try: model(bx, gx, input_meta=mx)
        except StopFwd as e: return e.x
        raise RuntimeError("trunk hook did not fire")
    return model, trunk, model.config
DEV = "cuda"
model, trunk, mc = load_katago_trunk(str(ROOT / ".scratch" / "e52" / "b18c384nbt-humanv0.ckpt"), ks, DEV)
colors = None
def run_block(bi):
    bits = np.load(K / f"blk{bi}_bits.npy", mmap_mode="r"); glob = np.load(K / f"blk{bi}_glob.npy"); loc = np.load(K / f"blk{bi}_loc.npy"); rows = np.load(K / f"blk{bi}_rows.npy")
    color = np.load(K / f"blk{bi}_color.npy")
    N = len(bits); vec = np.zeros((N, 768), np.float16); BS = 256
    with torch.no_grad():
        for a in range(0, N, BS):
            e = min(a + BS, N)
            b = np.unpackbits(np.asarray(bits[a:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
            bx = torch.from_numpy(b).to(DEV).float(); gx = torch.from_numpy(glob[a:e]).to(DEV); mx = torch.stack([mrows[int(c)] for c in color[a:e]]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16): f = trunk(bx, gx, mx).float()
            bm = f.mean((2, 3)); l = torch.from_numpy(loc[a:e].astype(np.int64)).to(DEV); pas = l < 0
            v = f.flatten(2).gather(2, l.clamp(min=0)[:, None, None].expand(-1, 384, 1))[:, :, 0]; v[pas] = bm[pas]
            vec[a:e] = torch.cat([v, bm], 1).cpu().numpy()
    gid, inv = np.unique(rows[:, 0], return_inverse=True); cnt = np.bincount(inv).astype(np.float32)
    gv = np.zeros((len(gid), 768), np.float32); np.add.at(gv, inv, vec.astype(np.float32)); gv /= cnt[:, None]
    np.save(K / f"blk{bi}_gv.npy", gv.astype(np.float16)); np.save(K / f"blk{bi}_gid.npy", gid); np.save(K / f"blk{bi}_cnt.npy", cnt.astype(np.int16))
    os.remove(K / f"blk{bi}_bits.npy")
    return N
if __name__ == "__main__":
    t0 = time.perf_counter(); bi = 0; tot = 0
    while True:
        if (K / f"blk{bi}_gv.npy").exists(): bi += 1; continue
        if (K / f"blk{bi}_bits.npy").exists() and (K / f"blk{bi}_rows.npy").exists() and ((K / f"blk{bi+1}_bits.npy").exists() or (K / "PREP_DONE").exists()):
            time.sleep(2); tot += run_block(bi); print(f"block {bi} done, positions {tot}, {time.perf_counter() - t0:.0f}s", flush=True); bi += 1; continue
        if (K / "PREP_DONE").exists() and not (K / f"blk{bi}_bits.npy").exists(): break
        time.sleep(10)
    print("all done", f"{time.perf_counter() - t0:.0f}s")
