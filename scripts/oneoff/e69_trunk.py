"""E69 GPU half (adapted from e62_trunk.py): KataGo b18c384nbt-humanv0 trunk state after 3, 6, 12, 15 blocks (= input of blocks[i]) and, for block 0 of each set only, after 9 blocks (check against E56's at-mid/mean-mid).
Per-GAME vectors (fp16) per block: [at 0:384 | mean 384:768 | std-over-positions-of-mean 768:1152]; 4 blocks concatenated in the order C.BLK_IDS = (3, 6, 12, 15) -> blk{i}_gv.npy (n_games, 4608) fp16.
at = value at the played point (board mean on passes); mean = over the 19x19 board.  Reads .scratch/e69/<set>/blk{i}_{bits,glob,loc,rows,color}.npy (e69_prep.py), writes blk{i}_{gv,gid,cnt}.npy (+ blk0_gv9.npy), deletes the block's bits.  System python3 + torch; submit with gpujob.
Exits when PREP_DONE and nothing is left, or after --idle-exit seconds without a ready block."""
import os, sys, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--idle-exit", type=int, default=1800); ap.add_argument("--ckpt", default=str(ROOT / ".scratch/e69/ckpt.ckpt")); ap.add_argument("--bs", type=int, default=256)
a = ap.parse_args()
K = ROOT / ".scratch" / "e69" / a.set; ks = str(ROOT / ".scratch" / "e52" / "katago_src")
mr = np.load(ROOT / ".scratch" / "e52" / "kata" / "meta_rows.npy"); mrows = {0: torch.tensor(mr[0]), 1: torch.tensor(mr[1])}
BLK_IDS = (3, 6, 12, 15); CHECK = 9; HOOK = BLK_IDS + (CHECK,); NB = len(HOOK)
def load_katago_trunk(ckpt, katago_src, device):
    sys.path.insert(0, str(Path(katago_src) / "python"))
    from katago.train.load_model import load_model
    model, _, _ = load_model(ckpt, use_swa=False, device=device, pos_len=19)
    class StopFwd(Exception):
        def __init__(self, x): self.x = x
    store = {}
    model.act_trunkfinal.register_forward_hook(lambda m, i, o: (_ for _ in ()).throw(StopFwd(o)))
    for i in HOOK: model.blocks[i].register_forward_pre_hook((lambda ii: (lambda m, args, kwargs: store.__setitem__(ii, args[0])))(i), with_kwargs=True)    # input of block i = trunk after i blocks
    def trunk(bx, gx, mx):
        try: model(bx, gx, input_meta=mx)
        except StopFwd as e: return {i: store.pop(i) for i in HOOK}
        raise RuntimeError("trunk hook did not fire")
    return model, trunk
DEV = "cuda"
model, trunk = load_katago_trunk(a.ckpt, ks, DEV)
def pos_vec(mids, loc):
    l = loc.clamp(min=0)[:, None, None].expand(-1, 384, 1); pas = loc < 0; out = []
    for i in HOOK:
        t = mids[i].float(); bm = t.mean((2, 3)); at = t.flatten(2).gather(2, l)[:, :, 0]; at[pas] = bm[pas]; out += [at, bm]
    return torch.cat(out, 1)          # (N, NB*768): per hooked block [at | mean]
def run_block(bi):
    bits = np.load(K / f"blk{bi}_bits.npy", mmap_mode="r"); glob = np.load(K / f"blk{bi}_glob.npy"); loc = np.load(K / f"blk{bi}_loc.npy"); rows = np.load(K / f"blk{bi}_rows.npy")
    color = np.load(K / f"blk{bi}_color.npy")
    N = len(bits); vec = np.zeros((N, NB * 768), np.float16); BS = a.bs
    with torch.no_grad():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            b = np.unpackbits(np.asarray(bits[s:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
            bx = torch.from_numpy(b).to(DEV).float(); gx = torch.from_numpy(glob[s:e]).to(DEV); mx = torch.stack([mrows[int(c)] for c in color[s:e]]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16): mids = trunk(bx, gx, mx)
            vec[s:e] = pos_vec(mids, torch.from_numpy(loc[s:e].astype(np.int64)).to(DEV)).cpu().numpy()
    gid, inv = np.unique(rows[:, 0], return_inverse=True); cnt = np.bincount(inv).astype(np.float32)
    order = np.argsort(inv, kind="stable"); starts = np.concatenate([[0], np.cumsum(cnt.astype(np.int64))[:-1]])
    outs = []
    for j, i in enumerate(HOOK):
        v32 = vec[:, j * 768:(j + 1) * 768].astype(np.float32)
        sm = np.add.reduceat(v32[order], starts, axis=0); gm = sm / cnt[:, None]                        # (n_games, 768) = [at-mean | mean-mean]
        sq = np.add.reduceat(v32[order][:, 384:768] ** 2, starts, axis=0); sd = np.sqrt(np.maximum(sq / cnt[:, None] - gm[:, 384:768] ** 2, 0))
        outs.append(np.concatenate([gm, sd], 1).astype(np.float16))                                   # (n_games, 1152)
    np.save(K / f"blk{bi}_gv.npy", np.concatenate(outs[:len(BLK_IDS)], 1)); np.save(K / f"blk{bi}_gid.npy", gid); np.save(K / f"blk{bi}_cnt.npy", cnt.astype(np.int16))
    if bi == 0: np.save(K / "blk0_gv9.npy", outs[len(BLK_IDS)])
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
