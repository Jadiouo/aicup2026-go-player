"""E80 GPU half 1 (system python3 + torch; submit with gpujob): per-position u=[at|mean] (act_trunkfinal, e70_gpu.trunk_u) -> standardise with E77 pca_full.npz -> first 64 PCA components (= E79 e79_gpu.py maths) -> fp16
for the blocks written by e80_prep.py (.scratch/e80/blocks/t*_*.npy); bits/glob/loc deleted after use.  Then assembles Player test2 tokens + the 200 same-model check games:
  .scratch/e80/test_tok.npz  : X (14200,24,64) fp16, mask, oidx bucket, gcol in the order rid-10M (pub: 0..7099 rows of Z_pub uids, priv: 100000..107099)  -> arrays tok_pub / tok_priv  (7100,24,64)
  .scratch/e80/chk_tok.npz   : same for the 200 bundle games + their bundle rows; compared with .scratch/e79/tok_full.npy (same-model check, PLAN 1 / 3a: corr median >= .9999, min >= .999, mean abs diff <= .001).
usage: e80_extract.py [--ckpt F] [--bs 512]"""
import sys, os, time, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(ROOT / "src"), str(ROOT)]
import numpy as np, torch
import e70_gpu as G
K = ROOT / ".scratch" / "e80"; B = K / "blocks"; dev = G.dev; log = G.log
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=str(K / "ckpt.ckpt")); ap.add_argument("--bs", type=int, default=512); a = ap.parse_args()
model = G.load(a.ckpt); z = np.load(ROOT / ".scratch/e77/pca_full.npz"); MU, SD, COMPS = z["mu"], z["sd"], z["comps"][:64]; assert COMPS.shape == (64, 768)
@torch.no_grad()
def u_block(bits, glob, loc, color):
    N = len(bits); out = np.zeros((N, 768), np.float16); loc = loc.astype(np.int64)
    for s in range(0, N, a.bs):
        e = min(s + a.bs, N); b = np.unpackbits(np.asarray(bits[s:e]), axis=1, count=22 * 361).reshape(-1, 22, 19, 19)
        bx = torch.from_numpy(b).to(dev).float(); gx = torch.from_numpy(glob[s:e]).to(dev); mx = torch.stack([G.MROWS[int(c)] for c in color[s:e]]).to(dev); l = torch.from_numpy(loc[s:e]).to(dev)
        out[s:e] = G.trunk_u(model, bx, gx, mx, l).cpu().numpy().astype(np.float16)
    return out
names = sorted(p.name[:-len("_bits.npy")] for p in B.glob("t*_bits.npy")); t0 = time.time(); log("blocks", names)
for nm in names:
    if (B / f"{nm}_pc.npy").exists(): continue
    bits = np.load(B / f"{nm}_bits.npy", mmap_mode="r"); U = u_block(bits, np.load(B / f"{nm}_glob.npy"), np.load(B / f"{nm}_loc.npy"), np.load(B / f"{nm}_color.npy"))
    P = (((U.astype(np.float32) - MU) / SD) @ COMPS.T).astype(np.float16); assert np.isfinite(P).all() and len(P) == len(bits); np.save(B / f"{nm}_pc.npy", P); del bits
    for s in ("bits", "glob", "loc"): os.remove(B / f"{nm}_{s}.npy")
    log(nm, "positions", len(P), f"{time.time() - t0:.0f}s maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
# assemble
names = sorted((p.name[:-len("_pc.npy")] for p in B.glob("t*_pc.npy")), key=lambda s: int(s[1:]))
pcs = np.concatenate([np.load(B / f"{n}_pc.npy") for n in names]); rows = np.concatenate([np.load(B / f"{n}_rows.npy") for n in names]); oi = np.concatenate([np.load(B / f"{n}_oidx.npy") for n in names]); col = np.concatenate([np.load(B / f"{n}_color.npy") for n in names])
rid = rows[:, 0]; ug, inv = np.unique(rid, return_inverse=True); G_ = len(ug); X = np.zeros((G_, 24, 64), np.float16); mk = np.zeros((G_, 24), bool); od = np.zeros((G_, 24), np.int8); gc = np.zeros(G_, np.int8)
j = rows[:, 1]; assert j.max() < 24 and not mk[inv, j].any(); X[inv, j] = pcs; mk[inv, j] = True; od[inv, j] = np.minimum(oi // 5, 23).astype(np.int8); gc[inv] = col
log("games with tokens", G_, "positions", len(pcs), "dist", np.bincount(mk.sum(1)).tolist())
out = {}
for si, s in enumerate(("pub", "priv")):
    zu = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False)["uids"]; want = 10_000_000 + si * 100_000 + np.arange(len(zu)); pos = np.searchsorted(ug, want); assert (ug[np.minimum(pos, G_ - 1)] == want).all(), f"test coverage {s}"
    assert mk[pos].any(1).all(); np.savez(K / f"tok_{s}.npz", X=X[pos], mask=mk[pos], oidx=od[pos], gcol=gc[pos]); log("test", s, len(zu), "coverage 100%")
cm = (ug >= 20_000_000); brow = ug[cm] - 20_000_000; np.savez(K / "chk_tok.npz", rows=brow, X=X[cm], mask=mk[cm], oidx=od[cm], gcol=gc[cm])
T = np.load(ROOT / ".scratch/e79/tok_full.npy", mmap_mode="r"); Tm = np.load(ROOT / ".scratch/e79/tok_full_mask.npy", mmap_mode="r"); Tg = np.load(ROOT / ".scratch/e79/tok_full_gcol.npy"); To = np.load(ROOT / ".scratch/e79/tok_full_oidx.npy", mmap_mode="r")
ref = np.stack([np.asarray(T[r]) for r in brow]).astype(np.float32); rm = np.stack([np.asarray(Tm[r]) for r in brow]); ro = np.stack([np.asarray(To[r]) for r in brow]); mine = X[cm].astype(np.float32); mm = mk[cm]
same_mask = bool((rm == mm).all()); same_oidx = bool((ro[mm] == od[cm][mm]).all()); same_col = bool((Tg[brow] == gc[cm]).all()); a_, b_ = mine[mm & rm], ref[mm & rm]; d = np.abs(a_ - b_)
cor = np.array([np.corrcoef(a_[:, k], b_[:, k])[0, 1] for k in range(64)])
r = {"n_games": int(len(brow)), "n_pos": int(len(a_)), "same_mask": same_mask, "same_oidx": same_oidx, "same_color": same_col, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "pc_std": float(b_.std()), "median_dim_corr": float(np.median(cor)), "min_dim_corr": float(cor.min())}
r["pass"] = bool(same_mask and same_oidx and same_col and r["median_dim_corr"] >= .9999 and r["min_dim_corr"] >= .999 and r["mean_abs_diff"] <= .001); (K / "chk_result.json").write_text(json.dumps(r, indent=1)); log("CHECK", r)
for n in names:
    for s in ("pc", "rows", "oidx", "win", "color"): os.remove(B / f"{n}_{s}.npy")
if not r["pass"]: sys.exit("same-model check FAILED")
log("done")
