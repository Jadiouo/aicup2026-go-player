"""E40 2a: episodic metric-learning MLP on the L0 Z vector (system python3 + torch; submit with gpujob).
Trains on sim TRAIN players only (eval players never seen), optionally on one half of them (cross-fit).  Saves L2-normalised per-game embeddings for ALL sim rows.
python3 e40_deep_mlp.py --out NAME [--half H|-1] [--epochs 40] [--dim 128] [--hidden 1024]"""
import argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--half", type=int, default=-1); ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--dim", type=int, default=128); ap.add_argument("--hidden", type=int, default=1024); ap.add_argument("--P", type=int, default=192)
ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--data", default="sim"); ap.add_argument("--hsl", type=int, default=0); ap.add_argument("--in-drop", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--scale", type=float, default=20.0)
a = ap.parse_args()
import os
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"   # repo root
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
Z = np.load(R + ".scratch/p1111/sim/Z.npy")                      # rows to EMBED (sim rows)
hf = {0: None, 1: ".scratch/e40/hsl_sim.npy", 2: ".scratch/e40/hsl120_sim.npy"}[a.hsl]
if hf: Z = np.hstack([Z, np.load(R + hf)])
dev = "cuda"
if a.data == "sim":
    mt = np.load(R + ".scratch/e40/sim_meta.npz"); pid, grp = mt["pid"], mt["grp"]; Ztr = Z
    tr_players = np.unique(pid[grp == 0])
    if a.half >= 0:
        perm = np.random.default_rng(99).permutation(len(tr_players)); h = perm[: len(perm) // 2] if a.half == 0 else perm[len(perm) // 2:]
        tr_players = np.sort(tr_players[h])
    MING = 28
elif a.data == "fin":
    bz = np.load(R + ".scratch/e40/fin/Z.npy")
    Ztr = np.hstack([bz] + ([np.load(R + f".scratch/e40/fin/{ {1: 'hsl60', 2: 'hsl120'}[a.hsl] }.npy")] if hf else [])) if hf else bz; del bz
    hv = np.load(R + ".scratch/e40/fin/halves.npz"); pid, flag = hv["pid"], hv["flag"]
    keep = np.ones(len(pid), bool) if a.half < 0 else (flag == a.half) | (flag == 2)
    tr_players = np.unique(pid[keep]); MING = 16
else:
    bz = np.load(R + ".scratch/e40/big/Z.npy"); Ztr = np.hstack([bz] + ([np.load(R + f".scratch/e40/big/{ {1: 'hsl60', 2: 'hsl120'}[a.hsl] }.npy")] if hf else [])); del bz
    hv = np.load(R + ".scratch/e40/big/halves.npz"); pid, flag = hv["pid"], hv["flag"]
    keep = np.ones(len(pid), bool) if a.half < 0 else (flag == a.half) | (flag == 2)
    tr_players = np.unique(pid[keep]); MING = 16
Zg = torch.from_numpy(Z).to(dev).half() if a.data != "fin" else None; Ztr_g = Zg if a.data == "sim" else torch.from_numpy(Ztr).to(dev).half()
if a.data == "fin": d_in_override = Ztr.shape[1]
rows = {p: np.flatnonzero(pid == p) for p in tr_players}
tr_players = [p for p in tr_players if len(rows[p]) >= MING]
print("players", len(tr_players), "rows", sum(len(rows[p]) for p in tr_players), flush=True)
d_in = Z.shape[1] if a.data != "fin" else d_in_override
class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Dropout(a.in_drop), nn.Linear(d_in, a.hidden), nn.LayerNorm(a.hidden), nn.GELU(), nn.Dropout(a.drop),
                              nn.Linear(a.hidden, a.hidden // 2), nn.LayerNorm(a.hidden // 2), nn.GELU(), nn.Dropout(a.drop), nn.Linear(a.hidden // 2, a.dim))
    def forward(s, x): return s.net(x)
enc = Enc().to(dev)
opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
def bag(idx_list):
    """idx_list: list of arrays of row ids -> (B,d) normalised bag means, one embedding pass."""
    flat = np.concatenate(idx_list); seg = np.repeat(np.arange(len(idx_list)), [len(i) for i in idx_list])
    e = Fn.normalize(enc(Ztr_g[torch.from_numpy(flat).to(dev)].float()), dim=-1)
    s = torch.zeros(len(idx_list), e.shape[1], device=dev).index_add_(0, torch.from_numpy(seg).to(dev), e)
    return Fn.normalize(s / torch.tensor([len(i) for i in idx_list], device=dev)[:, None], dim=-1)
t0 = time.perf_counter(); enc.train()
for step in range(a.steps):
    ps = rng.choice(len(tr_players), a.P, replace=False); A, B = [], []
    for j in ps:
        r = rows[tr_players[j]]; r = r[rng.permutation(len(r))]
        na = int(rng.choice([1, 2, 3, 5])); nb = min(10, len(r) - na)
        A.append(r[:na]); B.append(r[na:na + nb])
    qa, qb = bag(A), bag(B)
    logits = a.scale * qa @ qb.T; lab = torch.arange(a.P, device=dev)
    loss = (Fn.cross_entropy(logits, lab) + Fn.cross_entropy(logits.T, lab)) / 2
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0: print(step, round(loss.item(), 4), f"{time.perf_counter() - t0:.0f}s", flush=True)
torch.save(enc.state_dict(), R + f".scratch/e40/enc_{a.out}.pt")
if a.data == "fin":
    enc.eval()
    def emb(X):
        o = []
        with torch.no_grad():
            for i in range(0, len(X), 20000): o.append(Fn.normalize(enc(X[i:i + 20000].float()), dim=-1).cpu().numpy())
        return np.concatenate(o).astype(np.float16)
    ci = torch.from_numpy(np.load(R + ".scratch/e40/fin/cap_idx.npy")).to(dev)
    np.save(R + f".scratch/e40/emb_{a.out}_cap.npy", emb(Ztr_g[ci]))
    for sp in ("pub", "priv"): np.save(R + f".scratch/e40/emb_{a.out}_test_{sp}.npy", emb(torch.from_numpy(np.load(R + f".scratch/e40/fin/test_{sp}.npy")).to(dev)))
    print("saved enc+emb", a.out); raise SystemExit
enc.eval(); out = []
with torch.no_grad():
    for s in range(0, len(Z), 20000): out.append(Fn.normalize(enc(Zg[s:s + 20000].float()), dim=-1).cpu().numpy())
np.save(R + f".scratch/e40/emb_{a.out}.npy", np.concatenate(out).astype(np.float32)); print("saved", a.out, f"{time.perf_counter() - t0:.0f}s")
