"""E77 GPU half 2 (system python3 + torch; submit with gpujob): position-level encoder (attention pooling `attn` / 2-layer Transformer `tf`) + episodic InfoNCE.  PLAN section 2.
Per game: tokens = PCA-96 vectors (standardised) + own-move-index embedding (idx//5, 24 buckets) + colour embedding -> game vector (128, L2); bag profile = L2(mean of game vectors); loss = bidirectional InfoNCE (scale 20) between bag A (N in {1,2,3,5} games) and bag B (<=10 other games) of P=192 players.
Subcommands:
  grid  : the 12 pre-registered cells (arch x p_in x steps), seed 0, validation on the 200 validation players (E70 val_score); picks the best (ties -> fewer steps), trains nothing more; embeds the pilot games with it (emb_pilot_<tag>.npz); also seed-1 retrain of the selected cell (info).
Data: .scratch/e77/{tr,pilot}_{pc,idx}.npy/.npz (e77_gpu.py, e77_index.py)."""
import sys, os, time, math, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; K = ROOT / ".scratch" / "e77"; dev = os.environ.get("E77_DEV", "cuda")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class GameData:
    """game-level tensors: X (G,L,96) fp16, mask (G,L), oidx (G,L), col (G,)"""
    def __init__(self, pre, L=12):
        z = np.load(K / f"{pre}_idx.npz"); pc = np.load(K / f"{pre}_pc.npy", mmap_mode="r"); self.player, self.split, self.cnt, self.gid = z["player"], z["split"], z["cnt"], z["gid"]
        pos = z["pos"][:, :L]; m = pos >= 0; X = np.zeros((len(pos), L, pc.shape[1]), np.float16); X[m] = pc[pos[m]]
        self.X = torch.from_numpy(X).to(dev); self.mask = torch.from_numpy(m).to(dev); self.oidx = torch.from_numpy(np.maximum(z["oidx"][:, :L], 0).astype(np.int64)).to(dev); self.col = torch.from_numpy(z["gcol"].astype(np.int64)).to(dev)

class Enc(nn.Module):
    def __init__(self, arch, p_in, din=96, d=128, nb=24):
        super().__init__(); self.arch = arch; self.inp = nn.Sequential(nn.Linear(din, d), nn.LayerNorm(d)); self.pe = nn.Embedding(nb, d); self.ce = nn.Embedding(2, d); self.drop = nn.Dropout(p_in)
        if arch == "attn": self.ln = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d))
        else: self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, p_in, activation="gelu", batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02); self.pool = nn.MultiheadAttention(d, 4, batch_first=True); self.lnp = nn.LayerNorm(d); self.out = nn.Linear(d, d)
    def forward(self, x, oidx, col, mask):
        h = self.inp(x) + self.pe(torch.clamp(oidx // 5, max=23)) + self.ce(col)[:, None, :]; h = self.drop(h)
        if self.arch == "attn": h = h + self.mlp(self.ln(h))
        else: h = self.tf(h, src_key_padding_mask=~mask)
        h = self.lnp(h); o, _ = self.pool(self.q.expand(len(h), -1, -1), h, h, key_padding_mask=~mask, need_weights=False)
        return F.normalize(self.out(o[:, 0]), dim=-1)

def encode(model, D, games, mu, sd, drop_pos=0.0, bs=4096):
    outs = []
    for s in range(0, len(games), bs):
        g = games[s:s + bs]; x = (D.X[g].float() - mu) / sd; m = D.mask[g]
        if drop_pos > 0:
            r = torch.rand(m.shape, device=dev); r = torch.where(m, r, torch.full_like(r, -1.0)); keep = m & (r > drop_pos); keep = keep | (r == r.max(1, keepdim=True).values) & m; m = keep
        outs.append(model(x, D.oidx[g], D.col[g], m))
    return torch.cat(outs)

def val_score(E, player, seed=0, reps=5):
    """E (n,d) game embeddings of the validation games, player (n,) codes; 10 library games per player, N queries from the rest; cosine of bag means; score exp(-(rank-1)) top-5 (e70_gpu.val_score)."""
    ups = np.unique(player); byp = [np.flatnonzero(player == p) for p in ups]; rng = np.random.default_rng(seed); sc = {1: [], 2: [], 3: [], 5: []}
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    for _ in range(reps):
        libs, rests = [], []
        for ix in byp: r = rng.permutation(ix); libs.append(En[r[:10]].mean(0)); rests.append(r[10:])
        L = np.stack(libs); L /= np.linalg.norm(L, axis=1, keepdims=True)
        for n in (1, 2, 3, 5):
            q = np.stack([En[r[:n]].mean(0) for r in rests]); q /= np.linalg.norm(q, axis=1, keepdims=True); S = q @ L.T
            t = S[np.arange(len(S)), np.arange(len(S))]; rk = 1 + (S > t[:, None]).sum(1); sc[n].append(np.where(rk <= 5, np.exp(-(rk - 1.0)), 0.0).mean())
    return {str(n): float(np.mean(v)) for n, v in sc.items()} | {"all": float(np.mean([np.mean(v) for v in sc.values()]))}

def train(D, arch, p_in, steps, seed, mu, sd, games_of, elig, P=192, drop_pos=0.25, lr=1e-3, wd=0.05, scale=20.0, tag=""):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = Enc(arch, p_in, din=D.X.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps); t0 = time.time(); NS = (1, 2, 3, 5)
    for step in range(steps):
        pl = rng.choice(len(elig), P, replace=False); gids = np.zeros((P, 15), np.int64); wa = np.zeros((P, 15), np.float32); wb = np.zeros((P, 15), np.float32)
        for i, p in enumerate(pl):
            gs = rng.permutation(games_of[elig[p]]); n = NS[rng.integers(4)]; nb = min(10, len(gs) - n); gids[i, :n + nb] = gs[:n + nb]; wa[i, :n] = 1; wb[i, n:n + nb] = 1
        g = torch.from_numpy(gids.reshape(-1)).to(dev); model.train(); e = encode(model, D, g, mu, sd, drop_pos).reshape(P, 15, -1)
        A = F.normalize((e * torch.from_numpy(wa).to(dev)[:, :, None]).sum(1), dim=-1); B = F.normalize((e * torch.from_numpy(wb).to(dev)[:, :, None]).sum(1), dim=-1)
        lg = scale * A @ B.T; lab = torch.arange(P, device=dev); loss = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if step % 500 == 0 or step == steps - 1: log(f"{tag} step {step} loss {loss.item():.3f} acc {(lg.argmax(1) == lab).float().mean().item():.3f} {time.time() - t0:.0f}s maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    model.eval(); return model

def setup():
    D = GameData("tr"); tr = np.flatnonzero((D.split == 0) & (D.cnt == 12)); gof = {}
    for g in tr: gof.setdefault(int(D.player[g]), []).append(g)
    elig = [p for p, v in gof.items() if len(v) >= 12]; games_of = {p: np.array(v) for p, v in gof.items()}
    Xt = D.X[torch.from_numpy(tr).to(dev)].float().reshape(-1, D.X.shape[2]); mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; vg = np.flatnonzero((D.split == 1) & (D.cnt == 12))
    log("train players", len(elig), "train games", len(tr), "val games", len(vg), "val players", len(np.unique(D.player[vg]))); return D, games_of, elig, mu, sd, vg

@torch.no_grad()
def embed_np(model, D, games, mu, sd): return encode(model, D, torch.from_numpy(np.asarray(games)).to(dev), mu, sd, 0.0).cpu().numpy()

def cmd_grid(args):
    D, games_of, elig, mu, sd, vg = setup(); hist = json.load(open(K / "grid.json")) if (K / "grid.json").exists() else []; done = {(h["arch"], h["p_in"], h["steps"]) for h in hist}
    # reference: frozen direct average of the standardised PCA tokens (no learning) on the validation players
    Xm = ((D.X[torch.from_numpy(vg).to(dev)].float() - mu) / sd); mk = D.mask[torch.from_numpy(vg).to(dev)][:, :, None].float(); ref = ((Xm * mk).sum(1) / mk.sum(1)).cpu().numpy()
    log("VAL reference direct-average cosine (no learning):", val_score(ref, D.player[vg]))
    for arch in ("attn", "tf"):
        for p_in in (0.1, 0.3):
            for steps in (1500, 3000, 6000):
                if (arch, p_in, steps) in done: continue
                tag = f"{arch}_p{p_in}_s{steps}"; model = train(D, arch, p_in, steps, 0, mu, sd, games_of, elig, tag=tag); r = val_score(embed_np(model, D, vg, mu, sd), D.player[vg])
                hist.append(dict(arch=arch, p_in=p_in, steps=steps, **r)); json.dump(hist, open(K / "grid.json", "w"), indent=1); log("VAL", tag, r); torch.save(model.state_dict(), K / f"enc_{tag}.pt")
    best = sorted(hist, key=lambda h: (-round(h["all"], 12), h["steps"]))[0]; log("SELECTED", best); json.dump(best, open(K / "selected.json", "w"))
    Dp = GameData("pilot"); allp = np.arange(len(Dp.gid)); tag = f"{best['arch']}_p{best['p_in']}_s{best['steps']}"
    for seed in (0, 1):
        if seed == 0: model = Enc(best["arch"], best["p_in"]).to(dev); model.load_state_dict(torch.load(K / f"enc_{tag}.pt")); model.eval()
        else:
            model = train(D, best["arch"], best["p_in"], best["steps"], 1, mu, sd, games_of, elig, tag=tag + "_seed1"); r = val_score(embed_np(model, D, vg, mu, sd), D.player[vg]); log("VAL seed1", r); json.dump(r, open(K / "val_seed1.json", "w"))
        np.savez(K / f"emb_pilot_seed{seed}.npz", gid=Dp.gid, E=embed_np(model, Dp, allp, mu, sd).astype(np.float32)); log("pilot embeddings saved seed", seed)
    # controls need the direct-average vectors on the same standardisation: save them too (game means of standardised PCA tokens) for pilot and train games
    for nm, DD in (("pilot", Dp), ("tr", D)):
        Xm = ((DD.X.float() - mu) / sd); mk = DD.mask[:, :, None].float(); np.save(K / f"avg_{nm}.npy", ((Xm * mk).sum(1) / mk.sum(1).clamp(min=1)).cpu().numpy().astype(np.float32))
    log("direct-average game vectors saved")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); a = ap.parse_args(); log("args", vars(a)); {"grid": cmd_grid}[a.cmd](a); log("done")
