"""E86 full stage step 3 (runs ON THE COLAB VM, /content/Gogame; after e86_colab_prep.py): (1) same-model check (200 bundle games: Colab 24-position subset vs E79 tok_full tokens), (2) assemble the dense token array
(union games: E79's 24 tokens [exact-duplicate vectors dropped] + Colab's other positions; the 200 check games and all test2 games: Colab only; <= 118 positions per game), (3) train the six E77-size encoders (e80_enc recipe: attn, p_in .3, 6000 steps, seed 0, P=192, drop_pos .25,
NOT retuned; training-time encode in checkpointed chunks) and embed -> /content/enc_out/{pe_sim_h0,pe_sim_h1,pe_sim_full,pe_final_h0,pe_final_h1,pe_test_pub,pe_test_priv}.npy + enc_*.pt + enc_log.json.
Inputs: /content/out/b*_{pc,rows,oidx,rep,color}.npy, /content/meta86.npz, /content/e79_union.npz, /content/chk_e79.npz, /content/items_full.pkl (only for n_own of the check games).  usage: e86_colab_enc.py [--only sim,final] [--skip-check-fail]"""
import sys, os, json, time, argparse, pickle, glob
from pathlib import Path
ROOT = Path("/content/Gogame"); sys.path[:0] = [str(ROOT / "scripts" / "oneoff"), str(ROOT / "src"), str(ROOT)]
import numpy as np, pandas as pd, torch
import torch.utils.checkpoint as _ck
import e77_enc as M, e80_enc as E80
dev, log = M.dev, M.log; OUT = Path("/content/out"); EO = Path("/content/enc_out"); EO.mkdir(exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--only", default="sim,final"); ap.add_argument("--steps", type=int, default=6000); a = ap.parse_args()
# ---------------- memory: training-time encode in checkpointed chunks (same maths as e77_enc.encode)
_enc0 = M.encode
def _enc_ck(model, D, games, mu, sd, drop_pos=0.0, bs=4096):
    if not (torch.is_grad_enabled() and model.training): return _enc0(model, D, games, mu, sd, drop_pos, bs)
    def f(g):
        x = (D.X[g].float() - mu) / sd; m = D.mask[g]
        if drop_pos > 0:
            r = torch.rand(m.shape, device=M.dev); r = torch.where(m, r, torch.full_like(r, -1.0)); keep = m & (r > drop_pos); m = keep | (r == r.max(1, keepdim=True).values) & m
        return model(x, D.oidx[g], D.col[g], m)
    return torch.cat([_ck.checkpoint(f, games[s:s + 256], use_reentrant=False) for s in range(0, len(games), 256)])
M.encode = _enc_ck
# ---------------- load Colab positions
names = sorted((os.path.basename(p)[:-len("_pc.npy")] for p in glob.glob(str(OUT / "b*_pc.npy"))), key=lambda s: int(s[1:])); log("blocks", len(names))
pc = np.concatenate([np.load(OUT / f"{n}_pc.npy") for n in names]); rows = np.concatenate([np.load(OUT / f"{n}_rows.npy") for n in names]); oi = np.concatenate([np.load(OUT / f"{n}_oidx.npy") for n in names]).astype(np.int64); rp = np.concatenate([np.load(OUT / f"{n}_rep.npy") for n in names]); col = np.concatenate([np.load(OUT / f"{n}_color.npy") for n in names])
mz = np.load("/content/meta86.npz"); union = mz["union"]; NU = len(union); TEST = [10_000_000 + np.arange(7100), 10_100_000 + np.arange(7100)]; allrid = np.concatenate([union] + TEST); N = len(allrid); assert (np.diff(allrid) > 0).all()
gi = np.searchsorted(allrid, rows[:, 0]); assert (allrid[gi] == rows[:, 0]).all(); log("colab positions", len(pc), "games with positions", len(np.unique(gi)), "of", N)
# ---------------- (1) same-model check: 200 check games, Colab positions with rep>0 vs E79 tokens
ck = np.load("/content/chk_e79.npz"); items = {it[3]: it for it in pickle.load(open("/content/items_full.pkl", "rb")) if it[3] in set(ck["rid"].tolist())}
from scripts.experiments.e13 import lib; import e79_prep_local as P79; from gogame.sgf import parse_sgf
A_, B_ = [], []; chk_gi = set(np.searchsorted(allrid, ck["rid"]).tolist())
for k, rid in enumerate(ck["rid"]):
    _, color, sgf, _r = items[int(rid)]; mv = parse_sgf(sgf).moves; n_own = int((mv[:, 0] == color).sum()); i24 = np.concatenate([lib.own_move_idx_eval(n_own, k=12), P79.late_idx(n_own, 12)]); g = np.searchsorted(allrid, rid); sel_ = np.flatnonzero(gi == g)
    d = {int(oi[s]): s for s in sel_}
    for t, ix in enumerate(i24):
        if ck["mask"][k, t]: A_.append(pc[d[int(ix)]].astype(np.float32)); B_.append(ck["X"][k, t].astype(np.float32))
A_, B_ = np.stack(A_), np.stack(B_); dd = np.abs(A_ - B_); cor = np.array([np.corrcoef(A_[:, j], B_[:, j])[0, 1] for j in range(64)])
chk = {"n_pos": len(A_), "mean_abs_diff": float(dd.mean()), "max_abs_diff": float(dd.max()), "median_dim_corr": float(np.median(cor)), "min_dim_corr": float(cor.min())}; chk["pass"] = bool(chk["median_dim_corr"] >= .9999 and chk["min_dim_corr"] >= .999 and chk["mean_abs_diff"] <= .002)
json.dump(chk, open(EO / "check_colab_vs_e79.json", "w"), indent=1); log("CHECK", chk)
if not chk["pass"]: sys.exit("same-model check FAILED")
# ---------------- (2) dense tokens
L = 118; X = np.zeros((N, L, 64), np.float16); mk = np.zeros((N, L), bool); BK = np.zeros((N, L), np.int64); GC = np.zeros(N, np.int64)
u = np.load("/content/e79_union.npz"); X79, M79, O79, GC79 = u["X"], u["mask"], u["oidx"].astype(np.int64), u["gcol"].astype(np.int64)
start = np.zeros(N, np.int64); h = (np.random.default_rng(0).integers(1, 2**61, 16, dtype=np.int64))
for s in range(0, NU, 20000):                      # E79 tokens of the non-check union games; exact duplicate vectors (short games' overlapping windows) dropped
    xs = np.ascontiguousarray(X79[s:s + 20000]); hs = (xs.view(np.int64).reshape(len(xs), 24, 16) * h).sum(2); ms = M79[s:s + 20000].copy()
    for g in range(len(xs)):
        if (s + g) in chk_gi: ms[g] = False; continue
        seen = set(); 
        for t in np.flatnonzero(ms[g]):
            if hs[g, t] in seen: ms[g, t] = False
            else: seen.add(hs[g, t])
    for g in range(len(xs)):
        t = np.flatnonzero(ms[g]); n = len(t); X[s + g, :n] = xs[g, t]; mk[s + g, :n] = True; BK[s + g, :n] = O79[s + g, t]; start[s + g] = n
GC[:NU] = GC79
slot = start[gi] + rows[:, 1]; assert slot.max() < L; assert not mk[gi, slot].any(); X[gi, slot] = pc; mk[gi, slot] = True; BK[gi, slot] = np.minimum(oi // 5, 23); GC[gi] = col     # colour from the Colab rows (== E79 gcol, asserted below)
assert (GC79[gi[gi < NU]] == col[gi < NU]).all(), 'colour mismatch E79 vs Colab'; assert mk.any(1).all(), "game without tokens"
nb = np.bincount(mk.sum(1)); log("tokens per game: mean", float(mk.sum(1).mean()), "max", int(mk.sum(1).max()), "games", N); del pc
# ---------------- (3) encoders
sel = json.load(open("/content/selected.json")) if os.path.exists("/content/selected.json") else {"arch": "attn", "p_in": 0.3, "steps": 6000}; assert (sel["arch"], sel["p_in"], sel["steps"]) == ("attn", 0.3, 6000)
D = E80.mk(X, mk, BK, GC); del X; log("tokens on", dev, tuple(D.X.shape), f"maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
player = mz["player"]; res = {}
def run(name, cfg):
    f = EO / (f"pe_{name}_{cfg}.npy" if not (name == "final" and cfg == "full") else "pe_test_priv.npy")
    if f.exists(): log("skip", name, cfg); return
    ok = np.flatnonzero(mz[f"{name[:3] if name == 'sim' else 'fin'}_{cfg}"]); gof = {}
    for g in ok: gof.setdefault(int(player[g]), []).append(g)
    elig = [p for p, v in gof.items() if len(v) >= 12]; games_of = {p: np.array(v) for p, v in gof.items()}; mu, sd = E80.stats(D, ok, chunk=8000)
    log(name, cfg, "players", len(elig), "train games", len(ok), f"maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"); t0 = time.time()
    model = M.train(D, "attn", 0.3, a.steps, 0, mu, sd, games_of, elig, tag=f"{name}_{cfg}"); log("trained", f"{time.time() - t0:.0f}s")
    emb = mz["sim_pos"] if name == "sim" else mz["fin_pos"]; E = M.embed_np(model, D, emb, mu, sd).astype(np.float32); assert np.isfinite(E).all() and np.abs(np.linalg.norm(E, axis=1) - 1).max() < 1e-3
    torch.save({"sd": model.state_dict(), "mu": mu.cpu(), "std": sd.cpu()}, EO / f"enc_{name}_{cfg}.pt")
    if name == "sim":
        A = np.zeros((223_693, 128), np.float32); A[mz["sim_gid"]] = E; np.save(EO / f"pe_sim_{cfg}.npy", A)
        if cfg == "full": ev = np.flatnonzero(mz["sim_eval"]); r = M.val_score(E[ev], player[emb[ev]]); res["sim_full_eval_retrieval"] = r; log("INFO eval-player retrieval (not used for any selection; E80 .326, E77 .327):", r)
    else:
        if cfg != "full": A = np.zeros((int(mz["fin_nrows"]), 128), np.float32); A[mz["fin_gid"]] = E; np.save(EO / f"pe_final_{cfg}.npy", A)
        else:
            for si, s in enumerate(("pub", "priv")):
                Et = M.embed_np(model, D, NU + si * 7100 + np.arange(7100), mu, sd).astype(np.float32); assert np.isfinite(Et).all() and np.abs(np.linalg.norm(Et, axis=1) - 1).max() < 1e-3; np.save(EO / f"pe_test_{s}.npy", Et)
    json.dump(res, open(EO / "enc_log.json", "w")); log("saved", name, cfg, E.shape); del model; torch.cuda.empty_cache()
for name in a.only.split(","):
    for cfg in ("h0", "h1", "full"): run(name, cfg)
(EO / "ENC_DONE").write_text("ok"); log("done")
