"""E127 training (GPU).  usage: e127_train.py --net finh_h0 --kind kde|cross --out DIR [--steps N] [--init CKPT] ...
Listwise episodes (e127_data.episode): K=40 candidates (positive + S-cosine hard negatives), N in {1,2,3,5} query games, LS in {6,10} library games, M random positions per game.
Held-out monitor = the 300 val players of the same net (never trained on); the monitor is only a sanity/overfit read-out (final checkpoint is used, no selection on it)."""
import sys, os, json, time, math, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE)]
import numpy as np, torch, torch.nn.functional as F
from e127_data import Pool, episode, fit_sigma, OUT
from e127_model import Matcher


def score_from_logits(lg, label):
    r = 1 + (lg > lg[torch.arange(len(lg)), label][:, None]).sum(1); return torch.where(r <= 5, torch.exp(-(r - 1.0)), torch.zeros_like(r, dtype=torch.float)), r


@torch.no_grad()
def monitor(model, vp, seed=1234, nb=3, B=50, M=24, mq=None):
    model.eval(); g = torch.Generator(device="cuda"); g.manual_seed(seed); out = {}
    for N in (1, 3, 5):
        for LS in (10,):
            ce = []; sc = []; sb = []; top1 = []
            for _ in range(nb):
                ep = episode(vp, B, N, LS, M, K=40, gen=g); lg = model({k: ep[k] for k in ("qv", "qo", "qm", "qc", "yv", "yo", "ym", "yc")}); ce.append(F.cross_entropy(lg, ep["label"]).item())
                s_, r = score_from_logits(lg, ep["label"]); sc.append(s_.mean().item()); top1.append((r == 1).float().mean().item())
                qv_ = F.normalize(vp.GM[ep["qg"]].mean(1), dim=-1); cen = F.normalize(vp.GM[ep["cg"]].mean(2), dim=-1); sb.append(score_from_logits(torch.einsum("bd,bkd->bk", qv_, cen), ep["label"])[0].mean().item())
            out[f"N{N}"] = dict(ce=float(np.mean(ce)), score=float(np.mean(sc)), top1=float(np.mean(top1)), scos_score=float(np.mean(sb)))
    model.train(); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--net", required=True); ap.add_argument("--kind", default="kde"); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--B", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--lr_p", type=float, default=None); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=""); ap.add_argument("--cand_sa", type=int, default=1); ap.add_argument("--qry_sa", type=int, default=0); ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--d", type=int, default=128); ap.add_argument("--knorm", type=int, default=1); ap.add_argument("--hid", type=int, default=128); ap.add_argument("--resid", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=500); ap.add_argument("--mon_every", type=int, default=250); ap.add_argument("--M", default="12,24"); ap.add_argument("--warm", type=int, default=100); ap.add_argument("--max_hours", type=float, default=5.5)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = "cuda"; torch.manual_seed(a.seed); np.random.seed(a.seed); t_start = time.time()
    sig_f = OUT / f"sigma_{a.net}.npy"
    if not sig_f.exists(): np.save(sig_f, fit_sigma(a.net))
    sigma = np.load(sig_f); tp = Pool(a.net, ("trA", "trB"), sigma, dev); vp = Pool(a.net, ("val",), sigma, dev); print(f"train games {tp.n} players {len(tp.players)}; val games {vp.n} players {len(vp.players)}", flush=True)
    assert not (set(tp.players.tolist()) & set(vp.players.tolist()))
    model = Matcher(a.kind, d=a.d, cand_sa=a.cand_sa, qry_sa=a.qry_sa, drop=a.drop, knorm=bool(a.knorm), hid=a.hid, resid=bool(a.resid)).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location="cpu")["model"]; miss = model.load_state_dict(sd, strict=False); print("init from", a.init, "missing", len(miss.missing_keys), "unexpected", len(miss.unexpected_keys))
    nprm = sum(p.numel() for p in model.parameters()); print("params", nprm, flush=True)
    pp = [p for n, p in model.named_parameters() if n in ("P.weight", "logbeta", "scale")]; rest = [p for n, p in model.named_parameters() if n not in ("P.weight", "logbeta", "scale")]
    groups = [dict(params=pp, lr=a.lr_p or a.lr, weight_decay=0.0)] + ([dict(params=rest, lr=a.lr, weight_decay=a.wd)] if rest else []); opt = torch.optim.AdamW(groups); base = [g["lr"] for g in opt.param_groups]
    step0 = 0; hist = []
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]; hist = ck["hist"]; print("resumed at", step0, flush=True)
    Ms = [int(x) for x in a.M.split(",")]; g = torch.Generator(device=dev); g.manual_seed(a.seed * 1000 + step0); rs = np.random.default_rng(a.seed * 1000 + step0); scaler = torch.amp.GradScaler(); t0 = time.time()
    if step0 == 0:
        m0 = monitor(model, vp); hist.append(dict(step=0, **m0)); print("step 0 monitor", json.dumps(m0), flush=True)
    run_ce = []
    for step in range(step0 + 1, a.steps + 1):
        lrf = min(1.0, step / a.warm) * (0.5 * (1 + math.cos(math.pi * (step - 1) / a.steps)) * 0.95 + 0.05)
        for gr, b0 in zip(opt.param_groups, base): gr["lr"] = b0 * lrf
        N = int(rs.choice([1, 2, 3, 5])); LS = int(rs.choice([6, 10])); M = int(rs.choice(Ms))
        ep = episode(tp, a.B, N, LS, M, K=40, gen=g)
        with torch.autocast("cuda", dtype=torch.float16): lg = model({k: ep[k] for k in ("qv", "qo", "qm", "qc", "yv", "yo", "ym", "yc", "scos")})
        loss = F.cross_entropy(lg.float(), ep["label"]); opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); run_ce.append(loss.item())
        if step % 50 == 0: print(f"step {step} ce {np.mean(run_ce[-50:]):.4f} beta {model.logbeta.exp().item():.2f} scale {model.scale.item():.3f} {(time.time() - t0) / (step - step0):.2f}s/step mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        stop = (time.time() - t_start) > a.max_hours * 3600
        if step % a.mon_every == 0 or step == a.steps or stop:
            m1 = monitor(model, vp); hist.append(dict(step=step, train_ce=float(np.mean(run_ce[-250:])), **m1)); print(f"step {step} monitor", json.dumps(m1), flush=True)
        if step % a.ckpt_every == 0 or step == a.steps or stop:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, hist=hist, args=vars(a)), out / "last.pt.tmp"); os.replace(out / "last.pt.tmp", out / "last.pt")
            torch.save(dict(model=model.state_dict(), step=step, args=vars(a)), out / f"ckpt_{step:06d}.pt"); (out / "hist.json").write_text(json.dumps(hist, indent=1))
        if stop: print("max_hours reached, stopped at", step, flush=True); break
    print("TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
