"""E127 step 1 channel: X_new for the E110 pseudo-query dumps (Public / Private pools separately, structures a (tag '') and c).  GPU (system python3).
For every rep r in --reps: the 400 queries' 40-candidate shortlist (dump `top`), library = the dump's 6 games per candidate (never the query games), query games = qrows (<=4).  Matcher logits are z-scored over the shortlist
(row-z over the 40) and averaged over the two nets (h0 matcher on finh_h0 vectors, h1 matcher on finh_h1 vectors); entries outside the shortlist are 0 (never read by the combination).
--shuffle SEED: null control = every game lookup (library and query) goes through a fixed random permutation of the pool's 7,100 games (labels broken), as E112's shuffle.
Writes .scratch/e127/X/X_{name}_{tag}_{pool}[_shufSEED].npz  (X [8000,600] float32, rows of unlisted reps are 0)."""
import sys, os, json, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; sys.path[:0] = [str(HERE), str(ROOT / "scripts/experiments/e112"), str(ROOT / "scripts/experiments/e123")]
import numpy as np, torch
from e127_data import Pool, OUT
from e127_eval0 import load_model
import e112_stage2 as S
DUMPS = ROOT / ".scratch/e110/dumps"; POOLS = (("pub", "public", 0), ("priv", "private", 7100)); TAGS = ("", "c")


@torch.no_grad()
def net_logits(model, tp, base, row, z, r, M, perm, g, Bq=8):
    """-> logits [400,40] float32 and the shortlist top [400,40] for rep r."""
    m = np.flatnonzero(z["rep"] == r); top = z["top"][m].astype(np.int64); lib = z["lib"][r]; qr = z["qrows"][m]; dev = "cuda"
    gm = (lambda x: base + perm[x - base]) if perm is not None else (lambda x: x)
    libg = torch.from_numpy(gm(base + row[lib])).to(dev).long()                                        # [600,6] pool-test game index
    yv, yo, ym = tp.sample_pos(libg, M, g); yc = tp.EFF[libg][..., None].expand(600, 6, yv.shape[2]); b = dict(yv=yv.reshape(1, 600, -1, 128), yo=yo.reshape(1, 600, -1), ym=ym.reshape(1, 600, -1), yc=yc.reshape(1, 600, -1))
    with torch.autocast("cuda", dtype=torch.float16): enc = model.enc_cands(b)
    qvalid = torch.from_numpy(qr >= 0).to(dev); qg = torch.from_numpy(gm(base + row[np.where(qr >= 0, qr, 0)])).to(dev).long(); nq = qg.shape[1]
    qv, qo, qk = tp.sample_pos(qg, M, g); qk = qk & qvalid[..., None]; qc = tp.EFF[qg][..., None].expand(len(m), nq, qv.shape[2]); Mq = qv.shape[2]; out = []
    topt = torch.from_numpy(top).to(dev)
    for a in range(0, len(m), Bq):
        sl = slice(a, a + Bq); B = len(topt[sl]); e = {k: (v[0][topt[sl]] if torch.is_tensor(v) else v) for k, v in enc.items()}
        sub = dict(qv=qv[sl].reshape(B, -1, 128), qo=qo[sl].reshape(B, -1), qm=qk[sl].reshape(B, -1), qc=qc[sl].reshape(B, -1))
        with torch.autocast("cuda", dtype=torch.float16): out.append(model.score(sub, e).float().cpu())
    return torch.cat(out).numpy(), top, m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--models", required=True, help="path template with {net}"); ap.add_argument("--reps", default="0-14"); ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--shuffle", type=int, default=None); ap.add_argument("--tags", default=",".join(TAGS) if TAGS != ("",) else ""); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    lo, hi = [int(x) for x in a.reps.split("-")]; reps = range(lo, hi + 1); t0 = time.time(); tags = a.tags.split(",") if a.tags != "" else [""]; tags = [t for t in tags]
    if a.tags == ",".join(TAGS): tags = list(TAGS)
    nets = ("finh_h0", "finh_h1"); models = {n: load_model(a.models.format(net=n))[0] for n in nets}; pools = {n: Pool(n, ("test",), np.load(OUT / f"sigma_{n}.npy"), "cuda", with_players=False) for n in nets}; print("loaded", time.time() - t0, flush=True)
    sfx = f"_shuf{a.shuffle}" if a.shuffle is not None else ""
    for s, sp, base in POOLS:
        row = S.cand_rows(s, sp); assert len(row) == 6000; perm = np.random.default_rng([a.shuffle, base]).permutation(7100) if a.shuffle is not None else None
        for tag in tags:
            z = np.load(DUMPS / f"t2{tag}_{s}.npz"); X = np.zeros((len(z["rep"]), 600), np.float32)
            for r in reps:
                acc = np.zeros((400, 40)); 
                for n in nets:
                    g = torch.Generator(device="cuda"); g.manual_seed(a.seed * 100 + r); lg, top, m = net_logits(models[n], pools[n], base, row, z, r, a.M, perm, g); lz = (lg - lg.mean(1, keepdims=True)) / (lg.std(1, keepdims=True) + 1e-9); acc += lz / len(nets)
                Xm = X[m]; np.put_along_axis(Xm, top, acc.astype(np.float32), 1); X[m] = Xm
            np.savez(OUT / "X" / f"X_{a.name}_{tag}_{s}{sfx}.npz", X=X, reps=np.array(list(reps)), M=a.M); print("wrote", f"X_{a.name}_{tag}_{s}{sfx}.npz", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
