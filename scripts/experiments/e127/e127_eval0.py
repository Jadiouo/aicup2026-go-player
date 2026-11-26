"""E127 step 0 (held-out unseen players; deviation from the E109 309-player set, see RESULTS): per net 300 players x 16 games never used in training (and unseen by the net),
protocol = E109/E112-stage1 scoring: rng(0) per rep -> 10 library games + 5 rest games per player, N=1/2/3/5, 5 reps, per-player score = mean over reps then equal-weight mean over N,
fusion z_row(BASE) + w*z_row(X), 2-fold player-level CV of w, paired player bootstrap (2000).  BASE = cosine of bag means of S game vectors (the net's own pooled vector; E109 had S+pe86).
Channels: KDE_E112 (E112 fixed kernel exactly: P12, std-whitened over the val positions, beta 20, LR vs mean over candidates), kde_init (our model at init = the same kernel at M random positions, training-set sigma), and trained models.
usage: e127_eval0.py --models name=path{net}/last.pt,... [--M 24] [--tag NAME]   (players of the two nets are pooled for CV / bootstrap, 600 players)"""
import sys, os, json, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; sys.path[:0] = [str(HERE), str(ROOT / "scripts/experiments/e112")]
import numpy as np, torch, torch.nn.functional as F
from e127_data import Pool, OUT, prep
from e127_model import Matcher
from e112_lib import select_idx, nz, kde_game_sums, BETA
AMP = os.environ.get("E127_AMP", "1") == "1"
NS = (1, 2, 3, 5); WS = (.1, .25, .5, 1.0); REPS = 5; NETS = ("finh_h0", "finh_h1")


def sc_from(Sm):
    t = Sm[np.arange(len(Sm)), np.arange(len(Sm))]; r = 1 + (Sm > t[:, None]).sum(1); return np.where(r <= 5, np.exp(-(r - 1.0)), 0.0)


def zrow(Sm): return (Sm - Sm.mean(1, keepdims=True)) / (Sm.std(1, keepdims=True) + 1e-9)


def load_model(path, dev="cuda"):
    ck = torch.load(path, map_location="cpu"); a = ck["args"]; m = Matcher(a["kind"], d=a.get("d", 128), cand_sa=a.get("cand_sa", 1), qry_sa=a.get("qry_sa", 0), drop=0.0, knorm=bool(a.get('knorm', 0)), hid=a.get('hid', 128), resid=bool(a.get('resid', 0))); m.load_state_dict(ck["model"]); return m.to(dev).eval(), ck["step"]


@torch.no_grad()
def model_scores(model, vp, lib, rest, M, seed, tops, Bq=8):
    """lib [nP,10] rest [nP,5] game indices of vp; tops n -> [nP,40] shortlist (candidate ids per query) -> dict n -> [nP,40] logits (fp32 numpy).  Candidate-side encodings are computed once for all nP candidates and gathered."""
    g = torch.Generator(device="cuda"); g.manual_seed(seed); nP = len(lib); libt = torch.from_numpy(lib).cuda(); restt = torch.from_numpy(rest).cuda()
    yv, yo, ym = vp.sample_pos(libt, M, g); yc = vp.EFF[libt][..., None].expand(nP, 10, M)
    b = dict(yv=yv.reshape(1, nP, -1, 128), yo=yo.reshape(1, nP, -1), ym=ym.reshape(1, nP, -1), yc=yc.reshape(1, nP, -1))
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): enc = model.enc_cands(b)
    qv, qo, qk = vp.sample_pos(restt, M, g); qc = vp.EFF[restt][..., None].expand(nP, 5, M); out = {}
    for n in NS:
        Q = dict(qv=qv[:, :n].reshape(nP, -1, 128), qo=qo[:, :n].reshape(nP, -1), qm=qk[:, :n].reshape(nP, -1), qc=qc[:, :n].reshape(nP, -1)); top = torch.from_numpy(tops[n]).cuda(); L = []
        for a in range(0, nP, Bq):
            sub = {k: v[a:a + Bq] for k, v in Q.items()}; e = {k: (v[0][top[a:a + Bq]] if torch.is_tensor(v) else v) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): L.append(model.score(sub, e).float().cpu())
        out[n] = torch.cat(L).numpy().astype(np.float64)
    return out


def fuse(zfull, Xs, top, w):
    """shortlist fusion: zfull [nP,nP] base score (row-z), Xs [nP,40] channel on the shortlist -> row-z within the shortlist; fused = base + w*X on the shortlist, -inf elsewhere."""
    Z = zrow(Xs); F_ = np.full(zfull.shape, -np.inf); np.put_along_axis(F_, top, np.take_along_axis(zfull, top, 1) + w * Z, 1); return F_


def run_net(net, specs, M, dev="cuda", short=40):
    sigma = np.load(OUT / f"sigma_{net}.npy"); vp = Pool(net, ("val",), sigma, dev); d = np.load(OUT / f"pos_{net}_val.npz"); nP = len(vp.players)
    vec, off, ords = d["vec"], d["off"], d["ords"]; keep = [off[g] + select_idx(ords[off[g]:off[g + 1]], "P12") for g in range(vp.n)]; cnt = np.array([len(k) for k in keep]); koff = np.concatenate([[0], np.cumsum(cnt)]); kv = vec[np.concatenate(keep)].astype(np.float32)
    PW = nz(kv / kv.astype(np.float64).std(0).astype(np.float32)); GMn = vp.GM.cpu().numpy().astype(np.float64)
    models = {"kde_init": (Matcher("kde", knorm=True).to(dev).eval(), 0)}
    for nm, path in specs.items(): models[nm] = load_model(path.format(net=net), dev)
    rng = np.random.default_rng(0); chans = ["KDE_E112", "KDE_E112s"] + list(models); base = {n: np.zeros(nP) for n in NS}; alone = {(c, n): np.zeros(nP) for c in chans for n in NS}; fus = {(c, w, n): np.zeros(nP) for c in chans for w in WS for n in NS}
    fus2 = {(c, w, n): np.zeros(nP) for c in chans for w in WS for n in NS}; base2 = {n: np.zeros(nP) for n in NS}; t0 = time.time(); PGn = vp.PG.cpu().numpy(); recall = {n: 0.0 for n in NS}
    for rep in range(REPS):
        libs, rests = [], []
        for p in range(nP): r = rng.permutation(PGn[p, :vp.PCNT[p].item()]); libs.append(r[:10]); rests.append(r[10:15])
        libs = np.stack(libs); rests = np.stack(rests)
        lv = [np.concatenate([PW[koff[g]:koff[g + 1]] for g in libs[p]]) for p in range(nP)]; lens = np.array([len(x) for x in lv]); Lv = np.concatenate(lv)
        qg = rests.reshape(-1); Qv = np.concatenate([PW[koff[g]:koff[g + 1]] for g in qg]); Qslot = np.concatenate([np.full(cnt[g], j, np.int64) for j, g in enumerate(qg)])
        sums, qc_ = kde_game_sums(Qv, Qslot, len(qg), Lv, lens, BETA, dev=dev); sums = sums.reshape(nP, 5, nP); qc_ = qc_.reshape(nP, 5)
        Sb = {n: nz(GMn[rests[:, :n]].mean(1)) @ nz(GMn[libs].mean(1)).T for n in NS}; tops = {n: np.argsort(-Sb[n], 1)[:, :short] for n in NS}
        MS = {nm: model_scores(m, vp, libs, rests, M, seed=rep, tops=tops) for nm, (m, _) in models.items()}
        for n in NS:
            top = tops[n]; zb = zrow(Sb[n]); base[n] += sc_from(Sb[n]) / REPS; recall[n] += np.mean((top == np.arange(nP)[:, None]).any(1)) / REPS
            Kf = sums[:, :n].sum(1) / qc_[:, :n].sum(1)[:, None]; zb2 = zb + .5 * zrow(Kf); base2[n] += sc_from(fuse(zb2, np.take_along_axis(Kf, top, 1), top, 0.0)) / REPS     # w=0: ranking of zb2 inside the shortlist
            Sx = {"KDE_E112s": np.take_along_axis(Kf, top, 1)}; Sx.update({nm: MS[nm][n] for nm in models})
            alone[("KDE_E112", n)] += sc_from(Kf) / REPS
            for w in WS: fus[("KDE_E112", w, n)] += sc_from(zb + w * zrow(Kf)) / REPS
            for c, Xs in Sx.items():
                full = np.full((nP, nP), -np.inf); np.put_along_axis(full, top, Xs, 1); alone[(c, n)] += sc_from(full) / REPS
                for w in WS: fus[(c, w, n)] += sc_from(fuse(zb, Xs, top, w)) / REPS; fus2[(c, w, n)] += sc_from(fuse(zb2, Xs, top, w)) / REPS
        print(f"  [{net}] rep {rep} {time.time() - t0:.0f}s", flush=True)
    return dict(base=base, base2=base2, alone=alone, fus=fus, fus2=fus2, chans=chans, steps={nm: s for nm, (_, s) in models.items()}, nP=nP, shortlist_recall={str(n): float(v) for n, v in recall.items()})


def combine(res, key="fus", bkey="base"):
    """pool the two nets' players; per channel: CV-w delta vs BASE (per-player arrays, 2-fold CV over the pooled players)."""
    chans = res[0]["chans"]; nP = sum(r["nP"] for r in res); cat = lambda f: np.concatenate([f(r) for r in res])
    bpp = np.mean([cat(lambda r, n=n: r[bkey][n]) for n in NS], 0); rng = np.random.default_rng(0); half = rng.permutation(nP); h1, h2 = half[: nP // 2], half[nP // 2:]; B = np.random.default_rng(0).integers(0, nP, (2000, nP)); out = {}; per = {}
    for c in chans:
        if (c, WS[0], NS[0]) not in res[0][key]: continue
        pw = {w: np.mean([cat(lambda r, n=n, w=w: r[key][(c, w, n)]) for n in NS], 0) - bpp for w in WS}; w1 = max(WS, key=lambda w: pw[w][h1].mean()); w2 = max(WS, key=lambda w: pw[w][h2].mean())
        d = np.zeros(nP); d[h2] = pw[w1][h2]; d[h1] = pw[w2][h1]; per[c] = d; bs = d[B].mean(1)
        out[c] = dict(delta_cv=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), w_sel=[w1, w2], per_w={str(w): float(pw[w].mean()) for w in WS}, alone=float(np.mean([cat(lambda r, n=n: r["alone"][(c, n)]).mean() for n in NS])))
    return out, per, bpp, B


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default=""); ap.add_argument("--M", type=int, default=24); ap.add_argument("--tag", default="x"); a = ap.parse_args()
    specs = dict(x.split("=", 1) for x in a.models.split(",") if x); res = [run_net(n, specs, a.M) for n in NETS]; R = dict(M=a.M, specs=specs, steps=[r["steps"] for r in res], base=float(np.mean([np.concatenate([r["base"][n] for r in res]).mean() for n in NS])))
    R["shortlist_recall"] = [r["shortlist_recall"] for r in res]; R["base2"] = float(np.mean([np.concatenate([r["base2"][n] for r in res]).mean() for n in NS]))
    for key, bkey in (("fus", "base"), ("fus2", "base2")):
        out, per, bpp, B = combine(res, key, bkey); R[key] = out
        if key == "fus":
            for c in out:
                if c in ("KDE_E112",): continue
                d = per[c] - per["KDE_E112"]; bs = d[B].mean(1); out[c]["minus_KDE_E112"] = dict(point=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
                if c != "kde_init": d = per[c] - per["kde_init"]; bs = d[B].mean(1); out[c]["minus_kde_init"] = dict(point=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    (OUT / f"eval0_{a.tag}.json").write_text(json.dumps(R, indent=1)); print(f"BASE (S cos) {R['base']:.4f}   BASE+.5*KDE_E112 {R['base2']:.4f}")
    for key in ("fus", "fus2"):
        print(f"--- {key}: delta vs {'BASE' if key == 'fus' else 'BASE+.5*KDE_E112'}")
        for c, v in R[key].items(): print(f"  {c:10s} alone {v['alone']:.4f}  delta_cv {v['delta_cv']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] w {v['w_sel']} per_w {v['per_w']}" + (f"  minusKDE_E112 {v['minus_KDE_E112']['point']:+.4f} [{v['minus_KDE_E112']['lo']:+.4f},{v['minus_KDE_E112']['hi']:+.4f}]" if "minus_KDE_E112" in v else ""))


if __name__ == "__main__":
    main()
