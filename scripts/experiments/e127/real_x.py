"""E127 step 2 (alternate candidate): X_p1r for the REAL test2 queries.  GPU (system python3).  Data-feeding layer only; model forward = e127_x.net_logits (source-patched in one token, see below).
subcommands
  equiv   feed the PSEUDO-query dumps (.scratch/e110/dumps, tags '' and c, pools pub/priv, reps 0-14) through THIS layer and compare with the stored X_p1r_M58_{tag}_{pool}.npz  (writes final/equiv.json)
  real    library = all 10 games of each of the 600 candidates, query = its num_games games (export rows 6000+start+j), writes final/real_Xp1r_{pool}.npz
The only change to e127_x.net_logits: the hard-coded `expand(600, 6, ...)` (6 library games) becomes `expand(600, libg.shape[1], ...)` (via inspect.getsource + str.replace + exec; count asserted == 1).
Row-z over the 40 shortlist + 2-net average + put_along_axis are the lines inline in e127_x.main (not importable): copied literally in `channel_X`."""
import sys, os, json, csv, time, inspect, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; HERE = ROOT / ".scratch/e127/final"
sys.dont_write_bytecode = True; sys.path[:0] = [str(ROOT / "scripts/experiments/e127"), str(ROOT / "scripts/experiments/e112"), str(ROOT / "scripts/experiments/e123")]
import numpy as np, torch
import e127_x as EX
from e127_data import Pool, OUT
from e127_eval0 import load_model
import e112_stage2 as S
REAL = ROOT / ".scratch/e90/final/make"; NETS = ("finh_h0", "finh_h1"); POOLS = (("pub", "public", 0), ("priv", "private", 7100)); M = 58; CKPT = ".scratch/e127/p1r_{net}/last.pt"

_src = inspect.getsource(EX.net_logits); _old = "expand(600, 6, yv.shape[2])"; assert _src.count(_old) == 1
_ns = dict(EX.__dict__); exec(_src.replace(_old, "expand(600, libg.shape[1], yv.shape[2])"), _ns); net_logits_gen = _ns["net_logits"]


def channel_X(models, pools, base, rowx, lib, qrows, top, seed):
    """lib [600,L] and qrows [400,nq] (-1 pad) are indices into rowx (= candidate-table row -> pool position, then identity for query games 6000..7099); top [400,40] shortlist.  -> X [400,600] float32 (0 outside the shortlist)."""
    z = dict(rep=np.zeros(400, np.int64), top=top, lib={0: lib}, qrows=qrows); acc = np.zeros((400, 40)); parts = {}
    for n in NETS:
        g = torch.Generator(device="cuda"); g.manual_seed(seed); lg, top_, m = net_logits_gen(models[n], pools[n], base, rowx, z, 0, M, None, g); assert (top_ == top).all() and len(m) == 400
        lz = (lg - lg.mean(1, keepdims=True)) / (lg.std(1, keepdims=True) + 1e-9); acc += lz / len(NETS); parts[n] = lg
    X = np.zeros((400, 600), np.float32); np.put_along_axis(X, top, acc.astype(np.float32), 1); return X, parts


def load_all():
    models = {n: load_model(CKPT.format(net=n))[0] for n in NETS}; pools = {n: Pool(n, ("test",), np.load(OUT / f"sigma_{n}.npy"), "cuda", with_players=False) for n in NETS}; return models, pools


def rowx_of(s, sp):
    row = S.cand_rows(s, sp); assert len(row) == 6000; return np.concatenate([row, np.arange(6000, 7100)]).astype(np.int64)


def equiv():
    models, pools = load_all(); R = {}; t0 = time.time()
    # patched copy vs the ORIGINAL e127_x.net_logits on identical inputs (pub, tag a, rep 0, both nets): logits must match exactly
    s, sp, base = POOLS[0]; rowx = rowx_of(s, sp); z = np.load(ROOT / ".scratch/e110/dumps/t2_pub.npz"); zz = dict(rep=z["rep"], top=z["top"], lib=z["lib"], qrows=z["qrows"]); orig_vs_patched = {}
    for n in NETS:
        g1 = torch.Generator(device="cuda"); g1.manual_seed(0); a1 = EX.net_logits(models[n], pools[n], base, S.cand_rows(s, sp), zz, 0, M, None, g1)[0]
        g2 = torch.Generator(device="cuda"); g2.manual_seed(0); a2 = net_logits_gen(models[n], pools[n], base, S.cand_rows(s, sp), zz, 0, M, None, g2)[0]; orig_vs_patched[n] = float(np.abs(a1 - a2).max())
    print("original e127_x.net_logits vs source-patched copy, max|logit diff|:", orig_vs_patched, flush=True); R["orig_vs_patched_logits"] = orig_vs_patched
    for s, sp, base in POOLS:
        rowx = rowx_of(s, sp)
        for tag in ("", "c"):
            z = np.load(ROOT / f".scratch/e110/dumps/t2{tag}_{s}.npz"); st = np.load(OUT / "X" / f"X_p1r_M58_{tag}_{s}.npz"); X0 = st["X"]; assert int(st["M"]) == 58 and list(st["reps"]) == list(range(15))
            md = 0.0; per = []; top1 = 0; tot = 0
            for r in range(15):
                m = np.flatnonzero(z["rep"] == r); lib = z["lib"][r]; qr = z["qrows"][m]; top = z["top"][m].astype(np.int64)
                X, _ = channel_X(models, pools, base, rowx, lib, qr, top, r); d = np.abs(X - X0[m]); per.append(float(d.max())); md = max(md, per[-1])
                sh = np.take_along_axis(X, top, 1); sh0 = np.take_along_axis(X0[m], top, 1); top1 += int((sh.argmax(1) == sh0.argmax(1)).sum()); tot += len(m)
                o = np.ones((400, 600), bool); np.put_along_axis(o, top, False, 1); assert (X[o] == 0).all() and (X0[m][o] == 0).all()
            R[f"{s}/{tag or 'a'}"] = dict(max_abs_diff=md, per_rep_max=per, argmax_agree=f"{top1}/{tot}"); print(f"{s}/{tag or 'a'}: max|diff| {md:.3e}  argmax-in-shortlist agreement {top1}/{tot}  {time.time() - t0:.0f}s", flush=True)
    # outside the shortlist both are exactly 0?  (checked on the last cell's last rep above through put_along_axis; also whole-array check on rep 0 below)
    R["overall_max_abs_diff"] = max(v["max_abs_diff"] for k, v in R.items() if "/" in k); (HERE / "equiv.json").write_text(json.dumps(R, indent=1)); print("overall", R["overall_max_abs_diff"])


def real():
    models, pools = load_all(); t0 = time.time()
    for s, sp, base in POOLS:
        rowx = rowx_of(s, sp); row = rowx[:6000]
        c = list(csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", newline=""))); pid = np.array([r["player_id"] for r in c]); pl = sorted(set(pid)); assert len(pl) == 600
        y = np.array([pl.index(p) for p in pid]); assert (np.bincount(y) == 10).all(); lib = np.stack([np.flatnonzero(y == p) for p in range(600)]); assert lib.shape == (600, 10) and len(set(lib.ravel().tolist())) == 6000
        tq = list(csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_test2_{sp}.csv", newline=""))); ng = np.array([int(r["num_games"]) for r in tq]); assert len(tq) == 400 and ng.sum() == 1100 and ng.min() >= 1 and ng.max() <= 5
        z = np.load(ROOT / f".scratch/p1111/z_t2/Z_{s}.npz", allow_pickle=False); uids = z["uids"].astype(str); rz = np.load(REAL / f"real_v7q_S_{s}.npz", allow_pickle=True); qid = rz["qids"]; pool = rz["pool"]; top = rz["top"].astype(np.int64)
        assert [x.split("_", 1)[1] for x in pool] == [p.split("_", 1)[1] for p in pl], "pool order != sorted candidate ids"; assert top.shape == (400, 40) and top.min() >= 0 and top.max() < 600 and all(len(set(t)) == 40 for t in top.tolist())
        start = np.concatenate([[0], np.cumsum(ng)])[:-1]; qrows = np.full((400, 5), -1, np.int64)
        for i in range(400):
            assert qid[i] == tq[i]["question_id"].replace("pub2_", "pub_").replace("priv2_", "priv_")
            for j in range(ng[i]): assert uids[6000 + start[i] + j] == f"{qid[i]}#{j + 1}"; qrows[i, j] = 6000 + start[i] + j
        assert (qrows >= 0).sum() == 1100 and len(set(qrows[qrows >= 0].tolist())) == 1100
        for k in range(6000): assert uids[row[k]] == f"cand_{s}:{c[k]['game_id'].replace('pub2_', 'pub_').replace('priv2_', 'priv_')}"
        X, parts = channel_X(models, pools, base, rowx, lib, qrows, top, 0); assert np.isfinite(X).all()
        X2, _ = channel_X(models, pools, base, rowx, lib, qrows, top, 1); print(f"{s}: determinism across generator seeds, max|diff| = {np.abs(X - X2).max():.3e}")
        np.savez(HERE / f"real_Xp1r_{s}.npz", X=X, ng=ng, top=top); o = np.ones((400, 600), bool); np.put_along_axis(o, top, False, 1); sh = np.take_along_axis(X, top, 1); print(f"wrote real_Xp1r_{s}.npz  shortlist mean {sh.mean():.4f} std {sh.std():.4f} max-mean {sh.max(1).mean():.4f}  outside-shortlist nonzero: {int((X[o] != 0).sum())}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["equiv", "real"]); a = ap.parse_args(); {"equiv": equiv, "real": real}[a.cmd]()
