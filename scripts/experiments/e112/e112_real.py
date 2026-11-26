"""E112 amendment-1 candidate file (only after `E112A1 ADOPT`).  Subcommands
  compute   GPU (system python3 + torch).  X for the REAL test2 queries: per delivered S network (finh_full / finh_h0 / finh_h1) the same KDE likelihood-ratio channel as e112_stage2.compute
            (P12 positions, std over the pool's 6,000 candidate games, beta 20) but with the library = ALL 10 games of each of the 600 candidates and the query = its num_games query games
            (export rows 6000.. of the pool, uid 'pub_q_0001#1').  Row-z over the 600 candidates per network, mean over the three networks, no further standardisation (as in the pseudo-query X).
            Writes .scratch/e112/final/real_X_{pub,priv}.npz  (X [400,600] float32).  Reads no truths.
  make      CPU.  Production path v7s + channel, see make().  Writes .scratch/e112/final/ only.
  stats     CPU.  Basic statistics of X on the real queries next to the pseudo queries (structure a dump).
"""
import sys, os, json, csv, time, hashlib, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; sys.path[:0] = [str(HERE), str(ROOT / "scripts/lib")]; sys.dont_write_bytecode = True
import numpy as np
import e112_stage2 as S
from e112_lib import Pos, nz, kde_game_sums, BETA
OUT = ROOT / ".scratch/e112"; FIN = OUT / "final"; REAL = ROOT / ".scratch/e90/final/make"; V7S = ROOT / "data/submissions/player_v7s_一對一先驗k3.csv"; OLD800 = ROOT / ".scratch/e62/old_v6_800.csv"
POOLS = S.POOLS; NETS = S.NETS


def compute():
    import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True); t0 = time.time(); res = {s: [] for s, _, _ in POOLS}
    for net in NETS:
        P = Pos(OUT / f"t2_pos_{net}.npz"); assert P.n == 14200
        for s, sp, base in POOLS:
            row = S.cand_rows(s, sp); c = list(csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_candidates2_{sp}.csv", newline=""))); pid = np.array([r["player_id"] for r in c]); pl = sorted(set(pid)); assert len(pl) == 600
            tq = list(csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_test2_{sp}.csv", newline=""))); ng = np.array([int(r["num_games"]) for r in tq]); assert len(tq) == 400 and ng.sum() == 1100
            cg = np.unique(row); allc = np.concatenate([P.vec[P.off[base + g]:P.off[base + g + 1]] for g in cg]).astype(np.float32); sd = allc.astype(np.float64).std(0).astype(np.float32)
            W = {}
            def wpos(g):
                if g not in W: W[g] = nz(P.vec[P.off[base + g]:P.off[base + g + 1]].astype(np.float32) / sd)
                return W[g]
            segs = [np.concatenate([wpos(row[i]) for i in np.flatnonzero(pid == p)]) for p in pl]; assert all(len(np.flatnonzero(pid == p)) == 10 for p in pl)
            Lv = np.concatenate(segs); lens = np.array([len(q) for q in segs]); qstart = np.concatenate([[0], np.cumsum(ng)])[:-1]
            ug = np.arange(6000, 7100); parts = [wpos(int(g)) for g in ug]; Qv = np.concatenate(parts); Qslot = np.concatenate([np.full(len(p), j, np.int64) for j, p in enumerate(parts)])
            sums, cnt = kde_game_sums(Qv, Qslot, len(ug), Lv, lens, BETA, dev=dev); Xn = np.zeros((400, 600), np.float32)
            for i in range(400):
                js = np.arange(qstart[i], qstart[i] + ng[i]); Xn[i] = S.zrow((sums[js].sum(0) / cnt[js].sum())[None])[0]
            res[s].append(Xn); print(f"  {net} {s} done {time.time() - t0:.0f}s lib positions {len(Lv)} query positions {len(Qv)}", flush=True)
    for s, L in res.items(): X = np.mean(L, 0).astype(np.float32); np.savez(FIN / f"real_X_{s}.npz", X=X, nets=np.array(NETS)); print("wrote", f"real_X_{s}.npz", X.shape, "mean/std", float(X.mean()), float(X.std()))


def t2_csv(out, ws):
    """Production: v7s path (booster e -> csls_qq(k=3, alpha=.5, mode z) -> stable argsort -> top 5) with the channel inserted after the booster output:
    c = z_row(e over its shortlist) + w * X[q, shortlist]; then the SAME csls_qq call.  ws = {'pub': w, 'priv': w}."""
    from e81_lib import csls_qq
    rows = []
    for sp in ("pub", "priv"):
        z = np.load(REAL / f"real_v7q_S_{sp}.npz", allow_pickle=True); e, top, pool, qids = z["e"], z["top"].astype(np.int64), z["pool"], z["qids"]; w = ws[sp]
        zs = (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-9); X = np.load(FIN / f"real_X_{sp}.npz")["X"].astype(np.float64); Xs = np.take_along_axis(X, top, 1)
        c = zs + w * Xs; c = csls_qq(c, top, len(pool), 3, 0.5, "z")
        t5 = np.take_along_axis(top, np.argsort(-c, axis=1, kind="stable"), 1)[:, :5]
        rows += [[str(q)] + [str(x) for x in pool[t5[i]]] for i, q in enumerate(qids)]
    with open(out, "w", newline="") as f:
        w_ = csv.writer(f, lineterminator="\n"); w_.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]); w_.writerows(rows)


def assemble(t2, out):
    r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/oneoff/p1111_assemble.py"), "--old", str(OLD800), "--new2", str(t2), "--out", str(out)], cwd=ROOT, capture_output=True, text=True); return r.returncode, r.stdout.strip()


def check(p):
    r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/check_submission.py"), str(p), "--task", "player", "--data-dir", str(ROOT / ".scratch/p1111/t2")], cwd=ROOT, capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr).strip()


def rows_of(p):
    with open(p, newline="") as f: return list(csv.reader(f))


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make():
    res = json.loads((OUT / "result_amend1.json").read_text()); assert res["verdict"] == "E112A1 ADOPT"; wsel = res["w_for_candidate"]; ws = {"pub": wsel["public"], "priv": wsel["private"]}; R = dict(w=ws)
    for nm, w, fin in (("w0", {"pub": 0.0, "priv": 0.0}, "player_v7s_w0_repro.csv"), ("cand", ws, "player_v7s_posmatch.csv")):
        t2 = FIN / f"t2_{nm}.csv"; assert not t2.exists() and not (FIN / fin).exists(), "refusing to overwrite"; t2_csv(t2, w); rc, txt = assemble(t2, FIN / fin); assert rc == 0, txt
        R[nm] = dict(t2_sha=sha(t2), final_sha=sha(FIN / fin), assemble=txt, check_t2_rc=check(t2)[0], check_1600=check(FIN / fin))
    R["w0_final_bytes_equal_v7s"] = (FIN / "player_v7s_w0_repro.csv").read_bytes() == V7S.read_bytes()
    a, b = rows_of(V7S), rows_of(FIN / "player_v7s_posmatch.csv"); assert a[0] == b[0] and len(a) == len(b) == 1601 and [r[0] for r in a] == [r[0] for r in b]
    R["first800_identical_to_v7s"] = a[1:801] == b[1:801]; R["check_v7s_1600"] = check(V7S)
    shifts = {}
    for nm, pref in (("Public", "pub2_"), ("Private", "priv2_")):
        ia = [i for i in range(801, 1601) if a[i][0].startswith(pref)]; ch = [i for i in ia if a[i][1] != b[i][1]]
        old_rank = [a[i][1:].index(b[i][1]) + 1 if b[i][1] in a[i][1:] else 6 for i in ch]       # position of the new top-1 in the old top-5 (6 = not in old top 5)
        R[nm] = dict(rows=len(ia), top1_changed=len(ch), top5_ordered_changed=sum(a[i][1:] != b[i][1:] for i in ia), top5_set_changed=sum(set(a[i][1:]) != set(b[i][1:]) for i in ia),
                     new_top1_old_rank_dist={str(k): int(sum(1 for r in old_rank if r == k)) for k in range(2, 7)}, any_top5_changed=sum(a[i][1:] != b[i][1:] for i in ia))
    R["any_old_row_prefix_in_new_block"] = any(not a[i][0].startswith(("pub2_", "priv2_")) for i in range(801, 1601))
    with open(FIN / "make_result.json", "x") as f: f.write(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps(R, indent=1, ensure_ascii=False))


def stats():
    from e81_lib import csls_qq
    out = {}
    for s, sp, _ in POOLS:
        z = np.load(REAL / f"real_v7q_S_{s}.npz", allow_pickle=True); e, top = z["e"], z["top"].astype(np.int64); Xr = np.load(FIN / f"real_X_{s}.npz")["X"]
        d = np.load(ROOT / ".scratch/e110/dumps/t2_" f"{s}.npz"); ep, tp, nq = d["e"].astype(np.float64), d["top"].astype(np.int64), d["n"]; Xp = np.load(OUT / f"t2_X__{s}.npz")["X"]
        tq = list(csv.DictReader(open(ROOT / f"data/contest_1111/player_identification_test2_{sp}.csv", newline=""))); nr = np.array([int(r["num_games"]) for r in tq])
        def st(E, T, X, N):
            Xs = np.take_along_axis(X.astype(np.float64), T, 1); mx = Xs.max(1); agree = Xs.argmax(1) == E.argmax(1); allmax = X.max(1)
            r = dict(n=int(len(E)), short_max_z_mean=float(mx.mean()), short_max_z_sd=float(mx.std()), short_max_z_p10_p50_p90=[float(v) for v in np.percentile(mx, [10, 50, 90])], all600_max_z_mean=float(allmax.mean()),
                     x_top1_equals_booster_top1=float(agree.mean()), X_all_mean=float(X.mean()), X_all_sd=float(X.std()), X_short_mean=float(Xs.mean()), X_short_sd=float(Xs.std()))
            r["by_N"] = {str(int(n)): dict(n=int((N == n).sum()), short_max_z_mean=float(mx[N == n].mean()), x_top1_equals_booster_top1=float(agree[N == n].mean())) for n in sorted(set(N))}; return r
        out[f"{sp}/real"] = st(e, top, Xr, nr); out[f"{sp}/pseudo_a"] = st(ep, tp, Xp, nq)
    for k, v in out.items(): print(k, json.dumps(v))
    with open(FIN / "x_stats.json", "x") as f: f.write(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"compute": compute, "make": make, "stats": stats}[sys.argv[1]]()
