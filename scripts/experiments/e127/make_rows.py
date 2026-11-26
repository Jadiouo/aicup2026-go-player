"""E127 step B: choose training / held-out players + games for the matcher.  CPU, .venv python.  Writes .scratch/e127/rows_*.npy and meta_*.npz.
net finh_h0 never saw players that are only in the h1 roster; net finh_h1 never saw players only in the h0 roster (checked below: those players have 0 games in the net's roster mask).
Held-out (val) players: 300 per net, 16 games each (protocol: 10 library + 5 rest); train: all remaining players with >= 16 games, min(24, n) games each."""
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; OUT = ROOT / ".scratch/e127"; OUT.mkdir(exist_ok=True)
G = np.load(ROOT / ".scratch/e90/data/uh_games.npz"); player, eff, hd = G["player"], G["eff"], G["hd"]
R = np.load(ROOT / ".scratch/e90/data/rosters_h.npz"); h0, h1 = R["finh_h0"], R["finh_h1"]
ph0, ph1 = set(player[h0].tolist()), set(player[h1].tolist())
spec = {"finh_h0": (np.array(sorted(ph1 - ph0)), h0), "finh_h1": (np.array(sorted(ph0 - ph1)), h1)}
keep = np.flatnonzero(hd <= 1)   # test games have hd<=1 only
order = keep[np.argsort(player[keep], kind="stable")]; ps = player[order]; starts = np.searchsorted(ps, np.arange(player.max() + 2))
rng = np.random.default_rng(127); info = {}
for net, (pl, mask) in spec.items():
    assert not mask[np.isin(player, pl)].any(), "unseen players have games in the net's own roster"
    cnt = np.array([starts[p + 1] - starts[p] for p in pl]); ok = pl[cnt >= 16]; perm = rng.permutation(len(ok)); val, tr = np.sort(ok[perm[:300]]), np.sort(ok[perm[300:]])
    assert not (set(val.tolist()) & set(tr.tolist()))
    def pick(players, k):
        rows, pp = [], []
        for p in players:
            g = order[starts[p]:starts[p + 1]]; g = np.sort(rng.permutation(g)[:k]); rows.append(g); pp.append(np.full(len(g), p))
        return np.concatenate(rows).astype(np.int64), np.concatenate(pp)
    rv, pv = pick(val, 16); rt, pt = pick(tr, 24)
    h = len(tr) // 2; ia = np.isin(pt, tr[:h]); parts = {"val": (rv, pv), "trA": (rt[ia], pt[ia]), "trB": (rt[~ia], pt[~ia])}
    for nm, (r, p) in parts.items():
        np.save(OUT / f"rows_{net}_{nm}.npy", r); np.savez(OUT / f"meta_{net}_{nm}.npz", rows=r, player=p, eff=eff[r], hd=hd[r])
    info[net] = {k: len(v[0]) for k, v in parts.items()}; info[net].update(n_val_players=len(val), n_train_players=len(tr), hd_gt1_frac=float((hd[np.concatenate([rv, rt])] > 1).mean()))
    print(net, info[net])
import json; (OUT / "rows_info.json").write_text(json.dumps(info, indent=1))
