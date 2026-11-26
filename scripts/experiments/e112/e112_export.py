"""E112 per-position vector export (GPU, system python3 + torch).  Same code path as scripts/experiments/e90/train_e90h.py::embed_games / embed.py
(`_encode_ids` all own positions -> build_planes -> Net.pos_vec, eval mode, fp16 autocast on CUDA), but the 128-d position vectors are kept.
  --ckpt CKPT --games GAMES.npz --out OUT.npz [--rows ROWS.npy|export_spec_h.npz:KEY] [--pset PALL|P60|P12] [--limit N] [--cpu]
Output npz: vec [N,128] float16 (positions of the chosen set, game order = rows order), off [n+1] int64, ords [N] int16 (own-move ordinal of each stored position),
  rid [n] int32, gv_all [n,128] float32 = L2-normalised mean of ALL own-position vectors (original-export pooling, float32 sums), gv_first60 likewise for ordinal<60, npos_all [n].
Checkpoint: dict with key 'model' (E89 runA ckpt_latest.pt, or E90 final_model.pt); n_classes read from model['W'].  blocks 10, ch 128.
"""
import os, sys, json, time, hashlib, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".scratch/e112/numba_cache")); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True; sys.modules["coverage"] = None
sys.path[:0] = [str(ROOT / ".scratch/e89/pylib"), str(ROOT / "scripts/experiments/e90"), str(HERE)]
import numpy as np, torch
import train_e90h as T
from e112_lib import select_idx


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--games", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=""); ap.add_argument("--pset", default="PALL", choices=["PALL", "P60", "P12"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--block", type=int, default=512); ap.add_argument("--group", type=int, default=16); a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda"); amp = dev.type == "cuda"; torch.set_num_threads(2)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); sd = ck["model"]; model = T.Net(sd["W"].shape[0], 10, 128); model.load_state_dict(sd)
    model = model.to(dev).to(memory_format=torch.channels_last).eval()
    G = T.load_games(a.games)
    if a.rows:
        f, _, k = a.rows.partition(":"); rows = (np.load(f)[k] if k else np.load(f)).astype(np.int64)
    else: rows = np.arange(len(G["rid"]), dtype=np.int64)
    if a.limit: rows = rows[:a.limit]
    n = len(rows); sums = np.zeros((2, n, 128), np.float64); cnt = np.zeros((2, n), np.int64); vecs, ords, npos_sel = [], [], np.zeros(n, np.int64); t0 = time.time(); done = 0
    with torch.no_grad():
        for g0 in range(0, n, a.group):
            grp = rows[g0:g0 + a.group]; b = T._encode_ids(G, grp, None, flags=False); P = len(b["gid"]); t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}
            V = np.zeros((P, 128), np.float32)
            for x in range(0, P, a.block):
                sl = slice(x, min(x + a.block, P)); m = sl.stop - sl.start; sub = {k: v[sl] for k, v in t.items()}
                if m < a.block:
                    for k in ("grid", "lib", "prev_opp", "prev_own", "ko", "move", "ply", "oi", "eff"): sub[k] = torch.cat([sub[k], sub[k][:1].expand(a.block - m, *sub[k].shape[1:])])
                X = T.build_planes(sub, dev, torch.float16 if amp else torch.float32)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp): v = model.pos_vec(X, sub["move"].to(dev).long())[:m]
                V[sl] = v.float().cpu().numpy()
            gid = b["gid"]; oi = b["oi"].astype(np.int64)
            for j in range(len(grp)):
                s_, e_ = np.searchsorted(gid, j), np.searchsorted(gid, j, side="right"); vv, oo = V[s_:e_], oi[s_:e_]
                sums[0, g0 + j] = vv.sum(0, dtype=np.float64); cnt[0, g0 + j] = len(vv); m60 = oo < 60; sums[1, g0 + j] = vv[m60].sum(0, dtype=np.float64); cnt[1, g0 + j] = m60.sum()
                ix = select_idx(oo, a.pset); vecs.append(vv[ix].astype(np.float16)); ords.append(oo[ix].astype(np.int16)); npos_sel[g0 + j] = len(ix)
            done += P
            if (g0 // a.group) % 50 == 0: print(f"[export] game {g0}/{n} {done / (time.time() - t0):.0f} pos/s", flush=True)
    assert (cnt[0] > 0).all()
    un = lambda x: (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
    gv_all = un(sums[0] / cnt[0][:, None]); gv_f60 = un(sums[1] / np.maximum(cnt[1], 1)[:, None]); off = np.concatenate([[0], np.cumsum(npos_sel)]).astype(np.int64)
    out = Path(a.out); tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, vec=np.concatenate(vecs), off=off, ords=np.concatenate(ords), rid=G["rid"][rows], rows=rows, gv_all=gv_all, gv_first60=gv_f60, npos_all=cnt[0]); os.replace(tmp, out)
    info = dict(ckpt=a.ckpt, ckpt_sha16=sha16(a.ckpt), games=a.games, pset=a.pset, n_games=int(n), n_pos_stored=int(npos_sel.sum()), n_pos_all=int(cnt[0].sum()), device=dev.type, amp=amp, seconds=round(time.time() - t0))
    Path(str(out) + ".json").write_text(json.dumps(info, indent=1)); print("EXPORT DONE", json.dumps(info))


if __name__ == "__main__":
    main()
