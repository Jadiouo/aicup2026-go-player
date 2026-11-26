"""E110 analysis: CSLS-qq (k=10, alpha=.5) vs (k=3, alpha=.5), mode z.  Based on experiments/E90_player_csls_recheck/e90_an.py (same per-query scoring e81_lib.sc / rank_after / csls_qq on the stored float32 booster outputs,
same cluster bootstrap e81_lib.boot_ci with B=2000, seed 0, clusters = reps for pseudo queries, pools for simulation).  Only these two settings, no sweep.
Changes against e90_an.py: (a) two settings instead of one, delta = (3,.5) - (10,.5) is the quantity judged (each setting minus plain also reported);  (b) dump directories are arguments;  (c) outputs go to .scratch/e110/ only
(`open(...,"x")` for the formal result);  (d) mode `repro` recomputes (10,.5) minus plain on the existing v7p dumps (.scratch/e90_csls) and prints them next to the E90 RESULTS.md table;  mode `formal` = the pre-registered
PLAN.md criteria.
Usage:  .venv/bin/python scripts/experiments/e110/e110_an.py repro | formal"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/E81_player_error_analysis")]
from e81_lib import sc, csls_qq, rank_after, boot_ci
SETS = {"k10": 10, "k3": 3}; ALPHA, MODE = 0.5, "z"
V7P = ROOT / ".scratch/e90_csls"; V7Q = ROOT / ".scratch/e110/dumps"

def three(e, top, truth, P):
    out = {}; r0, _ = rank_after(e, top, truth); out["plain"] = sc(r0)
    for nm, k in SETS.items():
        r1, _ = rank_after(csls_qq(e, top, P, k, ALPHA, MODE), top, truth); out[nm] = sc(r1)
    return out

def summ(s, grp):
    pl, k10, k3 = s["plain"], s["k10"], s["k3"]
    def ci(d): lo, hi = boot_ci(d, grp, 2000, 0); return dict(delta=float(d.mean()), ci=[float(lo), float(hi)])
    return dict(n=int(len(pl)), plain=float(pl.mean()), k10=float(k10.mean()), k3=float(k3.mean()), k10_minus_plain=ci(k10 - pl), k3_minus_plain=ci(k3 - pl), k3_minus_k10=ci(k3 - k10))

def sim(path, structs=None):
    z = np.load(path); tags = [str(t) for t in z["tag"]]; res = {}
    for st in sorted({t[0] for t in tags}):
        parts, grp = [], []
        for i, t in enumerate(tags):
            if t[0] != st: continue
            parts.append(three(z["e"][i], z["top"][i].astype(np.int64), z["truth"][i], len(z["pl"][i]))); grp.append(np.full(len(parts[-1]["plain"]), i))
        s = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}; r = summ(s, np.concatenate(grp)); r["pools"] = len(parts); res[st] = r
    return res

def t2(dumpdir, tag):
    res = {}
    for s, nm in (("pub", "Public"), ("priv", "Private")):
        z = np.load(Path(dumpdir) / f"t2{tag}_{s}.npz"); P = len(z["players"]); rep = z["rep"]; parts, grp = [], []
        for r in sorted(set(rep)):
            m = rep == r; parts.append(three(z["e"][m], z["top"][m].astype(np.int64), z["truth"][m], P)); grp.append(np.full(int(m.sum()), r))
        s_ = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}; res[nm] = summ(s_, np.concatenate(grp))
    return res

def fmt(name, r):
    return (f"{name:34s} plain {r['plain']:.4f} | (10,.5) {r['k10']:.4f} d {r['k10_minus_plain']['delta']:+.4f} [{r['k10_minus_plain']['ci'][0]:+.4f}, {r['k10_minus_plain']['ci'][1]:+.4f}]"
            f" | (3,.5) {r['k3']:.4f} d {r['k3_minus_plain']['delta']:+.4f} [{r['k3_minus_plain']['ci'][0]:+.4f}, {r['k3_minus_plain']['ci'][1]:+.4f}]"
            f" | (3,.5)-(10,.5) {r['k3_minus_k10']['delta']:+.4f} [{r['k3_minus_k10']['ci'][0]:+.4f}, {r['k3_minus_k10']['ci'][1]:+.4f}]")

def repro():
    print("REPRO (a): v7p dumps in .scratch/e90_csls, (10,.5) minus plain; compare with experiments/E90_player_csls_recheck/RESULTS.md")
    for nm in ("s56a", "s56c", "s12"):
        for st, r in sim(V7P / f"sim_{nm}.npz").items(): print(fmt(f"sim {nm} struct {st} ({r['pools']} pools)", r))
    for tag in ("", "c", "b"):
        for sp, r in t2(V7P, tag).items(): print(fmt(f"t2 struct {tag or 'a'} {sp}", r))

def formal():
    out_path = ROOT / ".scratch/e110/result.json"
    assert not out_path.exists(), f"{out_path} exists: the formal reading runs once"
    R = dict(setting="(3,.5) - (10,.5); delta per query, cluster bootstrap B=2000 seed 0", t2={}, sim={})
    for tag in ("", "c", "b"): R["t2"][tag or "a"] = t2(V7Q, tag)
    R["sim"]["s56a_v7p_fallback"] = sim(V7P / "sim_s56a.npz")                      # criterion 3 (E90 precedent: seeds 5,6, fresh pools)
    R["sim"]["s12_v7p_fallback_info"] = sim(V7P / "sim_s12.npz"); R["sim"]["s56c_v7p_fallback_info"] = sim(V7P / "sim_s56c.npz")
    # info only: 16 pools of structure a (seeds 1,2 + 5,6) as in the consult c8 report
    za, zb = np.load(V7P / "sim_s12.npz"), np.load(V7P / "sim_s56a.npz"); parts, grp, i0 = [], [], 0
    for z in (za, zb):
        for i, t in enumerate(z["tag"]):
            if str(t)[0] != "a": continue
            parts.append(three(z["e"][i], z["top"][i].astype(np.int64), z["truth"][i], len(z["pl"][i]))); grp.append(np.full(len(parts[-1]["plain"]), i0)); i0 += 1
    R["sim"]["a_16pools_v7p_info"] = {"a": {**summ({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, np.concatenate(grp)), "pools": len(parts)}}
    d = lambda x: x["k3_minus_k10"]
    c1 = {sp: d(R["t2"]["a"][sp])["ci"][0] > 0 for sp in ("Public", "Private")}
    c2 = {sp: d(R["t2"]["c"][sp])["delta"] >= -0.005 for sp in ("Public", "Private")}
    c3 = d(R["sim"]["s56a_v7p_fallback"]["a"])["ci"][0] > 0
    R["criteria"] = dict(c1_pseudo_a_ci_lower_gt0=c1, c2_pseudo_c_point_ge_minus005=c2, c3_sim_a_ci_lower_gt0=bool(c3))
    ok = all(c1.values()) and all(c2.values()) and c3
    fails = [f"1(偽查詢a下界{ {sp: round(d(R['t2']['a'][sp])['ci'][0], 4) for sp in c1} })" for _ in [0] if not all(c1.values())] \
          + [f"2(偽查詢c點估計{ {sp: round(d(R['t2']['c'][sp])['delta'], 4) for sp in c2} })" for _ in [0] if not all(c2.values())] \
          + [f"3(模擬a下界{round(d(R['sim']['s56a_v7p_fallback']['a'])['ci'][0], 4)})" for _ in [0] if not c3]
    R["verdict"] = "E110 ADOPT" if ok else "E110 KEEP " + "、".join(fails)
    with open(out_path, "x") as f: f.write(json.dumps(R, indent=1, ensure_ascii=False))
    print("t2 pseudo queries (v7q booster dumps):")
    for st in ("a", "c", "b"):
        for sp, r in R["t2"][st].items(): print(fmt(f"  t2 struct {st} {sp}", r))
    print("simulation (v7p fallback dumps; the v7q simulation version could not be made):")
    for nm, v in R["sim"].items():
        for st, r in v.items(): print(fmt(f"  sim {nm} struct {st} ({r['pools']} pools)", r))
    print("criteria:", json.dumps(R["criteria"], ensure_ascii=False))
    print(f"  1. pseudo a: lower bounds Public {d(R['t2']['a']['Public'])['ci'][0]:+.4f}, Private {d(R['t2']['a']['Private'])['ci'][0]:+.4f} -> {'PASS' if all(c1.values()) else 'FAIL'}")
    print(f"  2. pseudo c: points Public {d(R['t2']['c']['Public'])['delta']:+.4f}, Private {d(R['t2']['c']['Private'])['delta']:+.4f} (>= -.005) -> {'PASS' if all(c2.values()) else 'FAIL'}")
    print(f"  3. sim a (v7p s56a): lower bound {d(R['sim']['s56a_v7p_fallback']['a'])['ci'][0]:+.4f} -> {'PASS' if c3 else 'FAIL'}")
    print(R["verdict"])

if __name__ == "__main__":
    {"repro": repro, "formal": formal}[sys.argv[1]]()
