#!/usr/bin/env python
"""P1111 step 1: format + statistics check of the 2026-11-11 Player test data (read-only on the data; writes only .scratch/p1111/).

  .venv/bin/python scripts/oneoff/p1111_inspect.py --data-dir data/contest_1111                 # format, stats, overlap with 11/04, SE/HSL plan
  .venv/bin/python scripts/oneoff/p1111_inspect.py --data-dir data/contest_1111 --exc            # + E30 exc statistic (needs L0 features of all games: minutes of CPU)
  .venv/bin/python scripts/oneoff/p1111_inspect.py --data-dir data/contest_1104 --ref-dir data/contest_1104 --exc --z-cache-dir .scratch/e27   # dry run on the old data

What it does (no guessing: anything it cannot classify makes it STOP with a raw-text excerpt, exit code 3):
  1. finds the CSV files, classifies each by HEADER (question file / candidate file) and by FILE NAME (rank/player, public/private); column aliases it accepted
     are printed in a mapping table.  Writes canonical-named copies (QUESTION_COLS / CAND_COLS of gogame.io) to .scratch/p1111/normalized/ when needed.
  2. per split: #questions, #candidates, games per candidate, N per question, colours (white?), root properties of the SGFs, one-to-one indicator
     (#questions == #candidates), duplicate ids, question games that also sit in a candidate library, gogame.io validation issue codes.
  3. overlap with the 11/04 data (--ref-dir, default data/contest_1104) by a MOVE-SEQUENCE key (colour + all B/W moves, header-insensitive); exact-same questions;
     candidates with identical game sets; which games already have SE (.scratch/e27/se) / HSL (.scratch/e27/hsl) features -> what must still be computed.
  4. writes the work plan to .scratch/p1111/plan/: se_todo.parquet + se_in/c*.parquet (10000-game chunks), hsl_todo.parquet (same schema as e27 hsl_games_in.parquet),
     reuse_se.parquet / reuse_hsl.parquet (new_uid -> old_uid), hsl_lib_sel.json (the library games that must have HSL), summary.json.
  5. (--exc) E30 detector: first-pass E04v2 scores (no SE/HSL) -> exc = c_all - (1 - mean p1) per split vs theta in experiments/E30_player_robustness/tuned.json.
Exit codes: 0 ok (read the VERDICT lines), 2 ok but needs human review (REVIEW flags), 3 stopped: format not understood.
"""
import argparse, csv, hashlib, json, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("GOGAME_COLOR_SEMANTICS", "actual")
import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)
OUT = ROOT / ".scratch" / "p1111"
OLD_SE = ROOT / ".scratch" / "e27" / "se" / "se_test_games.parquet"
OLD_HSL = ROOT / ".scratch" / "e27" / "hsl" / "hsl_test_games.parquet"
TUNED = ROOT / "experiments" / "E30_player_robustness" / "tuned.json"
SE_RATE, HSL_RATE = 23.6, 1.7          # games/s: E27 school SE 10000 games/424 s; E27 school HSL 1.37-1.76 (S9101-S9107)
HSL_SUB, HSL_SEED = 20, 20             # e27_player_fusion: 20 library games per candidate (hsl5s20)
CHUNK = 10000

QCAN = ["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)] + [f"color_{i}" for i in range(1, 21)]
CCAN = ["game_id", "player_id", "sgf_content", "color"]
ALIAS_Q = {"question_id": ["question_id", "qid", "id"], "num_games": ["num_games", "n_games", "ngames", "num_game"]}
ALIAS_C = {"game_id": ["game_id", "gid"], "player_id": ["player_id", "pid", "candidate_id"], "sgf_content": ["sgf_content", "sgf"], "color": ["color", "colour"]}
RE_SGF = re.compile(r"^(?:sgf|game)_?(\d+)$"); RE_COL = re.compile(r"^(?:color|colour)_?(\d+)$")
RE_MOVE = re.compile(r";([BW])\[([a-z]{0,2})\]")
RE_PROP = re.compile(r"(?<![A-Za-z])([A-Z]{1,2})\[")
KNOWN_ROOT = {"SZ", "HD", "CA", "RE", "AB", "AW"}
FLAGS: list[str] = []


def flag(msg):
    FLAGS.append(msg); print(f"  !! REVIEW: {msg}")


class Stop(Exception):
    pass


def raw_excerpt(path: Path, n=420):
    try:
        b = path.read_bytes()[:n]
        return b.decode("utf-8", "replace").replace("\n", "\\n")
    except Exception as e:      # noqa: BLE001
        return f"<unreadable: {e}>"


def stop(msg, paths=()):
    print("\n" + "=" * 100 + f"\nSTOP (format not understood, nothing guessed): {msg}")
    for p in paths:
        p = Path(p)
        print(f"--- {p} ({p.stat().st_size if p.exists() else '?'} bytes)")
        if p.suffix.lower() == ".zip":
            import zipfile
            print("zip members:", zipfile.ZipFile(p).namelist()[:30])
            continue
        try:
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                hdr = next(csv.reader(f), [])
                row = next(csv.reader(f), [])
            print(f"header ({len(hdr)} cols): {hdr[:45]}{' ...' if len(hdr) > 45 else ''}")
            print("first data row (cells cut to 70 chars):", [c[:70] for c in row[:12]], "..." if len(row) > 12 else "")
        except Exception:   # noqa: BLE001
            pass
        print("first bytes:", raw_excerpt(p))
    raise Stop(msg)


# ----------------------------------------------------------------------------- discovery and normalisation
def header_of(p: Path):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return next(csv.reader(f), [])


def pick(hdr, aliases):
    for a in aliases:
        if a in hdr:
            return a
    return None


def classify(p: Path):
    """returns dict(kind=question|candidate, task=rank|player, split=pub|priv, mapping={canonical: source}, extra=[...])"""
    hdr = header_of(p)
    name = p.name.lower()
    toks = re.split(r"[^a-z0-9]+", name)
    split = "pub" if ("public" in toks or "pub" in toks) else "priv" if ("private" in toks or "priv" in toks) else None
    task = "rank" if "rank" in toks else "player" if ("player" in toks or "identification" in toks or "candidates" in toks) else None
    qid, ng = pick(hdr, ALIAS_Q["question_id"]), pick(hdr, ALIAS_Q["num_games"])
    sg = {int(RE_SGF.match(h).group(1)): h for h in hdr if RE_SGF.match(h)}
    co = {int(RE_COL.match(h).group(1)): h for h in hdr if RE_COL.match(h)}
    cmap = {k: pick(hdr, v) for k, v in ALIAS_C.items()}
    if qid and ng and sg and co:
        if max(sg) > 20 or max(co) > 20:
            stop(f"{p.name}: sgf/color columns go beyond 20 (max sgf_{max(sg)}); gogame.io handles 20 games per question", [p])
        if set(sg) != set(co) or sorted(sg) != list(range(1, len(sg) + 1)):
            stop(f"{p.name}: sgf_i / color_i columns are not the contiguous pairs 1..n", [p])
        mapping = {"question_id": qid, "num_games": ng, **{f"sgf_{i}": sg[i] for i in sg}, **{f"color_{i}": co[i] for i in co}}
        used = set(mapping.values())
        return dict(kind="question", task=task, split=split, mapping=mapping, extra=[h for h in hdr if h not in used], header=hdr)
    if all(cmap.values()):
        used = set(cmap.values())
        return dict(kind="candidate", task="player", split=split, mapping=cmap, extra=[h for h in hdr if h not in used], header=hdr)
    return None


def load_norm(p: Path, info):
    """canonical-column DataFrame (all str).  QCAN gets empty cells for sgf_i/color_i beyond what the file has."""
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    m = info["mapping"]
    cols = QCAN if info["kind"] == "question" else CCAN
    out = pd.DataFrame({c: (df[m[c]] if c in m else "") for c in cols})
    return out, df


def discover(data_dir: Path, label: str):
    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    if not files:
        stop(f"{label}: {data_dir} has no files", [])
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    others = [p for p in files if p.suffix.lower() != ".csv"]
    for o in others:
        print(f"  [{label}] non-CSV file ignored: {o.name} ({o.stat().st_size} bytes)")
    if not csvs:
        stop(f"{label}: no CSV in {data_dir} (zip / parquet / json?): unpack or tell me the format", others[:6])
    found = {}
    for p in csvs:
        info = classify(p)
        if info is None:
            stop(f"{label}: cannot classify {p.name} (header matches neither a question file nor a candidate file)", [p])
        if info["split"] is None or (info["kind"] == "question" and info["task"] is None):
            stop(f"{label}: cannot tell task (rank/player) or split (public/private) from the file name {p.name}", [p])
        info["path"] = p
        key = (info["kind"], info["task"], info["split"])
        if key in found:
            stop(f"{label}: two files for {key}: {found[key]['path'].name} and {p.name} (E30: never mix candidate files; decide by hand)", [found[key]["path"], p])
        found[key] = info
    return found


# ----------------------------------------------------------------------------- game keys
def mkey(sgf: str, color: str) -> str:
    mv = RE_MOVE.findall(sgf)
    h = hashlib.sha1((color + "|" + ";".join(c + x for c, x in mv)).encode()).hexdigest()[:20]
    return h


def root_props(sgf: str):
    head = sgf.split(";B[", 1)[0].split(";W[", 1)[0]
    return RE_PROP.findall(head)


def explode_q(df: pd.DataFrame, split: str):
    """rows: (uid, qid, i, color, sgf) with gogame.io uid rule '<normalised qid>#<column index>'."""
    from gogame.io import normalize_qid
    rows = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        qid = normalize_qid(str(d["question_id"]), split)
        n = int(d["num_games"]) if str(d["num_games"]).isdigit() else 20
        k = 0
        for i in range(1, 21):
            s = d[f"sgf_{i}"]
            if s == "":
                continue
            rows.append((f"{qid}#{i}", qid, i, d[f"color_{i}"], s)); k += 1
            if k == n:
                break
    return pd.DataFrame(rows, columns=["uid", "qid", "slot", "color", "sgf"])


def cand_games(df: pd.DataFrame, split: str):
    from gogame.io import normalize_pid
    d = pd.DataFrame({"uid": [f"cand_{split}:{g}" for g in df.game_id], "pid": [normalize_pid(p, split) for p in df.player_id],
                      "color": df.color.to_numpy(), "sgf": df.sgf_content.to_numpy()})
    return d


# ----------------------------------------------------------------------------- per split report
def split_report(sp, qdf, cdf, info_q, info_c):
    r = {"split": sp}
    from gogame.io import validate_questions_frame
    qg = explode_q(qdf, sp)
    cg = cand_games(cdf, sp)
    nq, nc = len(qdf), cg.pid.nunique()
    r.update(nq=nq, n_cand=nc, q_games=len(qg), cand_games=len(cg))
    print(f"\n== split {sp}: questions {info_q['path'].name}  |  candidates {info_c['path'].name}")
    print(f"  questions {nq} (unique ids {qdf.question_id.nunique()}), candidates {nc}, candidate games {len(cg)}, query games {len(qg)}")
    if qdf.question_id.duplicated().any():
        flag(f"{sp}: duplicate question_id ({int(qdf.question_id.duplicated().sum())})")
    # N
    declared = pd.to_numeric(qdf.num_games, errors="coerce")
    actual = (qdf[[f"sgf_{i}" for i in range(1, 21)]] != "").sum(axis=1)
    nd = declared.value_counts(dropna=False).sort_index()
    r["N_dist"] = {str(k): int(v) for k, v in nd.items()}
    print("  N per question (declared):", dict(nd.items()), "| declared != non-empty sgf count:", int((declared != actual).sum()))
    if (declared != actual).any():
        flag(f"{sp}: num_games differs from the number of non-empty sgf columns in {(declared != actual).sum()} questions")
    odd = sorted(set(declared.dropna().astype(int)) - {5, 10, 15, 20})
    if odd:
        flag(f"{sp}: N values outside 5/10/15/20: {odd} (E30 S1 saw small N exceed theta: exc may false-alarm)")
    # colours
    qc = qdf[[f"color_{i}" for i in range(1, 21)]].apply(lambda row: "".join(sorted(set(x for x in row if x != ""))), axis=1)
    cc = qc.value_counts().to_dict()
    r["q_color"] = {k: int(v) for k, v in cc.items()}
    wq = int(sum(v for k, v in cc.items() if "W" in k))
    cw = int((cg.color == "W").sum()); cwp = int(cg[cg.color == "W"].pid.nunique())
    r.update(white_questions=wq, mixed_color_questions=int(sum(v for k, v in cc.items() if len(k) > 1)), cand_white_games=cw, cand_players_with_white=cwp)
    print(f"  question colours {cc} -> questions containing W: {wq};  candidate games: B {int((cg.color == 'B').sum())} W {cw} (players with W games: {cwp})")
    other = set(cg.color) - {"B", "W"}
    if other:
        stop(f"{sp}: candidate colour values outside B/W: {sorted(other)[:5]}", [info_c["path"]])
    if wq or cw:
        flag(f"{sp}: WHITE present (questions with W: {wq}, candidate W games: {cw}) -> io.py Q_NOT_BLACK blocks load_questions; see runbook section on white")
    # games per candidate
    gp = cg.groupby("pid").size()
    r["games_per_cand"] = dict(min=int(gp.min()), median=float(gp.median()), max=int(gp.max()))
    print(f"  games per candidate: min {gp.min()} median {gp.median():.0f} max {gp.max()}")
    if gp.min() < 20:
        flag(f"{sp}: some candidates have <20 games (min {gp.min()}): the HSL 20-per-candidate rule cannot hold; E30 S2 libraries <20 are weak")
    # one-to-one indicator
    r["nq_eq_nc"] = bool(nq == nc)
    print(f"  one-to-one indicator: nq == n_cand ? {nq == nc}  (nq {nq}, n_cand {nc}; necessary, not sufficient: truth is unknown)")
    if nq != nc:
        flag(f"{sp}: nq ({nq}) != n_cand ({nc}): not a bijection by construction -> E30 S4c/S5c-type structure, prefer the C* branch")
    # root properties
    cnt = {"HD": {}, "SZ": {}, "other": {}}
    for s in pd.concat([qg.sgf.sample(min(len(qg), 3000), random_state=0), cg.sgf.sample(min(len(cg), 6000), random_state=0)]):
        pr = root_props(s)
        for p in pr:
            if p in ("HD", "SZ"):
                m = re.search(p + r"\[([^\]]*)\]", s)
                cnt[p][m.group(1) if m else "?"] = cnt[p].get(m.group(1) if m else "?", 0) + 1
            elif p not in KNOWN_ROOT:
                cnt["other"][p] = cnt["other"].get(p, 0) + 1
    r["root_props_sample"] = cnt
    print(f"  root props (sample 9000 games): HD {cnt['HD']} SZ {cnt['SZ']} unknown {cnt['other']}")
    if cnt["other"]:
        flag(f"{sp}: unknown SGF root properties {cnt['other']}: run scripts/predict.py report (runbook_1104 1.5) and read them before going on")
    hd2 = {k: v for k, v in cnt["HD"].items() if k not in ("0", "1")}
    if hd2:
        flag(f"{sp}: handicap games (HD>=2) present in the sample {hd2}: colour semantics (actual) matters for them")
    # question games that also sit in candidate libraries
    qk = set(mkey(s, c) for s, c in zip(qg.sgf, qg.color)); ck = set(mkey(s, c) for s, c in zip(cg.sgf, cg.color))
    inter = len(qk & ck)
    r["query_games_also_in_library"] = inter
    print(f"  query games whose move sequence also appears in a candidate library: {inter} of {len(qk)} distinct")
    if inter:
        flag(f"{sp}: {inter} query games also appear in candidate libraries (E27 on 11/04 had 0): check before trusting the library")
    # gogame.io validation (what predict.py's load_questions would say)
    iss = validate_questions_frame(qdf, "player")
    codes = {}
    for i in iss:
        codes[i.code] = codes.get(i.code, 0) + 1
    r["io_issues"] = codes
    print("  gogame.io.validate_questions_frame(task='player'):", codes if codes else "no issues")
    if set(codes) - {"W_NUM_GAMES"}:
        flag(f"{sp}: predict.py load_questions would raise on {sorted(set(codes) - {'W_NUM_GAMES'})} (Q_NOT_BLACK = white questions)")
    return r, qg, cg


# ----------------------------------------------------------------------------- overlap + plan
def add_keys(g: pd.DataFrame):
    g = g.copy()
    g["key"] = [mkey(s, c) for s, c in zip(g.sgf, g.color)]
    g["tkey"] = [hashlib.sha1(s.encode()).hexdigest()[:20] for s in g.sgf]
    return g


def overlap_and_plan(new, ref, splits_new):
    """new/ref: dict split -> (qg, cg).  Returns summary dict; writes the plan."""
    plan = OUT / "plan"; (plan / "se_in").mkdir(parents=True, exist_ok=True)
    allnew = pd.concat([add_keys(pd.concat([qg.assign(kind="q", pid=qg.qid), cg.assign(kind="c", qid=cg.pid, slot=0)], ignore_index=True)).assign(split=sp)
                        for sp, (qg, cg) in new.items()], ignore_index=True)
    allold = pd.concat([add_keys(pd.concat([qg.assign(kind="q", pid=qg.qid), cg.assign(kind="c", qid=cg.pid, slot=0)], ignore_index=True)).assign(split=sp)
                        for sp, (qg, cg) in ref.items()], ignore_index=True)
    s = {"new_games": len(allnew), "ref_games": len(allold)}
    old_by_key = allold.drop_duplicates("key").set_index("key")
    allnew["old_uid"] = allnew.key.map(old_by_key.uid)
    allnew["old_tkey_same"] = allnew.tkey == allnew.old_uid.map(allold.drop_duplicates("uid").set_index("uid").tkey)
    m = allnew.old_uid.notna()
    s.update(games_in_ref=int(m.sum()), games_in_ref_exact_text=int((m & allnew.old_tkey_same).sum()))
    print(f"\n== overlap with the 11/04 data (key = colour + move sequence): {m.sum()} of {len(allnew)} new games already existed; exact-same sgf text: {(m & allnew.old_tkey_same).sum()}")
    if (m & ~allnew.old_tkey_same).any():
        flag(f"{int((m & ~allnew.old_tkey_same).sum())} games have the same moves as 11/04 but a different SGF text (header changed): reuse is by moves; check the colour/HD semantics before reusing SE/HSL")
    # questions / candidates identical to old ones
    for kind in ("q", "c"):
        a = allnew[allnew.kind == kind]; o = allold[allold.kind == kind]
        grp = lambda d: d.groupby("pid")["key"].apply(lambda x: hashlib.sha1("|".join(sorted(x)).encode()).hexdigest()[:20])   # noqa: E731
        gn, go = grp(a), grp(o)
        same = gn.isin(set(go))
        s[f"{kind}_units"] = int(len(gn)); s[f"{kind}_units_identical_to_ref"] = int(same.sum())
        lab = "questions" if kind == "q" else "candidates"
        print(f"  {lab}: {len(gn)} new, {int(same.sum())} have exactly the same game set as an 11/04 {lab[:-1]} (any id)")
    ids_new = set(allnew[allnew.kind == "q"].pid); ids_old = set(allold[allold.kind == "q"].pid)
    s["qid_in_both"] = len(ids_new & ids_old)
    if ids_new & ids_old:
        gn = allnew[allnew.kind == "q"].groupby("pid")["key"].apply(lambda x: tuple(sorted(x))); go = allold[allold.kind == "q"].groupby("pid")["key"].apply(lambda x: tuple(sorted(x)))
        com = sorted(ids_new & ids_old)
        diff = [q for q in com if gn[q] != go[q]]
        s["qid_same_content_differs"] = len(diff)
        print(f"  question ids present in both: {len(com)}; with DIFFERENT game content under the same id: {len(diff)}")
        if diff:
            flag(f"{len(diff)} question ids exist in 11/04 with different content: never reuse features/answers by id, only by move key")
    cn = set(allnew[allnew.kind == "c"].pid); co = set(allold[allold.kind == "c"].pid)
    s["cand_ids_in_both"] = len(cn & co)
    print(f"  candidate ids present in both: {len(cn & co)}")
    # feature coverage
    se_old = pd.read_parquet(OLD_SE, columns=["uid"]).uid if OLD_SE.exists() else pd.Series([], dtype=str)
    hsl_old = pd.read_parquet(OLD_HSL, columns=["uid"]).uid if OLD_HSL.exists() else pd.Series([], dtype=str)
    se_has, hsl_has = set(se_old), set(hsl_old)
    allnew["se_old"] = allnew.old_uid.where(allnew.old_uid.isin(se_has))
    allnew["hsl_old"] = allnew.old_uid.where(allnew.old_uid.isin(hsl_has))
    # SE: every candidate + query game
    # SE/HSL metrics were fitted on BLACK games only (player_sim3.meta_black) and never validated on white: features are computed for black games only,
    # white questions get no fusion bonus (v3 only) -- see runbook section 7.  All-black data: no effect.
    isB = allnew.color == "B"
    s["white_games_without_features"] = int((~isB).sum())
    se_todo = allnew[allnew.se_old.isna() & isB].drop_duplicates("uid")
    reuse_se = allnew[allnew.se_old.notna() & isB][["uid", "se_old"]].rename(columns={"uid": "new_uid", "se_old": "old_uid"})
    # HSL: queries all; library = HSL_SUB per candidate, reuse-first, then seeded fill
    sel_rows, lib_sel = [], {}
    for (sp, pid), g in allnew[(allnew.kind == "c") & isB].groupby(["split", "pid"], sort=True):
        g = g.sort_values("uid")
        have = g[g.hsl_old.notna()]
        k = min(HSL_SUB, len(g))
        take = list(have.uid[:k])
        if len(take) < k:
            rest = g[~g.uid.isin(take)]
            rg = np.random.default_rng(int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16))
            take += list(rest.uid.to_numpy()[rg.permutation(len(rest))[:k - len(take)]])
        lib_sel[pid] = sorted(take)
    selset = set(u for v in lib_sel.values() for u in v)
    need_h = allnew[(((allnew.kind == "q") & isB) | allnew.uid.isin(selset))].drop_duplicates("uid")
    hsl_todo = need_h[need_h.hsl_old.isna()]
    reuse_hsl = need_h[need_h.hsl_old.notna()][["uid", "hsl_old"]].rename(columns={"uid": "new_uid", "hsl_old": "old_uid"})
    sch = lambda d: pd.DataFrame({"uid": d.uid.to_numpy(), "color": d.color.map({"B": 0, "W": 1}).astype(np.int8).to_numpy(), "sgf": d.sgf.to_numpy()})   # noqa: E731
    sch(se_todo).to_parquet(plan / "se_todo.parquet", index=False)
    for f in (plan / "se_in").glob("c*.parquet"):
        f.unlink()
    for k, a in enumerate(range(0, len(se_todo), CHUNK)):
        sch(se_todo.iloc[a:a + CHUNK]).to_parquet(plan / "se_in" / f"c{k}.parquet", index=False)
    sch(hsl_todo).to_parquet(plan / "hsl_todo.parquet", index=False)
    reuse_se.to_parquet(plan / "reuse_se.parquet", index=False); reuse_hsl.to_parquet(plan / "reuse_hsl.parquet", index=False)
    (plan / "hsl_lib_sel.json").write_text(json.dumps(lib_sel))
    allnew[["uid", "split", "kind", "pid", "slot", "color", "key", "old_uid"]].to_parquet(plan / "games_new.parquet", index=False)
    s.update(se_total=int(allnew[isB].uid.nunique()), se_reuse=int(len(reuse_se)), se_todo=int(len(se_todo)), se_chunks=int(np.ceil(len(se_todo) / CHUNK)),
             hsl_total=int(len(need_h)), hsl_reuse=int(len(reuse_hsl)), hsl_todo=int(len(hsl_todo)),
             se_eta_min=round(len(se_todo) / SE_RATE / 60, 1), hsl_eta_min=round(len(hsl_todo) / HSL_RATE / 60, 1))
    print(f"\n== work plan (written to {plan})")
    print(f"  SE : need {s['se_total']} games, reuse {s['se_reuse']}, TO COMPUTE {s['se_todo']} in {s['se_chunks']} chunks -> ~{s['se_eta_min']} min on school ({SE_RATE} games/s, E27)")
    print(f"  HSL: need {s['hsl_total']} games (all query games + {HSL_SUB}/candidate), reuse {s['hsl_reuse']}, TO COMPUTE {s['hsl_todo']} -> ~{s['hsl_eta_min']} min = {s['hsl_eta_min'] / 60:.1f} h on school ({HSL_RATE} games/s, E27)")
    return s


# ----------------------------------------------------------------------------- E30 exc
def exc_stage(splits, cache_dir: Path):
    import predict as P
    from gogame import player_rerank as prr
    from gogame.io import explode_question, iter_candidate_games
    tj = json.loads(TUNED.read_text()); ch = tj["chosen"]
    pm = P.PLAYER_MODELS["e04v2"].load(ROOT / "data" / "models_final", 4)
    print(f"\n== E30 exc (first-pass E04v2 = data/models_final, include_holdout={pm.meta.get('include_holdout')}); decision stat {ch['stat']}, theta {ch['theta']:.4f}, F={ch['F']}")
    res = {}
    for sp, (qn, cn) in splits.items():
        cache_dir.mkdir(parents=True, exist_ok=True)
        pq = [(q, [g for g in explode_question(r, sp)]) for q, r in zip(qn.question_id, qn.to_dict("records"))]
        cands = {}
        from gogame.io import normalize_pid
        cn2 = cn.copy(); cn2["player_id"] = [normalize_pid(p, sp) for p in cn2.player_id]; cn2["uid"] = [f"cand_{sp}:{g}" for g in cn2.game_id]
        for g in iter_candidate_games(cn2, sp):
            cands.setdefault(g.player_id, []).append(g)
        pool = sorted(cands)
        games = [g for p in pool for g in cands[p]] + [g for _, gs in pq for g in gs]
        zp = cache_dir / f"Z_{sp}.npz"
        t0 = time.perf_counter()
        if zp.exists():
            z = np.load(zp, allow_pickle=False); uids, Z = z["uids"].astype(str), z["Z"]
            assert set(uids) == {g.uid for g in games}, f"{zp}: uid set differs from this data; use another --z-cache-dir"
            print(f"  [{sp}] Z cache {zp} ({len(uids)} games)")
        else:
            store = P.l0_store(games, 4)
            Z = np.empty((len(store.uids), len(pm.s["mu"]) + len(pm.s["cols"])), np.float32)
            for a in range(0, len(Z), 20000):
                Z[a:a + 20000] = pm._z(store, np.arange(a, min(a + 20000, len(Z))))
            uids = np.array(store.uids).astype(str); np.savez(zp, uids=uids, Z=Z)
            print(f"  [{sp}] L0 features for {len(uids)} games: {time.perf_counter() - t0:.0f}s -> {zp}")
        rowof = {u: i for i, u in enumerate(uids)}
        rows = lambda us: np.fromiter((rowof[u] for u in us), np.int64, len(us))   # noqa: E731
        lib_z = [Z[rows([g.uid for g in cands[p]])] for p in pool]
        q_z = [Z[rows([g.uid for g in gs])] for _, gs in pq]
        f = prr.candidate_features(lib_z, q_z, pm.s["sel"], prr.K, prr.N_DENSE)
        e = pm.booster.predict(f["X"], num_threads=4).reshape(len(q_z), -1)
        top = f["top"]
        st = det_stats(top, e, len(pool))
        st["flag"] = bool(st[ch["stat"]] > ch["theta"])
        res[sp] = st
        print(f"  [{sp}] c_all {st['c_all']:.3f} p1_mean {st['p1_mean']:.3f} exc {st['exc']:.4f}  (theta {ch['theta']:.4f})  -> {'ABOVE theta: non-1:1 suspected -> C* (decoder D)' if st['flag'] else 'below theta: one-to-one plausible -> B (v3 transductive)'}   nq {st['nq']} nc {st['nc']}")
        print(f"        E30 reference: all one-to-one scenarios max exc .248 (S1, small N), S4b/S4c/S5 min .266 ; margin to theta {ch['theta'] - st['exc']:+.4f}")
        if abs(st["exc"] - ch["theta"]) < 0.03:
            flag(f"{sp}: exc {st['exc']:.4f} is within .03 of theta {ch['theta']:.4f} (E30 margin is ~.001): decide by hand, see runbook table")
    return res


def det_stats(top0, e0, nc, tau=0.9):
    """verbatim logic of scripts/experiments/e30/common.det_stats (checked equal in the dry run)."""
    sm = lambda x: (lambda z: z / z.sum(1, keepdims=True))(np.exp(x - x.max(1, keepdims=True)))   # noqa: E731
    nq = e0.shape[0]
    order = np.argsort(-e0, axis=1, kind="stable")
    best = np.take_along_axis(top0, order[:, :1], 1)[:, 0]
    p1 = sm(e0).max(1)
    cnt = np.bincount(best, minlength=nc)
    coll = cnt[best] > 1
    c_all = float(coll.mean())
    return dict(c_all=c_all, exc=c_all - (1.0 - float(p1.mean())), p1_mean=float(p1.mean()), nq=int(nq), nc=int(nc))


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ref-dir", default=str(ROOT / "data" / "contest_1104"), help="old data used for the overlap (default 11/04)")
    ap.add_argument("--exc", action="store_true", help="also compute the E30 exc statistic (CPU, minutes)")
    ap.add_argument("--z-cache-dir", default=str(OUT / "z"), help="where Z_<split>.npz lives / is written (dry run: .scratch/e27)")
    ap.add_argument("--no-overlap", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = Path(a.data_dir).resolve()
    print(f"P1111 inspect  data={data_dir}  ref={a.ref_dir}  {time.strftime('%F %T')}")
    try:
        found = discover(data_dir, "new")
        print("\n== files found (classified by header + file name)")
        for k, i in sorted(found.items(), key=lambda kv: str(kv[0])):
            print(f"  {i['path'].name:50s} {k[0]:9s} task={k[1]} split={k[2]}  header cols {len(i['header'])}")
            ren = {c: s for c, s in i["mapping"].items() if c != s}
            if ren:
                print(f"      accepted column aliases (printed so you can check them): {ren}")
                flag(f"{i['path'].name}: columns renamed vs the 11/04 format {dict(list(ren.items())[:4])}")
            if i["extra"]:
                print(f"      extra columns not used by the pipeline: {i['extra'][:10]}")
                flag(f"{i['path'].name}: extra columns {i['extra'][:10]} (a new link between questions and candidate files? read the announcement)")
        pl = {k: v for k, v in found.items() if k[1] == "player"}
        rk = [k for k in found if k[1] == "rank"]
        if rk:
            print(f"  (rank files present, ignored here: {[found[k]['path'].name for k in rk]})")
        splits_found = sorted({k[2] for k in pl})
        norm_dir = OUT / "normalized"; norm_dir.mkdir(exist_ok=True)
        splits, report = {}, {"data_dir": str(data_dir), "splits": {}}
        for sp in splits_found:
            iq, ic = pl.get(("question", "player", sp)), pl.get(("candidate", "player", sp))
            if iq is None or ic is None:
                stop(f"split {sp}: need both a player question file and a candidate file; found question={iq is not None} candidate={ic is not None}",
                     [i["path"] for i in (iq, ic) if i])
            qn, _ = load_norm(iq["path"], iq); cn, _ = load_norm(ic["path"], ic)
            splits[sp] = (qn, cn, iq, ic)
            qn.to_csv(norm_dir / f"player_identification_test_{'public' if sp == 'pub' else 'private'}.csv", index=False)
            cn.to_csv(norm_dir / f"player_identification_candidates_{'public' if sp == 'pub' else 'private'}.csv", index=False)
        if not splits:
            stop("no player question/candidate files found", [i["path"] for i in found.values()][:6])
        print(f"\ncanonical-named copies written to {norm_dir} (use as --data-dir downstream if any alias was applied)")
        new = {}
        for sp, (qn, cn, iq, ic) in splits.items():
            r, qg, cg = split_report(sp, qn, cn, iq, ic)
            report["splits"][sp] = r; new[sp] = (qg, cg)
        if not a.no_overlap:
            rf = discover(Path(a.ref_dir), "ref")
            ref = {}
            for sp in new:
                iq, ic = rf.get(("question", "player", sp)), rf.get(("candidate", "player", sp))
                if iq is None or ic is None:
                    print(f"  ref has no {sp} files, overlap skipped for this split"); continue
                qn, _ = load_norm(iq["path"], iq); cn, _ = load_norm(ic["path"], ic)
                ref[sp] = (explode_q(qn, sp), cand_games(cn, sp))
            report["overlap_plan"] = overlap_and_plan(new, ref, list(new)) if ref else None
        if a.exc:
            report["exc"] = exc_stage({sp: (v[0], v[1]) for sp, v in splits.items()}, Path(a.z_cache_dir))
    except Stop:
        return 3
    # verdict
    print("\n" + "=" * 100 + "\nVERDICT")
    white = any(r["white_questions"] or r["cand_white_games"] for r in report["splits"].values())
    nonbij = any(not r["nq_eq_nc"] for r in report["splits"].values())
    flagged = any(v.get("flag") for v in report.get("exc", {}).values())
    if white:
        print("  branch: WHITE present -> runbook section 7 (Q_NOT_BLACK blocks predict.py/p1111_decode load; the black-only metrics/model are out of distribution for W-vs-B)")
    elif nonbij or flagged:
        print("  branch: C* (non one-to-one suspected: nq != n_cand, or exc > theta) -> decoder D (p1111_decode.py --decoder D)")
    elif "exc" not in report:
        print("  branch: provisional v6/v3 (one-to-one indicator ok) but exc NOT computed: run with --exc before choosing B")
    else:
        print("  branch: B (v3 transductive); with SE/HSL ready -> v6 fusion")
    if FLAGS:
        print(f"  {len(FLAGS)} REVIEW flag(s):")
        for f in FLAGS:
            print("   -", f)
    (OUT / "inspect_summary.json").write_text(json.dumps({**report, "flags": FLAGS}, indent=1, default=str))
    print(f"  summary: {OUT / 'inspect_summary.json'}")
    return 2 if FLAGS else 0


if __name__ == "__main__":
    sys.exit(main())
