"""E99: remote-side driver for ONE big network (copy of scripts/experiments/e90/driver_e90h.py, which stays untouched).  Runs train_e99.py (next to this file) as an ordinary subprocess.

  python driver_e99.py --roster finh_h1 --root /content/e99 --epochs 12 [--export auto]
Loop: start train_e99.py (it resumes from ckpt_latest.pt by itself) -> watch it; if the process exits non-zero, or status.json has not moved for 30 minutes, kill + restart from the last
checkpoint (at most 3 restarts).  Success = rc 0 and run_<roster>/results.json (train_e99 writes it LAST, after final model and export).  Then ROOT/DONE_<roster>.json is written.
The driver never starts another roster, never decides anything, never touches anything outside ROOT.  Recipe fixed here: --blocks 16 --ch 224 --accum 8 --workers 8 (train_e99.py asserts it again).
Files: ROOT/run_<roster>/ (train.log, status.json, steps_loss.tsv, ckpt_latest.pt, final_model.pt, vec_*.npz, results.json), ROOT/driver_<roster>.log, ROOT/driver_state_<roster>.json, ROOT/DONE_<roster>.json or ROOT/FAILED_<roster>.
Data files expected in ROOT: uh_games.npz rosters_h.npz export_spec_h.json export_spec_h.npz (+ fastboard.py next to this file).  finh_h1 exports games of the 'u' file only (no t_games.npz needed).
"""
import argparse, json, os, shlex, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--roster", required=True); ap.add_argument("--root", default="/content/e99"); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--export", default="auto")
ap.add_argument("--extra", default="", help="TEST ONLY: extra train_e99.py arguments (shlex-split)")
ap.add_argument("--stall-min", type=float, default=30.0); ap.add_argument("--max-restarts", type=int, default=3)
a = ap.parse_args()
assert 8 <= a.epochs <= 20, "E99 epochs must be 8..20"
root = Path(a.root); run = root / f"run_{a.roster}"; run.mkdir(parents=True, exist_ok=True); here = Path(__file__).resolve().parent
logf = open(root / f"driver_{a.roster}.log", "a"); state_p = root / f"driver_state_{a.roster}.json"
def log(*x):
    s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(i) for i in x); logf.write(s + "\n"); logf.flush()
def state(**kw):
    st = dict(roster=a.roster, driver_pid=os.getpid(), updated=time.time(), restarts=restarts, epochs=a.epochs, **kw); state_p.write_text(json.dumps(st)); return st
restarts = 0
cmd = [sys.executable, str(here / "train_e99.py"), "--data", str(root / "uh_games.npz"), "--rosters", str(root / "rosters_h.npz"), "--export-spec", str(root / "export_spec_h.json"),
       "--out", str(run), "--roster", a.roster, "--blocks", "16", "--ch", "224", "--epochs", str(a.epochs), "--accum", "8", "--workers", str(a.workers), "--export", a.export] + shlex.split(a.extra)
while True:
    p = subprocess.Popen(cmd, stdout=open(run / "stdout.log", "a"), stderr=subprocess.STDOUT); log("started train_e99.py pid", p.pid, "attempt", restarts, " ".join(cmd))
    state(state="running", train_pid=p.pid); last_beat = time.time(); last_seen = None; reason = None
    while p.poll() is None:
        time.sleep(30)
        try: t = json.loads((run / "status.json").read_text())["time"]
        except Exception: t = None
        if t is not None and t != last_seen: last_seen, last_beat = t, time.time()
        state(state="running", train_pid=p.pid)
        if time.time() - last_beat > a.stall_min * 60:
            reason = f"no progress for {a.stall_min} min"; log("STALL:", reason, "-> kill"); p.kill(); p.wait(); break
    rc = p.returncode; log("train_e99.py ended rc", rc, reason or "")
    if rc == 0 and (run / "results.json").exists(): break
    restarts += 1
    if restarts > a.max_restarts:
        log("giving up after", a.max_restarts, "restarts"); state(state="failed"); (root / f"FAILED_{a.roster}").write_text(f"rc {rc} {reason}"); sys.exit(1)
    log("restart", restarts, "from last checkpoint"); time.sleep(10)
res = json.loads((run / "results.json").read_text())
done = dict(roster=a.roster, steps=res["steps"], expected_steps=res["expected_steps"], epochs=res["epochs"], blocks=res["blocks"], ch=res["ch"], n_train=res["n_train"], n_classes=res["n_classes"],
            probe_h_in_sample=res["probe_final"]["h_all"], exports=res["exports"], accum=res["accum"], restarts=restarts, seconds_total_train_script=res["seconds"], finished=time.time())
(root / f"DONE_{a.roster}.json").write_text(json.dumps(done, indent=1)); log("DONE", json.dumps(done)); state(state="done")
