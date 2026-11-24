"""set_pose.py <result name> <pose> [--note TEXT] - exp1121 (2026-11-21, 3rd): posture label of a run that real_run.sh cannot take
(real_run.sh --pose accepts H|L|G|none only; P2 runs are started with --pose H = the same arm shape). Writes
goal1/real_prep4/<name>_pose.json {"pose": <pose>, "pose_cli": <pose in the name / exp.json>, "note": ..., "set_at": ...}; nothing
else is changed (exp.json, the name and real_run.sh stay as they are). exp_ana.py takes the pose from <name>_pose.json first, and a
replay (exp.json replay_of) from its recording's _pose.json. Run it right after each P2 real run (before its replays is not needed).
  python3 goal1/real_prep5/exp1121/set_pose.py real_c1k_effmass_H_t1_1121_190512 P2
(2026-11-22, night1122_s2) scenario labels S1 / S2 added (RUN_SHEET_next.md): S1 = right arm straight toward the path (fist
head-on), S2 = left arm bent 60 deg (fist toward +x and the path). Both are started with real_run.sh --pose H and labelled
right after the run:  python3 goal1/real_prep5/exp1121/set_pose.py <name> S1   (or S2). Other labels unchanged."""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("pose", choices=["P2", "H", "L", "G", "P", "none", "S1", "S2"]); ap.add_argument("--note", default="")
ap.add_argument("--dir", default=os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep4")); a = ap.parse_args()
n = os.path.basename(a.name)
for suf in ("_summary.json", "_exp.json", "_human.npz"):
    n = n[:-len(suf)] if n.endswith(suf) else n
ej = os.path.join(a.dir, n + "_exp.json")
if not os.path.exists(ej):
    sys.exit("no %s (a real_run.sh --human result?)" % ej)
E = json.load(open(ej)); out = os.path.join(a.dir, n + "_pose.json")
json.dump(dict(name=n, pose=a.pose, pose_cli=E.get("pose"), note=a.note, set_at=time.strftime("%Y-%m-%d %H:%M:%S")), open(out, "w"), indent=1)
print("%s: pose %s (real_run.sh pose %s) -> %s" % (n, a.pose, E.get("pose"), out))
