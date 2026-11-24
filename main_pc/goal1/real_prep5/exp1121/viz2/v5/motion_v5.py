"""motion_v5.py - viz2 v5 (sync 2026-11-24): robot motion state per step of a run, for the comparison scene choice (window_v5 --uni).
From the record (exp_ana Run, read only; steps npz goal_idx): the MPC goes A <-> B (rp2_mpc_rt: the goal switches on the measured state on
arrival, 1 cm (rp2 default; --arrive-pos-tol) / 0.05 rad, or after LEG_MAX_STEPS 400 steps = 6.7 s; there is no dwell at a goal).
  speed   TCP speed, m/s (exp_ana tcp_of(meas_q), central difference over +-SPAN steps)
  goal    goal index the step drives to (steps npz goal_idx after the step's switch; the switch step belongs to the new leg)
  leg     pass number: 0 = start -> first goal, +1 at every goal switch; dirn = "A->B" / "B->A" (from the goal of the leg)
  leg_k0  first step of the leg; dgoal = TCP distance to the goal of the leg; dnear = distance to the nearest goal point
  arriving  final approach of the leg: dgoal <= ARRIVE_M (0.03 m; the robot slows down to the 1 cm switch, "arrival wait")
  switch  how the leg ended: "arrival" (pe <= the arrival tolerance + 5 mm at the switch step) | "timeout" | "open" (run end);
          (arrtol 11-24) tolerance = the record's arrive_pos_tol (_exp.json, else _summary.json; old records 0.01 m = rp2 default)
Library: state(run) -> dict of arrays (+ legs list). CLI: motion_v5.py <run> [<run>] --t a,b [--every S] -> table."""
import json, os, sys
import numpy as np
E1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121")
SPAN = 3            # steps: speed = |tcp[k + 3] - tcp[k - 3]| / dt
ARRIVE_M = 0.03     # m: final approach of a leg
LEG_MAX_STEPS = 400   # rp2_mpc_rt / step2_scene (read only): leg timeout
POS_TOL_M = 0.01   # rp2 default arrival tolerance (old records)


def arrive_tol(p):   # (arrtol 11-24) the record's arrival position tolerance, m
    for f in (p + "_exp.json", p + "_summary.json"):
        try:
            v = json.load(open(f)).get("arrive_pos_tol")
        except (OSError, ValueError):
            v = None
        if v is not None:
            return float(v)
    return POS_TOL_M


def state(run_or_r):
    sys.path.insert(0, E1)
    import exp_ana as X   # read only
    r = X.Run(run_or_r) if isinstance(run_or_r, str) else run_or_r
    z = np.load(r.p + "_steps.npz"); n = r.n; t = np.asarray(r.t, float); tcp = np.asarray(r.tcp, float)
    gi = np.asarray(z["goal_idx"], int)[:n]
    sc = json.load(open(r.p + "_scene.json")); G = np.array([(lambda T: T[:3, 3] + X.M.TOOLZ * T[:3, 2])(X.M.gT(g)) for g in sc["goals_wxyz"]])
    a, b = np.clip(np.arange(n) - SPAN, 0, n - 1), np.clip(np.arange(n) + SPAN, 0, n - 1)
    speed = np.linalg.norm(tcp[b] - tcp[a], axis=1) / np.maximum(t[b] - t[a], 1e-6)
    sw = np.r_[False, np.diff(gi) != 0]; leg = np.cumsum(sw); k0s = np.r_[0, np.nonzero(sw)[0]]
    leg_k0 = k0s[leg]; dgoal = np.linalg.norm(tcp - G[gi], axis=1); dnear = np.linalg.norm(tcp[:, None] - G[None], axis=2).min(1)
    nm = "AB" if len(G) == 2 else "".join(chr(65 + i) for i in range(len(G)))
    legs = []; tol = arrive_tol(r.p)
    for j, k0 in enumerate(k0s):
        k1 = (k0s[j + 1] - 1) if j + 1 < len(k0s) else n - 1; g = int(gi[k0]); gp = int(gi[k0 - 1]) if k0 > 0 else -1
        end = "open" if j + 1 == len(k0s) else ("arrival" if dgoal[k1] <= tol + 0.005 else "timeout")
        legs.append(dict(leg=j, k0=int(k0), k1=int(k1), t0=float(t[k0]), t1=float(t[k1]), goal=g, dirn=("start" if gp < 0 else nm[gp]) + "->" + nm[g], end=end))
    dirn = np.array([legs[l]["dirn"] for l in leg])
    return dict(arrive_pos_tol=tol, t=t, tcp=tcp, speed=speed, goal=gi, leg=leg, leg_k0=leg_k0, dirn=dirn, dgoal=dgoal, dnear=dnear, arriving=dgoal <= ARRIVE_M, legs=legs, goals=G,
                name=os.path.basename(r.p))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--t", default=""); ap.add_argument("--every", type=float, default=0.5)
    A = ap.parse_args(); S = [state(p) for p in A.runs]
    for s in S:
        print("%s legs: %s" % (s["name"], "; ".join("%d %s %.2f-%.2f %s" % (l["leg"], l["dirn"], l["t0"], l["t1"], l["end"]) for l in s["legs"])))
    if A.t:
        ta, tb = [float(x) for x in A.t.split(",")]
        for x in np.arange(ta, tb + 1e-9, A.every):
            row = []
            for s in S:
                k = int(np.argmin(np.abs(s["t"] - x)))
                row.append("v %.3f m/s  leg %d %s  dgoal %.3f  dnear %.3f%s" % (s["speed"][k], s["leg"][k], s["dirn"][k], s["dgoal"][k], s["dnear"][k], "  ARRIVING" if s["arriving"][k] else ""))
            print("%6.2f  " % x + "  |  ".join(row))
