"""make_scenes.py [make|check] - demo motion candidates (2026-11-20 night), CPU only (numpy, RobotInertiaNP FK), no MPC.
Scene files (ym format, base frame, goal = the frame the MPC tracks, the user's coordinates; TCP = goal - 0.1034 m along z):
  ym  real_prep5/scene_AB_ym_rp5floor.json (reference, not changed)
  C1  demo/scene_C1_rp5floor.json: A (+0.45, -0.55, 0.3235) <-> B (-0.45, -0.55, 0.3235)
  C2  demo/scene_C2_rp5floor.json: A up-right (+0.35, -0.55, 0.45) <-> B low-left (-0.35, -0.55, ZLOW)
  C3  demo/scene_C3_rp5floor.json (rp2 --goal-cycle 1): pick (+0.35, -0.55, ZLOW) -> lift (+0.35, -0.55, ZUP) -> carry (-0.35, -0.55, ZUP)
      -> place (-0.35, -0.55, ZLOW) -> up (-0.35, -0.55, ZUP) -> back (+0.35, -0.55, ZUP) -> pick ..; "via" = the 4 upper points.
  C1-C3 and the ym copy real_prep5/scene_AB_ym_rp5floor_table.json get the plate robot_table_real = the real robot table as a solid box
  (2026-11-20 night, measured by the user: 0.67 x 0.67 m, base at its centre, x and y -0.335 .. +0.335, top z 0, down to the floor z -0.715).
  Table size / position: --table X0,X1,Y0,Y1,ZTOP,ZBOTTOM (base frame, m; default -0.335,0.335,-0.335,0.335,0,-0.715), or edit the
  "robot_table_real" entry of "plates" in the scene files (pose = box centre + wxyz, dims = sizes). (First version: plate worktop_assumed
  under the path, removed.)  Usage: python3 make_scenes.py make [--table ...] | check
  All goals gripper down (orientation of ym). q_start = IK at goal 0 (damped least squares, seeds; nearest to the ym start posture).
check: along every leg (straight goal line, 41 points, IK continued from the previous point) joint limit margin, robot spheres
(link3 .. fingers) above the table plane z 0 (lowest sphere surface), TCP height, self pinch (closest surface gap between
hand / link6-7 spheres and link0-3 spheres), elbow (link4 origin) and TCP smallest y (toward the person), joint path length per leg.
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/ws/src/effmass_robot"))
from effmass_robot.robot_inertia_np import RobotInertiaNP
D = os.path.dirname(os.path.abspath(__file__)); P5 = os.path.dirname(D)
RI = RobotInertiaNP(rotor=True); RAD = np.asarray(RI.sphere_radius, float); SIDX = np.asarray(RI.s_idx)
YM = json.load(open(os.path.join(P5, "scene_AB_ym_rp5floor.json")))
LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]); HI = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
ZLOW, ZUP, Y = 0.15, 0.40, -0.55
QUAT = YM["goals_wxyz"][0][3:]
TB = [-0.335, 0.335, -0.335, 0.335, 0.0, -0.715]   # real robot table box x0 x1 y0 y1 ztop zbottom (base frame)
if "--table" in sys.argv:
    TB = [float(v) for v in sys.argv[sys.argv.index("--table") + 1].split(",")]; assert len(TB) == 6 and TB[0] < TB[1] and TB[2] < TB[3] and TB[5] < TB[4]


def table_plate(tb=None):
    x0, x1, y0, y1, zt, zb = tb or TB
    return dict(name="robot_table_real", pose=[(x0 + x1) / 2, (y0 + y1) / 2, (zt + zb) / 2, 1, 0, 0, 0], dims=[x1 - x0, y1 - y0, zt - zb])


def box_gap(P, r, tb=None):
    """surface gap sphere (centre P (n,3), radius r) - table box (negative = inside), and the closest box point (n,3)"""
    x0, x1, y0, y1, zt, zb = tb or TB; lo = np.array([x0, y0, zb]); hi = np.array([x1, y1, zt])
    c = np.clip(P, lo, hi); out = np.linalg.norm(P - c, axis=-1); ins = np.minimum(P - lo, hi - P).min(-1)
    return np.where(out > 0, out, -ins) - r, c


def qmat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def gT(g):
    T = np.eye(4); T[:3, :3] = qmat(*g[3:]); T[:3, 3] = g[:3]; return T


W0 = RI._fk(np.array(YM["q_start"])); TG = np.linalg.inv(W0[7]) @ gT(YM["goals_wxyz"][0])   # flange -> goal frame (constant)
TOOLZ = 0.1034


def goal_of(q):
    return RI._fk(q)[7] @ TG


def err(q, T):
    G = goal_of(q); e = T[:3, 3] - G[:3, 3]; Rr = T[:3, :3] @ G[:3, :3].T
    w = 0.5 * np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]); return np.r_[e, w]


def ik(T, q0, iters=300):
    q = np.array(q0, float)
    for _ in range(iters):
        e = err(q, T)
        if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
            return q, True
        J = np.zeros((6, 7)); h = 1e-6
        for j in range(7):
            dq = np.zeros(7); dq[j] = h; J[:, j] = -(err(q + dq, T) - e) / h
        lam = 0.02; dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), LO + 1e-3, HI - 1e-3)
    return q, bool(np.linalg.norm(err(q, T)[:3]) < 1e-4)


def spheres(q):
    W = RI._fk(q); Ws = W[SIDX]; return np.einsum("sij,sj->si", Ws[:, :3, :3], RI.s_local) + Ws[:, :3, 3], W


SCN = dict(
    C1=dict(goals=[[0.45, Y, 0.3235], [-0.45, Y, 0.3235]], via=[]),
    C2=dict(goals=[[0.35, Y, 0.45], [-0.35, Y, ZLOW]], via=[]),
    C3=dict(goals=[[0.35, Y, ZLOW], [0.35, Y, ZUP], [-0.35, Y, ZUP], [-0.35, Y, ZLOW], [-0.35, Y, ZUP], [0.35, Y, ZUP]], via=[1, 2, 4, 5]))


def start_q(T):
    best = None
    for k in range(12):
        seed = np.array(YM["q_start"]) + (0 if k == 0 else np.random.default_rng(k).normal(0, 0.4, 7))
        q, ok = ik(T, seed)
        if ok:
            c = np.linalg.norm(q - np.array(YM["q_start"]))
            if best is None or c < best[0]:
                best = (c, q)
    return best[1]


def make():
    for n, d in SCN.items():
        G = [list(map(float, p)) + list(QUAT) for p in d["goals"]]; q0 = start_q(gT(G[0]))
        js = dict(YM); js.update(key=f"demo_{n}_rp5floor", goals_wxyz=G, q_start=[round(float(x), 6) for x in q0],
                                tcp_A=[G[0][0], G[0][1], G[0][2] - TOOLZ], tcp_B=[G[1][0], G[1][1], G[1][2] - TOOLZ], via=d["via"],
                                note=f"demo candidate {n} (2026-11-20 night, real_prep5/demo/make_scenes.py): goals in the base frame, gripper down "
                                     f"(orientation of scene_AB_ym_rp5floor.json), q_start = IK at goal 0. 'via' = pass-through goals for rp2 "
                                     f"--goal-cycle 1. Plates: floor as ym + robot_table_real (the real robot table as a solid box, see make_scenes.py --table). dy, end_gap, end_margin, line_min_margin copied from ym (not of this path).")
        js["plates"] = list(YM["plates"]) + [table_plate()]
        for k in ("rotated_from", "rotation"):
            js.pop(k, None)
        json.dump(js, open(os.path.join(D, f"scene_{n}_rp5floor.json"), "w"), indent=1); print(n, "q_start", np.round(q0, 3))
    js = dict(YM); js["plates"] = list(YM["plates"]) + [table_plate()]; js["key"] = YM["key"] + "_table"   # ym copy with the real table
    js["note"] = YM["note"] + " | 2026-11-20 night (real_prep5/demo/make_scenes.py make): copy of scene_AB_ym_rp5floor.json with the plate robot_table_real (real robot table as a solid box); nothing else changed."
    json.dump(js, open(os.path.join(P5, "scene_AB_ym_rp5floor_table.json"), "w"), indent=1); print("ym copy with the table written")


def legs(js):
    G = js["goals_wxyz"]; n = len(G); cyc = bool(js.get("via")) or n > 2
    return [(i, (i + 1) % n) for i in range(n)] if cyc else [(0, 1), (1, 0)]


def check():
    hand = np.isin(SIDX, [6, 7]); low = np.isin(SIDX, [0, 1, 2, 3]); arm = SIDX >= 3
    out = {}
    for n in ["ym", "C1", "C2", "C3"]:
        js = YM if n == "ym" else json.load(open(os.path.join(D, f"scene_{n}_rp5floor.json")))
        G = js["goals_wxyz"]; q = np.array(js["q_start"]); r = dict(jm=9.0, jm_j=-1, zs=9.0, tcpz=9.0, pinch=9.0, elby=9.0, tcpy=9.0, ok=True, dq=[], reach=[])
        for a, b in legs(js) * 2:   # twice round: the second lap starts from the first lap's end posture
            Pa, Pb = np.array(G[a][:3]), np.array(G[b][:3]); L = 0.0
            for s in np.linspace(0, 1, 41):
                T = gT(list(Pa + s * (Pb - Pa)) + list(QUAT)); qn, ok = ik(T, q); r["ok"] &= ok; L += np.abs(qn - q).sum() if s > 0 else 0.0; q = qn
                m = np.minimum(q - LO, HI - q); j = int(m.argmin())
                if m[j] < r["jm"]:
                    r["jm"], r["jm_j"] = float(m[j]), j + 1
                P, W = spheres(q); r["zs"] = min(r["zs"], float((P[arm, 2] - RAD[arm]).min()))
                tcp = (W[7] @ TG)[:3, 3] + TOOLZ * (W[7] @ TG)[:3, 2]; r["tcpz"] = min(r["tcpz"], float(tcp[2])); r["tcpy"] = min(r["tcpy"], float(tcp[1]))
                dd = np.linalg.norm(P[hand][:, None] - P[low][None], axis=2) - RAD[hand][:, None] - RAD[low][None]; r["pinch"] = min(r["pinch"], float(dd.min()))
                r["elby"] = min(r["elby"], float(W[3][1, 3]))
            r["dq"].append(round(float(L), 3)); r["reach"].append(round(float(np.linalg.norm(Pb - Pa)), 3))
        r["dq"] = r["dq"][:len(r["dq"]) // 2]; r["reach"] = r["reach"][:len(r["reach"]) // 2]
        out[n] = r
        print(f"{n}: IK ok {r['ok']}, joint margin min {r['jm']:.3f} rad (joint {r['jm_j']}), lowest sphere surface z {r['zs']:.3f}, TCP z min {r['tcpz']:.3f}, "
              f"pinch hand-link0..3 {r['pinch']:.3f}, elbow y min {r['elby']:.3f}, TCP y min {r['tcpy']:.3f}, leg length m {r['reach']}, joint path rad {r['dq']}")
    json.dump(out, open(os.path.join(D, "check_scenes.json"), "w"), indent=1)


if __name__ == "__main__":
    (make if (sys.argv[1:] or ["check"])[0] == "make" else check)()
