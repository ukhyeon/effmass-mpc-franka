"""exp_ana.py - exp1121 (2026-11-21): uniform vs effective-mass avoidance, real runs, real-robot replays and synthetic runs. CPU only, reads only.

  exp_ana.py one  <run> [--ref <run>]           posture windows of one run (table per robot pass)
  exp_ana.py pair <run A> <run B>               same human input, two methods: windows / passes matched, difference effmass - uniform
  exp_ana.py legs <run> ...                     path timing (TCP passes x 0.30 toward A, leaves A) to set the single-reach cue time
  exp_ana.py rel  <real run> <replay run>       replay reliability: the replay of the recording method against the real person run
  exp_ana.py agg  <run> <run> ... [--fig DIR]   pose x method x approach median [min, max]; pairs (same recording / same synthetic
                                                input); figures fig_pairs.png (per window: uniform -> effmass) and fig_pose.png

<run> = a result name or path prefix (<prefix>_summary.json, _steps.npz, _human.npz). Names are looked up in goal1/real_prep4,
real_prep5/exp1121/runs, real_prep5/demo/runs. Shell globs are fine for agg (e.g. real_prep4/real_c1k_*_summary.json).

Data. Robot: measured joints per step (steps npz meas_q), robot spheres of the MPC model (effmass_robot RobotInertiaNP, 61 spheres,
MPC radii). Human: the records the MPC used (_human.npz, rp2_mpc_rt.py --rec-human): per step the last record picked at or before the
step; a record the MPC's jump filter rejected is replaced by the last accepted one (as the MPC did); node world -> base with the
recorded frame (R, t). For a real run the "gap" is the gap to the measured human (camera + human node), not to the true body.
Gap = min over robot spheres and human capsules of the surface distance (centre distance - capsule radius - (sphere radius + 0.02 m)),
the definition of the MPC near check (summary min gap) and of the effmass-mpc ladder; checked against the MPC's per-step near_min.

Posture window: the arm capsules (upper arm, lower arm, hand; both sides) come within WIN_R (0.25 m, surface) of the robot path
(TCP polyline through the scene goals); pieces closer than 1.0 s are joined, pieces shorter than 0.5 s dropped, +1.0 s added at
the end. Pass: inside a window, a stretch where the TCP moves the same way along the path (|v| > 0.02 m/s, at least 0.3 s).
Per pass:
  gmin           smallest gap (m) and the human part at that step
  from           the side the robot comes from (+x / -x of the base); for a bent arm (lower arm more along the path than across,
                 |dx| > 0.7 |d|) "fist" if the robot comes from the side the hand points to, else "elbow"; straight arm: "side"
  angle          angle between u (robot sphere -> closest human point) and the lower-arm axis at gmin, folded to 0..90 deg
                 (< 45 = head-on, as the effmass-mpc ladder)
  m_h, m_r, mu   effective masses (kg) in u at gmin: m_h = 1/(u' inv_h u) of that segment, m_r = 1/(u' inv_r u) of that sphere,
                 mu = 1/(u'(inv_h + inv_r)u); eta_e = clamp(d_base + 0.10 max(sqrt(mu (1/m_iso + 1/m_r)) - 1, 0), d_base, 0.25)
                 (effective-mass clearance), eta_u = d_base (uniform clearance); d_base from the run summary (cost eta0)
  start          avoidance start gap: the gap when the TCP first goes more than 3 cm off the reference path for 0.2 s (ladder rule)  (MAIN metric, per side the robot comes from)
  ax_gap20       (3rd) the same with 20 deg (heavy cone: lower-arm mass half at about 17-22 deg, hand at about 8-15 deg)
  ax_gap         (2nd) near-axis gap: smallest gap between robot spheres and the lower arm / hand capsules of the arm in the path, over the
                 sphere-capsule pairs of the pass whose u lies within AX_DEG (30 deg) of the lower-arm axis (robot in front of the fist /
                 behind the elbow); "-" if the robot never was there. The overall smallest gap gmin is a secondary metric.
Pose not confirmed (4th): a real / replay run with _H_ in its name (or pose H) and no _pose.json (own or its recording's) is shown
as pose "H(unconfirmed)", with a warning at the top of agg and in its run line.
agg (3rd): MAIN table = the first reach of each run only (window 1: both runs still on the same path); SECONDARY = later reaches
with the phase flag. Pose label: <name>_pose.json (set_pose.py) first, a replay takes its recording's _pose.json.
Pairs (pair / agg): matched by window and direction; TCP x at the window start of both runs; a pair is flagged DIFFERENT PHASE if
they differ by more than PHASE_DX (0.15 m: about 0.3 s of travel at 0.5 m/s, half a lower arm) or the robot moves the other way.
Reflex stop = control PC EXIT= value 8 from <name>_ctrl_exit.json (real_run.sh, every run) or <name>_exp.json.
  dev            TCP off the reference path in the pass, mean / max (cm)
Reference path = the TCP path of a run of the same method without reaching (--ref, or in agg the newest run of the same kind and
method with pose "none" (real / replay; if there is none: same method of the other kind, then the other method of the same kind) or
the "_still" run of the same synthetic pose). Without a reference: start / dev "-".
Aborted runs (agg): control PC exit 9 (home move failed with real_run.sh --home-in-bridge; <name>_exp.json aborted) or no MPC
summary -> listed as ABORTED and left out of all tables; reflex stops (exit 8) stay in and are listed as REFLEX STOP.
Globs may match _summary.json or _exp.json files (aborted runs only have the latter).
Per run: arrivals, late steps (compute over), deadline misses, infeasible steps, bridge limit line of the control PC analysis
(<name>_ctrl_ana.txt), control PC exit code / reflex stop (<name>_exp.json, exit 8), MPC stop.
(R, 2026-11-22 night1122_s2) scenario labels S1 / S2 (set_pose.py <name> S1|S2; names with _S1_ / _sS1_ also count): BUNDLE S1 / S2
tables and, when any S1 / S2 first-reach pair is present, fig_groups shows the bundles that have data (else P2 / L as before).
--win-r M (one / pair / rel / agg; default 0.25 = WIN_R, unchanged): posture window radius. The S2 fist sits 0.29 m (surface) from the
TCP path line in the ARM_CHOICE simulation, so with 0.25 an S2 run has no window; use --win-r 0.35 for S1 / S2 (both).
All other labels and outputs are unchanged.
(11-23, arm avoidance demo) Recorded clearance parameters and arm-only analysis. From <name>_exp.json (old runs without an entry: the
value in brackets): "human_k_gain" (0.10), "human_eta_max" (0.25), "human_other_d_base" (0.10), "human_other_k_gain" (0.10),
"human_arm_parts" (both upper arms, lower arms, hands). Missing in exp.json: the same names in the summary human.part_cost (the MPC
log of the run); a run without _exp.json and part_cost (simulation) takes k_gain / eta_max of an effective-mass run from its summary
cost (p / eta_max); else the bracket values. eta_e uses k_gain and eta_max; a part not in human_arm_parts uses
other_d_base and other_k_gain (same eta_max). The gap (gmin, part, t, m_h / m_r / mu / eta_e, angle, avoid start), the posture window,
the side the robot meets and the near-axis gap use only the human_arm_parts capsules. The check against the MPC near_min (run line)
stays over every part (the MPC near check). The run line shows the recorded values only when one differs from the bracket values.
"""
import argparse, glob, json, os, re, sys
import numpy as np
G1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1"); P5 = os.path.join(G1, "real_prep5")
sys.path.insert(0, os.path.join(P5, "demo"))
import make_scenes as M   # noqa: E402  (RobotInertiaNP spheres, goal frames)
DIRS = [os.path.join(G1, "real_prep4"), os.path.join(P5, "exp1121", "runs"), os.path.join(P5, "demo", "runs")]
SEGN = ["pelvis", "left_thigh", "right_thigh", "abdomen", "left_calf", "right_calf", "left_foot", "right_foot", "chest", "neck", "left_shoulder",
        "right_shoulder", "head", "left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"]
ARMSEG = (13, 14, 15, 16, 17, 18); LOWER = {15: 17, 16: 18}   # lower arm -> hand of the same side
LIMIT_KO = "\ud55c\uacc4 \uc704\ubc18"   # "limit violations" line of the control PC analysis (Korean, escaped)
JUDGE_KO = "\ud310\uc815"   # "judgement" line
AX_DEG = 30.0   # (2nd) near-axis gap: u within this angle of the lower-arm axis
PHASE_DX = 0.15   # (2nd) pairs whose robot TCP x at the window start differs by more than this (or that move the other way) are flagged
RBUF = 0.02   # robot sphere = MPC radius + 0.02 m (as the MPC near check / the effmass-mpc ladder)
WIN_R, JOIN_S, MIN_S, TAIL_S, DEV_M, DEV_S, K_GAIN, ETA_MAX = 0.25, 1.0, 0.5, 1.0, 0.03, 0.2, 0.10, 0.25
OTHER_D, OTHER_K = 0.10, 0.10   # (11-23) defaults for runs that did not record human_other_d_base / human_other_k_gain
ARM_PARTS = ["left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"]   # (11-23) default human_arm_parts
ISO = None


def iso_mass():
    global ISO
    if ISO is None:
        j = json.load(open(os.path.join(G1, "data", "iso_15066.json")))
        sys.path.insert(0, os.path.join(G1, "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("g1m_names", os.path.join(G1, "scripts", "g1_meff_gpu.py"))
        src = open(spec.origin).read()   # only the name tables (no torch / warp import)
        ns = {}
        for nm in ("ISO_SEGMENTS", "ISO_ASSUMED"):
            m = re.search(r"^%s\s*=\s*(.+?)\n(?=\S)" % nm, src, re.M | re.S); exec(nm + " = " + m.group(1), ns)
        ISO = np.array([float(j[ns["ISO_ASSUMED"].get(n, n)]["m_h_iso"]) for n in ns["ISO_SEGMENTS"]])
    return ISO


def find(run):
    run = run[:-13] if run.endswith("_summary.json") else run
    if os.path.exists(run + "_summary.json"):
        return os.path.abspath(run)
    for d in DIRS:
        if os.path.exists(os.path.join(d, run + "_summary.json")):
            return os.path.join(d, run)
    raise SystemExit("no run %s (looked in %s)" % (run, ", ".join(DIRS)))


def seg_dist(P, a, b):   # points P (n,3) to segment a-b: distance, parameter
    d = b - a; L2 = max(float(d @ d), 1e-12); u = np.clip(((P - a) @ d) / L2, 0, 1); return np.linalg.norm(P - (a + u[:, None] * d), axis=1)


def poly_dist(P, Q):   # points P (n,3) to polyline Q (m,3)
    if len(Q) == 1:
        return np.linalg.norm(P - Q[0], axis=1)
    return np.min([seg_dist(P, Q[i], Q[i + 1]) for i in range(len(Q) - 1)], axis=0)


def tcp_of(q):
    W7 = np.array([M.RI._fk(x)[7].copy() for x in q]); Gm = np.einsum("tij,jk->tik", W7, M.TG); return Gm[:, :3, 3] + M.TOOLZ * Gm[:, :3, 2]


class Run:
    def __init__(self, run):
        self.p = find(run); self.name = os.path.basename(self.p)
        self.S = json.load(open(self.p + "_summary.json")); R = np.load(self.p + "_steps.npz")
        self.q = R["meas_q"].astype(float); self.t = (R["t_pub_ns"] - R["t_pub_ns"][0]) / 1e9; self.t0 = R["t_pub_ns"][0] / 1e9; self.n = len(self.q)
        self.E = json.load(open(self.p + "_exp.json")) if os.path.exists(self.p + "_exp.json") else {}
        self.C = json.load(open(self.p + "_ctrl_exit.json")) if os.path.exists(self.p + "_ctrl_exit.json") else {}   # (2nd) control PC EXIT= value
        self.ctrl_exit = self.C.get("control_exit", self.E.get("control_exit")); self.reflex = str(self.ctrl_exit) == "8"
        h = self.S.get("human") or {}; c = h.get("cost") or {}
        self.d_base = float(c.get("eta0", 0.065)); self.k = float(c.get("p", K_GAIN))
        self.method = h.get("method") or self.E.get("method") or ("uniform" if self.k == 0 else "effmass")
        # (11-23) recorded clearance parameters (exp.json; simulation runs: summary cost of an effective-mass run; else the defaults)
        pc = dict(h.get("part_cost") or {}); pc.update({k_: v_ for k_, v_ in self.E.items() if k_.startswith("human_")})   # exp.json first, then summary part_cost
        ef = self.method == "effmass" and not self.E
        self.k_gain = float(pc.get("human_k_gain", c.get("p") if ef and c.get("p") else K_GAIN))
        self.eta_max = float(pc.get("human_eta_max", c.get("eta_max") if ef and c.get("eta_max") else ETA_MAX))
        self.other_d = float(pc.get("human_other_d_base", OTHER_D)); self.other_k = float(pc.get("human_other_k_gain", OTHER_K))
        self.arm_parts = list(pc.get("human_arm_parts") or ARM_PARTS); self.arm_ids = tuple(SEGN.index(x) for x in self.arm_parts if x in SEGN)
        self.lower = [x for x in (15, 16) if x in self.arm_ids] or [15, 16]   # lower arms used for the side the robot meets
        io = pc.get("human_iso_override") or {}   # (shoulder 1124) ISO reference masses used by the run (exp.json dict / summary JSON text)
        io = json.loads(io) if isinstance(io, str) else dict(io); self.iso = iso_mass().copy()
        for k_, v_ in io.items():
            self.iso[SEGN.index(k_)] = float(v_)
        self.iso_override = io
        self.mass_mode = str(pc.get("human_mass_mode") or "reduced")   # (massmode 1124) eta reference recorded by the run (older runs: reduced)
        m = re.search(r"_s?(S1|S2|Hn|H|L|F|G|P2|P|none)_", "_" + self.name + "_")   # (R 11-22) S1 / S2   # (4th) also sP2_* (short runs)
        self.pose = self.E.get("pose") or (m.group(1) if m else "none")
        self.pose_file = None
        for pj in (self.p + "_pose.json", (self.E.get("replay_of") or "").replace("_human.npz", "_pose.json")):   # (3rd) set_pose.py label: own, else the recording's
            if pj and os.path.exists(pj):
                self.pose = json.load(open(pj)).get("pose", self.pose); self.pose_file = pj; break
        self.kind = self.E.get("human_source") or ("replay" if "_replay_" in self.name else "synth" if os.path.exists(self.p + "_synth.npz") else "live")
        # (4th) real / replay run started with --pose H (the P2 way) but no set_pose.py label: pose not confirmed
        self.pose_unconfirmed = self.kind in ("live", "replay") and self.pose_file is None and ("_H_" in self.name or self.pose == "H")
        if self.pose_unconfirmed:
            self.pose = "H(unconfirmed)"
        self.still = self.name.endswith("_still")
        sc = json.load(open(self.p + "_scene.json")) if os.path.exists(self.p + "_scene.json") else None
        if sc is None and os.path.exists(self.p + "_human.npz"):
            z = np.load(self.p + "_human.npz")
            sc = json.loads(str(z["scene_text"])) if "scene_text" in z.files and str(z["scene_text"]) else None
        self.scene = sc
        self.path = np.array([(lambda T: T[:3, 3] + M.TOOLZ * T[:3, 2])(M.gT(g)) for g in sc["goals_wxyz"]]) if sc else None
        if sc and len(self.path) == 2:
            pass
        elif sc and len(self.path) > 2:
            self.path = np.vstack([self.path, self.path[:1]])
        self._human()
        self.tcp = tcp_of(self.q)
        self.P = np.array([M.spheres(x)[0] for x in self.q])

    def _human(self):
        self.H = None
        f = self.p + "_human.npz"
        if not os.path.exists(f):
            return
        z = np.load(f); Rf, tf = z["frame_R"], z["frame_t"]
        ms = z["msg_step"].astype(int); rej = z["msg_jf_rejected"].astype(bool) if "msg_jf_rejected" in z.files else np.zeros(len(ms), bool)
        acc = np.arange(len(ms)); last = -1
        for i in range(len(ms)):   # rejected -> last accepted (as the MPC)
            if rej[i] and last >= 0:
                acc[i] = last
            elif not rej[i]:
                last = i
        tr = lambda x: x @ Rf.T + tf
        self.Hp0, self.Hp1 = tr(z["msg_p0"].astype(float)), tr(z["msg_p1"].astype(float)); self.Hrad = z["msg_rad"].astype(float)
        self.Hseg = z["msg_seg"].astype(int); self.Hcom = tr(z["msg_com"].astype(float))
        self.Hinv = np.einsum("ij,nsjk,lk->nsil", Rf, z["msg_inv"].astype(float), Rf)
        k = np.searchsorted(ms, np.arange(self.n), side="right") - 1   # last record picked at or before the step
        self.hidx = np.where(k >= 0, acc[np.maximum(k, 0)], -1); self.H = z
        self.rec_near = z["step_near_min"].astype(float) if "step_near_min" in z.files else None

    def gaps(self):   # per step: gap, sphere, shape, closest human point
        if hasattr(self, "_g"):
            return self._g
        n = self.n; g = np.full(n, np.nan); sp = np.full(n, -1); sh = np.full(n, -1); cp = np.full((n, 3), np.nan)
        arm = np.full(n, np.nan); gall = np.full(n, np.nan)
        for i in range(n):
            h = self.hidx[i] if self.H is not None else -1
            if h < 0:
                continue
            a, b, r = self.Hp0[h], self.Hp1[h], self.Hrad[h]; ok = (r > 0) & (np.abs(a).max(1) < 20)
            best = (np.inf, -1, -1, None); ga = np.inf; isarm = np.isin(self.Hseg[h], self.arm_ids)
            for kk in np.nonzero(ok)[0]:
                d = b[kk] - a[kk]; L2 = max(float(d @ d), 1e-12); u = np.clip(((self.P[i] - a[kk]) @ d) / L2, 0, 1); qp = a[kk] + u[:, None] * d
                dd = np.linalg.norm(self.P[i] - qp, axis=1) - r[kk] - M.RAD - RBUF; j = int(dd.argmin()); ga = min(ga, float(dd[j]))
                if isarm[kk] and dd[j] < best[0]:   # (11-23) arm parts only
                    best = (float(dd[j]), j, kk, qp[j])
            gall[i] = ga
            if best[2] >= 0:
                g[i], sp[i], sh[i], cp[i] = best[0], best[1], best[2], best[3]
            if self.path is not None:
                am = [float(poly_dist(np.linspace(a[kk], b[kk], 5), self.path).min()) - r[kk] for kk in np.nonzero(ok & isarm)[0]]
                arm[i] = min(am) if am else np.nan
        self._gall = gall; self._g = (g, sp, sh, cp, arm); return self._g

    def windows(self):
        g, _, _, _, arm = self.gaps(); inn = np.nan_to_num(arm, nan=9) < WIN_R; dt = np.median(np.diff(self.t)) if self.n > 1 else 1 / 60
        idx = np.nonzero(inn)[0]
        if not len(idx):
            return []
        pieces = [[idx[0], idx[0]]]
        for i in idx[1:]:
            if self.t[i] - self.t[pieces[-1][1]] <= JOIN_S:
                pieces[-1][1] = i
            else:
                pieces.append([i, i])
        out = []
        for a, b in pieces:
            if self.t[b] - self.t[a] < MIN_S:
                continue
            b2 = int(min(self.n - 1, np.searchsorted(self.t, self.t[b] + TAIL_S)))
            out.append((a, b2))
        return out

    def passes(self, w):
        a, b = w; d = self.path[-1] - self.path[0] if self.path is not None else np.array([1.0, 0, 0]); d = d / np.linalg.norm(d)
        v = np.gradient(self.tcp[:, :], self.t, axis=0) @ d; sg = np.where(np.abs(v) > 0.02, np.sign(v), 0)
        out = []; i = a
        while i <= b:
            if sg[i] == 0:
                i += 1; continue
            j = i
            while j + 1 <= b and sg[j + 1] in (sg[i], 0):
                j += 1
            if self.t[j] - self.t[i] >= 0.3:
                out.append((i, j, int(sg[i])))
            i = j + 1
        return out, d

    def pass_row(self, ps, d, ref=None):
        i0, i1, sgn = ps; g, sp, sh, cp, _ = self.gaps(); seg = np.arange(i0, i1 + 1); ok = seg[np.isfinite(g[seg])]
        if not len(ok):
            return None
        k = int(ok[np.argmin(g[ok])]); h = self.hidx[k]; s_ = int(sp[k]); kk = int(sh[k]); sgid = int(self.Hseg[h][kk])
        c = self.P[k, s_]; u = cp[k] - c; u = u / (np.linalg.norm(u) + 1e-9)
        ih = self.Hinv[h, sgid]; _, inv_r, _ = M.RI.compute(self.q[k]); ir = inv_r[s_]
        ah, ar = float(u @ ih @ u), float(u @ ir @ u); mu = 1.0 / (ah + ar)
        eta_e = self.eta_eff(sgid, mu, ar)
        # lower arm nearest to the path in the pass: bent (along the path) or straight
        fore = [x for x in self.lower if x in LOWER]; best = None
        for la in fore:
            e = self.Hcom[h, LOWER[la]] - self.Hcom[h, la]; dist = poly_dist(self.Hcom[h, [LOWER[la]]], self.path)[0] if self.path is not None else 0
            if best is None or dist < best[0]:
                best = (dist, la, e)
        _, la, e = best; along = float(e @ d); frm = "+x" if -sgn * d[0] > 0 else "-x"   # moving -x along d means coming from +x
        if abs(along) > 0.7 * np.linalg.norm(e):
            cls = "fist" if np.sign(along) == -sgn else "elbow"   # hand points against the robot's motion = toward it
        else:
            cls = "side"
        # angle u - lower arm axis (of the arm hit, else the nearest lower arm)
        lam = sgid if sgid in (15, 16) else (15 if sgid in (13, 17) else 16 if sgid in (14, 18) else la)
        ax = self.Hcom[h, LOWER[lam]] - self.Hcom[h, lam]; ang = float(np.degrees(np.arccos(min(1.0, abs(u @ ax) / (np.linalg.norm(ax) + 1e-9)))))
        row = dict(t0=float(self.t[i0]), t1=float(self.t[i1]), gmin=float(g[k]), t_min=float(self.t[k]), part=SEGN[sgid], link=str(np.asarray(M.RI.sphere_link)[s_]),
                   frm=frm, cls=cls, angle=ang, m_h=1.0 / ah, m_r=1.0 / ar if ar > 0 else np.inf, mu=mu, eta_e=eta_e, eta_u=self.d_base,
                   start=np.nan, dev_mean=np.nan, dev_max=np.nan, ax_gap=self.axis_gap(seg, la), ax_gap20=self.axis_gap(seg, la, 20.0))
        if ref is not None:
            dev = poly_dist(self.tcp[seg], ref.tcp); off = dev > DEV_M; need = max(1, int(round(DEV_S / np.median(np.diff(self.t)))))
            for j in range(len(seg) - need + 1):
                if off[j:j + need].all():
                    row["start"] = float(g[seg[j]]); break
            row.update(dev_mean=float(dev.mean() * 100), dev_max=float(dev.max() * 100))
        return row

    def eta_eff(self, sgid, mu, ar):   # (11-23) effective-mass clearance of part sgid with the recorded gain / cap (other parts: own base / gain)
        d, k = (self.d_base, self.k_gain) if sgid in self.arm_ids else (self.other_d, self.other_k)
        if self.mass_mode == "human":   # (massmode 1124) reference = the human effective mass m_h = 1 / (1/mu - u' inv_r u), as kernel mode gain_part_h
            mh = 1.0 / max(1.0 / mu - ar, 1e-9)
            return float(np.clip(d + k * max(np.sqrt(mh / self.iso[sgid]) - 1.0, 0.0), d, max(self.eta_max, d)))
        return float(np.clip(d + k * max(np.sqrt(mu * (1.0 / self.iso[sgid] + ar)) - 1.0, 0.0), d, max(self.eta_max, d)))   # (shoulder 1124) self.iso: recorded override

    def axis_gap(self, steps, la, deg=None):   # (2nd) smallest gap robot sphere - lower arm / hand of arm la over the pairs whose u is within AX_DEG of the lower-arm axis
        best = np.nan; hand = LOWER[la]; cmax = np.cos(np.radians(AX_DEG if deg is None else deg))
        for j in steps:
            h = self.hidx[j]
            if h < 0:
                continue
            seg_h = self.Hseg[h]; lo = np.nonzero(seg_h == la)[0]
            if not len(lo):
                continue
            ax = self.Hp1[h, lo[0]] - self.Hp0[h, lo[0]]; ax = ax / (np.linalg.norm(ax) + 1e-12)
            for kk in np.nonzero(np.isin(seg_h, [x for x in (la, hand) if x in self.arm_ids]) & (self.Hrad[h] > 0))[0]:
                a, b, r = self.Hp0[h, kk], self.Hp1[h, kk], self.Hrad[h, kk]; dv = b - a; L2 = max(float(dv @ dv), 1e-12)
                uu = np.clip(((self.P[j] - a) @ dv) / L2, 0, 1); qp = a + uu[:, None] * dv; v = qp - self.P[j]; nv = np.linalg.norm(v, axis=1)
                ok = np.abs(v @ ax) >= cmax * np.maximum(nv, 1e-12)
                if ok.any():
                    gg = (nv - r - M.RAD - RBUF)[ok].min()
                    best = gg if not np.isfinite(best) else min(best, gg)
        return float(best)

    def rows(self, ref=None):
        out = []
        for wi, w in enumerate(self.windows()):
            ps, d = self.passes(w)
            for pi, p in enumerate(ps):
                r = self.pass_row(p, d, ref)
                if r is not None:
                    i0 = w[0]; vx = np.gradient(self.tcp[:, 0], self.t)[i0]
                    r.update(win=wi + 1, pas=pi + 1, w0=float(self.t[w[0]]), w1=float(self.t[w[1]]), x0=float(self.tcp[i0, 0]), dir0=int(np.sign(vx)) if abs(vx) > 0.02 else 0)
                    out.append(r)
        return out

    def runinfo(self):
        S = self.S; ana = ""
        if os.path.exists(self.p + "_ctrl_ana.txt"):
            L = [l.strip() for l in open(self.p + "_ctrl_ana.txt", errors="ignore") if l.strip()]
            ana = "; ".join(l for l in L if l.startswith((LIMIT_KO,) if self.E.get("mode") == "fake" else (LIMIT_KO, JUDGE_KO))) or "-"   # fake: its judgement is real_run.sh's own
        ce = self.ctrl_exit; st = S.get("stop")
        chk = ""
        if self.rec_near is not None:
            self.gaps(); g = self._gall; m = np.isfinite(g) & np.isfinite(self.rec_near)   # (11-23) every part, as the MPC near check
            chk = " (check: recomputed gap - MPC near_min max |diff| %.4f m)" % float(np.abs(g[m] - self.rec_near[m]).max()) if m.any() else ""
        return (("[POSE NOT CONFIRMED] " if self.pose_unconfirmed else "") + "%s: kind %s, method %s (d_base %.3f, k_gain %.2f), pose %s; arrivals %s, late %s, deadline miss %s, infeasible %s, MPC stop %s; "
                "bridge limits %s; control exit %s%s%s" % (self.name, self.kind, self.method, self.d_base, self.k, self.pose, S.get("arrivals"), S.get("overrun_steps"),
                S.get("deadline_miss"), S.get("infeasible"), st.get("reason") if st else "none", ana or "-", ce if ce is not None else "-",
                " = REFLEX STOP" if str(ce) == "8" else "", chk) + self.rec_txt())

    def rec_txt(self):   # (11-23) recorded clearance parameters, only when they differ from the defaults
        dft = (abs(self.k_gain - K_GAIN) < 1e-6 and abs(self.eta_max - ETA_MAX) < 1e-6 and abs(self.other_d - OTHER_D) < 1e-6 and abs(self.other_k - OTHER_K) < 1e-6
               and self.arm_parts == ARM_PARTS)
        mm = "; eta reference: %s" % ("human effective mass m_h / m_iso" if self.mass_mode == "human" else "reduced mass mu / mu_iso")   # (massmode 1124) always shown
        return mm if dft else "; recorded clearance: gain %.2f, eta_max %.2f, other parts d_base %.2f / gain %.2f, arm parts %s" % (
            self.k_gain, self.eta_max, self.other_d, self.other_k, ",".join(self.arm_parts)) + mm


f2 = lambda x, f="%.3f": "-" if x is None or not np.isfinite(x) else f % x
HDR = "| window (s) | pass | from | robot meets | gap min m (part, t) | angle u-lower arm deg | m_h / m_r / mu kg | eta eff / uni m | avoid start gap m | TCP off ref cm mean / max |"


def row_txt(r):
    return ("| %d (%.1f-%.1f) | %d (%.1f-%.1f) | %s | %s | %s (%s, %.1f) | %.0f | %.2f / %.2f / %.2f | %.3f / %.3f | %s | %s / %s |"
            % (r["win"], r["w0"], r["w1"], r["pas"], r["t0"], r["t1"], r["frm"], r["cls"], f2(r["gmin"]), r["part"], r["t_min"], r["angle"], r["m_h"], r["m_r"], r["mu"],
               r["eta_e"], r["eta_u"], f2(r["start"]), f2(r["dev_mean"], "%.1f"), f2(r["dev_max"], "%.1f")))


def cmd_legs(a):   # (4th) path timing: when the TCP passes x 0.30 toward A and leaves A (t 0 = first publish), to set the cue time
    for n in a.runs:
        r = Run(n); x = r.tcp[:, 0]; t = r.t
        ga = [float(t[i]) for i in range(1, len(x)) if x[i - 1] < 0.30 <= x[i]]; la = [float(t[i]) for i in range(1, len(x)) if x[i - 1] >= 0.40 > x[i]]
        print("%s (%s): TCP passes x 0.30 toward A at %s s; leaves A at %s s" % (r.name, r.method, " ".join("%.2f" % v for v in ga[:4]), " ".join("%.2f" % v for v in la[:4])))


def cmd_one(a):
    R = Run(a.run); ref = Run(a.ref) if a.ref else None
    print(R.runinfo()); print("reference path: %s\n" % (ref.name if ref else "none (start / dev not computed)"))
    print(HDR); print("|---|---|---|---|---|---|---|---|---|---|")
    for r in R.rows(ref):
        print(row_txt(r))


def match(ra, rb):   # passes of two runs of the same human input: windows by start time (< 1.5 s), then the k-th pass of each direction
    out = []
    for wa in sorted({x["win"] for x in ra}):
        A = [x for x in ra if x["win"] == wa]; wb = [y["win"] for y in rb if abs(y["w0"] - A[0]["w0"]) < 1.5]
        if not wb:
            continue
        B = [y for y in rb if y["win"] == wb[0]]
        for f in ("+x", "-x"):
            out += list(zip([x for x in A if x["frm"] == f], [y for y in B if y["frm"] == f]))
    return sorted(out, key=lambda p: p[0]["t0"])


def phase(r, t):   # TCP x at time t
    return float(r.tcp[min(r.n - 1, int(np.searchsorted(r.t, t))), 0])


def cmd_pair(a):
    A, B = Run(a.a), Run(a.b)
    U, E = (A, B) if A.method == "uniform" else (B, A)
    ru, re_ = U.rows(Run(a.ref_u) if a.ref_u else None), E.rows(Run(a.ref_e) if a.ref_e else None)
    print(U.runinfo()); print(E.runinfo()); print()
    print("| window (s) | robot TCP x at window start uniform / effmass | from | robot meets | gap min m uniform / effmass / diff | avoid start gap m uniform / effmass / diff | eta eff m (effmass run) | angle deg | near-axis gap m uniform / effmass |")
    print("|---|---|---|---|---|---|---|---|---|")
    for u, e in match(ru, re_):
        far = abs(u["x0"] - e["x0"]) > PHASE_DX or u["dir0"] * e["dir0"] < 0   # (4th) opposite directions; standing (0) vs moving is not a different phase
        print("| %d (%.1f) | %+.2f / %+.2f%s | %s | %s | %s / %s / %+.3f | %s / %s / %s | %.3f | %.0f | %s / %s |" % (u["win"], u["w0"], u["x0"], e["x0"], " DIFFERENT PHASE" if far else "", u["frm"], e["cls"], f2(u["gmin"]), f2(e["gmin"]), e["gmin"] - u["gmin"],
              f2(u["start"]), f2(e["start"]), "%+.3f" % (e["start"] - u["start"]) if np.isfinite(e["start"] - u["start"]) else "-", e["eta_e"], e["angle"], f2(u["ax_gap"]), f2(e["ax_gap"])))
    print("\n(robot TCP x at window start: where the robot was on its path when the arm came in; pairs far apart in x met the arm in a different phase)")


def align(rep):   # replay timeline offset: pick time in the replay - recorded time of the same record (matched by input count)
    if not rep.E.get("replay_of"):
        return None
    try:
        src = np.load(rep.E["replay_of"]); new = rep.H
        Ts = dict(zip(src["msg_input_count"].tolist(), ((src["msg_stamp_ns"] - src["step_t_pub_ns"][0]) / 1e9).tolist()))
        tn = new["msg_pick_wall"] - rep.t0; off = [tn[i] - Ts[c] for i, c in enumerate(new["msg_input_count"].tolist()) if c in Ts and tn[i] > 0.5]
        return (float(np.median(off)), float(np.percentile(off, 95))) if off else None
    except Exception:
        return None


def cmd_rel(a):
    L, P = Run(a.real), Run(a.replay)
    print(L.runinfo()); print(P.runinfo())
    al = align(P); print("replay timeline: record pick time - recorded time median %s s, p95 %s s" % ((f2(al[0]), f2(al[1])) if al else ("-", "-")))
    m = min(L.n, P.n); dq = np.abs(L.q[0] - P.q[0]).max(); dt = np.linalg.norm(L.tcp[:m] - P.tcp[:m], axis=1)
    print("start joints max diff %.4f rad; TCP real - replay (same step) mean %.1f / max %.1f cm over the run\n" % (dq, dt.mean() * 100, dt.max() * 100))
    rl, rp = L.rows(), P.rows()
    print("| window (s) | robot TCP x at window start real / replay | from | gap min m real / replay / diff | t of gap min s real / replay | TCP real - replay in the pass cm mean / max |")
    print("|---|---|---|---|---|---|")
    for x, y in match(rl, rp):
        i0, i1 = np.searchsorted(L.t, x["t0"]), np.searchsorted(L.t, x["t1"]); s = slice(i0, min(i1 + 1, m))
        dd = np.linalg.norm(L.tcp[s] - P.tcp[s], axis=1) * 100
        print("| %d (%.1f) | %+.2f / %+.2f | %s | %s / %s / %+.3f | %.1f / %.1f | %.1f / %.1f |" % (x["win"], x["w0"], phase(L, x["w0"]), phase(P, y["w0"]), x["frm"], f2(x["gmin"]), f2(y["gmin"]), y["gmin"] - x["gmin"], x["t_min"], y["t_min"],
              dd.mean() if len(dd) else np.nan, dd.max() if len(dd) else np.nan))


def ref_for(r, runs):
    if r.kind == "synth" or r.name.startswith(("y_", "x_")):
        nm = re.sub(r"_s\d+$", "_still", r.name)
        c = [x for x in runs if x.name == nm]
        if not c and os.path.exists(os.path.join(os.path.dirname(r.p), nm) + "_summary.json"):   # (R 11-22) the still run next to the run (any folder)
            c = [Run(os.path.join(os.path.dirname(r.p), nm))]
        if not c:
            try:
                c = [Run(nm)]
            except SystemExit:
                c = []
        return c[0] if c else None
    nn = [x for x in runs if x.pose == "none" and x is not r and x.kind != "synth"]
    for ok in (lambda x: x.method == r.method and x.kind == r.kind, lambda x: x.method == r.method, lambda x: x.kind == r.kind):
        c = [x for x in nn if ok(x)]   # same method and kind; same method (real <-> replay); same kind (other method: printed in the run line)
        if c:
            return sorted(c, key=lambda x: x.name)[-1]
    return None


def aborted_of(prefix):   # (item 9) control PC code 9 (home move failed) or a run that never started the MPC -> aborted
    try:
        p = find(prefix)
    except SystemExit:
        p = None
    ej = (p or prefix) + "_exp.json"
    if not os.path.exists(ej):
        for d in DIRS:
            if os.path.exists(os.path.join(d, os.path.basename(prefix) + "_exp.json")):
                ej = os.path.join(d, os.path.basename(prefix) + "_exp.json"); break
    E = json.load(open(ej)) if os.path.exists(ej) else {}
    cj = (p or prefix) + "_ctrl_exit.json"   # (2nd) the control PC EXIT= value record of real_run.sh
    if os.path.exists(cj):
        E = dict(E, control_exit=json.load(open(cj)).get("control_exit"))
    if str(E.get("control_exit")) == "9" or E.get("aborted"):
        return "control exit %s (%s)" % (E.get("control_exit"), E.get("aborted") or "home move failed")
    return None if p else "no MPC summary"


def first_pairs(globs):   # (6th) first-reach pairs (window 1) of runs matched by globs, quietly (for --synth-ref)
    names = []
    for x in globs:
        names += sorted(glob.glob(x)) if any(ch in x for ch in "*?[") else [x]
    names = list(dict.fromkeys(re.sub(r"_(summary|exp|ctrl_exit)\.json$", "", n) for n in names))
    runs = [Run(x) for x in names]; runs = [r for r in runs if not r.still]; allrows = []
    for r in runs:
        for x in r.rows(ref_for(r, runs)):
            x.update(run=r.name, method=r.method, pose=r.pose, kind=r.kind); allrows.append(x)
    by = {}
    for r in runs:
        by.setdefault((r.E.get("replay_of") or re.sub(r"_(uniform|effmass)_", "_M_", r.name), r.pose), {})[r.method] = r
    out = []
    for k, d in by.items():
        if "uniform" in d and "effmass" in d:
            for u, e in match([x for x in allrows if x["run"] == d["uniform"].name], [x for x in allrows if x["run"] == d["effmass"].name]):
                if u["win"] == 1:
                    out.append((k, u, e, abs(u["x0"] - e["x0"]) > PHASE_DX or u["dir0"] * e["dir0"] < 0))
    return out


SYN_COL = "\ud569\uc131(\uc549\uc74c)"   # (6th) column title: synthetic (seated)


def cmd_agg(a):
    names = []
    for x in a.runs:
        names += sorted(glob.glob(x)) if any(ch in x for ch in "*?[") else [x]
    names = list(dict.fromkeys(re.sub(r"_(summary|exp|ctrl_exit)\.json$", "", n) for n in names))
    ab = [(n, aborted_of(n)) for n in names]; ab = [(n, w) for n, w in ab if w]
    for n, w in ab:
        print("ABORTED run (left out of the comparison): %s - %s" % (os.path.basename(n), w))
    names = [n for n in names if n not in dict(ab)]
    runs = [Run(x) for x in names]; runs = [r for r in runs if not r.still]
    unc = [r.name for r in runs if r.pose_unconfirmed]
    if unc:   # (4th) at the top: runs with _H_ in the name and no _pose.json (P2 runs need set_pose.py)
        print("WARNING: POSE NOT CONFIRMED (name has _H_ / pose H, no <name>_pose.json and no recording _pose.json; P2 runs need")
        print("  python3 goal1/real_prep5/exp1121/set_pose.py <name> P2). Listed separately as pose 'H(unconfirmed)':")
        for n in unc:
            print("  - " + n)
        print("  (S1 / S2 runs, started with --pose H: set_pose.py <name> S1 or S2)")   # (R 11-22)
    rx = [r.name for r in runs if r.reflex]
    if rx:
        print("REFLEX STOP runs (control exit 8; kept in the tables): " + ", ".join(rx))
    allrows = []
    for r in runs:
        ref = ref_for(r, runs); rr = r.rows(ref)
        print(r.runinfo() + "; reference %s; %d passes" % (ref.name if ref else "none", len(rr)))
        for x in rr:
            x.update(run=r.name, method=r.method, pose=r.pose, kind=r.kind); allrows.append(x)
    # pairs: same recording (replay_of) or same synthetic input (name without the method)
    key = lambda r: (r.E.get("replay_of") or re.sub(r"_(uniform|effmass)_", "_M_", r.name), r.pose)
    by = {}
    for r in runs:
        by.setdefault(key(r), {})[r.method] = r
    pairs = []
    for k, d in by.items():
        if "uniform" in d and "effmass" in d:
            ru = [x for x in allrows if x["run"] == d["uniform"].name]; re_ = [x for x in allrows if x["run"] == d["effmass"].name]
            for u, e in match(ru, re_):
                far = abs(u["x0"] - e["x0"]) > PHASE_DX or u["dir0"] * e["dir0"] < 0   # (4th) opposite directions; standing (0) vs moving is not a different phase
                pairs.append((k, u, e, far))
    fin = lambda v: [x for x in v if np.isfinite(x)]
    def mr(v, f="%+.1f"):
        v = fin(v)
        return "-" if not v else (f + " [" + f + ", " + f + "] (%d)") % (np.median(v), min(v), max(v), len(v))
    runinfo = {r.name: r for r in runs}
    def table(sel, title, flag_cols, syn=None):
        print("\n" + title + "\n")
        print("| pose | robot meets | avoid start diff (MAIN)%s | near-axis gap diff %.0f deg | near-axis gap diff 20 deg | gap min diff | gap min uniform: median / min / passes < 0 | gap min effmass: median / min / passes < 0 | arrivals per run uniform / effmass | reflex stop runs uniform / effmass | late steps %% uniform / effmass |%s"
              % (", same phase only" if flag_cols else "", AX_DEG, "" if syn is None else " %s: avoid start diff / gap min uniform / effmass |" % SYN_COL))
        print("|---|---|---|---|---|---|---|---|---|---|---|" + ("" if syn is None else "---|"))
        keys = {(k[1], e["cls"]) for k, u, e, far in sel}
        if syn is not None:   # (6th) also the sides seen only in the synthetic reference (real '-'), for the poses of this table
            keys |= {(k[1], e["cls"]) for k, u, e, far in syn if k[1] in {kk[1] for kk, _, _, _ in sel}}
        for p, c in sorted(keys):
            P_ = [(u, e, far) for k, u, e, far in sel if k[1] == p and e["cls"] == c]
            ds = lambda key: [(e[key] - u[key]) * 100 for u, e, far in P_ if not (flag_cols and far)]
            gm = lambda m: [(u if m == "uniform" else e)["gmin"] * 100 for u, e, far in P_]
            gtxt = lambda m: "-" if not gm(m) else "%.1f / %.1f / %d" % (np.median(gm(m)), min(gm(m)), sum(1 for v in gm(m) if v < 0))
            rr = lambda m: [runinfo[n] for n in sorted({(u if m == "uniform" else e)["run"] for u, e, far in P_})]
            arr = lambda m: ", ".join(str(r.S.get("arrivals")) for r in rr(m)) or "-"
            rfx = lambda m: "%d of %d" % (sum(r.reflex for r in rr(m)), len(rr(m)))
            late = lambda m: ", ".join("%.2f" % (100.0 * (r.S.get("deadline_miss") or 0) / max(r.S.get("steps") or 1, 1)) for r in rr(m)) or "-"
            sc = ""
            if syn is not None:   # (6th) the same pose / side in the synthetic reference (first reach, same-phase pairs)
                S_ = [(u, e) for k, u, e, far in syn if k[1] == p and e["cls"] == c and not far]
                sg = lambda m: [(u if m == "uniform" else e)["gmin"] * 100 for u, e in S_]
                sc = " %s / %s / %s |" % (mr([(e["start"] - u["start"]) * 100 for u, e in S_]), "%.1f" % np.median(sg("uniform")) if S_ else "-", "%.1f" % np.median(sg("effmass")) if S_ else "-")
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s / %s | %s / %s |%s" % (p, c, mr(ds("start")), mr(ds("ax_gap")), mr(ds("ax_gap20")), mr(ds("gmin")),
                  gtxt("uniform"), gtxt("effmass"), arr("uniform"), arr("effmass"), rfx("uniform"), rfx("effmass"), late("uniform"), late("effmass"), sc))
    first = [x for x in pairs if x[1]["win"] == 1]; later = [x for x in pairs if x[1]["win"] > 1]
    print("\nTables in cm; differences = effective mass - uniform over matched passes, median [min, max] (n). Near-axis gap = smallest gap")
    print("with u within the angle of the lower-arm axis. Phase flag: robot TCP x at the window start differs > %.2f m or moves the other way." % PHASE_DX)
    table(first, "MAIN: first reach of each run (window 1; the robot is still on the same path in both runs)", False)
    if first:
        nf = sum(1 for x in first if x[3])
        print("\n(first-reach pairs flagged DIFFERENT PHASE: %d of %d)" % (nf, len(first)))
    # (5th) the two experiment bundles, first reach only: P2 = front (lower arm head-on, large effective mass), L = side (straight arm, small)
    SYNF = first_pairs([g for g in a.synth_ref.split(",") if g.strip()]) if getattr(a, "synth_ref", "") else None
    for bp, bt in BUNDLES + BUNDLES_S:   # (R 11-22) S1 / S2 tables only when present (P2 / L as before)
        sel = [x for x in first if x[0][1] == bp]
        if sel:
            table(sel, "BUNDLE %s: %s - first reach" % (bp, bt), False, syn=SYNF)
    table(later, "SECONDARY: second and later reaches (pairs flagged DIFFERENT PHASE left out of the differences; all passes in the gap columns)", True)
    if later:
        print("\n(later pairs flagged DIFFERENT PHASE: %d of %d)" % (sum(1 for x in later if x[3]), len(later)))
    if pairs:
        print("\nPairs (per pass): robot TCP x at the window start uniform / effmass, flag, differences effmass - uniform (cm)\n")
        print("| input | pose | window | robot meets | TCP x0 uniform / effmass (dir) | phase flag | avoid start diff | near-axis gap diff | near-axis 20 deg | gap min diff |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        f1 = lambda v: "%+.1f" % (v * 100) if np.isfinite(v) else "-"
        for k, u, e, far in pairs:
            print("| %s | %s | %d (%.1f s) | %s | %+.2f / %+.2f (%+d / %+d) | %s | %s | %s | %s | %s |" % (os.path.basename(str(k[0])).replace("_human.npz", ""), k[1], u["win"], u["w0"], e["cls"],
                  u["x0"], e["x0"], u["dir0"], e["dir0"], "DIFFERENT PHASE" if far else ("first reach" if u["win"] == 1 else ""), f1(e["start"] - u["start"]), f1(e["ax_gap"] - u["ax_gap"]), f1(e["ax_gap20"] - u["ax_gap20"]), f1(e["gmin"] - u["gmin"])))
    if a.fig:
        figs(allrows, pairs, a.fig); fig_groups(first, a.fig, SYNF)


BUNDLES = (("P2", "front, lower arm head-on (large effective mass)"), ("L", "side, straight arm across the path (small effective mass)"))
BUNDLE_CLS = {"P2": ("fist",), "L": None}   # (5th) sides shown in fig_groups: P2 the fist side (the robot comes head-on); L every side
# (R 11-22) next experiment scenarios (RUN_SHEET_next.md); every side (a straight / 60 deg arm is class "side")
BUNDLES_S = (("S1", "right arm straight toward the path, fist head-on (large effective mass)"), ("S2", "left arm bent 60 deg, fist toward +x and the path"))
BUNDLE_CLS.update({"S1": None, "S2": None})
BUNDLE_SHORT = {"P2": "front, large m_eff", "L": "side, small m_eff", "S1": "straight arm, fist head-on", "S2": "bent 60 deg, fist head-on"}


def fig_groups(first, out, syn=None):   # (5th) avoidance start difference effmass - uniform per first-reach pair, P2 and L side by side
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True); INK, MUTE, C = "#222222", "#8a8a8a", "#2f6db3"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTE, "axes.labelcolor": INK})
    BUN = BUNDLES   # (R 11-22) with S1 / S2 pairs: the bundles that have data (first reach or synthetic reference); else P2 / L as before
    if any(k[1] in ("S1", "S2") for k, u, e, far in first):
        BUN = tuple(b for b in BUNDLES + BUNDLES_S if any(k[1] == b[0] for k, u, e, far in list(first) + list(syn or [])))
    fig, ax = plt.subplots(figsize=(4.8 if BUN is BUNDLES else max(4.8, 3.2 * len(BUN)), 3.6)); labs = []
    for gi, (bp, bt) in enumerate(BUN):
        cl = BUNDLE_CLS.get(bp)
        ok = lambda e, cl: cl is None or e["cls"] in cl
        if cl is not None and not any(k[1] == bp and ok(e, cl) for k, u, e, far in first):
            cl = None   # no pass on that side (e.g. the robot never reached it): all sides
        v = [(e["start"] - u["start"]) * 100 for k, u, e, far in first if k[1] == bp and ok(e, cl) and not far and np.isfinite(u["start"]) and np.isfinite(e["start"])]
        xs = gi + (np.arange(len(v)) - (len(v) - 1) / 2) * min(0.06, 0.5 / max(len(v), 1))
        ax.scatter(xs, v, s=26, color=C, edgecolors="white", linewidths=1, zorder=3)
        if v:
            ax.plot([gi - 0.22, gi + 0.22], [np.median(v)] * 2, color=INK, lw=2, solid_capstyle="round", zorder=4)
            ax.annotate("median %+.1f" % np.median(v), (gi - 0.25, np.median(v)), va="center", ha="right", fontsize=8, color=INK)
        if syn is not None:   # (6th) synthetic (seated) reference: hollow grey dots right of the real ones
            vs = [(e["start"] - u["start"]) * 100 for k, u, e, far in syn if k[1] == bp and ok(e, BUNDLE_CLS.get(bp)) and not far and np.isfinite(u["start"]) and np.isfinite(e["start"])]
            xs2 = gi + 0.32 + (np.arange(len(vs)) - (len(vs) - 1) / 2) * 0.03
            ax.scatter(xs2, vs, s=22, facecolors="none", edgecolors=MUTE, linewidths=1.2, zorder=3, label=("synthetic (seated)" if BUN is BUNDLES else "simulation reference") if gi == 0 else None)
            if vs:
                ax.plot([gi + 0.22, gi + 0.42], [np.median(vs)] * 2, color=MUTE, lw=2, zorder=4)
        side = "" if cl is None else ", %s side" % "/".join(cl)
        labs.append("%s%s\n%s\nn = %d" % (bp, side, BUNDLE_SHORT.get(bp, bt), len(v)))
    if syn is not None:
        ax.scatter([], [], s=26, color=C, label="real / replay"); ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.axhline(0, color=MUTE, lw=0.8, zorder=1); ax.set_xticks(range(len(BUN))); ax.set_xticklabels(labs); ax.set_xlim(-0.8, len(BUN) - 0.1)
    ax.set_ylabel("avoidance start gap difference (cm)\neffective mass - uniform"); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_title("First reach: one dot per paired pass, bar = median", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_groups.png"), dpi=150); plt.close(fig)
    print("figure: %s/fig_groups.png" % out)


def figs(allrows, pairs, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True); CU, CE, INK, MUTE = "#2f6db3", "#d1661b", "#222222", "#8a8a8a"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTE, "axes.labelcolor": INK})
    groups = sorted({(k[1], e["cls"]) for k, u, e, far in pairs})
    if groups:
        fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(groups), 3.4))
        for gi, gk in enumerate(groups):
            v = [(u, e) for k, u, e, far in pairs if (k[1], e["cls"]) == gk and np.isfinite(u["start"]) and np.isfinite(e["start"])]
            for j, (u, e) in enumerate(v):
                x0 = gi + (j - (len(v) - 1) / 2) * 0.05
                ax.plot([x0 - 0.15, x0 + 0.15], [u["start"] * 100, e["start"] * 100], color=MUTE, lw=1, zorder=1)
                ax.scatter([x0 - 0.15], [u["start"] * 100], s=22, color=CU, zorder=2, edgecolors="white", linewidths=1)
                ax.scatter([x0 + 0.15], [e["start"] * 100], s=22, color=CE, zorder=2, edgecolors="white", linewidths=1)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(["%s, %s" % g for g in groups]); ax.set_ylabel("avoidance start gap (cm)")
        ax.scatter([], [], color=CU, label="uniform"); ax.scatter([], [], color=CE, label="effective mass"); ax.legend(frameon=False, loc="upper left")
        ax.set_title("Same human input, two methods (line = one pass: uniform -> effective mass)", fontsize=9, color=INK); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_pairs.png"), dpi=150); plt.close(fig)
    groups = sorted({(x["pose"], x["cls"]) for x in allrows})
    if groups:
        fig, axs = plt.subplots(1, 2, figsize=(2.0 + 2.2 * len(groups), 3.4))
        for ax, k, lab in ((axs[0], "start", "avoidance start gap (cm)"), (axs[1], "ax_gap", "near-axis gap (cm), u within %.0f deg of the lower arm" % AX_DEG)):
            for gi, gk in enumerate(groups):
                for mi, (m, c) in enumerate((("uniform", CU), ("effmass", CE))):
                    v = np.array([x[k] for x in allrows if (x["pose"], x["cls"]) == gk and x["method"] == m], float) * 100; v = v[np.isfinite(v)]
                    if len(v):
                        xx = gi + (mi - 0.5) * 0.3
                        ax.plot([xx, xx], [v.min(), v.max()], color=c, lw=2, solid_capstyle="round"); ax.scatter([xx], [np.median(v)], s=40, color=c, zorder=3, edgecolors="white", linewidths=1.5)
            ax.set_xticks(range(len(groups))); ax.set_xticklabels(["%s, %s" % g for g in groups], rotation=0); ax.set_ylabel(lab); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        axs[0].scatter([], [], color=CU, label="uniform"); axs[0].scatter([], [], color=CE, label="effective mass"); axs[0].legend(frameon=False, loc="upper left")
        fig.suptitle("Pose x robot approach: median (dot) and range (bar) over passes", fontsize=9, color=INK)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_pose.png"), dpi=150); plt.close(fig)
    print("\nfigures: %s/fig_pairs.png, %s/fig_pose.png" % (out, out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("one"); p.add_argument("run"); p.add_argument("--ref", default=""); p.add_argument("--win-r", type=float, default=WIN_R)
    p = sp.add_parser("pair"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--ref-u", default=""); p.add_argument("--ref-e", default=""); p.add_argument("--win-r", type=float, default=WIN_R)
    p = sp.add_parser("legs"); p.add_argument("runs", nargs="+")
    p = sp.add_parser("rel"); p.add_argument("real"); p.add_argument("replay"); p.add_argument("--win-r", type=float, default=WIN_R)
    p = sp.add_parser("agg"); p.add_argument("runs", nargs="+"); p.add_argument("--fig", default=""); p.add_argument("--ax-deg", type=float, default=AX_DEG); p.add_argument("--win-r", type=float, default=WIN_R)
    p.add_argument("--synth-ref", default="", help="(6th) comma-separated globs of synthetic runs (e.g. the seated sP2_* / sL_*): BUNDLE tables get a column of their first-reach values, fig_groups their dots")
    a = ap.parse_args(); AX_DEG = getattr(a, "ax_deg", AX_DEG); WIN_R = getattr(a, "win_r", WIN_R); dict(one=cmd_one, pair=cmd_pair, rel=cmd_rel, agg=cmd_agg, legs=cmd_legs)[a.cmd](a)
