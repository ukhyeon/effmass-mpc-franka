"""rp5_jump_filter.py - H2 (2026-11-20): reject human frames whose torso jumps (sensing briefly placing the body on the robot).
Rule (from goal1/real_prep4/real_ym_1120_152448_human.npz: the torso jumped 0.85-0.92 m in one frame onto the robot base column, two
episodes of about 1 s; elsewhere the torso never moved more than 0.021 m per frame):
  torso = pelvis, abdomen, chest COM (segments 0, 3, 8). A frame whose torso moved more than --jump (0.2 m) from the PREVIOUS frame starts
  a hold: this and the following frames are rejected (the last accepted frame is used instead) until the torso is back within --back
  (0.2 m) of the last accepted torso, or more than --hold (1.5 s) after the last accepted frame (then the new position is taken).
Distances only, so the frame of the positions does not matter (node world or robot base).
Used by rp2_mpc_rt.py --human-jump-filter 1 (in its human reader) and by goal1/real_prep5/filt/jf_replay.py (offline replay).

AnchorFilter (mode "anchor", 2026-11-20 evening; opt-in rp2_mpc_rt.py --hjf-mode anchor, the hold mode above stays the default).
Why: in real_ym_1120_164039 the sensing put a second (ghost) body on the robot and switched between it and the real person for 4 s;
its temporal smoothing gave single in-between frames 0.3-0.4 m off the real place. The hold mode took those in-between frames as
"back" (within 0.2 m of the last accepted one), so its reference crept toward the robot, and the real person (standing still) was then
more than 0.2 m away and rejected for the full 1.5 s hold (8.80-10.24 s), while the arm reached toward the robot.
Rule (reference = last ACCEPTED torso, j = torso step from the previous RAW frame, max over pelvis/abdomen/chest):
  accept if j <= --jump (0.3) and the torso is within --back (0.2) of the reference          (normal motion; back after a glitch)
  or the torso has moved <= --settle-step (0.1 m) per frame for --settle-n (3) frames in a row and no torso segment COM is within
     --near (0.25 m) of a robot collision sphere surface                                       (a real new place, e.g. walked during a glitch)
  or more than --hold-max (3.0 s) since the last accepted frame                                (last resort, takes anything)
  else reject (the last accepted frame is used). No plain timeout: an in-between frame never becomes the reference.
near_fn(torso (3,3), same frame as com) -> smallest distance (m) to a robot sphere surface; None = no near test (settling then takes any place)."""
import numpy as np

TORSO = (0, 3, 8)


class JumpFilter:
    def __init__(self, jump=0.2, back=0.2, hold=1.5):
        self.jump, self.back, self.hold = float(jump), float(back), float(hold)
        self.prev = None; self.acc_tor = None; self.acc_t = None; self.holding = False
        self.n_seen = 0; self.n_rejected = 0; self.events = []   # [t_first_rejected, t_last_rejected, frames]

    def step(self, t, com):
        """t: frame time (s), com: (19, 3) segment COM. Returns True = use this frame, False = rejected (use the last accepted)."""
        tor = np.asarray(com, np.float64).reshape(-1, 3)[list(TORSO)]; self.n_seen += 1
        if self.prev is None or self.acc_tor is None:
            self.prev = tor; self.acc_tor = tor; self.acc_t = t; return True
        j = float(np.linalg.norm(tor - self.prev, axis=1).max()); self.prev = tor
        if not self.holding and j > self.jump:
            self.holding = True; self.events.append([t, t, 0])
        if self.holding:
            b = float(np.linalg.norm(tor - self.acc_tor, axis=1).max())
            if b < self.back or (t - self.acc_t) > self.hold:
                self.holding = False
        if not self.holding:
            self.acc_tor = tor; self.acc_t = t; return True
        self.n_rejected += 1; self.events[-1][1] = t; self.events[-1][2] += 1
        return False

    def describe(self):
        return dict(jump_m=self.jump, back_m=self.back, hold_s=self.hold)


def robot_near_fn(RI, get_q, R=None, t=None):
    """near_fn for AnchorFilter: torso in the node world frame (R, t: world -> robot base, p_base = R p + t; None = already base) and the
    robot collision spheres of RobotInertiaNP RI at the joints get_q() returns (measured). About 0.05 ms per call."""
    rad = np.asarray(RI.sphere_radius, float)

    def f(tor):
        q = get_q()
        if q is None:
            return np.inf
        W = RI._fk(np.asarray(q, float).reshape(-1)); Ws = W[RI.s_idx]
        P = np.einsum("sij,sj->si", Ws[:, :3, :3], RI.s_local) + Ws[:, :3, 3]
        x = np.asarray(tor, float) if R is None else np.asarray(tor, float) @ np.asarray(R).T + np.asarray(t)
        return float((np.linalg.norm(x[:, None] - P[None], axis=2) - rad[None]).min())
    return f


class AnchorFilter:
    """mode "anchor" (see the module text). Same interface as JumpFilter: step(t, com) -> True = use, False = rejected."""

    def __init__(self, jump=0.3, back=0.2, settle_n=3, settle_step=0.1, near=0.25, hold_max=3.0, near_fn=None):
        self.jump, self.back, self.settle_n, self.settle_step = float(jump), float(back), int(settle_n), float(settle_step)
        self.near, self.hold_max, self.near_fn = float(near), float(hold_max), near_fn
        self.prev = None; self.acc_tor = None; self.acc_t = None; self.run = 0; self.last_ok = True
        self.n_seen = 0; self.n_rejected = 0; self.events = []   # [t_first_rejected, t_last_rejected, frames]
        self.n_settled = 0; self.n_hold_max = 0; self.n_near_rejected = 0

    def step(self, t, com):
        tor = np.asarray(com, np.float64).reshape(-1, 3)[list(TORSO)]; self.n_seen += 1
        if self.prev is None or self.acc_tor is None:
            self.prev = tor; self.acc_tor = tor; self.acc_t = t; return True
        j = float(np.linalg.norm(tor - self.prev, axis=1).max()); self.prev = tor
        self.run = self.run + 1 if j <= self.settle_step else 0
        ok = j <= self.jump and float(np.linalg.norm(tor - self.acc_tor, axis=1).max()) <= self.back
        if not ok and self.run >= self.settle_n:
            nr = self.near_fn(tor) if (self.near_fn is not None and self.near > 0) else np.inf
            if nr >= self.near:
                ok = True; self.n_settled += 1
            else:
                self.n_near_rejected += 1
        if not ok and (t - self.acc_t) > self.hold_max:
            ok = True; self.n_hold_max += 1
        if ok:
            self.acc_tor = tor; self.acc_t = t; self.last_ok = True; return True
        if self.last_ok:
            self.events.append([t, t, 0])
        self.last_ok = False; self.n_rejected += 1; self.events[-1][1] = t; self.events[-1][2] += 1
        return False

    def describe(self):
        return dict(mode="anchor", jump_m=self.jump, back_m=self.back, settle_n=self.settle_n, settle_step_m=self.settle_step, near_m=self.near,
                    hold_max_s=self.hold_max, near_test=self.near_fn is not None)

    def counters(self):
        return dict(settled=self.n_settled, hold_max=self.n_hold_max, settled_refused_near_robot=self.n_near_rejected)
