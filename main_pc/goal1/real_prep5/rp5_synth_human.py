"""rp5_synth_human.py - OVERNIGHT_H1 (2026-11-20): synthetic human for the MPC tests, no camera, no human node at run time.
Writes the rp5_human_relay.py shared-memory record (float64[509], seqlock; same layout) at --rate Hz, so rp2_mpc_rt.py --human shm reads
it exactly like the relay. The human is ONE captured HumanInertiaState (h1n/cap_P4_side.npz: right arm straight and horizontal,
human node v2 shapes) moved rigidly (positions rotated + translated, velocities = rigid-body velocities, inverse inertia R L R^T), so the
arm does not articulate. Robot path of scene_AB_ym_rp5floor.json: base y -0.55, x -0.31 .. 0.29, hand height z 0.32.
Timeline (t = 0 at --start-log PATTERN, i.e. the MPC's first step; before that the human stands still at the start of motion 1):
  motion 1 (0-60 s)    "front": arm normal to the path (pointing +y), at x 0; the fingertip surface stops 0.10 m before the path line
                         (the robot passes in front of the fingertip: closest direction along the arm, large effective mass)
  motion 2 (60-120 s)  "side" : arm along the path (pointing +x), hand surface 0.10 m beside the path line (-y side); the hand slides in
                         from the B end, fingertip x -0.95 -> -0.02 (the robot passes alongside the hand: direction across the arm, small mass)
  motion 3 (120-180 s) "wave" : arm normal to the path, fingertip rests 0.25 m before the path line, whole body yaws +-35 deg about the
                         right shoulder at 1.5 Hz (fast arm swing beside the path)
  motions 1, 2: 10 s cycle = 2 s in (cosine), 5 s hold, 2 s out, 1 s hold; travel 0.60 m (1) / 0.93 m (2); fingertip height = captured (base z about 0.34)
  (first version tonight put the fingertip onto the path line: geometry-limited gaps, no mass effect; replaced)
  after 180 s: motion 1 far position, still.
Output frames are in the node world frame = base + (0, 0, 0.715) (live_frame_rp5.yml: R = I, t = (0, 0, -0.715)).
Log (--log npz): wall time, monotonic time, motion id, com (19,3), p0, p1 (24,3) of every written frame (base frame).
Teleport (H2 jump-filter test, 2026-11-20; default off): --teleport T1,T2,... [--teleport-dur 1.0] [--teleport-xy 0.12,0.09] moves the whole
body rigidly (com, shapes; velocities and inverse inertia unchanged) so that the pelvis is at base x, y = --teleport-xy (the robot base
column, as the sensing did on the real robot, real_ym_1120_152448) for --teleport-dur s from each Ti s of the timeline; the log (com,
p0, p1) keeps the TRUE body and gets teleported (0/1) per frame.
Walk (H2 jump-filter test, 2026-11-20 evening; default off): --walk "T,DX,DY,DUR[;T,DX,DY,DUR...]" moves the whole TRUE body rigidly by
(DX, DY) m (base frame) over DUR s from timeline time T (cosine speed profile, peak pi/2 x mean; velocities include it) and it stays
there; e.g. --walk "13,0.5,0,1" = 0.5 m sideways in 1 s. A teleport during a walk still puts the pelvis on --teleport-xy; the log keeps
the true (walking) body, walking (0/1) per frame. Checks that the jump filter does not reject a person who really moves.
Run with system python (numpy only): python3 rp5_synth_human.py --shm-name rp5_human --start-log <mpc.log> --duration 400 --log X.npz
"""
import argparse, math, os, time
import numpy as np
from multiprocessing import shared_memory
ap = argparse.ArgumentParser()
ap.add_argument("--shm-name", default="rp5_human"); ap.add_argument("--rate", type=float, default=30.0); ap.add_argument("--duration", type=float, default=400.0)
ap.add_argument("--cap", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1n/cap_P4_side.npz"))
ap.add_argument("--start-log", default=""); ap.add_argument("--start-pattern", default="warm reset"); ap.add_argument("--stop-file", default="")
ap.add_argument("--log", default=""); ap.add_argument("--motions", default="1,2,3", help="order of the 60 s motions")
ap.add_argument("--seg-len", type=float, default=60.0)
ap.add_argument("--teleport", default="", help="(H2 jf test) timeline times (s) of 1 s teleports of the body onto the robot base column, e.g. 4,14")
ap.add_argument("--walk", default="", help="(H2 jf test) 'T,DX,DY,DUR[;...]': rigid walk of the true body by DX, DY m in DUR s from T s")
ap.add_argument("--teleport-dur", type=float, default=1.0); ap.add_argument("--teleport-xy", default="0.12,0.09", help="pelvis x,y (base) while teleported")
a = ap.parse_args()
Z0 = 0.715; YP = -0.55
z = np.load(a.cap)
com0 = z["com_position"][-1].reshape(19, 3) - [0, 0, Z0]; inv0 = z["inv_inertia"][-1].reshape(19, 3, 3)
p00 = z["shape_p0"][-1].reshape(24, 3) - [0, 0, Z0]; p10 = z["shape_p1"][-1].reshape(24, 3) - [0, 0, Z0]
rad = z["shape_radius"][-1]; seg = z["shape_segment"][-1]; typ = z["shape_type"][-1]
UA, HAND, SH = 14, 18, 11   # right upper arm, right hand, right shoulder
ax0 = com0[HAND] - com0[UA]; ax0[2] = 0.0; ax0 /= np.linalg.norm(ax0); yaw0 = math.atan2(ax0[1], ax0[0])
hs = np.nonzero(seg == HAND)[0]; pts = np.vstack([p00[hs], p10[hs]]); _i = int(np.argmax(pts @ ax0)); tip0 = pts[_i]   # fingertip end of the hand shapes
R_TIP = float(np.concatenate([rad[hs], rad[hs]])[_i]); R_HAND = float(rad[hs].max())   # radius at the fingertip end / largest hand capsule
GAP = 0.10   # hand surface to the path line


def rz(t):
    c, s = math.cos(t), math.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pose(yaw, tip_xy, pivot=None, dyaw=0.0):
    """rotate the captured body so the arm points to yaw, put the fingertip at tip_xy (z kept); then extra yaw dyaw about pivot (shoulder)"""
    R = rz(yaw - yaw0); tip = R @ tip0; T = np.array([tip_xy[0] - tip[0], tip_xy[1] - tip[1], 0.0])
    f = lambda p: p @ R.T + T
    com, p0, p1 = f(com0), f(p00), f(p10)
    if dyaw:
        R2 = rz(dyaw); pv = com[SH] if pivot is None else pivot
        g = lambda p: (p - pv) @ R2.T + pv
        com, p0, p1, R = g(com), g(p0), g(p1), R2 @ R
    return com, p0, p1, R


def cyc(t, period=10.0):   # 0 -> 1 (2 s), hold 5 s, 1 -> 0 (2 s), hold 1 s (cosine ramps), and d/dt
    u = t % period
    if u < 2: s, ds = 0.5 - 0.5 * math.cos(math.pi * u / 2), 0.25 * math.pi * math.sin(math.pi * u / 2)
    elif u < 7: s, ds = 1.0, 0.0
    elif u < 9: s, ds = 0.5 + 0.5 * math.cos(math.pi * (u - 7) / 2), -0.25 * math.pi * math.sin(math.pi * (u - 7) / 2)
    else: s, ds = 0.0, 0.0
    return s, ds


def surf_off(yaw):   # (max over hand capsules of y + radius) - fingertip y, for the arm pointing to yaw (fingertip at y 0)
    _, p0, p1, _ = pose(yaw, (0.0, 0.0)); hs2 = np.nonzero(seg == HAND)[0]
    return float(np.max(np.concatenate([p0[hs2][:, 1] + rad[hs2], p1[hs2][:, 1] + rad[hs2]])))


OFF1, OFF2 = surf_off(math.pi / 2), surf_off(0.0)   # so that the hand SURFACE (not the axis) is GAP from the path line
MOT = [int(x) for x in a.motions.split(",")]
WALK = [[float(v) for v in w.split(",")] for w in a.walk.split(";") if w.strip()]
assert all(len(w) == 4 and w[3] > 0 for w in WALK), "--walk T,DX,DY,DUR[;...]"
TELE = [float(x) for x in a.teleport.split(",") if x.strip()]; TXY = np.array([float(x) for x in a.teleport_xy.split(",")])


def walk_off(t):   # (H2 jf test) offset and velocity of the --walk events at timeline time t
    o = np.zeros(3); v = np.zeros(3); on = False
    for t0_, dx, dy, du in WALK:
        u = min(max((t - t0_) / du, 0.0), 1.0); d = np.array([dx, dy, 0.0]); o += (0.5 - 0.5 * math.cos(math.pi * u)) * d
        if 0.0 < u < 1.0:
            v += 0.5 * math.pi / du * math.sin(math.pi * u) * d; on = True
    return o, v, on


def state(t):
    """motion id, com, p0, p1, R, com velocity (19,3)"""
    if t < 0 or t >= a.seg_len * len(MOT):
        m = MOT[0] if t < 0 else 0; t = 0.0
    else:
        m = MOT[int(t // a.seg_len)]; t = t - a.seg_len * int(t // a.seg_len)
    if m in (0, 1):
        s, ds = cyc(t) if m == 1 else (0.0, 0.0); y = YP - GAP - OFF1 - 0.60 + 0.60 * s
        com, p0, p1, R = pose(math.pi / 2, (0.0, y)); v = np.tile([0.0, 0.60 * ds, 0.0], (19, 1))
    elif m == 2:
        s, ds = cyc(t); x = -0.95 + 0.93 * s
        com, p0, p1, R = pose(0.0, (x, YP - GAP - OFF2)); v = np.tile([0.93 * ds, 0.0, 0.0], (19, 1))
    else:
        w = 2 * math.pi * 1.5; amp = math.radians(35.0); th = amp * math.sin(w * t); om = amp * w * math.cos(w * t)
        base_com, _, _, _ = pose(math.pi / 2, (0.0, YP - 0.25)); pv = base_com[SH].copy()
        com, p0, p1, R = pose(math.pi / 2, (0.0, YP - 0.25), pivot=pv, dyaw=th)
        v = np.cross(np.array([0, 0, om]), com - pv)
    return m, com, p0, p1, R, v


LEN = 509; OFF = dict(com=8, vel=65, inv=122, p0=293, p1=365, rad=437, seg=461, type=485)
try:
    old = shared_memory.SharedMemory(name=a.shm_name); old.close(); old.unlink()
except FileNotFoundError:
    pass
shm = shared_memory.SharedMemory(name=a.shm_name, create=True, size=LEN * 8); B = np.ndarray((LEN,), np.float64, buffer=shm.buf); B[:] = 0.0
LOG = dict(wall=[], mono=[], motion=[], com=[], p0=[], p1=[], teleported=[], walking=[]); n = 0; t_start = None; T0 = time.time(); dt = 1.0 / a.rate
print(f"[synth-human] shm {a.shm_name}, {a.rate} Hz, motions {MOT} x {a.seg_len} s, start at '{a.start_pattern}' in {a.start_log or '(now)'}", flush=True)
if not a.start_log:
    t_start = time.monotonic()
try:
    nxt = time.monotonic()
    while time.time() - T0 < a.duration and not (a.stop_file and os.path.exists(a.stop_file)):
        now = time.monotonic()
        if nxt > now:
            time.sleep(nxt - now)
        nxt += dt; now = time.monotonic()
        if t_start is None and n % 10 == 0 and os.path.exists(a.start_log):
            try:
                if a.start_pattern in open(a.start_log, errors="ignore").read():
                    t_start = now; print(f"[synth-human] timeline start (frame {n})", flush=True)
            except OSError:
                pass
        t = -1.0 if t_start is None else now - t_start
        m, com, p0, p1, R, v = state(t)
        wk = False
        if WALK and t >= 0:   # (H2 jf test) the true body walks
            wo, wv, wk = walk_off(t); com, p0, p1, v = com + wo, p0 + wo, p1 + wo, v + wv
        inv = np.einsum("ij,sjk,lk->sil", R, inv0, R)
        tele = t >= 0 and any(t0_ <= t < t0_ + a.teleport_dur for t0_ in TELE)
        cw, p0w, p1w = com, p0, p1
        if tele:   # (H2 jf test) the whole body moved so that the pelvis stands on the robot base column
            off = np.array([TXY[0] - com[0, 0], TXY[1] - com[0, 1], 0.0]); cw, p0w, p1w = com + off, p0 + off, p1 + off
        wall = time.time(); s = B[0]; B[0] = s + 1.0
        B[1] = int(wall); B[2] = int((wall - int(wall)) * 1e9); B[3] = wall; B[4] = n + 1; B[5] = 0.0; B[6] = now; B[7] = n + 1
        B[OFF["com"]:OFF["com"] + 57] = (cw + [0, 0, Z0]).ravel(); B[OFF["vel"]:OFF["vel"] + 57] = v.ravel(); B[OFF["inv"]:OFF["inv"] + 171] = inv.ravel()
        B[OFF["p0"]:OFF["p0"] + 72] = (p0w + [0, 0, Z0]).ravel(); B[OFF["p1"]:OFF["p1"] + 72] = (p1w + [0, 0, Z0]).ravel()
        B[OFF["rad"]:OFF["rad"] + 24] = rad; B[OFF["seg"]:OFF["seg"] + 24] = seg; B[OFF["type"]:OFF["type"] + 24] = typ
        B[0] = s + 2.0; n += 1
        if a.log:
            LOG["wall"].append(wall); LOG["mono"].append(now); LOG["motion"].append(m if t >= 0 else -1)
            LOG["com"].append(com.astype(np.float32)); LOG["p0"].append(p0.astype(np.float32)); LOG["p1"].append(p1.astype(np.float32)); LOG["teleported"].append(int(tele)); LOG["walking"].append(int(wk))
except KeyboardInterrupt:
    pass
finally:
    print(f"[synth-human] frames {n} in {time.time() - T0:.1f} s", flush=True)
    if a.log and LOG["wall"]:
        np.savez(a.log, **{k: np.asarray(v) for k, v in LOG.items()}, rad=rad, seg=seg, t_start_mono=t_start if t_start is not None else np.nan)
    del B; shm.close(); shm.unlink()
