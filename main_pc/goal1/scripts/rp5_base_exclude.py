"""rp5_base_exclude.py - baseex (2026-11-24): robot base / cart exclusion zone for person candidates (the camera Body Tracking sometimes
takes the robot on its cart as a person; nobody can stand on the cart). One rule, used by the sensing body choice
(real_prep5/night/rp5_sensing_bodies.py, env RP5_BASE_EXCLUDE), the MPC human reader (rp2_mpc_rt.py --human-base-exclude) and the
visualisation (viz2/v5 mesh_v5.py, compose_single.py):
  a person candidate is dropped when its PELVIS (root) lies inside the box  x0 - margin <= x <= x1 + margin, y0 - margin <= y <= y1 + margin
  (world frame of goal1/config/camera_pose_rp5.yaml = hrc node world: floor z = 0, origin below the robot base centre, axes = robot base
  axes; the robot base frame has the same x, y). Height is not limited. Hands / arms never count, only the pelvis.
File (yaml): frame: world, x: [x0, x1], y: [y0, y1], margin: m. See goal1/config/base_exclude.yaml.
Helpers: k4a_pelvis_world (pyk4a body list -> pelvis in world, with the camera depth -> colour extrinsics and the session camera pose),
camera_rot (hrc_ws camera_to_world_rotmat convention, re-written here: hrc_ws is not imported). ASCII."""
import json, math, os
import numpy as np

ROOT = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "")


def resolve(p):
    """path as given, else relative to ~/effmass_mpc, else relative to ~/effmass_mpc/goal1"""
    p = os.path.expandvars(os.path.expanduser(str(p)))
    for c in (p, os.path.join(ROOT, p), os.path.join(ROOT, "goal1", p)):
        if os.path.isfile(c):
            return os.path.realpath(c)
    raise FileNotFoundError("base exclude: no file %s (looked as given, under %s and %s/goal1)" % (p, ROOT, ROOT))


class BaseExclude:
    def __init__(self, path):
        import yaml
        self.path = resolve(path); d = yaml.safe_load(open(self.path))
        if str(d.get("frame", "world")) != "world":
            raise ValueError("%s: frame must be world" % self.path)
        self.x = [float(v) for v in d["x"]]; self.y = [float(v) for v in d["y"]]; self.margin = float(d.get("margin", 0.05))
        if not (self.x[0] < self.x[1] and self.y[0] < self.y[1] and self.margin >= 0):
            raise ValueError("%s: need x0 < x1, y0 < y1, margin >= 0" % self.path)
        self.lo = np.array([self.x[0] - self.margin, self.y[0] - self.margin]); self.hi = np.array([self.x[1] + self.margin, self.y[1] + self.margin])
        self.d = dict(d)

    def inside(self, p):
        """p (..., >= 2) world pelvis -> bool (...); NaN -> False"""
        q = np.asarray(p, np.float64)[..., :2]
        return np.all((q >= self.lo) & (q <= self.hi), axis=-1)

    def describe(self):
        return dict(file=self.path, frame="world", x=self.x, y=self.y, margin_m=self.margin, box_x=self.lo[0:1].tolist() + self.hi[0:1].tolist(),
                    box_y=self.lo[1:2].tolist() + self.hi[1:2].tolist(), rule="pelvis (root) inside the box -> not a person")


def camera_rot(roll_deg, pitch_deg, yaw_deg):
    """cam_R = Rx(roll) Ry(pitch) Rz(yaw); p_world = cam_R.T @ p_colour + t (goal1/config/camera_pose_rp5.yaml)"""
    r, p, y = [math.radians(float(v)) for v in (roll_deg, pitch_deg, yaw_deg)]
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rx @ ry @ rz


def load_extrinsics(path=None):
    """camera depth -> colour (R, t m) of this device (goal1/real_prep5/exp1121/viz/calib/device_intr.json, from its factory calibration)"""
    d = json.load(open(resolve(path or "goal1/real_prep5/exp1121/viz/calib/device_intr.json")))
    return np.array(d["depth_to_colour_R"], float), np.array(d["depth_to_colour_t_m"], float)


def load_camera_pose(path):
    """session camera pose yaml (<sensing>/debug_log/camera_pose_*.yaml or <run>_camera_pose.yaml) -> (R_cam, t)"""
    import yaml
    d = yaml.safe_load(open(path))
    return camera_rot(d["roll_deg"], d["pitch_deg"], d["yaw_deg"]), np.array(d["t"], float)


def k4a_pelvis_world(pose, R_dc, t_dc, R_cam, t_cam):
    """pyk4a pose (n, 32, 10) (x, y, z mm in the depth camera, cols 2:5; joint 0 = pelvis) -> (n, 3) world"""
    p = np.asarray(pose, np.float64)
    if len(p) == 0:
        return np.zeros((0, 3))
    pc = (p[:, 0, 2:5] / 1000.0) @ R_dc.T + t_dc
    return pc @ R_cam + t_cam   # = (R_cam.T @ pc.T).T + t


def last_kept(excluded):
    """bool (n,) excluded -> int (n,): index of the last NOT excluded item at or before i (the person kept), -1 if none yet"""
    out = np.full(len(excluded), -1, np.int64); last = -1
    for i, e in enumerate(np.asarray(excluded, bool)):
        if not e:
            last = i
        out[i] = last
    return out


def smpl_pelvis_world(trans, j0, R_cam, t_cam):
    """SMPL root (pelvis) joint = transl + J0(betas) for any pose (camera frame) -> world: p @ R_cam + t (as viz2 v5 mesh_v5)"""
    return (np.asarray(trans, np.float64) + np.asarray(j0, np.float64)) @ R_cam + t_cam


def episodes(times, gap=0.5):
    """sorted times -> [[t_first, t_last, count]] (a new episode after a gap > gap s)"""
    out = []
    for t in times:
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t; out[-1][2] += 1
        else:
            out.append([t, t, 1])
    return out
