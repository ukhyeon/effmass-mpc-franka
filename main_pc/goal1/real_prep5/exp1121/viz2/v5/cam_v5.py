"""cam_v5.py - viz2 v5 (2026-11-23): the real colour camera as an Isaac (USD) camera for one run.
Pose: the run's <run>_camera_pose.yaml (yaw, t) with roll / pitch from the sensing session's newest debug_log/camera_pose_*.yaml (else the
run's); world (human node frame) = robot base + (0, 0, 0.715); OpenCV camera Pc = R_C (Pw - t). Intrinsics: exp1121/viz/calib/device_intr.json
(colour K; the distortion is applied to the pinhole render afterwards with the same map as the SMPL layer, so the render matches the raw frame).
Prints one line: V5_CAM_M (16, row-major local->world for USD row vectors, base frame) V5_CAM_AP (focal, h / v aperture, h / v offset, mm)
eye (base frame) target (1 m ahead). Usage: cam_v5.py <run prefix> <sensing dir> [W H]"""
import glob, json, math, os, sys
import numpy as np, yaml
run, sens = sys.argv[1], sys.argv[2]; W, H = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1280, 720)
cp = yaml.safe_load(open(run + "_camera_pose.yaml")); fs = sorted(glob.glob(os.path.join(sens, "debug_log", "camera_pose_*.yaml")))
if fs:
    cs = yaml.safe_load(open(fs[-1])); cp["roll_deg"], cp["pitch_deg"] = cs["roll_deg"], cs["pitch_deg"]
def R(t, ax):
    c, s = math.cos(t), math.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[ax]
RC = R(math.radians(cp["roll_deg"]), "x") @ R(math.radians(cp["pitch_deg"]), "y") @ R(math.radians(cp["yaw_deg"]), "z"); TC = np.array(cp["t"], float)
D = np.diag([1.0, -1.0, -1.0]); Ru = RC.T @ D; t = TC - np.array([0, 0, 0.715])   # USD local axes (x right, y up, looks -z) -> base frame
M = np.eye(4); M[:3, :3] = Ru.T; M[3, :3] = t
K = np.array(json.load(open(os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121/viz/calib/device_intr.json")))["K"]); fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; f = 24.0
ap = [f, W * f / fx, H * f / fy, (W / 2.0 - cx) * f / fx, (cy - H / 2.0) * f / fy]
fwd = RC.T @ np.array([0, 0, 1.0])
print(",".join("%.9g" % v for v in M.reshape(-1)), ",".join("%.6g" % v for v in ap), ",".join("%.4f" % v for v in t), ",".join("%.4f" % v for v in t + fwd))
