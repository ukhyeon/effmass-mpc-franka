"""mosaic_v5.py - viz2 v5 (2026-11-24): privacy mosaic of the RGB panel defined as 3-D rectangles in the robot base frame
(mosaic_regions.yaml: name + 4 corners, m; world = base + (0, 0, 0.715)) and projected for every run with ITS camera pose.
Library (compose_single.py): load(yaml) -> regions; boxes(regions, b2c, pix, W, H, margin) -> [(name, x0, y0, x1, y1)], skipped names.
  b2c / pix are compose_single's own functions (base -> colour camera from <run>_camera_pose.yaml, cv2.projectPoints with the colour K and
  distortion of viz/calib/device_intr.json = the camera model of the SMPL layer and the distance line). A region is skipped when a corner
  is not in front of the camera (Z <= 0.05 m) or its box is fully outside the image; otherwise box = bounding box of the 4 projected
  corners + margin px, clipped to the image.
Init (how the yaml was made): mosaic_v5.py init <reference run prefix> <out yaml>
  The 11-23 pixel boxes (fitted on the frames before the camera tilt, run t6) are back-projected with the reference run's camera pose
  (same R_C, t as compose_single, inverse direction): no depth was recorded (the sensing sessions keep RGB only), so each region lies on
  a vertical plane facing the camera (normal = horizontal part of the optical axis) through the point where the ray of a hand-picked
  "foot" pixel (where the object stands on its table) meets the table top z = DESK_Z (world). For two runs with the same camera t
  (only roll / pitch changed) the projected box does not depend on that depth."""
import json, math, os, sys
import numpy as np
BASE_Z = 0.715
DESK_Z = 0.73   # assumed table-top height (world z, m) for all three objects
INIT = [   # name, pixel box x0,y0,x1,y1 (compose_single --blur 11-23), foot pixel (object on its table) in the same frames
    ("monitors_main", (278, 270, 366, 362), (322, 375)),
    ("monitor_side", (250, 268, 290, 418), (270, 418)),
    ("laptop", (530, 296, 598, 346), (565, 346)),
]


def load(path):
    import yaml
    Y = yaml.safe_load(open(path))
    return [(r["name"], np.asarray(r["corners"], float).reshape(4, 3)) for r in Y["regions"]]


def boxes(regions, b2c, pix, W, H, margin=20):
    out, skip = [], []
    for name, C in regions:
        Pc = b2c(C)
        if np.any(Pc[:, 2] <= 0.05):
            skip.append(name + " (behind the camera)"); continue
        uv = pix(Pc); x0, y0 = uv.min(0) - margin; x1, y1 = uv.max(0) + margin
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            skip.append(name + " (outside the image)"); continue
        out.append((name, int(max(math.floor(x0), 0)), int(max(math.floor(y0), 0)), int(min(math.ceil(x1), W)), int(min(math.ceil(y1), H))))
    return out, skip


def _camera(run):   # as compose_single.py: R_C = Rx(roll) Ry(pitch) Rz(yaw), Pc = R_C (Pw - t); colour K + distortion
    import yaml
    cp = yaml.safe_load(open(run + "_camera_pose.yaml"))
    def _r(t, ax):
        c, s = math.cos(t), math.sin(t)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[ax]
    RC = _r(math.radians(cp["roll_deg"]), "x") @ _r(math.radians(cp["pitch_deg"]), "y") @ _r(math.radians(cp["yaw_deg"]), "z")
    DI = json.load(open(os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121/viz/calib/device_intr.json")))
    return cp, RC, np.array(cp["t"], float), np.array(DI["K"]), np.array(DI["dist"])


def init(run, out):
    import cv2
    cp, RC, TC, K, DD = _camera(run)
    def ray(u, v):
        n = cv2.undistortPoints(np.array([[[u, v]]], float), K, DD).reshape(2); return RC.T @ np.array([n[0], n[1], 1.0])
    fw = RC.T @ np.array([0, 0, 1.0]); nh = np.array([fw[0], fw[1], 0.0]); nh /= np.linalg.norm(nh)
    L = ["# mosaic_regions.yaml - viz2 v5 privacy mosaic (2026-11-24), made by: mosaic_v5.py init %s" % os.path.basename(run),
         "# frame: robot base (world = base + (0, 0, %.3f)), metres; corners in order top-left, top-right, bottom-right, bottom-left" % BASE_Z,
         "# as seen from the camera. Source: the 11-23 pixel boxes (fitted before the camera tilt) back-projected with the camera pose of",
         "# %s (roll %.4f, pitch %.4f, yaw %.1f, t %s)." % (os.path.basename(run), cp["roll_deg"], cp["pitch_deg"], cp["yaw_deg"], list(TC)),
         "# No depth recorded: each region is a vertical plane facing the camera through the point where its foot pixel ray meets z = %.2f m" % DESK_Z,
         "# (world, assumed table top). Edit corners freely (e.g. measured positions); compose_single.py projects them per run, margin 20 px.",
         "frame: robot_base", "regions:"]
    for name, (x0, y0, x1, y1), (fu, fv) in INIT:
        d = ray(fu, fv); s = (DESK_Z - TC[2]) / d[2]; F = TC + s * d; Z = float((RC @ (F - TC))[2]); P = []
        for (u, v) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            r = ray(u, v); P.append(TC + (nh @ (F - TC)) / (nh @ r) * r - [0, 0, BASE_Z])
        L.append("  - name: %s   # pixel box %d,%d,%d,%d; foot pixel %d,%d -> camera depth %.2f m" % (name, x0, y0, x1, y1, fu, fv, Z))
        L.append("    corners:")
        L += ["      - [%.3f, %.3f, %.3f]" % tuple(p) for p in P]
    open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "init":
        init(sys.argv[2], sys.argv[3])
    else:
        print(__doc__); sys.exit(1)
