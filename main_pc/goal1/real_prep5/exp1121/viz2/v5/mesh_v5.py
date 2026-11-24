"""mesh_v5.py - viz2 v5 (2026-11-23): SMPL vertices for every human record of a run (real, replay, or simulation with the recorded person),
from the REAL run's sensing session (pose_npz rows, sensing_perf publish times) and betas, as viz2/v4 render_v4.py (same SMPL wrapper, CPU).
Record -> real record with the same msg_com (the replay / simulation hands over the recorded records unchanged) -> its message stamp ->
sensing perf row (publish time within 10 ms, as render_v4 msg_frames) -> SMPL (camera frame) -> world = Pc R_C + t_C (the real run's
camera pose yaml; world = human node world, the frame of the precomputed human file). Records without a match keep the previous mesh.
Writes <out> = npz verts (F, 6890, 3) float32, faces, matched (F,) bool."""
import json, os, sys
import numpy as np
P4 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep4")


def _rot(cp):
    import math
    r, p, y = [math.radians(cp[k]) for k in ("roll_deg", "pitch_deg", "yaw_deg")]
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]); ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]); return rx @ ry @ rz


def verts_for(real, sensing, betas, run, out, smpl_pkg=os.environ.get("HRC_SMPLX_PKG", "~/hrc_smplx_pkg"), body_models=os.environ.get("HRC_BODY_MODELS", "~/hrc_ws/src/body_models")):
    import yaml
    real = real if os.path.isabs(real) else os.path.join(P4, real)
    zr = np.load(real + "_human.npz"); zx = np.load(run + "_human.npz")
    key = lambda c: np.round(np.asarray(c, np.float64), 6).tobytes()
    rmap = {}
    for j, c in enumerate(zr["msg_com"]):
        rmap.setdefault(key(c), j)
    st = zr["msg_stamp_ns"].astype(np.float64) / 1e9
    perf = [l.split(",") for l in open(os.path.join(sensing, "sensing_perf", "paper_human_sensing_perf.csv")).read().splitlines()[1:]]
    PT = np.array([float(p[0]) for p in perf]); pzd = os.path.join(sensing, "pose_npz"); pz = np.load(os.path.join(pzd, sorted(os.listdir(pzd))[-1]))
    POSES, TRANS = pz["poses"], pz["trans"]; n_ = min(len(PT), len(POSES)); PT = PT[:n_]
    cp = yaml.safe_load(open(real + "_camera_pose.yaml")); RC = _rot(cp); TC = np.array(cp["t"], float)
    sys.path.insert(0, os.path.expanduser(smpl_pkg))
    import torch
    from smplx_kinect.common.body_models import SMPLWrapper
    SW = SMPLWrapper(os.path.expanduser(body_models), torch.device("cpu")); B = np.load(betas).astype(np.float32)
    F = len(zx["msg_com"]); V = None; out_v = []; ok = np.zeros(F, bool); cache = {}
    for i, c in enumerate(zx["msg_com"]):
        j = rmap.get(key(c)); row = None
        if j is not None:
            rw = int(np.clip(np.searchsorted(PT, st[j] - 0.002), 0, n_ - 1))
            if abs(PT[rw] - st[j]) < 0.01:
                row = rw
        if row is not None:
            if row not in cache:
                p = POSES[row]; bp = np.zeros(69, np.float32); bp[:63] = p[3:66]
                with torch.no_grad():
                    o = SW.get_output("male", B, bp, p[:3], TRANS[row].copy())
                cache[row] = (o.vertices[0].detach().cpu().numpy().astype(np.float64) @ RC + TC).astype(np.float32)
            V = cache[row]; ok[i] = True
        out_v.append(V)
    first = next((v for v in out_v if v is not None), None)
    if first is None:
        raise SystemExit("mesh_v5: no record of %s matched a sensing pose row of %s" % (run, real))
    out_v = [first if v is None else v for v in out_v]
    np.savez(out, verts=np.stack(out_v), faces=np.asarray(SW.models["male"].faces, np.int64), matched=ok)
    return dict(file=out, records=F, matched=int(ok.sum()), real=os.path.basename(real), sensing=sensing, betas=betas)
