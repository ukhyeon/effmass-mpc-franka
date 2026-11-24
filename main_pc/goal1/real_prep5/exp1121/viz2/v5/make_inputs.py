"""make_inputs.py - viz2 v5 (2026-11-23): converts one rp2 run (real, replay or lockstep simulation; exp_ana Run, read only) into the
inputs of the Isaac re-render (step2b --playback-q / --human-source precomputed / g1_viz_v5 line):
  <out>/human_v2.npz      precomputed human: one frame per human record (msg_com / vel / inv / p0 / p1, node world frame), shapes static
  <out>/human_verts.npz   (only with --mesh-real R --sensing S --betas B) SMPL vertices per record from the real run's sensing (world frame)
  <out>/step_frames.npy   human frame shown at step i = the record the MPC used at step i + 1 (frame i is captured before the human of
                          step i is pumped), jump-filter rejected records -> the last accepted one (as exp_ana)
  <out>/q.npz             q (n, 7) = steps meas_q (60 Hz = step2b steps)
  <out>/lines.npz         a = robot sphere surface point, b = arm surface point of the closest (robot sphere, arm capsule) pair, d = distance m
                          (exp_ana arm-only gap + 0.02 = MPC distance), per step; arm parts = exp_ana human_arm_parts
  (lines.npz also: ang = angle of that vector to the forearm axis of the same arm side, folded 0-90 deg; seg = arm segment of the pair)
  <out>/meta.json         method, d0 / gain / eta_max (summary cost; exp.json values when present), window, closest distance and time
(baseex 11-24) --base-exclude <yaml> (goal1/config/base_exclude.yaml; default "" = off = as before): a human record whose pelvis COM
(segment 0, node world) is inside the robot base / cart box (goal1/scripts/rp5_base_exclude.py) is not the person: steps showing it show
the last record before it that is not inside (step_frames.npy remapped; mesh and shapes follow); with no such record yet the step shows
an extra record F (a copy of record 0 moved 100 m down = no person in view), appended to human_v2.npz / human_verts.npz. meta base_exclude.
(robex 11-24) lines.npz leaves out the robot spheres of link0 / link1 (robex_v5.py; key robot_exclude); the MPC distance is unchanged.
(robex2 11-24) the links follow the run's record human_robot_exclude when present (robex_v5.links).
Usage: make_inputs.py <run> <out> [--window a,b] [--base-exclude yaml]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121"))
import exp_ana as X   # noqa: E402  (read only)
M = X.M
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--window", default="")
ap.add_argument("--mesh-real", default="", help="real run whose sensing gives the SMPL mesh (time = seconds since each run's MPC start)")
ap.add_argument("--sensing", default=""); ap.add_argument("--betas", default="")
ap.add_argument("--base-exclude", default="", help="(baseex 11-24) yaml of the robot base / cart box; empty = off")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
r = X.Run(A.run); z = r.H; n = r.n
F = len(z["msg_com"]); com = z["msg_com"].astype(np.float32).reshape(F, 19, 3); vel = z["msg_vel"].astype(np.float32).reshape(F, 19, 3)
inv = z["msg_inv"].astype(np.float32).reshape(F, 19, 3, 3); p0 = z["msg_p0"].astype(np.float32).reshape(F, 24, 3); p1 = z["msg_p1"].astype(np.float32).reshape(F, 24, 3)
rad = z["msg_rad"].reshape(F, 24); seg = z["msg_seg"].reshape(F, 24); typ = z["msg_type"].reshape(F, 24)
assert np.allclose(rad, rad[0]) and (seg == seg[0]).all(), "shape radius / segment change inside the run"
T0 = float(z["step_t_pub_ns"][0]) / 1e9; it = z["msg_stamp_ns"].astype(np.float64) / 1e9 - T0
np.savez(os.path.join(A.out, "human_v2.npz"), com=com, vel=vel, inv=inv, p0=p0, p1=p1, input_time=it, rate=np.array(30.0), shape_radius=rad[0].astype(np.float32),
         shape_segment=seg[0].astype(np.int64), shape_type=typ[0].astype(np.uint8), frame=np.arange(F), source=np.array(r.p), betas=np.zeros(10, np.float32), camera=np.zeros(6))
hidx = np.asarray(r.hidx, int); sf = np.r_[hidx[1:], hidx[-1:]]; sf = np.where(sf < 0, max(int(hidx[hidx >= 0][0]) if (hidx >= 0).any() else 0, 0), sf)
np.save(os.path.join(A.out, "step_frames.npy"), sf.astype(np.int64)); np.savez(os.path.join(A.out, "q.npz"), q=r.q)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import robex_v5   # noqa: E402  (robex 11-24) link0 / link1 spheres left out of the shown pair
robex_v5.apply(r); LN = robex_v5.lines(r); d = LN["d"]; print("[v5 inputs] " + r.robex["text"])
np.savez(os.path.join(A.out, "lines.npz"), **LN)
c = (r.S.get("human") or {}).get("cost") or {}; d0 = float(c.get("eta0", r.d_base)); kg = float(c.get("p", r.k)); em = float(c.get("eta_max", 0.25))
kind = "uniform" if kg == 0 else "effmass"
if A.window:
    wa, wb = [float(v) for v in A.window.split(",")]
else:
    X.WIN_R = 0.35; w = r.windows(); wa, wb = (float(r.t[w[0][0]]), float(r.t[w[0][1]])) if w else (0.0, float(r.t[-1]))
m = (r.t >= wa) & (r.t <= wb); k = int(np.nonzero(m)[0][np.nanargmin(d[m])])
mesh = None
if A.mesh_real:   # SMPL vertices per record at the same seconds since the MPC start in the real run (as render_v4 for replays)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mesh_v5
    mesh = mesh_v5.verts_for(A.mesh_real, A.sensing, A.betas, r.p, os.path.join(A.out, "human_verts.npz")); print("[v5 inputs] mesh %s" % mesh)
bex = None
if A.base_exclude:   # (baseex 11-24) records with the pelvis on the robot cart -> the last record not on it, none -> a record out of view
    sys.path.insert(0, os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/scripts")); import rp5_base_exclude as RBE
    BX = RBE.BaseExclude(A.base_exclude); exr = BX.inside(com[:, 0].astype(np.float64)); km = RBE.last_kept(exr); sf2 = km[sf]; hide = sf2 < 0
    if hide.any():   # extra record F = record 0 moved 100 m down (the renderer draws it out of view)
        dn = np.array([0, 0, -100.0], np.float32); hz = dict(np.load(os.path.join(A.out, "human_v2.npz")))
        for kk in ("com", "p0", "p1"):
            hz[kk] = np.concatenate([hz[kk], hz[kk][:1] + dn]).astype(np.float32)
        for kk in ("vel", "inv"):
            hz[kk] = np.concatenate([hz[kk], np.zeros_like(hz[kk][:1]) if kk == "vel" else hz[kk][:1]])
        hz["input_time"] = np.r_[hz["input_time"], hz["input_time"][:1]]; hz["frame"] = np.arange(F + 1)
        np.savez(os.path.join(A.out, "human_v2.npz"), **hz)
        hvf = os.path.join(A.out, "human_verts.npz")
        if os.path.exists(hvf):
            hv = dict(np.load(hvf)); hv["verts"] = np.concatenate([hv["verts"], hv["verts"][:1] + dn]).astype(np.float32); hv["matched"] = np.r_[hv["matched"], False]; np.savez(hvf, **hv)
        sf2 = np.where(hide, F, sf2)
    np.save(os.path.join(A.out, "step_frames.npy"), sf2.astype(np.int64))
    ep = RBE.episodes([float(r.t[i]) for i in np.nonzero(exr[sf])[0]], gap=0.2)
    bex = dict(BX.describe(), records=int(F), records_inside=int(exr.sum()), steps_remapped=int(exr[sf].sum()), steps_no_person=int(hide.sum()),
               episodes_s=[[round(a_, 2), round(b_, 2), int(c_)] for a_, b_, c_ in ep], note="steps whose shown record had the pelvis on the cart -> last record not on it")
    print("[v5 inputs] base exclude %s: %d of %d records inside, %d steps remapped, %d steps with no person; steps s %s" % (BX.path, exr.sum(), F, exr[sf].sum(), hide.sum(),
          " ".join("%.2f-%.2f(%d)" % tuple(e) for e in bex["episodes_s"]) or "none"))
json.dump(dict(run=r.p, name=r.name, kind=kind, d0=d0, gain=kg, eta_max=em, sim=not r.name.startswith("real_"), window=[wa, wb], closest_m=float(d[k]), t_closest=float(r.t[k]),
               k_closest=k, robot_exclude=dict(links=robex_v5.key(r), text=r.robex["text"]), steps=int(n), records=int(F), side=None, mesh=mesh, **({"base_exclude": bex} if bex is not None else {})), open(os.path.join(A.out, "meta.json"), "w"), indent=1)
print("[v5 inputs] %s: %s d0 %.2f gain %.2f eta_max %.2f, %d steps, %d records, closest %.1f cm at %.2f s (window %.1f-%.1f)" % (r.name, kind, d0, kg, em, n, F, 100 * d[k], r.t[k], wa, wb))
