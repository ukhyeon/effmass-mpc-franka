"""compose_single.py - viz2 v5 single-run mode (2026-11-23, matched view): one REAL run, two panels at the same time, each 1280 x 720:
  left : the real RGB frame (sensing vis_output, raw) at the run time t + 80 ms (camera -> host delay, as render_v4), privacy blur as v3 /
         v4 (time strip, --blur boxes), and the grey SMPL mesh of THAT frame drawn by the hrc_ws sensing renderer (VisualizerMeshSMPL,
         render mode full_smpl_collision_top3, colours / shading unchanged; pinhole layer remapped to the raw distortion), as v3_A. GPU.
  right: the Isaac re-render of the run seen from the REAL colour camera (render_v5.sh with cam_v5.py: same pose and K; the pinhole render is
         remapped with the same distortion map), labels "<method> (real robot)", "min distance N cm", at the line "N cm, A deg" (distance now,
         angle to the forearm axis). With --lookat 1 (run_v5 --cam) the right panel is not remapped and the line uses the look-at camera.
Outputs <out>/<name>_closest.png (2572 wide), <out>/<name>.mp4 (H.264, full size, 15 frames/s = a frame every 4 steps over --gif a,b,
the whole window), <out>/<name>.gif (1280 wide, 10 frames/s = a frame every 6 steps, closest step +-4 s; if over --gif-mb the span shrinks by
0.5 s per side, never the frame rate or the width), <out>/<name>.json. (run_v5 passes name = <run>_<tag>.)
(v5b, 11-23) Render grid --every 2 (step2b frame f<i> = step t0*60 + i). The distance "N cm, N deg" is a fixed box under the title box (it
covered the gripper and the fist next to the line); the scene keeps only a thin line between the two closest points.
(text 11-23) Top-left: ONE box, line 1 = method ("Effective mass" / "Uniform", method colour), line 2 = "gap N cm  (+E cm)": N = the
distance of this frame (lines.npz d, integer cm, sign kept), E = eta - d_base of the closest arm part (eta_v5.py; d_base from the run's
exp.json), only for an effective-mass run. No angle, no "min distance" line. Captions under the panels unchanged.
(mosaic 11-24) The privacy mosaic of the left panel = the 3-D regions of mosaic_regions.yaml (robot base frame) projected with THIS
run's _camera_pose.yaml and the colour K / distortion (b2c + pix below, the camera model of the line), bounding box + 20 px (mosaic_v5.py);
a run without _camera_pose.yaml falls back to the fixed pixel boxes --blur with a warning.
(win2 11-24) --mp4 a,b: the MP4 span (run_v5 passes the whole render range; the GIF stays the closest step +-4 s). An MP4 over 12 MB is
re-encoded with a higher crf only (mp4_v5.shrink; json mp4_crf when not 20). Part name of the closest pair from the 19-segment names.
(cmpview 11-24) --rgb-smpl on|off (default on = as before): off = the left RGB panel without the SMPL mesh (json rgb_smpl).
(baseex 11-24) --base-exclude <yaml> (default "" = off): a sensing SMPL row whose pelvis (root = transl + J0(betas), camera -> world with
this run's camera pose) is inside the robot base / cart box (goal1/scripts/rp5_base_exclude.py) is not drawn; the left panel shows the
last row before it that is not inside, or no mesh if there is none. Json base_exclude.
(wrap 11-24) --paper 0 by default (no clock / frame-number blur at the bottom left; --paper 1 = as before). --at T: PNG (from the main
render, the rendered step nearest T) and GIF around T (run_v5: the step with the largest eta extra among steps with the arm within 0.20 m);
the MP4 stays the whole run. Future path: path_v5 (the MPC plan if the steps npz has plan_q / plan_t, else the recorded TCP), caption
"path 0.5 s" / "planned path 0.5 s" in the footer, json future_path / future_source.
(final 11-24) Left panel: no privacy mosaic by default (--mosaic off); --mosaic on = the 3-D regions of mosaic_regions.yaml as before
(none = the fixed pixel boxes, <path> = another regions yaml). The time-strip blur (--paper) is unchanged. Right panel: the TCP future
path of the next --future 0.5 s as a dark red line (path_v5.RGB; cyan before pathred 11-24) fading toward its end (no MPC plan is recorded -> the recorded TCP, exp_ana tcp_of(meas_q);
the source is logged and in the json), and a 3-line box: method / "<closest arm part> <m_h> kg" / "gap N cm  (+E cm)"; part, m_h and
extra from the same step, part and u (label_v5.py, recorded mass mode and ISO override); --label-avg S averages m_h and extra over S s
(restart at a part change; default 0 = none, the values do not jitter); gap = lines.npz d as before; uniform: no bracket."""
import argparse, json, math, os, subprocess, sys, time
import numpy as np, cv2, yaml
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--work", required=True); ap.add_argument("--sensing", required=True)
ap.add_argument("--betas", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--still-tag", required=True); ap.add_argument("--still-span", type=int, default=30)
ap.add_argument("--t0", type=float, required=True); ap.add_argument("--every", type=int, required=True); ap.add_argument("--cam", default=""); ap.add_argument("--lookat", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--name", required=True); ap.add_argument("--gif", default=""); ap.add_argument("--gif-width", type=int, default=1280)
ap.add_argument("--gif-mb", type=float, default=10.0); ap.add_argument("--gif-half", type=float, default=4.0); ap.add_argument("--mp4-every", type=int, default=4); ap.add_argument("--gif-every", type=int, default=6); ap.add_argument("--delay-ms", type=float, default=80.0); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--paper", type=int, default=0); ap.add_argument("--blur", default="278,270,366,362;250,268,290,418;530,296,598,346", help="x0,y0,x1,y1;... pixel boxes (11-23 layout: monitors, laptop; used only for a run without _camera_pose.yaml)")
ap.add_argument("--mp4", default="", help="(win2 11-24) MP4 span a,b s (run_v5: the whole render range); empty = --gif")
ap.add_argument("--at", default="", help="(wrap 11-24) representative moment, s (run_v5: largest eta extra with the arm within 0.20 m); PNG / GIF there; empty = the closest step (still render)")
ap.add_argument("--rgb-smpl", default="on", choices=["on", "off"], help="(cmpview 11-24) off: no SMPL mesh on the left RGB panel (mosaic, text, 3-D panel unchanged)")
ap.add_argument("--base-exclude", default="", help="(baseex 11-24) yaml of the robot base / cart box for the left SMPL; empty = off")
ap.add_argument("--mosaic", default="off", help="(final 11-24) off (default) = no privacy mosaic; on = 3-D regions <v5>/mosaic_regions.yaml (mosaic 11-24); none = pixel --blur boxes; <yaml> = other regions"); ap.add_argument("--mosaic-margin", type=int, default=20)
ap.add_argument("--future", type=float, default=0.5, help="(final 11-24) TCP future path length, s (0 = none)"); ap.add_argument("--label-avg", type=float, default=0.0, help="(final 11-24) moving average of m_h / extra, s (0 = none: on t17 the step-to-step change is p95 0.04 kg / 0.05 cm, an average only adds lag)")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
E1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121"); HERE = os.path.dirname(os.path.abspath(__file__))
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COL = {"uniform": (25, 95, 215), "effmass": (240, 110, 10)}; W_, H_ = 1280, 720; FS = H_ / 480.0
m = json.load(open(os.path.join(A.work, "meta.json"))); L = np.load(os.path.join(A.work, "lines.npz")); kind = m["kind"]
R = np.load(A.run + "_steps.npz"); T0 = float(R["t_pub_ns"][0]) / 1e9
CPF = A.run + "_camera_pose.yaml"   # (mosaic 11-24) an old run without it: pixel blur boxes (and the look-at view only)
if not os.path.exists(CPF) and not A.lookat:
    raise SystemExit("no %s: the real-camera view needs it (use --cam)" % CPF)
cp = yaml.safe_load(open(CPF)) if os.path.exists(CPF) else dict(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, t=[0.0, 0.0, 0.0])
def _r(t, ax):
    c, s = math.cos(t), math.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[ax]
RC = _r(math.radians(cp["roll_deg"]), "x") @ _r(math.radians(cp["pitch_deg"]), "y") @ _r(math.radians(cp["yaw_deg"]), "z"); TC = np.array(cp["t"], float)
DI = json.load(open(os.path.join(E1, "viz", "calib", "device_intr.json"))); K = np.array(DI["K"]); DD = np.array(DI["dist"])
gu, gv = np.meshgrid(np.arange(W_, dtype=np.float32), np.arange(H_, dtype=np.float32))
_und = cv2.undistortPoints(np.stack([gu.ravel(), gv.ravel()], 1).reshape(-1, 1, 2), K, DD, P=K).reshape(H_, W_, 2).astype(np.float32); MAPX, MAPY = _und[..., 0].copy(), _und[..., 1].copy()
def b2c(Pb): return (np.asarray(Pb, float).reshape(-1, 3) + [0, 0, 0.715] - TC) @ RC.T   # robot base -> colour camera (OpenCV)
def pix(Pc):
    uv, _ = cv2.projectPoints(np.asarray(Pc, float).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DD); return uv.reshape(-1, 2)
vt = [l.strip().split(",") for l in open(os.path.join(A.sensing, "vis_times.csv")).read().splitlines()[1:]]
VF = [v[0] for v in vt]; VFi = np.array([int(v[0][:-4]) for v in vt]); VW = np.array([float(v[1]) for v in vt])
pzd = os.path.join(A.sensing, "pose_npz"); pz = np.load(os.path.join(pzd, sorted(os.listdir(pzd))[-1])); POSES, TRANS = pz["poses"], pz["trans"]
perf = [l.split(",") for l in open(os.path.join(A.sensing, "sensing_perf", "paper_human_sensing_perf.csv")).read().splitlines()[1:]]
PF = np.array([int(p[1]) for p in perf])[:min(len(perf), len(POSES))]; ROW = {int(f): j for j, f in enumerate(PF)}
sys.path.insert(0, os.path.expanduser(os.environ.get("HRC_SMPLX_PKG", "~/hrc_smplx_pkg")))
import torch
from smplx_kinect.common.body_models import SMPLWrapper
from smplx_kinect.common.smplx_vis import VisualizerMeshSMPL
DEV = torch.device(A.device); SW = SMPLWrapper(os.path.expanduser(os.environ.get("HRC_BODY_MODELS", os.environ.get("HRC_BODY_MODELS", "~/hrc_ws/src/body_models"))), DEV); BET = np.load(A.betas).astype(np.float32)
BXK = None; BXI = None   # (baseex 11-24) pose row -> row drawn (last row with the pelvis not on the cart), -1 = no mesh
if A.base_exclude:
    sys.path.insert(0, os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/scripts")); import rp5_base_exclude as RBE   # noqa: E402
    _BX = RBE.BaseExclude(A.base_exclude)
    with torch.no_grad():
        _J0 = SW.get_output("male", BET, np.zeros(69, np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32)).joints[0, 0].detach().cpu().numpy()
    _ex = _BX.inside(RBE.smpl_pelvis_world(TRANS[:len(PF)], _J0, RC, TC)); BXK = RBE.last_kept(_ex)
    BXI = dict(_BX.describe(), rows=int(len(_ex)), rows_inside=int(_ex.sum()), rows_with_no_earlier_person=int((_ex & (BXK < 0)).sum()), drawn_replaced=0, drawn_none=0)
    print("[compose_single] base exclude %s: %d of %d SMPL rows have the pelvis on the cart" % (_BX.path, _ex.sum(), len(_ex)))
VIS = VisualizerMeshSMPL(K=K, scale=1, size={"h": H_, "w": W_}, device=A.device, smpl_wrapper=SW, sides=False, normals=True)
BLUR = [tuple(int(v) for v in b.split(",")) for b in A.blur.split(";") if b.strip()]
MOSF = os.path.join(HERE, "mosaic_regions.yaml") if A.mosaic in ("on", "") else A.mosaic; MOS_SRC = "pixel boxes (--blur)"   # (mosaic 11-24) 3-D regions -> this run's pixels
if A.mosaic == "off":   # (final 11-24) default: no privacy mosaic
    BLUR = []; MOS_SRC = "off"
elif A.mosaic == "none":
    pass
elif not os.path.exists(CPF):
    print("[compose_single] WARNING: no %s -> privacy mosaic from the fixed pixel boxes %s (may miss the screens)" % (os.path.basename(CPF), A.blur))
elif not os.path.exists(MOSF):
    print("[compose_single] WARNING: no %s -> privacy mosaic from the fixed pixel boxes %s" % (MOSF, A.blur))
else:
    sys.path.insert(0, HERE); import mosaic_v5   # noqa: E402
    _mb, _ms = mosaic_v5.boxes(mosaic_v5.load(MOSF), b2c, pix, W_, H_, A.mosaic_margin); BLUR = [b[1:] for b in _mb]
    MOS_SRC = "%s projected with %s (+%d px)" % (os.path.basename(MOSF), os.path.basename(CPF), A.mosaic_margin)
    print("[compose_single] mosaic: " + ", ".join("%s %d,%d,%d,%d" % b for b in _mb) + ("; skipped " + ", ".join(_ms) if _ms else ""))


def rgb_left(k):   # raw frame + privacy blur + grey SMPL (hrc_ws renderer, as v3 smpl_layer)
    t = float(L["t"][k]); i = int(np.argmin(np.abs(VW - (T0 + t + A.delay_ms / 1000.0)))); img = cv2.imread(os.path.join(A.sensing, "vis_output", VF[i]))
    if A.paper:
        img[672:720, 0:360] = cv2.GaussianBlur(img[672:720, 0:360], (0, 0), 12)
    for (x0, y0, x1, y1) in BLUR:
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), 9)
    j = ROW.get(int(VFi[i])) if A.rgb_smpl == "on" else None   # (cmpview) --rgb-smpl off: raw frame only
    if j is not None and BXK is not None and BXK[j] != j:   # (baseex) pelvis on the cart: the last row not on it, or none
        BXI["drawn_replaced" if BXK[j] >= 0 else "drawn_none"] += 1; j = int(BXK[j]) if BXK[j] >= 0 else None
    if j is not None:
        p = POSES[j]; bp = np.zeros(69, np.float32); bp[:63] = p[3:66]
        with torch.no_grad():
            V = SW.get_output("male", BET, bp, p[:3], TRANS[j].copy()).vertices[0].detach().float().contiguous()
        lay = VIS.vertices2vis_test(V, vertex_colors=None, collision_vertex_colors=None, bg_img=np.zeros_like(img), render_mode="full_smpl_collision_top3")
        lay = cv2.remap(lay, MAPX, MAPY, cv2.INTER_LINEAR, borderValue=0); mk = np.any(lay > 10, axis=-1); img[mk] = lay[mk]
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), int(VFi[i])


if A.lookat:
    cv_ = [float(x) for x in A.cam.split(",")]; eye, tgt = np.array(cv_[:3]), np.array(cv_[3:6]); fw = (tgt - eye) / np.linalg.norm(tgt - eye)
    rt_ = np.cross(fw, [0, 0, 1.0]); rt_ /= np.linalg.norm(rt_); up_ = np.cross(rt_, fw)
def proj(Pb):
    if not A.lookat:
        return pix(b2c(Pb))[0]
    d = np.asarray(Pb, float) - eye; Zc = d @ fw; f = W_ * 24.0 / 20.955; return np.array([W_ / 2 + f * (d @ rt_) / Zc, H_ / 2 - f * (d @ up_) / Zc])


def box(d, xy, s, font, col, fill=(255, 255, 255, 235), bw=None):
    bw = int(3 * FS) if bw is None else bw; bb = d.textbbox((0, 0), s, font=font); x, y = xy
    d.rectangle([x, y, x + bb[2] - bb[0] + 20 * FS, y + bb[3] - bb[1] + 16 * FS], fill=fill, outline=col if bw else None, width=bw)
    d.text((x + 10 * FS, y + 8 * FS - bb[1]), s, font=font, fill=col); return y + bb[3] - bb[1] + 16 * FS


kc = int(m["k_closest"])
sys.path.insert(0, HERE); import eta_v5   # noqa: E402  (text 11-23) per-step eta of the closest arm part
ETA = eta_v5.load(A.run, A.work)
import label_v5   # noqa: E402  (final 11-24) part / m_h / extra of the same pair and u, TCP path
LAB = label_v5.load(A.run, A.work, A.label_avg); NF = int(round(A.future * 60))
import path_v5   # noqa: E402  (wrap 11-24) planned path if recorded, else the recorded TCP
PATH = path_v5.Path(A.run, LAB["tcp"], LAB["t"]); FUT_SRC = PATH.describe(A.future)
print("[v5 single] future path: " + (FUT_SRC if NF > 0 else "off"))


def cm_i(x):   # integer cm, sign kept (no "-0")
    return int(round(100.0 * float(x)))


def box2(d, xy, l1, f1, c1, l2, f2, c2, outline):   # one box, two lines
    b1 = d.textbbox((0, 0), l1, font=f1); b2 = d.textbbox((0, 0), l2, font=f2) if l2 else (0, 0, 0, 0); x, y = xy; pad = 10 * FS; gap = 8 * FS if l2 else 0
    w = max(b1[2] - b1[0], b2[2] - b2[0]) + 2 * pad; h = (b1[3] - b1[1]) + gap + (b2[3] - b2[1]) + 2 * pad
    d.rectangle([x, y, x + w, y + h], fill=(255, 255, 255, 235), outline=outline, width=int(3 * FS)); d.text((x + pad, y + pad - b1[1]), l1, font=f1, fill=c1)
    if l2:
        d.text((x + pad, y + pad + (b1[3] - b1[1]) + gap - b2[1]), l2, font=f2, fill=c2)
    return y + h


def gap_text(k):   # "gap N cm  (+E cm)"; the bracket only for an effective-mass run (final 11-24: E = label_v5 extra, averaged)
    if not np.isfinite(L["d"][k]):
        return ""
    s = "gap %d cm" % cm_i(L["d"][k])
    if kind == "effmass" and np.isfinite(LAB["extra_s"][k]):
        s += "  (%+d cm)" % cm_i(LAB["extra_s"][k])
    return s


def part_text(k):   # (final 11-24) "<closest arm part> <m_h> kg" (same step, part and u as the extra)
    sg = int(LAB["seg"][k])
    if sg < 0 or not np.isfinite(LAB["mh_s"][k]):
        return ""
    return "%s %.1f kg" % (label_v5.NICE.get(sg, str(sg)), float(LAB["mh_s"][k]))


def boxn(d, xy, lines, outline):   # (final 11-24) one box, n lines [(text, font, colour)]
    lines = [x for x in lines if x[0]]; bs = [d.textbbox((0, 0), t, font=f) for t, f, _ in lines]; x, y = xy; pad = 10 * FS; gap = 7 * FS
    w = max(b[2] - b[0] for b in bs) + 2 * pad; h = sum(b[3] - b[1] for b in bs) + gap * (len(bs) - 1) + 2 * pad
    d.rectangle([x, y, x + w, y + h], fill=(255, 255, 255, 235), outline=outline, width=int(3 * FS)); yy = y + pad
    for (t, f, c), b in zip(lines, bs):
        d.text((x + pad, yy - b[1]), t, font=f, fill=c); yy += (b[3] - b[1]) + gap
    return y + h


def future_path(d, k):   # (final 11-24) TCP line (colour path_v5.RGB, pathred 11-24) over the next --future s, alpha fading toward the end
    if NF <= 0:
        return
    P = PATH.points(k, A.future)   # (wrap) plan or recorded
    if len(P) < 2:
        return
    uv = np.array([proj(p) for p in P]); lw = max(4, int(3.5 * FS)); n = len(uv) - 1
    for i in range(n):   # white edge under the line (apart from the orange volumes and the dark closest-pair line)
        al = int(round(255 * (1.0 - 0.85 * i / max(n - 1, 1))))
        d.line([tuple(uv[i]), tuple(uv[i + 1])], fill=(255, 255, 255, int(0.6 * al)), width=lw + 4)
    for i in range(n):
        al = int(round(255 * (1.0 - 0.85 * i / max(n - 1, 1))))
        d.line([tuple(uv[i]), tuple(uv[i + 1])], fill=path_v5.RGB + (al,), width=lw)   # (pathred 11-24) colour from path_v5.RGB
def right_panel(k, still):
    p = os.path.join(A.work, "frames_" + (A.still_tag if still else A.tag), "cam0", "f%05d.png" % (k - ((kc - 3 * min(A.still_span // 3, kc // 3)) if still else int(round(A.t0 * 60)))))   # (gifarg 11-24) still span as run_v5 still()
    if not os.path.exists(p):
        return None
    fr = cv2.imread(p)
    if fr.shape[:2] != (H_, W_):
        fr = cv2.resize(fr, (W_, H_), interpolation=cv2.INTER_CUBIC)
    if not A.lookat:
        fr = cv2.remap(fr, MAPX, MAPY, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im, "RGBA"); c = COL[kind]
    fN = ImageFont.truetype(FB, int(24 * FS)); fL = ImageFont.truetype(FB, int(22 * FS)); fM = ImageFont.truetype(FB, int(18 * FS))
    future_path(d, k)   # (final 11-24) under the box and the closest-pair line
    boxn(d, (12 * FS, 10 * FS), [("Effective mass" if kind == "effmass" else "Uniform", fL, c), (part_text(k), fM, (30, 30, 30)), (gap_text(k), fM, (30, 30, 30))], c)   # (final 11-24) 3 lines
    a, b = L["a"][k], L["b"][k]
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):   # (v5b) thin line only in the scene
        pa, pb = proj(a), proj(b); lw = max(2, int(1.4 * FS)); d.line([tuple(pa), tuple(pb)], fill=(255, 255, 255, 200), width=lw + 2); d.line([tuple(pa), tuple(pb)], fill=(20, 20, 20), width=lw)
    return im


GAP, FOOT = 12, int(46 * FS)
def compose(k, still=False):
    R_ = right_panel(k, still)
    if R_ is None:
        return None, None
    L_, f = rgb_left(k); d = ImageDraw.Draw(L_, "RGBA"); box(d, (12 * FS, 10 * FS), "real RGB", ImageFont.truetype(FB, int(22 * FS)), (30, 30, 30))
    out = Image.new("RGB", (2 * W_ + GAP, H_ + FOOT), (255, 255, 255)); out.paste(L_, (0, 0)); out.paste(R_, (W_ + GAP, 0)); d = ImageDraw.Draw(out); fS = ImageFont.truetype(FR, int(16 * FS))
    gain = "effective-mass gain %.2f" % m["gain"] if kind == "effmass" else "uniform %.0f cm" % (100 * m["d0"])
    d.text((12 * FS, H_ + 6 * FS), gain + ("  \u00b7  " + PATH.caption(A.future) if NF > 0 else ""), font=fS, fill=(95, 95, 95))   # (wrap) path kind
    tx = "t = %.2f s" % float(L["t"][k]); bb = d.textbbox((0, 0), tx, font=fS); d.text((out.size[0] - (bb[2] - bb[0]) - 12 * FS, H_ + 6 * FS), tx, font=fS, fill=(95, 95, 95))
    return out, dict(t=round(float(L["t"][k]), 3), rgb_frame=f, dist_cm=round(100 * float(L["d"][k]), 1), angle_deg=None if not np.isfinite(L["ang"][k]) else round(float(L["ang"][k]), 1))


k0_ = int(round(A.t0 * 60)); km = None   # (wrap 11-24) --at: the rendered step (main frames) nearest the moment
if A.at:
    km = int(np.argmin(np.abs(L["t"] - float(A.at)))); km = km - ((km - k0_) % A.every)
t_ = time.time(); im, inf = compose(kc, still=True) if km is None else compose(km); p = os.path.join(A.out, "%s_closest.png" % A.name); im.save(p); sec = time.time() - t_
kp = kc if km is None else km   # the PNG / GIF step
v = L["a"][kc] - L["b"][kc]; sys.path.insert(0, HERE); import parts_v5   # noqa: E402  (win2 11-24) 19-segment names (the arm pair comes from the record's human_arm_parts)
SEGN = dict(enumerate(parts_v5.SEGN))
INFO = dict(run=os.path.basename(A.run), robot_exclude=m.get("robot_exclude"), kind=kind, gain=m["gain"], eta_max=m["eta_max"], window=m["window"], closest_cm=round(100 * m["closest_m"], 1), t_closest=m["t_closest"],
            part=SEGN.get(int(L["seg"][kc])), angle_deg=round(float(L["ang"][kc]), 1), horizontal_cm=round(100 * float(np.hypot(v[0], v[1])), 1), height_cm=round(100 * float(v[2]), 1),
            png=p, png_frame=inf, png_label=gap_text(kp), png_part=part_text(kp), label_raw=dict(part=label_v5.NICE.get(int(LAB["seg"][kp])), m_h=round(float(LAB["mh"][kp]), 3), extra_cm=round(100 * float(LAB["extra"][kp]), 2), mu=round(float(LAB["mu"][kp]), 3)),
            png_moment=dict(step=int(kp), t=round(float(L["t"][kp]), 3), source="--at %s" % A.at if km is not None else "closest step (still render)"),
            label_avg_s=A.label_avg, mass_rule=str(LAB["rule"]), future_path=FUT_SRC if NF > 0 else "off", future_source=PATH.source, paper_blur=A.paper, d_base=ETA["d_base"], d_base_src=ETA["d_base_src"], mosaic_src=MOS_SRC, mosaic_px=[list(b) for b in BLUR], png_size=list(im.size), view="look-at --cam" if A.lookat else "real colour camera (pose file + K, distortion applied)", compose_s_png=round(sec, 2))
print("[v5 single] %s %s" % (p, json.dumps({k_: INFO[k_] for k_ in ("closest_cm", "t_closest", "part", "angle_deg", "horizontal_cm", "height_cm")})))
if A.gif:   # (v5b) MP4: every --mp4-every steps over the whole window; GIF: every --gif-every steps around the closest step (both grids through kc)
    ga, gb = [float(x) for x in A.gif.split(",")]; k0 = int(round(A.t0 * 60)); T = L["t"]
    assert (kc - k0) % A.every == 0 and A.mp4_every % A.every == 0 and A.gif_every % A.every == 0, "frame grids must pass through the closest step"
    ma, mb = [float(x) for x in A.mp4.split(",")] if A.mp4 else (ga, gb)   # (win2) MP4 span
    ks = [k for k in range(k0, len(T)) if (k - kc) % A.mp4_every == 0 and ma - 1e-6 <= T[k] <= mb + 1e-6]
    mp4 = os.path.join(A.out, "%s.mp4" % A.name); fps = 60.0 / A.mp4_every; frs = 0; t1 = time.time(); ff = None; used = {}
    for k in ks:
        f_, _ = compose(k)
        if f_ is None:
            continue
        used[k] = int(k - k0); frs += 1
        if ff is None:
            Wv, Hv = f_.size[0] // 2 * 2, f_.size[1] // 2 * 2
            ff = subprocess.Popen(["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (Wv, Hv), "-r", "%g" % fps, "-i", "-",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
        ff.stdin.write(np.ascontiguousarray(np.asarray(f_)[:Hv, :Wv]).tobytes())
    ff.stdin.close(); ff.wait(); INFO["compose_s_per_frame"] = round((time.time() - t1) / max(frs, 1), 3)
    sys.path.insert(0, HERE); import mp4_v5   # noqa: E402  (win2) over 12 MB -> higher crf only
    _crf = mp4_v5.shrink(mp4)
    if _crf != mp4_v5.CRF0:
        INFO["mp4_crf"] = _crf; print("[v5 single] MP4 over 12 MB at crf 20 -> crf %d" % _crf)
    _ts = [float(T[k]) for k in used]
    INFO.update(mp4=mp4, mp4_frames=frs, mp4_fps=fps, mp4_size=[Wv, Hv], mp4_mb=round(os.path.getsize(mp4) / 1e6, 2), mp4_s=[round(_ts[0], 2), round(_ts[-1], 2)] if _ts else [ma, mb], isaac_frame_of_step={str(k): v for k, v in list(used.items())[::25]})
    gp = os.path.join(A.out, "%s.gif" % A.name); gfps = 60.0 / A.gif_every; tc = float(T[kp]); half = A.gif_half; gk = [k for k in range(k0, len(T)) if (k - kp) % A.gif_every == 0 and (k - k0) % A.every == 0]   # (wrap) around the PNG step
    gfr = {}
    for k in gk:
        if abs(T[k] - tc) <= half + 1e-6:
            f_, _ = compose(k)
            if f_ is not None:
                gh = int(round(A.gif_width * f_.size[1] / f_.size[0])); gfr[k] = path_v5.gif_frame(f_.resize((A.gif_width, gh), Image.LANCZOS))   # (pathgif 11-24) path colour kept in the palette
    while True:   # constant 100 ms frames, full GIF width; over the size limit -> shorter span (0.5 s per side), never fewer frames per second
        q = [gfr[k] for k in sorted(gfr) if abs(T[k] - tc) <= half + 1e-6]
        q[0].save(gp, save_all=True, append_images=q[1:], duration=int(round(1000.0 / gfps)), loop=0, optimize=True)
        if os.path.getsize(gp) / 1e6 <= A.gif_mb or half <= 1.0:
            break
        half -= 0.5
    ks_g = [k for k in sorted(gfr) if abs(T[k] - tc) <= half + 1e-6]
    INFO.update(gif=gp, gif_frames=len(q), gif_fps=gfps, gif_width=A.gif_width, gif_frame_s=round(1.0 / gfps, 3), gif_mb=round(os.path.getsize(gp) / 1e6, 2),
                gif_s=[round(float(T[ks_g[0]]), 2), round(float(T[ks_g[-1]]), 2)], gif_half_s=half)
    print("[v5 single] %s %d frames %.0f fps %.2f MB; %s %d frames %.0f fps (%.2f-%.2f s) %.2f MB" % (mp4, frs, fps, INFO["mp4_mb"], gp, len(q), gfps, INFO["gif_s"][0], INFO["gif_s"][1], INFO["gif_mb"]))
if A.rgb_smpl == "off":   # (cmpview 11-24) recorded only when off (default output unchanged)
    INFO["rgb_smpl"] = "off"
if BXI is not None:   # (baseex 11-24)
    INFO["base_exclude"] = BXI; print("[v5 single] base exclude: left-panel frames drawn with an earlier row %d, with no mesh %d" % (BXI["drawn_replaced"], BXI["drawn_none"]))
json.dump(INFO, open(os.path.join(A.out, "%s.json" % A.name), "w"), indent=1)
