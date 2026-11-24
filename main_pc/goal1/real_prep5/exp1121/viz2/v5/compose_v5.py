"""compose_v5.py - viz2 v5 comparison composition (CPU): two Isaac re-renders (render_v5.sh, same camera, same t0 / every) side by side,
uniform left, effective mass right. Per panel: top-left box "Uniform" / "gap N cm" and "Effective mass" / "gap N cm  (+D cm)" (D = the
two SHOWN integers' difference), a thin line robot point -> arm point (lines.npz a, b), captions under the panels.
(cmpview 11-24) Both panels at the SAME time, reference = the effective-mass run:
  - effective-mass panel at step k (its frame, its gap); uniform panel at the rendered uniform step whose replay time is nearest to the
    effective-mass step's time (frame and gap of that same step; the two replays drift up to 2 steps apart), so N, N and D are one moment.
  - PNG <out>/<name>_closest.png at the effective-mass closest step (meta k_closest), from the main render frames (no still render);
    caption "closest at N s" once. GIF / MP4 frames on grids through that step; caption "t = N s".
  - --realcam <run prefix> --sensing <dir>: the panels were rendered with the REAL colour camera (cam_v5.py: 1280 x 720 pinhole with the
    colour K); here the frames get the same distortion map as single mode (compose_single) and the line uses that camera model.
    Without --realcam the frames are a look-at camera (--cam eye,target; focal 24 mm, aperture 20.955 mm), as before.
  - captions: left margin 28 px (x panel height / 480; was 12: viewers that crop the edge showed "eal robot", "ffective-mass").
  --still-tag / --still-span are accepted and not used (the PNG no longer needs a still render).
Outputs <out>/<name>_closest.png, <name>.mp4 (H.264, a frame every --mp4-every 4 steps = 15 frames/s over --gif a,b), <name>.gif (a frame
every --gif-every 6 steps, 10 frames/s, reference closest step +-4 s inside --gif a,b, shorter by 0.5 s per side while > --gif-mb), <name>.json.
(win2 11-24) --at T: the reference moment (run_v5: the automatic choice, largest effective-mass eta extra with both gaps <= 0.20 m): PNG / GIF
around the effective-mass step nearest T, caption "selected at N s"; without --at: the effective-mass closest step, "closest at N s".
--mp4 a,b: MP4 span (run_v5: the whole render range); an MP4 over 12 MB is re-encoded with a higher crf only (mp4_v5; json mp4_crf).
(shviz 11-24) Pair of two runs of the same kind (e.g. two effective-mass runs: human / reduced eta reference, simulations): left = --a,
right = --b = the reference; D = right gap - left gap. Only in such a pair the method label gets the recorded eta reference
("Effective mass (human)" / "(reduced)"); otherwise the label is the method name only ("Effective mass", "Uniform").
(cmpfinal 11-24) Final layout as single mode: per panel ONE box of 3 fixed-height rows (rows aligned across the panels): method (colour) /
closest arm part of that panel's own robot (+ " <m_h> kg" in an effective-mass panel; label_v5.py, the single-mode computation: recorded
human_arm_parts incl. shoulders, mass mode, ISO override, --label-avg) / "gap N cm" (+ "  (+D cm)" on the effective-mass panel, D = shown gap
difference at the same time); the TCP future path of the next --future 0.5 s (label_v5 tcp = recorded TCP, as single mode) in dark red (path_v5.RGB, pathred 11-24) fading
toward its end on both panels; one caption line "same recorded person . closest at N s" (video: ". t = N s"). No RGB panel, no mosaic.
Times shown = the run's step times (steps t).
(passgif 11-24) --pass-json FILE (window_v5 --pass-json; run_v5 comparison mode without --window): GIF = the chosen pass from the later
start to the later arrival, frames on a time grid of --gif-fps (10) from the pass start (rendered step nearest each time); a panel whose
robot has arrived (its own arrival time, motion_v5 goal switch) stays on its arrival frame (rendered step at / before it) with a small
"arrived" under its box, and the shown gap difference is left out while a panel is held. Over --gif-mb: 8 frames/s, then narrower
(--gif-width 1100 -> 950 -> 800 -> 700 -> 600 -> 500); json gif_fps / gif_width / gif_tries. (passgif2) MP4 = the same pass (15 frames/s time
grid, full panel size, held panels); the held label is "arrived" (goal switch on arrival) or "timeout" (400-step leg limit). PNG as
before (--at = the file's PNG moment).
Usage: compose_v5.py --a <work uniform> --b <work effmass> --tag T --cam eye,target --t0 S --every N --out DIR --name NAME [--gif a,b]
       [--realcam RUN --sensing DIR] [--view LABEL]"""
import argparse, glob, json, math, os, subprocess, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--cam", required=True); ap.add_argument("--t0", type=float, required=True); ap.add_argument("--every", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--name", required=True); ap.add_argument("--gif", default=""); ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--gif-width", type=int, default=1100); ap.add_argument("--gif-every", type=int, default=6); ap.add_argument("--gif-half", type=float, default=4.0); ap.add_argument("--gif-mb", type=float, default=10.0)
ap.add_argument("--still-tag", default="", help="not used since cmpview 11-24 (PNG from the main frames at the effective-mass closest step)")
ap.add_argument("--still-span", type=int, default=30, help="not used since cmpview 11-24")
ap.add_argument("--mp4-every", type=int, default=4, help="(gifarg 11-24) MP4: a frame every N steps (60 / N frames/s)")
ap.add_argument("--realcam", default="", help="(cmpview) run prefix whose _camera_pose.yaml (+ --sensing roll / pitch) is the render camera")
ap.add_argument("--sensing", default=""); ap.add_argument("--view", default="", help="(cmpview) view label for the json / log")
ap.add_argument("--at", default="", help="(win2) reference moment, s (effective-mass run time); empty = its closest step")
ap.add_argument("--mp4", default="", help="(win2) MP4 span a,b s; empty = --gif")
ap.add_argument("--future", type=float, default=0.5, help="(cmpfinal) TCP future path, s (0 = none); as compose_single")
ap.add_argument("--label-avg", type=float, default=0.0, help="(cmpfinal) moving average of m_h / extra, s; as compose_single (default 0)")
ap.add_argument("--pass-json", default="", help="(passgif 11-24) window_v5 --pass-json file: GIF = that pass with held arrived panels, MP4 end")
ap.add_argument("--gif-fps", type=float, default=10.0, help="(passgif) GIF frame rate (first try; over --gif-mb: 8)")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
PJ = json.load(open(A.pass_json)) if A.pass_json else None   # (passgif 11-24)
HOLD = None if PJ is None else dict(ref=float(PJ["arrive"]["effmass"]), oth=float(PJ["arrive"]["uniform"]),   # reference = effective mass, other = uniform
                                    ref_how="arrived" if PJ["arrive_how"]["effmass"] == "arrival" else "timeout", oth_how="arrived" if PJ["arrive_how"]["uniform"] == "arrival" else "timeout")
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COL = {"uniform": (25, 95, 215), "effmass": (240, 110, 10)}
W5 = [dict(dir=d, meta=json.load(open(os.path.join(d, "meta.json"))), L=np.load(os.path.join(d, "lines.npz"))) for d in (A.a, A.b)]
WU = next((w for w in W5 if w["meta"]["kind"] == "uniform"), None); WE = next((w for w in W5 if w["meta"]["kind"] == "effmass"), None)
PAIR = WU is None or WE is None   # (shviz 11-24) two runs of the same kind: left = --a, right = --b (reference)
if PAIR:
    WU, WE = W5[0], W5[1]
REF, OTH = WE, WU   # (cmpview) reference = the effective-mass run (pair: --b); the other panel at the same time
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import parts_v5, label_v5, path_v5   # noqa: E402  (wrap) path_v5: planned / recorded future path (single-mode rule)


def title(w):   # (shviz) method name; the recorded eta reference only in a same-kind pair (e.g. "Effective mass (human)" vs "(reduced)")
    t = "Effective mass" if w["meta"]["kind"] == "effmass" else "Uniform"
    if not PAIR:
        return t
    mo, io, src = parts_v5.mass_rule(w["meta"]["run"])
    return t + (" (%s)" % mo if w["meta"]["kind"] == "effmass" and not src.startswith("default") else "")
k0 = int(round(A.t0 * 60)); cv = [float(x) for x in A.cam.split(",")]; eye, tgt = np.array(cv[:3]), np.array(cv[3:6])
fw = (tgt - eye) / np.linalg.norm(tgt - eye); rt = np.cross(fw, [0, 0, 1.0]); rt /= np.linalg.norm(rt); up = np.cross(rt, fw)
RCAM = None
if A.realcam:   # (cmpview) real colour camera: pose as cam_v5.py (run pose file, roll / pitch of the sensing session), K + distortion
    import cv2, yaml
    cp = yaml.safe_load(open(A.realcam + "_camera_pose.yaml")); fs_ = sorted(glob.glob(os.path.join(A.sensing, "debug_log", "camera_pose_*.yaml"))) if A.sensing else []
    if fs_:
        cs = yaml.safe_load(open(fs_[-1])); cp["roll_deg"], cp["pitch_deg"] = cs["roll_deg"], cs["pitch_deg"]
    def _r(t, ax):
        c, s = math.cos(t), math.sin(t)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[ax]
    DI = json.load(open(os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121/viz/calib/device_intr.json")))
    RCAM = dict(R=_r(math.radians(cp["roll_deg"]), "x") @ _r(math.radians(cp["pitch_deg"]), "y") @ _r(math.radians(cp["yaw_deg"]), "z"), t=np.array(cp["t"], float),
                K=np.array(DI["K"]), D=np.array(DI["dist"]), maps={})


def remap(im):   # (cmpview) pinhole render -> raw-camera distortion (same map as compose_single), at 1280 x 720 (the K size)
    W_, H_ = 1280, 720; a = np.asarray(im)[:, :, ::-1]
    if a.shape[:2] != (H_, W_):
        a = cv2.resize(a, (W_, H_), interpolation=cv2.INTER_CUBIC)
    if "m" not in RCAM["maps"]:
        gu, gv = np.meshgrid(np.arange(W_, dtype=np.float32), np.arange(H_, dtype=np.float32))
        u = cv2.undistortPoints(np.stack([gu.ravel(), gv.ravel()], 1).reshape(-1, 1, 2), RCAM["K"], RCAM["D"], P=RCAM["K"]).reshape(H_, W_, 2).astype(np.float32)
        RCAM["maps"]["m"] = (u[..., 0].copy(), u[..., 1].copy())
    mx, my = RCAM["maps"]["m"]; a = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return Image.fromarray(np.ascontiguousarray(a[:, :, ::-1]))


def proj(P, W, H):
    if RCAM is not None:   # robot base -> colour camera (OpenCV) -> raw pixels (with distortion), 1280 x 720 scaled to the panel
        Pc = (np.asarray(P, float).reshape(-1, 3) + [0, 0, 0.715] - RCAM["t"]) @ RCAM["R"].T
        uv, _ = cv2.projectPoints(Pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), RCAM["K"], RCAM["D"]); return uv.reshape(-1, 2)[0] * [W / 1280.0, H / 720.0]
    d = np.asarray(P, float) - eye; Z = d @ fw; f = W * 24.0 / 20.955
    return np.array([W / 2 + f * (d @ rt) / Z, H / 2 - f * (d @ up) / Z])


def frame(w, k):
    p = os.path.join(w["dir"], "frames_" + A.tag, "cam0", "f%05d.png" % (k - k0))
    if not os.path.exists(p):
        return None
    im = Image.open(p).convert("RGB")
    return remap(im) if RCAM is not None else im


def cm_i(x):   # (text 11-23) integer cm, sign kept
    return int(round(100.0 * float(x)))


def at_same_time(k):   # (cmpview) other run's rendered step (grid k0 + every * j) whose time is nearest to the reference run's step k
    T, To = REF["L"]["t"], OTH["L"]["t"]; tk = float(T[k]); kk = int(np.argmin(np.abs(To - tk)))
    best = None
    for c in (kk - (kk - k0) % A.every, kk - (kk - k0) % A.every + A.every):
        if 0 <= c < len(To) and c >= k0 and (best is None or abs(To[c] - tk) < abs(To[best] - tk)):
            best = c
    return best


def label(w, k):
    return "gap %d cm" % cm_i(w["L"]["d"][k]) if np.isfinite(w["L"]["d"][k]) else ""


LABS = {}   # (cmpfinal 11-24) per work dir: label_v5 values (closest part, m_h, extra, TCP) = the single-mode computation
NF = int(round(A.future * 60))


def lab(w):
    if w["dir"] not in LABS:
        LABS[w["dir"]] = label_v5.load(w["meta"]["run"], w["dir"], A.label_avg)
    return LABS[w["dir"]]


PATHS = {}


def path(w):   # (wrap 11-24) the MPC plan if the run recorded it, else its recorded TCP (path_v5, as single mode)
    if w["dir"] not in PATHS:
        PATHS[w["dir"]] = path_v5.Path(w["meta"]["run"], lab(w)["tcp"], lab(w)["t"])
    return PATHS[w["dir"]]


def path_caption():   # "path 0.5 s" / "planned path 0.5 s" (both panels); different sources -> both named
    s_ = {w["meta"]["kind"] if not PAIR else str(i): path(w) for i, w in enumerate((WU, WE))}
    srcs = set(p_.source for p_ in s_.values())
    if len(srcs) == 1:
        return next(iter(s_.values())).caption(A.future)
    return "; ".join("%s %s" % (k_, p_.caption(A.future)) for k_, p_ in s_.items())


def part_text(w, k):   # (cmpfinal) "<closest arm part>" (+ " <m_h> kg" in an effective-mass panel), as single mode part_text
    L5 = lab(w); sg = int(L5["seg"][k])
    if sg < 0:
        return ""
    t = label_v5.NICE.get(sg, str(sg))
    return t + (" %.1f kg" % float(L5["mh_s"][k]) if w["meta"]["kind"] == "effmass" and np.isfinite(L5["mh_s"][k]) else "")


def future_path(d, w, k, Wd, Hd, fs):   # (cmpfinal) TCP line (colour path_v5.RGB, pathred 11-24) over the next --future s, fading toward its end (single mode future_path)
    if NF <= 0:
        return
    P = path(w).points(k, A.future)   # (wrap) plan or recorded
    if len(P) < 2:
        return
    uv = np.array([proj(p, Wd, Hd) for p in P]); lw = max(4, int(3.5 * fs)); n = len(uv) - 1
    for i in range(n):
        al = int(round(255 * (1.0 - 0.85 * i / max(n - 1, 1)))); d.line([tuple(uv[i]), tuple(uv[i + 1])], fill=(255, 255, 255, int(0.6 * al)), width=lw + 4)
    for i in range(n):
        al = int(round(255 * (1.0 - 0.85 * i / max(n - 1, 1)))); d.line([tuple(uv[i]), tuple(uv[i + 1])], fill=path_v5.RGB + (al,), width=lw)   # (pathred 11-24) colour from path_v5.RGB


def box3(d, xy, lines, outline, fs):   # (cmpfinal) one box, 3 rows of FIXED height (font ascent + descent), so the rows line up across the panels
    x, y = xy; pad = 10 * fs; gap = 7 * fs; rows = []
    for t, f, c in lines:
        a_, de = f.getmetrics(); rows.append((t, f, c, a_ + de))
    w = max([d.textbbox((0, 0), t, font=f)[2] for t, f, c, _ in rows if t] + [0]) + 2 * pad; h = sum(r[3] for r in rows) + gap * (len(rows) - 1) + 2 * pad
    d.rectangle([x, y, x + max(w, 120 * fs), y + h], fill=(255, 255, 255, 235), outline=outline, width=max(3, int(3 * fs))); yy = y + pad; yb = y + h
    for t, f, c, rh in rows:
        if t:
            d.text((x + pad, yy), t, font=f, fill=c)
        yy += rh + gap
    return yb   # (passgif) bottom of the box


def panel(w, k, extra="", held=False):
    im = frame(w, k)
    if im is None:
        return None
    m = w["meta"]; c = COL[m["kind"]]; Wd, Hd = im.size; d = ImageDraw.Draw(im, "RGBA"); fs = Hd / 480.0
    fL = ImageFont.truetype(FB, int(22 * fs)); fM = ImageFont.truetype(FB, int(18 * fs))   # (cmpfinal) single mode sizes
    future_path(d, w, k, Wd, Hd, fs)   # under the box and the closest-pair line
    a, b = w["L"]["a"][k], w["L"]["b"][k]
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):   # (v5b) thin line only in the scene
        pa, pb = proj(a, Wd, Hd), proj(b, Wd, Hd); lw = max(2, int(1.4 * fs))
        d.line([tuple(pa), tuple(pb)], fill=(255, 255, 255, 200), width=lw + 2); d.line([tuple(pa), tuple(pb)], fill=(20, 20, 20), width=lw)
    l1 = title(w); l2 = part_text(w, k); l3 = label(w, k) + extra   # (cmpfinal) method / part (+ m_h) / gap (+ difference)
    yb = box3(d, (int(12 * fs), int(10 * fs)), [(l1, fL, c), (l2, fM, (30, 30, 30)), (l3, fM, (30, 30, 30))], c, fs)
    if held:   # (passgif 11-24) this robot has arrived: its panel stays on the arrival frame
        fA = ImageFont.truetype(FB, int(14 * fs)); tx_ = held if isinstance(held, str) else "arrived"; bb = d.textbbox((0, 0), tx_, font=fA); x_, y_ = int(12 * fs), yb + int(5 * fs)
        d.rectangle([x_, y_, x_ + bb[2] + int(10 * fs), y_ + bb[3] + int(6 * fs)], fill=(255, 255, 255, 220)); d.text((x_ + int(5 * fs), y_ + int(2 * fs)), tx_, font=fA, fill=(70, 70, 70))
    return im, (l2, l3)


def on_grid(w, ta):   # (passgif) rendered step (k0 + every * j) of run w at or before time ta
    Tw = w["L"]["t"]; kk = int(np.searchsorted(Tw, ta + 1e-9) - 1); kk = max(kk, k0)
    return kk - (kk - k0) % A.every


def compose(k, png=False, hold=None):
    """reference step k and the other run's step at the same time; None if a frame is missing. -> (image, record)
    (passgif) hold = dict(effmass=t, uniform=t): arrival times; a panel past its arrival shows its arrival frame with "arrived"."""
    ko = at_same_time(k)
    if ko is None:
        return None, None
    ke, ku = k, ko; he = hu = False
    if hold is not None:
        if float(REF["L"]["t"][k]) > hold["ref"] + 1e-9:
            ke = on_grid(REF, hold["ref"]); he = True
        if float(OTH["L"]["t"][ko]) > hold["oth"] + 1e-9:
            ko = on_grid(OTH, hold["oth"]); hu = True
        ku = ko
    extra = ""
    if np.isfinite(WE["L"]["d"][ke]) and np.isfinite(WU["L"]["d"][ku]) and not (he or hu):   # (passgif) no difference while a panel is held
        extra = "  (%+d cm)" % (cm_i(WE["L"]["d"][ke]) - cm_i(WU["L"]["d"][ku]))   # D = shown effective-mass gap - shown uniform gap
    hw = {id(REF): he and hold.get("ref_how", "arrived"), id(OTH): hu and hold.get("oth_how", "arrived")} if hold is not None else {id(REF): False, id(OTH): False}   # (passgif2) "arrived" / "timeout"
    pu = panel(WU, ku, held=hw[id(WU)]); pe = panel(WE, ke, extra, held=hw[id(WE)])   # uniform left, effective mass right
    if pu is None or pe is None:
        return None, None
    (iu, lu), (ie, le) = pu, pe
    Wd, Hd = iu.size; gap = 8; fs = Hd / 480.0; foot = int(34 * fs); xm = int(28 * fs)   # (cmpfinal) one caption line
    out = Image.new("RGB", (2 * Wd + gap, Hd + foot), (255, 255, 255)); out.paste(iu, (0, 0)); out.paste(ie, (Wd + gap, 0)); d = ImageDraw.Draw(out)
    fS = ImageFont.truetype(FR, int(16 * fs)); tr = float(REF["L"]["t"][k])
    tx = "same recorded person \u00b7 " + (("closest at %.2f s" % tr) if png else ("t = %.2f s" % tr)) + ("  \u00b7  " + path_caption() if NF > 0 else "")   # (cmpfinal) the only caption; (wrap) path kind
    d.text((xm, Hd + int(8 * fs)), tx, font=fS, fill=(95, 95, 95))
    rec = dict(t=round(tr, 3), step_ref=int(ke), step_other=int(ko), held=[n_ for n_, h_ in (("reference", he), ("other", hu)) if h_], t_other=round(float(OTH["L"]["t"][ko]), 3), dt_ms=round(1000 * abs(float(OTH["L"]["t"][ko]) - tr), 1),
               part_uniform=lu[0], label_uniform=lu[1], part_effmass=le[0], label_effmass=le[1],
               gap_uniform_cm=round(100 * float(WU["L"]["d"][ku]), 2), gap_effmass_cm=round(100 * float(WE["L"]["d"][ke]), 2))
    return out, rec


for _w in (WU, WE):   # (wrap) future path source per run (log + json)
    print("[v5] future path %s: %s" % (os.path.basename(_w["meta"]["run"]), path(_w).describe(A.future)))
T = REF["L"]["t"]; kr = int(np.argmin(np.abs(T - float(A.at)))) if A.at else int(REF["meta"]["k_closest"]); kr = kr - ((kr - k0) % A.every); kc = kr   # (win2) --at; (win2b) on the render grid
info = dict(name=A.name, future_path={os.path.basename(w["meta"]["run"]): path(w).describe(A.future) for w in (WU, WE)}, future_source={os.path.basename(w["meta"]["run"]): path(w).source for w in (WU, WE)},
            at=float(A.at) if A.at else None, view=A.view or "look-at", cam=A.cam, realcam=A.realcam or None, t0=A.t0, every=A.every, reference=REF["meta"]["kind"], png_step=kc, png_t=float(T[kc]),
            runs={w["meta"]["kind"]: w["meta"]["run"] for w in W5})
for w in W5:   # closest distance split into horizontal (base x-y) and height (base z) parts of the robot-point minus arm-point vector
    m = w["meta"]; k = m["k_closest"]; v = w["L"]["a"][k] - w["L"]["b"][k]
    info[m["kind"]] = dict(closest_cm=round(100 * m["closest_m"], 1), t_closest=m["t_closest"], horizontal_cm=round(100 * float(np.hypot(v[0], v[1])), 1), height_cm=round(100 * float(v[2]), 1))
t_ = time.time(); im, rec = compose(kc, png=True)
if im is None:
    raise SystemExit("[v5] no rendered frame at the reference closest step %d (t %.2f s) for both runs (render range / --tag?)" % (kc, T[kc]))
p = os.path.join(A.out, "%s_closest.png" % A.name); im.save(p); info.update(png=p, png_same_time=rec, compose_s_per_frame=round(time.time() - t_, 3))
print("[v5] %s  %s" % (p, json.dumps(rec)))
dts = []
if A.gif:   # (gifarg 11-24) MP4 over the whole --gif span, a frame every --mp4-every steps on a grid through the reference closest step
    ga, gb = [float(x) for x in (A.mp4 or A.gif).split(",")]; mp4 = os.path.join(A.out, "%s.mp4" % A.name)   # (win2) MP4 span
    if PJ is not None:   # (passgif2 11-24) MP4 = the pass (GIF span), held arrived panels
        ga, gb = [float(x) for x in PJ["gif"]]; info["mp4_rule"] = "pass %d %s (GIF span), held panels" % (PJ["pass_no"], PJ["dirn"])
    fps = 60.0 / A.mp4_every; ff = None; nf = 0; ts_ = []
    if PJ is not None:   # (passgif2) time grid of 60 / mp4_every frames/s from the pass start (rendered step nearest each time; real speed)
        Tg_ = np.array([kk for kk in range(k0, len(T)) if (kk - k0) % A.every == 0]); ks4 = []
        for x in np.arange(ga, gb + 1e-6, A.mp4_every / 60.0):
            kk = int(Tg_[np.argmin(np.abs(T[Tg_] - x))])
            if not ks4 or kk != ks4[-1]:
                ks4.append(kk)
    else:
        ks4 = [k for k in range(k0, len(T)) if (k - kr) % A.mp4_every == 0 and (k - k0) % A.every == 0 and ga - 1e-6 <= T[k] <= gb + 1e-6]
    for k in ks4:
        f, rc_ = compose(k, hold=HOLD)
        if f is None:
            continue
        dts.append(rc_["dt_ms"])
        if ff is None:
            Wv, Hv = f.size[0] // 2 * 2, f.size[1] // 2 * 2
            ff = subprocess.Popen(["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (Wv, Hv), "-r", "%g" % fps, "-i", "-",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
        ff.stdin.write(np.ascontiguousarray(np.asarray(f)[:Hv, :Wv]).tobytes()); nf += 1; ts_.append(float(T[k]))
    if ff is not None:
        ff.stdin.close(); ff.wait()
        import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mp4_v5   # (win2) over 12 MB -> higher crf only
        info["mp4_crf"] = mp4_v5.shrink(mp4)
        info.update(mp4=mp4, mp4_frames=nf, mp4_fps=fps, mp4_size=[Wv, Hv], mp4_mb=round(os.path.getsize(mp4) / 1e6, 2), mp4_s=[round(ts_[0], 2), round(ts_[-1], 2)])
        print("[v5] %s %d frames %.0f fps %.2f MB crf %d (%.2f-%.2f s)" % (mp4, nf, fps, info["mp4_mb"], info["mp4_crf"], ts_[0], ts_[-1]))
    else:
        print("[v5] no rendered frame in the --gif span %.2f-%.2f s: no MP4" % (ga, gb))
if A.gif and PJ is not None:   # (passgif 11-24) one whole pass, held arrived panels, 10 -> 8 frames/s, then narrower
    ta_, tb_ = [float(x) for x in PJ["gif"]]
    hold = HOLD
    Tgrid = np.array([kk for kk in range(k0, len(T)) if (kk - k0) % A.every == 0])
    cache = {}; tries = []; gp = os.path.join(A.out, "%s.gif" % A.name); ok_ = False
    for gfps in (A.gif_fps, 8.0):
        tg = np.arange(ta_, tb_ + 1e-6, 1.0 / gfps); ks = []
        for x in tg:
            kk = int(Tgrid[np.argmin(np.abs(T[Tgrid] - x))])
            if not ks or kk != ks[-1]:
                ks.append(kk)
        if T[ks[-1]] < tb_ - 1e-6 and ks[-1] != Tgrid[-1]:   # the last frame shows both arrived
            ks.append(int(Tgrid[min(np.searchsorted(T[Tgrid], tb_), len(Tgrid) - 1)]))
        for kk in ks:
            if kk not in cache:
                f, rc_ = compose(kk, hold=hold)
                cache[kk] = None if f is None else f.resize((A.gif_width, int(round(A.gif_width * f.size[1] / f.size[0]))), Image.LANCZOS)
                if f is not None:
                    dts.append(rc_["dt_ms"])
        for gw in ([A.gif_width] if gfps == A.gif_fps else [A.gif_width, 950, 800, 700, 600, 500]):
            frs = [path_v5.gif_frame(cache[kk] if gw == A.gif_width else cache[kk].resize((gw, int(round(gw * cache[kk].size[1] / cache[kk].size[0]))), Image.LANCZOS)) for kk in ks if cache[kk] is not None]
            if not frs:
                break
            frs[0].save(gp, save_all=True, append_images=frs[1:], duration=int(round(1000.0 / gfps)), loop=0, optimize=True); mb = os.path.getsize(gp) / 1e6
            tries.append(dict(fps=gfps, width=gw, frames=len(frs), mb=round(mb, 2)))
            if mb <= A.gif_mb:
                ok_ = True; break
        if ok_ or not frs:
            break
    if tries:
        tl = tries[-1]; kk_ = [kk for kk in ks if cache[kk] is not None]
        info.update(gif=gp, gif_frames=tl["frames"], gif_fps=tl["fps"], gif_width=tl["width"], gif_mb=tl["mb"], gif_tries=tries, gif_s=[round(float(T[kk_[0]]), 2), round(float(T[kk_[-1]]), 2)], gif_pass=PJ)
        print("[v5] %s %d frames %.0f fps width %d %.2f MB (%.2f-%.2f s, pass %d %s; held after arrival: effmass %.2f s, uniform %.2f s)%s" % (
            gp, tl["frames"], tl["fps"], tl["width"], tl["mb"], info["gif_s"][0], info["gif_s"][1], PJ["pass_no"], PJ["dirn"], hold["ref"], hold["oth"],
            "" if ok_ else "  WARNING: still over %.0f MB" % A.gif_mb))
        if len(tries) > 1:
            print("[v5] gif size tries: " + "; ".join("%g fps width %d -> %.2f MB" % (t_["fps"], t_["width"], t_["mb"]) for t_ in tries))
    else:
        print("[v5] no rendered frame in the pass %.2f-%.2f s: no GIF" % (ta_, tb_))
elif A.gif:
    # (v5b) closest step of the reference run +-4 s (the --gif a,b window bounds it), a frame every --gif-every steps on a grid through that step,
    # 100 ms each; over --gif-mb -> 0.5 s shorter per side (frame rate and width kept)
    ga, gb = [float(x) for x in A.gif.split(",")]; tc = float(T[kr]); half = A.gif_half   # GIF: around the reference moment, inside --gif
    ks = [k for k in range(k0, len(T)) if (k - kr) % A.gif_every == 0 and (k - k0) % A.every == 0 and ga <= T[k] <= gb and abs(T[k] - tc) <= half + 1e-6]
    fr = {}
    for k in ks:
        f, rc_ = compose(k)
        if f is not None:
            dts.append(rc_["dt_ms"]); h = int(round(A.gif_width * f.size[1] / f.size[0])); fr[k] = path_v5.gif_frame(f.resize((A.gif_width, h), Image.LANCZOS))   # (pathgif 11-24) path colour kept in the palette
    if fr:
        gp = os.path.join(A.out, "%s.gif" % A.name); gfps = 60.0 / A.gif_every
        while True:
            frs = [fr[k] for k in sorted(fr) if abs(T[k] - tc) <= half + 1e-6]
            frs[0].save(gp, save_all=True, append_images=frs[1:], duration=int(round(1000.0 / gfps)), loop=0, optimize=True)
            if os.path.getsize(gp) / 1e6 <= A.gif_mb or half <= 1.0:
                break
            half -= 0.5
        kk_ = [k for k in sorted(fr) if abs(T[k] - tc) <= half + 1e-6]
        info["gif"] = gp; info["gif_frames"] = len(frs); info["gif_fps"] = gfps; info["gif_mb"] = round(os.path.getsize(gp) / 1e6, 2); info["gif_s"] = [round(float(T[kk_[0]]), 2), round(float(T[kk_[-1]]), 2)]
        print("[v5] %s %d frames %.0f fps %.2f MB (%.2f-%.2f s)" % (gp, len(frs), gfps, info["gif_mb"], info["gif_s"][0], info["gif_s"][1]))
    else:
        print("[v5] no rendered frame in the GIF span: no GIF")
info["same_time_dt_ms_max"] = max(dts) if dts else rec["dt_ms"]
info["robot_exclude"] = WE["meta"].get("robot_exclude")   # (robex 11-24)
json.dump(info, open(os.path.join(A.out, "%s.json" % A.name), "w"), indent=1); print(json.dumps({k: info[k] for k in ("uniform", "effmass") if k in info}))
