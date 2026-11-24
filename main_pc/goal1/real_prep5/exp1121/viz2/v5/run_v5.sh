#!/usr/bin/env bash
# run_v5.sh <real run> <replay run 1> <replay run 2> [--sensing DIR] [--betas NPY] [--cam eye,target] [--tag T] [--window a,b]
# run_v5.sh [<real run>] [options]  (single-run mode, 2026-11-23: no replays; no run = the newest real run in goal1/real_prep4) left = real RGB frame with the SMPL mesh and the MPC arm
#   capsules drawn over it, right = the 3-D re-render of the real run (compose_single.py); window = the arm-near window (exp_ana, 0.35 m)
#   that holds the smallest arm distance (the whole arm-raised part); PNG at the closest moment, MP4 (15 frames/s) and GIF (<= 10 MB) over
#   the window. Default view = the real colour camera (cam_v5.py; --cam = a look-at view instead); panels 1280 x 720; the virtual keep-out
#   box of the scene is hidden in the render; left panel = grey SMPL of the hrc_ws renderer on the raw RGB (as v3_A).
# Any cwd; a run = name, path prefix or one of its files. Betas: --betas, else goal1/night1122_s2/betas/betas_<MMDD|sensing HHMMSS>.npy,
# else taken from the sensing session's ROS log (pid from the run's exp.json, the log must name the session folder) and saved there.
# Stops if the session has no pose_npz yet (camera still on).
# viz2 v5 (2026-11-23): 3-D re-render, Uniform vs Effective-mass side by side, from the run records only (no robot, no control PC, no sensing
# process; GPU only for Isaac, under the shared lock via render_v5.sh, nice -n 19). Names = goal1/real_prep4/<name> or a path prefix.
#   person : SMPL mesh of the REAL run's sensing session (default: the newest goal1/real_prep5/sensing_* not later than the real run;
#            betas: goal1/night1122_s2/betas/betas_<MMDD of the run>.npy, else betas_<HHMMSS of the sensing>.npy); --sensing none = MPC shapes
#   window : the real run's first meeting (exp_ana, radius 0.35 m, first pass of >= 1 s, as render_v4) unless --window
#   camera : the view chosen on S1 (low oblique from the robot side), placed relative to the fist at the closest moment, unless --cam
# Output viz2/v5/out/<real name>_<tag>_v5_closest.png (each panel at its own closest moment, time under the panel; fix3), _v5.gif (both panels
# at the same time), _v5.json; inputs and frames in viz2/v5/work/<run folder>__<run name>/.
# (v5b, 11-23) default tag v5b (earlier v5 results are kept; names <run>_<tag>_closest.png / .mp4 / .gif / .json in both modes). Isaac renders a
#   frame every 2 steps; MP4 = every 4 steps (15 frames/s, real speed) over the window; GIF = every 6 steps (10 frames/s, 100 ms each, full width)
#   over the closest step +-4 s (shorter if > 10 MB). One run_v5 per work dir at a time (<work>/.run_v5.lock): two runs on one work dir had
#   mixed their step cut files (the 3-D panel showed one pose; render_v5.sh now keeps them per tag and checks the steps step2b read).
#   The distance number is a fixed box under the title box; the scene keeps a thin line only.
# (robex 11-24) --robot-exclude LINKS (default link0,link1; none = all spheres): robot spheres of these links (robex_v5.py, MPC sphere model
#   link names) are left out of the shown closest pair: scene choice (window_v5), part / m_h / extra (label_v5, eta_v5), gap and line (lines.npz).
#   The MPC cost and exp_ana distances are unchanged. --compose-only rewrites lines.npz (robex_v5.py lines); the Isaac frames keep the line of
#   their render. Log line "[v5] robot exclude ...: N of 61 robot spheres left out".
# (perpbox 11-24) comparison default view = perp: fixed look-at perpendicular to the horizontal pelvis - robot base line, looking at its middle
#   (eye -2.274,-0.556,0.550 -> target 0.023,-0.434,0.200, base frame; chosen on t17 at 22.45 s: closest arm part 3 % hidden, see
#   out/occlusion_t17_22p45.json); --view real|side|fist as before; single mode stays the real camera. Every render (all views and modes)
#   hides the MPC-only virtual objects of the scene json (plates named *virtual* / *keepout*, e.g. keepout_virtual; render_v5.sh), the real
#   robot cart (robot_table_real) stays; render_v5.sh prints the scene objects the render drew / hid.
# (wrap 11-24) single mode: automatic moment = largest eta extra among steps with the arm within --pick-gap (0.20 m), >= 0.5 s inside the run
#   (window_v5 --arm-m; none / uniform run: the closest meeting, logged); PNG / GIF there (--at), MP4 the whole run. --paper 0|1 (default 0 =
#   no clock / frame-number blur), --mosaic off|on|none|<yaml> (default off), --label-avg S (default 0) go to compose_single.py. Future path:
#   path_v5 (MPC plan if recorded, else the recorded TCP; caption "planned path 0.5 s" / "path 0.5 s"), both modes.
# (shviz 11-24) comparison default view = real (no --view; --cam without --view = that look-at view; --view fist = the old fist look-at
#   view, names <run>_<tag>_cmp_*). Two runs of the same kind (e.g. simulations with human / reduced eta reference) = a pair: left = the
#   first, right = the second (reference; window without --window: its closest meeting). --at T: the PNG / GIF moment (s). The eta volumes
#   and labels follow each record's human_mass_mode / human_iso_override / human_arm_parts (shoulder spheres drawn).
# (win2 11-24) comparison automatic window: the moment = largest effective-mass eta extra among steps where the UNIFORM replay's arm gap
#   is <= --pick-gap (0.20 m; win2b: the effective-mass gap is not a condition), >= 0.5 s inside the run (window_v5.py --uni; none -> effective-mass closest, logged); PNG / GIF at that moment ("selected at N s", --at).
#   Single mode unchanged (REAL closest). Both modes render the WHOLE run (frame grid from step 0..3 through the moment); MP4 = the whole
#   render (448 frames at 15 frames/s for 30 s; over 12 MB -> higher crf only), GIF / PNG around the moment. --compose-only stops with
#   "render again" if the frames on disk do not cover the whole run. Eta volume parts = the record's human_arm_parts (parts_v5.py).
# (cmpview 11-24) automatic window (no --window) = the TCP pass holding the CLOSEST meeting (window_v5.py: base exclude applied, arm parts =
#   the run's _exp.json human_arm_parts) of the REAL run (single) / the EFFECTIVE-MASS replay (comparison); one reason line. --window a,b wins.
#   Comparison: both panels at the same time (effective-mass closest step = PNG "closest at N s"; GIF / MP4 around it; uniform panel at the
#   nearest-time step), no still render. --view real|side (comparison): real = the real colour camera as single mode (1280 x 720 panels),
#   side = fixed look-at from behind / above the seated person along +y (eye 0.05,-2.8,1.2 -> target 0.05,-0.55,0.25, base frame; whole robot and
#   upper body in view); both hide the virtual keep-out box as single mode; frames frames_<tag>_<view>, outputs <run>_<tag>_cmp_<view>_*.
#   No --view = the fist look-at view as before (<run>_<tag>_cmp_*). Single: --rgb-smpl off = no SMPL on the RGB panel, outputs <run>_<tag>_nosmpl_*.
# (gifarg 11-24) GIF window and still render fitted inside the render range (fit() below: clipped, then moved to keep its length); values that
#   may start with "-" are passed as --opt=value (argparse took "-0.32,3.95" for an option: "--gif: expected one argument"); every call checks
#   its required values first (need) and stops with one line naming the empty ones. Comparison mode also writes an MP4 (compose_v5.py), and its
#   outputs are now <real run>_<tag>_cmp_closest.png / .mp4 / .gif / .json (before: <real run>_<tag>.*, the same names as single mode, overwritten).
# (baseex 11-24) --base-exclude <yaml|none> (default goal1/config/base_exclude.yaml): person frames whose pelvis is on the robot cart are not
#   shown (make_inputs: the last record before it, none -> no person; compose_single: the same for the left SMPL); none = as before.
# (passgif 11-24) comparison automatic scene (no --window, uniform + effective mass): ONE whole pass (window_v5.py --pass-json, file
#   work/<effmass run>/pass.json): same pass number / direction, TCPs within 5 cm at the start, a uniform gap <= --pick-gap inside, largest
#   effective-mass eta extra; none -> the first pass (logged). GIF = later start .. later arrival (motion_v5 goal switch), an arrived panel
#   held on its arrival frame with "arrived" (10 frames/s; over 10 MB -> 8, then narrower); PNG = the largest extra in the pass; MP4 = the
#   render start .. the two TCPs first > 10 cm apart + 1 s. One log line names the pass, starts, arrivals, PNG and MP4 end. Single mode unchanged.
# (passgif2 11-24) pass tiers: both robots arrived > the effective-mass robot arrived > any end (each with the conditions above); comparison
#   MP4 = the same pass as the GIF (15 frames/s, full size, held panels); a held panel says "arrived" or "timeout".
# (text 11-23) --compose-only: no make_inputs, no Isaac; only the composition (labels, PNG / MP4 / GIF) from the inputs and frames of the
#   last render with the same tag (work/<run>/frames_<tag>, frames_<tag>s). Labels: "Effective mass" / "Uniform" + "gap N cm (+E cm)".
set -eu
V5=$(dirname "$(readlink -f "$0")"); GG=$(cd "$V5/../../../.." && pwd); P4=$GG/real_prep4; ROOT=$(cd "$GG/.." && pwd)   # paths from the script place (any cwd)
REAL=""; if (( $# )) && [[ $1 != --* ]]; then REAL=$1; shift; fi
SINGLE=1; RA=""; RB=""
if (( $# >= 2 )) && [[ $1 != --* && $2 != --* ]]; then SINGLE=0; RA=$1; RB=$2; shift 2; fi
SENS=""; BETAS=""; CAM=""; TAG=v5b; WIN=""; CO=0; BEX=$GG/config/base_exclude.yaml; VIEW=""; RGBSMPL=on; PICKG=0.20; ATO=""; PAPER=0; MOS=off; LAVG=0
while (( $# )); do case $1 in --sensing) SENS=$2; shift 2;; --betas) BETAS=$2; shift 2;; --cam) CAM=$2; shift 2;; --tag) TAG=$2; shift 2;; --window) WIN=$2; shift 2;; --compose-only) CO=1; shift;; --base-exclude) BEX=$2; shift 2;; --view) VIEW=$2; shift 2;; --rgb-smpl) RGBSMPL=$2; shift 2;; --pick-gap) PICKG=$2; shift 2;; --at) ATO=$2; shift 2;; --paper) PAPER=$2; shift 2;; --mosaic) MOS=$2; shift 2;; --label-avg) LAVG=$2; shift 2;; --robot-exclude) export V5_ROBOT_EXCLUDE=$2; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
[[ -z $VIEW || $VIEW == real || $VIEW == side || $VIEW == fist || $VIEW == perp ]] || { echo "--view perp|real|side|fist"; exit 1; }   # (cmpview 11-24)
if (( ! SINGLE )) && [[ -z $VIEW ]]; then [[ -n $CAM ]] && VIEW=fist || VIEW=perp; fi   # (perpbox 11-24) comparison default = perp (was real)
[[ $VIEW == fist ]] && VIEW=""   # the old look-at view (frames_<tag>, names <run>_<tag>_cmp_*)
[[ $RGBSMPL == on || $RGBSMPL == off ]] || { echo "--rgb-smpl on|off"; exit 1; }
export V5_ROBOT_EXCLUDE=${V5_ROBOT_EXCLUDE:-link0,link1}   # (robex 11-24) robot spheres left out of the shown pair (robex_v5.py)
echo "[v5] robot exclude ($V5_ROBOT_EXCLUDE): $(cd "$V5" && python3 -c 'import robex_v5; print(robex_v5.ids()[1])' 2>/dev/null)"
BEXA=(); [[ $BEX == none ]] || { [[ -f $BEX ]] || BEX=$ROOT/$BEX; [[ -f $BEX ]] || { echo "--base-exclude: no file $BEX"; exit 2; }; BEXA=(--base-exclude "$BEX"); }   # (baseex 11-24)
need() {   # (gifarg 11-24) need <what is called> VAR ...: stop with one line if a required value is empty (no shifted arguments)
  local w=$1 m="" v; shift
  for v in "$@"; do [[ -n ${!v:-} ]] || m+=" $v"; done
  [[ -z $m ]] || { echo "[v5] stop before $w: empty value(s):$m"; exit 3; }
}
FIT_PY='def fit(a, b, lo, hi, tol=4 / 60.0):   # (gifarg 11-24) [a, b] inside [lo, hi]: clip one side, extend the other to keep the length b - a (as far as the range allows)
    if a >= lo - tol and b <= hi + tol:   # out by less than one MP4 frame interval (4 steps; the render grid): no frame lost, keep as given
        return a, b
    n = b - a
    if a < lo:
        a, b = lo, min(hi, lo + n)
    if b > hi:
        a, b = max(lo, hi - n), hi
    return a, b
def render_cut(w, tag):   # (cmpview 11-24) (first step, steps) of the existing render <w>/frames_<tag>: its q_cut_<tag>.npz is a slice of q.npz
    import numpy as np
    try:
        q = np.load(w + "/q.npz")["q"]; qc = np.load(w + "/q_cut_%s.npz" % tag)["q"]
    except OSError:
        return None
    n = len(qc)
    for k in range(len(q) - n + 1):
        if np.array_equal(q[k], qc[0]) and np.array_equal(q[k:k + n], qc):
            return k, n
    return None
def full_cover(rc, n, tol=4):   # (win2 11-24) the render <rc> covers the whole run of n steps (start <= tol, end >= n - 1 - tol)
    return rc is not None and rc[0] <= tol and rc[0] + rc[1] - 1 >= n - 1 - tol
def still(kc, span=30, every=3):   # (gifarg 11-24) still render start / end steps: start >= 0 on the every-3 grid ending on kc, length span kept
    s0 = kc - every * min(span // every, kc // every); return s0, s0 + span'
export FIT_PY
pf() {   # run = name, path prefix, or a file of the run (<prefix>_summary.json, _steps.npz, ...); relative to the cwd or goal1/real_prep4
  local x=${1%/}; x=$(sed -E 's/_(summary\.json|steps\.npz|human\.npz|exp\.json|scene\.json|profile\.yml|camera_pose\.yaml|mpc\.log)$//' <<< "$x")
  for c in "$x" "$P4/$(basename "$x")" "$GG/$x"; do [[ -e ${c}_summary.json ]] && { realpath -s "$c"; return 0; }; done
  echo "no run $1 (looked in the cwd and $P4)" >&2; exit 2
}
if [[ -z $REAL ]]; then   # no run given: the newest real-robot run (not a replay) in goal1/real_prep4
  REAL=$(ls -t $P4/real_*_summary.json 2>/dev/null | grep -v "_replay_" | head -1); [[ -n $REAL ]] || { echo "no real run in $P4"; exit 2; }
  REAL=${REAL%_summary.json}; echo "[v5] no run given -> newest real run: $(basename "$REAL")"
fi
REAL=$(pf "$REAL"); RN=$(basename "$REAL")
(( SINGLE )) || { RA=$(pf "$RA"); RB=$(pf "$RB"); }
if [[ -z $SENS ]]; then   # newest sensing session started before the real run (name sensing_MMDD_HHMMSS, run name ..._MMDD_HHMMSS)
  ST=$(grep -oE "[0-9]{4}_[0-9]{6}$" <<< "$RN"); SENS=$(ls -d $GG/real_prep5/sensing_[0-9]*_[0-9]* | awk -F/ -v s="sensing_$ST" '{n=$NF} n<=s{b=$0} END{print b}')
elif [[ $SENS != none ]]; then
  for c in "$SENS" "$GG/$SENS" "$GG/real_prep5/$(basename "$SENS")"; do [[ -d $c ]] && { SENS=$(realpath "$c"); break; }; done
  [[ -d $SENS ]] || { echo "no sensing folder $SENS"; exit 2; }
fi
MESH=()
if [[ $SENS != none ]]; then
  if [[ -z $(ls "$SENS/pose_npz" 2>/dev/null) ]]; then   # pose_npz is written when the camera (sensing) is stopped
    echo "[v5] $(basename "$SENS") has no pose_npz yet (the camera is still on)."
    echo $'\uce74\uba54\ub77c\ub97c \uba3c\uc800 \ub044\uc138\uc694: cd ${EFFMASS_ROOT:-$HOME/effmass_mpc} && goal1/real_prep5/rp5_human_up.sh stop'; exit 4   # (Korean text as \u escapes: the script stays ASCII)
  fi
  if [[ -z $BETAS ]]; then
    MD=$(grep -oE "_[0-9]{4}_[0-9]{6}$" <<< "$RN" | cut -c2-5); SH=$(basename "$SENS" | grep -oE "[0-9]{6}$")
    for b in "$GG/night1122_s2/betas/betas_$MD.npy" "$GG/night1122_s2/betas/betas_$SH.npy" "$V5/work/betas/betas_$SH.npy"; do [[ -f $b ]] && { BETAS=$b; break; }; done
  fi
  if [[ -z $BETAS ]]; then   # (auto) take the CLIFF betas from THIS sensing session's ROS log, as goal1/scripts/save_cliff_betas.py log
    LOG=$(python3 - "$SENS" "$REAL" "$ROOT/env_check/ros_log" <<'PY'
import glob, json, os, re, sys, time
sens, real, d = sys.argv[1:4]; name = os.path.basename(sens.rstrip("/")); pid = None
try:
    pid = json.load(open(real + "_exp.json")).get("sensing_pid")   # real_run.sh records the sensing process
except Exception:
    pass
m = re.search(r"sensing_(\d{4})_(\d{6})$", name); t0 = None
if m:   # session start (folder name, local time) -> epoch
    yr = time.localtime().tm_year; t0 = time.mktime(time.strptime("%d%s%s" % (yr, m.group(1), m.group(2)), "%Y%m%d%H%M%S"))
cands = []
for f in glob.glob(os.path.join(d, "python3_*.log")):
    mm = re.match(r"python3_(\d+)_(\d+)\.log$", os.path.basename(f))
    if not mm:
        continue
    fp, ts = int(mm.group(1)), int(mm.group(2)) / 1000.0
    if pid and fp != int(pid):
        continue
    if not pid and (t0 is None or not (t0 - 120 <= ts <= t0 + 600)):
        continue
    txt = open(f, errors="replace").read()
    if "hrc_sensing_node" in txt and "cliff beta mean" in txt and ("/" + name + "/") in txt:   # the log names this session's folder
        cands.append((abs(ts - (t0 or ts)), f))
print(sorted(cands)[0][1] if cands else "")
PY
)
    if [[ -z $LOG ]]; then echo "[v5] stop: no betas file for $(basename "$SENS") and no 'cliff beta mean' line in its sensing log under $ROOT/env_check/ros_log (give --betas)"; exit 2; fi
    SH=$(basename "$SENS" | grep -oE "[0-9]{6}$"); BETAS=$GG/night1122_s2/betas/betas_$SH.npy
    echo "[v5] betas from the sensing log $LOG -> $BETAS"
    python3 "$GG/scripts/save_cliff_betas.py" log "$LOG" "$BETAS" | tail -1
  fi
  MESH=(--mesh-real "$REAL" --sensing "$SENS" --betas "$BETAS")
fi
if (( SINGLE )); then   # ---- single-run mode
  [[ $SENS != none ]] || { echo "single-run mode needs the sensing session (RGB frames)"; exit 2; }
  if [[ -z $WIN ]]; then   # (cmpview 11-24) the TCP pass holding the REAL run's closest meeting (reason line on stderr)
    WOUT=$(nice -n 19 python3 "$V5/window_v5.py" "$REAL" --arm-m "$PICKG" "${BEXA[@]}") || { echo "[v5 single] stop: no automatic window"; exit 3; }
    WIN=${WOUT%,*}; ATS=${WOUT##*,}   # (wrap 11-24) arm within --pick-gap, largest eta extra (closest if none); a,b + the moment
  fi
  W="$V5/work/$(basename "$(dirname "$REAL")")__$RN"; mkdir -p "$W"
  exec 9> "$W/.run_v5.lock"; flock -n 9 || { echo "[v5 single] another run_v5 uses $W -> stop"; exit 5; }   # (v5b)
  echo "[v5 single] $RN, arm window $WIN s, sensing $SENS, betas $BETAS"
  need make_inputs.py REAL W WIN
  if (( CO )); then nice -n 19 python3 "$V5/robex_v5.py" lines "$REAL" "$W" || exit 3; else   # (robex 11-24) compose-only: lines.npz with the spheres left out
  nice -n 19 python3 "$V5/make_inputs.py" "$REAL" "$W" --window="$WIN" "${MESH[@]}" "${BEXA[@]}"; fi   # (text 11-23) --compose-only: inputs and frames of the last render kept
  if (( CO )); then for f in meta.json lines.npz frames_$TAG/cam0 frames_${TAG}s/cam0; do [[ -e $W/$f ]] || { echo "[v5 single] --compose-only: no $W/$f (run without --compose-only first)"; exit 2; }; done; fi
  read -r CAM0 KC T0 T1 G0 G1 S0 S1 <<< "$(python3 - "$W" "$WIN" "$CO" "$TAG" <<'PY'
import json, os, sys, numpy as np
exec(os.environ["FIT_PY"])
w = sys.argv[1]; wa, wb = [float(x) for x in sys.argv[2].split(",")]; m = json.load(open(w + "/meta.json")); z = np.load(w + "/human_v2.npz"); k = m["k_closest"]
L = np.load(w + "/lines.npz"); h = int(np.load(w + "/step_frames.npy")[max(k - 1, 0)]); b = L["b"][k]; seg = z["shape_segment"]; rad = z["shape_radius"]
side = 18 if int(L["seg"][k]) in (14, 16, 18) else 17; hc = [i for i in range(len(seg)) if seg[i] == side and rad[i] > 0]; f = max(hc, key=lambda i: rad[i])
fist = 0.5 * (z["p0"][h][f] + z["p1"][h][f]) - np.array([0, 0, 0.715]); tgt = fist + np.array([-0.01, 0.18, 0.14]); eye = tgt + np.array([0.87, 0.67, 0.16])
t = L["t"]; k0 = k % 4; k1 = len(t) - 1   # (win2 11-24) the whole run on the 4-step grid through the closest step
if sys.argv[3] == "1":   # (cmpview 11-24) --compose-only: the frames on disk decide the render range (frame i = step k0 + i)
    rc = render_cut(w, sys.argv[4])
    if rc is None:
        raise SystemExit("[v5 single] --compose-only: cannot find the render range of frames_%s (q_cut_%s.npz)" % (sys.argv[4], sys.argv[4]))
    if not full_cover(rc, len(t)):
        raise SystemExit("[v5 single] --compose-only: frames_%s cover %.2f-%.2f s, not the whole run (MP4 = whole run): render again (run without --compose-only)" % (sys.argv[4], rc[0] / 60.0, (rc[0] + rc[1] - 1) / 60.0))
    if rc[0] != k0:
        print("[v5 single] --compose-only: frames_%s start at step %d (%.2f s), not %d: using the rendered range %.2f-%.2f s" % (sys.argv[4], rc[0], rc[0] / 60.0, k0, rc[0] / 60.0, (rc[0] + rc[1] - 1) / 60.0), file=sys.stderr)
    k0, k1 = rc[0], rc[0] + rc[1] - 1
    if not k0 <= k <= k1:
        raise SystemExit("[v5 single] --compose-only: the closest step %d is outside frames_%s (steps %d-%d): run without --compose-only" % (k, sys.argv[4], k0, k1))
ga, gb = fit(wa, wb, k0 / 60.0, k1 / 60.0)   # (gifarg 11-24) GIF / MP4 span inside the render range
if (round(ga, 2), round(gb, 2)) != (round(wa, 2), round(wb, 2)):
    print("[v5 single] gif span %.2f-%.2f s outside the render range %.2f-%.2f s -> %.2f-%.2f s" % (wa, wb, k0 / 60.0, k1 / 60.0, ga, gb), file=sys.stderr)
s0, s1 = still(k)
print(",".join("%.3f" % v for v in np.r_[eye, tgt]), k, "%.4f" % (k0 / 60.0), "%.4f" % (k1 / 60.0), "%.2f" % ga, "%.2f" % gb, "%.4f" % (s0 / 60.0), "%.4f" % (s1 / 60.0))
PY
)"
  NM=${RN}_$TAG   # (v5b) outputs <run>_<tag>_closest.png, .mp4, .gif, .json
  [[ $RGBSMPL == on ]] || NM=${NM}_nosmpl   # (cmpview 11-24) no SMPL on the RGB panel: other names
  HIDE=""   # (perpbox 11-24) render_v5.sh hides the MPC-only virtual objects of the scene json in every render
  if [[ -n $CAM ]]; then   # look-at view (--cam)
    LOOK=1; export V5R_EXTRA="$HIDE"; unset V5R_EYE
  else                     # (default) the real colour camera: run pose file (roll / pitch of the sensing session) + colour K
    need cam_v5.py REAL SENS
    LOOK=0; read -r CM CA EYE TGT <<< "$(python3 "$V5/cam_v5.py" "$REAL" "$SENS")"; CAM="$EYE,$TGT"
    export V5R_EYE=$EYE V5R_EXTRA="--v5-env V5_CAM_M=$CM --v5-env V5_CAM_AP=$CA $HIDE"
  fi
  echo "[v5 single] view $( (( LOOK )) && echo "look-at $CAM" || echo "real colour camera, eye $EYE"), closest step $KC, render $T0-$T1"
  need render_v5.sh W TAG T0 T1 CAM S0 S1
  ts=$(date +%s); (( CO )) || "$V5/render_v5.sh" "$W" "$TAG" "$T0" "$T1" 2 1280x720 "$CAM"; te=$(date +%s)   # (v5b) a frame every 2 steps (MP4 every 4, GIF every 6)
  (( CO )) || "$V5/render_v5.sh" "$W" "${TAG}s" "$S0" "$S1" 3 1280x720 "$CAM"; tf=$(date +%s)   # (gifarg) still: start >= 0, ends at or after the closest step
  while [[ -e $GG/real_prep5/REAL_RUN_ACTIVE ]]; do sleep 30; done   # the SMPL renderer uses the GPU: not during a real run
  need compose_single.py REAL W SENS BETAS TAG T0 CAM LOOK NM G0 G1
  flock "$GG/real_prep5/.gpu.lock" nice -n 19 python3 "$V5/compose_single.py" --run="$REAL" --work="$W" --sensing="$SENS" --betas="$BETAS" "${BEXA[@]}" --tag="$TAG" --still-tag="${TAG}s" \
    --t0="$T0" --every=2 --cam="$CAM" --lookat="$LOOK" --out="$V5/out" --name="$NM" --gif="$G0,$G1" --mp4="$T0,$T1" --rgb-smpl="$RGBSMPL" \
    --at="${ATO:-${ATS:-}}" --paper="$PAPER" --mosaic="$MOS" --label-avg="$LAVG"; tg=$(date +%s)   # (wrap 11-24) moment, clock blur, mosaic, label average
  echo "[v5 single] wall s: Isaac window $(( te - ts )), Isaac still $(( tf - te )), compose (SMPL renderer, PNG, MP4, GIF) $(( tg - tf ))"
  exit 0
fi
REFF=$(python3 - "$RA" "$RB" <<'PY'
import json, os, sys
def kind(p):   # method of the run: exp.json, else summary human.method, else the name
    for f, g in ((p + "_exp.json", lambda d: d.get("method")), (p + "_summary.json", lambda d: (d.get("human") or {}).get("method"))):
        try:
            v = g(json.load(open(f)))
            if v:
                return v
        except Exception:
            pass
    b = os.path.basename(p); return "effmass" if "_effmass_" in b else "uniform" if "_uniform_" in b else "?"
k = [kind(p) for p in sys.argv[1:3]]
if k[0] == k[1]:   # (shviz) pair of the same kind: reference = the second run
    print(sys.argv[2], "pair")
else:
    print(sys.argv[1 + k.index("effmass")] if "effmass" in k else "", "uniform-effmass")
PY
)
read -r REFF PAIRK <<< "$REFF"
[[ -n $REFF && $REFF != uniform-effmass ]] || { echo "[v5] stop: no effective-mass run among $(basename "$RA") $(basename "$RB")"; exit 3; }
RUNI=$RA; [[ $REFF == "$RA" ]] && RUNI=$RB; AT=""
if [[ $PAIRK == pair && -z $WIN ]]; then   # (shviz 11-24) pair: the reference run's closest meeting (window_v5 single rule)
  WOUT=$(nice -n 19 python3 "$V5/window_v5.py" "$REFF" "${BEXA[@]}") || { echo "[v5] stop: no automatic window"; exit 3; }; WIN=${WOUT%,*}; AT=${WOUT##*,}
fi
PJ=""
if [[ -z $WIN ]]; then   # (passgif 11-24) one whole pass (window_v5 --pass-json: chosen pass, starts / arrivals, PNG moment, MP4 end; one log line); "gif a,b,moment"
  PJ="$V5/work/$(basename "$(dirname "$REFF")")__$(basename "$REFF")/pass.json"; mkdir -p "$(dirname "$PJ")"
  WOUT=$(nice -n 19 python3 "$V5/window_v5.py" "$REFF" --uni "$RUNI" --uni-m "$PICKG" --pass-json "$PJ" "${BEXA[@]}") || { echo "[v5] stop: no automatic window"; exit 3; }
  WIN=${WOUT%,*}; AT=${WOUT##*,}
fi
[[ -n $ATO ]] && AT=$ATO   # (shviz) --at
echo "[v5] real $RN, replays $(basename $RA) $(basename $RB), window $WIN s, sensing ${SENS:-none}, betas ${BETAS:-none}"
WA=(); FD=6; for R in "$RA" "$RB"; do W="$V5/work/$(basename "$(dirname "$R")")__$(basename "$R")"; mkdir -p "$W"
  eval "exec $FD> \"\$W/.run_v5.lock\""; flock -n $FD || { echo "[v5] another run_v5 uses $W -> stop"; exit 5; }; FD=$(( FD + 1 ))   # (v5b) one run per work dir
  need make_inputs.py R W WIN
  if (( CO )); then [[ -f $W/meta.json ]] && { nice -n 19 python3 "$V5/robex_v5.py" lines "$R" "$W" || exit 3; }; else   # (robex 11-24) compose-only: lines.npz with the spheres left out
  nice -n 19 python3 "$V5/make_inputs.py" "$R" "$W" --window="$WIN" "${MESH[@]}" "${BEXA[@]}"; fi; WA+=("$W"); done
TAGR=$TAG; [[ -z $VIEW ]] || TAGR=${TAG}_$VIEW   # (cmpview) frames of this view
read -r WU WE CAM0 T0 T1 G0 G1 <<< "$(python3 - "${WA[@]}" "$WIN" "$CO" "$TAGR" "$AT" <<'PY'
import json, os, sys, numpy as np
exec(os.environ["FIT_PY"])
ws = sys.argv[1:3]; wa, wb = [float(x) for x in sys.argv[3].split(",")]; m = [json.load(open(w + "/meta.json")) for w in ws]   # argv 4 = compose-only, 5 = frame tag, 6 = moment
if sorted(x["kind"] for x in m) == ["effmass", "uniform"]:
    u = ws[[x["kind"] for x in m].index("uniform")]; e = ws[[x["kind"] for x in m].index("effmass")]
    ref = [x["kind"] for x in m].index("effmass")   # (cmpview) reference = the effective-mass run
else:   # (shviz 11-24) pair of the same kind: left = the first, right = the second (reference)
    u, e, ref = ws[0], ws[1], 1
L = np.load(ws[ref] + "/lines.npz"); k = int(np.argmin(np.abs(L["t"] - float(sys.argv[6])))) if sys.argv[6] else int(m[ref]["k_closest"])   # (win2) reference step = the moment
z = np.load(ws[ref] + "/human_v2.npz"); h = int(np.load(ws[ref] + "/step_frames.npy")[max(k - 1, 0)])
L = np.load(ws[ref] + "/lines.npz"); b = L["b"][k]; seg = z["shape_segment"]; rad = z["shape_radius"]
cand = [i for i in range(len(seg)) if seg[i] in (17, 18) and rad[i] > 0]   # the hand capsule (largest) nearest to the closest arm point
c = [0.5 * (z["p0"][h][i] + z["p1"][h][i]) - np.array([0, 0, 0.715]) for i in cand]; ib = min(range(len(cand)), key=lambda j: np.linalg.norm(c[j] - b))
side = 18 if seg[cand[ib]] == 18 else 17; hc = [i for i in cand if seg[i] == side]; f = max(hc, key=lambda i: rad[i]); fist = 0.5 * (z["p0"][h][f] + z["p1"][h][f]) - np.array([0, 0, 0.715])
tgt = fist + np.array([-0.01, 0.18, 0.14]); eye = tgt + np.array([0.87, 0.67, 0.16])   # S1 choice: target (0.38,-0.62,0.46), eye (1.25,0.05,0.62)
kc = k; k0 = kc % 4; t1 = (len(L["t"]) - 1) / 60.0   # (win2 11-24) the whole run on the 4-step grid through the reference step
if sys.argv[4] == "1":   # (cmpview 11-24) --compose-only: the frames on disk decide the render range
    rc = render_cut(ws[ref], sys.argv[5])
    if rc is None:
        raise SystemExit("[v5] --compose-only: cannot find the render range of frames_%s (q_cut_%s.npz)" % (sys.argv[5], sys.argv[5]))
    if rc[0] != k0:
        print("[v5] --compose-only: frames_%s start at step %d (%.2f s), not %d: using the rendered range" % (sys.argv[5], rc[0], rc[0] / 60.0, k0), file=sys.stderr)
    if not full_cover(rc, len(L["t"])):
        raise SystemExit("[v5] --compose-only: frames_%s cover %.2f-%.2f s, not the whole run (MP4 = whole run): render again (run without --compose-only)" % (sys.argv[5], rc[0] / 60.0, (rc[0] + rc[1] - 1) / 60.0))
    k0, t1 = rc[0], (rc[0] + rc[1] - 1) / 60.0
    if not k0 <= kc <= rc[0] + rc[1] - 1:
        raise SystemExit("[v5] --compose-only: the closest step %d (%.2f s) is outside frames_%s (%.2f-%.2f s): run without --compose-only" % (kc, kc / 60.0, sys.argv[5], k0 / 60.0, t1))
    if (kc - k0) % 2:   # (win2b) the moment between two rendered steps: the earlier rendered step (1 step = 17 ms)
        print("[v5] --compose-only: step %d is between rendered steps of frames_%s (start %d, every 2): using step %d" % (kc, sys.argv[5], k0, kc - 1), file=sys.stderr); kc -= 1
ga, gb = fit(wa - 0.4, wb + 0.4, k0 / 60.0, t1)   # (gifarg 11-24) GIF / MP4 span inside the render range [k0 / 60, t1]
if (round(ga, 2), round(gb, 2)) != (round(wa - 0.4, 2), round(wb + 0.4, 2)):
    print("[v5] gif span %.2f-%.2f s outside the render range %.2f-%.2f s -> %.2f-%.2f s" % (wa - 0.4, wb + 0.4, k0 / 60.0, t1, ga, gb), file=sys.stderr)
print(u, e, ",".join("%.3f" % v for v in np.r_[eye, tgt]), "%.4f" % (k0 / 60.0), "%.2f" % t1, "%.2f" % ga, "%.2f" % gb)
PY
)"
[[ -n ${T1:-} ]] || { echo "[v5] stop: the comparison inputs failed (see the lines above; one uniform and one effective-mass run are needed)"; exit 3; }
RES=640x480; NMC=${RN}_${TAG}_cmp; RCA=(); HIDE=""   # (cmpview 11-24) per view; (perpbox) virtual objects hidden by render_v5.sh
case $VIEW in
  real) [[ $SENS != none && -n $SENS ]] || { echo "[v5] --view real needs the sensing session (camera roll / pitch)"; exit 2; }
        need cam_v5.py REAL SENS
        read -r CM CA EYE TGT <<< "$(python3 "$V5/cam_v5.py" "$REAL" "$SENS")"; CAM="$EYE,$TGT"; RES=1280x720
        export V5R_EYE=$EYE V5R_EXTRA="--v5-env V5_CAM_M=$CM --v5-env V5_CAM_AP=$CA $HIDE"; RCA=(--realcam="$REAL" --sensing="$SENS");;
  side) CAM=${CAM:-0.05,-2.8,1.2,0.05,-0.55,0.25}; export V5R_EXTRA="$HIDE"; unset V5R_EYE;;
  perp) CAM=${CAM:--2.274,-0.556,0.550,0.023,-0.434,0.200}; export V5R_EXTRA="$HIDE"; unset V5R_EYE;;   # (perpbox 11-24) perpendicular to the pelvis - robot base line (t17), from -x
  *)    CAM=${CAM:-$CAM0};;
esac
[[ -z $VIEW ]] || NMC=${NMC}_$VIEW
[[ $PAIRK == pair ]] && NMC=${NMC}_pair_$(basename "$RA")_$(basename "$RB")   # (shviz) a same-kind pair does not overwrite the uniform / effmass comparison
echo "[v5] view ${VIEW:-fist look-at}, camera $CAM, render $T0-$T1 s ($RES, frames_$TAGR), gif $G0-$G1 s"
need render_v5.sh WU WE TAGR T0 T1 CAM RES
(( CO )) || for W in "$WU" "$WE"; do "$V5/render_v5.sh" "$W" "$TAGR" "$T0" "$T1" 2 "$RES" "$CAM"; done   # (v5b) a frame every 2 steps (GIF every 6, MP4 every 4)
if (( CO )); then for W in "$WU" "$WE"; do [[ -d $W/frames_$TAGR/cam0 ]] || { echo "[v5] --compose-only: no $W/frames_$TAGR (run without --compose-only first)"; exit 2; }; done; fi
need compose_v5.py WU WE TAGR CAM T0 NMC G0 G1
nice -n 19 python3 "$V5/compose_v5.py" --a="$WU" --b="$WE" --tag="$TAGR" --cam="$CAM" --t0="$T0" --every=2 --out="$V5/out" --name="$NMC" --gif="$G0,$G1" --mp4="$T0,$T1" --at="$AT" --view="${VIEW:-fist}" ${PJ:+--pass-json="$PJ"} "${RCA[@]}"
