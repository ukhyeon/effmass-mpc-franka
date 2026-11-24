#!/usr/bin/env bash
# render_v5.sh <work dir> <tag> <t0 s> <t1 s> <every> <WxH> <cams> - viz2 v5: ONE Isaac re-render (step2b through launch_v5.py) of the inputs in
# <work dir> (make_inputs.py) from t0 to t1 s of the run, a frame every <every> steps (60 Hz steps), cameras "eye,target[;eye,target]"
# (robot base frame). Under the shared GPU lock (flock goal1/real_prep5/.gpu.lock, waits while REAL_RUN_ACTIVE exists), nice -n 19.
# Shell look (fix3, same for both methods; chosen on the rendered S1 frames): opacity V5R_OPA (0.18), emission V5R_EMIS (0.1), ior 1 (no
# white Fresnel sheen), colours V5R_U (0.05,0.30,1.00) / V5R_E (0.95,0.20,0.00), dark outline V5R_EU / V5R_EE seen from the first camera eye
# (width V5R_EDGE_W 0.006 m). V5R_EYE = outline eye x,y,z (default: the first camera's eye); V5R_EXTRA = more launcher words
# ("--v5-env K=V ...": V5_CAM_M / V5_CAM_AP = matched real camera, V5_HIDE = prims hidden in the render). Frames: <work dir>/frames_<tag>/cam<j>/f<i>.png, frame i = step t0*60 + i. No robot, no control PC, no sensing; ROS local only (domain 81).
# step2b also writes its usual run files to goal1/logs/meff_isaac/v5_<tag>*.
# (robex2 11-24) eta volumes: the robot spheres of robex_v5.py ids <run> (the run's recorded human_robot_exclude, else V5_ROBOT_EXCLUDE =
# run_v5 --robot-exclude, default link0,link1) are never the nearest sphere (env V5_ROBEX_IDX of g1_viz_v5); none -> no env (as before).
# (v5b, 11-23) The step cut files are per tag (<work dir>/q_cut_<tag>.npz, step_frames_cut_<tag>.npy, lines_cut_<tag>.npz): before, every tag
# wrote q_cut.npz, so a still render (31 steps) of a second run_v5 on the same work dir replaced the window render's input while it waited for
# the GPU lock, and the window render showed one pose all along (step2b holds the last step). One render per work dir + tag at a time
# (<work dir>/.render_<tag>.lock, else exit 7); after Isaac the steps step2b read (log) must equal the cut, else exit 8.
set -u
W=$(realpath "${1:?work}"); TAG=${2:?tag}; T0=${3:?t0}; T1=${4:?t1}; EV=${5:?every}; RES=${6:?res}; CAMS=${7:?cams}
V5=$(dirname "$(readlink -f "$0")"); GG=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1; L=$GG/real_prep5/.gpu.lock; O=$GG/logs/meff_isaac; P=$O/v5_$TAG; mkdir -p $O
exec 8> "$W/.render_$TAG.lock"; flock -n 8 || { echo "$(date +%T) v5 $TAG: another render of $W with tag $TAG runs -> stop"; exit 7; }   # (v5b)
read -r D0 KG EM KIND SCN K0 N RGB EDGE HIDE5 <<< "$(python3 - "$W" "$T0" "$T1" "$TAG" <<'PY'
import json, sys, os, numpy as np
w, t0, t1, tag = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]; m = json.load(open(f"{w}/meta.json")); k0 = int(round(t0 * 60)); k1 = int(round(t1 * 60))
if k0 < 0 or k1 <= k0:
    sys.exit("render_v5: steps %d..%d (t0 %.4f s, t1 %.4f s): t0 must be >= 0 and before t1" % (k0, k1, t0, t1))   # (v5b fix: a negative t0 sliced q from the end)
q = np.load(f"{w}/q.npz")["q"]; sf = np.load(f"{w}/step_frames.npy"); L = np.load(f"{w}/lines.npz"); k1 = min(k1, len(q) - 1)
np.savez(f"{w}/q_cut_{tag}.npz", q=q[k0:k1 + 1]); np.save(f"{w}/step_frames_cut_{tag}.npy", sf[k0:k1 + 1]); np.savez(f"{w}/lines_cut_{tag}.npz", a=L["a"][k0:k1 + 1], b=L["b"][k0:k1 + 1])   # (v5b) per tag
z = np.load(f"{w}/human_v2.npz"); scn = str(np.load(m["run"] + "_human.npz")["scene_file"])
rgb = os.environ.get("V5R_U", "0.05,0.30,1.00") if m["kind"] == "uniform" else os.environ.get("V5R_E", "0.95,0.20,0.00")   # (fix3) shell colours (render_v5 env overrides)
rgb = rgb + " " + (os.environ.get("V5R_EU", "0.00,0.10,0.45") if m["kind"] == "uniform" else os.environ.get("V5R_EE", "0.50,0.10,0.00"))   # outline colour
try:   # (perpbox 11-24) MPC-only virtual objects of the scene json (plate names with "virtual" / "keepout"): hidden in every render
    hide = ["/World/obstacles/" + p["name"] for p in json.load(open(scn)).get("plates", []) if "virtual" in p["name"] or "keepout" in p["name"]]
except Exception:
    hide = ["/World/obstacles/keepout_virtual"]
print(m["d0"], m["gain"], m["eta_max"], m["kind"], scn, k0, k1 - k0 + 1, rgb, ",".join(hide) or "-")
PY
)"
[[ -n ${N:-} ]] || { echo "$(date +%T) v5 $TAG: no step cut (see the line above) -> stop"; exit 9; }   # (v5b fix)
HV=shapes; [[ -f $W/human_verts.npz ]] && HV=mesh
RUNP=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['run'])" "$W/meta.json")   # (win2 11-24) eta volumes on the record's arm parts
AP=$(python3 "$V5/parts_v5.py" "$RUNP"); AP=${AP:-none}; echo "$(date +%T) v5 $TAG: eta volume parts (segment ids) $AP"
read -r MM5 IO5 <<< "$(python3 "$V5/parts_v5.py" --rule "$RUNP")"; RULE5=""   # (shviz 11-24) recorded eta rule; env only when not the old default
if [[ ${MM5:-reduced} != reduced || ${IO5:--} != - ]]; then RULE5="--v5-env V5_MASS_MODE=$MM5 --v5-env V5_ISO_OVERRIDE=$IO5"; fi
echo "$(date +%T) v5 $TAG: eta rule ${MM5:-reduced}, ISO override ${IO5:--}; hidden (MPC-only virtual objects of $(basename "$SCN")): ${HIDE5:--}"
RX5=$(cd "$V5" && python3 robex_v5.py ids "$RUNP" 2> "$W/.robex_$TAG.txt"); RX5=${RX5:--}; RXA=""   # (robex2 11-24) robot spheres left out of the eta volumes
[[ $RX5 != - ]] && RXA="--v5-env V5_ROBEX_IDX=$RX5"; echo "$(date +%T) v5 $TAG: eta volumes: $(cat "$W/.robex_$TAG.txt" 2>/dev/null) -> indices $RX5"
rm -rf $W/frames_$TAG
echo "$(date +%T) v5 $TAG: $KIND d0 $D0 gain $KG eta_max $EM, steps $K0 + $N, every $EV, human $HV"
W0=0; while [[ -e $GG/real_prep5/REAL_RUN_ACTIVE ]]; do (( W0 )) || echo "$(date +%T) v5 $TAG: REAL_RUN_ACTIVE exists -> waiting (lock not held)"; W0=1; sleep 30; done
flock $L bash -c "while [[ -e $GG/real_prep5/REAL_RUN_ACTIVE ]]; do sleep 30; done; \
  EFFMASS_WS=1 RUN_ISAAC_NO_DISPLAY=1 ROS_METHOD=min RIR_DOMAIN_ID=81 RIR_LOCALHOST_ONLY=1 \
  timeout -s INT -k 20 900 nice -n 19 ${EFFMASS_ROOT:-$HOME/effmass_mpc}/env_check/scripts/run_isaac_ros.sh $V5/launch_v5.py --v5-env V5_RGB=$RGB --v5-env V5_LINES=$W/lines_cut_$TAG.npz --v5-env V5_OPACITY=${V5R_OPA:-0.18} --v5-env V5_EMIS=${V5R_EMIS:-0.1} --v5-env V5_EYE=${V5R_EYE:-${CAMS%%;*}} ${V5R_EXTRA:-} --v5-env V5_HIDE=${HIDE5:--} --v5-env V5_ARM_PARTS=$AP $RULE5 $RXA --v5-env V5_EDGE_W=${V5R_EDGE_W:-0.006} --v5-env V5_EDGE_RGB=$EDGE --v5-env V5_PERSON_RGB=0.32,0.32,0.34 --v5-env G1_MESH_OPACITY=0.6 --scene g1 --robot fr3v2_1 --no-diag --warmup 10 \
  --tag v5_$TAG --ros --ros-pose-topic none --human --human-mode full --human-source precomputed --ready-file $P.ready --done-file $P.done --ros-out $P \
  --scene g2 --shapes v2 --human-dir $GG/b2a/g2 --human-pre $W/human_v2.npz --human-frame $GG/config/live_frame_rp5.yml --human-step-frames $W/step_frames_cut_$TAG.npy \
  --human-viz $HV --viz-hide-com on --u-def nearest --cost-agg 0 --profile $GG/config/live_t1.yml --round-trip 1 --steps $N --scene-json $SCN \
  --meff on --meff-mode gain --meff-dbase $D0 --meff-muref 0.53 --meff-eta-max $EM --meff-agg max_all --meff-muref-mode part --b1-log 0 --meff-gain $KG --mppi-seed 0 \
  --playback-q $W/q_cut_$TAG.npz --viz-eta-shell arm '--viz-cam=$CAMS' --viz-record $W/frames_$TAG --viz-record-res $RES --viz-record-every $EV > $P.log 2>&1"
ec=$?
grep -ah "(v5) scene objects" $P.log | tail -1 | sed "s/^/$(date +%T) v5 $TAG: /"   # (perpbox 11-24) what the render drew
NQ=$(grep -aoE "playback of .*: [0-9]+ recorded steps" $P.log | grep -oE "[0-9]+ recorded" | cut -d' ' -f1); NF=$(grep -aoE "human frame per step from .*\(([0-9]+) steps\)" $P.log | grep -oE "[0-9]+ steps" | cut -d' ' -f1)
if (( ec == 0 )) && [[ ${NQ:-x} != $N || ${NF:-x} != $N ]]; then echo "$(date +%T) v5 $TAG: step2b read ${NQ:-?} robot steps / ${NF:-?} human steps, the cut has $N -> stop"; ec=8; fi   # (v5b)
echo "$(date +%T) v5 $TAG exit $ec, frames $(ls $W/frames_$TAG/cam0 2>/dev/null | wc -l)"
grep -aE "loop_wall_s|realtime_factor|\(viz\) \{" $P.log | cut -c1-200
grep -aE "Traceback" -A10 $P.log | head -14
exit $ec
