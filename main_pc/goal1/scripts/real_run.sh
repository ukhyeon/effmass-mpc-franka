#!/usr/bin/env bash
# real_run.sh <scene: ab|ym|ymw|c1|c1k|c2|c3> <MPC seconds> [--fake] [--fast] [--smooth[=<m/s>]] - one command for a run on the control PC (2026-11-20).
#   ab = goal1/real_prep5/scene_AB_rp5floor.json, ym = goal1/real_prep5/scene_AB_ym_rp5floor.json (floor fitted, real_prep5;
#   ym = the same goals turned -90 deg, robot's right; ymw = goal1/real_prep5/scene_AB_ymw_rp5floor.json, ym widened: A x +0.40, B x -0.40).
#   (2026-11-20 night) ym now uses goal1/real_prep5/scene_AB_ym_rp5floor_table.json = scene_AB_ym_rp5floor.json (kept) + the real robot
#   table as a solid box in the MPC world (plate robot_table_real: x, y -0.335 .. +0.335, top z 0, down to the floor; size / position:
#   goal1/real_prep5/demo/make_scenes.py --table or the plate entry). Demo scenes (goal1/real_prep5/demo/, same table plate):
#   c1 = scene_C1_rp5floor.json, wide left-right A x +0.45 <-> B x -0.45 (y -0.55, z 0.3235); c2 = scene_C2_rp5floor.json, diagonal
#   up-right (+0.35, -0.55, 0.45) <-> low-left (-0.35, -0.55, 0.15); c3 = scene_C3_rp5floor.json, pick-place cycle of 6 goals (MPC
#   gets --goal-cycle 1). Result names and archive as for the other scenes (<real|fake>_<scene>_<MMDD_HHMMSS>).
#   c1k (2026-11-21 night) = scene_C1k_rp5floor.json: c1 + a VIRTUAL keep-out box beside the table (x -0.8..0.8, y -1.0..-0.335,
#   below z 0.05; nothing real there) so that the robot does not duck under a hand beside the table (one c1 replay: 0.026 m).
#   1. shows the plan, asks for the word "start" in Korean (one time); anything else ends.
#   Remote start (2026-11-20): started directly by ssh the gateway could not reach the main PC (by hand it could; cause unknown).
#   So the commands are typed into the tmux session "franka" on the control PC (ssh + tmux send-keys; it must exist and sit at
#   a shell prompt, else stop). Progress from run.log, end from the exit-code file <RES>/<session>/exit_<step>.txt that
#   run_real_step3.sh writes ("EXIT=<code> <time> <result folder>"). Nothing is killed remotely from here.
#   2. step 0: run_real_step3.sh --execute 0 --confirmed (home), waits for exit_0.txt (120 s); skipped with --fake. Fails -> stop.
#   3. step 5: run_real_step3.sh --execute 5 --confirmed --end-after-idle 2 --duration 300 [--fake]; waits up to 60 s for the
#      "waiting for targets" line in its run.log. (2026-11-20 waits: idle 5 -> 2 s; remote polls every 0.3 s instead of 1 s.)
#   4. /c6/joint_states for 4 s (timeout -k 2 4; 10 s before 2026-11-20) with ros2 topic hz (daemon stopped before and after); not
#      900-1100 per s -> the MPC is not started. With --human the human re-check (4 s) runs at the same time.
#   5. rp4_run.sh c6 <name> <MPC seconds>, name = <real|fake>_<scene>_<MMDD_HHMMSS>, results goal1/real_prep4/<name>_*.
#   6. prints the main PC summary as soon as the MPC ends. Starts the control PC analysis of the bag (fake:
#      real_step3_fake/analyze_step3_fake.py) as soon as its run.log shows the stop is done (bag closed: "remaining processes" line),
#      while the control PC still checks the robot mode; then waits for its "end. result:" line and exit code, as before.
#   7. Ctrl+C: the MPC stops first (it handles SIGINT: no more targets, records kept). The control PC is left to end by itself
#      (no new targets for 5 s -> its own stop order); this script then still waits for its result. Ctrl+C again: leave at once.
#   --fast (2026-11-20): MPC profile goal1/config/real_c05_acc10.yml (sphere speed limit 0.50 m/s) instead of real_slowc_acc10.yml,
#   and step 5 on the control PC gets --vel-scale 0.90 --acc 9.8 --jerk $FAST_JERK (2026-11-20 speed: the bridge is a safety line near
#   the robot specification, the real speed is set on the MPC side; FAST_JERK at the top, 1500 from the control PC recalculation;
#   before: --vel-scale 0.40 --acc 8.0 --jerk 1500). Without --fast: as before.
#   Bridge smoothing (2026-11-20 evening): step 5 (only; step 0 would end with code 1) also gets --bridge-smooth; without --fast
#   --bridge-smooth --vel-scale 0.30, with --fast --vel-scale 0.90 --acc 9.8 --jerk $FAST_JERK --bridge-smooth. Shown in the header and
#   in the result summary ("bridge  :" line). Without --fast and without --speed: --vel-scale 0.55 (2026-11-20 evening, was 0.30:
#   1.6 x the joint speed demand 0.318 of spec in the no-human reference real_ym_1120_174138, rounded up to 0.05). The script waits for the "waiting for targets" line before the MPC starts (no fixed wait).
#   --speed V / --speed=V (2026-11-20; no default): one MPC speed value, V 0.05..0.60 m/s (inside the bridge --vel-scale 0.90).
#   The profile of this run (after --fast and the --human .h1_choice) is copied to goal1/real_prep4/<name>_profile.yml with
#   settings sphere_speed_limit = V and speed_cap_v = 0.8 x V (only these two lines changed, comments kept); the MPC gets the copy.
#   The original profile and rp2_mpc_rt.py are not changed. With --smooth also --lim-v V (with --smooth=X, X must equal V).
#   The plan line shows the copy path and the two values (profile -> copy).
#   --speed with --human (2026-11-20 evening): allowed; the copy is made from the .h1_choice PROFILE. Without --fast the bridge
#   --vel-scale (step 5) follows V: 1.6 x the joint speed demand at V (VS_K x V, of the FR3 joint speed specification; VS_K at the top,
#   from replays of real_ym_1120_174451, goal1/real_prep5/filt/sp_table.py), rounded up to 0.05, within 0.30..0.90. With --fast: 0.90.
#   The result summary shows V, lim-v, the profile copy and the bridge vel-scale ("speed   :" line).
#   --smooth[=V] (real_prep5 overnight filter, 2026-11-20): smoother MPC targets. The MPC sphere speed limiter scales the first point
#   toward the previously sent point instead of the measured joints (rp2_mpc_rt.py --lim-ref cmd --lim-v V, added to env MPC_EXTRA), so
#   the measured-joint noise no longer enters the target. V (m/s, commanded path) default 0.14, with --fast 0.28 = the speed the old
#   limiter gave on the real robot (goal1/real_prep5/OVERNIGHT_FILTER.md). Without --smooth: as before.
#   Each run uses its own control-PC session folder (<RES>/<YYYYmmdd_HHMMSS>_main_<scene>) so that nothing is overwritten.
#   --human (H2 prep, 2026-11-20; default off = as before): MPC with the live human input. The user starts sensing
#   (goal1/real_prep5/rp5_sensing.sh) and the human node (goal1/scripts/g1_run_human.sh node ...) first, domain 30 localhost (camera,
#   IMU correction); this script does not start them. Settings from goal1/real_prep5/.h1_choice (read as text, never executed;
#   KEY=VALUE words: MPS=0|1, PROFILE=<yml>, FAST_PROFILE=<yml> (used with --fast), MPC_EXTRA="<rp2_mpc_rt.py options>",
#   SKIP_LATE=0|1 (default 1), SENSING_MPS_THREADS=<1..100> (with MPS=1: the running sensing must have
#   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<value>, i.e. started with RP5_MPS_THREADS=<value>, else stop); other words are ignored and
#   listed). With --fast and a PROFILE but no FAST_PROFILE: stop (untested combination). MPC_EXTRA must not set --lim-* or
#   --skip-late (set by --smooth / SKIP_LATE). No file: the profile above + --skip-late 1.
#   MPS=1: the MPS daemon (goal1/real_prep5/rp5_mps.sh) must run and sensing must have started after it, else stop with the commands
#   (MPS is never switched on here: sensing has to start after it). Before the plan: sensing / human node processes, then
#   rp5_human_relay.py (domain 30, g1_run_human.sh py) -> shared memory rr5_human and a 5 s look (goal1/real_prep5/rp5_human_check.py;
#   runs while the plan waits for the start word, judged after it). Just before the MPC, at the same time as the joint-state
#   check: 4 s, at least 15 human messages per s, else no MPC (exit 10; the control PC
#   gets no targets and ends by itself). MPC (rp4_run.sh c6, EFFMASS_WS=1) gets --human shm --human-shm rr5_human --human-frame
#   goal1/config/live_frame_rp5.yml --skip-late 1 + MPC_EXTRA of the file, as rp5_run_human.sh. Summary: human rate and age, minimum gap
#   robot sphere - human shape (measured joints) with the robot link and the body part. --human also passes --rec-human 1:
#   goal1/real_prep4/<name>_human.npz = human records the MPC used, per-step record number / age, Top-K mask, nearest distance per
#   robot sphere and link (rp2_mpc_rt.py --rec-human, for a replay with the same human). With --smooth: V 0.14, with --fast 0.28 (as above).
#   Exit codes of this script: 0 done (control PC exit 0), 1 usage, 2 scene / file, 4 not confirmed, 5 tmux / step 0, 6 control PC not
#   waiting for targets, 7 joint-state rate (the main PC check, not the control PC's code 7), 8 no control PC end line, 9 the control PC
#   run ended with a non-zero code (incl. 9 = control PC home move failure with --home-in-bridge, before any target), 10 human input check, 11 .h1_choice / MPS / sensing order, 12 GPU check, 130 Ctrl+C before the MPC.
#   Exit code 12 is the MAIN PC GPU check of this script (below), not a control PC code.
#   Control PC exit value record (exp1121 2nd, every run that started anything on the control PC): goal1/real_prep4/<name>_ctrl_exit.json
#   = step, control_exit (the EXIT= value of the control PC exit file, null = none / no end line), real_run_exit (this script's code),
#   reflex_stop (control_exit 8); with --human also copied into <name>_exp.json. This script's code 8 has two meanings (no control PC
#   end line; control PC reflex during the --home-in-bridge move): the record tells them apart (analysis: reflex = control_exit 8).
#   --home-in-bridge (2026-11-21, item 9; default off = as before: step 0 then step 5): step 0 (--execute 0) is NOT called; step 5 gets
#   --home-in-bridge (control PC run_real_step3.sh: the bridge first moves to the step-0 home with Ruckig and drops targets meanwhile;
#   run.log, in Korean: "home move start" (only if it moves) -> "home arrived (...)" -> "gateway up" -> "waiting for targets"). The wait for it
#   is 60 + 40 s; meanwhile EXIT=9 / EXIT=8 in the control PC exit file or a "home move failed" line in its run.log stop the run before
#   the MPC starts (no target is ever sent; the human relay / replay is stopped as on any exit; the control PC runs its own stop order)
#   and this script waits for the control PC exit code (60 s) and ends with the SAME code: 9 = control PC home move failed, 8 = control
#   PC reflex stop during the home move (note: outside this case 8 / 9 keep their meanings below). Such runs get <name>_exp.json with
#   aborted "home" (code 9) for the analysis (goal1/real_prep5/exp1121/exp_ana.py lists code 9 runs as aborted and leaves them out).
#   The MPC is started only after "waiting for targets" (as without the option).
#   Reflex stop (2026-11-21 night; exp1121: code only): control PC exit code 8 (EXIT=8 in its exit file) is shown as
#   "robot reflex stop (collision detected) - clear the error in Desk" (step 0 line, "reflex  :" line of the result summary); the
#   judgement does not change (any non-zero code = failed, exit 9). The exit file text is not searched.
#   --method uniform|effmass, --pose H|L|none, --trial N (exp1121, 2026-11-21; with --human only): avoidance method of the human
#   cost (rp2_mpc_rt.py --human-method; effmass = default = as before, k_gain 0.10; uniform = k_gain 0, clearance d_base in every
#   direction; the risk field and all other options are the same .h1_choice ones), the arm posture of the comparison (H: elbow 90 deg,
#   fist toward the oncoming robot; L: straight arm into the path; G (exp1121 proposal): elbow 90 deg, lower arm on the path line just
#   beyond goal A, fist toward the robot coming to A; none: free) and the trial number (default: 1 + the number of results
#   of the same mode / scene / method / pose in goal1/real_prep4). With --human the result name is
#   <real|fake>_<scene>_<method>_<pose>_t<trial>_<MMDD_HHMMSS> (without --human: as before) and goal1/real_prep4/<name>_exp.json keeps
#   method, pose, trial, scene, speed, smooth and the MPC options (and, after the run, the control PC exit code); the result summary
#   has an "exp     :" line.
#   --k-gain K, --eta-max E, --other-d-base D, --other-k-gain K2 (kgain 2026-11-23; with --human or --human-replay; default: not given
#   = as before): rp2_mpc_rt.py --human-k-gain / --human-eta-max / --human-other-d-base / --human-other-k-gain (effective-mass gain,
#   thickness upper bound, base clearance and gain of the non-arm segments), added at the end of the MPC human options. --k-gain only
#   with --method effmass. <name>_exp.json keeps human_k_gain, human_eta_max, human_other_d_base, human_other_k_gain (values used,
#   also without the options) and human_arm_parts.
#   --mass-mode reduced|human (massmode 2026-11-24; with --human or --human-replay; default reduced = as before, nothing passed):
#   rp2_mpc_rt.py --human-mass-mode (eta reference: human-robot reduced mass mu, or the human effective mass m_h alone; the risk field
#   follows eta). <name>_exp.json keeps human_mass_mode (also without the option); MPC_EXTRA must not set --human-mass-mode.
#   (today 1124) .h1_choice MASS_MODE=reduced|human is the default of --mass-mode (the command option wins), for live, replay and
#   both methods; HUMAN_CHEST_SPINE3=<value> (set by the other session, read by the human-node side) is only recorded here.
#   <name>_exp.json: human_mass_mode (+ human_mass_mode_source command|.h1_choice|default), human_shoulder (0/1, from
#   --human-shoulder in the MPC options), human_chest_spine3 (the .h1_choice value, null without the key).
#   (recplan 1124) human_rec_plan 0/1 (rp2 --rec-plan in the MPC options, e.g. .h1_choice MPC_EXTRA).
#   (cam20 1124) --camera-pose <yml> (default .h1_choice CAMERA_POSE=<yml>, else none = as before; human runs): camera pose file of the
#   person's world coordinates. Live: the running sensing must use it (its newest debug_log camera_pose_*.yaml t = the file's t, else
#   stop: restart the sensing with rp5_human_up.sh, which gives CAMERA_POSE to the sensing as RP5_POSE). Replay: the recorded people
#   are moved by t(file) - t(recording's camera pose: <rec>_camera_pose.yaml, else the human npz camera_pose_text) with rp2
#   --human-shift (nothing when equal). _exp.json camera_pose {file, t, source, recording_t, human_shift}.
#   --leg-timeout S (cam20 1124; default: rp2 400 steps = 6.67 s): rp2 --leg-timeout S (also possible in .h1_choice MPC_EXTRA; the
#   command option wins). _exp.json leg_timeout_s (value used) and leg_timeout_source.
#   --arrive-pos-tol M (arrtol 1124; default: rp2 0.01 m): rp2 --arrive-pos-tol M (goal switch on arrival; orientation 0.05 rad
#   unchanged; also possible in .h1_choice MPC_EXTRA, the command option wins). _exp.json arrive_pos_tol + arrive_pos_tol_source.
#   --d-base D (arrtol 1124; human runs): rp2 --human-d-base D after the .h1_choice MPC_EXTRA (the last value wins, both methods).
#   _exp.json human_d_base = the value used (last --human-d-base of the MPC human options; rp2 default 0.065).
#   (robex 1124) human_robot_exclude: the links of rp2 --human-robot-exclude in the MPC options (e.g. .h1_choice MPC_EXTRA), null without
#   it; after the run the MPC summary human.robot_exclude (links, sphere indices, count) replaces it.
#   --human-replay <name>_human.npz (exp1121): human run WITHOUT the live human: the recorded human data of an earlier run
#   (rp2_mpc_rt.py --rec-human, e.g. goal1/real_prep4/real_c1k_effmass_H_t1_<stamp>_human.npz) is written at the recorded speed by
#   goal1/real_prep5/h2cost/rp5_replay_human.py into its own shared memory rr5_replay (30 records per s of the first record until the
#   timeline starts); the MPC reads that one. Timeline t = 0 = the "warm reset" line of this run's MPC log = the MPC start (as t = 0 of
#   the recording = its first publish), i.e. after step 0 (home) and the warm reset at the same start joints. The human checks (5 s,
#   4 s, 15 per s) look at the replay data. Sensing / human node / MPS may run or not (to keep the GPU load of the recording); if sensing
#   runs, the .h1_choice MPS checks apply as for --human. The recording must have the same scene (scene_text of the npz = this scene
#   file) -> else stop (exit 2). Implies --human (settings from .h1_choice); --pose defaults to the pose of the recording (its
#   <name>_exp.json), the method is --method (default effmass). Name <mode>_<scene>_replay_<method>_<pose>_t<trial>_<stamp>;
#   <name>_replay.json = replay facts; the summary "replay  :" line names the recording and the start joint difference.
#   --human-synth "<demo_synth.py options>" (exp1121, with --fake only; rehearsal): the synthetic person of
#   goal1/real_prep5/demo/demo_synth.py (timeline from the same "warm reset" line) instead of the live human; no sensing needed.
#   GPU check (2026-11-20 night; real and --fake alike): before the plan and again after the start word (before anything on the control
#   PC), nvidia-smi compute processes other than the MPS server (nvidia-cuda-mps-server), sensing (lv_sensing_wrapper.py and its child
#   processes), the human node (effmass_human.human_inertia_node and children) and this run's human relay are shown (pid, name, command)
#   and the run stops with code 12 (e.g. an Isaac app or another MPC left on the GPU). nvidia-smi failing also stops (code 12).
#   --profile <yml> (2026-11-20; no default): the MPC profile instead of real_slowc_acc10.yml / real_c05_acc10.yml (e.g. the no-human
#   reference with goal1/config/real_slowc_i3.yml). Refused with --human (the human run takes its profile from .h1_choice). With --speed
#   (allowed since 2026-11-20 evening) the --speed copy is made from this profile (no-human reference at the same V).
#   Replay archive (2026-11-20): before the MPC, goal1/real_prep4/<name>_scene.json and <name>_profile.yml (copy of the profile the MPC
#   uses; with --speed the MPC uses that copy itself); with --human also <name>_camera_pose.yaml = newest debug_log/camera_pose_*.yaml
#   of the running sensing (IMU-corrected), passed as rp2_mpc_rt.py --rec-camera-pose (paths and texts in <name>_human.npz).
#   Processes (2026-11-20): the MPC, sensing, human node and relay are found by program name (python) + arguments, never by the
#   text of a shell command line; Ctrl+C sends SIGINT only to the MPC python process.
#   Control PC codes (EXIT= in exit_<step>.txt): 0 ok, 7 stop confirmation failed, 128+N or -N ended by signal N, anything else
#   non-zero (or not a number) = failed. Any code but 0 counts as failed; the meaning is shown in the result.
set -u
VS_K=1.01   # --speed V without --fast: joint speed demand / spec per m/s of V (bridge --vel-scale = 1.6 x VS_K x V, 0.30..0.90)
FAST_JERK=1500   # --fast: jerk for the control PC bridge (step 5 --jerk); 1500 = control PC recalculation result (2026-11-20)
# python processes by program name and arguments ($1 = text in the arguments); a shell whose command line merely contains the text
# (e.g. "bash -c '... pgrep -f rp2_mpc_rt.py ...'") is never matched
# (executable argv[0] = .../python3, not the process name: torch renames the main thread to "pt_main_thread")
py_pids() { ps -u "$(id -u)" -o pid=,args= | awk -v pat="$1" '{x = $2; sub(/.*\//, "", x)} x ~ /^python[0-9.]*$/ && index($0, pat) {print $1}'; }
gpu_check() {   # (2026-11-20 night) $1 = when; prints the other GPU compute processes and returns 1 if there are any
  local apps keep line pid name
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null) || { echo "[real_run] GPU check ($1): nvidia-smi failed -> stop"; return 1; }
  keep=$(ps -eo pid=,ppid= | awk -v roots="$(py_pids "${HUMAN_PERCEPTION_PROC:-lv_sensing_wrapper.py}") $(py_pids effmass_human.human_inertia_node) ${HP:-}" \
    'BEGIN{n = split(roots, r, /[ \n]+/); for (i = 1; i <= n; i++) if (r[i] != "") k[r[i]] = 1} {par[$1] = $2}
     END{do {c = 0; for (p in par) if (!(p in k) && (par[p] in k)) {k[p] = 1; c = 1}} while (c); for (p in k) print p}')
  local bad=""
  while IFS=, read -r pid name; do
    pid=${pid// /}; name=${name# }; [[ -z $pid ]] && continue
    [[ $name == *nvidia-cuda-mps-server* ]] && continue
    grep -qx "$pid" <<< "$keep" && continue
    bad+="    pid $pid  $name  $(ps -o args= -p $pid 2>/dev/null | cut -c1-150)"$'\n'
  done <<< "$apps"
  [[ -z $bad ]] && return 0
  echo "[real_run] GPU check ($1): compute processes other than sensing / human node / MPS server are on the GPU -> stop. Stop them and run again:"
  printf '%s' "$bad"; return 1
}
SCENE=${1:-}; SEC=${2:-}; FAKE=0; FAST=0; SMOOTH=""; HUMAN=0; SPEED=""; PROF_OPT=""; METHOD=""; POSE=""; TRIAL=""; HSRC=live; HREP=""; HSYN=""; HOME_BR=0
ARGS=("${@:3}"); i=0
KGAIN=""; ETAMAX=""; ODBASE=""; OKGAIN=""   # (kgain 1123)
MMODE=""   # (massmode 1124) --mass-mode reduced|human
CAMPOSE=""; LEGT=""   # (cam20 1124) --camera-pose <yml>, --leg-timeout <s>
APTOL=""; DBASE=""   # (arrtol 1124) --arrive-pos-tol <m>, --d-base <m>
while (( i < ${#ARGS[@]} )); do
  a=${ARGS[i]}
  case "$a" in --fake) FAKE=1;; --fast) FAST=1;; --smooth) SMOOTH=def;; --smooth=*) SMOOTH=${a#--smooth=};; --human) HUMAN=1;;
    --speed) i=$(( i + 1 )); SPEED=${ARGS[i]:-}; [[ -n $SPEED ]] || { echo "--speed <m/s>"; exit 1; };; --speed=*) SPEED=${a#--speed=};;
    --profile) i=$(( i + 1 )); PROF_OPT=${ARGS[i]:-}; [[ -n $PROF_OPT ]] || { echo "--profile <yml>"; exit 1; };; --profile=*) PROF_OPT=${a#--profile=};;
    --method) i=$(( i + 1 )); METHOD=${ARGS[i]:-};; --method=*) METHOD=${a#--method=};;   # (exp1121)
    --pose) i=$(( i + 1 )); POSE=${ARGS[i]:-};; --pose=*) POSE=${a#--pose=};;
    --trial) i=$(( i + 1 )); TRIAL=${ARGS[i]:-};; --trial=*) TRIAL=${a#--trial=};;
    --human-replay) i=$(( i + 1 )); HREP=${ARGS[i]:-}; HSRC=replay; HUMAN=1;; --human-replay=*) HREP=${a#--human-replay=}; HSRC=replay; HUMAN=1;;
    --home-in-bridge) HOME_BR=1;;   # (item 9) home by the bridge instead of step 0
    --camera-pose) i=$(( i + 1 )); CAMPOSE=${ARGS[i]:-};; --camera-pose=*) CAMPOSE=${a#--camera-pose=};;   # (cam20 1124)
    --leg-timeout) i=$(( i + 1 )); LEGT=${ARGS[i]:-};; --leg-timeout=*) LEGT=${a#--leg-timeout=};;   # (cam20 1124)
    --arrive-pos-tol) i=$(( i + 1 )); APTOL=${ARGS[i]:-};; --arrive-pos-tol=*) APTOL=${a#--arrive-pos-tol=};;   # (arrtol 1124)
    --d-base) i=$(( i + 1 )); DBASE=${ARGS[i]:-};; --d-base=*) DBASE=${a#--d-base=};;   # (arrtol 1124)
    --mass-mode) i=$(( i + 1 )); MMODE=${ARGS[i]:-};; --mass-mode=*) MMODE=${a#--mass-mode=};;   # (massmode 1124)
    --k-gain) i=$(( i + 1 )); KGAIN=${ARGS[i]:-};; --eta-max) i=$(( i + 1 )); ETAMAX=${ARGS[i]:-};; --other-d-base) i=$(( i + 1 )); ODBASE=${ARGS[i]:-};; --other-k-gain) i=$(( i + 1 )); OKGAIN=${ARGS[i]:-};;   # (kgain 1123)
    --human-synth) i=$(( i + 1 )); HSYN=${ARGS[i]:-}; HSRC=synth; HUMAN=1;; --human-synth=*) HSYN=${a#--human-synth=}; HSRC=synth; HUMAN=1;;
    *) echo "options after the seconds: --fake, --fast, --smooth[=<m/s>], --human, --speed <m/s>, --profile <yml>, --method uniform|effmass, --pose H|L|none, --trial N, --human-replay <npz>, --human-synth "<args>" (--fake), --home-in-bridge, --camera-pose <yml>, --leg-timeout <s>, --arrive-pos-tol <m>, --d-base <m>, --k-gain K, --eta-max E, --other-d-base D, --other-k-gain K2, --mass-mode reduced|human"; exit 1;; esac
  i=$(( i + 1 ))
done
if [[ -n $METHOD$POSE$TRIAL ]]; then   # (exp1121) method / pose / trial: human runs only
  (( HUMAN )) || { echo "--method / --pose / --trial need --human"; exit 1; }
fi
if [[ $HSRC == replay ]]; then   # (exp1121) recorded human data instead of the live human
  [[ -n $HREP && -n $HSYN ]] && { echo "--human-replay and --human-synth: give one"; exit 1; }
  HREP=${HREP/#\~/$HOME}; [[ $HREP == /* ]] || HREP=$PWD/$HREP
  [[ -f $HREP ]] || { echo "--human-replay: no file $HREP"; exit 2; }
  if [[ -z $POSE ]]; then   # pose of the recording (its exp.json), else none
    POSE=$(python3 -I -c "import json, sys; print(json.load(open(sys.argv[1])).get('pose', 'none'))" "${HREP%_human.npz}_exp.json" 2>/dev/null || echo none)
  fi
fi
if [[ $HSRC == synth ]]; then
  (( FAKE )) || { echo "--human-synth is for --fake rehearsals only"; exit 1; }
  [[ $HSYN =~ ^[-A-Za-z0-9_.,=\ ]*$ ]] || { echo "--human-synth: only letters, digits, - _ . , = and spaces"; exit 1; }
fi
METHOD=${METHOD:-effmass}; POSE=${POSE:-none}
[[ $METHOD == uniform || $METHOD == effmass ]] || { echo "--method uniform|effmass"; exit 1; }
[[ $POSE == H || $POSE == L || $POSE == G || $POSE == none ]] || { echo "--pose H|L|G|none"; exit 1; }
[[ -z $TRIAL || $TRIAL =~ ^[1-9][0-9]{0,2}$ ]] || { echo "--trial N (1..999)"; exit 1; }
KG_OPTS=""   # (kgain 1123) effective-mass gain / thickness options -> end of the MPC human options (empty = as before)
for kv in "--k-gain:--human-k-gain:$KGAIN" "--eta-max:--human-eta-max:$ETAMAX" "--other-d-base:--human-other-d-base:$ODBASE" "--other-k-gain:--human-other-k-gain:$OKGAIN"; do
  IFS=: read -r ko mo vv <<< "$kv"; [[ -z $vv ]] && continue
  (( HUMAN )) || { echo "$ko needs --human or --human-replay"; exit 1; }
  awk -v v="$vv" 'BEGIN{exit !(v ~ /^[0-9]*\.?[0-9]+$/ && v + 0 <= 1.0)}' || { echo "$ko <number 0..1>"; exit 1; }
  KG_OPTS+=" $mo $vv"
done
[[ -n $KGAIN && $METHOD != effmass ]] && { echo "--k-gain needs --method effmass (uniform = k_gain 0)"; exit 1; }
if [[ -n $MMODE ]]; then   # (massmode 1124) eta reference -> rp2 --human-mass-mode (not given = rp2 default reduced)
  (( HUMAN )) || { echo "--mass-mode needs --human or --human-replay"; exit 1; }
  [[ $MMODE == reduced || $MMODE == human ]] || { echo "--mass-mode reduced|human"; exit 1; }
  KG_OPTS+=" --human-mass-mode $MMODE"
fi
if [[ -n $LEGT ]]; then   # (cam20 1124) leg timeout, s -> rp2 --leg-timeout (all runs)
  awk -v v="$LEGT" 'BEGIN{exit !(v ~ /^[0-9]*\.?[0-9]+$/ && v + 0 >= 1 && v + 0 <= 60)}' || { echo "--leg-timeout <seconds 1..60>"; exit 1; }
fi
if [[ -n $APTOL ]]; then   # (arrtol 1124) arrival position tolerance, m -> rp2 --arrive-pos-tol (all runs)
  awk -v v="$APTOL" 'BEGIN{exit !(v ~ /^[0-9]*\.?[0-9]+$/ && v + 0 >= 0.002 && v + 0 <= 0.2)}' || { echo "--arrive-pos-tol <m 0.002..0.2>"; exit 1; }
fi
if [[ -n $DBASE ]]; then   # (arrtol 1124) arm base clearance d0 -> rp2 --human-d-base after the .h1_choice MPC_EXTRA (last wins)
  (( HUMAN )) || { echo "--d-base needs --human or --human-replay"; exit 1; }
  awk -v v="$DBASE" 'BEGIN{exit !(v ~ /^[0-9]*\.?[0-9]+$/ && v + 0 >= 0.02 && v + 0 <= 0.3)}' || { echo "--d-base <m 0.02..0.3>"; exit 1; }
  KG_OPTS+=" --human-d-base $DBASE"
fi
if [[ -n $CAMPOSE ]]; then   # (cam20 1124) camera pose file: human runs only
  (( HUMAN )) || { echo "--camera-pose needs --human or --human-replay"; exit 1; }
  CAMPOSE=${CAMPOSE/#\~/$HOME}; [[ $CAMPOSE == /* ]] || CAMPOSE=$PWD/$CAMPOSE; [[ -f $CAMPOSE ]] || { echo "--camera-pose: no file $CAMPOSE"; exit 2; }
fi
MPC_PROFILE=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/config/real_slowc_acc10.yml; FAST5="--bridge-smooth --vel-scale 0.55"   # step 5 only (bridge smoothing 2026-11-20; 0.55 = 1.6 x demand 0.318 of real_ym_1120_174138, was 0.30)
(( FAST )) && { MPC_PROFILE=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/config/real_c05_acc10.yml; FAST5="--vel-scale 0.90 --acc 9.8 --jerk $FAST_JERK --bridge-smooth"; }
if [[ -n $PROF_OPT ]]; then   # (2026-11-20) --profile <yml>
  (( HUMAN )) && { echo "--profile with --human: the human run takes its profile from goal1/real_prep5/.h1_choice -> stop"; exit 1; }
  PROF_OPT=${PROF_OPT/#\~/$HOME}; [[ $PROF_OPT == /* ]] || PROF_OPT=$PWD/$PROF_OPT
  [[ -f $PROF_OPT ]] || { echo "--profile: no file $PROF_OPT"; exit 2; }
  python3 -I -c "import sys, yaml; s = (yaml.safe_load(open(sys.argv[1])) or {}).get('settings'); sys.exit(0 if isinstance(s, dict) and 'sphere_speed_limit' in s else 1)" "$PROF_OPT" \
    || { echo "--profile: $PROF_OPT has no settings.sphere_speed_limit (not an MPC profile?) -> stop"; exit 2; }
  MPC_PROFILE=$PROF_OPT
fi
SMOOTH_OPTS=""
if [[ -n $SMOOTH ]]; then   # (real_prep5 overnight filter) --smooth
  LIMV=$SMOOTH; [[ $LIMV == def ]] && LIMV=$( (( FAST )) && echo 0.28 || echo 0.14 )
  awk -v v="$LIMV" 'BEGIN{exit !(v ~ /^0?\.[0-9]+$/ && v + 0 >= 0.05 && v + 0 <= 0.50)}' || { echo "--smooth=<m/s>: 0.05..0.50"; exit 1; }
  SMOOTH_OPTS="--lim-ref cmd --lim-v $LIMV"
fi
CAL=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/real_prep5
HOSTS_ENV=${EFFMASS_HOSTS:-${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/config/hosts.env}; if [[ -f $HOSTS_ENV ]]; then source "$HOSTS_ENV"; fi   # CONTROL_PC, FASTDDS_MAIN_PC_XML, HUMAN_PERCEPTION_* (goal1/config/hosts.env.example)
SCENE_OPTS=""   # (2026-11-20 night) MPC options of the scene (c3: waypoint cycle)
case "$SCENE" in ab) SJ=$CAL/scene_AB_rp5floor.json;; ym) SJ=$CAL/scene_AB_ym_rp5floor_table.json;; ymw) SJ=$CAL/scene_AB_ymw_rp5floor.json;;
  c1) SJ=$CAL/demo/scene_C1_rp5floor.json;; c1k) SJ=$CAL/demo/scene_C1k_rp5floor.json;; c2) SJ=$CAL/demo/scene_C2_rp5floor.json;; c3) SJ=$CAL/demo/scene_C3_rp5floor.json; SCENE_OPTS="--goal-cycle 1";;
  *) echo "usage: real_run.sh <ab|ym|ymw|c1|c1k|c2|c3> <MPC seconds 5..240> [--fake]"; exit 1;; esac
[[ $SEC =~ ^[0-9]+$ ]] && (( SEC >= 5 && SEC <= 240 )) || { echo "MPC seconds: integer 5..240 (the control PC waits 275 s for targets)"; exit 1; }
[[ -f $SJ ]] || { echo "no scene file $SJ"; exit 2; }

# ---- (H2 prep) --human: settings from .h1_choice, MPS, sensing / human node (nothing started yet, nothing on the control PC)
GG=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1; HUMAN_OPTS=""; HUMAN_DESC=""; HSHM=rr5_human; HP=""
[[ $HSRC == replay ]] && HSHM=rr5_replay   # (exp1121) own shared memory for the replay
[[ $HSRC == synth ]] && HSHM=rr5_synth
if (( HUMAN )); then
  CHOICE=${H1_CHOICE:-$GG/config/h1_choice.env}
  if [[ -f $CHOICE ]]; then
    CH=$(python3 -I - "$CHOICE" "$( (( FAST )) && echo 1 || echo 0)" <<'PYEOF'
import os, re, shlex, sys
p, fast = sys.argv[1], sys.argv[2] == "1"; txt = open(p).read()
KEYS = ("MPS", "PROFILE", "FAST_PROFILE", "MPC_EXTRA", "SKIP_LATE", "SENSING_MPS_THREADS", "SENSING_BASE_EXCLUDE",   # (baseex 1124) last one: read by rp5_human_up.sh
        "MASS_MODE", "HUMAN_CHEST_SPINE3", "CAMERA_POSE")   # (cam20 1124) CAMERA_POSE: default of --camera-pose (also read by rp5_human_up.sh)   # (today 1124) MASS_MODE: default of --mass-mode; HUMAN_CHEST_SPINE3: recorded in _exp.json only
def fail(m):
    print("CH_ERR=" + shlex.quote(m)); sys.exit(0)
try:
    words = shlex.split(txt, comments=True)
except ValueError as e:
    fail("cannot read %s: %s" % (p, e))
d, other = {}, []
for w in words:
    k, eq, v = w.partition("=")
    if eq and k.upper() in KEYS:
        d[k.upper()] = v.rstrip(";,)")   # free text like .h1_done: trailing punctuation
    elif eq and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
        other.append(k)
if not d:
    fail("%s has none of %s (KEY=VALUE)" % (p, ", ".join(KEYS)))
mps = d.get("MPS", "0").lower()
if mps not in ("0", "1", "on", "off", "yes", "no", "true", "false"):
    fail("MPS=%s: use 0 or 1" % mps)
mps = "1" if mps in ("1", "on", "yes", "true") else "0"
sl = d.get("SKIP_LATE", "1")
if sl not in ("0", "1"):
    fail("SKIP_LATE=%s: use 0 or 1" % sl)
mm = d.get("MASS_MODE", "")   # (today 1124)
if mm and mm not in ("reduced", "human"):
    fail("MASS_MODE=%s: use reduced or human" % mm)
cs3 = d.get("HUMAN_CHEST_SPINE3", "")
if re.search(r"[^A-Za-z0-9_.+-]", cs3):
    fail("HUMAN_CHEST_SPINE3=%s: letters / digits only" % cs3)
smt = d.get("SENSING_MPS_THREADS", "")
if smt and not (smt.isdigit() and 1 <= int(smt) <= 100):
    fail("SENSING_MPS_THREADS=%s: use 1..100" % smt)
def path(v):
    v = os.path.expandvars(os.path.expanduser(v))
    for c in (v, os.path.join(os.path.expanduser("${EFFMASS_ROOT:-$HOME/effmass_mpc}"), v), os.path.join(os.path.expanduser("${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1"), v)):
        if os.path.isfile(c):
            return os.path.realpath(c)
    fail("profile %s not found" % v)
key = "FAST_PROFILE" if fast else "PROFILE"; prof = path(d[key]) if d.get(key) else ""
cam = d.get("CAMERA_POSE", "")   # (cam20 1124)
if cam:
    c_ = [x for x in (os.path.expanduser(cam), os.path.join(os.path.expanduser("${EFFMASS_ROOT:-$HOME/effmass_mpc}"), cam)) if os.path.isfile(x)]
    if not c_ or " " in os.path.realpath(c_[0]):
        fail("CAMERA_POSE=%s: no such file (or a space in the path)" % cam)
    cam = os.path.realpath(c_[0])
ex = d.get("MPC_EXTRA", "")
if re.search(r"[;&|<>`$\\(){}]", ex):
    fail("MPC_EXTRA has shell characters: %s" % ex)
SET_HERE = ("--link", "--js-source", "--shm-name", "--profile", "--duration", "--scene-json", "--out", "--human", "--human-shm", "--human-frame", "--k",
            "--lim-ref", "--lim-v", "--lim-guard", "--lim-reuse", "--skip-late", "--human-method", "--human-k-gain",
            "--human-eta-max", "--human-other-d-base", "--human-other-k-gain", "--human-mass-mode")   # (kgain 1123) real_run.sh options --k-gain / --eta-max / --other-*
bad = [t for t in ex.split() if t.split("=")[0] in SET_HERE]
if bad:
    fail("MPC_EXTRA sets %s (set by real_run.sh / rp4_run.sh)" % " ".join(bad))
for k, v in (("CH_MPS", mps), ("CH_PROFILE", prof), ("CH_EXTRA", ex), ("CH_SKIPL", sl), ("CH_OTHER", " ".join(other)), ("CH_SMT", smt), ("CH_MASS", mm), ("CH_SPINE3", cs3), ("CH_CAMPOSE", cam),
             ("CH_NOPROF", "1" if fast and not d.get("FAST_PROFILE") and d.get("PROFILE") else "0")):
    print("%s=%s" % (k, shlex.quote(v)))
PYEOF
) || { echo "[real_run] could not read $CHOICE -> stop"; exit 11; }
    CH_ERR=""; CH_MPS=0; CH_PROFILE=""; CH_EXTRA=""; CH_SKIPL=1; CH_OTHER=""; CH_NOPROF=0; CH_SMT=""; CH_MASS=""; CH_SPINE3=""; CH_CAMPOSE=""
    eval "$CH"   # only NAME='quoted value' lines from the reader above (shlex.quote)
    [[ -z $CH_ERR ]] || { echo "[real_run] .h1_choice: $CH_ERR -> stop"; exit 11; }
    [[ -n $CH_PROFILE ]] && MPC_PROFILE=$CH_PROFILE
    HUMAN_DESC="settings $CHOICE: MPS $CH_MPS, profile $(basename $MPC_PROFILE), skip-late $CH_SKIPL, extra [${CH_EXTRA}]"
    [[ -n $CH_OTHER ]] && HUMAN_DESC+=", ignored words: $CH_OTHER"
    if (( CH_NOPROF )); then   # (MPS check) --fast with a PROFILE but no FAST_PROFILE: the fast profile with these settings is untested
      echo "[real_run] --fast --human: $CHOICE has PROFILE but no FAST_PROFILE (the fast profile with these settings is untested) -> stop."
      echo "    Run without --fast, or add FAST_PROFILE=<yml> to $CHOICE after testing it."; exit 11
    fi
  else
    CH_MPS=0; CH_EXTRA=""; CH_SKIPL=1; CH_SMT=""; CH_MASS=""; CH_SPINE3=""; CH_CAMPOSE=""
    HUMAN_DESC="no $CHOICE: base settings ($(basename $MPC_PROFILE)) + --skip-late 1, MPS not required"
  fi
  MMSRC=command   # (today 1124) eta reference: command --mass-mode, else .h1_choice MASS_MODE, else rp2 default (reduced)
  if [[ -z $MMODE && -n $CH_MASS ]]; then MMODE=$CH_MASS; MMSRC=.h1_choice; KG_OPTS+=" --human-mass-mode $MMODE"; fi
  [[ -z $MMODE ]] && MMSRC=default
  HUMAN_DESC+="; mass mode ${MMODE:-reduced} ($MMSRC), chest spine3 ${CH_SPINE3:-not set}"
  MPSD=$(pgrep -u "$(id -u)" -f '^(/usr/bin/)?nvidia-cuda-mps-control' | head -1)   # process names are cut to 15 characters, so match the command line
  SENS=$(py_pids "${HUMAN_PERCEPTION_PROC:-lv_sensing_wrapper.py}" | head -1)
  HNODE=$(py_pids effmass_human.human_inertia_node | head -1)
  if [[ $HSRC == live ]]; then
    [[ -n $SENS ]] || { echo "[real_run] --human: sensing is not running (goal1/real_prep5/rp5_sensing.sh) -> stop"; exit 10; }
    [[ -n $HNODE ]] || { echo "[real_run] --human: the human node is not running (goal1/scripts/g1_run_human.sh node ...) -> stop"; exit 10; }
  else   # (exp1121) replay / synth: sensing and the node may run (same GPU load) or not
    HUMAN_DESC+="; human source $HSRC$( [[ $HSRC == replay ]] && echo " ($HREP)")$( [[ $HSRC == synth ]] && echo " (demo_synth.py $HSYN)"); sensing ${SENS:-off}, human node ${HNODE:-off}, MPS daemon ${MPSD:-off}"
  fi
  CAMSRC=command; CAMSHIFT=""; CAMTREC=""   # (cam20 1124) camera pose: command --camera-pose, else .h1_choice CAMERA_POSE, else none (as before)
  if [[ -z $CAMPOSE && -n ${CH_CAMPOSE:-} ]]; then CAMPOSE=$CH_CAMPOSE; CAMSRC=.h1_choice; fi
  [[ -z $CAMPOSE ]] && CAMSRC=none
  if [[ -n $CAMPOSE ]]; then
    CSOUT=$(tr '\0' '\n' < /proc/${SENS:-0}/environ 2>/dev/null | sed -n 's/^LV_SENSING_OUT=//p' | head -1)
    CAMR=$(python3 -I - "$CAMPOSE" "$HSRC" "${HREP:-}" "${CSOUT:-}" <<'PYEOF'
import glob, os, sys, yaml   # -> "OK <shift dx,dy,dz|-> <recording t|->" or "ERR <text>"
cam, src, rep, sout = sys.argv[1:5]; t = [float(x) for x in yaml.safe_load(open(cam))["t"]]
def tof(txt):
    return [float(x) for x in yaml.safe_load(txt)["t"]]
if src == "replay":
    tr = None; f = rep[:-len("_human.npz")] + "_camera_pose.yaml" if rep.endswith("_human.npz") else ""
    if f and os.path.isfile(f):
        tr = tof(open(f).read())
    else:
        import numpy as np
        z = np.load(rep)
        if "camera_pose_text" in z.files and str(z["camera_pose_text"]).strip():
            tr = tof(str(z["camera_pose_text"]))
    if tr is None:
        print("ERR the recording has no camera pose (<rec>_camera_pose.yaml or camera_pose_text): cannot shift it"); sys.exit(0)
    d = [a - b for a, b in zip(t, tr)]
    print("OK %s %s" % (",".join("%.6g" % x for x in d) if any(abs(x) > 1e-9 for x in d) else "-", ",".join("%.6g" % x for x in tr)))
elif src == "live":
    fs = sorted(glob.glob(os.path.join(sout, "debug_log", "camera_pose_*.yaml"))) if sout else []
    if not fs:
        print("ERR no camera_pose_*.yaml of the running sensing (%s): cannot check that it uses %s" % (sout or "out dir unknown", cam)); sys.exit(0)
    ts = tof(open(fs[-1]).read())
    if max(abs(a - b) for a, b in zip(t, ts)) > 1e-3:
        print("ERR the running sensing uses t %s (%s), the camera pose file %s has t %s: restart the sensing (goal1/real_prep5/rp5_human_up.sh stop, then start)" % (ts, fs[-1], cam, t)); sys.exit(0)
    print("OK - %s" % ",".join("%.6g" % x for x in ts))
else:
    print("OK - -")
PYEOF
    ) || { echo "[real_run] --camera-pose: cannot read $CAMPOSE -> stop"; exit 11; }
    [[ $CAMR == OK* ]] || { echo "[real_run] camera pose ($CAMSRC $CAMPOSE): ${CAMR#ERR } -> stop"; exit 11; }
    read -r _ CAMSHIFT CAMTREC <<< "$CAMR"; [[ $CAMSHIFT == - ]] && CAMSHIFT=""; [[ $CAMTREC == - ]] && CAMTREC=""
    [[ -n $CAMSHIFT ]] && KG_OPTS+=" --human-shift $CAMSHIFT"
  fi
  HUMAN_DESC+="; camera pose ${CAMPOSE:-not set} ($CAMSRC)${CAMSHIFT:+, recorded people moved by $CAMSHIFT m}"
  if [[ $CH_MPS == 1 && ( $HSRC == live || -n $SENS ) ]]; then
    if [[ -z $MPSD ]]; then
      echo "[real_run] --human: the settings use MPS but the MPS daemon is not running -> stop. MPS must be on before sensing starts:"
      echo "    1. stop sensing (Ctrl+C in its terminal), 2. $CAL/rp5_mps.sh on, 3. start sensing again, then the human node if it was stopped, 4. this command again"
      exit 11
    fi
    if (( $(ps -o etimes= -p $SENS || echo 0) > $(ps -o etimes= -p $MPSD || echo 0) )); then
      echo "[real_run] --human: sensing (pid $SENS) started before the MPS daemon (pid $MPSD), so it does not use MPS -> stop."
      echo "    Restart sensing (Ctrl+C in its terminal, then rp5_sensing.sh again) and run this command again."
      exit 11
    fi
    HUMAN_DESC+="; MPS daemon on (pid $MPSD), sensing started after it"
    if [[ -n $CH_SMT ]]; then   # (MPS check) the sensing's SM share must be the one of the settings
      SMT_NOW=$(tr '\0' '\n' < /proc/$SENS/environ 2>/dev/null | sed -n 's/^CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=//p' | head -1)
      if [[ $SMT_NOW != "$CH_SMT" ]]; then
        echo "[real_run] --human: the settings want sensing at CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CH_SMT, sensing (pid $SENS) has ${SMT_NOW:-none} -> stop."
        echo "    Restart sensing (Ctrl+C in its terminal, then RP5_MPS_THREADS=$CH_SMT $CAL/rp5_sensing.sh <out_dir>) and run this command again."
        exit 11
      fi
      HUMAN_DESC+=", sensing SM share $SMT_NOW %"
    fi
  elif [[ -n $MPSD && $CH_MPS != 1 ]]; then
    HUMAN_DESC+="; NOTE: MPS daemon is on although the settings do not use it (the MPC will be an MPS client too)"
  fi
  if [[ $HSRC == replay ]]; then   # (exp1121) the recording must be of this scene
    python3 -I - "$HREP" "$SJ" <<'PYEOF' || exit 2
import json, sys
import numpy as np
z = np.load(sys.argv[1])
need = ("msg_stamp_ns", "step_t_pub_ns", "msg_com", "msg_p0", "scene_text")
miss = [k for k in need if k not in z.files]
if miss:
    print("[real_run] --human-replay: %s has no %s (not a rp2_mpc_rt.py --rec-human record) -> stop" % (sys.argv[1], ", ".join(miss))); sys.exit(1)
try:
    same = json.loads(str(z["scene_text"])) == json.load(open(sys.argv[2]))
except Exception as e:
    print("[real_run] --human-replay: cannot compare the scenes (%s) -> stop" % e); sys.exit(1)
if not same:
    print("[real_run] --human-replay: the recording's scene (%s) is not this scene file %s -> stop" % (z["scene_file"], sys.argv[2])); sys.exit(1)
T = (z["msg_stamp_ns"] - z["step_t_pub_ns"][0]) / 1e9
print("[real_run] --human-replay: %d records, %.1f .. %.1f s of the recording, same scene" % (len(T), T[0], T[-1]))
PYEOF
  fi
  [[ -n $(py_pids "rp5_human_relay.py --shm-name $HSHM") ]] && { echo "[real_run] another rp5_human_relay.py writes $HSHM -> stop"; exit 10; }
  [[ -n $(py_pids "--shm-name $HSHM") ]] && { echo "[real_run] another process writes $HSHM -> stop"; exit 10; }
  HUMAN_OPTS="--human shm --human-shm $HSHM --human-frame $GG/config/live_frame_rp5.yml --skip-late $CH_SKIPL --rec-human 1 $CH_EXTRA --human-method $METHOD$KG_OPTS"
fi
gpu_check "before the plan" || exit 12   # (2026-11-20 night) real and --fake

# ---- (2026-11-20 speed) --speed V: after the profile is final (--fast / .h1_choice)
SPEED_DESC=""
if [[ -n $SPEED ]]; then
  awk -v v="$SPEED" 'BEGIN{exit !(v ~ /^0?\.[0-9]+$/ && v + 0 >= 0.05 && v + 0 <= 0.60)}' || { echo "--speed <m/s>: 0.05..0.60"; exit 1; }
  [[ -z $SMOOTH || $SMOOTH == def ]] || awk -v a="$SMOOTH" -v b="$SPEED" 'BEGIN{exit !(a + 0 == b + 0)}' || { echo "--smooth=$SMOOTH and --speed $SPEED differ: give one"; exit 1; }
  [[ -n $SMOOTH ]] && SMOOTH_OPTS="--lim-ref cmd --lim-v $SPEED"
  if (( ! FAST )); then   # (2026-11-20 evening) bridge vel-scale from V: 1.6 x demand (VS_K x V), rounded up to 0.05, 0.30..0.90
    SPEED_VS=$(awk -v v="$SPEED" -v k="$VS_K" 'BEGIN{x = 1.6 * k * v; x = int(x * 20 + 0.999999) / 20; if (x < 0.30) x = 0.30; if (x > 0.90) x = 0.90; printf "%.2f", x}')
    FAST5="--bridge-smooth --vel-scale $SPEED_VS"
  fi
fi

CTRL=${CONTROL_PC:?set CONTROL_PC=user@address in goal1/config/hosts.env}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 "$CTRL")
RDIR='~/franka_ws/controller_dev/real_step3'
RES=_real_results; ANA="$RDIR/analyze_step3.py"
(( FAKE )) && { RES=_real_results_fake; ANA='~/franka_ws/controller_dev/real_step3_fake/analyze_step3_fake.py'; }
STAMP=$(date +%Y%m%d_%H%M%S); SESSION=${STAMP}_main_${SCENE}; RSES="~/franka_ws/franka_ros2/$RES/$SESSION"
MODE=$( (( FAKE )) && echo fake || echo real ); NAME=${MODE}_${SCENE}_$(date +%m%d_%H%M%S)
OUT=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/real_prep4
if (( HUMAN )); then   # (exp1121) method, pose and trial in the name
  if [[ -z $TRIAL ]]; then
    TRIAL=$(( $(ls "$OUT"/${MODE}_${SCENE}_${METHOD}_${POSE}_t*_summary.json 2>/dev/null | wc -l) + 1 ))
  else
    TRIAL_GIVEN=1
  fi
  NAME=${MODE}_${SCENE}_${METHOD}_${POSE}_t${TRIAL}_$(date +%m%d_%H%M%S)
  if [[ $HSRC != live ]]; then   # (exp1121) replay / synth runs: own names and trial numbers
    KIND=$( [[ $HSRC == replay ]] && echo replay || echo synth )
    [[ -n ${TRIAL_GIVEN:-} ]] || TRIAL=$(( $(ls "$OUT"/${MODE}_${SCENE}_${KIND}_${METHOD}_${POSE}_t*_summary.json 2>/dev/null | wc -l) + 1 ))
    NAME=${MODE}_${SCENE}_${KIND}_${METHOD}_${POSE}_t${TRIAL}_$(date +%m%d_%H%M%S)
  fi
fi
if [[ -n $SPEED ]]; then   # (2026-11-20 speed) profile copy for this run: sphere_speed_limit = V, speed_cap_v = 0.8 V
  PCOPY=$OUT/${NAME}_profile.yml
  SPEED_DESC=$(python3 -I - "$MPC_PROFILE" "$PCOPY" "$SPEED" <<'PYEOF'
import re, sys, yaml
src, dst, v = sys.argv[1], sys.argv[2], float(sys.argv[3]); txt = open(src).read(); old = (yaml.safe_load(txt) or {}).get("settings") or {}
new = {"sphere_speed_limit": v, "speed_cap_v": round(0.8 * v, 4)}
for k, x in new.items():   # the one "  key: value  # comment" line under settings:
    pat = re.compile(r"^(  %s:[ \t]*)([^#\n]*?)([ \t]*(#.*)?)$" % k, re.M); m = pat.findall(txt)
    if len(m) != 1 or k not in old:
        print("ERR %s: %d lines '  %s:' (need one under settings)" % (src, len(m), k)); sys.exit(1)
    txt = pat.sub(lambda g: g.group(1) + repr(x) + g.group(3), txt)
txt = "# %s: copy of %s made by real_run.sh --speed %g (sphere_speed_limit, speed_cap_v changed; original not changed)\n" % (dst.rsplit("/", 1)[-1], src, v) + txt
chk = (yaml.safe_load(txt) or {}).get("settings") or {}
if any(abs(float(chk.get(k, -1)) - x) > 1e-9 for k, x in new.items()) or {k: y for k, y in chk.items() if k not in new} != {k: y for k, y in old.items() if k not in new}:
    print("ERR copy check failed"); sys.exit(1)
open(dst, "w").write(txt)
print("profile copy %s (from %s): sphere_speed_limit %s -> %g, speed_cap_v %s -> %g" % (dst, src.rsplit("/", 1)[-1], old["sphere_speed_limit"], new["sphere_speed_limit"], old["speed_cap_v"], new["speed_cap_v"]))
PYEOF
) || { echo "[real_run] --speed: $SPEED_DESC -> stop"; exit 2; }
  MPC_PROFILE=$PCOPY
  [[ -n $SMOOTH ]] && SPEED_DESC+="; --lim-v $SPEED"
fi
# Korean words written as UTF-8 bytes (the script stays ASCII)
W_START=$'\xec\x8b\x9c\xec\x9e\x91'                                                       # "start"
W_WAIT=$'\xeb\xaa\xa9\xed\x91\x9c\x20\xea\xb8\xb0\xeb\x8b\xa4\xeb\xa6\xbc'                # "waiting for targets"
W_END=$'\xeb\x81\x9d\x2e\x20\xea\xb2\xb0\xea\xb3\xbc\x3a'                                 # "end. result:"
W_JUDGE=$'\xed\x8c\x90\xec\xa0\x95'                                                       # "judgement"
W_LIMIT=$'\xed\x95\x9c\xea\xb3\x84\x20\xec\x9c\x84\xeb\xb0\x98'                            # "limit violations"
W_GAP=$'\xeb\xa1\x9c\xeb\xb4\x87\x20\xec\x8b\x9c\xea\xb0\x84\x20\xea\xb1\xb4\xeb\x84\x88\xeb\x9c\x80'   # "robot time gaps"
W_FERR=$'\x46\x72\x61\x6e\x6b\x61\x20\xec\x98\xa4\xeb\xa5\x98'                            # "Franka errors"
W_LEFT=$'\xeb\x82\xa8\xec\x9d\x80\x20\xed\x94\x84\xeb\xa1\x9c\xec\x84\xb8\xec\x8a\xa4'   # "remaining processes" (stop done, bag closed)
W_HOMEFAIL=$'home \xec\x9d\xb4\xeb\x8f\x99 \xec\x8b\xa4\xed\x8c\xa8'   # "home move failed" (item 9, control PC run.log)
W_HOMEOK=$'home \xeb\x8f\x84\xec\xb0\xa9'                                     # "home arrived"
W_GW=$'\xea\xb2\x8c\xec\x9d\xb4\xed\x8a\xb8\xec\x9b\xa8\xec\x9d\xb4\x20\xec\xa2\x85\xeb\xa3\x8c'        # "gateway ended"

INTR=0; MPC_STARTED=0
on_int() {
  if (( INTR )); then echo; echo "[real_run] second Ctrl+C: leaving now. The control PC run continues and ends by itself ($RSES)."; exit 130; fi
  INTR=1; echo
  for p in $(py_pids rp2_mpc_rt.py); do kill -INT "$p" 2>/dev/null; done   # (2026-11-20) the MPC python process only
  if (( MPC_STARTED )); then echo "[real_run] Ctrl+C: MPC stop requested (no more targets). The control PC ends by its idle stop; waiting for its result (Ctrl+C again to leave)."
  else echo "[real_run] Ctrl+C before the MPC started. The control PC run (if started) is not killed; it ends by its own --duration. Leaving."; exit 130; fi
}
trap on_int INT

TMUX_S=franka
pane_cmd() { "${SSH[@]}" "tmux display-message -p -t $TMUX_S '#{pane_current_command}'" 2>/dev/null; }
pane_idle() { [[ $(pane_cmd) =~ ^(bash|zsh|sh|fish)$ ]]; }
pane_tail() { "${SSH[@]}" "tmux capture-pane -p -t $TMUX_S -S -60" 2>/dev/null | grep -v '^\s*$' | tail -${1:-10}; }
remote_tmux() {   # run_real_step3.sh <args> typed into the tmux session (as by hand); exit code -> $RSES/exit_<step>.txt
  for i in $(seq 1 15); do pane_idle && break; sleep 1; done
  pane_idle || { echo "[real_run] tmux session $TMUX_S is busy (running: $(pane_cmd)) -> stop"; return 1; }
  "${SSH[@]}" "tmux send-keys -t $TMUX_S C-u && tmux send-keys -t $TMUX_S -l 'cd $RDIR && ./run_real_step3.sh $*' && tmux send-keys -t $TMUX_S Enter"
}
hz_rate() { awk '/^average rate:/{r=$3} END{printf "%.1f", r+0}' "$1"; }   # last "average rate: X" line of ros2 topic hz, 0 if none
remote_exit() { "${SSH[@]}" "grep -hoE '^EXIT=[^ ]+' $RSES/exit_$1.txt 2>/dev/null | head -1 | cut -d= -f2"; }   # $1 = step; any code (H2 prep: not only digits)

"${SSH[@]}" "tmux has-session -t $TMUX_S" 2>/dev/null || { echo "[real_run] no tmux session '$TMUX_S' on the control PC. On the control PC run: tmux new -d -s $TMUX_S"; exit 5; }
pane_idle || { echo "[real_run] tmux session $TMUX_S is not at a shell prompt (running: $(pane_cmd)) -> stop"; exit 5; }

REFLEX_MSG="robot reflex stop (collision detected) - clear the error in Desk"   # (night 2026-11-21) control PC code 8 / exit file text
reflex_of() {   # $1 = step, $2 = its exit code: prints REFLEX_MSG if the control PC code is 8 (exp1121: the code only, no text search)
  [[ ${2:-} == 8 ]] && echo "$REFLEX_MSG"
  return 0
}
ctrl_record() {   # (exp1121 2nd) $1 step, $2 control PC EXIT= value ("" = none), $3 this script's exit code: <name>_ctrl_exit.json, + exp.json
  python3 -I - "$OUT/${NAME}_ctrl_exit.json" "$OUT/${NAME}_exp.json" "$NAME" "$1" "${2:-}" "$3" <<'PYEOF' || true
import json, os, sys
f, ej, name, step, ce, rr = sys.argv[1:7]
d = dict(name=name, step=int(step), control_exit=ce or None, real_run_exit=int(rr), reflex_stop=ce == "8",
         note="control_exit = the EXIT= value of the control PC exit file (None = no exit file / no end line); real_run_exit = this script's code")
json.dump(d, open(f, "w"), indent=1)
if os.path.exists(ej):
    e = json.load(open(ej)); e.update(control_exit=ce or None, reflex_stop=ce == "8", real_run_exit=int(rr), control_step=int(step))
    try:   # (baseex 1124) MPC --human-base-exclude result (summary human.base_exclude), only when the MPC had the option
        be = (json.load(open(ej[:-len("_exp.json")] + "_summary.json")).get("human") or {}).get("base_exclude")
    except Exception:
        be = None
    if be is not None:
        e["human_base_exclude"] = be
    try:   # (robex 1124) MPC --human-robot-exclude result (summary human.robot_exclude), only when the MPC had the option
        rx = (json.load(open(ej[:-len("_exp.json")] + "_summary.json")).get("human") or {}).get("robot_exclude")
    except Exception:
        rx = None
    if rx is not None:
        e["human_robot_exclude"] = rx
    json.dump(e, open(ej, "w"), indent=1)
PYEOF
}
exit_text() {   # meaning of a control PC exit code
  case "${1:-}" in
    0) echo "ok";;
    8) echo "FAILED: $REFLEX_MSG";;
    7) echo "FAILED: stop confirmation failed (the control PC could not confirm that the robot stopped)";;
    "") echo "FAILED: no exit code (not ended / no exit file)";;
    -[0-9]*) echo "FAILED: ended by signal ${1#-} ($(kill -l ${1#-} 2>/dev/null))";;
    *) if [[ $1 =~ ^[0-9]+$ ]] && (( $1 > 128 && $1 < 160 )); then echo "FAILED: ended by signal $(( $1 - 128 )) ($(kill -l $(( $1 - 128 )) 2>/dev/null), $1 = 128 + $(( $1 - 128 )))"
       else echo "FAILED: non-zero exit code"; fi;;
  esac
}

# ---- (H2 prep) --human: human relay (domain 30 -> shared memory) and a first look, before the plan
if (( HUMAN )); then
  HSTOP=$OUT/${NAME}.humanstop; rm -f "$HSTOP"
  stop_hrelay() { [[ -n $HP ]] || return 0; touch "$HSTOP"; for i in $(seq 1 30); do kill -0 $HP 2>/dev/null || break; sleep 0.1; done
                  kill -0 $HP 2>/dev/null && kill -TERM $HP 2>/dev/null; wait $HP 2>/dev/null; rm -f "$HSTOP"; HP=""; }
  trap stop_hrelay EXIT
  case $HSRC in
    live) "$GG/scripts/g1_run_human.sh" py "$CAL/rp5_human_relay.py" --shm-name $HSHM --duration $(( SEC + 1800 )) --stop-file "$HSTOP" > "$OUT/${NAME}_hrelay.log" 2>&1 &;;
    replay) /usr/bin/python3 -I "$CAL/h2cost/rp5_replay_human.py" --replay "$HREP" --shm-name $HSHM --start-log "$OUT/${NAME}_mpc.log" --pre-rate 30 \
              --duration $(( SEC + 1800 )) --stop-file "$HSTOP" --info-json "$OUT/${NAME}_replay.json" > "$OUT/${NAME}_hrelay.log" 2>&1 &;;
    synth) /usr/bin/python3 -I "$CAL/demo/demo_synth.py" --shm-name $HSHM --duration $(( SEC + 1800 )) --stop-file "$HSTOP" --start-log "$OUT/${NAME}_mpc.log" \
              --log "$OUT/${NAME}_synth.npz" $HSYN > "$OUT/${NAME}_hrelay.log" 2>&1 &;;
  esac
  HP=$!
  /usr/bin/python3 -I "$CAL/rp5_human_check.py" --shm $HSHM --seconds 5 --min-rate 15 --wait 10 > "$OUT/${NAME}_humancheck.txt" 2>&1 &
  HCP=$!   # (2026-11-20 waits) judged after the start word
fi

echo "=== real_run: scene $SCENE ($(basename $SJ)), MPC $SEC s, $( (( FAKE )) && echo 'FAKE hardware (control PC --fake, step 0 skipped)' || echo 'REAL ROBOT (step 0 home first)' ) ==="
echo "    control PC $CTRL, session $RES/$SESSION; main PC results $OUT/${NAME}_*"
echo "    MPC: $(basename $MPC_PROFILE), K 1, --stale-ms 50, --warm-reset 1, --acc-clip 1 (rp4_run.sh c6)$( [[ -n $SMOOTH_OPTS ]] && echo ", smoothing $SMOOTH_OPTS")$( [[ -n $SCENE_OPTS ]] && echo ", scene options $SCENE_OPTS"); control PC step 5: $FAST5"
[[ -n $SPEED_DESC ]] && echo "    SPEED: $SPEED_DESC"
(( HOME_BR )) && echo "    HOME : --home-in-bridge: no step 0; step 5 moves to home in the bridge first (wait for targets up to 100 s; control PC code 9 / 8 -> stop before the MPC)"
(( HUMAN )) && { echo "    HUMAN: $HUMAN_DESC"; echo "    HUMAN: MPC options $HUMAN_OPTS (human data: 5 s look now, 4 s again just before the MPC)"
                 echo "    EXP  : method $METHOD, pose $POSE, trial $TRIAL (result name $NAME)"; }
read -r -p "Type the Korean word for start ($W_START) to go on: " ANS
[[ $ANS == "$W_START" ]] || { echo "not confirmed -> end"; exit 4; }
gpu_check "after the start word" || exit 12   # (2026-11-20 night) again: the plan may have waited long
if (( HUMAN )); then   # (2026-11-20 waits) the first human look ran while the plan waited
  wait $HCP; HCR=$?; cat "$OUT/${NAME}_humancheck.txt"
  (( HCR == 0 )) || { echo "[real_run] --human: human data not coming at 15 per s (relay log $OUT/${NAME}_hrelay.log) -> stop, nothing started on the control PC"; exit 10; }
fi

# ---- 2. step 0 (home), real robot only (item 9: not with --home-in-bridge)
if (( ! FAKE && ! HOME_BR )); then
  echo "[real_run] control PC step 0 (home) ..."
  remote_tmux --execute 0 --confirmed --session $SESSION || { echo "[real_run] could not start step 0 -> stop"; exit 5; }
  for (( T=SECONDS+120; SECONDS<T; )); do E=$(remote_exit 0); [[ -n $E ]] && break; sleep 0.3; done
  pane_tail 3
  [[ ${E:-} == 0 ]] || { RF0=$(reflex_of 0 "${E:-}"); ctrl_record 0 "${E:-}" 5; echo "[real_run] step 0 failed or did not end in 120 s (exit ${E:-none}: $(exit_text ${E:-}))$( [[ -n $RF0 && ${E:-} != 8 ]] && echo "; $RF0") -> stop"; exit 5; }
fi

# ---- 3. step 5 (live), in the tmux session
echo "[real_run] control PC step 5 (live) ..."
FK=$( (( FAKE )) && echo --fake )
(( HOME_BR )) && FK+="${FK:+ }--home-in-bridge"   # (item 9)
remote_tmux --execute 5 --confirmed --end-after-idle 2 --duration 300 --session $SESSION $FK $FAST5 || { echo "[real_run] could not start step 5 -> stop"; exit 5; }
READY=0
if (( HOME_BR )); then   # (item 9) up to 100 s; stop on EXIT=9 / 8 or a "home move failed" line, no target sent
  HB_FAIL=""
  for (( T=SECONDS+100; SECONDS<T; )); do
    "${SSH[@]}" "grep -qsF '$W_WAIT' $RSES/5_live_*/run.log" && { READY=1; break; }
    E=$(remote_exit 5); [[ -n $E ]] && { HB_FAIL="control PC exit $E"; break; }
    "${SSH[@]}" "grep -qsF '$W_HOMEFAIL' $RSES/5_live_*/run.log" && { HB_FAIL="run.log: $("${SSH[@]}" "grep -hF '$W_HOMEFAIL' $RSES/5_live_*/run.log | head -1")"; break; }
    sleep 0.3
  done
  "${SSH[@]}" "grep -hF '$W_HOMEOK' $RSES/5_live_*/run.log | tail -1"
  if (( ! READY )) && [[ -n $HB_FAIL ]]; then
    echo "[real_run] --home-in-bridge: $HB_FAIL -> no target sent, MPC not started; waiting for the control PC exit code (its stop order) ..."
    for (( T=SECONDS+60; SECONDS<T; )); do E=$(remote_exit 5); [[ -n $E ]] && break; sleep 0.5; done
    pane_tail 6
    if (( HUMAN )); then
      python3 -I -c "import json, sys; json.dump(dict(name=sys.argv[1], mode=sys.argv[2], scene=sys.argv[3], method=sys.argv[4], pose=sys.argv[5], human_source=sys.argv[6], home_in_bridge=True, control_exit=sys.argv[7] or None, aborted='home' if sys.argv[7] == '9' else 'home-reflex' if sys.argv[7] == '8' else 'home', reflex_stop=sys.argv[7] == '8'), open(sys.argv[8], 'w'), indent=1)" \
        "$NAME" "$MODE" "$SCENE" "$METHOD" "$POSE" "$HSRC" "${E:-}" "$OUT/${NAME}_exp.json"
    fi
    ctrl_record 5 "${E:-}" "$( [[ ${E:-} == 9 || ${E:-} == 8 ]] && echo ${E:-} || echo 6 )"
    case ${E:-} in
      9) echo "[real_run] control PC exit 9: home move failed (--home-in-bridge) -> end, code 9"; exit 9;;
      8) echo "[real_run] control PC exit 8: $REFLEX_MSG (during the home move) -> end, code 8"; exit 8;;
      *) echo "[real_run] control PC exit ${E:-none} ($(exit_text ${E:-})) -> end, code 6"; exit 6;;
    esac
  fi
else
for (( T=SECONDS+60; SECONDS<T; )); do
  "${SSH[@]}" "grep -qsF '$W_WAIT' $RSES/5_live_*/run.log" && { READY=1; break; }
  [[ -n $(remote_exit 5) ]] && break
  sleep 0.3
done
fi
if (( ! READY )); then
  echo "[real_run] control PC not waiting for targets within $( (( HOME_BR )) && echo 100 || echo 60 ) s (exit $(E=$(remote_exit 5); echo "${E:-none}: $(exit_text $E)")) -> MPC not started. Its tmux output:"
  pane_tail 10; exit 6
fi
"${SSH[@]}" "grep -hF '$W_WAIT' $RSES/5_live_*/run.log"

# ---- 4. joint states, 4 s (2026-11-20; was 10 s), the human re-check at the same time: as the hand check (this shell's environment + the exports below, daemon stopped before and
#      after). Raw output kept in <name>_hz.txt; PYTHONUNBUFFERED so that nothing printed is lost in the file when timeout ends hz.
HZRAW=$OUT/${NAME}_hz.txt
if (( HUMAN )); then   # (2026-11-20 waits) human data, 4 s, during the joint-state check
  /usr/bin/python3 -I "$CAL/rp5_human_check.py" --shm $HSHM --seconds 4 --min-rate 15 --wait 3 > "$OUT/${NAME}_humancheck2.txt" 2>&1 &
  HCP=$!
fi
( set +u; source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp PYTHONUNBUFFERED=1 \
         FASTRTPS_DEFAULT_PROFILES_FILE=${FASTDDS_MAIN_PC_XML:-${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/config/fastdds_main_pc.xml}
  ros2 daemon stop
  timeout -k 2 4 ros2 topic hz /c6/joint_states
  ros2 daemon stop ) > "$HZRAW" 2>&1
HZ=$(hz_rate "$HZRAW")
echo "[real_run] /c6/joint_states $HZ per s (raw: $HZRAW)"
if ! awk -v r="$HZ" 'BEGIN{exit !(r >= 900 && r <= 1100)}'; then
  echo "[real_run] joint-state rate not near 1000 per s -> MPC not started. The control PC ends by its --duration (no targets)."; exit 7
fi
if (( HUMAN )); then   # (H2 prep) human data, just before the MPC (started with the joint-state check)
  wait $HCP; HCR=$?; cat "$OUT/${NAME}_humancheck2.txt" | tee -a "$OUT/${NAME}_humancheck.txt"; rm -f "$OUT/${NAME}_humancheck2.txt"
  if (( HCR != 0 )); then
    echo "[real_run] --human: fewer than 15 human messages per s -> MPC not started. The control PC gets no targets and ends by itself."; exit 10
  fi
fi

# ---- (2026-11-20) replay archive: scene, profile, camera pose of the sensing run
cp "$SJ" "$OUT/${NAME}_scene.json"
[[ $MPC_PROFILE == "$OUT/${NAME}_profile.yml" ]] || cp "$MPC_PROFILE" "$OUT/${NAME}_profile.yml"
ARCH="$OUT/${NAME}_scene.json, $OUT/${NAME}_profile.yml"
if (( HUMAN )); then
  SOUT=$(tr '\0' '\n' < /proc/${SENS:-0}/environ 2>/dev/null | sed -n 's/^LV_SENSING_OUT=//p' | head -1)
  CPF=$(ls -t "${SOUT:-/nonexistent}"/debug_log/camera_pose_*.yaml 2>/dev/null | head -1)
  if [[ -n $CPF ]]; then
    cp "$CPF" "$OUT/${NAME}_camera_pose.yaml"; HUMAN_OPTS+=" --rec-camera-pose $OUT/${NAME}_camera_pose.yaml"; ARCH+=", $OUT/${NAME}_camera_pose.yaml (from $CPF)"
  else
    ARCH+="; NOTE: no camera_pose_*.yaml found for the running sensing (${SOUT:-out dir unknown})"
  fi
fi
if (( HUMAN )); then   # (exp1121) experiment record
  python3 -I - "$OUT/${NAME}_exp.json" <<PYEOF
import json, sys
def yaml_t(p):   # (cam20 1124)
    if not p:
        return None
    import yaml
    return [float(x) for x in yaml.safe_load(open(p))["t"]]
T = "$HUMAN_OPTS".split(); opt = lambda k: float(T[len(T) - 1 - T[::-1].index(k) + 1]) if k in T else None   # (kgain 1123) last value, as argparse
d0 = opt("--human-d-base"); d0 = 0.065 if d0 is None else d0; em = 0.25 if opt("--human-d-base") is None else max(0.25, d0)   # rp2_mpc_rt.py rules
em = em if opt("--human-eta-max") is None else max(opt("--human-eta-max"), d0); kg = 0.0 if "$METHOD" == "uniform" else (opt("--human-k-gain") if opt("--human-k-gain") is not None else 0.10)
KG = dict(human_k_gain=kg, human_eta_max=em, human_other_d_base=d0 if opt("--human-other-d-base") is None else opt("--human-other-d-base"),
          human_other_k_gain=kg if opt("--human-other-k-gain") is None else opt("--human-other-k-gain"),
          human_arm_parts=["left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"])
SH = int(opt("--human-shoulder") or 0); sopt = lambda k: T[len(T) - 1 - T[::-1].index(k) + 1] if k in T else None   # (shoulder 1124) rp2 --human-shoulder 1 (e.g. .h1_choice MPC_EXTRA)
KG.update(human_mass_mode=sopt("--human-mass-mode") or "reduced", human_mass_mode_source="$MMSRC")   # (massmode 1124) eta reference used (rp2 default reduced)
cs3 = "$CH_SPINE3"   # (today 1124) .h1_choice HUMAN_CHEST_SPINE3 (other session's switch), recorded as given; human_shoulder always
KG.update(human_shoulder=SH, human_chest_spine3=(int(cs3) if cs3.isdigit() else cs3) if cs3 else None)
_hx = sopt("--human-robot-exclude"); KG.update(human_robot_exclude=[x for x in _hx.split(",") if x] or None if _hx else None)   # (robex 1124) links (summary detail after the run)
KG.update(human_rec_plan=int(opt("--rec-plan") or 0))
_ap = "$APTOL"; KG.update(arrive_pos_tol=float(_ap) if _ap else (opt("--arrive-pos-tol") or 0.01), arrive_pos_tol_source="command" if _ap else (".h1_choice MPC_EXTRA" if opt("--arrive-pos-tol") else "default 0.01"))   # (arrtol 1124)
KG.update(human_d_base=d0, human_d_base_source="command --d-base" if "$DBASE" else (".h1_choice MPC_EXTRA" if opt("--human-d-base") is not None else "rp2 default"))   # (arrtol 1124) value used (last)
_lt = "$LEGT"; KG.update(leg_timeout_s=round(float(_lt) if _lt else (opt("--leg-timeout") or 400 / 60.0), 4),
                         leg_timeout_source="command" if _lt else (".h1_choice MPC_EXTRA" if opt("--leg-timeout") else "default 400 steps"))   # (cam20 1124)
KG.update(camera_pose=dict(file="$CAMPOSE" or None, t=yaml_t("$CAMPOSE"), source="$CAMSRC", recording_t=[float(x) for x in "$CAMTREC".split(",")] if "$CAMTREC" else None,
                           human_shift=[float(x) for x in "$CAMSHIFT".split(",")] if "$CAMSHIFT" else None))   # (cam20 1124)   # (recplan 1124) rp2 --rec-plan 1 (MPC plan 0.5 s ahead in the steps npz), e.g. .h1_choice MPC_EXTRA
if SH:   # the parts, ISO reference masses and sphere file rp2_mpc_rt.py uses with the option (same defaults)
    import os
    shp = os.path.realpath(os.path.expanduser(sopt("--human-shoulder-json") or "${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/data/shapes_v2_shoulder.json"))
    iso = os.path.realpath(os.path.expanduser(sopt("--human-iso-override-json") or "${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/data/iso_ref_override.json"))
    KG.update(human_arm_parts=KG["human_arm_parts"] + ["left_shoulder", "right_shoulder"], human_shoulder=1, human_shoulder_shapes=shp,
              human_iso_override=json.load(open(iso))["m_h_iso"], human_iso_override_file=iso)
json.dump(dict(name="$NAME", mode="$MODE", scene="$SCENE", method="$METHOD", pose="$POSE", trial=int("$TRIAL"), seconds=int("$SEC"),
               speed="$SPEED" or None, smooth="$SMOOTH" or None, fast=bool($FAST), mpc_human_options="$HUMAN_OPTS", smooth_options="$SMOOTH_OPTS",
               scene_options="$SCENE_OPTS", profile="$MPC_PROFILE", control_step5="$FAST5", human_source="$HSRC", replay_of="$HREP" or None,
               synth_args="$HSYN" or None, sensing_pid="${SENS:-}" or None, mps_daemon_pid="${MPSD:-}" or None, home_in_bridge=bool($HOME_BR), **KG), open(sys.argv[1], "w"), indent=1)
PYEOF
  ARCH+=", $OUT/${NAME}_exp.json"
fi
echo "[real_run] replay archive: $ARCH"

# ---- 5. MPC
echo "[real_run] MPC $SEC s: $NAME"
MPC_STARTED=1
(( HUMAN )) && export EFFMASS_WS=1 MPC_EXTRA="${MPC_EXTRA:-} $HUMAN_OPTS"   # (H2 prep) --human only
[[ -n $SMOOTH_OPTS ]] && export MPC_EXTRA="${MPC_EXTRA:-} $SMOOTH_OPTS"   # (real_prep5 overnight filter) --smooth only
[[ -n $SCENE_OPTS ]] && export MPC_EXTRA="${MPC_EXTRA:-} $SCENE_OPTS"   # (2026-11-20 night) c3 only
[[ -n $APTOL ]] && export MPC_EXTRA="${MPC_EXTRA:-} --arrive-pos-tol $APTOL"   # (arrtol 1124) last = wins over .h1_choice MPC_EXTRA
[[ -n $LEGT ]] && export MPC_EXTRA="${MPC_EXTRA:-} --leg-timeout $LEGT"   # (cam20 1124) last = wins over .h1_choice MPC_EXTRA
SCENE_JSON=$SJ MPC_PROFILE=$MPC_PROFILE "${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/scripts/rp4_run.sh" c6 "$NAME" "$SEC"
(( HUMAN )) && { stop_hrelay; tail -1 "$OUT/${NAME}_hrelay.log"; }

# ---- 6. main PC summary first (2026-11-20 waits), then the control PC result and analysis
echo
echo "================= result: $NAME ($MODE, scene $SCENE) ================="
(( HUMAN )) && echo "exp     : method $METHOD, pose $POSE, trial $TRIAL, human $HSRC ($OUT/${NAME}_exp.json)"
if [[ $HSRC == replay ]]; then   # (exp1121) recording name, start joints against the recording, timeline start
  python3 -I - "$HREP" "$OUT/${NAME}_human.npz" "$OUT/${NAME}_replay.json" <<'PYEOF'
import json, os, sys
import numpy as np
src, new, info = sys.argv[1:4]
try:
    a = np.load(src)["meas_q"][0].astype(float); b = np.load(new)["meas_q"][0].astype(float); dq = "%.4f rad" % np.abs(a - b).max()
except Exception as e:
    dq = "not known (%s)" % e
try:
    I = json.load(open(info)); st = "timeline started" if I.get("timeline_start_wall") else "timeline NOT started"
except Exception:
    st = "replay info missing"
print("replay  : recording %s; start joints vs the recording: max diff %s; %s" % (os.path.basename(src).replace("_human.npz", ""), dq, st))
PYEOF
fi
[[ -n $SPEED ]] && echo "speed   : V $SPEED m/s, lim-v $( [[ -n $SMOOTH ]] && echo "$SPEED" || echo "none (no --smooth)"), profile copy $PCOPY, bridge vel-scale $(sed -n 's/.*--vel-scale \([0-9.]*\).*/\1/p' <<< "$FAST5")"
echo "bridge  : bridge-smooth on, vel-scale $(sed -n 's/.*--vel-scale \([0-9.]*\).*/\1/p' <<< "$FAST5") (control PC step 5: $FAST5)"
python3 - "$OUT/${NAME}_summary.json" <<'PYEOF'
import json, sys
try:
    S = json.load(open(sys.argv[1]))
except Exception as e:
    print("main PC: no summary (" + str(e) + ")"); sys.exit(0)
if "msgs_per_s" not in S:
    print("main PC: sent %s steps (short run), stop %s" % (S.get("steps"), S.get("stop"))); sys.exit(0)
n = S["steps"]; st = S.get("stop")
print("main PC : sent %d (%.2f per s), late %d (%.2f %%), compute over %d" % (n, S["msgs_per_s"], S["deadline_miss"], 100.0 * S["deadline_miss"] / max(n, 1), S["overrun_steps"]))
print("          state age p50 %.2f / p99 %.2f / max %.2f ms, arrivals %d, infeasible %d, diff-acc > 10: %d"
      % (S["state_age_p50_ms"], S["state_age_ms"]["p99"], S["state_age_ms"]["max"], S["arrivals"], S["infeasible"], S["diffacc_over10_steps"]))
print("          stop: " + ("%s at step %d (age %.1f ms)" % (st["reason"], st["step"], st["age_ms"]) if st else "none (ran the full time)"))
h = S.get("human")
if h:   # (H2 prep) --human
    dur = S.get("duration_s") or n / 60.0; nr = h.get("near") or {}; ag = h.get("age_ms") or {}
    f = lambda x, fm="%.0f": "-" if x is None else fm % x
    print("human   : %s messages used in %.1f s (%s per s), new in %s steps, age at use mean %s / p99 %s / max %s ms, loss events %d"
          % (h.get("pickups"), dur, f(h["pickups"] / dur if h.get("pickups") is not None and dur else None, "%.1f"), h.get("new_steps"),
             f(ag.get("mean")), f(ag.get("p99")), f(ag.get("max")), len(h.get("loss_events") or [])))
    print("          min gap robot sphere - human shape: %s m at t %s s, robot %s - human %s; steps < 0.10 m %s, < 0.05 m %s"
          % (f(nr.get("min_m"), "%.3f"), f(nr.get("t_s"), "%.1f"), nr.get("sphere_link"), nr.get("segment"), nr.get("steps_below_0p10"), nr.get("steps_below_0p05")))
    rc = h.get("rec")
    if rc:   # (2026-11-20 rec-human)
        print("          human record: %s (%.1f MB), %s records used, relay records not seen %s"
              % (rc.get("file"), (rc.get("bytes") or 0) / 1e6, rc.get("messages"), rc.get("relay_records_not_seen")))
PYEOF
[[ -f $OUT/${NAME}_humancheck.txt ]] && sed 's/^/          /' "$OUT/${NAME}_humancheck.txt"
echo "[real_run] waiting for the control PC to end (idle 2 s -> stop order) ..."
# The bag analysis starts as soon as run.log shows the stop is done (bag closed: "remaining processes" line), in parallel with the
# control PC's own robot-mode check; if that line never comes, it starts after the end line (as before). Same analysis, same judgement.
RPATH=""; ANAPID=""; ANAF=$OUT/${NAME}_ctrl_ana.txt
start_ana() {   # $1 = control PC result folder (5_live_*)
  ( "${SSH[@]}" "B=\$(ls -d $1/bag_* | head -1); timeout 180 python3 -I $ANA \$B --launch-log $1/launch.log 2>&1 \
    | grep -E '^($W_LIMIT|$W_GAP|$W_FERR|$W_JUDGE)'" > "$ANAF" 2>/dev/null ) &
  ANAPID=$!
}
for (( T=SECONDS+360; SECONDS<T; )); do
  ST=$("${SSH[@]}" "for f in $RSES/5_live_*/run.log; do grep -qsF '$W_LEFT' \$f && echo L \$(dirname \$f); grep -hsF '$W_END' \$f | tail -1 | sed 's/.*: /E /'; done" 2>/dev/null)
  if [[ -z $ANAPID ]] && grep -q '^L ' <<< "$ST"; then
    start_ana "$(grep -m1 '^L ' <<< "$ST" | cut -c3-)"; echo "[real_run] control PC stop done (bag closed): its bag analysis started"
  fi
  RPATH=$(grep -m1 '^E ' <<< "$ST" | cut -c3-)
  [[ -n $RPATH ]] && break
  sleep 0.3
done
for (( T=SECONDS+15; SECONDS<T; )); do E5=$(remote_exit 5); [[ -n $E5 ]] && break; sleep 0.3; done
if [[ -z $RPATH ]]; then
  echo "control : no end line within 360 s (exit ${E5:-none}: $(exit_text ${E5:-})); see the tmux session $TMUX_S on the control PC"; ctrl_record 5 "${E5:-}" 8; exit 8
fi
echo "control : exit ${E5:-none} ($(exit_text ${E5:-})), result $RPATH"
RF5=$( [[ -n ${E5:-} && ${E5:-} != 0 ]] && reflex_of 5 "${E5:-}")   # (night 2026-11-21) reflex stop: shown in the summary, judgement unchanged (failed)
[[ -n $RF5 ]] && echo "reflex  : $RF5"
ctrl_record 5 "${E5:-}" "$( [[ ${E5:-} == 0 ]] && echo 0 || echo 9 )"   # (exp1121 2nd) every run: the control PC EXIT= value (also into exp.json)
GWLINE=$("${SSH[@]}" "grep -hF '$W_GW' $RPATH/run.log | tail -1")
[[ -n $ANAPID ]] || start_ana "$RPATH"
wait $ANAPID; ANAOUT=$(cat "$ANAF" 2>/dev/null)
# fake hardware: robot time gaps not applicable (debug topic recorded every 10 cycles); judgement = violations 0, Franka errors 0,
# targets sent by the MPC = received = forwarded by the gateway. Real robot: the analysis lines as they are.
python3 - "$FAKE" "$OUT/${NAME}_summary.json" "$GWLINE" "$ANAOUT" "${E5:-}" <<'PYEOF'
import ast, json, re, sys
fake, summ, gw, ana = sys.argv[1] == "1", sys.argv[2], sys.argv[3], sys.argv[4]
e5 = sys.argv[5] if len(sys.argv) > 5 else ""   # (H2 prep) control PC exit code: only 0 passes
print(gw)
GAP, JUDGE = "\ub85c\ubd07 \uc2dc\uac04 \uac74\ub108\ub700", "\ud310\uc815"
lines = [l for l in ana.splitlines() if l.strip()]
if not fake:
    print("\n".join(lines)); sys.exit(0)
d = {}
for l in lines:
    if l.startswith(GAP):
        print(GAP + ": \ud574\ub2f9 \uc5c6\uc74c (\uac00\uc9dc \ud558\ub4dc\uc6e8\uc5b4, debug 10\uc8fc\uae30 \uae30\ub85d)")
    elif l.startswith(JUDGE):
        m = re.search(r"\{.*\}", l); d = ast.literal_eval(m.group(0)) if m else {}
    else:
        print(l)
try:
    sent = int(json.load(open(summ))["steps"])
except Exception:
    sent = None
m1 = re.search("\ubc1b\uc740 \ubaa9\ud45c ([0-9.]+)", gw); m2 = re.search("\ub118\uae40 ([0-9.]+)", gw)
recv = int(float(m1.group(1))) if m1 else None; fwd = int(float(m2.group(1))) if m2 else None
v0, f0 = d.get("violations_zero") is True, d.get("franka_errors_zero") is True
match = sent is not None and sent == recv == fwd
ok = v0 and f0 and match and e5 == "0"
print("%s (\uac00\uc9dc \ud558\ub4dc\uc6e8\uc5b4): %s  \uc704\ubc18 0 %s, Franka \uc624\ub958 0 %s, \uc804\ub2ec \uc218 \uc77c\uce58 %s (MPC %s = \ubc1b\uc74c %s = \ub118\uae40 %s), control exit %s"
      % (JUDGE, "\ud1b5\uacfc" if ok else "\ubd88\ud1b5\uacfc", v0, f0, match, sent, recv, fwd, e5 or "none"))
PYEOF
if [[ ${E5:-} != 0 ]]; then   # (H2 prep) any non-zero control PC code = failed run
  echo "control : exit ${E5:-none} -> this run FAILED ($(exit_text ${E5:-}))"
  [[ -n ${RF5:-} && ${E5:-} != 8 ]] && echo "control : $RF5"
  echo "=================================================================="; exit 9
fi
echo "=================================================================="
