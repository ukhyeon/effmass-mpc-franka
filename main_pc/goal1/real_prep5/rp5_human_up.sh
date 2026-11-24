#!/usr/bin/env bash
# rp5_human_up.sh start|stop|status - real_prep5 H2 (2026-11-20): MPS, sensing and the human node with one command, in the tmux
# session "human" (window "sensing", window "node"). Watch them with: tmux attach -t human (leave with Ctrl+B then D).
#   Settings from goal1/real_prep5/.h1_choice (read as text, never executed): MPS=0|1, SENSING_MPS_THREADS=<1..100>; no file = MPS 0.
#   start : 1. refuses if the session "human" exists, or sensing / the human node already run (started by hand or elsewhere).
#           2. refuses if any CUDA process runs (as rp5_mps.sh on: MPS only applies to processes started after it).
#           3. MPS=1: rp5_mps.sh on, then checks the daemon. MPS=0 but a daemon runs: refuses (sensing would become an MPS client).
#           4. window "sensing": [RP5_MPS_THREADS=<SENSING_MPS_THREADS>] rp5_sensing.sh goal1/real_prep5/sensing_<MMDD_HHMM>; waits until
#              its perf log (<out>/sensing_perf/*.csv) has 30 frames (at most HU_SENSING_WAIT s, default 120), else stops and says why.
#              HU_SAVEVIS=1 (2026-11-20; default off): sensing also saves every camera colour frame (LV_SENSING_SAVEVIS=1:
#              <out>/vis_output/<index>.jpg with the capture time, <out>/vis_times.csv; about 30 jpg per s, mind the disk).
#              HU_BODY_LOG=1 (exp1121, 2026-11-21; default off): sensing is started with night/rp5_sensing_bodies.sh (same sensing,
#              same arguments) and RP5_BODY_LOG=1: every Body Tracking body list -> <out>/bodies.bin (record only, the body choice
#              is not changed: RP5_BODY_SELECT stays 0). Reader: night/rp5_sensing_bodies.py read_bodies(path).
#              Both for experiment recordings (exp1121):  HU_BODY_LOG=1 HU_SAVEVIS=1 goal1/real_prep5/rp5_human_up.sh start
#           5. window "node": g1_run_human.sh node -p shapes:=v2 -p pelvis_correction:=true (+ -p chest_spine3:=true with .h1_choice
#              HUMAN_CHEST_SPINE3=1, spine3 2026-11-24: chest orientation includes the SMPL spine3 rotation; default off).
#           6. human data check as real_run.sh: rp5_human_relay.py (domain 30) -> shared memory hu5_human (not real_run's rr5_human),
#              rp5_human_check.py 4 s, at least 15 messages per s (waits up to 30 s for the first message); one line.
#           7. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE of the sensing process (from /proc/<pid>/environ, as real_run.sh) against the settings.
#           On a failure the session is left as it is (to read the windows); "stop" ends it.
#   stop  : refuses while an MPC (rp2_mpc_rt.py) or real_run.sh runs (HU_FORCE=1 to go on). Ctrl+C to the node window, waits until
#           its shell prompt is back (20 s), then the same for sensing (40 s); a second Ctrl+C if needed; closes the session only if
#           both ended, then rp5_mps.sh off and rp5_mps.sh status. Processes not started in the session are not touched.
#   status: windows and what runs in them, sensing / node processes, MPS daemon and GPU processes, sensing MPS share, human data rate.
#   Exit: 0 ok, 1 usage, 2 already running / not running, 3 GPU busy, 4 MPS, 5 sensing, 6 human node / human data, 7 stop incomplete.
set -u
P=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/real_prep5; GG=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1; TS=human; CHOICE=${H1_CHOICE:-$GG/config/h1_choice.env}
HOSTS_ENV=${EFFMASS_HOSTS:-${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/config/hosts.env}; if [[ -f $HOSTS_ENV ]]; then source "$HOSTS_ENV"; fi   # CONTROL_PC, FASTDDS_MAIN_PC_XML, HUMAN_PERCEPTION_* (goal1/config/hosts.env.example)
SHM=hu5_human
tm() { tmux "$@" 2>/dev/null; }
have_session() { tm has-session -t "$TS"; }
pane_cmd() { tm display-message -p -t "$TS:$1" '#{pane_current_command}'; }
pane_busy() { [[ -n $(pane_cmd "$1") && ! $(pane_cmd "$1") =~ ^(bash|zsh|sh|fish)$ ]]; }
pane_tail() { tm capture-pane -p -t "$TS:$1" -S -40 | grep -v '^\s*$' | tail -${2:-8} | sed 's/^/    | /'; }
# Processes are found by program name + arguments, not by "pgrep -f" alone: a shell whose command line merely contains the text
# (e.g. another session's "bash -c '... until ! pgrep -f rp2_mpc_rt.py ...'") must not count (2026-11-20 fix: stop was refused
# after the run because of such a waiting loop).
# A python process is recognised by its executable (argv[0] = .../python3), not by the process name: torch renames the main thread
# to "pt_main_thread" (sensing, human node), so the name is not "python3" (2026-11-20 fix of the fix).
py_pid() { ps -u "$(id -u)" -o pid=,args= | awk -v pat="$1" '{x = $2; sub(/.*\//, "", x)} x ~ /^python[0-9.]*$/ && index($0, pat) {print $1; exit}'; }   # $1 = text in a python command line
sens_pid() { py_pid "${HUMAN_PERCEPTION_PROC:-lv_sensing_wrapper.py}"; }   # perception process (hosts.env)
node_pid() { py_pid "effmass_human.human_inertia_node"; }
mpc_running() {   # prints the MPC / real_run.sh processes (pid, name, arguments), nothing if none
  ps -eo pid=,comm=,args= | awk '
    {x = $3; sub(/.*\//, "", x)}
    x ~ /^python[0-9.]*$/ && index($0, "rp2_mpc_rt.py") {print; next}  # the MPC (Isaac python; executable .../python3) itself
    $2 == "real_run.sh" || $2 == "rp4_run.sh" {print; next}            # the scripts run as ./real_run.sh, ./rp4_run.sh
    $2 ~ /^(bash|sh)$/ && ($4 ~ /(^|\/)(real_run|rp4_run)[.]sh$/) {print}  # run as "bash real_run.sh" (argument 1, not inside -c text)
  ' | cut -c1-160
}
mps_pid() { pgrep -u "$(id -u)" -f '^(/usr/bin/)?nvidia-cuda-mps-control' | head -1; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null; }
sens_share() { local p; p=$(sens_pid); [[ -n $p ]] && tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=//p' | head -1; }
read_choice() {   # CH_MPS, CH_SMT, CH_BEX, CH_SP3, CH_CAM from .h1_choice (words KEY=VALUE, shell quoting allowed, nothing executed)
  CH_MPS=0; CH_SMT=""; CH_BEX=""; CH_SP3=0; CH_CAM=""
  [[ -f $CHOICE ]] || return 0
  local out
  out=$(python3 -I - "$CHOICE" <<'PYEOF'
import shlex, sys
d = {}
for w in shlex.split(open(sys.argv[1]).read(), comments=True):
    k, eq, v = w.partition("=")
    if eq:
        d[k.upper()] = v.rstrip(";,)")
m = d.get("MPS", "0").lower(); m = "1" if m in ("1", "on", "yes", "true") else "0" if m in ("0", "off", "no", "false") else "bad"
t = d.get("SENSING_MPS_THREADS", "")
t = t if (t == "" or (t.isdigit() and 1 <= int(t) <= 100)) else "bad"
b = d.get("SENSING_BASE_EXCLUDE", "")   # (baseex 1124) yaml of the robot base / cart box for the sensing body choice
if b:
    import os
    c = [x for x in (os.path.expanduser(b), os.path.join(os.path.expanduser("${EFFMASS_ROOT:-$HOME/effmass_mpc}"), b)) if os.path.isfile(x)]
    b = os.path.realpath(c[0]) if c and " " not in os.path.realpath(c[0]) else "bad"
c = d.get("HUMAN_CHEST_SPINE3", "0").lower()   # (spine3 1124) human node chest orientation with spine3 (node parameter chest_spine3)
c = "1" if c in ("1", "on", "yes", "true") else "0" if c in ("0", "off", "no", "false") else "bad"
p = d.get("CAMERA_POSE", "")   # (cam20 1124) camera pose file for the sensing (RP5_POSE); none = rp5_sensing.sh default camera_pose_rp5.yaml
if p:
    import os
    c_ = [x for x in (os.path.expanduser(p), os.path.join(os.path.expanduser("${EFFMASS_ROOT:-$HOME/effmass_mpc}"), p)) if os.path.isfile(x)]
    p = os.path.realpath(c_[0]) if c_ and " " not in os.path.realpath(c_[0]) else "bad"
print(m, t or "-", b or "-", c, p or "-")
PYEOF
) || { echo "[human-up] cannot read $CHOICE"; exit 4; }
  read -r CH_MPS CH_SMT CH_BEX CH_SP3 CH_CAM <<< "$out"; [[ $CH_SMT == - ]] && CH_SMT=""; [[ $CH_BEX == - ]] && CH_BEX=""; [[ $CH_CAM == - ]] && CH_CAM=""
  [[ $CH_CAM != bad ]] || { echo "[human-up] $CHOICE: CAMERA_POSE file not found (path as given or under ${EFFMASS_ROOT:-$HOME/effmass_mpc}, no spaces) -> stop"; exit 4; }
  [[ $CH_SP3 != bad ]] || { echo "[human-up] $CHOICE: HUMAN_CHEST_SPINE3 must be 0 or 1 -> stop"; exit 4; }
  [[ $CH_MPS != bad && $CH_SMT != bad ]] || { echo "[human-up] $CHOICE: MPS must be 0/1 and SENSING_MPS_THREADS 1..100 -> stop"; exit 4; }
  [[ $CH_BEX != bad ]] || { echo "[human-up] $CHOICE: SENSING_BASE_EXCLUDE file not found (path as given or under ${EFFMASS_ROOT:-$HOME/effmass_mpc}, no spaces) -> stop"; exit 4; }
}
human_rate() {   # $1 = seconds, $2 = wait for the first message (s); prints one line, returns the check's exit code
  local st=$P/.hu5_relay_stop_$$ rp rc
  rm -f "$st"
  "$GG/scripts/g1_run_human.sh" py "$P/rp5_human_relay.py" --shm-name $SHM --duration $(( $1 + $2 + 30 )) --stop-file "$st" > /dev/null 2>&1 &
  rp=$!
  /usr/bin/python3 -I "$P/rp5_human_check.py" --shm $SHM --seconds "$1" --min-rate 15 --wait "$2"; rc=$?
  touch "$st"; for i in $(seq 1 30); do kill -0 $rp 2>/dev/null || break; sleep 0.1; done
  kill -0 $rp 2>/dev/null && kill -TERM $rp 2>/dev/null; wait $rp 2>/dev/null; rm -f "$st"
  return $rc
}
show_share() {
  local now; now=$(sens_share)
  if [[ -n $CH_SMT ]]; then
    [[ $now == "$CH_SMT" ]] && echo "[human-up] sensing CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$now (settings $CH_SMT): ok" \
      || { echo "[human-up] sensing CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${now:-none}, settings want $CH_SMT: MISMATCH"; return 1; }
  else
    echo "[human-up] sensing CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${now:-none} (settings: not set)"
  fi
}
watch_line() { echo "[human-up] watch: tmux attach -t $TS   (windows: sensing, node; leave with Ctrl+B then D)"; }

do_status() {
  read_choice
  echo "[human-up] settings: MPS $CH_MPS, SENSING_MPS_THREADS ${CH_SMT:-not set}${CH_BEX:+, SENSING_BASE_EXCLUDE $CH_BEX}$( [[ ${CH_SP3:-0} == 1 ]] && echo ", HUMAN_CHEST_SPINE3 1") ($( [[ -f $CHOICE ]] && echo "$CHOICE" || echo "no $CHOICE"))"
  if have_session; then
    tm list-windows -t "$TS" -F '[human-up] window #{window_name}: #{pane_current_command} (pid #{pane_pid})'
  else
    echo "[human-up] no tmux session $TS"
  fi
  echo "[human-up] sensing pid $(sens_pid || true), human node pid $(node_pid || true)"
  local m; m=$(mps_pid)
  echo "[human-up] MPS daemon: ${m:-off}; GPU processes: $(gpu_apps | tr '\n' ';' | sed 's/;$//')"
  [[ -n $(sens_pid) ]] && show_share
  if [[ -n $(node_pid) ]]; then human_rate 3 5; else echo "[human-up] human data: no human node"; fi
  have_session && watch_line
  return 0
}

do_start() {
  read_choice
  if have_session || [[ -n $(sens_pid) || -n $(node_pid) ]]; then
    echo "[human-up] already running (session $TS: $(have_session && echo yes || echo no), sensing pid $(sens_pid), human node pid $(node_pid)) -> not started again"
    have_session && watch_line; exit 2
  fi
  local a; a=$(gpu_apps)
  [[ -z $a ]] || { echo "[human-up] CUDA processes are running, stop them first (MPS / sensing must start before any GPU user):"; echo "$a"; exit 3; }
  if [[ $CH_MPS == 1 ]]; then
    "$P/rp5_mps.sh" on || { echo "[human-up] rp5_mps.sh on failed -> stop"; exit 4; }
    [[ -n $(mps_pid) ]] || { echo "[human-up] MPS daemon not running after rp5_mps.sh on -> stop"; exit 4; }
  elif [[ -n $(mps_pid) ]]; then
    echo "[human-up] the settings say MPS 0 but an MPS daemon runs (pid $(mps_pid)); sensing would become an MPS client -> stop."
    echo "    Run $P/rp5_mps.sh off first, or set MPS=1 in $CHOICE."; exit 4
  fi
  # (exp1121) recording runs: body lists + colour frames together:  HU_BODY_LOG=1 HU_SAVEVIS=1 rp5_human_up.sh start
  [[ ${HU_BODY_LOG:-0} == 1 && ${HU_SAVEVIS:-0} == 1 ]] || echo "[human-up] note: for experiment recordings start with HU_BODY_LOG=1 HU_SAVEVIS=1 $0 start (body lists + colour frames; sensing behaviour unchanged)"
  local out=$P/sensing_$(date +%m%d_%H%M%S) env=""
  [[ $CH_MPS == 1 && -n $CH_SMT ]] && env="RP5_MPS_THREADS=$CH_SMT "
  [[ ${HU_SAVEVIS:-0} == 1 ]] && env+="LV_SENSING_SAVEVIS=1 "   # (2026-11-20) colour frames saved to <out>/vis_output
  local sens_sh=${HUMAN_PERCEPTION_START:-$P/rp5_sensing.sh}   # perception start command <out_dir> (hosts.env; hrc_ws, not in this repository)
  if [[ ${HU_BODY_LOG:-0} == 1 ]]; then   # (exp1121) body list record only (RP5_BODY_SELECT 0 = sensing takes body 0 as before)
    sens_sh=${HUMAN_PERCEPTION_START_BODIES:-$P/night/rp5_sensing_bodies.sh}; env+="RP5_BODY_LOG=1 RP5_BODY_SELECT=0 "
    [[ -x $sens_sh ]] || { echo "[human-up] HU_BODY_LOG=1: $sens_sh not found / not executable -> stop"; exit 5; }
  fi
  if [[ -n ${CH_BEX:-} ]]; then   # (baseex 1124) .h1_choice SENSING_BASE_EXCLUDE=<yaml>: bodies with the pelvis on the robot cart removed before body 0 is taken
    sens_sh=${HUMAN_PERCEPTION_START_BODIES:-$P/night/rp5_sensing_bodies.sh}; env+="RP5_BASE_EXCLUDE=$CH_BEX "
    [[ -x $sens_sh ]] || { echo "[human-up] SENSING_BASE_EXCLUDE: $sens_sh not found / not executable -> stop"; exit 5; }
  fi
  [[ -n ${CH_CAM:-} ]] && env+="RP5_POSE=$CH_CAM "   # (cam20 1124) .h1_choice CAMERA_POSE=<yml>: camera pose of the sensing (rp5_sensing.sh / rp5_sensing_bodies.sh RP5_POSE)
  tmux new-session -d -s "$TS" -n sensing -c "$P" || { echo "[human-up] cannot create tmux session $TS -> stop"; exit 5; }
  tmux new-window -d -t "$TS" -n node -c "$GG/scripts"
  tmux send-keys -t "$TS:sensing" "${env}$sens_sh $out" Enter
  echo "[human-up] sensing started in $TS:sensing -> $out (MPS $CH_MPS${CH_SMT:+, share $CH_SMT %}$( [[ ${HU_SAVEVIS:-0} == 1 ]] && echo ", saving colour frames to $out/vis_output")$( [[ ${HU_BODY_LOG:-0} == 1 ]] && echo ", body lists to $out/bodies.bin")${CH_BEX:+, base exclude $CH_BEX}${CH_CAM:+, camera pose $CH_CAM})"
  local wmax=${HU_SENSING_WAIT:-120} t0=$SECONDS n=0
  while (( SECONDS - t0 < wmax )); do
    n=$(cat "$out"/sensing_perf/*.csv 2>/dev/null | wc -l)
    (( n >= 31 )) && break
    if (( SECONDS - t0 > 5 )) && ! pane_busy sensing; then
      echo "[human-up] sensing ended before giving frames -> stop. Its window:"; pane_tail sensing; watch_line; exit 5
    fi
    sleep 1
  done
  (( n >= 31 )) || { echo "[human-up] sensing gave $(( n > 0 ? n - 1 : 0 )) frames in $wmax s (need 30) -> stop. Its window:"; pane_tail sensing; watch_line; exit 5; }
  echo "[human-up] sensing gives frames ($(( n - 1 )) after $(( SECONDS - t0 )) s)"
  local sp3=""; [[ ${CH_SP3:-0} == 1 ]] && sp3=" -p chest_spine3:=true"   # (spine3 1124) .h1_choice HUMAN_CHEST_SPINE3=1
  tmux send-keys -t "$TS:node" "$GG/scripts/g1_run_human.sh node -p shapes:=v2 -p pelvis_correction:=true$sp3" Enter
  echo "[human-up] human node started in $TS:node"
  local hrc; human_rate 4 30; hrc=$?
  local src=0; show_share || src=1
  watch_line
  (( hrc == 0 )) || { echo "[human-up] human data below 15 per s (stand in front of the camera?). Node window:"; pane_tail node; exit 6; }
  (( src == 0 )) || exit 4
  echo "[human-up] ready"
}

do_stop() {
  have_session || { echo "[human-up] no tmux session $TS (processes started by hand are not touched: sensing pid $(sens_pid), node pid $(node_pid))"; exit 2; }
  local busy; busy=$(mpc_running)
  if [[ ${HU_FORCE:-0} != 1 && -n $busy ]]; then
    echo "[human-up] an MPC / real_run.sh is running; stopping the human input now would cut it -> not stopped (HU_FORCE=1 to stop anyway):"
    echo "$busy" | sed 's/^/    /'; exit 7
  fi
  local w ok=1
  for w in node:20 sensing:40; do
    local win=${w%%:*} lim=${w##*:}
    if pane_busy "$win"; then
      tmux send-keys -t "$TS:$win" C-c
      for (( T=SECONDS+lim; SECONDS<T; )); do pane_busy "$win" || break; sleep 0.5; done
      if pane_busy "$win"; then
        tmux send-keys -t "$TS:$win" C-c
        for (( T=SECONDS+10; SECONDS<T; )); do pane_busy "$win" || break; sleep 0.5; done
      fi
      pane_busy "$win" && { echo "[human-up] $win did not end (running: $(pane_cmd "$win")):"; pane_tail "$win"; ok=0; } || echo "[human-up] $win ended"
    else
      echo "[human-up] $win: nothing running"
    fi
    (( ok )) || break
  done
  (( ok )) || { echo "[human-up] session $TS kept open (tmux attach -t $TS); MPS left as it is"; exit 7; }
  tmux kill-session -t "$TS" && echo "[human-up] tmux session $TS closed"
  "$P/rp5_mps.sh" off; local mrc=$?
  "$P/rp5_mps.sh" status
  (( mrc == 0 )) || exit 7
}

case "${1:-}" in
  start) do_start;;
  stop) do_stop;;
  status) do_status;;
  *) echo "usage: $0 start|stop|status   (watch: tmux attach -t $TS; recordings: HU_BODY_LOG=1 HU_SAVEVIS=1 $0 start)"; exit 1;;
esac
