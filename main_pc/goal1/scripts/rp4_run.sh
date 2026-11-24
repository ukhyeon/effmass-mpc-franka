#!/usr/bin/env bash
# rp4_run.sh <net> <name> [duration_s] - real_prep4 (2026-11-19): Isaac-free MPC (rp2_mpc_rt.py) on the two-PC link to the control-PC gateway.
#   net c6   : ROS_DOMAIN_ID 42, ROS_LOCALHOST_ONLY 0, RMW_IMPLEMENTATION rmw_fastrtps_cpp,
#              FASTRTPS_DEFAULT_PROFILES_FILE ${FASTDDS_MAIN_PC_XML:-${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/config/fastdds_main_pc.xml} (holds the PC addresses; not copied into the repo).
#              Starts the relay and the MPC only; the control-PC gateway publishes /c6/joint_states and reads /mpc/joint_trajectory.
#   net local: in-PC check: ROS_DOMAIN_ID ${RP4_DOMAIN:-77}, ROS_LOCALHOST_ONLY 1, rmw_fastrtps_cpp, no profile file. Also starts
#              rp2_fake_robot.py like the gateway (/c6/joint_states best effort, 1 kHz, joint names 1,3,2,4..7; /mpc/joint_trajectory best effort).
#              Env FAKE_STOP_AFTER (s; the fake robot stops publishing joint states then; default 0 = never), TAU_MS (default 10).
#   Both processes get the same ROS environment: the relay (rp2_js_relay.py, system python, /opt/ros/humble) and the MPC (Isaac python via
#   env_check/scripts/run_isaac_ros.sh with RIR_* variables). MPC: --link c6 --js-source shm --stale-ms 50, profile real_slowc_acc10.yml, K 1,
#   --acc-clip 1, --warm-reset 1; env MPC_EXTRA adds options. Env MPC_PROFILE (2026-11-20): profile yml instead of real_slowc_acc10.yml. Results goal1/real_prep4/<name>_* (OUT_DIR to change).
#   Env SCENE_JSON: scene (goals, start configuration); default goal1/real_prep5/demo/scene_C1k_rp5floor.json (the c1k scene of real_run.sh) (real_prep4 2026-11-20:
#   scene_AB_ym.json = the same goals turned -90 deg about the base z axis, rp4_scene_ym.py).
#   Refuses to start while an Isaac / MPC process is running.
set -u
NET=${1:?net c6|local}; NAME=${2:?name}; DUR=${3:-60}
GG=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1; SC=$GG/scripts; OUT=${OUT_DIR:-$GG/real_prep4}; mkdir -p $OUT
PROFILE_XML=${FASTDDS_MAIN_PC_XML:-${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/config/fastdds_main_pc.xml}
case "$NET" in
  c6)    DOM=42; LOC=0; XML=$PROFILE_XML; [[ -f $XML ]] || { echo "no profile $XML"; exit 2; } ;;
  local) DOM=${RP4_DOMAIN:-77}; LOC=1; XML="" ;;
  *) echo "net is c6 or local"; exit 2 ;;
esac
n=$(ps -eo comm=,args= | awk '$1!="bash" && $1!="grep" && $1!="sh"' | grep -cE 'step2b_mpc_isaac|kit/python|rp2_mpc_rt')
[[ $n == 0 ]] || { echo "an Isaac / MPC process is running; stop it first"; exit 3; }
ROSENV=(env -i HOME="$HOME" USER="${USER:-$(id -un)}" LANG=C.UTF-8 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PYTHONUNBUFFERED=1
        ROS_LOG_DIR="${EFFMASS_ROOT:-$HOME/effmass_mpc}/env_check/ros_log" ROS_DOMAIN_ID=$DOM ROS_LOCALHOST_ONLY=$LOC RMW_IMPLEMENTATION=rmw_fastrtps_cpp)
[[ -n $XML ]] && ROSENV+=(FASTRTPS_DEFAULT_PROFILES_FILE="$XML")
syspy() { "${ROSENV[@]}" bash -c 'cd "$HOME"; set +u; source /opt/ros/humble/setup.bash; set -u; exec /usr/bin/python3 "$@"' _ "$@"; }
SJ=${SCENE_JSON:-$GG/real_prep5/demo/scene_C1k_rp5floor.json}; [[ -f $SJ ]] || { echo "no scene $SJ"; exit 2; }
echo "net $NET: domain $DOM, localhost_only $LOC, profile ${XML:-none}, scene $SJ" | tee $OUT/${NAME}_net.txt
FP=""
if [[ $NET == local ]]; then
  Q0=$(python3 -c "import json;print(','.join(str(x) for x in json.load(open('$SJ'))['q_start']))")
  syspy $SC/rp2_fake_robot.py --js-topic /c6/joint_states --jt-topic /mpc/joint_trajectory --js-qos best_effort --jt-qos best_effort --name-order gateway \
    --tau-ms ${TAU_MS:-10} --q0=$Q0 --stop-pub-after ${FAKE_STOP_AFTER:-0} --duration $(( DUR + 240 )) --out $OUT/$NAME > $OUT/${NAME}_fake.log 2>&1 &
  FP=$!; sleep 2
fi
syspy $SC/rp2_js_relay.py --js-topic /c6/joint_states --js-qos best_effort --shm-name rp4_js --duration $(( DUR + 240 )) --stop-file $OUT/${NAME}.relaystop > $OUT/${NAME}_relay.log 2>&1 &
RP=$!; sleep 2
RIR_DOMAIN_ID=$DOM RIR_LOCALHOST_ONLY=$LOC RIR_RMW=rmw_fastrtps_cpp RIR_FASTRTPS_PROFILE="$XML" ROS_METHOD=min ${EFFMASS_ROOT:-$HOME/effmass_mpc}/env_check/scripts/run_isaac_ros.sh \
  $SC/rp2_mpc_rt.py --link c6 --js-source shm --shm-name rp4_js --stale-ms 50 --profile ${MPC_PROFILE:-$GG/config/real_slowc_acc10.yml} --k 1 --acc-clip 1 --warm-reset 1 \
  --duration $DUR --scene-json $SJ --out $OUT/$NAME ${MPC_EXTRA:-} > $OUT/${NAME}_mpc.log 2>&1
echo "mpc exit $?"
touch $OUT/${NAME}.relaystop; wait $RP; rm -f $OUT/${NAME}.relaystop; tail -1 $OUT/${NAME}_relay.log
[[ -n $FP ]] && { wait $FP; tail -1 $OUT/${NAME}_fake.log; }
grep -a "^\[rp2\]\|Traceback\|Error" $OUT/${NAME}_mpc.log | tail -3 | cut -c1-600
