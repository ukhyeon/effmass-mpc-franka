#!/usr/bin/env bash
# run_isaac_ros.sh — run_isaac.sh 와 같은 깨끗한 환경(env -i)에 ROS 2 Humble 사용 경로만 더해 Isaac Python 을 실행한다.
#
#   사용법: ROS_METHOD=min|source run_isaac_ros.sh <script.py> [args...]
#           ROS_METHOD=... run_isaac_ros.sh --print-env
#
#   ROS_METHOD=min    (필요 경로만, 기본값)
#     - LD_LIBRARY_PATH : /opt/ros/humble/lib, /opt/ros/humble/lib/x86_64-linux-gnu, ~/hrc_ws/install/hrc_interfaces/lib
#     - AMENT_PREFIX_PATH : ~/hrc_ws/install/hrc_interfaces:/opt/ros/humble  (rmw 구현 라이브러리 위치 탐색용)
#     - Python 경로는 PYTHONPATH 로 넣지 않는다 (python.sh 가 PYTHONPATH 를 Isaac 경로보다 앞에 두기 때문).
#       대신 STEP3_ROS_PYPATHS 로 넘겨, 스크립트가 sys.path 맨 뒤에 붙인다 (step3_ros_common.add_ros_paths()).
#   ROS_METHOD=source (비교용: ~/.bashrc 의 ROS 부분과 같게 setup.bash 2개를 source)
#     - source /opt/ros/humble/setup.bash; source ~/hrc_ws/install/setup.bash
#   공통: ROS_DISTRO=humble, ROS_DOMAIN_ID=30, ROS_LOCALHOST_ONLY=1 (이 PC 의 ~/.bashrc 값과 같게), RMW_IMPLEMENTATION 은 지정하지 않음(기본 fastrtps)
#   이 스크립트는 ~/.bashrc, /opt/ros, ~/hrc_ws 를 읽기만 하고 바꾸지 않는다.
set -euo pipefail
ISAAC_ROOT="${ISAAC_ROOT:-${EFFMASS_ROOT:-$HOME/effmass_mpc}/isaacsim}"
PY="$ISAAC_ROOT/python.sh"
METHOD="${ROS_METHOD:-min}"
ROS=/opt/ros/humble
HRC_WS=${HRC_WS:-$HOME/hrc_ws}   # perception workspace (hrc_ws, not in this repository): its message package is on the ROS paths
HRC=$HRC_WS/install/hrc_interfaces

CLEAN_ENV=(
  env -i
  HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}"
  SHELL=/bin/bash TERM="${TERM:-xterm-256color}" LANG="${LANG:-C.UTF-8}"
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
  ISAAC_ROOT="$ISAAC_ROOT" EFFMASS_ROOT="${EFFMASS_ROOT:-$HOME/effmass_mpc}" HRC_WS="$HRC_WS"
  ROS_DISTRO=humble ROS_VERSION=2 ROS_PYTHON_VERSION=3
  ROS_DOMAIN_ID="${RIR_DOMAIN_ID:-30}" ROS_LOCALHOST_ONLY="${RIR_LOCALHOST_ONLY:-1}"
  STEP3_ROS_METHOD="$METHOD"
  ROS_LOG_DIR="${EFFMASS_ROOT:-$HOME/effmass_mpc}/env_check/ros_log"   # rclpy 로그를 ~/.ros/log 대신 여기로 (3단계 사용자 결정)
  STEP_ROBOT="${STEP_ROBOT:-panda}"   # (goal1) 로봇 선택 panda|fr3 (step2_scene.robot_name)
  STEP_FR3_ARM_GAINS="${STEP_FR3_ARM_GAINS:-625,10}"   # (goal1) FR3 팔 구동 이득 (step2_scene.set_fr3_arm_gains)
)
# (goal1 1-2d 2안, 2026-11-16) Isaac 안 numpy(OpenBLAS64 pthreads) LAPACK 호출 뒤 스레드 바쁜 대기 방지: BLAS·OpenMP 1스레드. 이 실행 환경에만 적용
#   EFFMASS_THREADS1=0 이면 넣지 않음 (비교 측정용). warp 커널 캐시는 ${EFFMASS_ROOT:-$HOME/effmass_mpc}/cache/warp (기존 ~/.cache/warp 는 그대로 둠)
# (real_prep4, 2026-11-19) opt-in ROS network for the two-PC link; unset = as before (domain 30, localhost only, default rmw, no profile)
if [[ -n "${RIR_RMW:-}" ]]; then CLEAN_ENV+=(RMW_IMPLEMENTATION="$RIR_RMW"); fi
if [[ -n "${RIR_FASTRTPS_PROFILE:-}" ]]; then CLEAN_ENV+=(FASTRTPS_DEFAULT_PROFILES_FILE="$RIR_FASTRTPS_PROFILE"); fi
if [[ "${EFFMASS_THREADS1:-1}" == "1" ]]; then CLEAN_ENV+=(OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1); fi
CLEAN_ENV+=(WARP_CACHE_PATH="${EFFMASS_ROOT:-$HOME/effmass_mpc}/cache/warp")
for v in DISPLAY XAUTHORITY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS; do
  if [[ "${RUN_ISAAC_NO_DISPLAY:-0}" == "1" && ( "$v" == "DISPLAY" || "$v" == "XAUTHORITY" ) ]]; then continue; fi
  [[ -n "${!v:-}" ]] && CLEAN_ENV+=("$v=${!v}")
done

case "$METHOD" in
  min)
    CLEAN_ENV+=(
      LD_LIBRARY_PATH="$ROS/lib:$ROS/lib/x86_64-linux-gnu:$HRC/lib"
      AMENT_PREFIX_PATH="$HRC:$ROS"
      STEP3_ROS_PYPATHS="$HRC/local/lib/python3.10/dist-packages:$ROS/lib/python3.10/site-packages:$ROS/local/lib/python3.10/dist-packages"
    )
    if [[ "${EFFMASS_WS:-0}" == "1" ]]; then
      # (goal1 1-2d) ${EFFMASS_ROOT:-$HOME/effmass_mpc}/ws 의 effmass_msgs(메시지 타입 지원 라이브러리)·effmass_robot 을 쓸 때만. 기본 0 = 이전과 같음
      WS=${EFFMASS_ROOT:-$HOME/effmass_mpc}/ws/install
      CLEAN_ENV+=(
        LD_LIBRARY_PATH="$ROS/lib:$ROS/lib/x86_64-linux-gnu:$HRC/lib:$WS/effmass_msgs/lib"
        AMENT_PREFIX_PATH="$WS/effmass_msgs:$HRC:$ROS"
        STEP3_ROS_PYPATHS="$HRC/local/lib/python3.10/dist-packages:$ROS/lib/python3.10/site-packages:$ROS/local/lib/python3.10/dist-packages:$WS/effmass_msgs/local/lib/python3.10/dist-packages:$WS/effmass_robot/lib/python3.10/site-packages"
        EFFMASS_WS=1
      )
    fi
    INNER='exec "$@"'
    ;;
  source)
    # setup.bash 안의 python3 호출이 cwd 영향을 받지 않도록 홈에서 source
    INNER='cd "$HOME"; set +u; source /opt/ros/humble/setup.bash; source "$HRC_WS/install/setup.bash"; set -u; cd "$ISAAC_ROOT"; exec "$@"'
    ;;
  *) echo "ROS_METHOD 는 min 또는 source" >&2; exit 1;;
esac

if [[ "${1:-}" == "--print-env" ]]; then
  exec "${CLEAN_ENV[@]}" bash -c "${INNER/exec \"\$@\"/env | sort}" _
fi
[[ -x "$PY" ]] || { echo "[run_isaac_ros] $PY 없음" >&2; exit 2; }
cd "$ISAAC_ROOT"
exec "${CLEAN_ENV[@]}" bash -c "$INNER" _ "$PY" "$@"
