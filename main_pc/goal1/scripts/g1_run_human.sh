#!/usr/bin/env bash
# g1_run_human.sh <validate|node|py SCRIPT> [args] — 1-2b 사람 사본 노드 실행 환경 (환경 점검 (F) 설정과 같음)
#   env -i + /opt/ros/humble + ~/hrc_ws/install (hrc_interfaces·원본 비교용, 읽기만) + ${EFFMASS_ROOT:-$HOME/effmass_mpc}/ws/install
#   CPU 1스레드: CUDA_VISIBLE_DEVICES= (빈 값), OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, 노드는 torch.set_num_threads(1)
#   PYTHONDONTWRITEBYTECODE=1 (~/hrc_ws 에 __pycache__ 안 생기게), ROS_LOG_DIR=${EFFMASS_ROOT:-$HOME/effmass_mpc}/env_check/ros_log, ROS_DOMAIN_ID=30, localhost
set -euo pipefail
MODE="${1:?validate|node|py}"; shift
exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" SHELL=/bin/bash LANG=C.UTF-8 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  EFFMASS_ROOT="${EFFMASS_ROOT:-$HOME/effmass_mpc}" HRC_WS="${HRC_WS:-$HOME/hrc_ws}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ROS_DOMAIN_ID=30 ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR="${EFFMASS_ROOT:-$HOME/effmass_mpc}/env_check/ros_log" G1_MODE="$MODE" \
  bash -c 'cd "$HOME"; set +u; source /opt/ros/humble/setup.bash; source "$HRC_WS/install/setup.bash"; source "${EFFMASS_ROOT:-$HOME/effmass_mpc}/ws/install/setup.bash"; set -u
    case "$G1_MODE" in
      validate) exec /usr/bin/python3 "${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/scripts/g1_human_validate.py" "$@" ;;
      node) exec /usr/bin/python3 -c "from effmass_human.human_inertia_node import main; main()" --ros-args "$@" ;;
      py) exec /usr/bin/python3 "$@" ;;
    esac' _ "$@"
