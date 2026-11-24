#!/usr/bin/env bash
# rp5_mps.sh on|off|status - NVIDIA MPS (multi-process service) for this user, no sudo (2026-11-20).
#   on     : starts the MPS control daemon (nvidia-cuda-mps-control -d). Pipe directory = the default /tmp/nvidia-mps, so that clients started
#            from env -i (run_isaac.sh, run_isaac_ros.sh, rp5_sensing.sh) find it without any variable. Log directory
#            goal1/real_prep5/mps_log (the default /var/log/nvidia-mps needs root). Refuses while a CUDA process runs: MPS only applies to
#            processes started after it, and a process started before would keep its own context next to the MPS server.
#   off    : echo quit | nvidia-cuda-mps-control, waits until the server and the daemon are gone (clients must have ended first).
#   status : daemon / server processes and the GPU process list (an MPS server shows as nvidia-cuda-mps-server, its clients as type M+C).
#   The server itself starts with the first CUDA client. Per-client SM share: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE in that client's
#   environment before CUDA starts (rp5_gpu_load.py --mps-threads P sets it inside the process).
set -u
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${EFFMASS_ROOT:-$HOME/effmass_mpc}/goal1/real_prep5/mps_log
# process names are cut to 15 characters ("nvidia-cuda-mps"), so match the command line from its start
dmn() { pgrep -u "$(id -u)" -f '^(/usr/bin/)?nvidia-cuda-mps-control'; }
procs() { dmn; pgrep -u "$(id -u)" -f '^(/usr/bin/)?nvidia-cuda-mps-server'; }
apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null; }
case "${1:-status}" in
  on)
    if dmn > /dev/null; then echo "[mps] daemon already running"; exit 0; fi
    a=$(apps); [[ -z $a ]] || { echo "[mps] CUDA processes are running, stop them first:"; echo "$a"; exit 3; }
    mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d || { echo "[mps] daemon start failed"; exit 4; }
    sleep 1; dmn > /dev/null && echo "[mps] daemon on (pipe $CUDA_MPS_PIPE_DIRECTORY, log $CUDA_MPS_LOG_DIRECTORY); the server starts with the first CUDA client" \
      || { echo "[mps] daemon not running, see $CUDA_MPS_LOG_DIRECTORY/control.log"; exit 4; } ;;
  off)
    if ! dmn > /dev/null; then echo "[mps] daemon not running"; else
      echo quit | nvidia-cuda-mps-control
      for i in $(seq 1 20); do [[ -z $(procs) ]] && break; sleep 0.5; done
      if [[ -n $(procs) ]]; then echo "[mps] still running (clients left?):"; ps -o pid,args -p "$(procs | paste -sd,)"; apps; exit 5; fi
      echo "[mps] off"
    fi ;;
  status)
    p=$(procs); if [[ -n $p ]]; then echo "[mps] processes:"; ps -o pid,etime,args -p "$(echo "$p" | paste -sd,)"; else echo "[mps] no daemon / server"; fi
    nvidia-smi | sed -n '/Processes:/,$p' ;;
  *) echo "usage: $0 on|off|status"; exit 2 ;;
esac
