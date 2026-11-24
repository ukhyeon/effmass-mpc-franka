"""rp5_human_relay.py - real_prep5 H1: human receiver in its own process (as rp2_js_relay.py for the joint state).
Subscribes to effmass_msgs/HumanInertiaState on /effmass/human_inertia_state (QoS as g1_human_link.HumanInertiaLink: best effort,
depth 5) in its own ROS domain (the human node's, e.g. 30 localhost) and overwrites one record in POSIX shared memory --shm-name
(no queue). Nothing is published. rp2_mpc_rt.py --human shm reads the record once per step.
Record float64[509], seqlock (seq odd while writing; a reader accepts a copy only if seq is even and unchanged):
  0 seq, 1 input_stamp.sec, 2 input_stamp.nanosec, 3 receive wall time (time.time), 4 input_count, 5 compute_ms, 6 receive time
  (time.monotonic), 7 messages written so far, then com_position 57 @8, com_velocity 57 @65, inv_inertia 171 @122, shape_p0 72 @293,
  shape_p1 72 @365, shape_radius 24 @437, shape_segment 24 @461, shape_type 24 @485 (node world frame, as received).
Creates the block, removes it on exit. Stops after --duration s, on a stop file, or on Ctrl+C.
Run with: goal1/scripts/g1_run_human.sh py <this file> [--duration 900] [--stop-file F]"""
import argparse, os, time
import numpy as np
import rclpy
from multiprocessing import shared_memory
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from effmass_msgs.msg import HumanInertiaState

ap = argparse.ArgumentParser(); ap.add_argument("--shm-name", default="rp5_human"); ap.add_argument("--duration", type=float, default=900.0)
ap.add_argument("--stop-file", default=""); a = ap.parse_args()
LEN = 509
OFF = [("com_position", 8, 57), ("com_velocity", 65, 57), ("inv_inertia", 122, 171), ("shape_p0", 293, 72), ("shape_p1", 365, 72),
       ("shape_radius", 437, 24), ("shape_segment", 461, 24), ("shape_type", 485, 24)]
try:
    old = shared_memory.SharedMemory(name=a.shm_name); old.close(); old.unlink()
except FileNotFoundError:
    pass
shm = shared_memory.SharedMemory(name=a.shm_name, create=True, size=LEN * 8); B = np.ndarray((LEN,), np.float64, buffer=shm.buf); B[:] = 0.0
rclpy.init(); node = rclpy.create_node("rp5_human_relay"); N = [0, 0]   # written, skipped (unexpected sizes)


def cb(m):
    arrs = [np.asarray(getattr(m, f), np.float64) for f, _, _ in OFF]
    if any(len(x) != n for x, (_, _, n) in zip(arrs, OFF)):
        N[1] += 1
        return
    s = B[0]; B[0] = s + 1.0   # odd: writing
    B[1] = m.input_stamp.sec; B[2] = m.input_stamp.nanosec; B[3] = time.time(); B[4] = m.input_count; B[5] = m.compute_ms; B[6] = time.monotonic()
    B[7] = N[0] + 1
    for x, (_, o, n) in zip(arrs, OFF):
        B[o:o + n] = x
    B[0] = s + 2.0; N[0] += 1


qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
node.create_subscription(HumanInertiaState, "/effmass/human_inertia_state", cb, qos); t0 = time.time()
print(f"[human-relay] /effmass/human_inertia_state (domain {os.environ.get('ROS_DOMAIN_ID', '?')}) -> shared memory {a.shm_name}", flush=True)
try:
    nxt = t0 + 5.0
    while time.time() - t0 < a.duration and not (a.stop_file and os.path.exists(a.stop_file)):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() >= nxt:
            nxt += 5.0; print(f"[human-relay] {N[0]} messages ({N[1]} skipped)", flush=True)
except KeyboardInterrupt:
    pass
finally:
    print(f"[human-relay] messages {N[0]} (skipped {N[1]}) in {time.time() - t0:.1f} s", flush=True)
    node.destroy_node(); rclpy.try_shutdown() if hasattr(rclpy, "try_shutdown") else rclpy.shutdown(); del B; shm.close(); shm.unlink()
