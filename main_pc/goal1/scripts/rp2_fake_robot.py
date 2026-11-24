"""rp2_fake_robot.py - real_prep2 test robot (no simulator): takes the first point of each trajectory_msgs/JointTrajectory on
--jt-topic as its joint target and publishes sensor_msgs/JointState (fr3_joint1..7) at --rate-hz (default 1000) on --js-topic.
  tau > 0: first-order lag q += (target - q) (1 - exp(-dt / tau)) per tick (dt = 1 / rate), velocity = (target - q) / tau
  tau = 0: q = target at once, velocity = the point's velocity
Starts at --q0 (7 values, comma separated). Stops after --duration s, or 5 s after the last trajectory once one has arrived.
Writes <out>_fake.npz: receive wall times and first-point positions of every trajectory, publish wall times (1 kHz ticks).
Run with ROS_LOCALHOST_ONLY=1 (goal1/scripts/g1_run_human.sh py sets ROS_DOMAIN_ID 30, ROS_LOCALHOST_ONLY 1).
real_prep4 options (defaults = as before): --jt-qos best_effort (depth 1, as the c6 gateway), --name-order gateway (joint names published
as fr3_joint1,3,2,4..7 like the gateway), --stop-pub-after S (stop publishing joint states S s after start; trajectories are still
received and recorded)."""
import argparse, math, threading, time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
ap = argparse.ArgumentParser()
ap.add_argument("--js-topic", default="/NS_1/franka/joint_states"); ap.add_argument("--jt-topic", default="/mpc/joint_trajectory")
ap.add_argument("--tau-ms", type=float, default=10.0); ap.add_argument("--q0", required=True); ap.add_argument("--duration", type=float, default=600.0)
ap.add_argument("--out", required=True)
ap.add_argument("--js-qos", default="reliable", choices=["reliable", "best_effort"], help="QoS of the joint-state publisher (best_effort = rclpy qos_profile_sensor_data)")
ap.add_argument("--rate-hz", type=float, default=1000.0, help="joint state publish rate (and model update rate), per s")
ap.add_argument("--jt-qos", default="reliable", choices=["reliable", "best_effort"])
ap.add_argument("--name-order", default="sorted", choices=["sorted", "gateway"])
ap.add_argument("--stop-pub-after", type=float, default=0.0, help="s; 0 = never")
a = ap.parse_args()
NAMES = [f"fr3_joint{k + 1}" for k in range(7)]; q = np.array([float(x) for x in a.q0.split(",")]); tgt = q.copy(); tv = np.zeros(7); v = np.zeros(7)
LOCK = threading.Lock(); RECV_T, RECV_Q = [], []
rclpy.init(); node = rclpy.create_node("rp2_fake_robot")


def on_jt(m):
    global tgt, tv
    if not m.points:
        return
    idx = {n: i for i, n in enumerate(m.joint_names)}; p = m.points[0]
    t = np.array([p.positions[idx[n]] for n in NAMES]); vv = np.array([p.velocities[idx[n]] for n in NAMES]) if len(p.velocities) == len(m.joint_names) else np.zeros(7)
    with LOCK:
        tgt, tv = t, vv
    RECV_T.append(time.time_ns()); RECV_Q.append(t)


node.create_subscription(JointTrajectory, a.jt_topic, on_jt, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT) if a.jt_qos == "best_effort" else QoSProfile(depth=10));
ORD = [0, 2, 1, 3, 4, 5, 6] if a.name_order == "gateway" else list(range(7))
pub = node.create_publisher(JointState, a.js_topic, qos_profile_sensor_data if a.js_qos == "best_effort" else QoSProfile(depth=10))
ex = SingleThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
dt = 1.0 / a.rate_hz; alpha = 1.0 - math.exp(-dt / (a.tau_ms / 1000.0)) if a.tau_ms > 0 else 1.0; PUB_T = []; t0 = time.perf_counter(); k = 0
print(f"[fake] rate {a.rate_hz} Hz, tau {a.tau_ms} ms, js {a.js_topic}, jt {a.jt_topic}", flush=True)
while True:
    k += 1; due = t0 + k * dt; now = time.perf_counter()
    if due > now:
        time.sleep(due - now)
    with LOCK:
        t, vv = tgt.copy(), tv.copy()
    if a.tau_ms > 0:
        q = q + alpha * (t - q); v = (t - q) / (a.tau_ms / 1000.0)
    else:
        q = t; v = vv
    if not (a.stop_pub_after > 0 and time.perf_counter() - t0 > a.stop_pub_after):
        m = JointState(); m.name = [NAMES[i] for i in ORD]; m.position = [float(q[i]) for i in ORD]; m.velocity = [float(v[i]) for i in ORD]; m.effort = [0.0] * 7
        m.header.stamp = node.get_clock().now().to_msg(); pub.publish(m); PUB_T.append(time.time_ns())
    el = time.perf_counter() - t0
    if el > a.duration or (RECV_T and (time.time_ns() - RECV_T[-1]) / 1e9 > 5.0):
        break
ex.shutdown(); node.destroy_node(); rclpy.shutdown()
np.savez(a.out + "_fake.npz", recv_t_ns=np.array(RECV_T, np.int64), recv_q=np.array(RECV_Q, np.float32), pub_t_ns=np.array(PUB_T, np.int64), tau_ms=a.tau_ms, rate_hz=a.rate_hz)
pt = np.diff(np.array(PUB_T)) / 1e6
print(f"[fake] trajectories received {len(RECV_T)}, joint states published {len(PUB_T)} ({len(PUB_T) / max(el, 1e-9):.0f} per s, period mean {pt.mean():.3f} ms, p99 {np.percentile(pt, 99):.3f}, max {pt.max():.3f})", flush=True)
