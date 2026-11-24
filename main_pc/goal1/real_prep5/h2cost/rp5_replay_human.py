"""rp5_replay_human.py --replay <name>_human.npz --shm-name X --start-log <mpc.log> [--duration S] [--stop-file F] - H2 cost study (2026-11-20).
Writes the RECORDED human records (rp2_mpc_rt.py --rec-human: every record the real MPC picked up, raw, incl. frames its jump filter
rejected) into the rp5_human_relay.py shared-memory record (float64[509], seqlock, same layout), at the recorded times: timeline t = 0
when --start-pattern ("warm reset") appears in --start-log (= the MPC's first step), recorded time = record stamp - first publish of
the recorded run. One new seqlock record per recorded record; stamps / receive times = now. Before t = 0 the first record with t >= 0 is
written once per second (exp1121: --pre-rate R = R times per s, e.g. 30 so that a human-rate check sees the replay before the
timeline starts; --info-json F: replay facts (file, records, times, timeline start wall time) written at the end).
Positions as recorded (node world frame). No ROS. System python (numpy only)."""
import argparse, os, time
import numpy as np
from multiprocessing import shared_memory
ap = argparse.ArgumentParser(); ap.add_argument("--replay", required=True); ap.add_argument("--shm-name", default="rp5_human")
ap.add_argument("--start-log", default=""); ap.add_argument("--start-pattern", default="warm reset"); ap.add_argument("--duration", type=float, default=400.0)
ap.add_argument("--stop-file", default=""); ap.add_argument("--pre-rate", type=float, default=1.0, help="(exp1121) records per s before the timeline start")
ap.add_argument("--info-json", default="", help="(exp1121) replay facts -> this file at the end"); a = ap.parse_args()
z = np.load(a.replay); T = (z["msg_stamp_ns"] - z["step_t_pub_ns"][0]) / 1e9
F = dict(com=z["msg_com"], vel=z["msg_vel"], inv=z["msg_inv"], p0=z["msg_p0"], p1=z["msg_p1"], rad=z["msg_rad"], seg=z["msg_seg"], type=z["msg_type"])
OFF = dict(com=8, vel=65, inv=122, p0=293, p1=365, rad=437, seg=461, type=485); LEN = 509
i0 = int(np.argmax(T >= 0))
try:
    old = shared_memory.SharedMemory(name=a.shm_name); old.close(); old.unlink()
except FileNotFoundError:
    pass
shm = shared_memory.SharedMemory(name=a.shm_name, create=True, size=LEN * 8); B = np.ndarray((LEN,), np.float64, buffer=shm.buf); B[:] = 0.0
n = 0; last = -1; t_start = None; T0 = time.time(); nxt_pre = 0.0; t_start_wall = None; n_pre = 0
print(f"[replay-human] {a.replay}: {len(T)} records, {T[0]:.2f} .. {T[-1]:.2f} s; shm {a.shm_name}", flush=True)


def write(ri):
    global n
    wall = time.time(); s = B[0]; B[0] = s + 1.0
    B[1] = int(wall); B[2] = int((wall - int(wall)) * 1e9); B[3] = wall; B[4] = float(z["msg_input_count"][ri]); B[5] = 0.0; B[6] = time.monotonic(); B[7] = n + 1
    for k, o in OFF.items():
        v = np.asarray(F[k][ri], np.float64).ravel(); B[o:o + v.size] = v
    B[0] = s + 2.0; n += 1


try:
    while time.time() - T0 < a.duration and not (a.stop_file and os.path.exists(a.stop_file)):
        now = time.monotonic()
        if t_start is None and a.start_log and os.path.exists(a.start_log):
            try:
                if a.start_pattern in open(a.start_log, errors="ignore").read():
                    t_start = now; t_start_wall = time.time(); n_pre = n; print(f"[replay-human] timeline start ({n} records written before)", flush=True)
            except OSError:
                pass
        if t_start is None:
            if now >= nxt_pre:
                write(i0); nxt_pre = now + 1.0 / max(a.pre_rate, 1e-3)
            time.sleep(0.02); continue
        ri = int(np.searchsorted(T, now - t_start, side="right")) - 1
        if ri >= i0 and ri != last:
            write(ri); last = ri
        time.sleep(0.002)   # after the last record: keep the block (no new records) until the stop file
except KeyboardInterrupt:
    pass
finally:
    print(f"[replay-human] records written {n} in {time.time() - T0:.1f} s", flush=True)
    if a.info_json:   # (exp1121)
        import json
        json.dump(dict(replay=os.path.abspath(a.replay), records=int(len(T)), t_first_s=float(T[0]), t_last_s=float(T[-1]), first_timeline_record=i0,
                       written=n, written_before_start=n_pre, timeline_start_wall=t_start_wall, start_pattern=a.start_pattern, start_log=a.start_log,
                       pre_rate=a.pre_rate, rec_meas_q0=[float(x) for x in z["meas_q"][0]] if "meas_q" in z.files else None,
                       rec_scene_file=str(z["scene_file"]) if "scene_file" in z.files else None), open(a.info_json, "w"), indent=1)
    del B; shm.close(); shm.unlink()
