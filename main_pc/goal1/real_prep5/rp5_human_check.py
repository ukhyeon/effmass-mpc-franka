"""rp5_human_check.py --shm <name> [--seconds 10] [--min-rate 15] [--wait 8] - real_prep5 H2 prep (2026-11-20): is the human data coming?
Reads the shared-memory record that rp5_human_relay.py writes (/dev/shm/<name>, float64[509], seqlock; layout in rp5_human_relay.py)
as a plain file (np.fromfile; no SharedMemory object, so nothing is registered or removed). Waits up to --wait s for the block and the
first message, then counts the messages written (record index 7) for --seconds s.
One line: messages per s, largest gap between messages, input stamp -> relay receive delay (mean / max, only if the stamp is set),
human shapes in the last record. Exit 0: rate >= --min-rate; 1: rate too low; 2: no block / no message within --wait s.
Nothing is published or subscribed. Run with the system python: /usr/bin/python3 -I rp5_human_check.py --shm rr5_human"""
import argparse, os, sys, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--shm", required=True); ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--min-rate", type=float, default=15.0); ap.add_argument("--wait", type=float, default=8.0); a = ap.parse_args()
P = "/dev/shm/" + a.shm; LEN = 509


def rec():   # consistent copy of the record or None
    for _ in range(20):
        try:
            x = np.fromfile(P, np.float64, count=LEN)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if len(x) < LEN:
            return None
        if int(x[0]) % 2 == 0:
            y = np.fromfile(P, np.float64, count=LEN)
            if len(y) == LEN and y[0] == x[0]:
                return y
        time.sleep(0.001)
    return None


t_end = time.time() + a.wait; r = None
while time.time() < t_end:
    r = rec()
    if r is not None and r[7] >= 1:
        break
    time.sleep(0.05)
if r is None or r[7] < 1:
    print(f"human check: {'no shared memory ' + P if r is None else 'no human message'} within {a.wait:.0f} s -> rate 0 per s (need {a.min_rate:g})")
    sys.exit(2)
n0, t0 = r[7], time.time(); last_n, last_t, gap_max = n0, t0, 0.0; lat = []; seen = set()
while time.time() - t0 < a.seconds:
    time.sleep(0.005)
    r2 = rec()
    if r2 is None:
        continue
    now = time.time()
    if r2[7] != last_n:
        gap_max = max(gap_max, now - last_t); last_n, last_t = r2[7], now; r = r2
        st = r2[1] + r2[2] * 1e-9
        if st > 0 and r2[6] not in seen:
            seen.add(r2[6]); lat.append((r2[3] - st) * 1000.0)
el = time.time() - t0; gap_max = max(gap_max, time.time() - last_t)
rate = (last_n - n0) / el
radius = r[437:461]; nshape = int((radius > 0).sum())
ls = f"stamp->receive {np.mean(lat):.0f} / {np.max(lat):.0f} ms" if lat else "stamp->receive n/a (stamp not set)"
ok = rate >= a.min_rate
print(f"human check {el:.1f} s: {rate:.1f} messages per s (need {a.min_rate:g}), largest gap {gap_max * 1000:.0f} ms, {ls}, "
      f"shapes with radius > 0 in the last record {nshape}/24 -> {'ok' if ok else 'TOO FEW'}")
sys.exit(0 if ok else 1)
