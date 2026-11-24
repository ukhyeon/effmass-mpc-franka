"""rp5_out_filter.py - real_prep5 (2026-11-20): causal smoothing of the 60 Hz MPC joint target stream (output side).

Why: the MPC targets carry sample-mean jitter of the sampling optimizer; half of the target acceleration is above 15 Hz while the
motion itself is 1-3 Hz. The control-PC bridge turns that into torque ripple (noise). These filters act on the target sequence
only, are causal, and are reset to the current (measured) joints on (re)start.

  f = OutFilter(kind, f_hz=8.0, fs=60.0, n=9, deg=2, m=4)
  q, qd, qdd = f.reset(q_meas)          # state = q_meas, velocity 0, acceleration 0
  q, qd, qdd = f.step(q_target)         # every 60 Hz step; returns the smoothed target and its finite-difference velocity /
                                        # acceleration (qd = (q - q_prev) fs, qdd = (qd - qd_prev) fs)
  q, qd, qdd = f.step(q_target, plan_q=H)   # kind "blend" also needs the step's horizon positions H (n_points, J)

kinds (definitions as the control-PC analysis):
  none  : output = input
  ema   : first order, y += a (x - y), a = 1 - exp(-2 pi f / fs)
  bw2   : second-order Butterworth low-pass, fs 60 Hz, bilinear transform with pre-warping (as scipy.signal.butter(2, f, fs=fs)),
          direct form II transposed, state initialised to the steady state of the first value
  cd2   : two ema in series with the same a (critically damped second order, no overshoot)
  sg    : causal polynomial end-point fit (Savitzky-Golay end point): least-squares polynomial of degree deg over the last n
          targets, evaluated at the newest one (FIR, unity DC gain)
  blend : plan blending: mean over the last m plans of their prediction for this step: plan of step k-j, horizon point j
          (horizon point 0 = 1/60 s ahead, uniform spacing for the first 16 points); fewer plans right after a reset
Chain: OutFilter(kind="blend+bw2", ...) = blend, then bw2 on its output.
"""
import math

import numpy as np

KINDS = ("none", "ema", "bw2", "cd2", "sg", "blend")


def bw2_coeffs(f_hz, fs):
    """(b0, b1, b2, a1, a2) of the 2nd-order Butterworth low-pass, bilinear transform with pre-warping (scipy.signal.butter(2, f, fs))."""
    K = math.tan(math.pi * f_hz / fs); s2 = math.sqrt(2.0); nrm = 1.0 / (1.0 + s2 * K + K * K)
    b0 = K * K * nrm
    return b0, 2.0 * b0, b0, 2.0 * (K * K - 1.0) * nrm, (1.0 - s2 * K + K * K) * nrm


def sg_weights(n, deg):
    """weights w (n,) so that sum(w * x[-n:]) = value at t = 0 of the LS polynomial of degree deg over t = -(n-1) .. 0"""
    t = np.arange(-(n - 1), 1, dtype=np.float64)
    A = np.vander(t, deg + 1, increasing=True)          # columns 1, t, t^2 ...
    return np.linalg.pinv(A)[0]                          # value at t = 0 = coefficient of t^0


class _Stage:
    def __init__(self, kind, f_hz, fs, n, deg, m):
        assert kind in KINDS, kind
        self.kind, self.f, self.fs, self.n, self.deg, self.m = kind, float(f_hz), float(fs), int(n), int(deg), int(m)
        if kind in ("ema", "cd2"):
            self.a = 1.0 - math.exp(-2.0 * math.pi * self.f / self.fs)
        if kind == "bw2":
            self.c = bw2_coeffs(self.f, self.fs)
        if kind == "sg":
            self.w = sg_weights(self.n, self.deg)

    def reset(self, q):
        q = np.asarray(q, np.float64).copy()
        if self.kind in ("ema",):
            self.y = q.copy()
        elif self.kind == "cd2":
            self.y1 = q.copy(); self.y = q.copy()
        elif self.kind == "bw2":
            b0, b1, b2, a1, a2 = self.c
            self.z2 = (b2 - a2) * q; self.z1 = (b1 - a1) * q + self.z2
        elif self.kind == "sg":
            self.hist = np.repeat(q[None], self.n, 0)
        elif self.kind == "blend":
            self.plans = []
        return q

    def step(self, x, plan_q=None):
        x = np.asarray(x, np.float64)
        k = self.kind
        if k == "none":
            return x.copy()
        if k == "ema":
            self.y = self.y + self.a * (x - self.y); return self.y.copy()
        if k == "cd2":
            self.y1 = self.y1 + self.a * (x - self.y1); self.y = self.y + self.a * (self.y1 - self.y); return self.y.copy()
        if k == "bw2":
            b0, b1, b2, a1, a2 = self.c
            y = b0 * x + self.z1; self.z1 = b1 * x - a1 * y + self.z2; self.z2 = b2 * x - a2 * y
            return y
        if k == "sg":
            self.hist = np.concatenate([self.hist[1:], x[None]], 0); return self.w @ self.hist
        if k == "blend":
            if plan_q is None:   # no plan (held / infeasible step): restart the blend from this target
                self.plans = []; return x.copy()
            P = np.asarray(plan_q, np.float64)
            self.plans = ([P] + self.plans)[: self.m]
            # plan of step k-j predicts this step at its horizon point j (point 0 of the newest plan = x)
            vals = [x] + [p[j] for j, p in enumerate(self.plans) if j >= 1 and j < len(p)]
            return np.mean(vals, 0)
        raise ValueError(k)


class OutFilter:
    def __init__(self, kind="bw2", f_hz=8.0, fs=60.0, n=9, deg=2, m=4):
        self.kind = kind; self.fs = float(fs)
        self.stages = [_Stage(k, f_hz, fs, n, deg, m) for k in kind.split("+")]
        self.needs_plan = any(s.kind == "blend" for s in self.stages)
        self.q = self.v = None

    def reset(self, q_meas):
        q = np.asarray(q_meas, np.float64).copy()
        for s in self.stages:
            s.reset(q)
        self.q = q.copy(); self.v = np.zeros_like(q)
        return self.q.copy(), self.v.copy(), np.zeros_like(q)

    def step(self, q_target, plan_q=None):
        assert self.q is not None, "reset() first"
        y = np.asarray(q_target, np.float64)
        for s in self.stages:
            y = s.step(y, plan_q if s.kind == "blend" else None)
        v = (y - self.q) * self.fs; a = (v - self.v) * self.fs
        self.q = y.copy(); self.v = v
        return y.copy(), v.copy(), a

    def describe(self):
        return dict(kind=self.kind, stages=[dict(kind=s.kind, f_hz=s.f, n=s.n, deg=s.deg, m=s.m) for s in self.stages], fs=self.fs)


def _selftest():
    rng = np.random.default_rng(0); fs = 60.0
    try:
        from scipy import signal
        for f in (5.0, 8.0, 12.0):
            b, a = signal.butter(2, f, fs=fs); c = bw2_coeffs(f, fs)
            assert np.allclose(b, c[:3], atol=1e-12) and np.allclose(a[1:], c[3:], atol=1e-12), (f, b, a, c)
            x = rng.normal(size=(300, 3)).cumsum(0); x0 = x[0]
            F = OutFilter("bw2", f); F.reset(x0); y = np.array([F.step(r)[0] for r in x])
            zi = signal.lfilter_zi(b, a); ys = np.stack([signal.lfilter(b, a, x[:, j], zi=zi * x0[j])[0] for j in range(3)], 1)
            assert np.allclose(y, ys, atol=1e-9), np.abs(y - ys).max()
        print("bw2 matches scipy.signal.butter / lfilter (steady-state init)")
    except ImportError:
        print("scipy not available: bw2 check skipped")
    for kind in ("ema", "cd2", "sg", "bw2", "none"):
        F = OutFilter(kind, 8.0); F.reset(np.ones(7)); out = [F.step(np.ones(7) * 2.0)[0] for _ in range(400)]
        assert np.allclose(out[-1], 2.0, atol=1e-6), (kind, out[-1])   # unity DC gain
        F.reset(np.ones(7)); q, v, a = F.step(np.ones(7)); assert np.allclose(q, 1) and np.allclose(v, 0) and np.allclose(a, 0), kind   # reset = steady
    F = OutFilter("blend", m=3); F.reset(np.zeros(1)); H = np.arange(1, 31, dtype=float)[:, None]
    assert F.step(np.ones(1), H)[0][0] == 1.0
    assert abs(F.step(np.ones(1) * 2, H + 1)[0][0] - np.mean([2.0, 2.0])) < 1e-12   # plan k-1 at point 1 = H[1] = 2
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
