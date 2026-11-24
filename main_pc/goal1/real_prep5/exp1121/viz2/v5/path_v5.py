"""path_v5.py - viz2 v5 (wrap 2026-11-24): the TCP future path drawn on the 3-D panels (single and comparison mode, one rule).
If the run's _steps.npz has plan_q and plan_t (the MPC plan), the path of step k = the planned joints of step k within the next
horizon_s (plan_q[k] (H, 7) joint positions; plan_t (H,) or (n, H) times of the plan points, seconds after the step, or absolute (then
the step time is subtracted)) -> TCP with exp_ana tcp_of. Else the recorded TCP of the next horizon_s (label_v5 tcp = exp_ana
tcp_of(meas_q)), as before. Source "planned" | "recorded"; caption "planned path 0.5 s" | "path 0.5 s".
Line colour: RGB (dark red 220, 0, 0; pathred 11-24), drawn by compose_single.py and compose_v5.py.
Usage (library): P = Path(run_prefix, tcp_recorded, step_times); P.source, P.caption(h), P.points(k, h); path_v5.RGB;
gif_frame(im) = GIF palette frame that keeps RGB (both composers' GIFs, pathgif 11-24)."""
import os, sys
import numpy as np
E1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121")
RGB = (220, 0, 0)   # (pathred 11-24) colour of the future path line (single and comparison mode read it here); was cyan (0, 185, 200)


class Path:
    def __init__(self, run, tcp_rec, t_steps):
        self.tcp = np.asarray(tcp_rec, float); self.t = np.asarray(t_steps, float); self.plan_q = self.plan_t = None; self.cache = {}
        try:
            z = np.load(run + "_steps.npz")
            if "plan_q" in z.files and "plan_t" in z.files:
                self.plan_q = np.asarray(z["plan_q"], float); self.plan_t = np.asarray(z["plan_t"], float)
        except OSError:
            pass
        self.source = "planned" if self.plan_q is not None else "recorded"

    def caption(self, h=0.5):
        return ("planned path %.1f s" if self.source == "planned" else "path %.1f s") % h

    def describe(self, h=0.5):
        return ("MPC plan of the step (steps npz plan_q / plan_t, %d points) within the next %.2f s, TCP by exp_ana tcp_of" % (self.plan_q.shape[1], h)) \
            if self.source == "planned" else ("recorded TCP of the next %.2f s (exp_ana tcp_of(meas_q); no plan_q / plan_t in the steps npz)" % h)

    def points(self, k, h=0.5):   # (m, 3) robot base frame, from the TCP at step k
        if self.source == "recorded":
            return self.tcp[k:k + int(round(h * 60)) + 1]
        if k in self.cache:
            return self.cache[k]
        q = self.plan_q[min(k, len(self.plan_q) - 1)]; tt = self.plan_t if self.plan_t.ndim == 1 else self.plan_t[min(k, len(self.plan_t) - 1)]
        tt = np.asarray(tt, float)
        if np.nanmax(np.abs(tt)) > 1e5:   # absolute times (ns or s) -> seconds after the plan's first point (= the step)
            tt = tt / (1e9 if np.nanmax(np.abs(tt)) > 1e12 else 1.0); tt = tt - tt[np.isfinite(tt)][0]
        m = np.isfinite(tt) & (tt <= h + 1e-9) & np.all(np.isfinite(q), axis=1)
        sys.path.insert(0, E1)
        import exp_ana as X   # read only
        P = np.vstack([self.tcp[k:k + 1], X.tcp_of(q[m])]) if m.any() else self.tcp[k:k + 1]
        self.cache[k] = P
        return P


GIF_KEEP = [(25, 95, 215), (240, 110, 10), (20, 20, 20), (95, 95, 95), (255, 255, 255)]   # (pathgif) method colours (composers' COL), text, white


def gif_frame(im, colors=128, keep=GIF_KEEP):   # (pathgif 11-24) GIF palette frame: median cut of colors - 1 - len(keep) entries + RGB and the
    from PIL import Image          # fixed drawing colours always in the palette (a 128-colour median cut alone merged the few path pixels
                                   # into the orange / brown of the volumes; no dithering, as before)
    fix = [tuple(RGB)] + [tuple(c) for c in keep]; m = colors - len(fix)
    q = im.convert("RGB").quantize(colors=m, method=Image.MEDIANCUT, dither=Image.NONE); n = min(m, len(q.getpalette()) // 3)
    pal = q.getpalette()[:3 * n] + [v for c in fix for v in c]; pal += list(RGB) * (256 - len(pal) // 3); pi = Image.new("P", (1, 1)); pi.putpalette(pal)   # unused = RGB
    return im.convert("RGB").quantize(palette=pi, dither=Image.NONE)
