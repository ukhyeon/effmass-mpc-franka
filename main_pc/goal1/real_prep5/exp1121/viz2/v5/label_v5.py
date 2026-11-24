"""label_v5.py - viz2 v5 single mode (final 2026-11-24): per-step label values of the 3-D panel and the TCP future path.
For every step (exp_ana Run of the run, read only): the closest (robot sphere, arm-part shape) pair of Run.gaps() (human_arm_parts of the
record, shoulders included when recorded; the same pair as lines.npz d and eta_v5), u = robot sphere centre -> nearest shape point (the
MPC kernel direction, mode "closest"), ah = u' inv_h u (recorded node inverse inertia of that segment), ar = u' inv_r u (robot, RobotInertiaNP):
  m_h   = 1 / ah                                  (human effective mass along u)
  eta   = recorded rule (eta_v5.of_run, exp_ana eta_eff): human_mass_mode reduced: clip(d0 + k (sqrt(mu (1/m_iso + ar)) - 1)+, d0, eta_max),
          mu = 1 / (ah + ar); human: clip(d0 + k (sqrt(m_h / m_iso) - 1)+, ...); m_iso = ISO with the recorded override (shoulders 3 kg)
  extra = eta - d0
So m_h and extra come from the same step, part and u. Smoothed copies (mh_s, extra_s): trailing moving average over avg_s (compose_single
--label-avg, default 0 = the raw values) that restarts when the part changes. tcp (n, 3): exp_ana tcp_of(meas_q), robot base frame (future path = tcp[k .. k + 0.5 s]).
Cached in <work>/label.npz (key: run name + rule + avg + left-out robot spheres).
(robex 11-24) The pair leaves out the robot spheres of link0 / link1 (robex_v5.py; the MPC and exp_ana distances are unchanged). Usage (library): label_v5.load(run_prefix, work_dir) -> dict."""
import os, sys
import numpy as np
E1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121"); HERE = os.path.dirname(os.path.abspath(__file__))
NICE = {10: "left shoulder", 11: "right shoulder", 13: "left upper arm", 14: "right upper arm", 15: "left forearm", 16: "right forearm",
        17: "left hand", 18: "right hand"}


def smooth(x, seg, n):   # trailing mean of the last n finite values within the same part run (restart at a part change)
    y = np.full(len(x), np.nan); s0 = 0
    for i in range(len(x)):
        if i == 0 or seg[i] != seg[i - 1]:
            s0 = i
        w = x[max(s0, i - n + 1):i + 1]; w = w[np.isfinite(w)]
        y[i] = w.mean() if len(w) and np.isfinite(x[i]) else np.nan
    return y


def load(run, work, avg_s=0.2):
    sys.path.insert(0, HERE); import eta_v5, robex_v5   # noqa: E402  (robex 11-24)
    name = os.path.basename(run); rule = eta_v5.rule_of(run); cache = os.path.join(work, "label.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if str(z["name"]) == name and str(z["rule"]) == rule and abs(float(z["avg_s"]) - avg_s) < 1e-9 and str(z["robex"] if "robex" in z.files else "") == robex_v5.key(run):   # (robex) same left-out spheres
            return {k: z[k] for k in z.files}
    sys.path.insert(0, E1)
    import exp_ana as X   # read only
    r = X.Run(run); robex_v5.apply(r); g, sp, sh, cp, _ = r.gaps(); n = r.n   # (robex 11-24) link0 / link1 spheres left out of the pair
    E = eta_v5.of_run(r); d0 = float(E["d_base"])
    seg = np.full(n, -1, np.int16); mh = np.full(n, np.nan); mu = np.full(n, np.nan)
    for i in range(n):
        h = int(r.hidx[i])
        if h < 0 or sp[i] < 0 or not np.isfinite(g[i]):
            continue
        s_ = int(sp[i]); kk = int(sh[i]); sgid = int(r.Hseg[h][kk]); c = r.P[i, s_]; u = cp[i] - c; u = u / (np.linalg.norm(u) + 1e-9)
        _, inv_r, _ = X.M.RI.compute(r.q[i]); ar = float(u @ inv_r[s_] @ u); ah = float(u @ r.Hinv[h, sgid] @ u)
        seg[i] = sgid; mh[i] = 1.0 / max(ah, 1e-9); mu[i] = 1.0 / (ah + ar)
    ns = max(1, int(round(avg_s * 60)))
    out = dict(name=np.array(name), rule=np.array(rule), robex=np.array(robex_v5.key(r)), avg_s=np.array(avg_s), seg=seg, mh=mh, mu=mu, eta=E["eta"], extra=E["extra"], d_base=np.array(d0),
               mh_s=smooth(mh, seg, ns), extra_s=smooth(np.asarray(E["extra"], float), seg, ns), tcp=X.tcp_of(r.q), t=r.t)
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    o = load(sys.argv[1], sys.argv[2]); k = int(np.nanargmin(np.where(o["seg"] >= 0, np.arange(len(o["seg"])), np.inf))) if (o["seg"] >= 0).any() else 0
    print("rule %s, d_base %.3f, steps with a part %d; m_h median %.1f kg" % (o["rule"], float(o["d_base"]), int((o["seg"] >= 0).sum()), np.nanmedian(o["mh"])))
