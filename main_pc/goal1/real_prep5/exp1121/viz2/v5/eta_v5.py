"""eta_v5.py - viz2 v5 text (2026-11-23): per-step target clearance of the closest arm part, for the "gap N cm (+E cm)" label.
For every step: the closest (robot sphere, arm capsule) pair of exp_ana Run.gaps() (arm parts only, the same pair as lines.npz d),
mu = 1 / (u'(inv_h + inv_r)u) along robot point -> arm point, eta = clip(d_base + k (sqrt(mu (1/m_iso + u'inv_r u)) - 1)+, d_base,
max(eta_max, d_base)) (exp_ana eta_eff, arm branch), extra = eta - d_base.
d_base (uniform base clearance) from the run's _exp.json: key "human_d_base", else "--human-d-base X" in its "mpc_human_options",
else the summary human cost eta0 (simulation runs without exp.json); the source is returned with the values.
k / eta_max = exp_ana Run (exp.json human_k_gain / human_eta_max, else summary). Cached in <work>/eta.npz (keyed by the run name).
(shviz 11-24) The eta rule follows the run's record (exp_ana Run): human_mass_mode reduced (as above) | human (eta = clip(d_base + k
(sqrt(m_h / m_iso) - 1)+, ...), m_h = 1 / (u' inv_h u), kernel mode gain_part_h); m_iso = ISO per segment with the recorded human_iso_override
(e.g. shoulders 3 kg). Old records (no keys): reduced, no override = the values before. Cache key "rule" = mode + override.
eta_v5.of_run(exp_ana Run) (win2 11-24): the same values for a Run object (no cache; window_v5 uses it on a base-excluded Run).
(robex 11-24) load() leaves out the robot spheres of link0 / link1 (robex_v5.py) before the pair; of_run(r) uses r as given (callers apply it).
Usage (library): eta_v5.load(run_prefix, work_dir) -> dict(eta (n,), extra (n,), d_base, d_base_src, k_gain, eta_max)."""
import json, os, re, sys
import numpy as np
E1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121")


def d_base_of(run, summary_eta0):
    p = run + "_exp.json"
    if os.path.exists(p):
        E = json.load(open(p))
        if "human_d_base" in E:
            return float(E["human_d_base"]), "exp.json human_d_base"
        m = re.search(r"--human-d-base\s+([0-9.eE+-]+)", str(E.get("mpc_human_options", "")))
        if m:
            return float(m.group(1)), "exp.json mpc_human_options --human-d-base"
    return float(summary_eta0), "summary human cost eta0"


def load(run, work):
    cache = os.path.join(work, "eta.npz"); name = os.path.basename(run)
    if os.path.exists(cache):
        z = np.load(cache)
        if str(z["name"]) == name and str(z["rule"] if "rule" in z.files else "reduced") == rule_of(run) and str(z["robex"] if "robex" in z.files else "") == _rx().key(run):   # (shviz) same run and eta rule; (robex) same left-out spheres
            return dict(eta=z["eta"], extra=z["extra"], d_base=float(z["d_base"]), d_base_src=str(z["d_base_src"]), k_gain=float(z["k_gain"]), eta_max=float(z["eta_max"]))
    sys.path.insert(0, E1)
    import exp_ana as X   # read only
    r = X.Run(run); _rx().apply(r); out = of_run(r)   # (robex 11-24) link0 / link1 spheres left out of the pair
    np.savez(cache, name=np.array(name), rule=np.array(rule_of(run)), robex=np.array(_rx().key(r)), **{k: np.asarray(v) for k, v in out.items()})
    return out


def _rx():   # (robex 11-24)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import robex_v5
    return robex_v5


def rule_of(run):   # (shviz 11-24) the recorded eta rule as text ("reduced" for old records)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import parts_v5
    mode, io, _ = parts_v5.mass_rule(run)
    return mode + ("" if not io else " " + ",".join("%s:%g" % kv for kv in sorted(io.items())))


def of_run(r):   # (win2 11-24) the same per-step eta / extra for an exp_ana Run object (its hidx may be remapped by the caller)
    sys.path.insert(0, E1)
    import exp_ana as X   # read only
    g, sp, sh, cp, _ = r.gaps(); n = r.n
    d0, src = d_base_of(r.p, r.d_base); kg, em = float(r.k_gain), float(r.eta_max)
    iso = np.asarray(getattr(r, "iso", X.iso_mass()), float); mode = str(getattr(r, "mass_mode", "reduced"))   # (shviz) recorded reference masses / rule
    eta = np.full(n, np.nan)
    for i in range(n):
        h = int(r.hidx[i])
        if h < 0 or sp[i] < 0 or not np.isfinite(g[i]):
            continue
        s_ = int(sp[i]); kk = int(sh[i]); sgid = int(r.Hseg[h][kk]); c = r.P[i, s_]; u = cp[i] - c; u = u / (np.linalg.norm(u) + 1e-9)
        _, inv_r, _ = X.M.RI.compute(r.q[i]); ar = float(u @ inv_r[s_] @ u); ah = float(u @ r.Hinv[h, sgid] @ u); mu = 1.0 / (ah + ar)
        if mode == "human":   # (shviz) human effective mass reference (kernel gain_part_h)
            eta[i] = float(np.clip(d0 + kg * max(np.sqrt((1.0 / max(ah, 1e-9)) / iso[sgid]) - 1.0, 0.0), d0, max(em, d0)))
        else:
            eta[i] = float(np.clip(d0 + kg * max(np.sqrt(mu * (1.0 / iso[sgid] + ar)) - 1.0, 0.0), d0, max(em, d0)))
    return dict(eta=eta, extra=eta - d0, d_base=d0, d_base_src=src, k_gain=kg, eta_max=em)


if __name__ == "__main__":
    o = load(sys.argv[1], sys.argv[2]); e = o["extra"]
    print("d_base %.3f (%s), k %.2f, eta_max %.2f; extra cm: median %.1f, max %.1f" % (o["d_base"], o["d_base_src"], o["k_gain"], o["eta_max"], 100 * np.nanmedian(e), 100 * np.nanmax(e)))
