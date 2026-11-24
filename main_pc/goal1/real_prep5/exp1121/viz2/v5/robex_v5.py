"""robex_v5.py - viz2 v5 (robex 2026-11-24): robot spheres left out of the SHOWN closest pair (scene choice, part / m_h / extra, gap
and line of the panels). The robot base (link0) and link 1 hardly move, so a person cannot be avoided by them; a hand meeting them
(t18 comparison: m_h about 5 kg, extra 19.1 cm) is not a scene of the method. The MPC cost and exp_ana's own distances are unchanged:
only the viz copy of the robot sphere centres (exp_ana Run.P) has these spheres moved 1 km away before Run.gaps() is called.
Spheres = the MPC robot sphere model (exp_ana M.RI.sphere_link, RobotInertiaNP, 61 spheres) whose link name ends with "_<link>" for a
link of the list: default "link0,link1" (fr3v2_1_link0 2 spheres, fr3v2_1_link1 4 spheres); env V5_ROBOT_EXCLUDE (run_v5.sh
--robot-exclude) = another comma list, "none" = off (all spheres, as before).
Library: links() -> list; ids(links) -> (sphere indices, text); apply(Run) -> number left out (clears the Run's gap cache); key() -> cache
key text; lines(Run) -> lines.npz arrays (as make_inputs).
CLI: robex_v5.py lines <run prefix> <work dir> -> rewrites <work>/lines.npz (a, b, d, t, ang, seg, robot_exclude) and the closest step of
<work>/meta.json (inside its window) with the spheres left out (run_v5 --compose-only: the inputs of the last render, no Isaac).
(robex2 11-24) The links follow the run's record when it has them: _exp.json human_robot_exclude (list, or the dict real_run copies from
the MPC summary), else _summary.json human.robot_exclude (rp2 --human-robot-exclude); a missing / null key -> V5_ROBOT_EXCLUDE as before.
links(src) / key(src) take a Run or a run prefix for that; apply(r) stores r.robex (links, n, text, source, ids).
CLI: robex_v5.py ids <run prefix> -> stdout the left-out sphere indices "i,j,..." or "-" (render_v5.sh: env V5_ROBEX_IDX of g1_viz_v5,
eta volumes and line), stderr the text line."""
import json, os, sys
import numpy as np
E1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121")
DEFAULT = "link0,link1"
FAR = 1.0e3   # m: an excluded sphere centre is put this far away (never the closest)


def rec_links(src):   # (robex2) the links the MPC left out, from the run's record; None = not recorded (key missing or null)
    p = getattr(src, "p", src)
    for f, get in ((p + "_exp.json", lambda d: d.get("human_robot_exclude")), (p + "_summary.json", lambda d: (d.get("human") or {}).get("robot_exclude"))):
        try:
            v = get(json.load(open(f)))
        except Exception:
            continue
        if isinstance(v, dict):
            v = v.get("links")
        if isinstance(v, str):
            v = v.split(",")
        if v is None:
            continue
        return [str(x).strip() for x in v if str(x).strip() and str(x).strip().lower() not in ("none", "off", "0")]
    return None


def links(src=None):   # (robex2) src = Run or run prefix: its recorded links win; else env V5_ROBOT_EXCLUDE (run_v5 --robot-exclude)
    rl = rec_links(src) if src is not None else None
    if rl is not None:
        return rl
    v = os.environ.get("V5_ROBOT_EXCLUDE", DEFAULT).strip()
    return [] if v.lower() in ("", "none", "off", "0") else [x.strip() for x in v.split(",") if x.strip()]


def key(src=None):
    if src is not None and hasattr(src, "robex"):
        return src.robex["links"]
    return ",".join(links(src)) or "none"


def ids(lk=None):
    sys.path.insert(0, E1)
    import exp_ana as X   # read only
    lk = links() if lk is None else lk; L = [str(x) for x in np.asarray(X.M.RI.sphere_link)]
    ix = [i for i, n in enumerate(L) if any(n == l or n.endswith("_" + l) for l in lk)]
    cnt = {}
    for i in ix:
        cnt[L[i]] = cnt.get(L[i], 0) + 1
    txt = ("%d of %d robot spheres left out of the shown pair (%s)" % (len(ix), len(L), ", ".join("%s x%d" % kv for kv in cnt.items()))) if ix else \
        "no robot sphere left out of the shown pair (all %d)" % len(L)
    return ix, txt


def apply(r):
    rl = rec_links(r); lk = links(r); ix, txt = ids(lk); src = "record" if rl is not None else "run_v5"
    if rl is not None:
        txt += " [links from the run's record]"
    r.robex = dict(links=",".join(lk) or "none", n=len(ix), text=txt, source=src, ids=list(ix))
    if ix:
        r.P = np.array(r.P, float, copy=True); r.P[:, ix] = FAR
        if hasattr(r, "_g"):
            del r._g
    return len(ix)


def lines(r):   # make_inputs lines.npz arrays from the Run (its spheres already left out by apply)
    sys.path.insert(0, E1)
    import exp_ana as X   # read only
    n = r.n; hidx = np.asarray(r.hidx, int); g, sp, sh, cp, _ = r.gaps(); a = np.full((n, 3), np.nan); b = np.full((n, 3), np.nan); d = g + X.RBUF
    for i in range(n):
        if sp[i] < 0 or hidx[i] < 0:
            continue
        c = r.P[i][sp[i]]; u = cp[i] - c; L = np.linalg.norm(u); u = u / max(L, 1e-9)
        a[i] = c + u * X.M.RAD[sp[i]]; b[i] = cp[i] - u * r.Hrad[hidx[i]][sh[i]]
    ang = np.full(n, np.nan); segc = np.full(n, -1)   # angle of (robot point - arm point) to the forearm axis of that arm side, folded to 0-90 deg
    for i in range(n):
        if sp[i] < 0 or hidx[i] < 0 or not np.all(np.isfinite(a[i])):
            continue
        h = hidx[i]; sg_ = int(r.Hseg[h][sh[i]]); segc[i] = sg_; fa = 15 if sg_ in (13, 15, 17) else 16; kf = np.nonzero((r.Hseg[h] == fa) & (r.Hrad[h] > 0))[0]
        if len(kf):
            ax = r.Hp1[h][kf[0]] - r.Hp0[h][kf[0]]; v = a[i] - b[i]
            ang[i] = np.degrees(np.arccos(min(1.0, abs(float(ax @ v)) / (np.linalg.norm(ax) * np.linalg.norm(v) + 1e-12))))
    return dict(a=a, b=b, d=d, t=r.t, ang=ang, seg=segc, robot_exclude=np.array(key(r)))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "ids":   # (robex2) for render_v5.sh: the sphere indices left out of the eta volumes and the line
        lk = links(sys.argv[2]); ix, txt = ids(lk); src = "the run's record" if rec_links(sys.argv[2]) is not None else "run_v5 --robot-exclude"
        sys.stderr.write("%s (links %s from %s)\n" % (txt, ",".join(lk) or "none", src)); print(",".join(str(i) for i in ix) or "-"); raise SystemExit(0)
    if len(sys.argv) != 4 or sys.argv[1] != "lines":
        raise SystemExit("usage: robex_v5.py lines <run prefix> <work dir> | robex_v5.py ids <run prefix>")
    run, work = sys.argv[2], sys.argv[3]; f = os.path.join(work, "lines.npz")
    sys.path.insert(0, E1)
    import exp_ana as X   # read only
    r = X.Run(run); apply(r); o = lines(r); np.savez(f, **o)
    mf = os.path.join(work, "meta.json"); m = json.load(open(mf)); wa, wb = m["window"]; t = o["t"]; d = o["d"]; w = (t >= wa) & (t <= wb)
    if w.any() and np.isfinite(d[w]).any():
        k = int(np.nonzero(w)[0][np.nanargmin(d[w])]); m.update(closest_m=float(d[k]), t_closest=float(t[k]), k_closest=k)
    m["robot_exclude"] = dict(links=key(r), text=r.robex["text"]); json.dump(m, open(mf, "w"), indent=1)
    ch = int(np.sum(np.isfinite(d))); print("[v5 robex] %s: %s; lines.npz rewritten (%d steps with a pair), closest in window %.1f cm at %.2f s"
                                            % (os.path.basename(run), r.robex["text"], ch, 100 * m["closest_m"], m["t_closest"]))
