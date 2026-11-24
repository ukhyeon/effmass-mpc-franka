"""window_v5.py - viz2 v5 (cmpview 2026-11-24, win2 11-24): automatic meeting window of a run = the robot TCP pass that holds the chosen moment.
One run (single mode): the moment = the run's CLOSEST meeting (smallest arm gap of exp_ana Run.gaps() over the whole run).
--uni <uniform replay> (comparison mode, the main run = the effective-mass replay; win2 11-24, win2b): candidates = effective-mass steps where
  the UNIFORM replay's arm gap (at the nearest replay time) is <= --uni-m (0.20 m); the effective-mass gap is not a condition (win2b: "both
  <= 0.20 m" dropped the strongest head-on scene, where the effective-mass robot keeps its distance); the moment = the candidate with the largest effective-
  mass eta extra (eta - d_base of the closest arm part, eta_v5.of_run: d_base, k, eta_max and the arm parts from the run's _exp.json), i.e.
  the strongest head-on meeting; not the largest gap difference (that can be a phase shift of the two robots' back-and-forth paths). No
  candidate: the effective-mass replay's closest meeting (as before), said in the reason line. Candidates lie --edge-s (0.5 s) inside the run
  (the last rendered frame ends before the last step; a moment at the run end has no frames after it).
Arm parts = the run's _exp.json human_arm_parts (exp_ana); with --base-exclude <yaml> human records with the pelvis on the robot cart
(goal1/scripts/rp5_base_exclude.py) are replaced by the last record that is not, before anything is measured.
Pass: exp_ana Run.passes over the whole run (TCP moving along the path direction, >= 0.3 s); the pass holding the moment, else the nearest
pass; under --min-s (1.0 s) -> --min-s centred on the moment (inside the run).
Prints "a,b,t" (window start, end, moment; s, run step times) on stdout and one reason line on stderr.
--arm-m M (wrap 11-24; run_v5 single mode 0.20): one run, the moment = the largest eta extra (eta_v5, the run's recorded rule) among steps
  with the arm gap <= M, >= --edge-s inside the run; none (or a uniform run, extra 0): the closest meeting, logged.
(robex 11-24) robot spheres of link0 / link1 (robex_v5.py, env V5_ROBOT_EXCLUDE) are left out of every gap here (choice and its part).
(sync 11-24) --uni: --sync 1 (default) keeps only steps where both robots move (TCP >= --sync-v 0.05 m/s, not within 3 cm of the leg goal; motion_v5.py)
  on the same pass (pass number = goal switches of the record, same direction) whose start (the later start of the two) has the two TCPs
  within --sync-d 0.05 m; none -> the first effective-mass pass where the uniform gap is <= --uni-m (largest extra inside it), logged.
(passgif 11-24) --pass-json FILE (with --uni; run_v5 comparison mode): the scene = ONE whole pass (motion_v5 legs, pass number = goal
  switches). Chosen pass: (1) same pass number and direction in both runs, the two TCPs within --sync-d (0.05 m) at the pass start (the
  later start of the two), (2) a step inside the pass span with the uniform gap <= --uni-m (0.20 m), (3) of those the largest effective-mass
  eta extra. None -> the first pass (leg 0, both start at home), logged. Pass span = the later start .. the later arrival (arrival = the goal
  switch of motion_v5, i.e. rp2's switch rule; a timeout / run end is named). PNG moment = the largest eta extra inside the chosen pass
  among the steps with the uniform gap <= --uni-m (>= --edge-s inside the run); none -> the largest extra in the pass, then its closest gap.
  (passgif2) Tiers: passes where both robots ARRIVED (goal switch on arrival, not timeout) first, then passes where the effective-mass
  robot arrived, then any; each tier keeps the conditions above. MP4 = the GIF span (the TCP divergence time, --div-m, is only recorded). The file holds
  the pass, both starts / arrivals (s, how), the GIF span, the PNG moment and the MP4 end; stdout "gif start,gif end,PNG moment"; one log line.
Usage: window_v5.py <run prefix> [--uni <uniform run prefix>] [--base-exclude yaml] [--min-s S] [--both-m M] [--pass-json FILE]"""
import argparse, os, sys
import numpy as np
E1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/real_prep5/exp1121"); sys.path.insert(0, E1); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_ana as X   # noqa: E402  (read only)
import robex_v5   # noqa: E402  (robex 11-24) robot spheres left out of the shown pair (link0, link1)
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--uni", default=""); ap.add_argument("--base-exclude", default="")
ap.add_argument("--min-s", type=float, default=1.0); ap.add_argument("--uni-m", type=float, default=0.20, help="(win2b) candidates: uniform arm gap <= this, m (the effective-mass gap is not a condition)")
ap.add_argument("--arm-m", type=float, default=0.0, help="(wrap) one run: candidates = steps with the arm gap <= this (m) -> largest eta extra (0 = the closest meeting)")
ap.add_argument("--sync", type=int, default=1, help="(sync 11-24) comparison: only steps where both robots move on the same pass (1 = on, 0 = as before)")
ap.add_argument("--sync-v", type=float, default=0.05, help="(sync) TCP speed of both robots >= this, m/s"); ap.add_argument("--sync-d", type=float, default=0.05, help="(sync) TCP distance of the two robots at the pass start <= this, m")
ap.add_argument("--edge-s", type=float, default=0.5, help="(win2) candidates at least this far from the run start / end (frames before and after the moment)")
ap.add_argument("--pass-json", default="", help="(passgif 11-24) comparison: whole-pass scene, written to this json (see the doc)")
ap.add_argument("--div-m", type=float, default=0.10, help="(passgif) MP4 end: the two TCPs apart more than this, m ...")
ap.add_argument("--div-after", type=float, default=1.0, help="(passgif) ... plus this, s")
A = ap.parse_args()
BX = None
if A.base_exclude:
    sys.path.insert(0, os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/scripts")); import rp5_base_exclude as RBE   # noqa: E402
    BX = RBE.BaseExclude(A.base_exclude)


def load(p):   # exp_ana Run with the base exclusion applied to its human record index (records on the cart -> the last one not on it)
    r = X.Run(p); robex_v5.apply(r)   # (robex 11-24) before any gap
    if r.H is None:
        raise SystemExit("window_v5: %s has no human record" % p)
    r.nbx = 0
    if BX is not None:
        ex = BX.inside(np.asarray(r.H["msg_com"], np.float64)[:, 0]); km = RBE.last_kept(ex); r.nbx = int(ex.sum())
        h = np.asarray(r.hidx, int); r.hidx = np.where(h >= 0, km[np.maximum(h, 0)], -1)
        if hasattr(r, "_g"):
            del r._g
    return r


def part_at(r, k):
    g, sp, sh, cp, _ = r.gaps(); h = int(r.hidx[k])
    return X.SEGN[int(r.Hseg[h][sh[k]])] if h >= 0 and sh[k] >= 0 else "?"


def window(r, k):
    ps, _ = r.passes((0, r.n - 1)); t = r.t
    if ps:
        inside = [p for p in ps if p[0] <= k <= p[1]]
        a, b = (inside[0] if inside else min(ps, key=lambda p: min(abs(t[p[0]] - t[k]), abs(t[p[1]] - t[k]))))[:2]
        how = "TCP pass holding it" if inside else "nearest TCP pass (the moment is between passes)"
    else:
        a = b = k; how = "no TCP pass: around the moment"
    ta, tb, tk = float(t[a]), float(t[b]), float(t[k])
    if tb - ta < A.min_s:   # (win2) shorter than min_s: min_s centred on the moment, inside the run
        ta = max(float(t[0]), tk - A.min_s / 2); tb = min(float(t[-1]), ta + A.min_s); ta = max(float(t[0]), tb - A.min_s); how += ", %.1f s centred" % A.min_s
    return ta, tb, how


re_ = load(A.run); ge = re_.gaps()[0] + X.RBUF   # shown gap (lines.npz d)
parts = "%s: %s" % ("exp.json" if re_.E.get("human_arm_parts") else "default", ",".join(re_.arm_parts))
bxs = ("%d records on the cart replaced" % re_.nbx) if BX is not None else "off"
bxs += "; " + re_.robex["text"]   # (robex 11-24) in every reason line
if not np.isfinite(ge).any():
    raise SystemExit("window_v5: no arm gap in %s" % A.run)
k = None
if A.uni and A.pass_json:   # (passgif 11-24) one whole pass: GIF span, PNG moment, MP4 end (see the doc)
    import json, eta_v5, motion_v5   # noqa: E402
    ru = load(A.uni); gu = ru.gaps()[0] + X.RBUF; te, tu = np.asarray(re_.t, float), np.asarray(ru.t, float)
    ku = np.array([int(np.argmin(np.abs(tu - x))) for x in te]); gun = gu[ku]
    ET = eta_v5.of_run(re_); ext = np.where(np.isfinite(ET["extra"]), ET["extra"], -np.inf)
    edge = (te >= te[0] + A.edge_s) & (te <= te[-1] - A.edge_s); cand = np.isfinite(ge) & np.isfinite(gun) & (gun <= A.uni_m) & edge
    ME, MU = motion_v5.state(re_), motion_v5.state(ru)

    def arrive(M, j):   # end of leg j: the goal switch step (first step of leg j + 1) -> (time s, how)
        if j + 1 < len(M["legs"]):
            return float(M["t"][M["legs"][j + 1]["k0"]]), M["legs"][j]["end"]
        return float(M["t"][-1]), "open (run end)"
    rows = []
    for j in range(min(len(ME["legs"]), len(MU["legs"]))):
        lE, lU = ME["legs"][j], MU["legs"][j]; ts = max(lE["t0"], lU["t0"])
        kE0 = int(np.argmin(np.abs(te - ts))); kU0 = int(np.argmin(np.abs(tu - ts))); dS = float(np.linalg.norm(ME["tcp"][kE0] - MU["tcp"][kU0]))
        aE, aU = arrive(ME, j), arrive(MU, j); tend = max(aE[0], aU[0]); inp = (te >= ts - 1e-9) & (te <= tend + 1e-9); c = cand & inp
        rows.append(dict(j=j, dirn=lE["dirn"], dirn_u=lU["dirn"], ts=ts, tsE=lE["t0"], tsU=lU["t0"], dS=dS, aE=aE, aU=aU, tend=tend, inp=inp, c=c, nc=int(c.sum()),
                         ex=float(ext[c].max()) if c.any() else -np.inf))
    ok0 = [r_ for r_ in rows if r_["dirn"] == r_["dirn_u"] and r_["dS"] <= A.sync_d and r_["nc"] > 0]   # (passgif2) tiers: both arrived > effmass arrived > any
    tiers = [("both robots arrived", [r_ for r_ in ok0 if r_["aE"][1] == "arrival" and r_["aU"][1] == "arrival"]),
             ("the effective-mass robot arrived (uniform: %s)", [r_ for r_ in ok0 if r_["aE"][1] == "arrival"]), ("any end (arrival / timeout)", ok0)]
    tn, ok = next(((n_, l_) for n_, l_ in tiers if l_), (None, []))
    if ok:
        P = max(ok, key=lambda r_: r_["ex"]); tn = tn % P["aU"][1] if "%s" in tn else tn
        why = "pass %d = largest effective-mass eta extra (%.1f cm) of %d passes with %s, the same number / direction, TCPs within %.0f cm at the start and a uniform gap <= %.2f m (fitting passes: %d, both arrived %d, effmass arrived %d)" % (
            P["j"], 100 * P["ex"], len(ok), tn, 100 * A.sync_d, A.uni_m, len(ok0), len(tiers[0][1]), len(tiers[1][1]))
    else:
        P = rows[0]; n1 = sum(r_["dirn"] == r_["dirn_u"] for r_ in rows); n2 = sum(r_["dirn"] == r_["dirn_u"] and r_["dS"] <= A.sync_d for r_ in rows)
        why = "no pass fits (of %d passes: same direction %d, + TCPs within %.0f cm at the start %d, + uniform gap <= %.2f m 0; start distances cm %s): first pass (both from home)" % (
            len(rows), n1, 100 * A.sync_d, n2, A.uni_m, " ".join("%d:%.1f" % (r_["j"], 100 * r_["dS"]) for r_ in rows))
    inpe = P["inp"] & edge
    if P["c"].any():
        k = int(np.argmax(np.where(P["c"], ext, -np.inf))); how = "largest eta extra among %d steps with the uniform gap <= %.2f m" % (P["nc"], A.uni_m)
    elif (inpe & np.isfinite(ext)).any() and np.where(inpe, ext, -np.inf).max() > -np.inf:
        k = int(np.argmax(np.where(inpe, ext, -np.inf))); how = "largest eta extra in the pass (no uniform gap <= %.2f m)" % A.uni_m
    else:
        k = int(np.nanargmin(np.where(P["inp"], ge, np.nan))); how = "closest effective-mass gap in the pass"
    dv = np.linalg.norm(np.asarray(re_.tcp, float) - np.asarray(ru.tcp, float)[ku], axis=1); kd = np.nonzero(dv > A.div_m)[0]
    tdiv = float(te[kd[0]]) if len(kd) else None; mp4e = P["tend"]   # (passgif2) MP4 = the same pass as the GIF (TCP divergence only recorded)
    out = dict(pass_no=P["j"], dirn=P["dirn"], start_s=round(P["ts"], 3), start=dict(effmass=round(P["tsE"], 3), uniform=round(P["tsU"], 3)), start_tcp_cm=round(100 * P["dS"], 1),
               arrive=dict(effmass=round(P["aE"][0], 3), uniform=round(P["aU"][0], 3)), arrive_how=dict(effmass=P["aE"][1], uniform=P["aU"][1]), gif=[round(P["ts"], 3), round(P["tend"], 3)],
               png_t=round(float(te[k]), 3), png_how=how, png_part=part_at(re_, k), png_extra_cm=round(100 * float(ET["extra"][k]), 2), png_gap_cm=dict(effmass=round(100 * float(ge[k]), 1), uniform=round(100 * float(gun[k]), 1)),
               tcp_div_t=tdiv, div_m=A.div_m, mp4_end=round(mp4e, 3), reason=why, uniform_run=ru.p, run=re_.p)
    json.dump(out, open(A.pass_json, "w"), indent=1)
    print("%.3f,%.3f,%.3f" % (P["ts"], P["tend"], float(te[k])))
    print("[v5] pass %d %s (%s); start effmass %.2f s, uniform %.2f s (TCPs %.1f cm apart at %.2f s); arrival effmass %.2f s (%s), uniform %.2f s (%s); GIF %.2f-%.2f s; "
          "PNG %.2f s (%s: %s, eta extra %.1f cm, gap effmass %.1f / uniform %.1f cm); MP4 = GIF span (TCPs > %.0f cm apart first at %s) (arm parts %s; base exclude %s)"
          % (P["j"], P["dirn"], why, P["tsE"], P["tsU"], 100 * P["dS"], P["ts"], P["aE"][0], P["aE"][1], P["aU"][0], P["aU"][1], P["ts"], P["tend"], float(te[k]), how, out["png_part"],
             out["png_extra_cm"], out["png_gap_cm"]["effmass"], out["png_gap_cm"]["uniform"], 100 * A.div_m, ("%.2f s" % tdiv) if tdiv is not None else "never", parts, bxs), file=sys.stderr)
    raise SystemExit(0)
if A.uni:   # (win2) both replays within --both-m at the same replay time -> largest effective-mass eta extra
    import eta_v5   # noqa: E402
    ru = load(A.uni); gu = ru.gaps()[0] + X.RBUF; te, tu = re_.t, ru.t
    ku = np.array([int(np.argmin(np.abs(tu - x))) for x in te]); gun = gu[ku]
    cand = np.isfinite(ge) & np.isfinite(gun) & (gun <= A.uni_m) & (te >= te[0] + A.edge_s) & (te <= te[-1] - A.edge_s)   # (win2b) uniform gap only; not at the run edges
    ET = eta_v5.of_run(re_); ex = np.where(cand & np.isfinite(ET["extra"]), ET["extra"], -np.inf)
    SY = ""   # (sync 11-24) both robots moving on the same pass: speed >= --sync-v, not in the final approach (3 cm), same pass number and direction,
    if A.sync:   # the two TCPs within --sync-d at that pass start (the later of the two starts); none -> the first pass with the uniform gap <= --uni-m
        import motion_v5   # noqa: E402
        ME, MU = motion_v5.state(re_), motion_v5.state(ru); vE, vU = ME["speed"], MU["speed"][ku]; lE, lU = ME["leg"], MU["leg"][ku]
        ts = np.maximum(te[ME["leg_k0"]], tu[MU["leg_k0"][ku]]); kE0 = np.searchsorted(te, ts).clip(0, len(te) - 1); kU0 = np.searchsorted(tu, ts).clip(0, len(tu) - 1)
        dS = np.linalg.norm(ME["tcp"][kE0] - MU["tcp"][kU0], axis=1)
        mv = (vE >= A.sync_v) & (vU >= A.sync_v) & ~ME["arriving"] & ~MU["arriving"][ku]; same = (lE == lU) & (ME["dirn"] == MU["dirn"][ku]); syn = mv & same & (dS <= A.sync_d)
        def st(k_):
            return "effmass v %.2f m/s pass %d %s, uniform v %.2f m/s pass %d %s, TCP distance at the pass start %.1f cm (%.2f s)" % (vE[k_], lE[k_], ME["dirn"][k_], vU[k_], lU[k_], MU["dirn"][ku[k_]], 100 * dS[k_], ts[k_])
        if (cand & syn).any():
            cand = cand & syn; SY = "both moving on the same pass (%d of the steps with the uniform gap <= %.2f m): " % (int(cand.sum()), A.uni_m)
        else:
            c0 = np.nonzero(cand)[0]
            if len(c0):
                nm_, ns_, nd_ = int((cand & mv).sum()), int((cand & mv & same).sum()), int((cand & mv & same & (dS <= A.sync_d)).sum()); n0 = len(c0); lg = int(lE[c0[0]]); cand = cand & (lE == lg)
                SY = ("no step with both robots moving on the same pass (of the %d steps with the uniform gap <= %.2f m: both moving %d, + same pass %d, + start within %.0f cm %d): "
                      "the first effmass pass with the uniform gap <= %.2f m (pass %d %s, from %.2f s): " % (n0, A.uni_m, nm_, ns_, 100 * A.sync_d, nd_, A.uni_m, lg, ME["legs"][lg]["dirn"], te[c0[0]]))
        ET = eta_v5.of_run(re_); ex = np.where(cand & np.isfinite(ET["extra"]), ET["extra"], -np.inf)
    if np.isfinite(ex).any() and ex.max() > -np.inf:
        k = int(np.argmax(ex)); ta, tb, how = window(re_, k)
        print("%.2f,%.2f,%.3f" % (ta, tb, float(re_.t[k])))
        print("[v5] " + SY + ("state at the moment: " + st(k) + "\n[v5] " if A.sync else "") + "window %.2f-%.2f s = %s: moment %.2f s = largest effective-mass eta extra among %d steps with the uniform gap <= %.2f m (>= 0.5 s inside the run): %s, eta extra %.1f cm "
              "(d_base %.3f %s), gap effmass %.1f cm, uniform %.1f cm (at %.2f s), difference %+.1f cm (arm parts %s; base exclude %s)"
              % (ta, tb, how, float(re_.t[k]), int(cand.sum()), A.uni_m, part_at(re_, k), 100 * float(ET["extra"][k]), ET["d_base"], ET["d_base_src"],
                 100 * float(ge[k]), 100 * float(gun[k]), float(tu[ku[k]]), 100 * float(ge[k] - gun[k]), parts, bxs), file=sys.stderr)
    else:
        print("[v5] no step with the uniform arm gap <= %.2f m: back to the effective-mass closest meeting" % A.uni_m, file=sys.stderr)
if k is None and A.arm_m > 0 and not A.uni:   # (wrap 11-24) single mode: arm within --arm-m, largest eta extra, not at the run edges
    import eta_v5   # noqa: E402
    te = re_.t; cand = np.isfinite(ge) & (ge <= A.arm_m) & (te >= te[0] + A.edge_s) & (te <= te[-1] - A.edge_s)
    ET = eta_v5.of_run(re_); ex = np.where(cand & np.isfinite(ET["extra"]), ET["extra"], -np.inf)
    if cand.any() and ex.max() > 0:
        k = int(np.argmax(ex)); ta, tb, how = window(re_, k)
        print("%.2f,%.2f,%.3f" % (ta, tb, float(te[k])))
        print("[v5] window %.2f-%.2f s = %s: moment %.2f s = largest eta extra among %d steps with the arm gap <= %.2f m (>= %.1f s inside the run): %s, eta extra %.1f cm "
              "(d_base %.3f %s), gap %.1f cm (arm parts %s; base exclude %s)" % (ta, tb, how, float(te[k]), int(cand.sum()), A.arm_m, A.edge_s, part_at(re_, k),
              100 * float(ET["extra"][k]), ET["d_base"], ET["d_base_src"], 100 * float(ge[k]), parts, bxs), file=sys.stderr)
    else:
        print("[v5] no step with the arm gap <= %.2f m and a positive eta extra (%d steps within; uniform run or none): back to the closest meeting" % (A.arm_m, int(cand.sum())), file=sys.stderr)
if k is None:
    k = int(np.nanargmin(ge)); ta, tb, how = window(re_, k)
    print("%.2f,%.2f,%.3f" % (ta, tb, float(re_.t[k])))
    print("[v5] window %.2f-%.2f s = %s: closest meeting of %s at %.2f s, %s, %.1f cm (arm parts %s; base exclude %s)"
          % (ta, tb, how, re_.name, float(re_.t[k]), part_at(re_, k), 100 * float(ge[k]), parts, bxs), file=sys.stderr)
