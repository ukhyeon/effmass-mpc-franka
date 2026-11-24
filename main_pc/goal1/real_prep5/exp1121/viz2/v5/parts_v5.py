"""parts_v5.py - viz2 v5 (win2 2026-11-24): the effective-mass arm parts of a run, from its record (same order as exp_ana):
<run>_exp.json human_arm_parts, else <run>_summary.json human.part_cost.human_arm_parts, else the 6 arm parts (old records).
arm_parts(run prefix) -> (names, segment ids, source). drawable(ids) -> (ids the v5 eta volumes can draw = 10, 11 (shoulder spheres, shviz 11-24), 13..18, skipped ids): other parts are skipped
with one warning line. mass_rule(run) -> recorded human_mass_mode and human_iso_override (shviz 11-24).
CLI: parts_v5.py <run prefix> -> prints "13,14,..." (drawable ids; warning on stderr)."""
import json, os, sys

SEGN = ["pelvis", "left_thigh", "right_thigh", "abdomen", "left_calf", "right_calf", "left_foot", "right_foot", "chest", "neck", "left_shoulder",
        "right_shoulder", "head", "left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"]
DEFAULT = ["left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"]
DRAWABLE = (10, 11, 13, 14, 15, 16, 17, 18)   # (shviz 11-24) + shoulders (record shape slots 21, 22: spheres)


def arm_parts(run):
    for f, get in ((run + "_exp.json", lambda d: d.get("human_arm_parts")),
                   (run + "_summary.json", lambda d: ((d.get("human") or {}).get("part_cost") or {}).get("human_arm_parts"))):
        try:
            v = get(json.load(open(f)))
        except (OSError, ValueError):
            v = None
        if v:
            names = [str(x) for x in v]; return names, [SEGN.index(x) for x in names if x in SEGN], os.path.basename(f)
    return list(DEFAULT), [SEGN.index(x) for x in DEFAULT], "default (no human_arm_parts in the record)"


def mass_rule(run):   # (shviz 11-24) recorded eta rule: (human_mass_mode, {segment: ISO kg override}, source); old records: ("reduced", {}, ...)
    for f, get in ((run + "_exp.json", lambda d: d), (run + "_summary.json", lambda d: (d.get("human") or {}).get("part_cost") or {})):
        try:
            d = get(json.load(open(f)))
        except (OSError, ValueError):
            continue
        if d.get("human_mass_mode") or d.get("human_iso_override"):
            io = d.get("human_iso_override") or {}; io = json.loads(io) if isinstance(io, str) else dict(io)
            return str(d.get("human_mass_mode") or "reduced"), {str(k): float(v) for k, v in io.items()}, os.path.basename(f)
    return "reduced", {}, "default (no human_mass_mode in the record)"


def drawable(ids, run="", warn=True):
    ok = [i for i in ids if i in DRAWABLE]; skip = [i for i in ids if i not in DRAWABLE]
    if skip and warn:
        print("[v5] WARNING: %s arm parts %s have no eta volume in the v5 render yet (drawn: %s)" % (os.path.basename(run) or "run", ",".join(SEGN[i] for i in skip),
              ",".join(SEGN[i] for i in ok)), file=sys.stderr)
    return ok, skip


if __name__ == "__main__":
    if sys.argv[1] == "--rule":   # (shviz) "mode override" for render_v5.sh: "reduced -" (old records) / "human left_shoulder:3,right_shoulder:3"
        mo, io, _ = mass_rule(sys.argv[2]); print(mo, ",".join("%s:%g" % kv for kv in sorted(io.items())) or "-")
    else:
        n, ids, src = arm_parts(sys.argv[1]); ok, _ = drawable(ids, sys.argv[1]); print(",".join(str(i) for i in ok))
