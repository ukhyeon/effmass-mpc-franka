"""human_shoulder.py - (shoulder 1124) rp2_mpc_rt.py --human-shoulder 1: shoulder spheres and ISO reference override.
ShoulderSpheres(shapes_json, iso_json).apply(r, hoff) writes, in place, one sphere per entry of goal1/data/shapes_v2_shoulder.json into
a relay record r (rp5_human_relay.py layout, offsets hoff = rp2 HOFF): slot k gets p0 = p1 = com[segment] (the message's segment centre
of mass, node world frame), radius, segment, type 1. The slot must be inactive (radius 0) or already hold the same segment's sphere
(a record written by an earlier --human-shoulder 1 run, e.g. its _human.npz replayed: rewritten from com, counted in .rewritten);
another shape there -> ValueError.
.iso = {segment name: m_h_iso kg} from goal1/data/iso_ref_override.json (for g1_meff_gpu.MeffChannel.set_parts).
Not used without the option (rp2 then never imports this file)."""
import json, os
import numpy as np
G1 = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1")
SHAPES_JSON = os.path.join(G1, "data", "shapes_v2_shoulder.json")
ISO_JSON = os.path.join(G1, "data", "iso_ref_override.json")


class ShoulderSpheres:
    def __init__(self, shapes_json=SHAPES_JSON, iso_json=ISO_JSON):
        self.shapes_json = os.path.realpath(os.path.expanduser(shapes_json or SHAPES_JSON)); self.iso_json = os.path.realpath(os.path.expanduser(iso_json or ISO_JSON))
        S = json.load(open(self.shapes_json)); I = json.load(open(self.iso_json))
        self.sph = [(int(e["slot"]), int(e["segment"]), float(e["radius"]), str(e["name"])) for e in S["spheres"]]
        self.iso = {str(k): float(v) for k, v in I["m_h_iso"].items()}
        self.version = (S.get("version", ""), I.get("version", "")); self.n = 0; self.rewritten = 0

    def apply(self, r, hoff):
        com = r[hoff["com"][0]:hoff["com"][0] + 57].reshape(19, 3)
        for slot, seg, rad, name in self.sph:
            if r[hoff["rad"][0] + slot] > 0.0 and int(round(r[hoff["seg"][0] + slot])) == seg:   # (slotfix 1124) same shoulder sphere already there
                self.rewritten += 1
            elif r[hoff["rad"][0] + slot] > 0.0:
                raise ValueError("human record slot %d is not free (radius %.3f): the shoulder sphere %s needs an inactive slot" % (slot, r[hoff["rad"][0] + slot], name))
            c = com[seg].copy()
            r[hoff["p0"][0] + 3 * slot:hoff["p0"][0] + 3 * slot + 3] = c; r[hoff["p1"][0] + 3 * slot:hoff["p1"][0] + 3 * slot + 3] = c
            r[hoff["rad"][0] + slot] = rad; r[hoff["seg"][0] + slot] = seg; r[hoff["type"][0] + slot] = 1.0
        self.n += 1
        return r

    def describe(self):
        return dict(shapes_json=self.shapes_json, iso_json=self.iso_json, spheres=[dict(name=n, segment=g, slot=k, radius=r) for k, g, r, n in self.sph],
                    m_h_iso=self.iso, version=list(self.version), records=self.n, rewritten=self.rewritten)
