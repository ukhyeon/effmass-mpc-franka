"""(viz2 v5 copy of goal1/scripts/g1_viz.py, 2026-11-23; loaded by launch_v5.py instead of g1_viz. Changes marked (v5): eta shells on the
arm parts only (upper arm, lower arm, hand; the hand = its largest capsule, no thumb / finger), one shell per part in the method colour
(env V5_RGB), and a line robot point -> arm point per step from env V5_LINES (npz a, b (n,3), base frame).
(robex2 11-24) env V5_ROBEX_IDX "i,j,..." (render_v5.sh, robex_v5.py ids): robot spheres never taken as the nearest one of an eta volume;
unset or "-" = all spheres (as before). The line (V5_LINES) is already made without them (make_inputs / robex_v5 lines)."""
"""g1_viz.py - display-only helpers for step2b_mpc_isaac.py (demo). Nothing here feeds the MPC: it reads the current robot spheres and the
human inputs after the solver step.
  eta shells (--viz-eta-shell on): one translucent capsule per v2 hand shape, radius = shape radius + eta of the robot sphere closest to that shape,
      eta by the run's cost mode (mass: set A of mu; iso: 0.6 kg hand; uniform: eta0; off: hidden). Colour by mu (blue 0.5 kg -> red 3.0 kg).
  trail (--viz-trail on): TCP path of the last --viz-trail-sec seconds (TCP = hand link - 0.1034 m in z, as in the analyses), condition colour.
  ghost (--viz-ghost cond,cond --viz-ghost-tag <tag with {cond}>): saved TCP paths of other runs (B1 rows) as grey curves.
  record (--viz-record DIR): cameras (--viz-cam eye,target[;eye,target]) frames every step through replicator rgb annotators (DIR/cam<j>/), DIR_cam<j>.mp4 at the end.
Timing of the per-step update is kept (ms) and written to <DIR or log dir>/<tag>_viz.json."""
import json, os, time, subprocess
from collections import deque
import numpy as np

COND_RGB = {0: (0.55, 0.55, 0.55), 1: (0.93, 0.63, 0.0), 2: (0.92, 0.41, 0.20), 3: (0.16, 0.47, 0.84)}   # off, uniform, mass, iso


def capsule_mesh(p0, p1, rad, na=36, nt=6, nc=8):   # (v5) capsule surface points, unit normals, triangles (as viz2/v4 render_v4.py)
    ax = p1 - p0; L = np.linalg.norm(ax); w = ax / L if L > 1e-6 else np.array([0, 0, 1.0])
    a = np.cross(w, [1.0, 0, 0]); a = a if np.linalg.norm(a) > 0.1 else np.cross(w, [0, 1.0, 0]); a /= np.linalg.norm(a); b = np.cross(w, a)
    rings = [(p0, -np.sin(np.pi / 2 * k / nc), np.cos(np.pi / 2 * k / nc)) for k in range(nc, 0, -1)] + [(p0 + t * ax, 0.0, 1.0) for t in np.linspace(0, 1, nt)]
    rings += [(p1, np.sin(np.pi / 2 * k / nc), np.cos(np.pi / 2 * k / nc)) for k in range(1, nc + 1)]
    ang = np.linspace(0, 2 * np.pi, na, endpoint=False); P, N = [], []
    for c, sw, cr in rings:
        n = sw * w[None] + cr * (np.cos(ang)[:, None] * a[None] + np.sin(ang)[:, None] * b[None]); n /= np.linalg.norm(n, axis=1, keepdims=True); P.append(c + rad * n); N.append(n)
    nr = len(rings); i, j = np.meshgrid(np.arange(nr - 1), np.arange(na), indexing="ij"); i, j = i.ravel(), j.ravel()
    q0, q1, q2, q3 = i * na + j, i * na + (j + 1) % na, (i + 1) * na + (j + 1) % na, (i + 1) * na + j
    return np.vstack(P), np.vstack(N), np.vstack([np.stack([q0, q2, q1], 1), np.stack([q0, q3, q2], 1)])


def silhouette(P, F, eye):   # (v5 fix3) contour of the view-facing sign over the triangle mesh P, F seen from eye: segments (n, 2, 3)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]); vn = np.zeros_like(P)
    for j in range(3):
        np.add.at(vn, F[:, j], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12; vd = P - eye; f = (vn * vd).sum(1) / (np.linalg.norm(vd, axis=1) + 1e-12)
    pts = []; cnt = np.zeros(len(F), int); seg = np.zeros((len(F), 2, 3))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        fa, fb = f[F[:, a]], f[F[:, b]]; m = (fa > 0) != (fb > 0); t = fa / np.where(m, fa - fb, 1.0)
        x = P[F[:, a]] + t[:, None] * (P[F[:, b]] - P[F[:, a]]); k = np.nonzero(m & (cnt < 2))[0]; seg[k, cnt[k]] = x[k]; cnt[k] += 1
    return seg[cnt == 2]


def mu_rgb(mu, lo=0.5, hi=3.0):
    t = float(np.clip((mu - lo) / (hi - lo), 0, 1)); return (0.16 + 0.76 * t, 0.47 - 0.22 * t, 0.84 - 0.70 * t)


class Viz:
    def __init__(self, stage, args, meff, human, tag, log, n_spheres):
        from pxr import Gf, UsdGeom, UsdShade, Sdf, Vt
        self.Gf, self.UsdGeom, self.UsdShade, self.Sdf, self.Vt = Gf, UsdGeom, UsdShade, Sdf, Vt
        self.stage, self.args, self.meff, self.human, self.tag, self.log, self.S = stage, args, meff, human, tag, log, n_spheres
        self.t_upd = []; self.t_cap = []; self.shells = []; self.trail = None; self.rec = None
        mode = int(meff.mode[0]) if meff is not None else 0; self.mode = mode
        from pxr import UsdLux
        if not stage.GetPrimAtPath("/World/viz/dome").IsValid():   # display light (the run scene has only the default light)
            UsdLux.DomeLight.Define(stage, "/World/viz/dome").CreateIntensityAttr(700)
        if getattr(args, "viz_zone", ""):   # (V1) static zone surface
            _bm = "uniform" not in os.path.basename(args.viz_zone)
            if getattr(args, "viz_zone_style", "cage") == "cage":
                log(f"(viz) zone {add_zone_cage(stage, os.path.expanduser(args.viz_zone), by_mu=_bm)}")
            else:
                log(f"(viz) zone {add_zone(stage, os.path.expanduser(args.viz_zone), opacity=getattr(args, 'viz_zone_opacity', 0.35), by_mu=_bm)}")
        self.zseq = None
        if getattr(args, "viz_zone_seq", ""):   # (R1) zone cages along the retract motion; the one closest to the current frame's retract distance is shown
            zs = json.load(open(os.path.expanduser(args.viz_zone_seq))); cond = args.viz_zone_seq_cond; hz = np.load(zs["human"] if "human" in zs else self.human.pre_path)["s_retract"]; prims = []   # retract distance per frame of the human file in use
            for i, fn in enumerate(zs["files"][cond]):
                add_zone_cage(stage, fn, path=f"/World/viz/zoneseq/c{i:02d}", by_mu=(cond != "uniform")); pr = stage.GetPrimAtPath(f"/World/viz/zoneseq/c{i:02d}")
                UsdGeom.Imageable(pr).MakeInvisible(); prims.append(pr)
            self.zseq = dict(S=np.array(zs["s"]), prims=prims, hs=hz, cur=-1); log(f"(viz) zone sequence {cond}: {len(prims)} cages")
        for hp in [p for p in os.environ.get("V5_HIDE", "").split(",") if p and p != "-"]:   # (v5 matched) render only: hide these prims (e.g. a virtual keep-out box)
            pr = stage.GetPrimAtPath(hp)
            if pr.IsValid():
                UsdGeom.Imageable(pr).MakeInvisible(); log(f"(v5) hidden in the render: {hp}")
            else:
                log(f"(v5) no prim {hp} to hide")
        _ob = stage.GetPrimAtPath("/World/obstacles")   # (perpbox 11-24) one line: the scene objects drawn / hidden in this render
        if _ob.IsValid():
            _ch = list(_ob.GetChildren()); _vis = [c.GetName() for c in _ch if UsdGeom.Imageable(c).ComputeVisibility() != UsdGeom.Tokens.invisible]
            log(f"(v5) scene objects drawn: {', '.join(_vis) or 'none'}; hidden: {', '.join(c.GetName() for c in _ch if c.GetName() not in _vis) or 'none'}")
        if getattr(args, "viz_hide_com", "off") == "on":   # hide the human segment COM markers (PointInstancer of the human link)
            pr = stage.GetPrimAtPath("/World/human/instancer")
            if pr.IsValid():
                UsdGeom.Imageable(pr).MakeInvisible()
        if args.viz_eta_shell in ("on", "arm", "armelbow") and meff is not None:
            # (live prep2) "arm": shells on every active shape of both forearms and hands (segments 15-18), "on": right hand only
            # (live cmp) "armelbow": "arm" plus a sphere shell at each elbow (forearm capsule end p0, forearm radius + eta of the forearm)
            segs = range(15, 19) if args.viz_eta_shell in ("arm", "armelbow") else (18,)
            h = human.last_h; self.hand = [k for k in range(len(h["rad"])) if int(h["seg"][k]) in segs and float(h["rad"][k]) > 0]
            if os.environ.get("V5_RGB"):   # (v5) arm parts, one capsule per part: upper / lower arm as recorded, hand = largest hand capsule
                self.hand = []   # (win2 11-24) parts = V5_ARM_PARTS (render_v5.sh: the record's human_arm_parts, drawable ids 13-18; "none" = no part); unset = 13-18
                _ap = os.environ.get("V5_ARM_PARTS", "")
                _segs = list(range(13, 19)) if not _ap else ([] if _ap == "none" else [int(x) for x in _ap.split(",") if x.strip()])
                _ok5 = (10, 11) + tuple(range(13, 19))   # (shviz 11-24) shoulders 10, 11 = the record's sphere slots (zero-length capsules)
                _bad = [x for x in _segs if x not in _ok5]
                if _bad:
                    log(f"(v5) WARNING: eta volume parts {_bad} cannot be drawn (10, 11, 13-18 only): skipped"); _segs = [x for x in _segs if x in _ok5]
                for sg in _segs:
                    ks = [k for k in range(len(h["rad"])) if int(h["seg"][k]) == sg and float(h["rad"][k]) > 0]
                    if ks:
                        self.hand += ks if sg < 17 else [max(ks, key=lambda k: float(h["rad"][k]))]
                log(f"(v5) eta volume parts {_segs} (V5_ARM_PARTS {_ap or 'unset'}): {len(self.hand)} capsules {self.hand}")
                args.viz_eta_shell = "arm"
            self.elbow = set()
            if args.viz_eta_shell == "armelbow":
                fk = [k for k in self.hand if int(h["seg"][k]) in (15, 16)]; self.elbow = set(range(100, 100 + len(fk))); self.elbow_src = dict(zip(sorted(self.elbow), fk))
            self.v5_mesh = {}; self.v5_edge = {}; self.v5_eye = np.array([float(x) for x in os.environ["V5_EYE"].split(",")[:3]]) if os.environ.get("V5_EYE") else None
            if os.environ.get("V5_RGB"):   # (v5) direction-dependent shells: capsule surface + normal x eta(normal) (eta rule of the MPC gain_part cost)
                rgb5 = tuple(float(x) for x in os.environ["V5_RGB"].split(","))
                for k in self.hand:
                    _P, _N, _F = capsule_mesh(np.asarray(h["p0"][k], float), np.asarray(h["p1"][k], float), float(h["rad"][k]))
                    g = UsdGeom.Mesh.Define(stage, f"/World/viz/v5_shell_{k:02d}"); g.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(_F)))
                    g.CreateFaceVertexIndicesAttr(Vt.IntArray(_F.reshape(-1).tolist())); g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(_P, np.float32)))
                    g.CreateSubdivisionSchemeAttr("none"); g.CreateDoubleSidedAttr(True); g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(_N, np.float32))); g.SetNormalsInterpolation("vertex")
                    m = UsdShade.Material.Define(stage, f"/World/viz/Looks/v5shell_{k:02d}"); sh = UsdShade.Shader.Define(stage, f"/World/viz/Looks/v5shell_{k:02d}/s"); sh.CreateIdAttr("UsdPreviewSurface")
                    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb5)); sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(os.environ.get("V5_OPACITY", "0.22")))
                    em5 = float(os.environ.get("V5_EMIS", "0.25"))   # (v5 fix3) emission factor: the colour stays saturated under the bright dome light
                    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85); sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1); sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0)); sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(os.environ.get("V5_IOR", "1.0"))); sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[em5 * x for x in rgb5])); m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                    UsdGeom.PrimvarsAPI(g.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True); UsdShade.MaterialBindingAPI(g.GetPrim()).Bind(m)
                    self.v5_mesh[k] = g
                    if os.environ.get("V5_EYE"):   # (v5 fix3) dark outline of the shell as seen from the recording camera
                        ce = tuple(float(x) for x in os.environ.get("V5_EDGE_RGB", ",".join(str(0.55 * x) for x in rgb5)).split(","))
                        c5 = UsdGeom.BasisCurves.Define(stage, f"/World/viz/v5_edge_{k:02d}"); c5.CreateTypeAttr("linear"); c5.CreateWidthsAttr(Vt.FloatArray([float(os.environ.get("V5_EDGE_W", "0.004"))]))
                        c5.SetWidthsInterpolation("constant"); c5.CreateCurveVertexCountsAttr(Vt.IntArray([2])); c5.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, -5), Gf.Vec3f(0, 0, -5.001)]))
                        c5.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*ce)])); UsdShade.MaterialBindingAPI(c5.GetPrim()).Bind(self._mat(f"v5edge_{k:02d}", ce))
                        UsdGeom.PrimvarsAPI(c5.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True); self.v5_edge[k] = c5
                self.hand = []
            for k in self.hand + sorted(self.elbow):
                g = UsdGeom.Capsule.Define(stage, f"/World/viz/eta_shell_{k:02d}"); g.CreateAxisAttr("Z"); x = UsdGeom.Xformable(g)
                m = UsdShade.Material.Define(stage, f"/World/viz/Looks/shell_{k:02d}"); sh = UsdShade.Shader.Define(stage, f"/World/viz/Looks/shell_{k:02d}/s"); sh.CreateIdAttr("UsdPreviewSurface")
                col = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f); col.Set(Gf.Vec3f(0.5, 0.5, 0.5)); sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(os.environ.get("V5_OPACITY", "0.22")))
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3); m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                UsdGeom.PrimvarsAPI(g.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
                UsdShade.MaterialBindingAPI(g.GetPrim()).Bind(m); g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.5, 0.5, 0.5)])); g.CreateDisplayOpacityAttr(Vt.FloatArray([0.28]))
                self.shells.append((k, g, x.AddTranslateOp(), x.AddOrientOp(), col, g.GetDisplayColorAttr()))
            if mode == 0:
                for _, g, *_r in self.shells:
                    UsdGeom.Imageable(g).MakeInvisible()
        self.v5_rgb = tuple(float(x) for x in os.environ["V5_RGB"].split(",")) if os.environ.get("V5_RGB") else None   # (v5)
        _rx5 = os.environ.get("V5_ROBEX_IDX", "")   # (robex2 11-24) robot spheres left out of the eta volumes (render_v5.sh: robex_v5.py ids); unset / "-" = all (as before)
        self.robex_idx = np.array([int(x) for x in _rx5.split(",")], int) if _rx5 and _rx5 != "-" else None
        if self.robex_idx is not None:
            log(f"(v5) robot spheres left out of the eta volumes: {len(self.robex_idx)} of {n_spheres} {self.robex_idx.tolist()}")
        self.v5_lines = None
        if os.environ.get("V5_LINES"):   # (v5) closest robot point -> arm point, per step
            zl = np.load(os.environ["V5_LINES"]); self.v5_lines = (zl["a"], zl["b"])
            c = UsdGeom.BasisCurves.Define(stage, "/World/viz/v5_line"); c.CreateTypeAttr("linear"); c.CreateWidthsAttr(Vt.FloatArray([0.010])); c.SetWidthsInterpolation("constant")
            c.CreateCurveVertexCountsAttr(Vt.IntArray([2])); c.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, -5), Gf.Vec3f(0, 0, -5.001)]))
            lc = (0.08, 0.08, 0.08); c.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*lc)])); UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(self._mat("v5_line", lc)); self.v5_curve = c
            self.v5_ends = []
            for j in range(2):
                sp_ = UsdGeom.Sphere.Define(stage, f"/World/viz/v5_end{j}"); sp_.CreateRadiusAttr(0.012); UsdShade.MaterialBindingAPI(sp_.GetPrim()).Bind(self._mat("v5_line", lc))
                self.v5_ends.append(UsdGeom.Xformable(sp_).AddTranslateOp())
        if args.viz_trail == "on":
            c = UsdGeom.BasisCurves.Define(stage, "/World/viz/trail"); c.CreateTypeAttr("linear"); c.CreateWidthsAttr(Vt.FloatArray([0.006])); c.SetWidthsInterpolation("constant")
            c.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*COND_RGB.get(mode, (0.5, 0.5, 0.5)))]))
            self.trail = c; self.buf = deque(maxlen=max(2, int(args.viz_trail_sec * 60)))
            UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(self._mat("trail", COND_RGB.get(mode, (0.5, 0.5, 0.5))))
        if args.viz_ghost:
            O = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/logs/meff_isaac")
            for j, cond in enumerate([x for x in args.viz_ghost.split(",") if x]):
                f = f"{O}/{args.viz_ghost_tag.format(cond=cond)}_b1.npz"
                if not os.path.exists(f):
                    log(f"(viz) ghost {cond}: {f} not found"); continue
                P = np.load(f)["rows"][:, 8:11] - np.array([0, 0, 0.1034])
                g = UsdGeom.BasisCurves.Define(stage, f"/World/viz/ghost_{cond}"); g.CreateTypeAttr("linear"); g.CreateCurveVertexCountsAttr(Vt.IntArray([len(P)]))
                g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P, np.float32))); g.CreateWidthsAttr(Vt.FloatArray([0.004])); g.SetWidthsInterpolation("constant")
                g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*([0.45, 0.45, 0.45] if j == 0 else [0.65, 0.65, 0.65]))]))
                UsdShade.MaterialBindingAPI(g.GetPrim()).Bind(self._mat(f"ghost_{cond}", (0.45, 0.45, 0.45) if j == 0 else (0.65, 0.65, 0.65)))
                log(f"(viz) ghost {cond}: {len(P)} points from {os.path.basename(f)}")
        if args.viz_record:
            import omni.replicator.core as rep
            W, H = [int(x) for x in args.viz_record_res.split("x")]; self.rec = []; self.rec_dir = os.path.expanduser(args.viz_record)
            for j, cs in enumerate([c for c in args.viz_cam.split(";") if c]):   # several cameras: "eye,target;eye,target"
                v = [float(x) for x in cs.split(",")]; cp = f"/World/viz/rec_cam{j}"
                cam = UsdGeom.Camera.Define(stage, cp); cam.CreateFocalLengthAttr(24.0); cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
                if j == 0 and os.environ.get("V5_CAM_M"):   # (v5 matched) camera = the real colour camera: local->world matrix (row-major, USD row vectors) + film back
                    mv = [float(x) for x in os.environ["V5_CAM_M"].split(",")]; UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*mv))
                    fl, hap, vap, hof, vof = [float(x) for x in os.environ["V5_CAM_AP"].split(",")]
                    cam.GetFocalLengthAttr().Set(fl); cam.CreateHorizontalApertureAttr(hap); cam.CreateVerticalApertureAttr(vap)
                    cam.CreateHorizontalApertureOffsetAttr(hof); cam.CreateVerticalApertureOffsetAttr(vof)
                else:
                    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*v[:3]), Gf.Vec3d(*v[3:]), Gf.Vec3d(0, 0, 1)).GetInverse())
                rp = rep.create.render_product(cp, (W, H)); a = rep.AnnotatorRegistry.get_annotator("rgb"); a.attach([rp])
                self.rec.append(a); os.makedirs(f"{self.rec_dir}/cam{j}", exist_ok=True)
        log(f"(viz) eta shell {args.viz_eta_shell} ({len(self.shells)} hand shapes, cost mode {mode}), trail {args.viz_trail} {args.viz_trail_sec} s, "
            f"ghost '{args.viz_ghost}', record '{args.viz_record}'")

    def _mat(self, name, rgb):
        UsdShade, Sdf, Gf = self.UsdShade, self.Sdf, self.Gf
        m = UsdShade.Material.Define(self.stage, f"/World/viz/Looks/{name}"); sh = UsdShade.Shader.Define(self.stage, f"/World/viz/Looks/{name}/s"); sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb)); sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[0.5 * x for x in rgb]))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); return m

    def _robex(self, d):   # (robex2 11-24) d (robot spheres, shapes): the left-out spheres are never the nearest; indices out of range -> unchanged + warning
        ix = getattr(self, "robex_idx", None)
        if ix is None or not len(ix):
            return d
        if ix.max() >= d.shape[0] or len(ix) >= d.shape[0]:
            if not getattr(self, "_robex_warned", False):
                self.log(f"(v5) WARNING: robot sphere exclusion {ix.tolist()} does not fit {d.shape[0]} spheres: all spheres used"); self._robex_warned = True
            return d
        d = np.array(d, copy=True); d[ix] = np.inf; return d

    def _eta(self, mu, mr, seg=18):
        p = self.meff.par.cpu().numpy() if not hasattr(self, "_par") else self._par; self._par = p
        e0, mref, pw, emin, emax = [float(x) for x in p[1:6]]
        if self.mode == 1:
            return e0
        if self.mode in (4, 5):   # (live prep2) gain modes, same rule as the kernel (5 = per-part reference mu_iso)
            inv_r = 1.0 / mr if np.isfinite(mr) and mr > 0 else 0.0
            ratio = mu * (float(self.meff.inv_iso[seg]) + inv_r) if self.mode == 5 else mu / mref
            return float(np.clip(e0 + pw * max(np.sqrt(ratio) - 1.0, 0.0), emin, emax))
        if self.mode == 3:
            mu = 1.0 / (float(self.meff.inv_iso[18]) + (1.0 / mr if np.isfinite(mr) and mr > 0 else 0.0))
        return float(np.clip(e0 * (mu / mref) ** pw, emin, emax))

    def update(self, i, sph, ee):
        t0 = time.perf_counter(); Gf = self.Gf
        if self.zseq is not None and self.human.cur_frame is not None:
            z = self.zseq; j = int(np.argmin(np.abs(z["S"] - z["hs"][min(self.human.cur_frame, len(z["hs"]) - 1)])))
            if j != z["cur"]:
                if z["cur"] >= 0:
                    self.UsdGeom.Imageable(z["prims"][z["cur"]]).MakeInvisible()
                self.UsdGeom.Imageable(z["prims"][j]).MakeVisible(); z["cur"] = j
        if getattr(self, "v5_mesh", None) and self.mode != 0:   # (v5) per-vertex eta at the robot sphere nearest to the part
            qa = self.meff.query_all(sph); d = self._robex(qa["dist"].cpu().numpy()); h = self.human.last_h; I = self.meff.inp   # (robex2) nearest sphere without the left-out ones
            inv_h = I.inv_h.detach().cpu().numpy(); inv_r = I.inv_r.detach().cpu().numpy(); mo = self.meff.mass_on.detach().cpu().numpy(); iso = self.meff.inv_iso.detach().cpu().numpy()
            _io5 = os.environ.get("V5_ISO_OVERRIDE", "")   # (shviz 11-24) the run's recorded ISO reference override "name:kg,..." (render_v5.sh); unset = kernel values
            if _io5 and _io5 != "-":
                _ISN = ["pelvis", "left_thigh", "right_thigh", "abdomen", "left_calf", "right_calf", "left_foot", "right_foot", "chest", "neck", "left_shoulder",
                        "right_shoulder", "head", "left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"]   # = g1_meff_gpu ISO_SEGMENTS
                iso = iso.copy()
                for _kv in _io5.split(","):
                    _n, _v = _kv.split(":"); iso[_ISN.index(_n)] = 1.0 / float(_v)
            _hm5 = os.environ.get("V5_MASS_MODE", "reduced") == "human"   # (shviz) recorded human_mass_mode (reduced = as before)
            p = self.meff.par.detach().cpu().numpy().astype(float); e0, pw, emin, emax = p[1], p[3], p[4], p[5]
            for k, g in self.v5_mesh.items():
                sg = int(h["seg"][k]); s_ = int(np.argmin(d[:, k])); _P, _N, _F = capsule_mesh(np.asarray(h["p0"][k], float), np.asarray(h["p1"][k], float), float(h["rad"][k]))
                ah = np.einsum("ni,ij,nj->n", _N, inv_h[sg], _N); ar = np.einsum("ni,ij,nj->n", _N, inv_r[s_], _N) * float(mo[s_] == 1); mu = 1.0 / (ah + ar)
                e = np.clip(e0 + pw * np.maximum(np.sqrt(mu * (iso[sg] + ar)) - 1.0, 0.0), emin, max(emax, emin)) if self.mode == 5 else np.full(len(_N), e0)
                if _hm5 and self.mode == 5:   # (shviz 11-24) human effective mass reference m_h = 1 / (u' inv_h u) (kernel gain_part_h)
                    e = np.clip(e0 + pw * np.maximum(np.sqrt(iso[sg] / np.maximum(ah, 1e-9)) - 1.0, 0.0), emin, max(emax, emin))
                Pe = _P + _N * e[:, None]; g.GetPointsAttr().Set(self.Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(Pe, np.float32)))
                if k in self.v5_edge:   # (v5 fix3) silhouette segments
                    sg5 = silhouette(Pe, _F, self.v5_eye)
                    if len(sg5):
                        self.v5_edge[k].GetPointsAttr().Set(self.Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(sg5.reshape(-1, 3), np.float32)))
                        self.v5_edge[k].GetCurveVertexCountsAttr().Set(self.Vt.IntArray([2] * len(sg5)))
        if self.shells and self.mode != 0:
            qa = self.meff.query_all(sph); d = self._robex(qa["dist"].cpu().numpy()); mu = qa["mu"].cpu().numpy(); mr = qa["mr"].cpu().numpy(); h = self.human.last_h
            for k, g, tr, ori, col, dcol in self.shells:
                kk = self.elbow_src[k] if k in self.elbow else k   # elbow sphere: forearm shape kk, zero length at its end p0
                s = int(np.argmin(d[:, kk])); e = self._eta(float(mu[s, kk]), float(mr[s, kk]), int(h["seg"][kk])); a, b = h["p0"][kk], (h["p0"][kk] if k in self.elbow else h["p1"][kk]); v = b - a; n = float(np.linalg.norm(v))
                g.GetRadiusAttr().Set(float(h["rad"][kk]) + e); g.GetHeightAttr().Set(n); tr.Set(Gf.Vec3d(*((a + b) / 2).tolist()))
                z = np.array([0, 0, 1.0]); vv = v / max(n, 1e-12); ax = np.cross(z, vv); sn = np.linalg.norm(ax)
                ori.Set(Gf.Quatf(1, 0, 0, 0) if sn < 1e-9 else Gf.Quatf(float(np.cos(np.arctan2(sn, z @ vv) / 2)), *[float(x) for x in ax / sn * np.sin(np.arctan2(sn, z @ vv) / 2)]))
                rgb = self.v5_rgb or mu_rgb(float(mu[s, kk])); col.Set(Gf.Vec3f(*rgb)); dcol.Set(self.Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
        if self.v5_lines is not None:   # (v5) the line of the step whose frame is captured next (i + 1)
            a_, b_ = self.v5_lines; j = min(i + 1, len(a_) - 1)
            if np.all(np.isfinite(a_[j])) and np.all(np.isfinite(b_[j])):
                P = [Gf.Vec3f(*[float(x) for x in a_[j]]), Gf.Vec3f(*[float(x) for x in b_[j]])]
            else:
                P = [Gf.Vec3f(0, 0, -5), Gf.Vec3f(0, 0, -5.001)]
            self.v5_curve.GetPointsAttr().Set(self.Vt.Vec3fArray(P))
            for e_, p_ in zip(self.v5_ends, P):
                e_.Set(Gf.Vec3d(float(p_[0]), float(p_[1]), float(p_[2])))
        if self.trail is not None and i % 2 == 0:
            self.buf.append(ee.detach().cpu().numpy().reshape(3) - np.array([0, 0, 0.1034]))
            if len(self.buf) >= 2:
                P = np.ascontiguousarray(np.array(self.buf), np.float32); self.trail.GetPointsAttr().Set(self.Vt.Vec3fArray.FromNumpy(P))
                self.trail.GetCurveVertexCountsAttr().Set(self.Vt.IntArray([len(P)]))
        self.t_upd.append((time.perf_counter() - t0) * 1000)

    def capture(self, i):
        if self.rec is None:
            return
        t0 = time.perf_counter(); from PIL import Image
        if getattr(self, "_pool", None) is None:   # (live prep2) files written by 3 threads; get_data stays in this thread
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(3); self._fut = []; self.t_get = []
        ext = os.environ.get("G1_VIZ_REC_EXT", "png")
        for j, a in enumerate(self.rec):
            dat = np.asarray(a.get_data()); tg = time.perf_counter()
            if dat.size:
                im = Image.fromarray(np.ascontiguousarray(dat[:, :, :3])); fp = f"{self.rec_dir}/cam{j}/f{i:05d}.{ext}"
                self._fut.append(self._pool.submit(im.save, fp, **({"quality": 92} if ext == "jpg" else {"compress_level": 1})))
            self.t_get.append((tg - t0) * 1000)
        self._fut = [f for f in self._fut if not f.done()]
        while len(self._fut) > 24:
            self._fut.pop(0).result()
        self.t_cap.append((time.perf_counter() - t0) * 1000)

    def finish(self, out_dir):
        st = dict(update_ms_mean=float(np.mean(self.t_upd)) if self.t_upd else None, update_ms_p95=float(np.percentile(self.t_upd, 95)) if self.t_upd else None,
                  capture_ms_mean=float(np.mean(self.t_cap)) if self.t_cap else None, n_updates=len(self.t_upd))
        if self.rec is not None and getattr(self, "_pool", None) is not None:
            for f in self._fut:
                f.result()
            self._pool.shutdown(wait=True); st["get_data_ms_mean"] = float(np.mean(self.t_get)) if self.t_get else None
        if self.rec is not None:
            st["mp4"] = []
            for j in range(len(self.rec)):
                if not os.path.exists(f"{self.rec_dir}/cam{j}/f00001.png"):   # (live prep2) --viz-record-every > 1: no raw clip
                    continue
                mp4 = f"{self.rec_dir.rstrip('/')}_cam{j}.mp4"
                r = subprocess.run(["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", f"{self.rec_dir}/cam{j}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4])
                st["mp4"].append(mp4 if r.returncode == 0 else None)
            st["mp4_note"] = "raw camera clips, 30 fps for 60 Hz steps (0.5x)"
        json.dump(st, open(os.path.join(out_dir, f"{self.tag}_viz.json"), "w"), indent=1); self.log(f"(viz) {st}")
        return st


def add_zone(stage, npz, path="/World/viz/zone", opacity=0.35, by_mu=True, rgb=(0.62, 0.72, 0.85), roughness=0.9):
    """(V1) static translucent avoidance-zone surface from v1_zone.py (zone_<cond>_mesh.npz, world frame); vertex colour by mu (blue 0.5 -> red 3.0 kg)"""
    from pxr import Gf, UsdGeom, UsdShade, Sdf, Vt
    z = np.load(npz); P = np.ascontiguousarray(z["verts"], np.float32); Fc = np.asarray(z["faces"], np.int32)
    m = UsdGeom.Mesh.Define(stage, path); m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P)); m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*P.min(0).tolist()), Gf.Vec3f(*P.max(0).tolist())]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(Fc))); m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(Fc.reshape(-1))); m.CreateSubdivisionSchemeAttr("none"); m.CreateDoubleSidedAttr(True)
    C = np.array([mu_rgb(float(x)) for x in z["mu"]], np.float32) if by_mu else np.tile(np.array(rgb, np.float32), (len(P), 1))
    pv = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex); pv.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(C)))
    UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    # translucent colour by mu: faces binned into 8 mu bins (0.5 - 3.0 kg), one UsdPreviewSurface material per bin (colour + opacity) bound to a face subset
    # (a single material reading the displayColor primvar rendered white in RTX; displayColor alone ignores displayOpacity)
    fm = np.asarray(z["mu"], float)[Fc].mean(1) if by_mu else np.zeros(len(Fc)); edges = np.linspace(0.5, 3.0, 9); bins = np.clip(np.digitize(fm, edges) - 1, 0, 7)
    for bi in (range(8) if by_mu else [0]):
        idx = np.nonzero(bins == bi)[0] if by_mu else np.arange(len(Fc))
        if not len(idx):
            continue
        col = mu_rgb(0.5 * (edges[bi] + edges[bi + 1])) if by_mu else rgb
        sub = UsdGeom.Subset.CreateGeomSubset(UsdGeom.Imageable(m), f"mu{bi}", "face", Vt.IntArray.FromNumpy(idx.astype(np.int32)), "materialBind")
        mat = UsdShade.Material.Define(stage, f"{path}_mat{bi}"); sh = UsdShade.Shader.Define(stage, f"{path}_mat{bi}/s"); sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col)); sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness)); sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02)); mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
    return dict(verts=len(P), faces=len(Fc), file=npz)


def add_zone_cage(stage, npz, path="/World/viz/zone_cage", dz=0.02, dx=0.03, width=0.0035, by_mu=True, rgb=(0.35, 0.45, 0.65)):
    """(V2) zone boundary drawn as contour rings (see-through): sections of the v1 zone surface by horizontal planes every dz and by planes normal to x
    every dx; segments coloured by mu (8 bins, blue 0.5 -> red 3.0 kg; one emissive material per bin)"""
    from pxr import Gf, UsdGeom, UsdShade, Sdf, Vt
    z = np.load(npz); V = np.asarray(z["verts"], float); Fc = np.asarray(z["faces"]); MU = np.asarray(z["mu"], float); T = V[Fc]; TM = MU[Fc]; segs, mus = [], []
    for axis, step in ((2, dz), (0, dx)):
        lo, hi = V[:, axis].min(), V[:, axis].max()
        for val in np.arange(np.ceil(lo / step) * step, hi, step):
            sv = T[:, :, axis] - val; cross = ~((sv > 0).all(1) | (sv < 0).all(1))
            for tri, sv_, tm in zip(T[cross], sv[cross], TM[cross]):
                pts, mm = [], []
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    if (sv_[i] > 0) != (sv_[j] > 0):
                        w = sv_[i] / (sv_[i] - sv_[j]); pts.append(tri[i] + w * (tri[j] - tri[i])); mm.append(tm[i] + w * (tm[j] - tm[i]))
                if len(pts) == 2:
                    segs.append(pts); mus.append(0.5 * (mm[0] + mm[1]))
    segs = np.asarray(segs, np.float32); mus = np.asarray(mus); edges = np.linspace(0.5, 3.0, 9); bins = np.clip(np.digitize(mus, edges) - 1, 0, 7) if by_mu else np.zeros(len(mus), int)
    for bi in sorted(set(bins.tolist())):
        sel = segs[bins == bi]; col = mu_rgb(0.5 * (edges[bi] + edges[bi + 1])) if by_mu else rgb
        c = UsdGeom.BasisCurves.Define(stage, f"{path}/b{bi}"); c.CreateTypeAttr("linear"); c.CreateCurveVertexCountsAttr(Vt.IntArray([2] * len(sel)))
        c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(sel.reshape(-1, 3)))); c.CreateWidthsAttr(Vt.FloatArray([width])); c.SetWidthsInterpolation("constant")
        c.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*col)])); UsdGeom.PrimvarsAPI(c.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        m = UsdShade.Material.Define(stage, f"{path}/m{bi}"); sh = UsdShade.Shader.Define(stage, f"{path}/m{bi}/s"); sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col)); sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[0.6 * x for x in col]))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(m)
    return dict(segments=int(len(segs)), file=npz, style="cage")
