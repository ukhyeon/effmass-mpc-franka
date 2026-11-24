"""robot_inertia_np.py — FR3 v2.1 로봇 구별 3x3 역관성, numpy 만 (goal1 1-2d 결정 1, 안 C). Pinocchio 를 import 하지 않는다.

정의는 robot_inertia.RobotInertia (Pinocchio, 1-2c) 와 같다:
  - fr3v2_1_curobo_narrow.urdf, 손가락 두 관절 0.04 고정 (손가락 링크는 손에 붙은 강체), nv = 7
  - M(q) = Σ_링크 m Jvᵀ Jv + Jωᵀ (R I Rᵀ) Jω  (링크 COM 기준, 7자유도라 링크 야코비안 합으로 직접; 결과는 CRBA 와 같은 M)
  - 회전자 반사 관성 diag(N² Jm) (URDF <dynamics gear_ratio motor_inertia>), 기본 켬, rotor=False 로 끔
  - 구 중심 점 야코비안 J_p (관절 축 z_k × (p − o_k), 그 구가 붙은 몸체까지의 관절만) → Λ⁻¹ = J_p M⁻¹ J_pᵀ, 구 61개 한 번에
  - 구 목록·순서는 RobotInertia 와 같음 (fr3v2_1_port.yml, collision_link_names 순서, attached_object 제외)
시뮬레이션(Isaac)과 실기(목표 2)에서 이 구현을 공통으로 쓰고, Pinocchio(RobotInertia)는 오프라인 대조 기준으로 둔다.
"""
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

URDF = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc")) + "/goal1/fr3v2_1/fr3v2_1_curobo_narrow.urdf"
SPHERES = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc")) + "/goal1/fr3v2_1/spheres/fr3v2_1_port.yml"
COLLISION_LINKS = ["fr3v2_1_link0", "fr3v2_1_link1", "fr3v2_1_link2", "fr3v2_1_link3", "fr3v2_1_link4", "fr3v2_1_link5", "fr3v2_1_link6",
                   "fr3v2_1_link7", "fr3v2_1_hand", "fr3v2_1_leftfinger", "fr3v2_1_rightfinger"]
ARM = [f"fr3v2_1_joint{i}" for i in range(1, 8)]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def spd_inv(M):
    """대칭 양정치 n×n 역행렬, LAPACK 을 부르지 않음 (1-2d 1안).
    Isaac numpy(OpenBLAS64 0.3.23 pthreads)의 np.linalg.inv/solve 는 호출 뒤 스레드 6개가 바쁜 대기를 해 같은 프로세스의 Isaac 물리·GPU MPC library 를 느리게 함 [확인].
    Cholesky M = L Lᵀ (열마다 반복 n 회) → L⁻¹ (행마다 반복 n 회) → M⁻¹ = L⁻ᵀ L⁻¹"""
    n = M.shape[0]
    L = np.zeros_like(M)
    for j in range(n):
        d = M[j, j] - L[j, :j] @ L[j, :j]
        if d <= 0.0:
            raise np.linalg.LinAlgError("spd_inv: 양정치 아님")
        L[j, j] = np.sqrt(d)
        L[j + 1:, j] = (M[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    Li = np.zeros_like(M)
    for i in range(n):
        Li[i, i] = 1.0 / L[i, i]
        Li[i, :i] = -(L[i, :i] @ Li[:i, :i]) / L[i, i]
    return Li.T @ Li


def _T(el):
    A = np.eye(4)
    if el is not None:
        A[:3, :3] = _rpy(*[float(x) for x in el.get("rpy", "0 0 0").split()]); A[:3, 3] = [float(x) for x in el.get("xyz", "0 0 0").split()]
    return A


class RobotInertiaNP:
    def __init__(self, urdf=URDF, spheres_yml=SPHERES, rotor=True, finger_q=0.04, links=COLLISION_LINKS):
        root = ET.parse(urdf).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        base = [l.get("name") for l in root.findall("link") if l.get("name") not in joints][0]
        # 링크마다 (몸체 번호 b: -1 = 고정 베이스, 0..6 = 관절 1..7 뒤 몸체, 몸체 좌표에서 링크 좌표 변환)
        self.body_of = {base: (-1, np.eye(4))}
        X = [None] * 7; N = np.zeros(7); Jm = np.zeros(7)

        def resolve(link):
            if link in self.body_of:
                return self.body_of[link]
            j = joints[link]; pb, pT = resolve(j.find("parent").get("link")); O = _T(j.find("origin")); typ = j.get("type")
            if typ == "revolute":
                k = ARM.index(j.get("name"))
                ax = np.array([float(x) for x in j.find("axis").get("xyz").split()]); assert np.allclose(ax, [0, 0, 1]), "축이 z 가 아님"
                assert pb == k - 1, "직렬 사슬이 아님"
                X[k] = pT @ O                                       # 부모 몸체 좌표 → 관절 k 좌표 (회전 전)
                d = j.find("dynamics"); N[k] = float(d.get("gear_ratio")); Jm[k] = float(d.get("motor_inertia"))
                r = (k, np.eye(4))
            elif typ == "prismatic":                                 # 손가락: finger_q 로 고정
                B = np.eye(4); B[:3, 3] = np.array([float(x) for x in j.find("axis").get("xyz").split()]) * finger_q
                r = (pb, pT @ O @ B)
            else:
                r = (pb, pT @ O)
            self.body_of[link] = r
            return r

        for l in root.findall("link"):
            resolve(l.get("name"))
        assert all(x is not None for x in X)
        self.X = np.stack(X)                                         # (7,4,4)
        self.gear_ratio, self.motor_inertia = N, Jm
        self.rotor_diag = N ** 2 * Jm
        self.rotor = bool(rotor)
        # 질량 링크 (몸체 0..6 에 붙은 것만; 베이스는 M 에 기여 없음)
        lb, lm, lc, lI = [], [], [], []
        for l in root.findall("link"):
            i = l.find("inertial"); b, T = self.body_of[l.get("name")]
            if i is None or b < 0:
                continue
            Ti = T @ _T(i.find("origin")); I = i.find("inertia"); g = lambda k: float(I.get(k))
            Il = np.array([[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")], [g("ixz"), g("iyz"), g("izz")]])
            lb.append(b); lm.append(float(i.find("mass").get("value"))); lc.append(Ti[:3, 3]); lI.append(Ti[:3, :3] @ Il @ Ti[:3, :3].T)
        self.l_body = np.array(lb); self.l_mass = np.array(lm); self.l_com = np.array(lc); self.l_I = np.array(lI)
        self.l_mask = (np.arange(7)[None, :] <= self.l_body[:, None]).astype(float)            # (L,7)
        self.l_sqrtm = np.sqrt(self.l_mass)
        # 구
        sph = yaml.safe_load(open(spheres_yml))["collision_spheres"]
        self.sphere_link, loc, rad = [], [], []
        for ln in links:
            for s in sph.get(ln, []):
                self.sphere_link.append(ln); loc.append(s["center"]); rad.append(s["radius"])
        self.sphere_radius = np.asarray(rad, float)
        sb, sl = [], []
        for ln, c in zip(self.sphere_link, loc):
            b, T = self.body_of[ln]; sb.append(b); sl.append((T @ np.r_[c, 1.0])[:3])
        self.s_body = np.array(sb); self.s_local = np.array(sl)
        self.s_mask = (np.arange(7)[None, :] <= self.s_body[:, None]).astype(float)            # (S,7), 베이스 구 = 0
        self.s_idx = self.s_body + 1                                                           # 0 = 베이스(단위), 1..7 = 몸체
        self.l_idx = self.l_body + 1
        self._W = np.zeros((8, 4, 4)); self._W[0] = np.eye(4)
        self.timing = {}

    def _fk(self, q):
        W = self._W
        c, s = np.cos(q), np.sin(q)
        for k in range(7):
            Rz = np.array([[c[k], -s[k], 0.0, 0.0], [s[k], c[k], 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            W[k + 1] = W[k] @ self.X[k] @ Rz
        return W

    def compute(self, q, timed=False):
        """q (7,) → centers (S,3), inv_inertia (S,3,3) [1/kg], M (7,7)"""
        pc = time.perf_counter
        t0 = pc()
        q = np.asarray(q, float)
        W = self._fk(q)
        O = W[1:, :3, 3]; Z = W[1:, :3, 2]                                                      # 관절 축 원점·방향 (7,3)
        t1 = pc()
        Wl = W[self.l_idx]; R = Wl[:, :3, :3]
        pl = np.einsum("lij,lj->li", R, self.l_com) + Wl[:, :3, 3]                             # 링크 COM (L,3)
        Jv = np.cross(Z[None], pl[:, None] - O[None]) * self.l_mask[..., None]                # (L,7,3)
        Jw = Z[None] * self.l_mask[..., None]                                                  # (L,7,3)
        Iw = R @ self.l_I @ R.transpose(0, 2, 1)
        A = (Jv * self.l_sqrtm[:, None, None]).transpose(1, 0, 2).reshape(7, -1)
        M = A @ A.T + np.einsum("lki,lij,lmj->km", Jw, Iw, Jw)
        if self.rotor:
            M = M + np.diag(self.rotor_diag)
        t2 = pc()
        Minv = spd_inv(M)                      # (1-2d 1안) np.linalg.inv 대신 LAPACK 없는 Cholesky
        Minv = 0.5 * (Minv + Minv.T)
        t3 = pc()
        Ws = W[self.s_idx]
        P = np.einsum("sij,sj->si", Ws[:, :3, :3], self.s_local) + Ws[:, :3, 3]               # 구 중심 (S,3)
        Js = (np.cross(Z[None], P[:, None] - O[None]) * self.s_mask[..., None]).transpose(0, 2, 1)   # (S,3,7)
        t4 = pc()
        L = Js @ Minv @ Js.transpose(0, 2, 1)
        L = 0.5 * (L + L.transpose(0, 2, 1))
        t5 = pc()
        if timed:
            self.timing = dict(fk=(t1 - t0) * 1e6, mass_matrix=(t2 - t1) * 1e6, inverse=(t3 - t2) * 1e6, point_jacobian=(t4 - t3) * 1e6, inv_inertia=(t5 - t4) * 1e6,
                               total=(t5 - t0) * 1e6)
        return P, L, M

    def point_inv_inertia(self, q, body_link, point_local=(0.0, 0.0, 0.0)):
        """임의 링크(URDF 이름) 좌표의 점 하나 역관성 (비교용)"""
        W = self._fk(np.asarray(q, float)); O = W[1:, :3, 3]; Z = W[1:, :3, 2]
        b, T = self.body_of[body_link]; p = (W[b + 1] @ T @ np.r_[np.asarray(point_local, float), 1.0])[:3]
        mask = (np.arange(7) <= b).astype(float)
        J = (np.cross(Z, p - O) * mask[:, None]).T
        _, _, M = self.compute(q)
        return p, J @ np.linalg.solve(M, J.T)
