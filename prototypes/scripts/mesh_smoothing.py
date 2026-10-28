"""White-surface meshing and smoothing, and fronts seeded from a surface (finding 28).

Prototype, except the mesh operations that moved to the package (baillarger.mesh_smoothing:
edges, boundary_vertices, vertex_normals, relax, gauss_steps, twice, smooth_normals), which are
imported from there. Three parts.

1. Phantoms with a known smooth white surface. Each gives, at any point X (voxel
   units of the 1x grid, voxel centres at integer coordinates), the signed distance
   to the WM (+ outside), the foot point on the surface, the outward normal there and
   the principal curvatures there (per voxel, + where convex seen from outside). The
   label is the voxel centre's side (WM where the distance < 0); cortex is
   0 < d <= T. Shapes: a ball; a wavy body (WM below a graph cos x cos y); a nearly
   grid-aligned plane (long staircase terraces); a thin blade (WM within w/2 of a
   wavy cylinder sheet, cortex on both sides). All are rotated off the grid axes
   except where stated.

2. Start surfaces:
   - contour(wm, sigma): marching tetrahedra at 0.5 of the label smoothed by sigma
     voxels (sigma = 0: the binary label itself; every vertex at a grid-edge
     midpoint). baillarger.frame.white_mesh is contour(sigma) with its normals
     smoothed along edges.
   - relax: tangential relaxation (vertices move toward their neighbours' centroid
     within the tangent plane), which evens the triangles without moving the surface
     along its normal. The marching-tetrahedra triangles of a binary label are
     already about half a voxel across, so this stands in for remeshing; no edge
     splits or collapses (a limit).
   - taubin(v, f, cutoff): Taubin lambda|mu smoothing (umbrella Laplacian). The pass
     band is set from a wavelength in voxels: k_PB = pi^2 <e^2> / L^2, the umbrella
     operator's eigenvalue for a plane wave of wavelength L on a regular mesh of edge
     e. Wavelengths above L pass unchanged (no shrinkage); shorter ones are damped.
     Box-face boundary vertices are fixed.

3. A copy of baillarger.fronts.march (march_band) with per-seed start values: each
   seed voxel has its own arrival, arc length and root position, and is frozen. The
   root position is carried like arc length but without a source term
   (sum_j rho_j a_j (X - X_j) = 0), which is open decision 4's proposal. band_init
   gives the seeds from a mesh: cortex voxels within `band` voxels of the mesh, and
   every cortex voxel 6-adjacent to the WM label, take the distance from the mesh's
   tangent plane at the closest point, in the metric (|(x - p).n| / sqrt(n^T D n)),
   the metric foot point as root, and the Euclidean distance to it as arc length.
   The domain is the cortex only. Limits: the band uses the closest triangle's plane
   (exact for a plane, second order in the band width on a curved mesh); the march
   is the package's first-order scheme.

Units: voxels of the 1x grid unless stated; H_MM converts.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy import sparse
from scipy.spatial import cKDTree

from baillarger._jit import njit
from baillarger.fronts.march import _heap_pop, _heap_push, _reverse, _stencils
from baillarger.fronts.selling import selling
from baillarger.mesh import marching_tetrahedra
from baillarger.mesh_smoothing import (  # noqa: F401  (the package's, re-exported)
    boundary_vertices, edges, gauss_steps, relax, smooth_normals, twice, umbrella, vertex_normals)

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

H_MM = 0.16
_INF = np.inf


# ------------------------------------------------------------------ phantoms


def rotation(axis, deg) -> Float:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


@dataclass
class Truth:
    d: Float  # signed distance to the white surface (+ outside the WM)
    foot: Float  # (N, 3) foot point
    n: Float  # (N, 3) outward unit normal at the foot
    k1: Float  # principal curvatures at the foot, k1 >= k2 (+ convex)
    k2: Float
    side: Int  # which sheet (blades: +1 / -1; else 0)


@dataclass
class Phantom:
    name: str
    shape: tuple[int, int, int]
    T: float  # cortex thickness (voxels)
    kind: str
    p: dict = field(default_factory=dict)

    # -- the true surface
    def truth(self, X) -> Truth:
        X = np.asarray(X, float).reshape(-1, 3)
        if self.kind == "ball":
            c, R = np.asarray(self.p["c"]), self.p["R"]
            r = X - c
            rn = np.linalg.norm(r, axis=1)
            u = r / np.maximum(rn, 1e-12)[:, None]
            k = np.full(len(X), 1.0 / R)
            return Truth(rn - R, c + R * u, u, k, k.copy(), np.zeros(len(X), np.int64))
        if self.kind == "graph":
            return self._graph(X)
        if self.kind == "blade":
            return self._blade(X)
        if self.kind == "fin":
            return self._fin(X)
        raise ValueError(self.kind)

    def _h(self, u, v):
        """Graph height and derivatives (h, hu, hv, huu, huv, hvv) in local coordinates."""
        p = self.p
        z0, A, lam, tilt = p["z0"], p.get("A", 0.0), p.get("lam", 1.0), p.get("tilt", 0.0)
        w = 2 * np.pi / lam
        cu, su, cv, sv = np.cos(w * u), np.sin(w * u), np.cos(w * v), np.sin(w * v)
        h = z0 + A * cu * cv + tilt * v
        hu = -A * w * su * cv
        hv = -A * w * cu * sv + tilt
        huu = -A * w * w * cu * cv
        huv = A * w * w * su * sv
        hvv = -A * w * w * cu * cv
        return h, hu, hv, huu, huv, hvv

    def _graph(self, X) -> Truth:
        Q, o = self.p["Q"], self.p["o"]
        L = (X - o) @ Q  # local coordinates (columns of Q are the local axes)
        # start from the nearest of a dense sample of the surface (Newton from the
        # point's own (u, v) diverges beyond the focal distance of a trough)
        lo, hi = L[:, :2].min(0) - 30, L[:, :2].max(0) + 30
        gu, gv = (np.arange(lo[i], hi[i] + 0.25, 0.25) for i in range(2))
        UU, VV = np.meshgrid(gu, gv, indexing="ij")
        UU, VV = UU.ravel(), VV.ravel()
        _, j = cKDTree(np.c_[UU, VV, self._h(UU, VV)[0]]).query(L)
        u, v = UU[j], VV[j]
        for _ in range(40):
            h, hu, hv, huu, huv, hvv = self._h(u, v)
            gx, gy, gz = u - L[:, 0], v - L[:, 1], h - L[:, 2]
            Fu = gx + gz * hu
            Fv = gy + gz * hv
            Auu = 1 + hu * hu + gz * huu
            Auv = hu * hv + gz * huv
            Avv = 1 + hv * hv + gz * hvv
            det = Auu * Avv - Auv * Auv
            du = (Avv * Fu - Auv * Fv) / det
            dv = (-Auv * Fu + Auu * Fv) / det
            u -= du
            v -= dv
            if not du.size or max(np.abs(du).max(), np.abs(dv).max()) < 1e-12:
                break
        h, hu, hv, huu, huv, hvv = self._h(u, v)
        W = np.sqrt(1 + hu * hu + hv * hv)
        nl = np.stack([-hu, -hv, np.ones_like(hu)], 1) / W[:, None]
        footl = np.stack([u, v, h], 1)
        d = np.sum((L - footl) * nl, 1)
        # shape operator S = I^-1 II, II = Hess h / W; kappa (convex +) = -eig
        I11, I12, I22 = 1 + hu * hu, hu * hv, 1 + hv * hv
        II11, II12, II22 = huu / W, huv / W, hvv / W
        detI = I11 * I22 - I12 * I12
        S11 = (I22 * II11 - I12 * II12) / detI
        S12 = (I22 * II12 - I12 * II22) / detI
        S21 = (-I12 * II11 + I11 * II12) / detI
        S22 = (-I12 * II12 + I11 * II22) / detI
        tr, dt = S11 + S22, S11 * S22 - S12 * S21
        disc = np.sqrt(np.maximum(tr * tr / 4 - dt, 0))
        e_lo, e_hi = tr / 2 - disc, tr / 2 + disc
        return Truth(d, o + footl @ Q.T, nl @ Q.T, -e_lo, -e_hi, np.zeros(len(X), np.int64))

    def _blade(self, X) -> Truth:
        Q, o, w = self.p["Q"], self.p["o"], self.p["w"]
        a, lam = self.p["a"], self.p["lam"]
        k = 2 * np.pi / lam
        L = (X - o) @ Q
        gt = np.arange(L[:, 1].min() - 30, L[:, 1].max() + 30, 0.05)
        _, j = cKDTree(np.c_[a * np.sin(k * gt), gt]).query(L[:, :2])
        t = gt[j]
        for _ in range(40):
            c, c1, c2 = a * np.sin(k * t), a * k * np.cos(k * t), -a * k * k * np.sin(k * t)
            gx, gy = c - L[:, 0], t - L[:, 1]
            F1 = gx * c1 + gy
            F2 = c1 * c1 + 1 + gx * c2
            dt = F1 / F2
            t -= dt
            if not dt.size or np.abs(dt).max() < 1e-12:
                break
        c, c1, c2 = a * np.sin(k * t), a * k * np.cos(k * t), -a * k * k * np.sin(k * t)
        W = np.sqrt(1 + c1 * c1)
        nc = np.stack([np.ones_like(t), -c1, np.zeros_like(t)], 1) / W[:, None]
        S = np.stack([c, t, L[:, 2]], 1)
        s = np.sum((L - S) * nc, 1)
        side = np.where(s >= 0, 1, -1)
        d = np.abs(s) - w / 2
        footl = S + (side * w / 2)[:, None] * nc
        kc = -c2 / W**3
        kap = np.where(side > 0, kc / (1 + kc * w / 2), -kc / (1 - kc * w / 2))
        nl = side[:, None] * nc
        z = np.zeros_like(kap)
        return Truth(d, o + footl @ Q.T, nl @ Q.T, np.maximum(kap, z), np.minimum(kap, z), side)

    def _fin(self, X) -> Truth:
        """A blade of thickness w (centre x' = a sin(2 pi y' / lam)) on a base (WM below
        z' = zbase), up to a rounded tip at z' = ztip (a half cylinder of radius w/2
        round the tip line). Distance: the smaller of the blade's and the base's (exact
        outside the WM; the concave junction is a crease, and the tip's curvature
        ignores the centre curve's own bend)."""
        Q, o, w, ztip, zb = self.p["Q"], self.p["o"], self.p["w"], self.p["ztip"], self.p["zbase"]
        tb = self._blade(X)
        L = (X - o) @ Q
        zt = L[:, 2]
        # signed distance across the centre sheet: |s| = d + w/2, sign = side
        s = (tb.d + w / 2) * tb.side
        nc_l = (tb.n @ Q) * tb.side[:, None]  # centre-sheet normal (local)
        above = zt > ztip
        q = s[:, None] * nc_l + (zt - ztip)[:, None] * np.array([0, 0, 1.0])
        rr = np.linalg.norm(q, axis=1)
        u = q / np.maximum(rr, 1e-12)[:, None]
        foot_tip_l = (L - q) + (w / 2) * u
        d = np.where(above, rr - w / 2, tb.d)
        foot = np.where(above[:, None], o + foot_tip_l @ Q.T, tb.foot)
        n = np.where(above[:, None], u @ Q.T, tb.n)
        k1 = np.where(above, 2 / w, tb.k1)
        k2 = np.where(above, 0.0, tb.k2)
        side = np.where(above, 2, tb.side)
        db = zt - zb
        base = db < d
        Lb = L.copy()
        Lb[:, 2] = zb
        d = np.where(base, db, d)
        foot = np.where(base[:, None], o + Lb @ Q.T, foot)
        n = np.where(base[:, None], np.broadcast_to(Q[:, 2], n.shape), n)
        k1 = np.where(base, 0.0, k1)
        k2 = np.where(base, 0.0, k2)
        side = np.where(base, 0, side)
        return Truth(d, foot, n, k1, k2, side)

    # -- grids
    def grid(self, scale: int = 1) -> Float:
        """Points of the grid with spacing 1/scale (1x voxel units), shape (*shape_s, 3)."""
        n = [scale * (s - 1) + 1 for s in self.shape]
        ax = [np.arange(m) / scale for m in n]
        return np.stack(np.meshgrid(*ax, indexing="ij"), -1)

    def labels(self, scale: int = 1):
        """(wm, cortex, truth over the grid) at grid spacing 1/scale."""
        G = self.grid(scale)
        tr = self.truth(G.reshape(-1, 3))
        d = tr.d.reshape(G.shape[:-1])
        return d < 0, (d > 0) & (d <= self.T), tr


def ball(R=8.0, T=15.0) -> Phantom:
    n = int(2 * (R + T + 5)) + 1
    c = np.full(3, (n - 1) / 2) + np.array([0.31, -0.17, 0.23])  # off the grid
    return Phantom(f"ball_R{R:g}", (n, n, n), T, "ball", dict(c=c, R=R))


def wavy(A=2.0, lam=48.0, T=15.0, n=72) -> Phantom:
    Q = rotation((1, 2, 3), 25)
    o = np.array([n / 2, n / 2, n / 2 - T / 2]) - 0.37 * Q[:, 2]
    return Phantom("wavy", (n, n, n), T, "graph", dict(Q=Q, o=o, z0=0.0, A=A, lam=lam))


def plane(tilt_deg=4.0, T=15.0, n=64) -> Phantom:
    """A plane tilted tilt_deg from the grid's z axis about x: long terraces."""
    Q = np.eye(3)
    o = np.array([0.0, n / 2, n / 2 - T / 2 + 0.29])
    return Phantom(f"plane_{tilt_deg:g}deg", (n, n, n), T, "graph",
                   dict(Q=Q, o=o, z0=0.0, tilt=np.tan(np.radians(tilt_deg))))


def fin(w=2.0, height=12.0, a=2.0, lam=48.0, T=12.0, n=64, deg=20.0) -> Phantom:
    Q = rotation((2, 1, 3), deg)
    o = np.full(3, n / 2) + np.array([0.21, -0.33, 0.12])
    zb = -height / 2 - 4
    return Phantom(f"fin_w{w:g}", (n, n, n), T, "fin",
                   dict(Q=Q, o=o, w=w, a=a, lam=lam, zbase=zb, ztip=zb + height))


def blade(w=3.0, a=2.0, lam=48.0, T=12.0, n=64, deg=25.0) -> Phantom:
    Q = rotation((2, 1, 3), deg)
    o = np.full(3, n / 2) + np.array([0.21, -0.33, 0.12])
    return Phantom(f"blade_w{w:g}", (n, n, n), T, "blade", dict(Q=Q, o=o, w=w, a=a, lam=lam))


# ------------------------------------------------------------------ meshes


def contour(wm: Bool, sigma: float) -> tuple[Float, Int]:
    """Marching tetrahedra at 0.5 of the label smoothed by sigma voxels; faces
    oriented outward (out of the WM)."""
    F = ndi.gaussian_filter(wm.astype(np.float64), sigma) if sigma > 0 else wm.astype(np.float64)
    v, f = marching_tetrahedra(F, 0.5)
    return v, f[:, [0, 2, 1]].copy()  # marching_tetrahedra points into the WM


def _umbrella(v: Float, f: Int):
    A, deg = umbrella(v, f)
    return A, deg, edges(f)


def taubin(v: Float, f: Int, iters: int, lam: float = 0.6307, mu: float = -0.6732) -> Float:
    """Taubin lambda|mu smoothing, umbrella Laplacian, with Taubin's standard pass
    band (k_PB = 1/lam + 1/mu = 0.1). Its gain exceeds 1 inside the pass band, by
    a factor 1 + lam mu k (k - k_PB) per pair, which compounds over iterations: a pass
    band moved up to set a short cutoff blows up (measured: a cutoff of 2 voxels with
    lam 0.5 gave 1.05 per pair, x4.8 over 30 pairs). Box-face boundary vertices are fixed."""
    A, deg, _ = _umbrella(v, f)
    fixed = boundary_vertices(f, len(v))
    x = v.copy()
    for _ in range(iters):
        for s in (lam, mu):
            d = (A @ x) / deg[:, None] - x
            d[fixed] = 0
            x = x + s * d
    return x


def min_angles(v: Float, f: Int) -> Float:
    P = v[f]
    out = []
    for k in range(3):
        a, b = P[:, (k + 1) % 3] - P[:, k], P[:, (k + 2) % 3] - P[:, k]
        c = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
        out.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.min(out, 0)


# ------------------------------------------------------------------ closest points


def _closest_on_tri(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (N, 3) (Ericson 5.1.5),
    and their barycentric weights (N, 3)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, 1), np.sum(ac * ap, 1)
    bp = p - b
    d3, d4 = np.sum(ab * bp, 1), np.sum(ac * bp, 1)
    cp = p - c
    d5, d6 = np.sum(ab * cp, 1), np.sum(ac * cp, 1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    w = vc / den
    bary = np.stack([1 - v - w, v, w], 1)  # interior
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)  # edge ab
        t = d1 / (d1 - d3)
        bary[m] = np.stack([1 - t, t, 0 * t], 1)[m]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)  # edge ac
        t = d2 / (d2 - d6)
        bary[m] = np.stack([1 - t, 0 * t, t], 1)[m]
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)  # edge bc
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bary[m] = np.stack([0 * t, 1 - t, t], 1)[m]
    for m, k in (((d1 <= 0) & (d2 <= 0), 0), ((d3 >= 0) & (d4 <= d3), 1), ((d6 >= 0) & (d5 <= d6), 2)):
        e = np.zeros(3)
        e[k] = 1
        bary[m] = e
    out = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:] * c
    return out, bary


def pseudonormals(v: Float, f: Int):
    """Face normals, angle-weighted vertex normals, and per face edge normals (the
    sum of the two adjacent faces' normals; edges k: (f[k], f[k + 1])). The sign of
    (x - p).n with n the pseudonormal of the feature holding the closest point p is
    the inside/outside test of Baerentzen & Aanaes (2005), exact for a closed
    manifold mesh."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-30
    vn = np.zeros_like(v)
    for k in range(3):
        a, b = v[f[:, (k + 1) % 3]] - v[f[:, k]], v[f[:, (k + 2) % 3]] - v[f[:, k]]
        c = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
        np.add.at(vn, f[:, k], np.arccos(np.clip(c, -1, 1))[:, None] * fn)
    nv = len(v)
    key = np.stack([np.minimum(f, np.roll(f, -1, 1)) * nv + np.maximum(f, np.roll(f, -1, 1))], 0)[0]
    uk, inv = np.unique(key.ravel(), return_inverse=True)
    en_u = np.zeros((len(uk), 3))
    np.add.at(en_u, inv, np.repeat(fn, 3, 0))
    en = en_u[inv].reshape(len(f), 3, 3)  # (face, edge k, xyz)
    return fn, vn, en


def closest_on_mesh(v: Float, f: Int, X: Float, k: int = 8, chunk: int = 20000):
    """For points X: closest point on the mesh, its face, and the unit normal of the
    closest point: (X - P) / |X - P|, signed to the face's outward side (the face
    normal where X is on the mesh or the closest point is inside the face)."""
    tree = cKDTree(v)
    nv = len(v)
    order = np.argsort(f.ravel(), kind="stable")
    fv = f.ravel()[order]
    fid = order // 3
    st = np.searchsorted(fv, np.arange(nv + 1))
    val = np.diff(st)
    maxv = int(val.max())
    table = -np.ones((nv, maxv), np.int64)
    for j in range(maxv):
        has = val > j
        table[has, j] = fid[st[:-1][has] + j]
    fn, vn, en = pseudonormals(v, f)
    P = np.zeros_like(X)
    F = np.zeros(len(X), np.int64)
    Bc = np.zeros((len(X), 3))
    for s in range(0, len(X), chunk):
        x = X[s : s + chunk]
        _, nn = tree.query(x, k=k)
        cand = table[nn].reshape(len(x), -1)  # (n, k * maxv)
        best = np.full(len(x), np.inf)
        bp = np.zeros_like(x)
        bf = np.zeros(len(x), np.int64)
        bb = np.zeros((len(x), 3))
        for j in range(cand.shape[1]):
            c = cand[:, j]
            ok = c >= 0
            cc = np.where(ok, c, 0)
            q, bary = _closest_on_tri(x, v[f[cc, 0]], v[f[cc, 1]], v[f[cc, 2]])
            d = np.sum((q - x) ** 2, 1)
            d[~ok] = np.inf
            better = d < best
            best[better] = d[better]
            bp[better] = q[better]
            bf[better] = cc[better]
            bb[better] = bary[better]
        P[s : s + chunk] = bp
        F[s : s + chunk] = bf
        Bc[s : s + chunk] = bb
    # the pseudonormal of the feature holding the closest point gives the side
    nz = Bc > 1e-12
    cnt = nz.sum(1)
    pn = fn[F].copy()
    e2 = cnt == 2  # on edge k = the corner not used, edges (k+1, k+2) -> index (k+1) % 3
    if e2.any():
        zero = np.argmin(nz[e2], 1)
        pn[e2] = en[F[e2], (zero + 1) % 3]
    e1 = cnt == 1
    if e1.any():
        corner = np.argmax(nz[e1], 1)
        pn[e1] = vn[f[F[e1], corner]]
    r = X - P
    rn = np.linalg.norm(r, axis=1)
    sgn = np.where(np.sum(r * pn, 1) >= 0, 1.0, -1.0)
    n = fn[F] * sgn[:, None] * np.sign(np.sum(fn[F] * pn, 1))[:, None]
    use = rn > 1e-9
    n[use] = (r[use] / rn[use, None]) * sgn[use, None]
    return P, F, n


# ------------------------------------------------------------------ the march, seeded per voxel


@njit
def _update_r(q, lab, rhs, nbr, offsets, weights, accepted, arrival, label, length, root,
              spacing, wv, wr, wo, ws, wi):
    """baillarger.fronts.march._update, also carrying the root position."""
    k = 0
    for i in range(6):
        r = weights[q, i]
        if r <= 0.0:
            continue
        best = _INF
        bs = 0.0
        sg = 0.0
        bi = -1
        for side in range(2):
            n = nbr[q, 2 * i + side]
            if n >= 0 and accepted[n] and label[n] == lab and arrival[n] < best:
                best = arrival[n]
                bs = length[n]
                bi = n
                sg = -1.0 if side == 0 else 1.0
        if best < _INF:
            j = k
            while j > 0 and wv[j - 1] > best:
                wv[j] = wv[j - 1]
                wr[j] = wr[j - 1]
                ws[j] = ws[j - 1]
                wi[j] = wi[j - 1]
                wo[j, 0] = wo[j - 1, 0]
                wo[j, 1] = wo[j - 1, 1]
                wo[j, 2] = wo[j - 1, 2]
                j -= 1
            wv[j] = best
            wr[j] = r
            ws[j] = bs
            wi[j] = bi
            wo[j, 0] = sg * offsets[q, i, 0]
            wo[j, 1] = sg * offsets[q, i, 1]
            wo[j, 2] = sg * offsets[q, i, 2]
            k += 1
    if k == 0:
        return _INF, np.nan, -1.0, -1.0, -1.0
    a = 0.0
    b = 0.0
    c = -rhs
    u = _INF
    used = 0
    for j in range(k):
        a += wr[j]
        b += wr[j] * wv[j]
        c += wr[j] * wv[j] * wv[j]
        disc = b * b - a * c
        if disc < 0.0:
            disc = 0.0
        u = (b + np.sqrt(disc)) / a
        used = j + 1
        if j + 1 == k or u <= wv[j + 1]:
            break
    fx = 0.0
    fy = 0.0
    fz = 0.0
    tot = 0.0
    s = 0.0
    rx = 0.0
    ry = 0.0
    rz = 0.0
    for j in range(used):
        g = wr[j] * (u - wv[j])
        fx += g * wo[j, 0]
        fy += g * wo[j, 1]
        fz += g * wo[j, 2]
        tot += g
        s += g * ws[j]
        rx += g * root[wi[j], 0]
        ry += g * root[wi[j], 1]
        rz += g * root[wi[j], 2]
    speed = np.sqrt((fx * spacing[0]) ** 2 + (fy * spacing[1]) ** 2 + (fz * spacing[2]) ** 2)
    if tot > 0.0:
        s = (s + speed) / tot
        rx /= tot
        ry /= tot
        rz /= tot
    else:
        s = ws[0]
        rx = root[wi[0], 0]
        ry = root[wi[0], 1]
        rz = root[wi[0], 2]
    return u, s, rx, ry, rz


@njit
def _march_band(nbr, count, rev, offsets, weights, seeds, seed_labels, seed_t, seed_s, seed_r,
                spacing, rhs):
    m = nbr.shape[0]
    arrival = np.full(m, _INF)
    label = np.full(m, -1, dtype=np.int64)
    length = np.full(m, np.nan)
    root = np.full((m, 3), np.nan)
    accepted = np.zeros(m, dtype=np.bool_)
    fixed = np.zeros(m, dtype=np.bool_)
    t1 = np.full(m, _INF)
    l1 = np.full(m, -1, dtype=np.int64)
    s1 = np.full(m, np.nan)
    r1 = np.full((m, 3), np.nan)
    cap = 13 * m + seeds.shape[0] + 16
    hv = np.empty(cap)
    hp = np.empty(cap, dtype=np.int64)
    hl = np.empty(cap, dtype=np.int64)
    size = 0
    wv = np.empty(6)
    wr = np.empty(6)
    wo = np.empty((6, 3))
    ws = np.empty(6)
    wi = np.empty(6, dtype=np.int64)
    for n in range(seeds.shape[0]):
        p = seeds[n]
        if seed_t[n] < t1[p]:
            t1[p] = seed_t[n]
            l1[p] = seed_labels[n]
            s1[p] = seed_s[n]
            r1[p, 0] = seed_r[n, 0]
            r1[p, 1] = seed_r[n, 1]
            r1[p, 2] = seed_r[n, 2]
            fixed[p] = True
            size = _heap_push(hv, hp, hl, size, seed_t[n], p, seed_labels[n])
    while size > 0:
        v = hv[0]
        p = hp[0]
        lab = hl[0]
        size = _heap_pop(hv, hp, hl, size)
        if accepted[p] or lab != l1[p] or v > t1[p]:
            continue
        accepted[p] = True
        arrival[p] = t1[p]
        label[p] = l1[p]
        length[p] = s1[p]
        root[p, 0] = r1[p, 0]
        root[p, 1] = r1[p, 1]
        root[p, 2] = r1[p, 2]
        for e in range(count[p], count[p + 1]):
            q = rev[e]
            if accepted[q] or fixed[q]:
                continue
            u, s, rx, ry, rz = _update_r(q, lab, rhs[lab], nbr, offsets, weights, accepted, arrival,
                                         label, length, root, spacing, wv, wr, wo, ws, wi)
            if u == _INF:
                continue
            if u < t1[q]:
                t1[q] = u
                l1[q] = lab
                s1[q] = s
                r1[q, 0] = rx
                r1[q, 1] = ry
                r1[q, 2] = rz
                size = _heap_push(hv, hp, hl, size, u, q, lab)
    return arrival, label, length, root


@dataclass
class Band:
    arrival: Float
    label: Int
    length: Float
    root: Float  # (..., 3) root position, in the coordinates the seeds gave


def march_band(metric, mask: Bool, seeds: Bool, seed_t, seed_s, seed_r, seed_lab=None,
               spacing=(1.0, 1.0, 1.0)) -> Band:
    """Fast marching (the package's scheme) from frozen seed voxels with their own
    arrival, arc length and root. seeds: boolean, in the mask; seed_* in
    np.flatnonzero(seeds) order. Labels compete as in march (rhs 1)."""
    shape = mask.shape
    h = np.asarray(spacing, float)
    met = np.asarray(metric, float)
    cells = np.flatnonzero(mask.ravel()).astype(np.int64)
    ids = np.full(mask.size, -1, np.int64)
    ids[cells] = np.arange(cells.size)
    scale = h[:, None] * h[None, :]
    if met.shape == (3, 3):
        off1, w1 = selling(np.linalg.inv(met) / scale)
        off = np.ascontiguousarray(np.broadcast_to(off1, (cells.size, 6, 3)))
        w = np.ascontiguousarray(np.broadcast_to(w1, (cells.size, 6)))
    else:
        off, w = selling(np.linalg.inv(met.reshape(-1, 3, 3)[cells]) / scale)
    nbr = _stencils(mask, off, ids, cells)
    count, rev = _reverse(nbr)
    sc = np.flatnonzero(seeds.ravel())
    assert np.all(mask.ravel()[sc]), "a seed lies outside the mask"
    lab = np.zeros(sc.size, np.int64) if seed_lab is None else np.asarray(seed_lab, np.int64)
    nl = int(lab.max()) + 1 if lab.size else 1
    a, lb, ln, rt = _march_band(nbr, count, rev, off, w, ids[sc], lab, np.asarray(seed_t, float),
                                np.asarray(seed_s, float), np.ascontiguousarray(seed_r, float), h,
                                np.ones(nl))

    def full(x, fill):
        out = np.full((mask.size,) + x.shape[1:], fill, dtype=x.dtype)
        out[cells] = x
        return out.reshape(shape + x.shape[1:])

    return Band(full(a, np.inf), full(lb, -1), full(ln, np.nan), full(rt, np.nan))


def band_seeds(points: Float, P: Float, n: Float, D: Float | None):
    """Seed values at points (N, 3) from the closest surface points P and unit
    normals n (outward): arrival (signed metric distance from the tangent plane),
    arc length and root (the metric foot point). D: dual metric (N, 3, 3) or None
    (isotropic)."""
    s = np.sum((points - P) * n, 1)
    if D is None:
        Dn = n
        nDn = np.ones(len(points))
    else:
        Dn = np.einsum("nij,nj->ni", D, n)
        nDn = np.sum(n * Dn, 1)
    t = s / np.sqrt(nDn)
    foot = points - (s / nDn)[:, None] * Dn
    return t, np.linalg.norm(points - foot, axis=1) * np.sign(s), foot


def surface_seeds(ctx: Bool, wm: Bool, X: Float, closest, D: Float | None = None,
                  band: float = 1.0, scale: int = 1):
    """Seeds for a surface start on a grid of spacing 1/scale: cortex voxels within
    `band` (1x voxels) of the surface, and every cortex voxel 6-adjacent to the WM
    label. X: the grid's points (..., 3) in 1x voxel units. closest(points) returns
    (closest surface points, outward unit normals). D: dual metric over the grid
    (..., 3, 3) or None. Returns (seed mask, arrival, arc length, root)."""
    adj = ndi.binary_dilation(wm, ndi.generate_binary_structure(3, 1)) & ctx
    near = ctx & (ndi.distance_transform_edt(~wm) / scale <= band + 1.5)
    idx = np.flatnonzero(near.ravel())
    pts = X.reshape(-1, 3)[idx]
    P, n = closest(pts)
    Dd = None if D is None else D.reshape(-1, 3, 3)[idx]
    t, s, foot = band_seeds(pts, P, n, Dd)
    keep = (np.abs(np.sum((pts - P) * n, 1)) <= band) | adj.ravel()[idx]
    seeds = np.zeros(ctx.shape, bool)
    seeds.ravel()[idx[keep]] = True
    return seeds, t[keep], s[keep], foot[keep]


def mesh_closest(v: Float, f: Int):
    def closest(pts):
        P, _, n = closest_on_mesh(v, f, pts)
        return P, n
    return closest


def truth_closest(ph: Phantom):
    def closest(pts):
        tr = ph.truth(pts)
        return tr.foot, tr.n
    return closest


# ------------------------------------------------------------------ implicit and voxel-face starts


def implicit_closest(wm: Bool, sigma: float, iters: int = 40):
    """Closest points on the zero set of phi = sigma probit(label smoothed by sigma)
    (the 0.5 level of the smoothed label), phi the tricubic (cubic B-spline)
    interpolant of the grid values, by Chopp's
    (2001) iteration, with d2 halved: x <- x + d1 + d2 / 2, d1 = -phi grad / |grad|^2 (onto the zero
    set), d2 = (x0 - x) minus its component along grad (toward the point's own
    foot). No mesh. Returns closest(points) -> (feet, unit normal of the point's
    side: (x0 - foot)/|x0 - foot| signed + outside the WM)."""
    from scipy.special import ndtri

    F = ndi.gaussian_filter(wm.astype(np.float64), sigma, truncate=6.0)
    # the smoothed label is a sigmoid across the boundary (width about sigma), on
    # which Newton steps overshoot from more than about a width away; sigma *
    # probit(F) is exactly the signed distance for a blurred plane, with the same
    # zero set and a gradient near 1 across the band
    L = sigma * ndtri(np.clip(F, 1e-9, 1 - 1e-9))
    coef = ndi.spline_filter(L, order=3)
    h = 1e-4

    def phi(p):
        return ndi.map_coordinates(coef, p.T, order=3, prefilter=False, mode="nearest")

    def grad(p):
        g = np.zeros_like(p)
        for a in range(3):
            e = np.zeros(3)
            e[a] = h
            g[:, a] = (phi(p + e) - phi(p - e)) / (2 * h)
        return g

    def iterate(x0, x):
        for _ in range(iters):
            f, g = phi(x), grad(x)
            g2 = np.maximum(np.sum(g * g, 1), 1e-12)
            d1 = -(f / g2)[:, None] * g
            r = x0 - x
            d2 = r - (np.sum(r * g, 1) / g2)[:, None] * g
            # half of d2: at points farther than the zero set's local radius of
            # curvature (staircase wiggles of a lightly smoothed label) the full
            # step cycles (22 % of a 2-voxel band at sigma 0.5 stalled)
            step = d1 + 0.5 * d2
            # keep a step within a voxel (Newton from far points can overshoot)
            sn = np.linalg.norm(step, axis=1, keepdims=True)
            x = x + step * np.minimum(1.0, 1.0 / np.maximum(sn, 1e-12))
        return x

    def closest(pts):
        x0 = np.asarray(pts, float)
        x = iterate(x0, x0.copy())
        # where phi is flat (a lightly smoothed label beyond about a voxel) the
        # iteration stalls; restart those points from the feet of their nearest
        # converged points (Chopp's start is for points next to the interface,
        # the rest is an extension)
        for _ in range(3):
            bad = np.abs(phi(x)) > 1e-6
            if not bad.any() or bad.all():
                break
            _, j = cKDTree(x0[~bad]).query(x0[bad], k=4)
            cand = x[~bad][j]  # (nb, 4, 3)
            best = np.argmin(np.linalg.norm(cand - x0[bad][:, None], axis=2), 1)
            x[bad] = iterate(x0[bad], cand[np.arange(bad.sum()), best])
        side = np.where(phi(x0) > 0, -1.0, 1.0)  # phi > 0 inside the WM
        r = x0 - x
        rn = np.linalg.norm(r, axis=1)
        g = grad(x)
        n = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)  # outward
        use = rn > 1e-9
        n[use] = r[use] / rn[use, None] * side[use, None]
        return x, n

    return closest


def voxel_face_mesh(wm: Bool) -> tuple[Float, Int]:
    """The exposed faces of the WM label (the 'Lego' surface), two triangles each,
    welded, oriented outward. Box faces are not exposed."""
    quads = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for sgn in (-1, 1):
            nb = np.ones_like(wm)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if sgn > 0:
                src[a], dst[a] = slice(1, None), slice(0, -1)
            else:
                src[a], dst[a] = slice(0, -1), slice(1, None)
            nb[tuple(dst)] = wm[tuple(src)]
            v = np.argwhere(wm & ~nb).astype(float)
            ctr = v.copy()
            ctr[:, a] += 0.5 * sgn
            eb, ec = np.zeros(3), np.zeros(3)
            eb[b], ec[c] = 0.5, 0.5
            q = np.stack([ctr - eb - ec, ctr + eb - ec, ctr + eb + ec, ctr - eb + ec], 1)
            if sgn < 0:
                q = q[:, ::-1]
            quads.append(q)
    Q = np.concatenate(quads)
    tri = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    key = np.round(tri.reshape(-1, 3) * 2).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    return uk / 2.0, inv.reshape(-1, 3)


def aspect_ratios(v: Float, f: Int) -> Float:
    """Longest edge x perimeter / (4 sqrt 3 area): 1 for an equilateral triangle."""
    P = v[f]
    e = np.stack([np.linalg.norm(P[:, (k + 1) % 3] - P[:, k], axis=1) for k in range(3)], 1)
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    return e.max(1) * e.sum(1) / (4 * np.sqrt(3) * np.maximum(area, 1e-30))


def dual_areas(v: Float, f: Int) -> Float:
    """Barycentric dual-cell area per vertex (a third of each incident triangle)."""
    P = v[f]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    out = np.zeros(len(v))
    for k in range(3):
        np.add.at(out, f[:, k], area / 3)
    return out
