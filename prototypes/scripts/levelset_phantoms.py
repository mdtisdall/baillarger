"""Phantoms with an analytic white surface, for levelset_fill (test plan §5). Voxels; all 3D.

Extruded phantoms (along y) give the WM's cross-section in (x, z) as a function g (> 0 in WM, 0 on
the white surface), contoured on a fine grid (0.05 voxel) to a polyline. Signed distance phi to that
polyline (> 0 outside the WM), the foot point and the outward normal follow from the nearest polyline
point. A WM cross-section given only as voxels is the union of the voxel squares (nearest upsampling,
so the contour sits on the voxel faces to 0.025 voxel). The sphere is analytic.

The known normal (the phantoms' 'data normal') is the unit vector from the foot point: the gradient of
the distance to the white surface.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import find_contours

type Bool = npt.NDArray[np.bool_]
type Float = npt.NDArray[np.float64]

FINE = 0.05
BAND_WM = 3.0  # band depth into the WM (voxels): at least the update's look-back, h <= sqrt 3 plus the
# trilinear cell (test plan §3 step 1; test_band_depth_tilted_slab)


@dataclass
class Phantom:
    name: str
    wm: Bool
    cortex: Bool
    fluid: Bool
    band: Bool
    phi: Float  # signed distance to the white surface, > 0 outside the WM (everywhere)
    foot: Float  # (..., 3)
    normal: Float  # (..., 3) outward surface normal at the foot point
    n: Float  # (..., 3) the known layer normal (unsigned use)
    extra: dict


def contour_g(g, nx: int, nz: int, step: float = FINE) -> tuple[Float, Float]:
    """Points (m, 2) in (x, z) on g = 0, from a fine grid over the box, and the outward unit normal at
    each (perpendicular to the polyline, pointing to g < 0). The normal is needed from the curve itself:
    at a voxel lying on the surface, x - foot has no direction."""
    # offset by half a step, so that voxel faces (x.5) fall midway between samples, where the contour of
    # a voxel-square union is placed
    xs = np.arange(-0.5 + step / 2, nx - 0.5, step)
    zs = np.arange(-0.5 + step / 2, nz - 0.5, step)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    G = g(X, Z)
    pts, nrm, cid = [], [], []
    for ci, c in enumerate(find_contours(G, 0.0)):
        p = np.c_[xs[0] + c[:, 0] * step, zs[0] + c[:, 1] * step]
        # repeated points (the curve through a grid sample) would give a zero tangent
        keep = np.r_[True, np.any(np.abs(np.diff(p, axis=0)) > 1e-9, axis=1)]
        p = p[keep]
        if len(p) < 3:
            continue
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        nv = np.c_[t[:, 1], -t[:, 0]]
        probe = p + 2 * step * nv
        flip = g(probe[:, 0], probe[:, 1]) > 0
        nv[flip] *= -1
        pts.append(p)
        nrm.append(nv)
        cid.append(np.full(len(p), ci))
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(cid)


def project(curve: Float, cid, pts: Float) -> tuple[Float, Float, Float]:
    """Nearest point on the polyline (projected onto the two segments next to the nearest sample, so
    the foot is exact to the polyline, not to its 0.05-voxel sampling). Returns (dist, foot, index of
    the nearest sample)."""
    tree = cKDTree(curve)
    _, j = tree.query(pts)
    best = np.linalg.norm(pts - curve[j], axis=1)
    foot = curve[j].copy()
    for o in (-1, 1):
        k = j + o
        ok = (k >= 0) & (k < len(curve))
        k = np.clip(k, 0, len(curve) - 1)
        ok &= cid[k] == cid[j]
        a, b = curve[j], curve[k]
        ab = b - a
        t = np.clip(np.sum((pts - a) * ab, 1) / np.maximum(np.sum(ab * ab, 1), 1e-18), 0, 1)
        q = a + t[:, None] * ab
        dq = np.linalg.norm(pts - q, axis=1)
        better = ok & (dq < best)
        best[better] = dq[better]
        foot[better] = q[better]
    return best, foot, j


def g_from_mask2d(m: Bool):
    """g for the union of the voxel squares of a 2D mask (x, z): +1 inside, -1 outside, nearest."""
    nx, nz = m.shape

    def g(X, Z):
        i = np.clip(np.round(X).astype(int), 0, nx - 1)
        k = np.clip(np.round(Z).astype(int), 0, nz - 1)
        return np.where(m[i, k], 1.0, -1.0)

    return g


def extruded(name: str, g, ny: int, nx: int, nz: int, cortex2d: Bool, fluid2d: Bool, extra=None,
             band_wm: float = BAND_WM) -> Phantom:
    """Phantom from g (> 0 in WM) on an (nx, nz) cross-section extruded along y (ny voxels)."""
    X, Z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    wm2 = g(X.astype(float), Z.astype(float)) > 0
    curve, cn, cid = contour_g(g, nx, nz)
    pts = np.c_[X.ravel(), Z.ravel()].astype(float)
    dist, fp, j = project(curve, cid, pts)
    sgn = np.where(wm2.ravel(), -1.0, 1.0)
    ext = lambda a: np.broadcast_to(a[:, None], (nx, ny) + a.shape[1:]).copy()  # noqa: E731
    wm = ext(wm2)
    c3 = cortex2d if cortex2d.ndim == 3 else ext(cortex2d)  # 3D masks pass through
    f3 = fluid2d if fluid2d.ndim == 3 else ext(fluid2d)
    cortex = c3 & ~wm
    fluid = f3 & ~wm & ~cortex
    phi = np.zeros((nx, ny, nz))
    foot = np.zeros((nx, ny, nz, 3))
    normal = np.zeros((nx, ny, nz, 3))
    tree = cKDTree(curve)
    done: dict = {}
    for y in range(ny):
        fl = fluid[:, y, :]
        key = fl.tobytes()
        if key not in done:
            done[key] = visible_foot(pts, sgn, dist, fp, j, curve, cn, tree, fl)
        d_y, fp_y, n_y = done[key]
        phi[:, y, :] = (sgn * d_y).reshape(nx, nz)
        foot[:, y, :, 0] = fp_y[:, 0].reshape(nx, nz)
        foot[:, y, :, 2] = fp_y[:, 1].reshape(nx, nz)
        foot[:, y, :, 1] = y
        nv = n_y
        normal[:, y, :, 0] = nv[:, 0].reshape(nx, nz)
        normal[:, y, :, 2] = nv[:, 1].reshape(nx, nz)
    return finish(name, wm, cortex, fluid, phi, foot, normal, extra or {}, band_wm)


def visible_foot(pts, sgn, dist, fp, j, curve, cn, tree, fluid2: Bool):
    """Distance, foot point and outward normal, with the foot visible from the point without crossing
    fluid: a layer normal does not come from WM on the far side of a fluid gap. Where the nearest point
    is hidden, the nearest visible curve sample is taken (all samples searched: a nearer limit missed
    the far bank's own plate at P8, test_apposed_seam). The normal is (x - foot) signed outward, or the curve's own normal on the surface itself."""
    nx, nz = fluid2.shape
    t = np.linspace(0.05, 0.95, 19)

    def blocked(a, b):
        q = a[:, None, :] + t[None, :, None] * (b - a)[:, None, :]
        qi = np.clip(np.round(q[..., 0]).astype(int), 0, nx - 1)
        qk = np.clip(np.round(q[..., 1]).astype(int), 0, nz - 1)
        return fluid2[qi, qk].any(1)

    d_out, fp_out = dist.copy(), fp.copy()
    cand = np.flatnonzero(dist > 0.75)
    hidden = np.zeros(len(pts), bool)
    if cand.size:
        hidden[cand] = blocked(pts[cand], fp[cand])
    for i in np.flatnonzero(hidden):
        dd = np.linalg.norm(curve - pts[i], axis=1)
        jj = np.argsort(dd)
        dd = dd[jj]
        ok = np.zeros(len(jj), bool)
        for c0 in range(0, len(jj), 2000):  # nearest first, stop at the first visible chunk
            sl = slice(c0, c0 + 2000)
            ok[sl] = ~blocked(np.repeat(pts[i][None], len(jj[sl]), 0), curve[jj[sl]])
            if ok[sl].any():
                break
        if ok.any():
            m = int(np.argmax(ok))
            d_out[i], fp_out[i] = dd[m], curve[jj[m]]
    v = (pts - fp_out) * sgn[:, None]
    lv = np.linalg.norm(v, axis=1, keepdims=True)
    n_out = np.where(lv >= 0.05, v / np.maximum(lv, 1e-12), cn[j])
    return d_out, fp_out, n_out


def finish(name, wm, cortex, fluid, phi, foot, normal, extra, band_wm: float = BAND_WM) -> Phantom:
    near = ndi.binary_dilation(cortex, np.ones((3, 3, 3)), iterations=int(np.ceil(band_wm)))
    band = (wm & near & (phi >= -band_wm)) | (cortex & (phi < 1.0))
    return Phantom(name, wm, cortex, fluid, band, phi, foot, normal, normal.copy(), extra)


# ------------------------------------------------------------------ shapes


def slab(t: int = 12, nx: int = 16, ny: int = 8, nz: int = 34, z0: int = 10) -> Phantom:
    """WM z < z0, cortex z0 <= z < z0 + t, fluid above. White surface z = z0 - 0.5."""
    g = lambda X, Z: (z0 - 0.5) - Z  # noqa: E731
    Z = np.arange(nz)[None, :]
    c2 = np.broadcast_to((Z >= z0) & (Z < z0 + t), (nx, nz))
    f2 = np.broadcast_to(Z >= z0 + t, (nx, nz))
    return extruded("slab", g, ny, nx, nz, c2, f2, dict(t=t, z0=z0 - 0.5))


def cylinder(r0: float = 10.0, r1: float = 26.0, ny: int = 8) -> Phantom:
    """WM r < r0, cortex r0 <= r < r1, fluid beyond (axis along y)."""
    m = int(np.ceil(r1)) + 3
    nx = nz = 2 * m + 1
    c = float(m)
    g = lambda X, Z: r0 - np.hypot(X - c, Z - c)  # noqa: E731
    X, Z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    r = np.hypot(X - c, Z - c)
    return extruded("cylinder", g, ny, nx, nz, (r >= r0) & (r < r1), r >= r1, dict(r0=r0, r1=r1, c=c))


def sphere(r0: float = 10.0, r1: float = 26.0) -> Phantom:
    m = int(np.ceil(r1)) + 3
    n = 2 * m + 1
    c = float(m)
    X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    P = np.stack([X, Y, Z], -1).astype(float) - c
    r = np.linalg.norm(P, axis=-1)
    u = P / np.maximum(r, 1e-12)[..., None]
    wm = r < r0
    cortex = (r >= r0) & (r < r1)
    fluid = r >= r1
    foot = c + r0 * u
    return finish("sphere", wm, cortex, fluid, r - r0, foot, u, dict(r0=r0, r1=r1, c=c))


def from_rim(name: str, rim, extra=None) -> Phantom:
    """A baillarger.phantoms rim (1 outside, 2 WM, 3 cortex) whose WM is extruded along y: the white
    surface is the boundary of the WM's voxel squares in (x, z)."""
    wm2 = rim[:, 0, :] == 2
    assert np.all((rim == 2) == wm2[:, None, :])
    nx, ny, nz = rim.shape
    return extruded(name, g_from_mask2d(wm2), ny, nx, nz, rim == 3, rim == 1, extra)


def p4() -> Phantom:
    from baillarger import phantoms

    rim, _, e = phantoms.p4_kissing_dark_line()
    return from_rim("P4", rim, dict(known=25.5, region=e["patch_mask"].any(0), split=26))


def p8() -> Phantom:
    from baillarger import phantoms

    rim, _, e, known, _ = phantoms.p8_natural_contact()
    return from_rim("P8", rim, dict(known=float(known), region=e["patch_mask"].any(0), split=known))


def wedge(theta_deg: float, mu_a: float = 12.0, mu_b: float = 18.0, ny: int = 6, z0: float = 5.0) -> Phantom:
    """Two banks on the walls of a V (opening angle theta, apex at (cx, z0), opening upward), cortex
    where min(d_A / mu_A, d_B / mu_B) <= 1, fluid in the rest of the V, WM outside. The exact seam is
    the plane d_A / mu_A = d_B / mu_B through the apex (constant normals on each wall). extra holds both
    walls' inward normals and the exact distances; ph.n is version B (the nearer wall's normal: the
    distance gradient, flipping at the bisector)."""
    h = np.radians(theta_deg) / 2
    nA = np.array([np.cos(h), np.sin(h)])
    nB = np.array([-np.cos(h), np.sin(h)])
    aa = np.arctan2(np.sin(2 * h), mu_b / mu_a + np.cos(2 * h))  # seam angle from wall A
    L = mu_a / np.sin(aa)  # apex to where the seam reaches tau = 1
    nz = int(z0 + 1.15 * L + 6)
    half = (nz - z0) * np.tan(h) + max(mu_a, mu_b) / np.cos(h) + 4
    nx = int(2 * half) + 1
    cx = nx // 2 + 0.25  # off the voxel grid, so the bisector is not a voxel plane
    X, Z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")

    def dists(X, Z):
        px, pz = X - cx, Z - z0
        return px * nA[0] + pz * nA[1], px * nB[0] + pz * nB[1]

    def g(X, Z):
        dA, dB = dists(X, Z)
        return -np.minimum(dA, dB)

    dA, dB = dists(X.astype(float), Z.astype(float))
    inside = (dA > 0) & (dB > 0)
    ctx = inside & (np.minimum(dA / mu_a, dB / mu_b) <= 1)
    flu = inside & ~ctx
    ph = extruded(f"wedge{int(theta_deg)}", g, ny, nx, nz, ctx, flu,
                  dict(theta=theta_deg, mu_a=mu_a, mu_b=mu_b, cx=cx, z0=z0, nA=nA, nB=nB, L=L))
    ext = lambda a: np.broadcast_to(a[:, None], (nx, ny, nz)).copy()  # noqa: E731
    ph.extra["dA"], ph.extra["dB"] = ext(dA), ext(dB)
    return ph


def wedge_normals(ph: Phantom, version: str) -> tuple[Float, Float]:
    """(n, coherence) for a wedge: A, each bank's own wall normal up to the exact tau seam; B, the
    nearer wall's normal (flips at the bisector); C, B with coherence 0 within 1 voxel of the bisector
    (a band about 2 voxels wide), so the march's blend to the front's own normal takes over there."""
    e = ph.extra
    nA = np.array([e["nA"][0], 0.0, e["nA"][1]])
    nB = np.array([e["nB"][0], 0.0, e["nB"][1]])
    dA, dB = e["dA"], e["dB"]
    coh = np.ones(ph.wm.shape)
    if version == "A":
        own_a = dA / e["mu_a"] <= dB / e["mu_b"]
    else:
        own_a = dA <= dB
    n = np.where(own_a[..., None], nA, nB)
    if version == "C":
        X = np.arange(ph.wm.shape[0])[:, None, None] * np.ones(ph.wm.shape)
        coh[np.abs(X - e["cx"]) <= 1.0] = 0.0
    return n, coh


def u_slab(R: float, t: float = 16.0, ny: int = 8) -> Phantom:
    """synth_contacts.u_slab with the analytic white surface: a V-shaped sulcus (half-angle 25 deg, apex
    zo) with a rounded fundus of radius R, lined by cortex of thickness t (analytic distance), fluid
    inside; WM below ztop outside the cavity, cortex also over the WM's top."""
    nx, nz = 100, 112
    cx, alpha, ztop = 50.0, np.radians(25), 94
    zo = 20.0 if R < t else 30.0

    def wd(X, Z):
        px, pz = X - cx, Z - zo
        ins = (pz >= 0) & (np.abs(px) <= pz * np.tan(alpha))
        best = np.full(np.shape(px), np.inf)
        for sgn in (-1, 1):
            d = np.array([sgn * np.sin(alpha), np.cos(alpha)])
            s = np.maximum(px * d[0] + pz * d[1], 0)
            best = np.minimum(best, np.hypot(px - s * d[0], pz - s * d[1]))
        return np.where(ins, 0.0, best)

    g = lambda X, Z: np.minimum(wd(X, Z) - R, (ztop - 0.5) - Z)  # noqa: E731
    ph0 = extruded("tmp", g, 1, nx, nz, np.zeros((nx, nz), bool), np.zeros((nx, nz), bool))
    phi2 = ph0.phi[:, 0, :]
    wm2 = ph0.wm[:, 0, :]
    ctx = ~wm2 & (phi2 <= t)
    hmax = (t - R) / np.sin(alpha) if R < t else 0.0
    return extruded(f"uslab_R{int(R)}", g, ny, nx, nz, ctx, ~wm2 & ~ctx,
                    dict(cx=cx, zo=zo, zmax=zo + hmax, R=R, t=t))


def omega(tl: float = 12.0, tr: float = 18.0, ny: int = 8) -> Phantom:
    """synth_contacts.omega with the analytic white surface: a round chamber (radius 34) with a neck
    (30 wide) to the surface; cortex tl on the left (foot x < cx), tr on the right, the neck filled, so
    the two sides touch at their natural thicknesses at x = cx - 3.5."""
    nx, nz = 110, 116
    cx, zc, rc, ztop = 55.0, 45.0, 34.0, 92
    x0, x1 = cx - 15 - 0.5, cx + 15 - 0.5  # neck faces

    def g(X, Z):
        ch = np.hypot(X - cx, Z - zc) - rc
        neck = np.maximum(np.maximum(x0 - X, X - x1), zc - Z)
        return np.minimum(np.minimum(ch, neck), (ztop - 0.5) - Z)

    # as synth_contacts.omega: within tl of the left WM or within tr of the right WM (the neck then
    # fills exactly, the two sides touching at x = cx - 3.5)
    X, Z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    wm2 = g(X.astype(float), Z.astype(float)) > 0
    curve, _, _ = contour_g(g, nx, nz)
    pts = np.c_[X.ravel(), Z.ravel()].astype(float)
    dl = cKDTree(curve[curve[:, 0] < cx]).query(pts)[0].reshape(nx, nz)
    dr = cKDTree(curve[curve[:, 0] >= cx]).query(pts)[0].reshape(nx, nz)
    ctx = ~wm2 & ((dl <= tl + 1e-6) | (dr <= tr + 1e-6))
    zneck = zc + np.sqrt(rc**2 - 15.0**2)
    zz = np.arange(nz)
    region = np.broadcast_to(((zz > zneck + 1) & (zz < ztop))[None, :], (ny, nz))
    return extruded("omega", g, ny, nx, nz, ctx, ~wm2 & ~ctx,
                    dict(cx=cx, known=cx - 3.5, region=region, tl=tl, tr=tr))


def blade(t: float = 12.0, ny: int = 8) -> Phantom:
    """A WM blade 2 voxels thick (voxels cx - 1, cx; faces cx - 1.5 and cx + 0.5), from the bottom of
    the box to ztip, with cortex t on both sides and over its tip; fluid beyond."""
    nx, nz = 2 * int(t) + 12, 60
    cx, ztip = nx // 2, 40
    g = lambda X, Z: np.minimum(np.minimum(X - (cx - 1.5), (cx + 0.5) - X), (ztip - 0.5) - Z)  # noqa: E731
    ph0 = extruded("tmp", g, 1, nx, nz, np.zeros((nx, nz), bool), np.zeros((nx, nz), bool))
    phi2, wm2 = ph0.phi[:, 0, :], ph0.wm[:, 0, :]
    ctx = ~wm2 & (phi2 <= t)
    return extruded("blade", g, ny, nx, nz, ctx, ~wm2 & ~ctx, dict(cx=cx, ztip=ztip, mid=cx - 0.5))


def tilted_slab(deg: float = 30.0, t: float = 12.0, band_wm: float = BAND_WM, ny: int = 6) -> Phantom:
    """A flat white surface tilted deg about y: WM below the plane through (nx/2, z0) with normal
    (-sin, cos), cortex 0 < phi <= t above it, fluid beyond."""
    nx, nz, z0 = 60, 50, 18.3
    a = np.radians(deg)
    nrm = np.array([-np.sin(a), np.cos(a)])
    cx = nx / 2

    def g(X, Z):
        return -((X - cx) * nrm[0] + (Z - z0) * nrm[1])

    X, Z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    d = -g(X.astype(float), Z.astype(float))
    return extruded(f"tilted{int(deg)}", g, ny, nx, nz, (d > 0) & (d <= t), d > t,
                    dict(nrm=nrm, cx=cx, z0=z0, t=t), band_wm=band_wm)


def plane3d(normal=(1.0, 1.0, 1.0), t: float = 12.0, band_wm: float = BAND_WM, n: int = 40) -> Phantom:
    """A flat white surface with any normal (analytic; not extruded): WM where (x - c).nu < 0, cortex
    0 < phi <= t, fluid beyond."""
    nu = np.asarray(normal, float)
    nu /= np.linalg.norm(nu)
    c = np.array([n / 2 + 0.31, n / 2 - 0.17, n / 2 - 6.4])
    X = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).astype(float)
    phi = (X - c) @ nu
    wm, cortex, fluid = phi < 0, (phi >= 0) & (phi <= t), phi > t
    foot = X - phi[..., None] * nu
    normal_ = np.broadcast_to(nu, X.shape).copy()
    return finish("plane3d", wm, cortex, fluid, phi, foot, normal_, dict(nu=nu, c=c), band_wm)
