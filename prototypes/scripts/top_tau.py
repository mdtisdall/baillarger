"""τ behind a fixed top surface Σ (T3 of docs/plans/top-surface.md), and the
step direction ∇τ for the Σ start (user, 2026-11-19: "Step along ∇τ").

Between two start steps, the τ solve of global_tau_solver runs on the voxel
centres between the white surface and Σ, with τ = 0 where an axis crosses Σ
(a ghost value at θh, 0 < θ ≤ 1, from the voxel centre), a warm start from
the last τ and at most LM_ITERS Levenberg-Marquardt iterations. The energy
is that of global_tau_energy without the stop term (Σ is fixed), with the
thickness prior kept (user, 2026-11-19). The direction of a vertex of Σ is
the mean of ∇τ over the τ voxels inside Σ within one voxel of it, weighted
by 1 - distance (voxels); where none is near, the caller uses the mesh
normal (user, 2026-11-19).

Inside Σ (user, 2026-11-19: "Signed distance"): the signed distance s
from each voxel centre within BAND_VOX voxels of Σ to its closest point on
Σ, with the sign from the pseudonormal of the face, edge or vertex that
the closest point lies on; inside where s < 0. Each connected part of the
voxels off the band takes the side of the band voxels next to it. s, and
so the stop term of top_solve, is a continuous function of Σ; a voxel
enters or leaves the domain only where s = 0. (The earlier test, a
majority of votes along the axis lines, flipped voxels far from Σ for
moves of 0.0005 h; its disagreement with the new test is counted.) The
domain is the voxels inside Σ that are outside the white surface
(site.inside) or are inner voxels of global_tau_solver.make_domain; every
domain voxel gets the ghost where an axis crosses Σ, and a connected part
with no ghost is left out (its τ has no τ = 0 to fix it).
"""

from __future__ import annotations

import time
from typing import Any

import global_tau_energy as GE
import global_tau_prep as TP
import global_tau_solver as GS
import jax.numpy as jnp
import numba as nb
import numpy as np
from scipy import ndimage as ndi

LM_ITERS = 3  # user, 2026-11-19: "A few LM iterations"
THETA_MIN = 0.05  # a voxel centre on Σ: the ghost at 0.05 h, not at 0
EPS_B, EPS_C = 1.37e-7, 2.91e-7  # voxels: the offset of the grid lines (no exact edge hits)


@nb.njit(cache=True)
def _axis_crossings(
    P: np.ndarray, F: np.ndarray, a: int, shape: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The crossings of the grid lines along axis a (through the voxel
    centres, integer coordinates on the other two axes) with the faces of
    the mesh (P in voxel coordinates): line coordinates (u, v), the position
    t on the axis, the sign of the face normal's component on the axis, and
    the face."""
    b, c = (a + 1) % 3, (a + 2) % 3
    cap = 16 * len(F) + 16
    U = np.empty(cap, np.int64)
    W = np.empty(cap, np.int64)
    T = np.empty(cap, np.float64)
    S = np.empty(cap, np.int64)
    Fx = np.empty(cap, np.int64)
    n = 0
    for f in range(len(F)):
        p0, p1, p2 = P[F[f, 0]], P[F[f, 1]], P[F[f, 2]]
        nx = (p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1])
        ny = (p1[2] - p0[2]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[2] - p0[2])
        nz = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        nrm = (nx, ny, nz)
        na = nrm[a]
        if na == 0.0:
            continue
        lo_b = int(np.ceil(min(p0[b], p1[b], p2[b]) - EPS_B))
        hi_b = int(np.floor(max(p0[b], p1[b], p2[b]) - EPS_B))
        lo_c = int(np.ceil(min(p0[c], p1[c], p2[c]) - EPS_C))
        hi_c = int(np.floor(max(p0[c], p1[c], p2[c]) - EPS_C))
        for u in range(max(lo_b, 0), min(hi_b, shape[b] - 1) + 1):
            for w in range(max(lo_c, 0), min(hi_c, shape[c] - 1) + 1):
                # the line is moved by a tiny offset (EPS_B, EPS_C), so it
                # never passes exactly through an edge or a vertex of Σ and
                # each crossing counts once
                ub, wc = u + EPS_B, w + EPS_C
                x0, y0 = p0[b], p0[c]
                x1, y1 = p1[b], p1[c]
                x2, y2 = p2[b], p2[c]
                den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if den == 0.0:
                    continue
                l0 = ((y1 - y2) * (ub - x2) + (x2 - x1) * (wc - y2)) / den
                l1 = ((y2 - y0) * (ub - x2) + (x0 - x2) * (wc - y2)) / den
                l2 = 1.0 - l0 - l1
                if l0 < 0 or l1 < 0 or l2 < 0:
                    continue
                if n == cap:
                    raise ValueError("_axis_crossings: buffer full")
                U[n] = u
                W[n] = w
                T[n] = l0 * p0[a] + l1 * p1[a] + l2 * p2[a]
                S[n] = 1 if na > 0 else -1
                Fx[n] = f
                n += 1
    return U[:n], W[:n], T[:n], S[:n], Fx[:n]


@nb.njit(cache=True)
def _line_votes(
    U: np.ndarray, W: np.ndarray, T: np.ndarray, S: np.ndarray, Fx: np.ndarray,
    a: int, shape: np.ndarray, vote: np.ndarray, voted: np.ndarray,
    ghost: np.ndarray, gdist: np.ndarray, gface: np.ndarray,
) -> None:
    """Per voxel on each line with crossings: an inside vote (next crossing
    above leaves, or none above and the last below enters), and the ghost
    sides: a crossing between the centre and the neighbour's centre, with
    θ = its distance from the centre (the nearest crossing on that side)
    and its face (gface)."""
    b, c = (a + 1) % 3, (a + 2) % 3
    # sort by line, then by t
    key = (U * shape[c] + W).astype(np.float64) * (4.0 * shape[a] + 8.0) + (T + 2.0)
    order = np.argsort(key, kind="mergesort")
    i = 0
    N = len(order)
    idx = np.zeros(3, np.int64)
    while i < N:
        j = i
        line = U[order[i]] * shape[c] + W[order[i]]
        while j < N and U[order[j]] * shape[c] + W[order[j]] == line:
            j += 1
        u, w = U[order[i]], W[order[i]]
        ts = T[order[i:j]]
        ss = S[order[i:j]]
        fs = Fx[order[i:j]]
        m = j - i
        for k in range(shape[a]):
            # next crossing above k
            q = 0
            while q < m and ts[q] <= k:
                q += 1
            inside = False
            if q < m:
                inside = ss[q] > 0
            else:
                inside = ss[m - 1] < 0
            idx[a] = k
            idx[b] = u
            idx[c] = w
            vote[idx[0], idx[1], idx[2]] += 1 if inside else -1
            voted[idx[0], idx[1], idx[2]] += 1
            # ghost sides: a crossing in (k, k + 1] (plus side), [k - 1, k) (minus side)
            if q < m and ts[q] <= k + 1:
                ghost[idx[0], idx[1], idx[2], a, 1] = True
                gdist[idx[0], idx[1], idx[2], a, 1] = ts[q] - k
                gface[idx[0], idx[1], idx[2], a, 1] = fs[q]
            if q > 0 and ts[q - 1] >= k - 1:
                ghost[idx[0], idx[1], idx[2], a, 0] = True
                gdist[idx[0], idx[1], idx[2], a, 0] = k - ts[q - 1]
                gface[idx[0], idx[1], idx[2], a, 0] = fs[q - 1]
        i = j


BAND_VOX = 1.5  # the signed distance is exact within this many voxels of Σ


@nb.njit(cache=True)
def _closest_on_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[float, float, float]:
    """The barycentric weights of the closest point to p on the triangle
    (a, b, c) (Ericson, Real-Time Collision Detection, 5.1.5). A weight is
    exactly 0 when the point is on the opposite edge or at a vertex."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return 1.0, 0.0, 0.0
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return 0.0, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return 1.0 - v, v, 0.0
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return 0.0, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return 1.0 - w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return 0.0, 1.0 - w, w
    den = 1.0 / (va + vb + vc)
    v, w = vb * den, vc * den
    return 1.0 - v - w, v, w


@nb.njit(cache=True)
def _nearest_faces(
    V: np.ndarray, F: np.ndarray, vox: np.ndarray, shape: np.ndarray, band: float,
    dist2: np.ndarray, near: np.ndarray,
) -> None:
    """For every voxel centre (k vox, mm) within band (mm) of a face: the
    squared distance to the nearest face and that face. Each face visits the
    voxels in its bounding box widened by band, so a voxel whose nearest
    face is within band always sees it: the result is exact in the band."""
    p = np.empty(3)
    for f in range(len(F)):
        a, b, c = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
        lo = np.empty(3, np.int64)
        hi = np.empty(3, np.int64)
        for ax in range(3):
            mn = min(a[ax], b[ax], c[ax]) - band
            mx = max(a[ax], b[ax], c[ax]) + band
            lo[ax] = max(int(np.ceil(mn / vox[ax])), 0)
            hi[ax] = min(int(np.floor(mx / vox[ax])), shape[ax] - 1)
        for i in range(lo[0], hi[0] + 1):
            p[0] = i * vox[0]
            for j in range(lo[1], hi[1] + 1):
                p[1] = j * vox[1]
                for k in range(lo[2], hi[2] + 1):
                    p[2] = k * vox[2]
                    w0, w1, w2 = _closest_on_triangle(p, a, b, c)
                    q = w0 * a + w1 * b + w2 * c
                    d2 = ((p - q) ** 2).sum()
                    if d2 < dist2[i, j, k]:
                        dist2[i, j, k] = d2
                        near[i, j, k] = f


@nb.njit(cache=True)
def _closest_weights(X: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    w = np.empty((len(X), 3))
    for i in range(len(X)):
        w[i, 0], w[i, 1], w[i, 2] = _closest_on_triangle(X[i], A[i], B[i], C[i])
    return w


def _pseudonormals(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit face normals (nf, 3), the edge pseudonormal of each face's edge
    k = (F[k], F[k + 1]) (nf, 3, 3: the sum of the unit normals of the one or
    two faces on that edge) and the angle-weighted vertex pseudonormals (nv,
    3) (Bærentzen and Aanæs 2005): the sign of (x - q) · (pseudonormal of the
    feature that q lies on) is the side of Σ for an oriented mesh with no
    self-intersection."""
    e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(e1, e2)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    E = np.stack([F, np.roll(F, -1, axis=1)], -1).reshape(-1, 2)
    key = np.sort(E, 1)
    _, inv = np.unique(key[:, 0] * (len(V) + 1) + key[:, 1], return_inverse=True)
    esum = np.zeros((inv.max() + 1, 3))
    np.add.at(esum, inv, np.repeat(fn, 3, axis=0))
    en = esum[inv].reshape(len(F), 3, 3)
    vn = np.zeros((len(V), 3))
    for k in range(3):
        u = V[F[:, (k + 1) % 3]] - V[F[:, k]]
        w = V[F[:, (k + 2) % 3]] - V[F[:, k]]
        cosang = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300)
        np.add.at(vn, F[:, k], np.arccos(np.clip(cosang, -1.0, 1.0))[:, None] * fn)
    return fn, en, vn


def signed_distance_band(
    V_mm: np.ndarray, F: np.ndarray, vox: np.ndarray, shape: tuple[int, ...], band_vox: float = BAND_VOX
) -> dict[str, np.ndarray]:
    """The signed distance s (mm, positive outside Σ) from the voxel centres
    within band_vox × h of Σ, with what ∂s/∂V needs (user, 2026-11-19:
    "Signed distance"; T4 of docs/plans/top-surface.md).

    s = ± |x - q|, q the closest point on Σ, the sign from the pseudonormal
    of the face, edge or vertex that q lies on. s is a continuous function
    of the vertices. With q = Σ w_i V_i on the nearest face (barycentric
    weights w), ∂s/∂V_i = -w_i u, u = (x - q) / s (the unit vector from q
    toward x for a voxel outside, away from x inside; the face normal when
    s = 0): the derivatives of the weights and of the face normal are
    tangent to the feature that q lies on, so they add nothing.

    Returns the band voxels (ijk), s, the face, the weights w and u."""
    vox = np.asarray(vox, float)
    h = float(vox.mean())
    band = band_vox * h
    shp = np.array(shape, np.int64)
    dist2 = np.full(tuple(shape), np.inf)
    near = np.full(tuple(shape), -1, np.int64)
    Vc = np.ascontiguousarray(V_mm, np.float64)
    Fc = np.ascontiguousarray(F, np.int64)
    _nearest_faces(Vc, Fc, vox, shp, band, dist2, near)
    inb = (near >= 0) & (dist2 < band * band)
    ijk = np.argwhere(inb)
    fc = near[inb]
    x = ijk * vox
    A, B, C = Vc[Fc[fc, 0]], Vc[Fc[fc, 1]], Vc[Fc[fc, 2]]
    w = _closest_weights(x, A, B, C)
    q = w[:, :1] * A + w[:, 1:2] * B + w[:, 2:] * C
    fn, en, vn = _pseudonormals(Vc, Fc)
    nz = w > 0.0
    cnt = nz.sum(1)
    pn = fn[fc].copy()
    one = cnt == 1  # at a vertex
    pn[one] = vn[Fc[fc[one], nz[one].argmax(1)]]
    two = cnt == 2  # on an edge: the edge k = (F[k], F[k + 1]) is the one whose opposite weight is 0
    k_edge = (np.argmin(nz[two], 1) + 1) % 3
    pn[two] = en[fc[two], k_edge]
    d = x - q
    dn = np.linalg.norm(d, axis=1)
    sgn = np.where((d * pn).sum(1) < 0.0, -1.0, 1.0)
    s = sgn * dn
    u = fn[fc].copy()
    ok = dn > 1e-12 * h
    u[ok] = d[ok] / s[ok, None]
    return {"ijk": ijk, "s": s, "face": fc, "w": w, "u": u}


def inside_from_band(sd: dict[str, np.ndarray], shape: tuple[int, ...]) -> tuple[np.ndarray, dict[str, int]]:
    """Inside Σ for every voxel: s < 0 in the band; elsewhere, each
    connected part (6 neighbours) of the voxels off the band takes the side
    of the band voxels next to it. A step between two neighbouring centres
    that crosses Σ has one centre within h/2 of Σ, inside the band
    (band_vox > 1/2), so the band separates the two sides, except through
    the open edges of Σ at the crop boundary. Counted for diagnosis: parts
    whose band neighbours disagree (mixed_parts, with the minority's
    neighbour count mixed_votes) and parts with no band neighbour (taken as
    outside)."""
    band = np.zeros(shape, bool)
    neg = np.zeros(shape, bool)
    X = tuple(sd["ijk"].T)
    band[X] = True
    neg[X] = sd["s"] < 0.0
    comp, n = ndi.label(~band)
    vin = np.zeros(n + 1, np.int64)
    vout = np.zeros(n + 1, np.int64)
    for ax in range(3):
        for sh in (1, -1):
            nb_comp = np.roll(comp, sh, axis=ax)
            edge = [slice(None)] * 3
            edge[ax] = slice(0, 1) if sh == 1 else slice(-1, None)
            nb_comp[tuple(edge)] = 0  # no wrap-around
            m = band & (nb_comp > 0)
            vin += np.bincount(nb_comp[m & neg], minlength=n + 1)
            vout += np.bincount(nb_comp[m & ~neg], minlength=n + 1)
    part_in = vin > vout
    part_in[0] = False
    inside = np.where(band, neg, part_in[comp])
    seen = (vin + vout)[1:] > 0
    mixed = (vin[1:] > 0) & (vout[1:] > 0)
    info = {
        "band_voxels": int(band.sum()),
        "mixed_parts": int(mixed.sum()),
        "mixed_votes": int(np.minimum(vin, vout)[1:].sum()),
        "parts_no_band": int((~seen).sum()),
    }
    return inside, info


def sigma_top(
    site: TP.Site, V_mm: np.ndarray, F: np.ndarray, tau_prev: np.ndarray | None, s: GS.Settings
) -> tuple[GS.SigmaTop, dict[str, int]]:
    """The fixed top surface Σ on the site's grid (see the module docstring):
    domain, inner voxels, ghosts (θ ≥ THETA_MIN) and τ0, and counts.

    Inside Σ is from the signed distance (signed_distance_band,
    inside_from_band). τ0 is the last τ where it exists; else s in the band
    and, off the band, s of the nearest band voxel inside Σ minus the
    distance to it (exact for a flat Σ).

    Every domain voxel is inside Σ: the tissue voxels, and the inner voxels
    inside Σ (while Σ still lies in the WM voxels, at the start, it is
    between inner voxels). Every domain voxel with a neighbour across Σ has
    the ghost τ = 0 on that side (T3). A connected part of the domain (6
    neighbours, the stencil) with no ghost has no τ = 0 to fix its offset:
    it is left out, and counted ("unanchored_voxels")."""
    shape = np.array(site.labels.shape, np.int64)
    shp = tuple(int(x) for x in shape)
    vox = np.asarray(site.vox, float)
    P = V_mm / vox
    Fi = np.ascontiguousarray(F, np.int64)
    vote = np.zeros(shp, np.int64)
    voted = np.zeros(shp, np.int64)
    ghost = np.zeros((*shp, 3, 2), bool)
    gdist = np.ones((*shp, 3, 2))
    gface = np.full((*shp, 3, 2), -1, np.int64)
    # The ghosts (τ = 0 at θh on an axis side) still come from the axis
    # crossings. TO TEST (user, 2026-11-19): measure whether θ jumps as Σ
    # moves (a ghost side that appears or goes away on a voxel that stays
    # in the domain; θ that changes face along a line) and how much of the
    # change in E_τ at a frozen τ comes from it. If it matters, replace θ
    # with θ = s / |u_a| from the signed distance (signed_distance_band),
    # which would also remove the axis crossings.
    for a in range(3):
        U, W, T, S, Fx = _axis_crossings(np.ascontiguousarray(P), Fi, a, shape)
        _line_votes(U, W, T, S, Fx, a, shape, vote, voted, ghost, gdist, gface)
    band = signed_distance_band(V_mm, Fi, vox, shp)
    inside_sigma, finfo = inside_from_band(band, shp)
    # the line votes of the old inside test, for diagnosis only
    vote_inside = (voted > 0) & (vote > 0)
    geo = {"inside": inside_sigma, "theta": gdist.copy(), "face": gface, "band": band}
    _, inner = GS.make_domain(site, s)
    tissue = inside_sigma & ~site.inside
    inner &= inside_sigma
    domain = tissue | inner
    ghost &= domain[..., None, None]
    comp, n = ndi.label(domain)
    anchored = np.zeros(n + 1, bool)
    anchored[np.unique(comp[ghost.any((3, 4))])] = True
    anchored[0] = False
    free = domain & ~anchored[comp]
    domain &= ~free
    inner &= domain
    ghost &= domain[..., None, None]
    gdist = np.clip(gdist, THETA_MIN, 1.0)
    # τ0: s in the band; off it, from the nearest band voxel inside Σ
    sgrid = np.full(shp, np.nan)
    sgrid[tuple(band["ijk"].T)] = band["s"]
    src = np.isfinite(sgrid) & inside_sigma
    tau0 = np.zeros(shp)
    if src.any():
        dd, idx = ndi.distance_transform_edt(~src, sampling=vox, return_indices=True)
        tau0 = sgrid[tuple(idx)] - dd
    if tau_prev is not None:
        ok = domain & np.isfinite(tau_prev)
        tau0[ok] = tau_prev[ok]
    info = {
        "tissue_voxels": int((domain & ~inner).sum()),
        "inner_voxels": int(inner.sum()),
        "ghost_sides": int(ghost.sum()),
        "unanchored_parts": int(n - anchored[1:].sum()),
        "unanchored_voxels": int(free.sum()),
        **finfo,
        "vote_disagree": int(((voted > 0) & (vote_inside != inside_sigma)).sum()),
    }
    h = float(vox.mean())
    # s for the fade-in (global_tau_energy._fade): band values, and -h (ω =
    # 1) for the voxels inside Σ off the band
    sdist = np.where(np.isfinite(sgrid), sgrid, -h)
    sig = GS.SigmaTop(domain, inner, ghost, gdist, tau0, sdist=sdist)
    sig.geo = geo
    return sig, info


class TauBehind:
    """τ behind Σ for one site, solved again before each start step."""

    def __init__(self, site: TP.Site, lm_iters: int = LM_ITERS):
        self.site = site
        self.s = GS.Settings(lm_iters=lm_iters, face_test=False)
        t0 = time.time()
        self.pt = TP.make_patches(site, self.s.patch_mm)
        self.centre_vox = np.array(site.labels.shape) / 2.0
        self.tau_grid: np.ndarray | None = None
        self.log: list[dict[str, Any]] = []
        self.setup_s = time.time() - t0

    def solve(self, V_mm: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, GS.GlobalTau | None]:
        """τ behind Σ = (V_mm, F): returns τ on the grid (nan off the
        domain), ∇τ on the grid ((X, Y, Z, 3), nan off the domain: every
        domain voxel is inside Σ) and the solver (None when the domain is
        empty)."""
        t0 = time.time()
        sig, ginfo = sigma_top(self.site, V_mm, F, self.tau_grid, self.s)
        shape = self.site.labels.shape
        grad = np.full((*shape, 3), np.nan)
        if not sig.domain.any():
            self.log.append({**ginfo, "s": time.time() - t0})
            return np.full(shape, np.nan), grad, None
        gt = GS.GlobalTau(self.site, self.pt, self.s, self.centre_vox, sigma=sig)
        info, _ = gt.solve_global("sigma")
        pb = gt._pb
        x = np.zeros(pb.S1)
        x[: pb.S] = gt.tau
        g = np.asarray(GE.gradient(jnp.asarray(x), pb.P))[: pb.S]
        tau = gt.G.grid(gt.tau)
        X = tuple(gt.G.coords.T - GS.PAD)
        grad[X] = g
        grad[~sig.domain] = np.nan
        self.tau_grid = tau
        self.log.append(
            {
                **ginfo,
                "tau_min": float(np.nanmin(tau)),
                "tau_max": float(np.nanmax(tau)),
                "lm_iters": len(info["iters"]),
                "e0": info["e0"],
                "e1": info["e1"],
                "s": time.time() - t0,
            }
        )
        return tau, grad, gt

    def directions(self, V_mm: np.ndarray, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per vertex: the unit mean of ∇τ over the τ voxels inside Σ within
        one voxel, weighted by 1 - distance (voxels), and whether it has one."""
        p = V_mm / self.site.vox
        shape = np.array(grad.shape[:3])
        lo = np.floor(p).astype(np.int64)
        acc = np.zeros((len(p), 3))
        wsum = np.zeros(len(p))
        for di in (-1, 0, 1, 2):
            for dj in (-1, 0, 1, 2):
                for dk in (-1, 0, 1, 2):
                    q = lo + np.array([di, dj, dk])
                    ok = np.all((q >= 0) & (q < shape), 1)
                    d = np.linalg.norm(q - p, axis=1)
                    w = np.where(ok, np.maximum(1.0 - d, 0.0), 0.0)
                    qc = np.clip(q, 0, shape - 1)
                    g = grad[qc[:, 0], qc[:, 1], qc[:, 2]]
                    good = (w > 0) & np.isfinite(g).all(1)
                    acc[good] += w[good, None] * g[good]
                    wsum[good] += w[good]
        n = np.linalg.norm(acc, axis=1)
        has = (wsum > 0) & (n > 0)
        D = np.zeros_like(acc)
        D[has] = acc[has] / n[has, None]
        return D, has
