"""The signed distance to Σ near Σ: the parts of top_tau.py (prototype
branch claude/tau-shell-cc08e1) that the Σ stop stage uses, unchanged.

The signed distance s from each voxel centre within BAND_VOX voxels of Σ to
its closest point on Σ, with the sign from the pseudonormal of the face,
edge or vertex that the closest point lies on; inside where s < 0. s is a
continuous function of Σ's vertices.

stop_field.py uses the closest-point kernels (_closest_on_triangle,
_closest_weights, _pseudonormals). signed_distance_band (on a grid-sized
array) and inside_from_band are the dense reference of its tests.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from scipy import ndimage as ndi

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
    V: np.ndarray,
    F: np.ndarray,
    vox: np.ndarray,
    shape: np.ndarray,
    band: float,
    dist2: np.ndarray,
    near: np.ndarray,
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
def _closest_weights(
    X: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> np.ndarray:
    w = np.empty((len(X), 3))
    for i in range(len(X)):
        w[i, 0], w[i, 1], w[i, 2] = _closest_on_triangle(X[i], A[i], B[i], C[i])
    return w


def _pseudonormals(
    V: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        cosang = (u * w).sum(1) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300
        )
        np.add.at(vn, F[:, k], np.arccos(np.clip(cosang, -1.0, 1.0))[:, None] * fn)
    return fn, en, vn


def signed_distance_band(
    V_mm: np.ndarray,
    F: np.ndarray,
    vox: np.ndarray,
    shape: tuple[int, ...],
    band_vox: float = BAND_VOX,
) -> dict[str, np.ndarray]:
    """The signed distance s (mm, positive outside Σ) from the voxel centres
    within band_vox × h of Σ, with what ∂s/∂V needs (user, 2026-11-19:
    "Signed distance"; T4 of the plan docs/plans/top-surface.md on the
    prototype branch).

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
    two = (
        cnt == 2
    )  # on an edge: the edge k = (F[k], F[k + 1]) is the one whose opposite weight is 0
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


def inside_from_band(
    sd: dict[str, np.ndarray], shape: tuple[int, ...]
) -> tuple[np.ndarray, dict[str, int]]:
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
