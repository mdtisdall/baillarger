"""Reference for baillarger.mesh_intersect: the plain numpy implementation of
the prototype, kept only to check the numba one against it.

Exact triangle-triangle intersection counts for triangle meshes.

Candidate pairs come from a uniform grid over the triangles' bounding boxes;
each pair is tested with the interval test of Moller (1997): each triangle
must straddle the other's plane, and the two triangles' intervals on the line
where the planes meet must overlap. Touching pairs (a vertex or edge on the
other triangle, within a tolerance) and coplanar pairs count as not intersecting.
In a self test, triangles that share a vertex are not tested (neighbours touch
by construction).
"""

from typing import Any, overload

import numpy as np
import numpy.typing as npt

type Arr = npt.NDArray[Any]


def candidate_pairs(
    lo: Arr,
    hi: Arr,
    cell: float | None = None,
    max_pairs: int | None = None,
    max_span: int = 4,
) -> Arr:
    """Pairs (i < j) of boxes [lo, hi] (M, 3) that share a grid cell and overlap.
    Boxes spanning more than max_span cells in some direction (a few stretched
    triangles) are compared with all boxes directly instead of being put in the
    grid. Raises RuntimeError when there would be more than max_pairs candidate
    pairs (crowded triangles: a sign of a collapsed mesh)."""
    ext = hi - lo
    if cell is None:
        cell = 2.0 * np.median(ext.max(1)) + 1e-12
    c0 = np.floor(lo / cell).astype(np.int64)
    c1 = np.floor(hi / cell).astype(np.int64)
    span = c1 - c0 + 1
    big = np.any(span > max_span, 1)
    small = np.where(~big)[0]
    keys_l: list[Arr] = []
    tris_l: list[Arr] = []
    sp = span[small]
    for dx in range(sp[:, 0].max(initial=0)):
        for dy in range(sp[:, 1].max(initial=0)):
            for dz in range(sp[:, 2].max(initial=0)):
                d = np.array([dx, dy, dz])
                m = np.all(d[None, :] < sp, 1)
                c = c0[small[m]] + d
                keys_l.append(
                    c[:, 0] * 73856093 ^ c[:, 1] * 19349663 ^ c[:, 2] * 83492791
                )
                tris_l.append(small[m])
    out = []
    if keys_l:
        keys, tris = np.concatenate(keys_l), np.concatenate(tris_l)
        o = np.lexsort((tris, keys))
        keys, tris = keys[o], tris[o]
        start = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
        size = np.diff(np.r_[start, len(keys)])
        total = int((size * (size - 1) // 2).sum())
        if max_pairs is not None and total > max_pairs:
            raise RuntimeError(
                f"{total} candidate pairs (more than {max_pairs}); "
                f"largest cell has {size.max()} triangles"
            )
        end = np.repeat(start + size, size)
        for k in range(1, size.max()):
            i = np.flatnonzero(np.arange(len(keys)) + k < end)
            out.append(np.stack([tris[i], tris[i + k]], 1))
    for b in np.where(big)[0]:
        m = np.all((lo <= hi[b]) & (lo[b] <= hi), 1)
        m[b] = False
        out.append(np.stack([np.full(m.sum(), b), np.where(m)[0]], 1))
        if max_pairs is not None and sum(len(x) for x in out) > max_pairs:
            raise RuntimeError(
                f"more than {max_pairs} candidate pairs, "
                f"with {big.sum()} stretched triangles"
            )
    if not out:
        return np.zeros((0, 2), np.int64)
    P = np.sort(np.concatenate(out), 1)
    P = P[P[:, 0] != P[:, 1]]
    P = np.unique(P, axis=0)
    ov = np.all((lo[P[:, 0]] <= hi[P[:, 1]]) & (lo[P[:, 1]] <= hi[P[:, 0]]), 1)
    return P[ov]


def _interval(d: Arr, p: Arr) -> tuple[Arr, Arr]:
    """The interval of a triangle on the line, from its signed distances d and line
    coordinates p (M, 3): crossings of its edges, and its vertices on the plane."""
    vals = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cross = d[:, i] * d[:, j] < 0
        with np.errstate(invalid="ignore", divide="ignore"):
            t = p[:, i] + (p[:, j] - p[:, i]) * d[:, i] / (d[:, i] - d[:, j])
        vals.append(np.where(cross, t, np.nan))
    for i in range(3):
        vals.append(np.where(d[:, i] == 0, p[:, i], np.nan))
    v = np.stack(vals, 1)
    return np.nanmin(v, 1), np.nanmax(v, 1)


def tri_tri(A: Arr, B: Arr, rel_tol: float = 1e-4) -> Arr:
    """Whether triangles A[k] and B[k] (M, 3, 3) intersect. Distances to a plane
    under rel_tol x the pair's mean edge length count as 0, and the intervals must
    overlap by more than that: near-touching triangles do not count."""

    def plane(T: Arr) -> tuple[Arr, Arr]:
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        return n, -np.einsum("ij,ij->i", n, T[:, 0])

    edge = 0.5 * (
        np.linalg.norm(np.diff(A, axis=1, append=A[:, :1]), axis=2).mean(1)
        + np.linalg.norm(np.diff(B, axis=1, append=B[:, :1]), axis=2).mean(1)
    )
    tol = rel_tol * edge
    n2, d2 = plane(B)
    nn2 = np.linalg.norm(n2, axis=1)
    da = np.einsum("mij,mj->mi", A, n2) + d2[:, None]
    da = np.where(np.abs(da) < (tol * nn2)[:, None], 0, da)
    n1, d1 = plane(A)
    nn1 = np.linalg.norm(n1, axis=1)
    db = np.einsum("mij,mj->mi", B, n1) + d1[:, None]
    db = np.where(np.abs(db) < (tol * nn1)[:, None], 0, db)
    sep = (
        np.all(da >= 0, 1)
        | np.all(da <= 0, 1)
        | np.all(db >= 0, 1)
        | np.all(db <= 0, 1)
    )
    D = np.cross(n1, n2)
    nD = np.linalg.norm(D, axis=1)
    pa = np.einsum("mij,mj->mi", A, D)
    pb = np.einsum("mij,mj->mi", B, D)
    out = np.zeros(len(A), bool)
    k = ~sep
    if k.any():
        a0, a1 = _interval(da[k], pa[k])
        b0, b1 = _interval(db[k], pb[k])
        out[k] = np.minimum(a1, b1) - np.maximum(a0, b0) > (tol * nD)[k]
    return out


@overload
def intersecting_faces(
    V: Arr,
    F: Arr,
    V2: None = None,
    F2: None = None,
    chunk: int = 500000,
    max_pairs: int | None = None,
) -> tuple[Arr, int]: ...


@overload
def intersecting_faces(
    V: Arr,
    F: Arr,
    V2: Arr,
    F2: Arr,
    chunk: int = 500000,
    max_pairs: int | None = None,
) -> tuple[Arr, Arr, int]: ...


def intersecting_faces(
    V: Arr,
    F: Arr,
    V2: Arr | None = None,
    F2: Arr | None = None,
    chunk: int = 500000,
    max_pairs: int | None = None,
) -> tuple[Arr, int] | tuple[Arr, Arr, int]:
    """Faces that intersect another face: within (V, F) when V2 is None (pairs
    sharing a vertex skipped), else between (V, F) and (V2, F2). Returns a boolean
    mask over F (and over F2 for two meshes) and the number of intersecting pairs."""
    if V2 is None:
        T = V[F]
        P = candidate_pairs(T.min(1), T.max(1), max_pairs=max_pairs)
        share = np.zeros(len(P), bool)
        for i in range(3):
            for j in range(3):
                share |= F[P[:, 0], i] == F[P[:, 1], j]
        P = P[~share]
        T1 = T2 = T
    else:
        T1, T2 = V[F], V2[F2]
        T = np.concatenate([T1, T2])
        P = candidate_pairs(T.min(1), T.max(1), max_pairs=max_pairs)
        n1 = len(T1)
        P = P[(P[:, 0] < n1) & (P[:, 1] >= n1)]
        P[:, 1] -= n1
    hit = np.zeros(len(P), bool)
    for c in range(0, len(P), chunk):
        s = slice(c, c + chunk)
        hit[s] = tri_tri(T1[P[s, 0]], T2[P[s, 1]])
    m1 = np.zeros(len(T1), bool)
    m1[P[hit, 0]] = True
    if V2 is None:
        m1[P[hit, 1]] = True
        return m1, int(hit.sum())
    m2 = np.zeros(len(T2), bool)
    m2[P[hit, 1]] = True
    return m1, m2, int(hit.sum())


def faces_hitting(
    V: Arr,
    F: Arr,
    sub: Arr,
    V2: Arr | None = None,
    F2: Arr | None = None,
    chunk: int = 500000,
    max_pairs: int | None = None,
) -> Arr:
    """Which of the faces F[sub] intersect a face of (V, F) (sub's own faces and
    faces sharing a vertex with them not counted) or, with V2, a face of (V2, F2).
    Only the faces near sub are tested, so a step that moves a few vertices is
    cheap to check."""
    sub = np.asarray(sub)
    Ts = V[F[sub]]
    To = V[F] if V2 is None else V2[F2]
    lo_s, hi_s = Ts.min(1), Ts.max(1)
    lo_o, hi_o = To.min(1), To.max(1)
    # only the other faces whose boxes reach the box around sub
    near = np.all((lo_o <= hi_s.max(0)) & (lo_s.min(0) <= hi_o), 1)
    on = np.where(near)[0]
    lo = np.concatenate([lo_s, lo_o[on]])
    hi = np.concatenate([hi_s, hi_o[on]])
    P = candidate_pairs(lo, hi, max_pairs=max_pairs)
    ns = len(sub)
    P = P[(P[:, 0] < ns) & (P[:, 1] >= ns)]
    a, b = P[:, 0], on[P[:, 1] - ns]
    if V2 is None:
        keep = b != sub[a]
        for i in range(3):
            for j in range(3):
                keep &= F[sub[a], i] != F[b, j]  # in a self test the other faces are F
        a, b = a[keep], b[keep]
    hit = np.zeros(len(a), bool)
    for c in range(0, len(a), chunk):
        s = slice(c, c + chunk)
        hit[s] = tri_tri(Ts[a[s]], To[b[s]])
    out = np.zeros(ns, bool)
    out[a[hit]] = True
    return out
