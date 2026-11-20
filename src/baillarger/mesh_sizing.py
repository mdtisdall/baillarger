"""Target edge lengths for meshes of a surface and of the volume round it.

One sizing for the surface remesher (the coarse Σ start) and for a graded
τ volume mesh. Each caller passes its own constants (user, 2026-11-20).

On the surface (curvature_sizes):
  - shape: a flat triangle with edges L over a surface of curvature radius R
    deviates from it by about L² / (8 R). With an allowed deviation (the sag)
    this gives L <= sqrt(8 sag / κ), κ the larger absolute principal
    curvature (baillarger.frame.mesh_curvatures, smoothed along the surface).
    `scale` multiplies it (the remesher keeps edges up to 4/3 × the target,
    so the Σ start uses 3/4 of the bound);
  - facing sheets (optional): L <= the distance along the outward normal to
    a facing part of the surface (facing_distances), so that two sheets
    across thin tissue are not joined by one face;
  - L is clipped to [lo, hi].

In the volume (graded_sizes): the target at a point x is
min over surface vertices v of (s_v + grade |x - v|), up to `far`: the
largest field that keeps the surface's sizes and grows at most `grade` per
unit of distance. A volume mesh that the surface cuts must not have a cell
cut by two sheets: `facing` caps s_v by facing_factor × the vertex's facing
distance, below the surface clip (a cell near facing sheets at most about
gap / 2, the user's rule of 2026-11-20).

From sigma_mesh.sizing (feature/top-surface, 79d876d, 8b7ac75) and
tau_tet.sigma_sizes / sizing_field (feature/tau-coarse, b1d2cee, fda2d1e).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from baillarger._jit import njit
from baillarger.frame import mesh_curvatures
from baillarger.mesh_checks import face_normals

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.int64]


def curvature_sizes(
    V: npt.ArrayLike,
    F: npt.ArrayLike,
    normals: npt.ArrayLike,
    sag: float,
    smooth: float,
    lo: float,
    hi: float,
    scale: float = 1.0,
    facing: Float | None = None,
) -> tuple[Float, Float]:
    """The target edge length at each vertex, scale × sqrt(8 sag / κ), at
    most `facing` (per vertex, if given), clipped to [lo, hi]; and |κ|max
    (per unit length). normals: outward unit normals per vertex; smooth: the
    curvature's smoothing length along the surface. All lengths in the
    units of V."""
    Vf = np.asarray(V, np.float64)
    Fi = np.asarray(F, np.int64)
    k1, k2 = mesh_curvatures(Vf, Fi, np.asarray(normals, np.float64), smooth)
    kap: Float = np.maximum(np.abs(k1), np.abs(k2))
    L = scale * np.sqrt(8.0 * sag / np.maximum(kap, 1e-12))
    if facing is not None:
        L = np.minimum(L, facing)
    out: Float = np.clip(L, lo, hi)
    return out, kap


@njit(cache=True)
def _facing_hits(
    V: Float,
    N: Float,
    F: Int,
    FN: Float,
    ptr: Int,
    idx: Int,
    lmax: float,
) -> Float:
    """For each vertex i (candidate faces idx[ptr[i]:ptr[i + 1]]): the
    distance along +N[i] to the nearest face (not one of its own) whose
    normal points against N[i], within lmax (Möller-Trumbore); inf if
    none."""
    n = len(V)
    best = np.full(n, np.inf)
    for i in range(n):
        o = V[i]
        d = N[i]
        for q in range(ptr[i], ptr[i + 1]):
            f = idx[q]
            a, b, c = F[f, 0], F[f, 1], F[f, 2]
            if a == i or b == i or c == i:
                continue
            if FN[f, 0] * d[0] + FN[f, 1] * d[1] + FN[f, 2] * d[2] >= 0.0:
                continue
            e1 = V[b] - V[a]
            e2 = V[c] - V[a]
            p = np.cross(d, e2)
            det = e1 @ p
            if abs(det) < 1e-18:
                continue
            tv = o - V[a]
            u = (tv @ p) / det
            if u < 0.0 or u > 1.0:
                continue
            qv = np.cross(tv, e1)
            w = (d @ qv) / det
            if w < 0.0 or u + w > 1.0:
                continue
            t = (e2 @ qv) / det
            if 0.0 < t <= lmax and t < best[i]:
                best[i] = t
    return best


def facing_distances(
    V: npt.ArrayLike, F: npt.ArrayLike, normals: npt.ArrayLike, lmax: float
) -> Float:
    """For each vertex: the distance along its outward normal to the nearest
    face of the surface that faces it (its normal points back), within
    lmax; inf where there is none. The candidates are the faces whose
    centroid is within lmax / 2 + the largest face radius of the point
    lmax / 2 out along the normal."""
    Vf = np.ascontiguousarray(V, np.float64)
    Fi = np.ascontiguousarray(F, np.int64)
    N = np.ascontiguousarray(normals, np.float64)
    FN = face_normals(Vf, Fi)
    cen = Vf[Fi].mean(1)
    rad = float(np.linalg.norm(Vf[Fi] - cen[:, None], axis=2).max()) if len(Fi) else 0.0
    raw: Any = cKDTree(cen).query_ball_point(Vf + 0.5 * lmax * N, 0.5 * lmax + rad)
    lists: list[list[int]] = list(raw)
    ptr = np.zeros(len(Vf) + 1, np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    idx = np.fromiter((j for x in lists for j in x), np.int64, int(ptr[-1]))
    return _facing_hits(Vf, N, Fi, FN, ptr, idx, float(lmax))


def graded_sizes(
    P: npt.ArrayLike,
    V: npt.ArrayLike,
    size: npt.ArrayLike,
    grade: float,
    far: float,
    facing: Float | None = None,
    facing_factor: float = 0.5,
    chunk: int = 2048,
) -> Float:
    """The target edge length at points P: min over the surface vertices v
    of (s_v + grade |P - v|), at most far, with s_v = size_v, or
    min(size_v, facing_factor × facing_v) where `facing` is given (that cap
    is not clipped below). Costs O(len(P) × len(V)), in chunks of P."""
    Pf = np.asarray(P, np.float64).reshape(-1, 3)
    Vf = np.asarray(V, np.float64)
    s = np.asarray(size, np.float64)
    if facing is not None:
        s = np.minimum(s, facing_factor * facing)
    out: Float = np.full(len(Pf), float(far))
    for a in range(0, len(Pf), chunk):
        d = np.linalg.norm(Pf[a : a + chunk, None] - Vf[None], axis=2)
        out[a : a + chunk] = np.minimum(
            out[a : a + chunk], (s[None] + grade * d).min(1)
        )
    return out


def kuhn_grid(
    lo: npt.ArrayLike, hi: npt.ArrayLike, spacing: float
) -> tuple[Float, Int]:
    """A background tetrahedral mesh of the box [lo, hi]: a grid of about
    `spacing` (at least 2 points per axis, ends on lo and hi), each cube cut
    into the 6 tetrahedra on its main diagonal (Kuhn). Points (n, 3) and
    tetrahedra (m, 4)."""
    lo_ = np.asarray(lo, np.float64)
    hi_ = np.asarray(hi, np.float64)
    n = np.maximum(np.ceil((hi_ - lo_) / spacing).astype(np.int64), 1) + 1
    ax = [np.linspace(lo_[a], hi_[a], int(n[a])) for a in range(3)]
    G: Float = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    idx = np.arange(len(G)).reshape(tuple(int(x) for x in n))
    c: list[Any] = [
        idx[i : n[0] - 1 + i, j : n[1] - 1 + j, k : n[2] - 1 + k].ravel()
        for i in (0, 1)
        for j in (0, 1)
        for k in (0, 1)
    ]  # cube corners, bits (i, j, k)
    paths = ((1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6))
    T: Int = np.concatenate([np.stack([c[0], c[a], c[b], c[7]], 1) for a, b in paths])
    return G, T.astype(np.int64)
