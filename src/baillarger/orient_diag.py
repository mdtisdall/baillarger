"""Diagnostics of orient (phase 1.2, Part E): the agreement of the white
surface's normals with the data normal.

For each white-surface vertex: the cortex voxels within one voxel outward
along its normal, their data normals weighted by their coherence, and the
angle of that weighted mean to the vertex's normal.
"""

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
Box = tuple[slice, slice, slice]

#: the samples along the vertex normal, in voxels (of the smallest voxel
#: size): the voxels that hold them are the ones within 1 voxel outward
SAMPLES = (0.5, 1.0)


def surface_agreement(
    x: npt.ArrayLike,
    n: npt.ArrayLike,
    normal: npt.ArrayLike,
    coherence: npt.ArrayLike,
    sign: npt.ArrayLike,
    box: Box,
    affine: npt.ArrayLike,
) -> tuple[F64, F64]:
    """The angle (degrees) and the weight of each vertex.

    x (V, 3), n (V, 3): the vertices and their unit normals, scanner mm.
    normal (3, box), coherence (box), sign (box): orient's output on box
    (grid points). For vertex v, the voxels are those that hold
    x_v + s n_v for s = k vs_min (k in SAMPLES, vs_min the smallest voxel
    size), each voxel once, in the box, where the data normal is defined.
    m = sum_i c_i s_i d_i, with d_i the data normal, c_i its coherence, and
    s_i = 1 where orient signed d_i (sign not 0), else sign(d_i . n_v) (an
    unsigned normal is compared as an axis). The angle is
    arccos(m . n_v / |m|) in [0, 180]; the weight is sum_i c_i. NaN angle
    and weight 0 where no voxel counts."""
    A = np.asarray(affine, np.float64)
    inv = np.linalg.inv(A)
    vs = np.linalg.norm(A[:3, :3], axis=0)
    X = np.asarray(x, np.float64)
    N = np.asarray(n, np.float64)
    nf = np.asarray(normal, np.float64)
    cf = np.asarray(coherence, np.float64)
    sf = np.asarray(sign)
    lo = np.array([b.start for b in box])
    shape = np.array(cf.shape)
    V = len(X)
    m = np.zeros((V, 3))
    w = np.zeros(V)
    seen: list[npt.NDArray[np.int64]] = []
    for k in SAMPLES:
        p = X + (k * float(vs.min())) * N
        v = p @ inv[:3, :3].T + inv[:3, 3]
        idx = np.floor(v + 0.5).astype(np.int64) - lo
        ok = np.all((idx >= 0) & (idx < shape), axis=1)
        safe = np.where(ok[:, None], idx, 0)
        flat = np.where(ok, np.ravel_multi_index(tuple(safe.T), tuple(shape)), -1)
        for prev in seen:
            ok &= flat != prev
        seen.append(flat)
        sel = np.nonzero(ok)[0]
        ii = tuple(idx[sel].T)
        d = nf[(slice(None), *ii)].T  # (s, 3)
        c = cf[ii]
        sg = sf[ii]
        good = np.isfinite(c) & np.all(np.isfinite(d), 1)
        sel, d, c, sg = sel[good], d[good], c[good], sg[good]
        dot = np.einsum("sa,sa->s", d, N[sel])
        flip = np.where(sg == 0, np.where(dot < 0, -1.0, 1.0), 1.0)
        np.add.at(m, sel, (c * flip)[:, None] * d)
        np.add.at(w, sel, c)
    norm = np.linalg.norm(m, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("va,va->v", m, N) / norm
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    ang = np.where((w > 0) & (norm > 0), ang, np.nan)
    return ang, w
