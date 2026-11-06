"""Maps of the white surface's smoothed normals, from the mesh and its normals
only (wave 2.1, docs/plans/wave-2.1-normals.md): no echoes, no rays.

The maps test whether the published normals (white-surf/1,
desc-smoothnormals) agree with themselves and with the mesh. They do not
test whether the normals are correct: normals can be smooth and point the
wrong way.

One value for each vertex (MAPS):

- nbrmax: the largest angle (degrees) between the vertex's smoothed normal
  and the smoothed normals of its 1-ring neighbours.
- nbrmaxpermm: for the same neighbour, that angle divided by the edge length
  (degrees / mm). Of neighbours with the same largest angle, the largest
  ratio.
- normlen: |n̄|, as given (the published desc-smoothnormlen).
- renorm: the angle (degrees) between the smoothed normal n and m / |m|, where
  m takes the same smoothing steps with no division (linear_smoothed). It
  shows where the division at each step changed the normal.
- smoothing: the angle (degrees) between the raw normal
  (baillarger.decimate.normals, made unit) and n.
- misfacingraw: the faces that touch the vertex and are misfacing against
  the raw normals (baillarger.wm_sites.misfacing, on the surface in mm).
- misfacingsmooth: the same against the smoothed normals: the face's normal
  · (n[a] + n[b] + n[c]) <= 0.

A vertex that is in no face gets NaN in every map; a vertex with no edge gets
NaN in nbrmax and nbrmaxpermm.

The frame: white_surf smooths the normals on the vertices in voxel
coordinates of the grid and writes them with no transform. So every map that
uses the smoothed normals is computed there too: the raw normals, m and the
face normals of misfacingsmooth on the vertices in voxel coordinates. Only the
edge lengths (mm) and misfacingraw (as wm_sites) use scanner mm.

ranked_sites gives the ranked lists of the review: take the largest value,
keep its vertex as a site, drop the vertices within a distance of it, and
repeat. No threshold.

Memory, for a mesh of V vertices, F faces and E (about 1.5 F) edges: the
edges (16 E bytes), the umbrella matrix of linear_smoothed (about 16 E
bytes in float32, more while it is built), the raw normals (24 V) and the
maps (4 V each). The edges are walked in chunks.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from baillarger import decimate, wm_sites
from baillarger import mesh_smoothing as S

type Float = npt.NDArray[np.float64]
type F32 = npt.NDArray[np.float32]
type Int = npt.NDArray[np.int64]
type Faces = npt.NDArray[np.integer[Any]]

#: the maps, in the order of the maps file
MAPS = (
    "nbrmax",
    "nbrmaxpermm",
    "normlen",
    "renorm",
    "smoothing",
    "misfacingraw",
    "misfacingsmooth",
)
#: the unit of each map
UNITS = dict(
    nbrmax="degree",
    nbrmaxpermm="degree / mm",
    normlen="",
    renorm="degree",
    smoothing="degree",
    misfacingraw="faces",
    misfacingsmooth="faces",
)
#: edges or faces per chunk
CHUNK = 4_000_000
#: candidates per chunk of ranked_sites
SITE_CHUNK = 4096


def angle_deg(a: npt.ArrayLike, b: npt.ArrayLike) -> Float:
    """Row by row, the angle (degrees) between vectors a and b, which need not
    be unit: atan2(|a × b|, a · b), exact also near 0 and 180. NaN where
    either is zero."""
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cr = np.linalg.norm(np.cross(x, y), axis=1)
    d = np.sum(x * y, axis=1)
    out: Float = np.degrees(np.arctan2(cr, d))
    zero = (np.linalg.norm(x, axis=1) == 0) | (np.linalg.norm(y, axis=1) == 0)
    out[zero] = np.nan
    return out


def used_vertices(f: Faces, nv: int) -> npt.NDArray[np.bool_]:
    """The vertices that are in at least one face."""
    u = np.zeros(nv, bool)
    u[np.asarray(f).ravel()] = True
    return u


def neighbour_maps(
    n: npt.ArrayLike, v_mm: npt.ArrayLike, E: Int, chunk: int = CHUNK
) -> tuple[F32, F32]:
    """nbrmax and nbrmaxpermm of normals n on the edges E (k, 2), edge lengths
    from v_mm. Two passes over the edges in chunks: the largest angle at each
    vertex, then the ratio at the edges that give it."""
    nn, vm = np.asarray(n), np.asarray(v_mm)
    nv = len(nn)
    best = np.full(nv, -np.inf)
    for e0 in range(0, len(E), chunk):
        a, b = E[e0 : e0 + chunk, 0], E[e0 : e0 + chunk, 1]
        ang = angle_deg(nn[a], nn[b])
        np.fmax.at(best, a, ang)
        np.fmax.at(best, b, ang)
    per = np.full(nv, -np.inf)
    for e0 in range(0, len(E), chunk):
        a, b = E[e0 : e0 + chunk, 0], E[e0 : e0 + chunk, 1]
        ang = angle_deg(nn[a], nn[b])  # the same values as the first pass
        ln = np.linalg.norm(
            np.asarray(vm[a], np.float64) - np.asarray(vm[b], np.float64), axis=1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            r = ang / ln
        for end in (a, b):
            hit = ang == best[end]
            np.fmax.at(per, end[hit], r[hit])
    best[~np.isfinite(best)] = np.nan
    per[per == -np.inf] = np.nan
    return best.astype(np.float32), per.astype(np.float32)


def linear_smoothed[T: np.floating[Any]](
    v: npt.NDArray[T], f: Faces, iters: int
) -> npt.NDArray[T]:
    """m: the unit area-weighted vertex normals after iters steps of
    m = 0.5 m + 0.5 (the mean of m over the neighbours), with no division, as
    mesh_smoothing.smooth_normals_and_length smooths its length (which gives
    only |m|). m is linear in the start normals."""
    if iters < 1:
        raise ValueError("iters must be at least 1")
    m = S.vertex_normals(v, f)
    A, deg = S.umbrella(v, f)
    for _ in range(iters):
        m = 0.5 * m + 0.5 * (A @ m) / deg[:, None]
    return m


def face_counts(f: Faces, bad: npt.NDArray[np.bool_], nv: int) -> F32:
    """For each vertex, the number of faces with `bad` that touch it."""
    c = np.bincount(np.asarray(f)[bad].ravel(), minlength=nv)
    return c.astype(np.float32)


def misfacing_smooth(
    v: npt.ArrayLike, f: Faces, n: npt.ArrayLike
) -> npt.NDArray[np.bool_]:
    """Per face: its normal · (n[a] + n[b] + n[c]) <= 0, with n the smoothed
    normals (wm_sites.misfacing's rule, with n in place of the raw normals).
    CHUNK faces at a time."""
    vv, nn, ff = np.asarray(v), np.asarray(n, np.float64), np.asarray(f)
    out = np.zeros(len(ff), bool)
    for a in range(0, len(ff), CHUNK):
        fc = ff[a : a + CHUNK]
        p0 = np.asarray(vv[fc[:, 0]], np.float64)
        fn = np.cross(
            np.asarray(vv[fc[:, 1]], np.float64) - p0,
            np.asarray(vv[fc[:, 2]], np.float64) - p0,
        )
        s = nn[fc[:, 0]] + nn[fc[:, 1]] + nn[fc[:, 2]]
        out[a : a + len(fc)] = np.sum(fn * s, 1) <= 0
    return out


def normal_maps(
    v_vox: npt.ArrayLike,
    v_mm: npt.ArrayLike,
    f: Faces,
    n: npt.ArrayLike,
    normlen: npt.ArrayLike,
    steps: int,
    log: Callable[[str], object] = lambda m: None,
) -> dict[str, F32]:
    """The maps (MAPS, the module docstring) of the surface (v_vox: vertices in
    voxel coordinates, as the normals were smoothed; v_mm: the same in scanner
    mm; faces f) with its published smoothed normals n, their length normlen
    and the smoothing's steps (NormalSmoothingSteps)."""
    vx = np.asarray(v_vox, np.float64)
    ff = np.asarray(f, np.int64)
    nn = np.asarray(n, np.float32)
    nv = len(vx)
    used = used_vertices(ff, nv)
    out: dict[str, F32] = {}
    E = S.edges(ff)
    out["nbrmax"], out["nbrmaxpermm"] = neighbour_maps(nn, v_mm, E)
    del E
    log("nbrmax, nbrmaxpermm")
    out["normlen"] = np.asarray(normlen, np.float32).reshape(-1).copy()
    # float32, as white_surf smooths them
    m = linear_smoothed(vx.astype(np.float32), ff, steps)
    out["renorm"] = angle_deg(nn, m).astype(np.float32)
    del m
    log("renorm")
    raw = decimate.normals(vx, ff)
    out["smoothing"] = angle_deg(raw, nn).astype(np.float32)
    del raw
    log("smoothing")
    vm = np.asarray(v_mm, np.float64)
    out["misfacingraw"] = face_counts(ff, wm_sites.misfacing(vm, ff), nv)
    del vm
    log("misfacingraw")
    out["misfacingsmooth"] = face_counts(ff, misfacing_smooth(vx, ff, nn), nv)
    log("misfacingsmooth")
    for k in MAPS:
        out[k][~used] = np.nan
    return {k: out[k] for k in MAPS}


def ranked_sites(
    values: npt.ArrayLike,
    points_mm: npt.NDArray[np.floating[Any]],
    n: int,
    spacing: float,
    order: str = "high",
) -> Int:
    """The ranked list: the largest finite value (order "low": the smallest)
    is a site; every vertex within spacing mm of it (distance <= spacing) is
    dropped; repeat until n sites or no candidate is left. NaN is never a
    site; ties go to the lower vertex id. The finite values are sorted once
    and walked in chunks; each chunk is tested against a KD-tree of the sites
    kept so far (at most n points), and its own conflicts one by one. (The
    viewer's overlay, baillarger.viewer.overlay, has the same rule.)"""
    if order not in ("high", "low"):
        raise ValueError(f"order must be high or low, not {order!r}")
    val = np.asarray(values)
    cand = np.flatnonzero(np.isfinite(val))
    key = -val[cand] if order == "high" else val[cand]
    cand = cand[np.argsort(key, kind="stable")]
    kept: list[int] = []
    tree: cKDTree | None = None
    for c0 in range(0, len(cand), SITE_CHUNK):
        if len(kept) >= n:
            break
        ids = cand[c0 : c0 + SITE_CHUNK]
        xyz = np.asarray(points_mm[ids], np.float64)
        if tree is not None:
            near = tree.query_ball_point(xyz, spacing, return_length=True)
            ids, xyz = ids[near == 0], xyz[near == 0]
        new = len(kept)
        mine: list[Float] = []
        for v, p in zip(ids, xyz, strict=True):
            if mine and np.any(np.linalg.norm(np.asarray(mine) - p, axis=1) <= spacing):
                continue
            kept.append(int(v))
            mine.append(p)
            if len(kept) >= n:
                break
        if len(kept) > new:
            tree = cKDTree(np.asarray(points_mm[np.asarray(kept)], np.float64))
    return np.asarray(kept, np.int64)


def ranked_table(
    maps: dict[str, F32],
    name: str,
    points_mm: npt.NDArray[np.floating[Any]],
    keys: npt.ArrayLike,
    n: int = 200,
    spacing: float = 1.0,
    order: str = "high",
) -> list[dict[str, Any]]:
    """The ranked list of map `name` (highest first; order "low": lowest
    first) as rows: rank, vertex, edge key, RAS x, y, z (mm), the value, and
    the other maps at that vertex."""
    sites = ranked_sites(maps[name], points_mm, n, spacing, order)
    kk = np.asarray(keys)
    rows = []
    for i, v in enumerate(sites):
        p = np.asarray(points_mm[v], np.float64)
        row: dict[str, Any] = dict(
            rank=i + 1,
            vertex=int(v),
            edge_key=int(kk[v]),
            x=float(p[0]),
            y=float(p[1]),
            z=float(p[2]),
            value=float(maps[name][v]),
        )
        row.update({k: float(maps[k][v]) for k in MAPS if k != name})
        rows.append(row)
    return rows
