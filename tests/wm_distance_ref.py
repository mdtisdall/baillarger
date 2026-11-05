"""Reference for baillarger.wm_distance: a brute-force geodesic distance through
tissue, and phantoms with known answers, kept only to check the fast marching
sampler against.

The sampler solves the eikonal equation on the voxel grid with scikit-fmm; the
source is the interface between WM voxels and cortex voxels (faces shared by a
WM voxel and a face-adjacent cortex voxel), values sit at voxel centres, paths
use free voxels (WM or cortex) only, and voxels with any other label are
obstacles. `geodesic_reference` computes the same absolute distance by
Dijkstra on a graph of refined sub-voxels joined by a long stencil, so its
error is bounded and known (see `reference_bound`), unlike the first-order
staircase of a 6-neighbour graph.

Metrication error of the stencil. Joining each sub-voxel to every sub-voxel
with offset (a, b, c), max(|a|, |b|, |c|) <= radius, gcd(|a|, |b|, |c|) = 1,
makes graph distance in free space a polyhedral norm. Its unit ball is the
convex hull of the stencil steps (each divided by its own length), so the worst
ratio of graph distance to Euclidean distance over all directions is
1 / (distance from the origin to the nearest facet of that hull). `stencil_bound`
computes this with scipy's convex hull. For radius 2 on an isotropic grid it is
0.0494: graph distances overestimate straight-line distances by at most 4.94 %
(radius 1 gives 12.8 %, radius 3 gives 2.5 %). The graph never underestimates
the true distance when the straight edges are free.

Source discretization. The sources are the sub-voxels touching an interface
face, started at half a sub-voxel (the distance of their centre to the face
plane). For axis-aligned interfaces this is exact; for a stair-stepped
interface the face-normal start differs from the true perpendicular distance
by up to about half a sub-voxel, and the value is read at the sub-voxel that
sits at the voxel centre. `reference_bound` allows one full largest sub-voxel
for both.
"""

import itertools
import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull

type Arr = npt.NDArray[Any]

WM = 1
CTX = 2


def _stencil(radius: int) -> list[tuple[int, ...]]:
    """Integer offsets (a, b, c) with max-norm <= radius, not all zero, gcd 1."""
    out = []
    for o in itertools.product(range(-radius, radius + 1), repeat=3):
        if any(o) and math.gcd(math.gcd(abs(o[0]), abs(o[1])), abs(o[2])) == 1:
            out.append(o)
    return out


def stencil_bound(
    radius: int, vox: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """Worst relative overestimate of a straight-line distance by shortest paths on
    the infinite stencil graph of the given radius, for grid spacing vox (default
    isotropic; the bound depends only on the ratios of vox).
    Computed as 1 / (inradius of the convex hull of unit step directions) - 1."""
    steps = np.array(_stencil(radius), dtype=np.float64) * np.asarray(vox)
    unit = steps / np.linalg.norm(steps, axis=1)[:, None]
    hull = ConvexHull(unit)
    inradius = float(np.min(-hull.equations[:, 3]))
    return 1.0 / inradius - 1.0


STENCIL_BOUND = 0.049417428813491204  # stencil_bound(2): radius 2, isotropic


def reference_bound(
    d_mm: float,
    vox: tuple[float, float, float],
    refine: int,
    radius: int,
) -> float:
    """Allowed absolute error (mm) of `geodesic_reference` against a true
    distance d_mm: the stencil overestimate d_mm * stencil_bound(radius, vox)
    plus max(vox) / refine for the source and read-out discretization (a half
    sub-voxel each)."""
    return d_mm * stencil_bound(radius, vox) + max(vox) / refine


def geodesic_reference(
    wm: Arr,
    ctx: Arr,
    vox: tuple[float, float, float],
    refine: int = 3,
    radius: int = 2,
) -> Arr:
    """Absolute distance (mm) from each free voxel centre to the WM/cortex
    interface through free voxels of its own side, by Dijkstra on refine^3
    sub-voxels per voxel (refine odd). NaN at obstacles, inf where the interface
    is not reached. Error bound: see `reference_bound`."""
    if refine % 2 != 1:
        raise ValueError("refine must be odd")
    shape = wm.shape
    lab = np.zeros(shape, dtype=np.int8)
    lab[wm] = WM
    lab[ctx] = CTX
    for ax in range(3):
        lab = np.repeat(lab, refine, axis=ax)
    sshape = lab.shape
    n = lab.size
    h = np.asarray(vox, dtype=np.float64) / refine
    ids = np.arange(n, dtype=np.int64).reshape(sshape)

    rows: list[Arr] = []
    cols: list[Arr] = []
    wts: list[Arr] = []
    nsamp = 4 * radius
    for off in _stencil(radius):
        if next(o for o in off if o != 0) < 0:
            continue  # the other half comes from directed=False
        src = tuple(
            slice(0, s - o) if o >= 0 else slice(-o, s)
            for o, s in zip(off, sshape, strict=True)
        )
        dst = tuple(
            slice(o, s) if o >= 0 else slice(0, s + o)
            for o, s in zip(off, sshape, strict=True)
        )
        lsrc = lab[src]
        ok = (lsrc > 0) & (lab[dst] == lsrc)
        for k in range(1, nsamp):
            sh = np.rint(np.asarray(off) * (k / nsamp)).astype(int)
            mid = tuple(
                slice(s.start + d, s.stop + d)
                if s.start is not None
                else slice(d, s.stop + d)
                for s, d in zip(src, sh, strict=True)
            )
            ok &= lab[mid] == lsrc
        length = float(np.linalg.norm(np.asarray(off) * h))
        r = ids[src][ok]
        rows.append(r)
        cols.append(ids[dst][ok])
        wts.append(np.full(r.size, length))

    # sources: sub-voxels with a face shared with the other side
    init = np.full(n, np.inf)
    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = slice(0, -1)
        hi[ax] = slice(1, None)
        a = lab[tuple(lo)]
        b = lab[tuple(hi)]
        cross = (a > 0) & (b > 0) & (a != b)
        for sel, part in ((tuple(lo), a), (tuple(hi), b)):
            del part
            idx = ids[sel][cross]
            init[idx] = np.minimum(init[idx], h[ax] / 2.0)
    src_nodes = np.flatnonzero(np.isfinite(init))
    rows.append(np.full(src_nodes.size, n, dtype=np.int64))
    cols.append(src_nodes)
    wts.append(init[src_nodes])

    r_all = np.concatenate(rows)
    c_all = np.concatenate(cols)
    w_all = np.concatenate(wts)
    graph = coo_matrix((w_all, (r_all, c_all)), shape=(n + 1, n + 1)).tocsr()
    dist = dijkstra(graph, directed=False, indices=n)[:n].reshape(sshape)

    c = refine // 2
    out = np.array(dist[c::refine, c::refine, c::refine], dtype=np.float64)
    out[~(wm | ctx)] = np.nan
    return out


def plane_phantom(
    shape: tuple[int, int, int], axis: int, level: int
) -> tuple[Arr, Arr]:
    """WM where the index along axis is < level, cortex elsewhere. The interface
    lies at index level - 0.5."""
    i = np.arange(shape[axis]).reshape([-1 if a == axis else 1 for a in range(3)])
    wm = np.broadcast_to(i < level, shape).copy()
    return wm, ~wm


def plane_exact(
    shape: tuple[int, int, int],
    axis: int,
    level: int,
    vox: tuple[float, float, float],
) -> Arr:
    """Signed exact distance (mm) for `plane_phantom`: (i - level + 0.5) * vox[axis],
    negative in WM, positive in cortex."""
    i = np.arange(shape[axis]).reshape([-1 if a == axis else 1 for a in range(3)])
    return np.broadcast_to((i - level + 0.5) * vox[axis], shape).astype(np.float64)


def oblique_plane_phantom(
    shape: tuple[int, int, int],
    normal: tuple[float, float, float],
    offset: float,
    vox: tuple[float, float, float],
) -> tuple[Arr, Arr]:
    """WM where the voxel centre (index * vox, mm) has coordinate along the unit
    normal (normal is in mm space, normalised here) below offset (mm); cortex
    elsewhere. The staircase makes the exact distance only approximate:
    n . x - offset, within about max(vox) / 2."""
    nrm = np.asarray(normal, dtype=np.float64)
    nrm = nrm / np.linalg.norm(nrm)
    g = np.indices(shape).astype(np.float64)
    s = sum(g[a] * vox[a] * nrm[a] for a in range(3))
    wm = np.asarray(s < offset)
    return wm, ~wm


def sulcus_phantom(shape: tuple[int, int, int]) -> tuple[Arr, Arr, Arr]:
    """Returns (wm, ctx, detour). Axis 0 points up from the WM.

    WM: indices x < 6 and y < w, with w = shape[1] // 2 - 1. Obstacle wall:
    y in [w, w + 2) (2 voxels thick), x < shape[0] - 4, the full extent of
    axis 2; it stands on the bottom of the volume beside the WM and leaves a gap
    of 4 voxels at the top. Cortex: every other voxel. Cortex voxels on the near
    side (y < w) sit straight above the WM. Cortex voxels on the far side
    (y >= w + 2) have no WM beneath them and must go up through the gap, over
    the wall's top edge and down to the WM top at x = 5.5.

    detour: far-side cortex voxels with x >= 6; their geodesic is strictly
    longer than the straight-line distance (x - 5.5) * vox[0] to the plane
    of the interface."""
    nx, ny, _ = shape
    x, y, _z = np.indices(shape)
    w = ny // 2 - 1
    wm = (x < 6) & (y < w)
    wall = (y >= w) & (y < w + 2) & (x < nx - 4)
    ctx = ~wm & ~wall
    detour = ctx & (y >= w + 2) & (x >= 6)
    return wm, ctx, detour


def obstacle_label_phantom(shape: tuple[int, int, int]) -> tuple[Arr, Arr, Arr]:
    """Returns (wm, ctx, blocked). Plane phantom along axis 0 with level 6
    (WM x < 6), plus a block of a third label (neither WM nor cortex) at
    x in [7, 10), y and z in [n // 4, 3n // 4) of their extents. blocked: cortex
    voxels at x >= 10 inside the block's y, z range, whose straight path to the
    WM crosses the block and must go round it."""
    nx, ny, nz = shape
    x, y, z = np.indices(shape)
    wm = x < 6
    inside = (y >= ny // 4) & (y < 3 * ny // 4) & (z >= nz // 4) & (z < 3 * nz // 4)
    block = (x >= 7) & (x < 10) & inside
    ctx = ~wm & ~block
    blocked = ctx & inside & (x >= 10)
    return wm, ctx, blocked
