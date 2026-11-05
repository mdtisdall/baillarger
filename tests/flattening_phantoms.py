"""Phantoms for the tests of baillarger.flattening.

A plane grid of columns (roots on a plane, normals along +z) with label
intervals, a regular triangulation of the grid, and echo volumes in which the
laminar pattern of test_flatten.pattern is warped per column by a known
warp. The echoes go through the real ColumnVolumeSampler, on the numpy
reference device.

Layout. Column c = i * ny + j is the root at voxel (i, j, K_ROOT). The voxel
size is (sp, sp, step) mm. Voxel k along z is at z = k step - CAP_IN, so
every sample of every column is on a voxel centre along z.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from test_flatten import pattern

from baillarger import rays as R
from baillarger.blocks import ArraySource
from baillarger.column_volume import ColumnVolume, ColumnVolumeSampler
from baillarger.flattening import MARGIN_MM, SurfaceFaces

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

#: sample step of the columns, mm
STEP = 0.04
#: the offset of the grid in scanner space, mm
ORIGIN = (-1.0, 3.0, -R.CAP_IN)
#: the warp of the echo phantom (as test_flatten.grid_volume)
ALPHA = 0.2
BETA = 0.12
TOP = 4.6
#: the constant factor and the exponent of each echo of the phantom
ECHO_SCALE = (60.0, 40.0, 25.0)
ECHO_POWER = (1.0, 1.0, 0.7)


def affine(sp: float = 0.16, step: float = STEP) -> F64:
    """The voxel-to-scanner affine: diagonal (sp, sp, step), offset ORIGIN."""
    a = np.diag([sp, sp, step, 1.0])
    a[:3, 3] = ORIGIN
    return a


def grid_xy(nx: int, ny: int) -> tuple[F64, F64]:
    """x and y (n,) of the columns, each in [-1, 1], column c = i * ny + j
    (as test_flatten.grid_volume)."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    return xs.ravel(), ys.ravel()


def grid_edges(nx: int, ny: int) -> I64:
    """The edges of the triangulation of plane_grid as pairs of column
    indices, smaller first, sorted: along i, along j, and the diagonal from
    (i, j) to (i + 1, j + 1). Built from the grid, not from the faces."""
    out = []
    for i in range(nx):
        for j in range(ny):
            c = i * ny + j
            if i + 1 < nx:
                out.append((c, c + ny))
            if j + 1 < ny:
                out.append((c, c + 1))
            if i + 1 < nx and j + 1 < ny:
                out.append((c, c + ny + 1))
    return np.array(sorted(out), np.int64)


Interval = tuple[float, float, int]


def build_rays(
    root: npt.ArrayLike,
    normal: npt.ArrayLike,
    intervals: Sequence[Sequence[Interval]],
    k_in: npt.ArrayLike,
    k_out: npt.ArrayLike,
    step: float = STEP,
    vertex: npt.ArrayLike | None = None,
) -> R.Rays:
    """Rays of straight columns, with its own k_in and k_out for each column.

    intervals[c] lists (s0, s1, class) of column c. The label of an interval
    is set equal to its class code. The sample positions are root + s normal.
    The vertex ids default to the column indices."""
    x0 = np.asarray(root, np.float32).reshape(-1, 3)
    nrm = np.asarray(normal, np.float32).reshape(-1, 3)
    n = len(x0)
    ki = np.broadcast_to(np.asarray(k_in, np.int64), (n,))
    ko = np.broadcast_to(np.asarray(k_out, np.int64), (n,))
    ss, pp, counts = [], [], []
    for c in range(n):
        sk = np.arange(-ki[c], ko[c] + 1) * step
        ss.append(sk.astype(np.float32))
        pp.append(
            (x0[c].astype(np.float64) + sk[:, None] * nrm[c].astype(np.float64)).astype(
                np.float32
            )
        )
        counts.append(len(sk))
    i0 = np.concatenate([[0], np.cumsum([len(iv) for iv in intervals])])
    flat = [x for iv in intervals for x in iv]
    vtx = np.arange(n, dtype=np.int64) if vertex is None else np.asarray(vertex)
    roots = R.Roots(
        edge_key=np.arange(n, dtype=np.int64),
        vertex=vtx.astype(np.int64),
        owned=np.ones(n, np.bool_),
        root=x0,
        normal=nrm,
        normal_length=np.ones(n, np.float32),
    )
    return R.Rays(
        roots=roots,
        method="normal",
        step=step,
        k_in=ki.astype(np.int32),
        k_out=ko.astype(np.int32),
        sample_start=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        s=np.concatenate(ss),
        position=np.concatenate(pp),
        interval_start=i0.astype(np.int64),
        interval_s0=np.array([x[0] for x in flat], np.float32),
        interval_s1=np.array([x[1] for x in flat], np.float32),
        interval_label=np.array([x[2] for x in flat], np.int32),
        interval_class=np.array([x[2] for x in flat], np.uint8),
        derived_class=np.full(n, R.NEITHER, np.uint8),
        area_coef=np.zeros((0, 3), np.float64),
        focus=np.zeros((0, 2), np.float64),
        area=np.zeros(0, np.float32),
        volume=np.zeros(0, np.float32),
        past_focus=np.zeros(0, np.bool_),
    )


@dataclass(frozen=True)
class Plane:
    """A plane grid of columns."""

    nx: int
    ny: int
    sp: float
    step: float
    #: (4, 4) voxel to scanner
    affine: F64
    #: (n, 3) root in voxel coordinates, and in scanner mm
    voxel: F64
    root_mm: F64
    #: (n,) the root's vertex id in the mesh (not the column index)
    vertex: I64
    #: (n,) the end of the WM and the end of the cortex along each column, mm
    s_wm: F64
    s_top: F64
    rays: R.Rays
    #: (m, 3) triangles of vertex ids; surf holds the same faces in blocks
    faces: I64
    surf: SurfaceFaces

    @property
    def n(self) -> int:
        return self.nx * self.ny


def plane_grid(
    nx: int,
    ny: int,
    sp: float = 0.16,
    s_wm: float | npt.ArrayLike = 0.0,
    s_top: float | npt.ArrayLike = 3.0,
    step: float = STEP,
    margin: float = MARGIN_MM,
    blocks: int = 1,
) -> Plane:
    """Roots on a plane grid with sp mm between them, normals +z, one
    column for each node.

    Intervals of column c: WM from -CAP_IN to s_wm[c], CORTEX to s_top[c],
    OUTSIDE to CAP_OUT. The columns go from -e_in to e_out, with
    e_in = clip(margin - s_wm, 0, CAP_IN) and
    e_out = clip(s_top + margin, 0, CAP_OUT), floored to the step.

    The mesh has one vertex for each node, with the id (37 c + 5) mod n (a
    permutation of 0..n-1 when 37 and n have no common factor, which holds
    for the sizes that the tests use). Each cell (i, j) has two triangles,
    split by the diagonal from (i, j) to (i + 1, j + 1). The faces go into
    `blocks` blocks of consecutive cell columns i, each with the bounding
    box of the root voxels that its faces use."""
    n = nx * ny
    sw = np.broadcast_to(np.asarray(s_wm, np.float64), (n,)).copy()
    st = np.broadcast_to(np.asarray(s_top, np.float64), (n,)).copy()
    aff = affine(sp, step)
    k_root = int(round(R.CAP_IN / step))
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    voxel = np.stack([ii.ravel(), jj.ravel(), np.full(n, k_root)], axis=1).astype(
        np.float64
    )
    root_mm = voxel @ aff[:3, :3].T + aff[:3, 3]
    e_in = np.clip(margin - sw, 0.0, R.CAP_IN)
    e_out = np.clip(st + margin, 0.0, R.CAP_OUT)
    k_in = np.floor(e_in / step + 1e-6).astype(np.int64)
    k_out = np.floor(e_out / step + 1e-6).astype(np.int64)
    iv = [
        [
            (-R.CAP_IN, float(sw[c]), R.WM),
            (float(sw[c]), float(st[c]), R.CORTEX),
            (float(st[c]), R.CAP_OUT, R.OUTSIDE),
        ]
        for c in range(n)
    ]
    vertex = (37 * np.arange(n, dtype=np.int64) + 5) % n
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    rays = build_rays(root_mm, nrm, iv, k_in, k_out, step, vertex)
    faces, start, stop, bmin, bmax = [], [0], [], [], []
    edges_i = np.array_split(np.arange(nx - 1), blocks)
    for cells in edges_i:
        f = []
        for i in cells:
            for j in range(ny - 1):
                c = i * ny + j
                f.append((vertex[c], vertex[c + ny], vertex[c + ny + 1]))
                f.append((vertex[c], vertex[c + ny + 1], vertex[c + 1]))
        fa = np.array(f, np.int64).reshape(-1, 3)
        faces.append(fa)
        stop.append(start[-1] + len(fa))
        start.append(stop[-1])
        used = np.isin(vertex, fa)
        bmin.append(voxel[used].min(0) if used.any() else np.zeros(3))
        bmax.append(voxel[used].max(0) if used.any() else np.zeros(3))
    allf = np.concatenate(faces)
    surf = SurfaceFaces(
        allf,
        np.array(start[:-1], np.int64),
        np.array(stop, np.int64),
        np.array(bmin, np.float64),
        np.array(bmax, np.float64),
    )
    return Plane(
        nx, ny, sp, step, aff, voxel, root_mm, vertex, sw, st, rays, allf, surf
    )


# ------------------------------------------------------------ the warp


def horizon(x: F64, y: F64, u: npt.ArrayLike, top: float = TOP) -> F64:
    """The place s (mm) of the pattern depth u on each column:
    Z(u) = u (1 + ALPHA x) + BETA y sin(pi u / top), with the sine term
    taken at u clipped to [0, top] (so Z is continuous and linear outside).
    Shape (n, len(u))."""
    uu = np.atleast_1d(np.asarray(u, np.float64))[None]
    uc = np.clip(uu, 0.0, top)
    out: F64 = uu * (1 + ALPHA * x[:, None]) + BETA * y[:, None] * np.sin(
        np.pi * uc / top
    )
    return out


def warped_volumes(
    plane: Plane,
    x: F64,
    y: F64,
    z_max: float = R.CAP_OUT,
    top: float = TOP,
) -> list[npt.NDArray[np.float32]]:
    """Three echo volumes (nx + 1, ny + 1, nz) on the grid of the plane.

    Voxel k along z is at s = k step - CAP_IN. There, the echo a of
    column c is ECHO_SCALE[a] exp(ECHO_POWER[a] pattern(u)), where u solves
    horizon(x, y, u) = s on column c. The volume ends at s = z_max."""
    nz = int(round((z_max + R.CAP_IN) / plane.step)) + 1
    s = np.arange(nz) * plane.step - R.CAP_IN
    uf = np.linspace(-6.0, 14.0, 20001)
    vols = [np.zeros((plane.nx + 1, plane.ny + 1, nz), np.float32) for _ in ECHO_SCALE]
    zf = horizon(x, y, uf, top)
    for c in range(plane.n):
        i, j = divmod(c, plane.ny)
        p = pattern(np.interp(s, zf[c], uf))
        for a, v in enumerate(vols):
            v[i, j] = ECHO_SCALE[a] * np.exp(ECHO_POWER[a] * p)
    # one more node on the high side of x and y, a copy of the last one: the
    # sampler's validity needs a voxel above the last column
    for v in vols:
        v[-1] = v[-2]
        v[:, -1] = v[:, -2]
    return vols


def sample_plane(plane: Plane, vols: Sequence[npt.NDArray[np.float32]]) -> ColumnVolume:
    """The column volume of the plane's rays on the volumes, by the real
    sampler on the numpy device."""
    shape = vols[0].shape
    smp = ColumnVolumeSampler(
        [ArraySource(v) for v in vols],
        plane.affine,
        [plane.affine] * len(vols),
        (shape[0], shape[1], shape[2]),
    )
    return smp.sample(plane.rays, "numpy")
