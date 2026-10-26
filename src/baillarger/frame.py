"""The columns and depth frame: streamlines from the fixed white surface.

The frame depends only on the WM and on the missing tissue, not on the pial
surface, so it does not change in the outer loop (PLAN_LAYER_REFINE.md,
"Columns and depth frame"):

- Domain: cortex, seams and fluid (rim labels 3 and 1), without missing
  tissue. Cortex and fluid are the same to the frame, so changing the pial
  surface changes nothing here.
- White mesh: marching tetrahedra on the WM label smoothed by sigma voxels,
  at 0.5. Vertex normals point out of the WM.
- Distance D: the distance (mm) from the WM through the domain, by fast
  marching (scikit-fmm, second order); WM voxels next to missing tissue are
  sources too, but the missing tissue itself is an obstacle. Inside the WM D is
  negative (the signed distance), so the smoothed gradient is defined across
  the white surface.
- Streamlines: from each white vertex, first enter_mm along its normal, then
  along the gradient of D smoothed by sigma voxels (normalized convolution over
  the domain and the WM), with midpoint (RK2) steps of step_mm. A streamline
  stops at the ridge of D (the gradient reverses or fades: where streamlines
  from facing banks meet, in fused cortex the medial surface between them),
  where it would enter missing tissue, the WM or other labels, at the crop
  faces, or where D reaches d_max.
- Missing tissue: blocks cut out of the specimen before the scan, which the
  aseg labels as tissue (imputed) but where the FLASH is black
  (missing_tissue). It is not in the domain, its WM is no source of D, and a
  streamline stops on entering it. Such a streamline is cut (path_cut): its
  column is truncated by the cut, not by the pial surface, so the voxels
  nearest to it take no column and it does not count for T_c; its column is
  flagged (column_cut). The white mesh is made from all the WM, imputed WM too,
  so the edge of a cut does not become a false white surface.
- Columns: geodesic Voronoi cells of the white mesh. Seeds are chosen by
  farthest-point sampling along the mesh edges until every vertex is within
  column_mm / 2 of a seed, and each vertex goes to its nearest seed along the
  mesh. A column never straddles two facing banks.
- Voxels: each domain voxel goes to the nearest streamline point within
  reach_mm (so to the column of the nearest streamline), and its depth d is
  the arc length there, plus its offset along the streamline. Streamlines fan
  out over gyral crowns, so neighbouring ones can be 1 mm apart there; reach_mm
  (default 1.5 mm) bridges that. Voxels that no streamline comes within
  reach_mm of have column -1 and depth NaN.

Relative depth r = d / T_c (relative_depth) uses the column thickness T_c,
which the outer loop updates (column_thickness gives it from a tissue mask).
"""

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import skfmm
from scipy import ndimage as ndi
from scipy import sparse
from scipy.spatial import cKDTree

from baillarger._jit import njit
from baillarger.mesh import marching_tetrahedra

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I32 = npt.NDArray[np.int32]
type I64 = npt.NDArray[np.int64]

#: Why a streamline ended (Frame.path_end).
END_RIDGE, END_MISSING, END_OBSTACLE, END_CROP, END_DMAX, END_LENGTH = range(6)
END_NAMES = ("ridge", "missing", "obstacle", "crop", "d_max", "length")


@dataclass(frozen=True)
class Frame:
    """The columns and depth frame of one patch. Points are in voxel
    coordinates of the patch, lengths in mm."""

    voxel_mm: tuple[float, float, float]
    #: white mesh: vertices (V, 3), faces (F, 3), unit outward normals (V, 3)
    #: in mm space
    verts: F64
    faces: I64
    normals: F64
    #: column of each vertex, and the seed vertex of each column
    vertex_column: I64
    column_seed: I64
    #: distance from the WM through the domain (mm); NaN outside the domain
    #: and the WM, negative in the WM
    distance: F32
    #: streamlines, one per vertex, as ragged arrays: the points of path i are
    #: path_points[path_start[i]:path_start[i + 1]], with arc lengths path_arc
    #: (mm), one point every sample_mm or less
    path_start: I64
    path_points: F32
    path_arc: F32
    path_end: npt.NDArray[np.int8]
    #: streamlines that end at missing tissue: their columns are truncated, so
    #: they take no voxels and do not count for the thickness
    path_cut: Bool
    #: columns with at least one cut streamline
    column_cut: Bool
    #: column (-1 none), streamline (-1 none) and depth d (mm, NaN none) of
    #: each voxel
    voxel_column: I32
    voxel_path: I32
    depth: F32

    @property
    def n_columns(self) -> int:
        return len(self.column_seed)

    def path_length(self) -> F32:
        """Arc length of each streamline (mm)."""
        s = self.path_start
        out = np.zeros(len(s) - 1, np.float32)
        has = s[1:] > s[:-1]
        out[has] = self.path_arc[s[1:][has] - 1]
        return out


def missing_tissue(
    rim: npt.NDArray[np.integer[Any]],
    flash: npt.ArrayLike,
    voxel_mm: tuple[float, float, float],
    frac: float = 0.25,
    sigma: float = 2.0,
    min_vox: int = 250,
    margin_mm: float = 0.3,
) -> Bool:
    """Tissue the aseg labels (WM or cortex) where the FLASH is black: blocks
    cut out of the specimen before the scan, imputed by the segmentation.

    The FLASH smoothed by sigma voxels is below frac x the cortex median, in
    connected blocks of at least min_vox voxels, grown by margin_mm.
    """
    f = np.asarray(flash, np.float64)
    tis = (rim == 2) | (rim == 3)
    gm_med = np.median(f[rim == 3])
    blk = tis & (ndi.gaussian_filter(f, sigma) < frac * gm_med)
    lab, n = ndi.label(blk)
    big = np.zeros(n + 1, bool)
    big[1:] = np.bincount(lab.ravel(), minlength=n + 1)[1:] >= min_vox
    miss: Bool = big[lab]
    it = int(round(margin_mm / min(voxel_mm)))
    if it:
        miss = ndi.binary_dilation(miss, iterations=it) & tis
    return miss


# ---------------------------------------------------------------- white mesh


def _vertex_normals(v: F64, faces: I64) -> F64:
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    out: F64 = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)
    return out


def _edges(faces: I64) -> I64:
    e = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1
    )
    out: I64 = np.unique(e, axis=0)
    return out


def white_mesh(
    wm: Bool,
    voxel_mm: tuple[float, float, float],
    sigma: float = 1.0,
    normal_iters: int = 10,
) -> tuple[F64, I64, F64]:
    """Marching tetrahedra on the WM label smoothed by sigma voxels, at 0.5.

    Returns vertices (voxel coordinates), faces and unit vertex normals in mm
    space, pointing out of the WM: area-weighted face normals, then
    normal_iters steps of Laplacian smoothing along the mesh edges.
    """
    vox = np.asarray(voxel_mm, float)
    field = ndi.gaussian_filter(np.asarray(wm, np.float64), sigma)
    verts, faces = marching_tetrahedra(field, 0.5)
    if not len(faces):
        return verts, faces, np.zeros_like(verts)
    # marching_tetrahedra orients the faces toward increasing field: into the WM
    n = -_vertex_normals(verts * vox, faces)
    E = _edges(faces)
    nv = len(verts)
    C = sparse.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(nv, nv))
    A = (C + C.T).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)[:, None]
    for _ in range(normal_iters):
        n = 0.5 * n + 0.5 * (A @ n) / deg
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return verts, faces, n


# ---------------------------------------------------------------- distance


def domain_mask(rim: npt.NDArray[np.integer[Any]], missing: Bool | None) -> Bool:
    """Cortex and fluid (rim 3 and 1), without missing tissue."""
    dom: Bool = (rim == 1) | (rim == 3)
    if missing is not None:
        dom &= ~missing
    return dom


def wm_distance(
    rim: npt.NDArray[np.integer[Any]],
    voxel_mm: tuple[float, float, float],
    missing: Bool | None = None,
) -> F32:
    """Signed distance (mm) from the WM through the domain, by second-order
    fast marching: positive in the domain, negative in the WM, NaN elsewhere
    (missing tissue, other labels, and domain voxels with no path to the WM)."""
    wm = rim == 2
    if missing is not None:
        wm = wm & ~missing
    dom = domain_mask(rim, missing)
    free = dom | wm
    if not wm.any() or not dom.any():
        return np.full(rim.shape, np.nan, np.float32)
    phi = np.ma.MaskedArray(np.where(wm, -1.0, 1.0), mask=~free)
    D = skfmm.distance(phi, dx=np.asarray(voxel_mm, float), order=2)
    out = np.ma.filled(np.ma.MaskedArray(D).astype(np.float64), np.nan)
    return np.where(free, out, np.nan).astype(np.float32)


def _smoothed_gradient(
    D: F32, sigma: float, voxel_mm: tuple[float, float, float]
) -> F32:
    """Gradient (per mm) of D smoothed by sigma voxels with normalized
    convolution over the voxels where D is defined; shape (*shape, 3)."""
    ok = np.isfinite(D)
    w = ndi.gaussian_filter(ok.astype(np.float64), sigma)
    Ds = ndi.gaussian_filter(np.where(ok, D, 0.0).astype(np.float64), sigma)
    Ds = np.where(w > 1e-3, Ds / np.maximum(w, 1e-3), 0.0)
    vox = np.asarray(voxel_mm, float)
    G = np.stack([np.gradient(Ds, axis=k) / vox[k] for k in range(3)], -1)
    return G.astype(np.float32)


# ---------------------------------------------------------------- streamlines


@njit
def _trilinear(vol: F32, x: float, y: float, z: float) -> float:
    """vol at (x, y, z), clamped to the array."""
    nx, ny, nz = vol.shape
    x = min(max(x, 0.0), nx - 1.0)
    y = min(max(y, 0.0), ny - 1.0)
    z = min(max(z, 0.0), nz - 1.0)
    i = min(int(x), nx - 2) if nx > 1 else 0
    j = min(int(y), ny - 2) if ny > 1 else 0
    k = min(int(z), nz - 2) if nz > 1 else 0
    fx, fy, fz = x - i, y - j, z - k
    i1, j1, k1 = min(i + 1, nx - 1), min(j + 1, ny - 1), min(k + 1, nz - 1)
    c00 = vol[i, j, k] * (1 - fx) + vol[i1, j, k] * fx
    c10 = vol[i, j1, k] * (1 - fx) + vol[i1, j1, k] * fx
    c01 = vol[i, j, k1] * (1 - fx) + vol[i1, j, k1] * fx
    c11 = vol[i, j1, k1] * (1 - fx) + vol[i1, j1, k1] * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return float(c0 * (1 - fz) + c1 * fz)


@njit
def _direction(
    G: F32, x: float, y: float, z: float, gmin: float
) -> tuple[float, float, float, bool]:
    """Unit direction of G at (x, y, z), and whether |G| >= gmin."""
    gx = _trilinear(G[..., 0], x, y, z)
    gy = _trilinear(G[..., 1], x, y, z)
    gz = _trilinear(G[..., 2], x, y, z)
    n = (gx * gx + gy * gy + gz * gz) ** 0.5
    if n < gmin:
        return 0.0, 0.0, 0.0, False
    return gx / n, gy / n, gz / n, True


@njit
def _label(code: npt.NDArray[np.int8], x: float, y: float, z: float) -> int:
    """code at the voxel nearest (x, y, z); -1 outside the array."""
    i, j, k = int(np.floor(x + 0.5)), int(np.floor(y + 0.5)), int(np.floor(z + 0.5))
    if (
        i < 0
        or j < 0
        or k < 0
        or i >= code.shape[0]
        or j >= code.shape[1]
        or k >= code.shape[2]
    ):
        return -1
    return int(code[i, j, k])


@njit
def _trace(
    start: F64,
    normal: F64,
    G: F32,
    D: F32,
    code: npt.NDArray[np.int8],
    vox: F64,
    step: float,
    enter: float,
    d_max: float,
    gmin: float,
    max_len: float,
    stride: int,
    pts: F32,
    arc: F32,
    npts: I64,
    end: npt.NDArray[np.int8],
) -> None:
    """Trace each streamline into pts / arc (one row per path, a point every
    stride steps and the last point), npts and end. code: 0 domain, 1 WM,
    2 missing tissue, 3 other labels. The first step (along the normal) may
    still land in a WM voxel."""
    for p in range(len(start)):
        x, y, z = start[p, 0], start[p, 1], start[p, 2]
        ux, uy, uz = normal[p, 0], normal[p, 1], normal[p, 2]
        pts[p, 0, 0], pts[p, 0, 1], pts[p, 0, 2] = x, y, z
        arc[p, 0] = 0.0
        n = 1
        s = 0.0
        k = 0
        h = enter
        why = END_LENGTH
        while True:
            # the next point: along the normal first, then along the field
            if k == 0:
                dx, dy, dz = ux, uy, uz
            else:
                ax, ay, az, ok = _direction(G, x, y, z, gmin)
                if not ok or ax * ux + ay * uy + az * uz < 0:
                    why = END_RIDGE
                    break
                mx = x + 0.5 * h * ax / vox[0]
                my = y + 0.5 * h * ay / vox[1]
                mz = z + 0.5 * h * az / vox[2]
                bx, by, bz, ok = _direction(G, mx, my, mz, gmin)
                if not ok or bx * ax + by * ay + bz * az < 0:
                    bx, by, bz = ax, ay, az
                dx, dy, dz = bx, by, bz
            nx = x + h * dx / vox[0]
            ny = y + h * dy / vox[1]
            nz = z + h * dz / vox[2]
            lab = _label(code, nx, ny, nz)
            if lab < 0:
                why = END_CROP
                break
            if lab == 2:
                why = END_MISSING
                break
            if lab == 3 or (lab == 1 and k > 0):
                why = END_OBSTACLE
                break
            if k > 0 and _trilinear(D, nx, ny, nz) >= d_max:
                why = END_DMAX
                break
            if s + h > max_len:
                why = END_LENGTH
                break
            x, y, z = nx, ny, nz
            ux, uy, uz = dx, dy, dz
            s += h
            k += 1
            h = step
            if k % stride == 0 and n < pts.shape[1] - 1:
                pts[p, n, 0], pts[p, n, 1], pts[p, n, 2] = x, y, z
                arc[p, n] = s
                n += 1
        # the last point, if it was not kept
        if k > 0 and arc[p, n - 1] < s:
            pts[p, n, 0], pts[p, n, 1], pts[p, n, 2] = x, y, z
            arc[p, n] = s
            n += 1
        npts[p] = n
        end[p] = why


# ---------------------------------------------------------------- columns


@njit
def _farthest_point_voronoi(
    indptr: I64, indices: I64, length: F64, radius: float
) -> tuple[I64, I64]:
    """Farthest-point sampling on a graph (CSR, edge lengths): seeds until every
    node is within radius of one; returns each node's nearest seed (index into
    the seeds) and the seeds. The farthest node comes from a lazy max-heap of
    (-distance, node): stale entries are skipped when popped."""
    n = len(indptr) - 1
    dist = np.full(n, np.inf)
    owner = -np.ones(n, np.int64)
    seeds: list[int] = []
    far = [(-np.inf, np.int64(v)) for v in range(n)]
    heapq.heapify(far)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    while len(far):
        key, v = heapq.heappop(far)
        if -key != dist[v]:
            continue  # stale: v got nearer to a seed since
        if dist[v] <= radius:
            break
        c = len(seeds)
        seeds.append(int(v))
        dist[v] = 0.0
        owner[v] = c
        heapq.heappush(heap, (0.0, np.int64(v)))
        while len(heap):
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for e in range(indptr[u], indptr[u + 1]):
                w = indices[e]
                nd = d + length[e]
                if nd < dist[w]:
                    dist[w] = nd
                    owner[w] = c
                    heapq.heappush(heap, (nd, w))
                    heapq.heappush(far, (-nd, w))
    return owner, np.array(seeds, np.int64)


def mesh_columns(
    verts: F64,
    faces: I64,
    voxel_mm: tuple[float, float, float],
    column_mm: float = 0.7,
) -> tuple[I64, I64]:
    """Geodesic Voronoi cells of the mesh, about column_mm across: the column
    of each vertex and the seed vertex of each column."""
    if not len(faces):
        return -np.ones(len(verts), np.int64), np.zeros(0, np.int64)
    E = _edges(faces)
    vox = np.asarray(voxel_mm, float)
    w = np.linalg.norm((verts[E[:, 0]] - verts[E[:, 1]]) * vox, axis=1)
    nv = len(verts)
    A = sparse.coo_matrix(
        (np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
        shape=(nv, nv),
    ).tocsr()
    A.sort_indices()
    return _farthest_point_voronoi(
        A.indptr.astype(np.int64),
        A.indices.astype(np.int64),
        A.data.astype(np.float64),
        0.5 * column_mm,
    )


# ---------------------------------------------------------------- voxels


def _assign(
    path_start: I64,
    pts: F32,
    arc: F32,
    path_column: I64,
    path_cut: Bool,
    dom: Bool,
    vox: F64,
    reach: float,
) -> tuple[I32, I32, F32]:
    """Each domain voxel takes the nearest streamline point within reach: its
    column, its path, and its arc length plus the voxel's offset along the
    path there. A voxel whose nearest streamline is cut (path_cut) takes
    nothing: it is next to missing tissue, where the column is truncated."""
    shape = dom.shape
    col = -np.ones(shape, np.int32)
    path = -np.ones(shape, np.int32)
    depth = np.full(shape, np.nan, np.float32)
    npath = len(path_start) - 1
    counts = np.diff(path_start)
    pid = np.repeat(np.arange(npath), counts)
    # only paths with two points or more: a lone start point has no direction
    use = counts[pid] >= 2
    if not use.any():
        return col, path, depth
    P = pts.astype(np.float64) * vox
    # tangents from the neighbouring points of the same path
    q = np.arange(len(P))
    a, b = path_start[pid], path_start[pid + 1] - 1
    t = P[np.minimum(q + 1, b)] - P[np.maximum(q - 1, a)]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    idx = np.flatnonzero(use)
    tree = cKDTree(P[idx])
    V = np.argwhere(dom)
    dist, jt = tree.query(V * vox, k=1, distance_upper_bound=reach)
    hit = np.isfinite(dist)
    V, j = V[hit], idx[np.asarray(jt)[hit]]
    ok = ~path_cut[pid[j]]
    V, j = V[ok], j[ok]
    off = np.sum((V * vox - P[j]) * t[j], axis=1)
    vi = (V[:, 0], V[:, 1], V[:, 2])
    col[vi] = path_column[pid[j]]
    path[vi] = pid[j]
    depth[vi] = (arc[j] + off).astype(np.float32)
    return col, path, depth


def build_frame(
    rim: npt.NDArray[np.integer[Any]],
    voxel_mm: tuple[float, float, float],
    missing: Bool | None = None,
    sigma: float = 1.0,
    normal_iters: int = 10,
    step_mm: float = 0.04,
    enter_mm: float = 0.08,
    d_max: float = 4.0,
    gmin: float = 0.3,
    sample_mm: float | None = None,
    column_mm: float = 0.7,
    reach_mm: float = 1.5,
) -> Frame:
    """The frame of one patch from its rim labels (only the WM and the domain
    matter, see the module notes) and its missing tissue.

    gmin: the smoothed |grad D| (per mm, about 1 away from ridges) below which
    a streamline counts as at a ridge. sample_mm: the spacing of the kept
    streamline points (default: the smallest voxel size). Streamlines are at
    most 2 x d_max long.
    """
    vox = np.asarray(voxel_mm, np.float64)
    vt = (float(vox[0]), float(vox[1]), float(vox[2]))
    rim = np.asarray(rim)
    wm = rim == 2
    verts, faces, normals = white_mesh(wm, vt, sigma, normal_iters)
    vertex_column, column_seed = mesh_columns(verts, faces, vt, column_mm)
    D = wm_distance(rim, vt, missing)
    G = _smoothed_gradient(D, sigma, vt)
    dom = domain_mask(rim, missing)
    code = np.full(rim.shape, 3, np.int8)
    code[rim == 2] = 1
    code[dom] = 0
    if missing is not None:
        code[missing] = 2
    Dt = np.where(np.isfinite(D), D, np.inf).astype(np.float32)
    sample = float(vox.min()) if sample_mm is None else sample_mm
    stride = max(1, int(round(sample / step_mm)))
    max_len = 2.0 * d_max
    width = int(np.ceil(max_len / (stride * step_mm))) + 3
    nv = len(verts)
    pts = np.full((nv, width, 3), np.nan, np.float32)
    arc = np.full((nv, width), np.nan, np.float32)
    npts = np.zeros(nv, np.int64)
    end = np.zeros(nv, np.int8)
    _trace(
        verts.astype(np.float64),
        normals.astype(np.float64),
        G,
        Dt,
        code,
        vox,
        step_mm,
        enter_mm,
        d_max,
        gmin,
        max_len,
        stride,
        pts,
        arc,
        npts,
        end,
    )
    path_start = np.r_[0, np.cumsum(npts)].astype(np.int64)
    # cut: the streamline ends at missing tissue (also when it starts there),
    # so its column is truncated
    path_cut = end == END_MISSING
    keep = np.arange(width)[None, :] < npts[:, None]
    path_points = pts[keep]
    path_arc = arc[keep]
    del pts, arc
    voxel_column, voxel_path, depth = _assign(
        path_start,
        path_points,
        path_arc,
        vertex_column,
        path_cut,
        dom,
        vox,
        reach_mm,
    )
    column_cut = np.zeros(len(column_seed), bool)
    column_cut[vertex_column[path_cut & (vertex_column >= 0)]] = True
    return Frame(
        voxel_mm=vt,
        verts=verts,
        faces=faces,
        normals=normals,
        vertex_column=vertex_column,
        column_seed=column_seed,
        distance=D,
        path_start=path_start,
        path_points=path_points,
        path_arc=path_arc,
        path_end=end,
        path_cut=path_cut,
        column_cut=column_cut,
        voxel_column=voxel_column,
        voxel_path=voxel_path,
        depth=depth,
    )


def column_thickness(frame: Frame, tissue: Bool) -> F64:
    """T_c of each column: the median, over its streamlines, of the depth of
    their last tissue voxel (NaN for a column with no tissue voxel). Cut
    streamlines take no voxels, so they do not count."""
    m = tissue & (frame.voxel_path >= 0)
    npath = len(frame.path_start) - 1
    last = np.full(npath, -np.inf)
    np.maximum.at(last, frame.voxel_path[m], frame.depth[m].astype(np.float64))
    has = np.isfinite(last)
    col = frame.vertex_column[has]
    val = last[has]
    T = np.full(frame.n_columns, np.nan)
    if len(val):
        order = np.lexsort((val, col))
        col, val = col[order], val[order]
        starts = np.r_[0, np.flatnonzero(np.diff(col)) + 1]
        for a, b in zip(starts, np.r_[starts[1:], len(col)], strict=True):
            T[col[a]] = np.median(val[a:b])
    return T


def relative_depth(frame: Frame, thickness: npt.ArrayLike) -> F32:
    """r = d / T_c for each voxel (NaN where the voxel has no column or its
    column no thickness)."""
    T = np.asarray(thickness, np.float64)
    c = frame.voxel_column
    r = np.full(c.shape, np.nan, np.float32)
    m = c >= 0
    r[m] = (frame.depth[m] / T[c[m]]).astype(np.float32)
    return r
