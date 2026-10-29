"""The streamlines at a point of the white surface, and the surfaces around it (plan:
reports/Crossing paths and contacts from laminar profiles.md, *A pia editor on an
inflated surface*).

Whatever tracer made the column volume (its `tracer_name`, if saved), the traces are
shown as they are: the positions the column volume saved, nothing re-traced.
- From the point: every trace rooted within `radius` of the clicked vertex along the
  white mesh (geodesic, so that roots on the facing bank of a sulcus, near in space but
  far along the mesh, are not counted).
- To the point: every other trace whose last sample is within `radius` of the point in
  space (mostly crossing traces from the facing bank that end at this one).
- The white mesh cropped to a box, and the boundary of a label (the cortex) contoured
  by marching tetrahedra (smoothed by sigma voxels, for display only), clipped to the
  box and left open where the box cuts it.
- TCK: the MRtrix track format, which NiiVue reads.
Everything is 3D.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from baillarger.mesh import marching_tetrahedra
from baillarger.streamlines import STEP_MM

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def geodesic(G: csr_matrix, root: int, radius: float) -> Float:
    """Distances (mm) along the mesh graph G from the vertex root; inf beyond radius."""
    d: Float = dijkstra(G, indices=root, limit=radius)
    return d


def select(
    G: csr_matrix,
    root: int,
    x0: Float,
    tv: npt.ArrayLike,
    pos: npt.NDArray[Any],
    ns: npt.NDArray[Any],
    vox: npt.ArrayLike,
    radius: float = 1.0,
) -> tuple[Int, Int]:
    """(from, to): traces rooted within radius of the vertex root along G, and the
    other traces whose last sample lies within radius of x0 (mm). tv: each trace's root
    vertex (-1: none on this mesh); pos (n, nmax, 3) in voxels; ns the sample counts;
    vox the voxel size (mm)."""
    near = np.isfinite(geodesic(G, root, radius))
    t = np.asarray(tv)
    fr = (t >= 0) & near[np.maximum(t, 0)]
    ends = pos[np.arange(len(ns)), np.maximum(ns, 1) - 1].astype(float) * np.asarray(
        vox
    )
    to = ~fr & (ns > 0) & (np.linalg.norm(ends - x0, axis=1) <= radius)
    return np.flatnonzero(fr), np.flatnonzero(to)


def crop_mesh(
    v: Float, f: npt.NDArray[Any], lo: npt.ArrayLike, hi: npt.ArrayLike
) -> tuple[Float, Int, Int]:
    """The faces of (v, f) with every vertex in the box [lo, hi]: (vertices, faces
    reindexed, the kept vertices' indices in v)."""
    inside = np.all((v >= np.asarray(lo, float)) & (v <= np.asarray(hi, float)), axis=1)
    fk = f[inside[f].all(1)]
    keep = np.unique(fk).astype(np.int64)
    new = -np.ones(len(v), np.int64)
    new[keep] = np.arange(keep.size)
    return v[keep], new[fk], keep


def contour(mask: npt.NDArray[np.bool_], sigma: float) -> tuple[Float, Int]:
    """Marching tetrahedra at 0.5 of the label smoothed by sigma voxels (vertices in
    voxels), faces oriented out of the label."""
    F = (
        ndi.gaussian_filter(mask.astype(np.float64), sigma)
        if sigma > 0
        else mask.astype(np.float64)
    )
    v, f = marching_tetrahedra(F, 0.5)
    return v, np.asarray(f, np.int64)[:, [0, 2, 1]].copy()


def label_surface(
    mask: npt.NDArray[np.bool_],
    lo: npt.ArrayLike,
    hi: npt.ArrayLike,
    sigma: float = 1.0,
) -> tuple[Float, Int]:
    """The boundary of a label inside the box [lo, hi) of voxels: the contour of the
    label smoothed by sigma voxels (from a crop padded so that the smoothing sees past
    the box), clipped to the box. Returns (vertices in voxels, faces)."""
    pad = int(np.ceil(3 * sigma)) + 1
    a = np.maximum(np.asarray(lo, int) - pad, 0)
    b = np.minimum(np.asarray(hi, int) + pad, mask.shape)
    crop = mask[a[0] : b[0], a[1] : b[1], a[2] : b[2]]
    if not crop.any() or crop.all():
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    v, f = contour(crop, sigma)
    v = v + a
    v, f, _ = crop_mesh(v, f, np.asarray(lo, float), np.asarray(hi, float) - 1)
    return v, f


def at_depth(
    pos: npt.NDArray[Any],
    ns: npt.NDArray[Any],
    depth: npt.ArrayLike,
    vox: npt.ArrayLike,
) -> Float:
    """Per trace the point (mm) at arc length depth (mm) along it, interpolated between
    samples; nan where depth is not finite or is past the trace's end."""
    s = np.asarray(depth, float) / STEP_MM
    ok = np.isfinite(s) & (s >= 0) & (s <= ns - 1)
    i = np.clip(np.floor(np.nan_to_num(s)).astype(int), 0, pos.shape[1] - 2)
    w = (np.nan_to_num(s) - i)[:, None]
    k = np.arange(len(ns))
    nxt = pos[k, np.minimum(i + 1, pos.shape[1] - 1)]
    p = (1 - w) * pos[k, i] + w * np.where(w > 0, nxt, 0)
    out: Float = p.astype(float) * np.asarray(vox)
    out[~ok] = np.nan
    return out


def tck(lines: list[Float]) -> bytes:
    """MRtrix .tck bytes (float32 little-endian) of polylines in mm: a row of nan, then
    each line's points and a row of nan; an inf row ends the file. NiiVue (0.69) takes
    each nan row as the end of a line and the first offset as 0 only after a nan row:
    without the leading one it draws every line but the first (none of one line)."""
    body = [np.full((1, 3), np.nan, np.float32)]
    for p in lines:
        body.append(np.asarray(p, np.float32).reshape(-1, 3))
        body.append(np.full((1, 3), np.nan, np.float32))
    body.append(np.full((1, 3), np.inf, np.float32))
    head = f"mrtrix tracks\ndatatype: Float32LE\ncount: {len(lines)}\nfile: . "
    off = len(head) + 20
    hdr = f"{head}{off}\nEND\n".ljust(off).encode()
    return hdr + np.vstack(body).astype("<f4").tobytes()


def read_tck(data: bytes) -> list[Float]:
    """The polylines of .tck bytes as tck writes them (a leading nan row starts no
    line)."""
    head = data[: data.index(b"END\n")].decode()
    off = int(next(x for x in head.splitlines() if x.startswith("file:")).split()[-1])
    a = np.frombuffer(data[off:], "<f4").reshape(-1, 3).astype(float)
    if len(a) and np.isnan(a[0]).any():
        a = a[1:]
    out: list[Float] = []
    cur: list[Float] = []
    for row in a:
        if np.isinf(row).any():
            break
        if np.isnan(row).any():
            out.append(np.array(cur).reshape(-1, 3))
            cur = []
        else:
            cur.append(row)
    return out
