"""The streamlines at a point of the white surface, for the pia editor's local 3D panel (plan:
reports/Crossing paths and contacts from laminar profiles.md, *A pia editor on an inflated surface*).

Whatever tracer made the column volume (its `tracer_name`, if saved), the panel shows its traces as
they are: the positions saved in column_volume_<site>.npz, nothing re-traced.
- From the point: every trace rooted within `radius` of the clicked vertex along the white mesh
  (geodesic, so that roots on the facing bank of a sulcus, near in space but far along the mesh,
  are not counted).
- To the point: every other trace whose last sample is within `radius` of the point in space
  (mostly crossing traces from the facing bank that end at this one).
- The white mesh and the GM boundary inside a box around the point: the mesh cropped to the box,
  the GM boundary the marching-tetrahedra contour of the cortex label (smoothed by sigma voxels,
  for display only) clipped to the box, left open where the box cuts it.
Positions in mm. Everything is 3D.
"""

import numpy as np
from scipy.sparse.csgraph import dijkstra

import mesh_smoothing as MS
from crossing_lib import STEP_MM


def geodesic(G, root, radius):
    """Distances (mm) along the mesh graph G from the vertex root; inf beyond radius."""
    return dijkstra(G, indices=root, limit=radius)


def select(G, root, x0, tv, pos, ns, vox, radius=1.0):
    """(from, to): traces rooted within radius of the vertex root along G, and the other traces
    whose last sample lies within radius of x0 (mm). tv: each trace's root vertex (-1: none on
    this mesh); pos (n, nmax, 3) in voxels; ns the sample counts; vox the voxel size (mm)."""
    near = np.isfinite(geodesic(G, root, radius))
    tv = np.asarray(tv)
    fr = (tv >= 0) & near[np.maximum(tv, 0)]
    ends = pos[np.arange(len(ns)), np.maximum(ns, 1) - 1].astype(float) * vox
    to = ~fr & (ns > 0) & (np.linalg.norm(ends - x0, axis=1) <= radius)
    return np.flatnonzero(fr), np.flatnonzero(to)


def crop_mesh(v, f, lo, hi):
    """The faces of (v, f) with every vertex in the box [lo, hi]: (vertices, faces reindexed, the
    kept vertices' indices in v)."""
    inside = np.all((v >= lo) & (v <= hi), axis=1)
    fk = f[inside[f].all(1)]
    keep = np.unique(fk)
    new = -np.ones(len(v), np.int64)
    new[keep] = np.arange(keep.size)
    return v[keep], new[fk], keep


def gm_surface(mask, vox, lo, hi, sigma=1.0):
    """The boundary of a label inside the box [lo, hi) of voxels: the contour of the label smoothed
    by sigma voxels (from a crop padded so that the smoothing sees past the box), clipped to the
    box. Returns (vertices in mm, faces)."""
    pad = int(np.ceil(3 * sigma)) + 1
    a = np.maximum(np.asarray(lo) - pad, 0)
    b = np.minimum(np.asarray(hi) + pad, mask.shape)
    crop = mask[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    if not crop.any() or crop.all():
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    v, f = MS.contour(crop, sigma)
    v = np.asarray(v, float) + a
    v, f, _ = crop_mesh(v, np.asarray(f, np.int64), np.asarray(lo, float), np.asarray(hi, float) - 1)
    return v * vox, f


def at_depth(pos, ns, depth, vox):
    """Per trace the point (mm) at arc length depth (mm) along it, interpolated between samples;
    nan where depth is not finite or is past the trace's end."""
    s = np.asarray(depth, float) / STEP_MM
    ok = np.isfinite(s) & (s >= 0) & (s <= ns - 1)
    i = np.clip(np.floor(np.nan_to_num(s)).astype(int), 0, pos.shape[1] - 2)
    w = (np.nan_to_num(s) - i)[:, None]
    k = np.arange(len(ns))
    p = (1 - w) * pos[k, i] + w * np.where(w > 0, pos[k, np.minimum(i + 1, pos.shape[1] - 1)], 0)
    out = p.astype(float) * vox
    out[~ok] = np.nan
    return out
