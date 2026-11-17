"""Inputs of the tube front-growth τ solver for one cropped site.

A site is the npz that tube_crop.py writes: echoes (3, X, Y, Z), labels,
affine (voxel -> mm), white_v (mm), white_f. This module derives:

- the white surface in voxel coordinates, its unit normals (out of the WM),
  its vertex areas and its mean curvature H̄ (smoothed over CURV_MM along the
  surface, baillarger.frame.mesh_curvatures);
- the signed distance d (mm, > 0 outside the WM) of the voxels near the white
  surface, and the inside mask (d < 0 near the surface, label 41 elsewhere);
- the laminar normal n̂ and its coherence from the structure tensor of echoes 2
  and 3 (as in baillarger.column_traces: log, trend removed, cortex label);
- the patches: farthest-point sampling on the vertices that face cortex
  (Euclidean), Voronoi cells along the mesh edges, coarse levels, patch
  areas, the patch adjacency and the same-bank pairs (geodesic distance of
  the patch centres along the patch graph <= SAME_MM).

Settings (the user's choices, 2026-11-17, items M and O of the open-choice
list): PATCH_MM = 0.5 (centre spacing), coarse levels 1, 2, 4 mm, SAME_MM = 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from baillarger import streamlines as S
from baillarger.frame import mesh_curvatures

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

WM, CORTEX, OUTSIDE = 41, 42, 0
PATCH_MM = 0.5
COARSE_MM = (1.0, 2.0, 4.0)
SAME_MM = 5.0
CURV_MM = 5.0
TREND = 4.0  # column_traces.TREND (voxels)
NEAR_VOX = 3  # signed distance computed within this many voxels of the WM edge


@dataclass
class Site:
    echoes: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int16]
    affine: F64
    vox: F64  # voxel size (mm) per axis
    v: F64  # white vertices, voxel coordinates
    f: I64
    normal: F64  # unit, out of the WM, voxel-coordinate frame (mm-scaled)
    varea: F64  # vertex area (mm²)
    H: F64  # mean curvature (k1 + k2) / 2 per mm, smoothed
    dist: F64  # signed distance (mm), nan far from the surface
    inside: npt.NDArray[np.bool_]
    nhat: npt.NDArray[np.float32]  # (X, Y, Z, 3)
    coh: npt.NDArray[np.float32]


@dataclass
class Patches:
    vpatch: I64  # patch of each vertex, -1 where the vertex does not face cortex
    centre: I64  # centre vertex of each patch
    area: F64  # mm²
    adj: npt.NDArray[np.int64]  # (n_pairs, 2) patch pairs that share a mesh edge
    coarse: list[I64]  # per level: coarse patch of each fine patch
    same_ptr: I64  # CSR over patches: same-bank partners (sorted), incl. itself
    same_idx: npt.NDArray[np.int32]
    same_dist: npt.NDArray[np.float32]


def vertex_normals_areas(v_mm: F64, f: I64) -> tuple[F64, F64]:
    """Area-weighted unit vertex normals (face orientation) and vertex areas."""
    cr = np.cross(v_mm[f[:, 1]] - v_mm[f[:, 0]], v_mm[f[:, 2]] - v_mm[f[:, 0]])
    n = np.zeros_like(v_mm)
    a = np.zeros(len(v_mm))
    fa = 0.5 * np.linalg.norm(cr, axis=1)
    for k in range(3):
        np.add.at(n, f[:, k], cr)
        np.add.at(a, f[:, k], fa / 3)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n, a


def mm_to_vox(affine: F64, p: F64) -> F64:
    out: F64 = (np.linalg.inv(affine) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
    return out


def sample_labels(labels: npt.NDArray[np.integer], pv: F64) -> npt.NDArray[np.int64]:
    """Labels at the nearest voxel of points pv (voxel coordinates); -1 outside."""
    i = np.round(pv).astype(np.int64)
    ok = np.all((i >= 0) & (i < np.array(labels.shape)), axis=1)
    out = np.full(len(pv), -1, np.int64)
    out[ok] = labels[i[ok, 0], i[ok, 1], i[ok, 2]]
    return out


def signed_distance(
    labels: npt.NDArray[np.integer], v: F64, n: F64, vox: F64
) -> tuple[F64, npt.NDArray[np.bool_]]:
    """Signed distance (mm, > 0 outside the WM) by the nearest vertex and its
    normal, for voxels within NEAR_VOX of the WM label's edge; inside mask."""
    wm = labels == WM
    edge = wm ^ ndi.binary_erosion(wm)
    near = ndi.binary_dilation(edge, iterations=NEAR_VOX + 1)
    idx = np.argwhere(near)
    tree = cKDTree(v * vox)
    _, j = tree.query(idx * vox)
    d = np.einsum("ij,ij->i", idx * vox - v[j] * vox, n[j])
    dist = np.full(labels.shape, np.nan)
    dist[tuple(idx.T)] = d
    inside = wm.copy()
    inside[tuple(idx.T)] = d < 0
    return dist, inside


def laminar_normal(
    echoes: npt.NDArray[np.float32], labels: npt.NDArray[np.integer]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """n̂ and coherence from echoes 2 and 3 over the cortex label
    (column_traces.tissue + base_tensor + data_normal, with S.eigen)."""
    ctx = (labels == CORTEX).astype(float)
    tensor = None
    for i in (1, 2):
        L = np.log(np.maximum(echoes[i].astype(float), 1.0))
        det = (L - S.masked_smooth(L, ctx, TREND)) * ctx
        t = S.structure_tensor(det, ctx)
        tensor = t if tensor is None else tensor + t
    assert tensor is not None
    nh, coh = S.eigen(tensor)
    coh = np.where(ctx > 0, coh, 0).astype(np.float32)
    return nh, coh


def load_site(path: Path) -> Site:
    z = np.load(path)
    A = z["affine"].astype(float)
    vox = np.abs(np.diag(A)[:3]).astype(float)
    labels = z["labels"].astype(np.int16)
    v = mm_to_vox(A, z["white_v"].astype(float))
    f = z["white_f"].astype(np.int64)
    n, area = vertex_normals_areas(v * vox, f)
    # orient out of the WM: the label 0.5 voxel along the normal
    out = sample_labels(labels, v + 0.5 * n / vox) != WM
    inn = sample_labels(labels, v - 0.5 * n / vox) == WM
    if np.mean(out & inn) < np.mean(~out & ~inn):
        n = -n
        f = f[:, ::-1].copy()
    k1, k2 = mesh_curvatures(v * vox, f, n, CURV_MM)
    dist, inside = signed_distance(labels, v, n, vox)
    nh, coh = laminar_normal(z["echoes"], labels)
    return Site(
        z["echoes"], labels, A, vox, v, f, n, area, 0.5 * (k1 + k2), dist, inside, nh, coh
    )


def facing_cortex(site: Site) -> npt.NDArray[np.bool_]:
    """Vertices whose outward normal meets the cortex label within one voxel."""
    vox = site.vox
    hit = np.zeros(len(site.v), bool)
    for t in (0.5, 1.0):
        hit |= sample_labels(site.labels, site.v + t * vox.mean() * site.normal / vox) == CORTEX
    return hit


def fps(p: F64, radius: float, start: int = 0) -> I64:
    """Farthest-point sampling (Euclidean) until every point is within radius
    of a seed. Returns the seed indices in the order chosen."""
    tree = cKDTree(p)
    mind = np.full(len(p), np.inf)
    seeds = [start]
    cur = start
    while True:
        r = float(mind.max()) if np.isfinite(mind.max()) else np.inf
        ids = tree.query_ball_point(p[cur], r=r if np.isfinite(r) else 1e9)
        ids = np.asarray(ids, np.int64)
        dd = np.linalg.norm(p[ids] - p[cur], axis=1)
        mind[ids] = np.minimum(mind[ids], dd)
        cur = int(np.argmax(mind))
        if mind[cur] <= radius:
            break
        seeds.append(cur)
    return np.asarray(seeds, np.int64)


def mesh_graph(v_mm: F64, f: I64, keep: npt.NDArray[np.bool_]) -> sparse.csr_matrix:
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = e[keep[e[:, 0]] & keep[e[:, 1]]]
    w = np.linalg.norm(v_mm[e[:, 0]] - v_mm[e[:, 1]], axis=1)
    g = sparse.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(v_mm),) * 2).tocsr()
    return g.maximum(g.T).tocsr()


def make_patches(site: Site, patch_mm: float = PATCH_MM) -> Patches:
    vmm = site.v * site.vox
    face = facing_cortex(site)
    ids = np.flatnonzero(face)
    # hexagonal packing: centre spacing s <-> covering radius s / sqrt(3)
    seeds = ids[fps(vmm[ids], patch_mm / np.sqrt(3))]
    g = mesh_graph(vmm, site.f, face)
    _, _, src = csgraph.dijkstra(g, indices=seeds, min_only=True, return_predecessors=True)
    lab = np.full(len(vmm), -1, np.int64)
    seed_id = -np.ones(len(vmm), np.int64)
    seed_id[seeds] = np.arange(len(seeds))
    ok = face & (src >= 0)
    lab[ok] = seed_id[src[ok]]
    area = np.bincount(lab[ok], weights=site.varea[ok], minlength=len(seeds))
    e = np.concatenate([site.f[:, [0, 1]], site.f[:, [1, 2]], site.f[:, [2, 0]]])
    pe = lab[e]
    pe = pe[(pe[:, 0] >= 0) & (pe[:, 1] >= 0) & (pe[:, 0] != pe[:, 1])]
    adj = np.unique(np.sort(pe, 1), axis=0)
    cen = vmm[seeds]
    coarse = []
    for s in COARSE_MM:
        cs = fps(cen, s / np.sqrt(3))
        _, j = cKDTree(cen[cs]).query(cen)
        coarse.append(j.astype(np.int64))
    # same-bank pairs: patch graph, edge = centre distance, limit SAME_MM
    n = len(seeds)
    w = np.linalg.norm(cen[adj[:, 0]] - cen[adj[:, 1]], axis=1)
    pg = sparse.coo_matrix((w, (adj[:, 0], adj[:, 1])), shape=(n, n)).tocsr()
    pg = pg.maximum(pg.T).tocsr()
    rows, cols, ds = [], [], []
    for a in range(0, n, 512):
        dm = csgraph.dijkstra(pg, indices=np.arange(a, min(a + 512, n)), limit=SAME_MM)
        r, c = np.nonzero(np.isfinite(dm))
        rows.append(r + a)
        cols.append(c)
        ds.append(dm[r, c])
    R, C, Dd = np.concatenate(rows), np.concatenate(cols), np.concatenate(ds)
    o = np.lexsort((C, R))
    R, C, Dd = R[o], C[o], Dd[o]
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, R + 1, 1)
    ptr = np.cumsum(ptr)
    return Patches(lab, seeds, area, adj, coarse, ptr, C.astype(np.int32), Dd.astype(np.float32))


def patch_geodesic(site: Site, pt: Patches, pairs: npt.NDArray[np.int64]) -> F64:
    """Geodesic distance (mm, along the mesh's facing vertices... all vertices)
    between the centres of the patch pairs; inf when not connected in the crop."""
    if len(pairs) == 0:
        return np.zeros(0)
    vmm = site.v * site.vox
    g = mesh_graph(vmm, site.f, np.ones(len(vmm), bool))
    src = np.unique(pairs[:, 0])
    out = np.full(len(pairs), np.inf)
    for a in range(0, len(src), 64):
        s = src[a : a + 64]
        dm = csgraph.dijkstra(g, indices=pt.centre[s])
        pos = {int(x): k for k, x in enumerate(s)}
        for k, (i, j) in enumerate(pairs):
            if int(i) in pos:
                out[k] = dm[pos[int(i)], pt.centre[j]]
    return out
