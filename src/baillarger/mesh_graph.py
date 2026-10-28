"""Distances along a triangle mesh: its edge graph in mm, geodesic Voronoi
patches, the distances between patch centres, and the vertex where a path
that ended inside the surface crossed it.

The column traces use these on the white mesh (baillarger.column_traces) to
tell a trace that crosses a contact to the far bank from one that returns to
its own bank: the two ends' patches are far apart along the white surface,
or near. Distances along the mesh go round a thin WM blade's tip, so its two
sides are far apart even where they are a voxel apart in space.

Graph distances are shortest paths (Dijkstra) along the mesh's edges and
across each pair of neighbouring triangles (the edge between their far
vertices). Along the edges alone, the error would be up to 41 % (sqrt 2): a
marching-tetrahedra mesh of a voxel label has all its square cells split by
the same diagonal, so a path across the other diagonal zigzags, and
smoothing moves the vertices but keeps the edges. With the crossing edges a
grid-aligned plane gets an 8-neighbour graph: at most 8.2 % (1 / cos 22.5
deg) over the straight line.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from baillarger.mesh_smoothing import edges

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def _across(f: npt.NDArray[np.integer[Any]]) -> I64:
    """For each edge shared by two triangles, the pair of their far vertices."""
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    far = np.concatenate([f[:, 2], f[:, 0], f[:, 1]])
    key = np.sort(e, 1)
    o = np.lexsort((key[:, 1], key[:, 0]))
    key, far = key[o], far[o]
    same = np.all(key[1:] == key[:-1], 1)
    # interior manifold edges: two triangles (an edge with more has no pair)
    pair = same & ~np.r_[same[1:], False] & ~np.r_[False, same[:-1]]
    i = np.flatnonzero(pair)
    out: I64 = np.c_[far[i], far[i + 1]].astype(np.int64)
    keep: I64 = out[out[:, 0] != out[:, 1]]
    return keep


def edge_graph(
    v: F64, f: npt.NDArray[np.integer[Any]], voxel_mm: npt.ArrayLike
) -> csr_matrix:
    """The symmetric graph of a mesh with vertices v (voxel coordinates): its
    edges and the edges across each pair of neighbouring triangles, weighted
    by their straight-line lengths in mm."""
    E = np.concatenate([edges(f), _across(f)])
    vox = np.asarray(voxel_mm, np.float64)
    w = np.linalg.norm((v[E[:, 0]] - v[E[:, 1]]) * vox, axis=1)
    G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v),) * 2).tocsr()
    out: csr_matrix = G.maximum(G.T).tocsr()
    return out


def geodesic_patches(graph: csr_matrix, spacing: float) -> tuple[I64, I64]:
    """Centres chosen greedily in index order, each at least `spacing` along
    the graph from every earlier one, then every node assigned to its nearest
    centre (a geodesic Voronoi partition: each cell is connected). Returns
    the centre node of each patch and the patch of each node (-1 for nodes
    no centre reaches)."""
    n = graph.shape[0]
    covered = np.zeros(n, bool)
    centres = []
    for i in range(n):
        if covered[i]:
            continue
        centres.append(i)
        d = dijkstra(graph, indices=i, limit=spacing)
        covered |= d < spacing
    cen = np.array(centres, np.int64)
    _, _, src = dijkstra(graph, indices=cen, min_only=True, return_predecessors=True)
    lookup = -np.ones(n + 1, np.int64)
    lookup[cen] = np.arange(cen.size)
    return cen, lookup[np.where(src < 0, n, src)]


def patch_distances(graph: csr_matrix, rep: I64, limit: float, chunk: int = 200) -> F32:
    """Distances along the graph between patch centres (npatch, npatch),
    float32; inf beyond limit or where not connected."""
    n = rep.size
    D = np.full((n, n), np.inf, np.float32)
    for a in range(0, n, chunk):
        d = dijkstra(graph, indices=rep[a : a + chunk], limit=limit)
        D[a : a + chunk] = d[:, rep]
    return D


def crossing_vertex(
    v: F64,
    nrm: F64,
    x: npt.ArrayLike,
    d: npt.ArrayLike,
    voxel_mm: npt.ArrayLike,
    k: int = 8,
) -> I64:
    """For paths that ended at x (n, 3, voxel coordinates) going along d
    (n, 3) into the region the mesh bounds (its normals nrm point out of it):
    the nearest of the k nearest vertices (in mm) whose normal faces the path
    (n . d < 0), the side the path came through. The nearest vertex if none
    of the k does. On a thin blade this picks the side the path entered, not
    the nearer side."""
    vox = np.asarray(voxel_mm, np.float64)
    xa = np.asarray(x, np.float64)
    da = np.asarray(d, np.float64)
    k = min(k, len(v))
    _, j = cKDTree(v * vox).query(xa * vox, k=k)
    j = j.reshape(len(xa), k)
    faces = np.einsum("nkc,nc->nk", nrm[j], da) < 0
    first = np.where(faces.any(1), np.argmax(faces, 1), 0)
    out: I64 = j[np.arange(len(xa)), first].astype(np.int64)
    return out
