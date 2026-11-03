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

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from baillarger.mesh import MeshBlocks
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


def _graph(
    v: F64, f: npt.NDArray[np.integer[Any]], voxel_mm: npt.ArrayLike
) -> csr_matrix:
    """The graph of edge_graph over len(v) nodes."""
    E = np.sort(np.concatenate([edges(f), _across(f)]), 1)
    E = np.unique(E, axis=0)  # a pair that occurs twice gets its length once
    vox = np.asarray(voxel_mm, np.float64)
    w = np.linalg.norm((v[E[:, 0]] - v[E[:, 1]]) * vox, axis=1)
    n = (len(v),) * 2
    out: csr_matrix = (
        coo_matrix((w, (E[:, 0], E[:, 1])), shape=n)
        + coo_matrix((w, (E[:, 1], E[:, 0])), shape=n)
    ).tocsr()
    return out


def edge_graph(
    v: F64, f: npt.NDArray[np.integer[Any]], voxel_mm: npt.ArrayLike
) -> csr_matrix:
    """The symmetric graph of a mesh with vertices v (voxel coordinates): its
    edges and the edges across each pair of neighbouring triangles, weighted
    by their straight-line lengths in mm."""
    return _graph(v, f, voxel_mm)


@dataclass(frozen=True)
class BlockIndex:
    """The block index of a mesh as arrays, one row per block (white-surf's
    block index): the block's index, the vertices it owns and its faces
    ([start, stop) ranges), and the bounding box of the vertices its faces
    use, in voxel coordinates."""

    ijk: I64
    vertex_start: I64
    vertex_stop: I64
    face_start: I64
    face_stop: I64
    box_min: F64
    box_max: F64


def block_index(
    v: F64, f: npt.NDArray[np.integer[Any]], blocks: MeshBlocks
) -> BlockIndex:
    """The BlockIndex of a mesh made by marching_tetrahedra_blocks."""
    k = len(blocks.origin)
    lo = np.empty((k, 3))
    hi = np.empty((k, 3))
    for b in range(k):
        used = v[np.unique(f[blocks.face_start[b] : blocks.face_stop[b]])]
        lo[b], hi[b] = used.min(0), used.max(0)
    return BlockIndex(
        blocks.origin // blocks.cells,
        blocks.vertex_start.astype(np.int64),
        blocks.vertex_stop.astype(np.int64),
        blocks.face_start.astype(np.int64),
        blocks.face_stop.astype(np.int64),
        lo,
        hi,
    )


@dataclass(frozen=True)
class LocalGraph:
    """A superblock's local graph (local_graph)."""

    #: the graph over the local vertices, weights in mm
    graph: csr_matrix
    #: (n,) global vertex id of each local vertex, ascending
    vertex: I64
    #: (n,) which local vertices are roots of the superblock with its halo
    root: Bool


def local_graph(
    v: F64,
    f: npt.NDArray[np.integer[Any]],
    blocks: BlockIndex,
    superblock: npt.ArrayLike,
    level: int,
    cells: int,
    halo_mm: float,
    voxel_mm: npt.ArrayLike,
) -> LocalGraph:
    """The part of edge_graph(v, f, voxel_mm) that a superblock with a halo
    of halo_mm needs, from the faces of the blocks near it alone.

    The superblock (I, J, K) at `level` m is the aligned cube of 2^m blocks a
    side with block indices i // 2^m = I etc.; its cells span, on axis a,
    voxels I_a 2^m C to (I_a + 1) 2^m C (C = cells). Its halo box is that
    span widened by halo_mm / voxel_mm[a] on each axis. Its roots are the
    vertices owned by its blocks and the other vertices inside the halo box.
    v (n, 3, voxel coordinates; may be a memory map: only the rows used are
    read) and f (m, 3) are the whole mesh. Only the faces of the blocks whose
    bounding box meets the halo box are read (f is indexed by one slice for
    each such block).

    The graph is built as edge_graph builds it, on the vertices those faces
    use; `vertex` gives their global ids and `root` marks the roots.

    The distances from a root are exact up to halo_mm: for a root r of the
    superblock and a vertex u with a whole-mesh graph distance d(r, u) <= L
    <= halo_mm, the local graph's distance equals d(r, u) (exactly, in
    floating point: the same path adds the same weights in the same order).
    Every vertex x on a path of length <= L from r lies in the halo box (r is
    in the superblock's span or in the halo box, and x is at most L away in
    mm, so within L / voxel_mm[a] on each axis; for an owned root, the span
    holds it, since it lies on an edge of a cell of the superblock). A face
    that uses x belongs to a block whose bounding box contains x, so that
    block meets the halo box and the face is read. A graph edge of the path
    is a mesh edge (a triangle with both ends on the path) or an edge across
    two triangles (both use path vertices); both triangles are read, so every
    edge of the path is in the local graph, with the same weight. The local
    graph has no edge that the whole graph lacks: its mesh edges are the
    whole mesh's, and an edge across a pair of triangles that share an edge
    is also across that pair in the whole mesh (on a manifold mesh the pair
    is unique, as both triangles are read whenever the edge's vertices are
    both inside). So local distances are never shorter than the whole
    graph's, and not longer for the paths up to L. With a halo smaller than
    L, a path can leave the halo box and its faces are missing.
    """
    vox = np.asarray(voxel_mm, np.float64)
    sb = np.asarray(superblock, np.int64)
    span = 1 << level
    lo = sb * span * cells
    hi = (sb + 1) * span * cells
    h = halo_mm / vox
    box_lo, box_hi = lo - h, hi + h
    near = np.all((blocks.box_max >= box_lo) & (blocks.box_min <= box_hi), axis=1)
    inside = np.all(blocks.ijk // span == sb, axis=1)
    rows = [
        f[a:b]
        for a, b in zip(blocks.face_start[near], blocks.face_stop[near], strict=True)
        if b > a
    ]
    fl = np.concatenate(rows) if rows else np.zeros((0, 3), np.int64)
    vertex = np.unique(fl)
    vl = np.asarray(v[vertex], np.float64)
    local = np.searchsorted(vertex, fl)
    graph = _graph(vl, local, vox)
    root = np.all((vl >= box_lo) & (vl <= box_hi), axis=1)
    for a, b in zip(
        blocks.vertex_start[inside], blocks.vertex_stop[inside], strict=True
    ):
        root |= (vertex >= a) & (vertex < b)
    return LocalGraph(graph, vertex, root)


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
