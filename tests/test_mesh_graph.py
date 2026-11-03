"""Tests for baillarger.mesh_graph: distances along a mesh, patches, and the
vertex a path crossed, on meshes with known geodesics."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from baillarger import mesh_graph as G
from baillarger.mesh import MeshBlocks, marching_tetrahedra, marching_tetrahedra_blocks
from baillarger.mesh_smoothing import edges, smooth_normals

VOX = (0.16, 0.16, 0.16)


def plane(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """The plane x = 10.3 across an n x n box: vertices (voxels), faces."""
    F = np.indices((21, n, n))[0].astype(float)
    return marching_tetrahedra(F, 10.3)


def test_edge_graph_weights_are_straight_line_lengths_in_mm() -> None:
    v, f = plane(8)
    g = G.edge_graph(v, f, (0.16, 0.2, 0.32)).tocoo()
    d = (v[g.row] - v[g.col]) * np.array([0.16, 0.2, 0.32])
    assert np.allclose(g.data, np.linalg.norm(d, axis=1))
    A = G.edge_graph(v, f, VOX).toarray()
    assert np.array_equal(A, A.T)


def ball() -> tuple[np.ndarray, np.ndarray]:
    """A small ball of radius 4.3 voxels: vertices (voxels), faces."""
    c = np.indices((13, 13, 13)).astype(float) - 6.0
    return marching_tetrahedra(np.linalg.norm(c, axis=0), 4.3)


def _lengths(g: csr_matrix, v: np.ndarray, vox: tuple[float, ...]) -> np.ndarray:
    c = g.tocoo()
    d: np.ndarray = np.linalg.norm((v[c.row] - v[c.col]) * np.asarray(vox), axis=1)
    return d


def test_edge_graph_weights_on_a_tetrahedron() -> None:
    """Each edge of a tetrahedron is a mesh edge and also the edge across a
    pair of faces: no weight is the sum of two."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    vox = (0.16, 0.2, 0.32)
    g = G.edge_graph(v, f, vox)
    assert g.nnz == 12
    assert np.allclose(g.data, _lengths(g, v, vox))


def test_edge_graph_weights_on_a_ball() -> None:
    """On this ball, some edges across a pair of faces occur twice in the
    same order (the bug doubled their weights)."""
    v, f = ball()
    E = np.concatenate([edges(f), G._across(f)])
    assert len(E) - len(np.unique(E, axis=0)) > 0
    g = G.edge_graph(v, f, VOX)
    assert np.allclose(g.data, _lengths(g, v, VOX), rtol=1e-12, atol=0)


def test_edge_graph_edge_set_is_the_mesh_edges_and_the_edges_across() -> None:
    v, f = ball()
    g = G.edge_graph(v, f, VOX)
    n = len(v)
    E = np.concatenate([edges(f), G._across(f)])
    want = np.zeros((n, n), bool)
    want[E[:, 0], E[:, 1]] = True
    want |= want.T
    got = g.toarray() > 0
    assert np.array_equal(got, want)


def test_geodesics_on_a_plane_are_within_the_8_neighbour_bound() -> None:
    """With the edges across neighbouring triangles, a grid-aligned plane's
    graph is 8-connected: at most 1 / cos(22.5 deg) over the straight line
    (along the mesh edges alone, sqrt 2)."""
    v, f = plane()
    g = G.edge_graph(v, f, VOX)
    a = int(np.argmin(np.linalg.norm(v - [10.3, 5, 5], axis=1)))
    d = dijkstra(g, indices=a)
    far = np.linalg.norm((v - v[a]) * 0.16, axis=1)
    ok = far > 1.0
    ratio = d[ok] / far[ok]
    assert ratio.min() >= 1.0 - 1e-9
    assert ratio.max() <= 1.0 / np.cos(np.radians(22.5)) + 1e-9


def test_patches_are_a_geodesic_voronoi_partition_at_the_spacing() -> None:
    v, f = plane()
    g = G.edge_graph(v, f, VOX)
    rep, patch = G.geodesic_patches(g, 0.96)
    d = dijkstra(g, indices=rep)
    assert d[:, rep][~np.eye(rep.size, dtype=bool)].min() >= 0.96 - 1e-9
    assert np.all(patch >= 0)
    assert np.allclose(d[patch, np.arange(len(v))], d.min(0))
    D = G.patch_distances(g, rep, 3.0)
    assert np.allclose(D[np.isfinite(D)], d[:, rep][np.isfinite(D)])
    assert np.isinf(D[d[:, rep] > 3.0]).all()


def test_on_a_thin_blade_the_two_sides_are_far_apart_and_kept_apart() -> None:
    """A blade 2 voxels thick: its two faces are about 0.3 mm apart in space
    but the geodesic goes round the tip; a path ending in the blade from one
    side is assigned a vertex on that side."""
    wm = np.zeros((40, 40, 20), bool)
    wm[19:21, 4:30, :] = True  # the blade, x in [19, 20], tip at y = 29
    v, f = marching_tetrahedra(wm.astype(float), 0.5)
    f = f[:, [0, 2, 1]].copy()
    n = smooth_normals(v, f)
    g = G.edge_graph(v, f, VOX)
    # midway between the blade's base (y = 4) and tip (y = 29): round either
    # end is at least 2 x 12 voxels
    a = int(np.argmin(np.linalg.norm(v - [18.5, 16.5, 10], axis=1)))
    b = int(np.argmin(np.linalg.norm(v - [20.5, 16.5, 10], axis=1)))
    assert np.linalg.norm((v[a] - v[b]) * 0.16) < 0.4
    assert dijkstra(g, indices=a)[b] > 2 * 12 * 0.16
    # paths ending just inside the blade, coming from -x and from +x
    x = np.array([[19.6, 16.5, 10.0], [19.6, 16.5, 10.0]])
    d = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    j = G.crossing_vertex(v, n, x, d, VOX)
    assert v[j[0], 0] < 19.5 and v[j[1], 0] > 19.5


# --- local form: a superblock with a halo ---------------------------------

CELLS = 8


def phantom() -> tuple[np.ndarray, np.ndarray, G.BlockIndex, MeshBlocks]:
    """A ball of radius 17.3 voxels on a 40-voxel grid in blocks of 8 cells
    (5 blocks a side, 37 or so crossed): vertices, faces, block index."""
    c = np.indices((41, 41, 41)).astype(float) - 20.0
    r = np.linalg.norm(c, axis=0) + 1.5 * np.sin(c[0] / 3.0) * np.cos(c[1] / 4.0)
    v, f, mb = marching_tetrahedra_blocks(r, 17.3, CELLS)
    return v, f, G.block_index(v, f, mb), mb


def _superblocks(
    blocks: G.BlockIndex, level: int, count: int
) -> list[tuple[int, int, int]]:
    sup = blocks.ijk // (1 << level)
    ids = sorted({(int(a), int(b), int(c)) for a, b, c in sup})
    step = max(1, len(ids) // count)
    return ids[::step][:count]


def _owned(blocks: G.BlockIndex, sb: tuple[int, int, int], level: int) -> np.ndarray:
    ins = np.all(blocks.ijk // (1 << level) == np.array(sb), axis=1)
    return np.concatenate(
        [
            np.arange(a, b)
            for a, b in zip(
                blocks.vertex_start[ins], blocks.vertex_stop[ins], strict=True
            )
        ]
        or [np.zeros(0, np.int64)]
    )


def _scatter(lg: G.LocalGraph, d: np.ndarray, n: int) -> np.ndarray:
    out = np.full((len(d), n), np.inf)
    out[:, lg.vertex] = d
    return out


def _exactness(vox: tuple[float, float, float], L: float, halo: float) -> list[bool]:
    """Per (level, superblock, owned root sample): is the local distance
    array equal to the whole graph's (inf beyond L)?"""
    v, f, blocks, _ = phantom()
    whole = G.edge_graph(v, f, vox)
    rng = np.random.default_rng(3)
    same = []
    for level in (0, 1):
        for sb in _superblocks(blocks, level, 4):
            lg = G.local_graph(v, f, blocks, sb, level, CELLS, halo, vox)
            owned = _owned(blocks, sb, level)
            assert owned.size
            src = rng.choice(owned, min(12, owned.size), replace=False)
            assert lg.root[np.searchsorted(lg.vertex, src)].all()
            dw = dijkstra(whole, indices=src, limit=L)
            dl = dijkstra(lg.graph, indices=np.searchsorted(lg.vertex, src), limit=L)
            same.append(np.array_equal(dw, _scatter(lg, dl, len(v))))
    return same


def test_local_distances_equal_the_whole_graphs_up_to_the_halo() -> None:
    L = 0.9
    for vox in ((0.16, 0.16, 0.16), (0.16, 0.16, 0.32)):
        v, f, blocks, _ = phantom()
        whole = G.edge_graph(v, f, vox)
        rng = np.random.default_rng(1)
        for level in (0, 1):
            for sb in _superblocks(blocks, level, 4):
                for halo in (L, L + 0.3):
                    lg = G.local_graph(v, f, blocks, sb, level, CELLS, halo, vox)
                    owned = _owned(blocks, sb, level)
                    src = rng.choice(owned, min(15, owned.size), replace=False)
                    dw = dijkstra(whole, indices=src, limit=L)
                    assert np.isfinite(dw).sum() > len(src) * 10
                    loc = np.searchsorted(lg.vertex, src)
                    dl = dijkstra(lg.graph, indices=loc, limit=L)
                    np.testing.assert_array_equal(dw, _scatter(lg, dl, len(v)))


def test_a_halo_smaller_than_the_limit_can_miss_vertices() -> None:
    """With no halo, a root near the superblock's face has neighbours within
    L that the local graph lacks: this is why the halo must be >= L."""
    L = 0.9
    same = _exactness((0.16, 0.16, 0.16), L, 0.0)
    assert not all(same)


def test_local_graph_is_a_subgraph_of_the_whole() -> None:
    v, f, blocks, _ = phantom()
    vox = (0.16, 0.16, 0.32)
    whole = G.edge_graph(v, f, vox).toarray()
    sb = _superblocks(blocks, 1, 3)[1]
    lg = G.local_graph(v, f, blocks, sb, 1, CELLS, 0.5, vox)
    a = lg.graph.toarray()
    assert lg.vertex.size < len(v)
    assert np.array_equal(lg.vertex, np.unique(lg.vertex))
    sub = whole[np.ix_(lg.vertex, lg.vertex)]
    assert np.all((a == 0) | (a == sub))


def test_roots_are_the_owned_and_the_halo_box_vertices() -> None:
    v, f, blocks, _ = phantom()
    vox = np.array((0.16, 0.16, 0.32))
    for level in (0, 1):
        for sb in _superblocks(blocks, level, 4):
            halo = 0.6
            lg = G.local_graph(v, f, blocks, sb, level, CELLS, halo, vox)
            span = (1 << level) * CELLS
            lo = np.array(sb) * span - halo / vox
            hi = (np.array(sb) + 1) * span + halo / vox
            want = np.zeros(len(v), bool)
            want[_owned(blocks, sb, level)] = True
            want |= np.all((v >= lo) & (v <= hi), axis=1)
            # every wanted root is a local vertex, and only they are roots
            assert np.isin(np.flatnonzero(want), lg.vertex).all()
            np.testing.assert_array_equal(lg.vertex[lg.root], np.flatnonzero(want))


class _Spy:
    """A face array that records the row ranges read from it."""

    def __init__(self, f: np.ndarray) -> None:
        self.f = f
        self.read: list[tuple[int, int]] = []

    def __getitem__(self, s: slice) -> np.ndarray:
        assert isinstance(s, slice)
        self.read.append((int(s.start), int(s.stop)))
        return self.f[s]


def test_only_the_faces_of_blocks_meeting_the_halo_box_are_read() -> None:
    v, f, blocks, _ = phantom()
    vox = np.array((0.16, 0.16, 0.16))
    sb, level, halo = _superblocks(blocks, 0, 5)[2], 0, 0.5
    spy = _Spy(f)
    G.local_graph(v, spy, blocks, sb, level, CELLS, halo, vox)  # type: ignore[arg-type]
    lo = np.array(sb) * CELLS - halo / vox
    hi = (np.array(sb) + 1) * CELLS + halo / vox
    meets = np.all((blocks.box_max >= lo) & (blocks.box_min <= hi), axis=1)
    assert 0 < meets.sum() < len(meets)
    want = np.zeros(len(f), bool)
    for a, b in zip(blocks.face_start[meets], blocks.face_stop[meets], strict=True):
        want[a:b] = True
    got = np.zeros(len(f), bool)
    for a, b in spy.read:
        got[a:b] = True
    np.testing.assert_array_equal(got, want)


def _graph_by_rows(v: np.ndarray, f: np.ndarray, vox: tuple[float, ...]) -> csr_matrix:
    """The edge graph with the duplicate pairs removed by np.unique(axis=0):
    the reference for the one-key-per-pair version."""
    from scipy.sparse import coo_matrix

    E = np.sort(np.concatenate([edges(f), G._across(f)]), 1)
    E = np.unique(E, axis=0)
    w = np.linalg.norm((v[E[:, 0]] - v[E[:, 1]]) * np.asarray(vox), axis=1)
    n = (len(v),) * 2
    out: csr_matrix = (
        coo_matrix((w, (E[:, 0], E[:, 1])), shape=n)
        + coo_matrix((w, (E[:, 1], E[:, 0])), shape=n)
    ).tocsr()
    return out


def _same_bytes(a: csr_matrix, b: csr_matrix) -> bool:
    return (
        a.shape == b.shape
        and a.indptr.dtype == b.indptr.dtype
        and a.indices.dtype == b.indices.dtype
        and a.data.dtype == b.data.dtype
        and a.indptr.tobytes() == b.indptr.tobytes()
        and a.indices.tobytes() == b.indices.tobytes()
        and a.data.tobytes() == b.data.tobytes()
    )


def test_edge_graph_equals_the_row_unique_version_byte_for_byte() -> None:
    meshes = [plane(8), ball()]
    rng = np.random.default_rng(0)
    for n in (5, 30, 200):
        # random triangles: repeated edges, repeated pairs, shared edges
        f = rng.integers(0, n, (4 * n, 3))
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        meshes.append((rng.random((n, 3)) * 10, f))
    for v, f in meshes:
        assert _same_bytes(G.edge_graph(v, f, VOX), _graph_by_rows(v, f, VOX))


def test_pair_keys_of_nothing_and_the_vertex_limit() -> None:
    assert G._pair_keys(np.empty((0, 2), np.int64), 5).shape == (0,)
    n = G._MAX_VERTICES
    assert n * n + n < np.iinfo(np.int64).max  # the largest key fits
    E = np.array([[n - 1, n - 2], [n - 2, n - 1], [0, n - 1]], np.int64)
    assert G._pair_keys(E, n).tolist() == [n - 1, (n - 2) * n + n - 1]
    with np.testing.assert_raises(AssertionError):
        G._pair_keys(E, n + 1)


def test_edge_graph_in_small_chunks_is_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v, f = ball()
    whole = _graph_by_rows(v, f, VOX)
    monkeypatch.setattr(G, "_CHUNK", 7)
    assert _same_bytes(G.edge_graph(v, f, VOX), whole)
