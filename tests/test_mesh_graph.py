"""Tests for baillarger.mesh_graph: distances along a mesh, patches, and the
vertex a path crossed, on meshes with known geodesics."""

import numpy as np
from scipy.sparse.csgraph import dijkstra

from baillarger import mesh_graph as G
from baillarger.mesh import marching_tetrahedra
from baillarger.mesh_smoothing import smooth_normals

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
