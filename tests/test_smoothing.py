"""Tests for baillarger.mesh_smoothing: exact answers on planes and balls."""

import numpy as np
import numpy.typing as npt
import pytest

from baillarger import mesh_smoothing as S
from baillarger.mesh import marching_tetrahedra

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def plane_mesh() -> tuple[Float, Int, Float]:
    g = np.stack(np.meshgrid(*[np.arange(20.0)] * 3, indexing="ij"), -1)
    nrm = np.array([0.3, -0.5, 0.81])
    nrm /= np.linalg.norm(nrm)
    v, f = marching_tetrahedra(g @ nrm, 9.3)
    return v, f, nrm


def ball_mesh(r: float, n: int = 32) -> tuple[Float, Int, Float]:
    """The binary ball's contour, faces out of the ball, and its centre."""
    c = np.array([15.3, 15.6, 15.1])
    g = np.indices((n, n, n)).astype(float)
    wm = ((g - c[:, None, None, None]) ** 2).sum(0) <= r * r
    v, f = marching_tetrahedra(wm.astype(np.float64), 0.5)
    return v, f[:, [0, 2, 1]].copy(), c


def test_edges_and_boundary_of_a_square() -> None:
    f = np.array([[0, 1, 2], [0, 2, 3]])
    assert S.edges(f).tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
    assert S.boundary_vertices(f, 5).tolist() == [True, True, True, True, False]


def test_vertex_normals_of_a_plane() -> None:
    v, f, nrm = plane_mesh()
    n = S.vertex_normals(v, f)
    # marching_tetrahedra points toward increasing field: along +nrm
    np.testing.assert_allclose(n, np.broadcast_to(nrm, n.shape), atol=1e-9)


def test_smoothing_keeps_a_plane_and_its_boundary() -> None:
    v, f, nrm = plane_mesh()
    b = S.boundary_vertices(f, len(v))
    for x in (S.twice(v, f, 2.0), S.relax(v, f)):
        np.testing.assert_allclose(x @ nrm, 9.3, atol=1e-9)
        np.testing.assert_array_equal(x[b], v[b])
    assert not np.allclose(S.relax(v, f), v)  # it moves vertices in the plane


def test_twice_keeps_a_ball_radius_where_gauss_shrinks() -> None:
    v, f, c = ball_mesh(10.0)
    r0 = np.linalg.norm(v - c, axis=1).mean()
    A, deg = S.umbrella(v, f)
    x = v.copy()
    for _ in range(S.gauss_steps(v, f, 2.0)):
        x = x + 0.5 * ((A @ x) / deg[:, None] - x)
    shrink = r0 - np.linalg.norm(x - c, axis=1).mean()
    rt = np.linalg.norm(S.twice(v, f, 2.0) - c, axis=1).mean()
    assert shrink > 0.1 and abs(rt - r0) < 0.1 * shrink


def test_twice_never_amplifies() -> None:
    # 1 - (1 - g^n)^2 lies in [0, 1] for every eigenvalue g of the
    # (symmetrized) step operator: no mode grows
    v, f, _ = ball_mesh(5.0, 16)
    A, deg = S.umbrella(v, f)
    Ds = np.diag(1 / np.sqrt(deg))
    g = np.linalg.eigvalsh(0.5 * np.eye(len(v)) + 0.5 * Ds @ A.toarray() @ Ds)
    resp = 1 - (1 - g ** S.gauss_steps(v, f, 1.5)) ** 2
    assert g.min() >= -1e-12 and resp.max() <= 1 + 1e-12 and resp.min() >= -1e-12


def test_smooth_normals_of_a_plane_and_a_ball() -> None:
    v, f, nrm = plane_mesh()
    n = S.smooth_normals(v, f)
    np.testing.assert_allclose(n, np.broadcast_to(nrm, n.shape), atol=1e-9)
    v, f, c = ball_mesh(10.0)
    n = S.smooth_normals(v, f)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)
    assert np.all(np.sum(n * (v - c), 1) > 0)  # out of the ball


def test_normal_length_is_one_on_a_plane() -> None:
    """A flat triangulated grid in z = 0: every area-weighted normal is +z,
    also at the boundary, because each face adds +z. Averages of equal
    vectors keep their length, so the length is 1 at every vertex."""
    i, j = np.meshgrid(np.arange(9), np.arange(9), indexing="ij")
    v = np.stack([i.ravel(), j.ravel(), np.zeros(81)], 1).astype(np.float64)
    q = (i[:-1, :-1] * 9 + j[:-1, :-1]).ravel()
    f = np.concatenate(
        [np.stack([q, q + 9, q + 10], 1), np.stack([q, q + 10, q + 1], 1)]
    )
    n, length = S.smooth_normals_and_length(v, f)
    np.testing.assert_allclose(length, 1.0, atol=1e-12)
    np.testing.assert_allclose(n, np.broadcast_to([0.0, 0.0, 1.0], n.shape), atol=1e-12)
    assert length.shape == (81,) and length.dtype == n.dtype


def blade_length(thick: int) -> tuple[Float, Float]:
    """The normal length at the vertices of a closed blade of thick voxels,
    and each vertex's distance to the blade's crest (its edge and corners)."""
    g = np.zeros((24, 24, 24))
    g[10 : 10 + thick, 4:20, 4:20] = 1.0
    v, f = marching_tetrahedra(g, 0.5)
    _, length = S.smooth_normals_and_length(v, f)
    d = np.minimum(
        np.minimum(v[:, 1] - 3.5, 19.5 - v[:, 1]),
        np.minimum(v[:, 2] - 3.5, 19.5 - v[:, 2]),
    )
    return length, d


def test_normal_length_falls_at_the_crest_of_a_thin_blade() -> None:
    """A blade 16 x 16 voxels wide. Its two big faces have opposite normals
    and meet at the crest, so the smoothed normal there averages opposite
    vectors. Far from the crest the length is 1. At the crest it is lower,
    and lower for the blade 1 voxel thick than for the blade 2 voxels thick,
    where the faces are further apart."""
    crest = {}
    for thick in (1, 2):
        length, d = blade_length(thick)
        assert length[d >= 6].min() > 1 - 1e-9
        crest[thick] = length[d <= 1].min()
        assert length.max() < 1 + 1e-9
    assert crest[1] < 0.94
    assert crest[2] < 0.96
    assert crest[1] < crest[2]


def test_normal_length_of_an_octahedron_after_one_step() -> None:
    """The octahedron with vertices +-e_x, +-e_y, +-e_z, faces out. Its 8
    faces have equal area and the normals (+-1, +-1, +-1)/sqrt(3). At e_z the
    4 faces add up to 4 e_z / sqrt(3), so the unit normal is e_z (and by
    symmetry the unit normal at a vertex is the vertex). The 4 neighbours of
    e_z are +-e_x and +-e_y, whose normals have the mean 0. With iters = 1
    the length is |0.5 e_z + 0.5 * 0| = 0.5 at every vertex."""
    v = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float
    )
    f = np.array(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4]]
        + [[2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    )
    n, length = S.smooth_normals_and_length(v, f, 1)
    np.testing.assert_allclose(length, 0.5, atol=1e-15)
    np.testing.assert_allclose(n, v, atol=1e-15)


def test_old_and_new_normals_are_the_same() -> None:
    v, f, _ = ball_mesh(8.0)
    for iters in (1, 3, 10):
        n, _ = S.smooth_normals_and_length(v, f, iters)
        assert np.array_equal(n, S.smooth_normals(v, f, iters))
    with pytest.raises(ValueError):
        S.smooth_normals_and_length(v, f, 0)


def test_pieces() -> None:
    f = np.array([[0, 1, 2], [3, 4, 5], [4, 5, 6], [1, 2, 7]])
    vlab, flab = S.pieces(f, 9)
    assert vlab.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, -1]
    assert flab.tolist() == [0, 1, 1, 0]
