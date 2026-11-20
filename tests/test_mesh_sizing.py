"""Exact tests of baillarger.mesh_sizing: the curvature sizes on a sphere
(the Σ start's and the τ mesh's constants), the facing distance between two
sheets, the graded volume field and its facing cap, and the Kuhn grid."""

import numpy as np

from baillarger import mesh_sizing as S
from baillarger.mesh import marching_tetrahedra


def _sphere(R: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.indices((n, n, n)).astype(float) - (n - 1) / 2
    v, f = marching_tetrahedra(np.sqrt((g**2).sum(0)), R)
    V = v.astype(np.float64) - (n - 1) / 2
    return V, f.astype(np.int64), V / np.linalg.norm(V, axis=1, keepdims=True)


def _plane(n: int, z: float, up: bool) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(n, dtype=float)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]] if up else [[a, c, b], [a, d, c]]
    return V, np.array(F, np.int64)


def test_curvature_sizes_on_a_sphere() -> None:
    """R = 20: κ = 1/R. The Σ start's constants (sag h/4, 3/4 of the bound,
    clip [1, 5] h) give 0.75 sqrt(8 · 0.25 · 20) = 4.74; the τ mesh's (no
    scale, clip [1, 32] h) give sqrt(8 · 0.25 · 20) = 6.32."""
    R = 20.0
    V, F, N = _sphere(R, 48)
    L, kap = S.curvature_sizes(
        V, F, N, sag=0.25, smooth=2.5, lo=1.0, hi=5.0, scale=0.75
    )
    np.testing.assert_allclose(np.median(kap), 1 / R, rtol=0.05)
    np.testing.assert_allclose(np.median(L), 0.75 * np.sqrt(8 * 0.25 * R), rtol=0.05)
    L2, _ = S.curvature_sizes(V, F, N, sag=0.25, smooth=4.0, lo=1.0, hi=32.0)
    np.testing.assert_allclose(np.median(L2), np.sqrt(8 * 0.25 * R), rtol=0.05)
    L3, _ = S.curvature_sizes(V, F, N, sag=0.25, smooth=4.0, lo=1.0, hi=3.0)
    assert L3.max() <= 3.0
    cap = np.full(len(V), 2.0)
    L4, _ = S.curvature_sizes(
        V, F, N, sag=0.25, smooth=4.0, lo=1.0, hi=32.0, facing=cap
    )
    np.testing.assert_allclose(L4, 2.0)


def test_facing_distances_between_two_sheets() -> None:
    """Two planes 3 apart facing each other: every vertex sees the other
    sheet at 3 along its normal; with a search of 2, none; a sheet facing
    away is not a hit."""
    A, FA = _plane(9, 0.0, up=True)
    B, FB = _plane(9, 3.0, up=False)
    V = np.r_[A, B]
    F = np.r_[FA, FB + len(A)]
    N = np.r_[np.tile([0, 0, 1.0], (len(A), 1)), np.tile([0, 0, -1.0], (len(B), 1))]
    d = S.facing_distances(V, F, N, 4.0)
    np.testing.assert_allclose(d, 3.0)
    assert np.isinf(S.facing_distances(V, F, N, 2.0)).all()
    F2 = np.r_[FA, FB[:, ::-1] + len(A)]  # the upper sheet now faces up
    assert np.isinf(S.facing_distances(V, F2, N, 4.0)[: len(A)]).all()


def test_graded_sizes_and_the_facing_cap() -> None:
    V = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    size = np.array([1.0, 4.0])
    P = np.array([[0.0, 0, 0], [0, 0, 2], [10, 0, 0], [10, 0, 20], [5, 0, 0]])
    s = S.graded_sizes(P, V, size, grade=0.5, far=8.0)
    np.testing.assert_allclose(s, [1.0, 2.0, 4.0, 8.0, 3.5])
    capped = S.graded_sizes(P, V, size, 0.5, 8.0, facing=np.array([np.inf, 0.5]))
    np.testing.assert_allclose(capped, [1.0, 2.0, 0.25, 8.0, 2.75])


def test_kuhn_grid_tiles_the_box() -> None:
    G, T = S.kuhn_grid([0, 0, 0], [3.0, 2.0, 1.0], 1.0)
    assert len(G) == 4 * 3 * 2 and len(T) == 6 * 3 * 2 * 1
    vol = np.linalg.det(G[T[:, 1:]] - G[T[:, :1]])
    assert (np.abs(vol) > 0).all()
    np.testing.assert_allclose(np.abs(vol).sum() / 6, 6.0)
