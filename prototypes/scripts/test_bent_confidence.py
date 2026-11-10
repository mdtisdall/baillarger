"""Tests of bent_confidence: the soft terms, the per-column measures, the
product and the patches."""

import bent_confidence as BC
import numpy as np


def test_soft_is_one_in_range_and_gaussian_past_it() -> None:
    out = BC.soft(np.array([0.0, 1.0, 2.0, np.nan]), 1.0, 1.0)
    assert np.allclose(out[:2], 1.0)
    assert np.isclose(out[2], np.exp(-0.5))
    assert out[3] == 1.0


def test_soft_low_falls_below_the_limit() -> None:
    out = BC.soft_low(np.array([2.0, 1.0, 0.5]), 1.0, 0.5)
    assert np.allclose(out[:2], 1.0)
    assert np.isclose(out[2], np.exp(-0.5))


def test_largest_fall_stops_at_the_top() -> None:
    s = np.arange(5, dtype=np.float64)
    d = np.array([[0.0, 2.0, 1.5, 3.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
    d = np.vstack([d, d[0]])
    top = np.array([3.0, 4.0, np.nan])
    out = BC.largest_fall(d, top, s)
    assert np.isclose(out[0], 0.5)  # the drop to 0 is past the top
    assert out[1] == 0.0
    assert np.isnan(out[2])


def test_grad_mm_uses_the_affine() -> None:
    i = np.arange(5, dtype=np.float64)
    vol = i[:, None, None] * np.ones((5, 4, 3))  # 1 per voxel along axis 0
    g = BC.grad_mm(vol, np.diag([0.5, 1.0, 1.0, 1.0]))
    assert np.allclose(g[..., 0], 2.0)  # 2 per mm
    assert np.allclose(g[..., 1:], 0.0)


def test_arrival_angle_of_square_and_slanted_probes() -> None:
    s = np.arange(0, 3.01, BC.STEP)
    d = np.array([[0.0, 0, 1], [0, np.sin(np.pi / 3), np.cos(np.pi / 3)]])
    P = np.zeros((3, len(s), 3)) + np.array([5.0, 5, 5])
    P[0] += s[:, None] * d[0]
    P[1] += s[:, None] * d[1]
    P[2] += s[:, None] * d[0]
    top = np.array([1.0, 1.0, np.nan])
    grad = np.zeros((12, 12, 12, 3))
    grad[..., 2] = 1.0  # the outward normal is +z
    out = BC.arrival_deg(P, top, grad, lambda x: x)
    assert np.isclose(out[0], 0.0, atol=1e-6)
    assert np.isclose(out[1], 60.0)
    assert np.isnan(out[2])


def test_neighbour_msd_zero_for_equal_profiles_and_sees_an_outlier() -> None:
    s = np.arange(5, dtype=np.float64)
    prof = np.array([0.0, 1, -1, 1, 0])
    f = np.tile(prof, (5, 1))
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], np.int64)
    top = np.full(5, 4.0)
    assert np.allclose(BC.neighbour_msd(f, top, s, edges, 2), 0.0)
    f2 = f.copy()
    f2[2] = -prof
    out = BC.neighbour_msd(f2, top, s, edges, 2)
    assert out[2] > out[0] > 0


def test_neighbour_msd_ignores_samples_past_a_neighbours_top() -> None:
    s = np.arange(5, dtype=np.float64)
    f = np.array([[1.0, 1, 1, 1, 1], [1, 1, 99, 99, 99]])
    top = np.array([4.0, 1.0])
    out = BC.neighbour_msd(f, top, s, np.array([[0, 1]], np.int64), 1)
    assert np.isclose(out[0], 0.0)


def test_top_step_counts_no_top_as_infinite() -> None:
    top = np.array([1.0, 1.5, np.nan, np.nan])
    e = np.array([[0, 1], [1, 2], [2, 3]], np.int64)
    out = BC.top_step(top, e)
    assert np.isclose(out[0], 0.5)
    assert np.isinf(out[1])
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_combine_is_the_product_and_the_gate_gives_zero() -> None:
    terms = dict(a=np.array([0.5, 0.5]), b=np.array([0.5, 1.0]))
    out = BC.combine(terms, ["a", "b"], np.array([True, False]))
    assert np.isclose(out[0], 0.25)
    assert out[1] == 0.0


def test_components_are_ranked_by_size() -> None:
    mask = np.array([True, True, False, True, True, True, False])
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], np.int64)
    lab = BC.components(mask, e)
    assert lab.tolist() == [1, 1, -1, 0, 0, 0, -1]


def test_bands() -> None:
    cols = dict(
        P=np.array([0.0, 0.2, 0.6, 0.9, 0.9]),
        support=np.array([1.0, 1.0, 1.0, 0.1, 0.5]),
        reach=np.array([0.0, 1, 1, 1, 1]),
        fold=np.ones(5),
    )
    assert BC.bands(cols).tolist() == [0, 1, 2, 3, 4]
