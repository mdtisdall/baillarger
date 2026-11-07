"""Tests of bent_probes: the top crossing, the l1 heading rule, straight
probes."""

import bent_probes as BP
import numpy as np


def test_first_crossing_is_linear_between_samples() -> None:
    s = np.arange(5, dtype=np.float64) * 0.5
    f = np.array(
        [[-1.0, -0.5, 0.5, 1.0, 1.0], [-1.0, -1, -1, -1, -1], [0.2, 1, 1, 1, 1]]
    )
    t = BP.first_crossing(f, s)
    assert np.isclose(t[0], 0.75)
    assert np.isnan(t[1])
    assert t[2] == 0.0


def test_l1_heading_keeps_sign_and_falls_back() -> None:
    h = np.array([[0.0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    v = np.array([[0.0, 0, -1], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    c = np.array([0.9, 0.9, 0.1, 0.9])
    ok = np.array([True, True, True, False])
    out = BP.l1_heading(v, c, ok, h, 0.3, 45.0)
    assert np.allclose(out[0], [0, 0, 1])  # sign flipped toward h
    assert np.allclose(out[1], h[1])  # turn of 90 deg: keep heading
    assert np.allclose(out[2], h[2])  # low coherence: keep heading
    assert np.allclose(out[3], h[3])  # no tensor: keep heading


def test_straight_probe_points() -> None:
    r = np.array([[1.0, 2, 3]])
    d = np.array([[0.0, 0, 2]])
    P = BP.straight(r, d)
    s = BP.s_grid()
    assert P.shape == (1, len(s), 3)
    assert np.allclose(P[0, :, 2], 3 + s)
    assert np.allclose(P[0, -1], [1, 2, 3 + BP.S_MAX])


def test_fold_dets_sees_a_fold() -> None:
    g = np.arange(-5, 6, dtype=np.float64) * 0.1
    X, Y = np.meshgrid(g, g, indexing="ij")
    R = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    N = np.tile([0.0, 0, 1], (len(R), 1))
    idx = np.array([60], np.int64)  # the centre
    same = BP.fold_dets(R, N, R + [0, 0, 1], idx, 0.25)
    assert np.isclose(same[0], 1.0)
    flipped = R * [-1, 1, 1]
    assert BP.fold_dets(R, N, flipped, idx, 0.25)[0] < 0
