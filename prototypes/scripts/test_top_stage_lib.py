import numpy as np
import pytest

from top_stage_lib import behind_root_lengths, gaussian_log_term, nearest_class, two_mode_log_prior


def _bump(n: int, at: int, height: float, width: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((np.arange(n) - at) / width) ** 2)


def test_two_modes_each_scored_against_its_own_peak() -> None:
    dens = _bump(120, 1, 10.0, 3.0) + _bump(120, 60, 2.0, 8.0)  # the near-0 mode is the taller one
    logg, pk, trough = two_mode_log_prior(dens)
    assert pk.tolist() == [1, 60]
    assert 1 < trough < 60
    assert logg[1] == pytest.approx(0.0)
    assert logg[60] == pytest.approx(0.0)  # not penalised for being the smaller mode
    assert logg[trough] < logg[60] - 1


def test_one_mode_is_relative_to_its_peak() -> None:
    logg, pk, trough = two_mode_log_prior(_bump(80, 40, 3.0, 5.0))
    assert trough == -1
    assert len(pk) == 1
    assert logg.max() == pytest.approx(0.0)


def test_floor() -> None:
    dens = _bump(200, 100, 1.0, 2.0)
    logg, _, _ = two_mode_log_prior(dens, floor=-6.0)
    assert logg.min() == -6.0


def test_nearest_class_uses_the_weighted_distance() -> None:
    means = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x = np.array([[0.1, 0.0], [0.9, 0.1], [0.0, 0.8]])
    assert nearest_class(x, means, np.eye(2)).tolist() == [0, 1, 2]
    # a point between classes 0 and 1: the weighting decides (axis 0 almost free)
    mid = np.array([[0.4, 0.45]])
    assert nearest_class(mid, means, np.diag([1e-6, 1.0])).tolist() == [0]
    assert nearest_class(mid, means, np.diag([1.0, 1e-6])).tolist() == [0]
    assert nearest_class(np.array([[0.7, 0.45]]), means, np.diag([1.0, 1e-6])).tolist() == [1]


def test_behind_root_lengths() -> None:
    cls = np.array([[0, 0, 1, 2, 0], [2, 2, 2, 0, 0], [1, 1, 1, 1, 1]])
    gm, fl = behind_root_lengths(cls, 3, 0.04)
    np.testing.assert_allclose(gm, [0.08, 0.0, 0.0])
    np.testing.assert_allclose(fl, [0.0, 0.12, 0.0])


def test_gaussian_log_term() -> None:
    t = gaussian_log_term(np.array([0.0, 0.3, 10.0, np.nan]), 0.3)
    assert t[0] == 0.0
    assert t[1] == pytest.approx(-0.5)
    assert t[2] == -8.0
    assert np.isnan(t[3])
