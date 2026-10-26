"""Tests for baillarger.bands: plan test 6 (band DP against brute force, and
cross-validation finds the true bands), with one echo and with three."""

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from baillarger import bands as bd

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def brute_force_sse(W: F64, S: F64, Q: F64, cuts: I64) -> float:
    G = len(W)
    e = np.concatenate([[0], cuts, [G]])
    tot = 0.0
    for i, j in zip(e[:-1], e[1:], strict=True):
        w, s, q = W[i:j].sum(0), S[i:j].sum(0), Q[i:j].sum(0)
        tot += float(np.sum(q - np.where(w > 0, s * s / np.where(w > 0, w, 1), 0)))
    return tot


@pytest.mark.parametrize("E", [1, 3])
@pytest.mark.parametrize("min_w", [1, 2])
def test_band_dp_brute_force(E: int, min_w: int) -> None:
    """Plan test 6: the DP split against every split of a 12-bin grid."""
    rng = np.random.default_rng(7 + E)
    G = 12
    W = rng.uniform(0.5, 2, (G, E))
    x = rng.standard_normal((G, E))
    S, Q = W * x, W * x * x + rng.uniform(0, 0.1, (G, E))
    fits = bd.band_dp(W, S, Q, 4, min_w)
    for k in range(1, 5):
        splits = [
            np.array(c, np.int64)
            for c in itertools.combinations(range(1, G), k - 1)
            if np.all(np.diff(np.concatenate([[0], c, [G]])) >= min_w)
        ]
        best = min(brute_force_sse(W, S, Q, c) for c in splits)
        assert fits[k][1] == pytest.approx(best)
        assert brute_force_sse(W, S, Q, fits[k][0]) == pytest.approx(best)
        assert np.all(np.diff(np.concatenate([[0], fits[k][0], [G]])) >= min_w)


def test_band_dp_stops_where_k_is_not_possible() -> None:
    W = np.ones((10, 1))
    fits = bd.band_dp(W, W, W, 8, 3)
    assert sorted(fits) == [1, 2, 3]  # 3 bands of at least 3 bins in 10


def region(
    true: F64,
    levels: F64,
    n_cols: int = 36,
    per_col: int = 100,
    noise: float = 0.5,
    seed: int = 8,
) -> tuple[I64, F64, F64, F64, F64, F64]:
    """A region of vertical columns with a piecewise-constant profile (one
    vector of levels per band, shape (K, E)), a gain and an offset per column
    and echo, and noise. Returns col, m, f (raw), xh, w (normalized with the
    true gain and offset) and sigma2."""
    rng = np.random.default_rng(seed)
    E = levels.shape[1]
    col = np.repeat(np.arange(n_cols), per_col)
    m = np.tile((np.arange(per_col) + 0.5) / per_col, n_cols)
    a = rng.uniform(50, 150, (n_cols, E))
    s = rng.uniform(5, 20, (n_cols, E))
    band = bd.band_of(m, true)
    f = a[col] + s[col] * (levels[band] + noise * rng.standard_normal((len(m), E)))
    sigma2 = (noise * s) ** 2
    xh = (f - a[col]) / s[col]
    w = s[col] ** 2 / sigma2[col]
    return col, m, f, xh, w, sigma2


@pytest.mark.parametrize("E", [1, 3])
def test_cv_finds_true_bands(E: int) -> None:
    """Plan test 6: on a piecewise-constant profile with noise, the
    cross-validation finds the true K and the boundaries within one grid bin."""
    true = np.array([0.2, 0.44, 0.8])
    base = np.array([0.0, 2.0, 0.5, 1.5])
    levels = np.stack([base, 0.8 * base, 1.2 * base][:E], 1)
    col, m, f, xh, w, sigma2 = region(true, levels)
    h = 1 / 50
    K, Z, cv = bd.bands_for_region(
        col, m, f, xh, w, sigma2, 36, h, 8, 0.05, np.random.default_rng(0)
    )
    assert K == 4
    assert np.all(np.abs(Z - true) <= h + 1e-9)
    assert cv[K - 1] == cv.min()


def test_three_echoes_find_a_band_that_one_echo_misses() -> None:
    """A band with the same level as its neighbour at echo 2 but a different
    decay (as the stria against cortex) is found with three echoes, not with
    echo 2 alone."""
    true = np.array([0.3, 0.6])
    levels = np.array([[1.0, 1.0, 1.0], [0.4, 1.0, 1.6], [1.0, 1.0, 1.0]])
    col, m, f, xh, w, sigma2 = region(true, levels, noise=0.3, seed=3)
    h = 1 / 50
    args = (36, h, 8, 0.05)
    K3, Z3, _ = bd.bands_for_region(
        col, m, f, xh, w, sigma2, *args, np.random.default_rng(0)
    )
    one = slice(1, 2)
    K1, _, _ = bd.bands_for_region(
        col,
        m,
        f[:, one],
        xh[:, one],
        w[:, one],
        sigma2[:, one],
        *args,
        np.random.default_rng(0),
    )
    assert K3 == 3 and np.all(np.abs(Z3 - true) <= h + 1e-9)
    assert K1 == 1


@pytest.mark.parametrize("E", [1, 3])
def test_flat_profile_has_one_band(E: int) -> None:
    for seed in range(5):
        col, m, f, xh, w, sigma2 = region(
            np.zeros(0), np.zeros((1, E)), noise=1.0, seed=seed
        )
        K, Z, _ = bd.bands_for_region(
            col, m, f, xh, w, sigma2, 36, 1 / 50, 8, 0.05, np.random.default_rng(0)
        )
        assert K == 1 and len(Z) == 0


def test_the_prototype_rule_never_finds_one_band() -> None:
    """Why the default rule is not the prototype's: refitting the held-out
    gain favours every K >= 2 over a flat template, and the least residual
    then falls at random."""
    Ks = []
    for seed in range(5):
        col, m, f, xh, w, sigma2 = region(
            np.zeros(0), np.zeros((1, 3)), noise=1.0, seed=seed
        )
        K, _, cv = bd.bands_for_region(
            col,
            m,
            f,
            xh,
            w,
            sigma2,
            36,
            1 / 50,
            8,
            0.05,
            np.random.default_rng(0),
            refit=True,
            one_se=False,
        )
        assert cv[0] > cv[1:].max()
        Ks.append(K)
    assert min(Ks) >= 2


def test_choose_k() -> None:
    cv = np.array([10.0, 5.0, 4.9, 4.8])
    # four columns: K = 2 is within one standard error of K = 4
    percol = np.array(
        [[2.5] * 4, [1.0, 1.5, 1.2, 1.3], [1.2, 1.2, 1.3, 1.2], [1.2, 1.2, 1.2, 1.2]]
    )
    assert bd.choose_k(cv, percol, one_se=False) == 4
    assert bd.choose_k(cv, percol, gain_frac=0.0) == 2
    assert bd.choose_k(cv, percol) == 2
    # with very many columns the standard error is tiny, and the gain term
    # decides: 5 % of the whole gain (10 - 4.8) is 0.26, so K = 3 (4.9) is
    # within it and K = 2 (5.5) is not
    many = np.repeat(percol, 1000, axis=1) / 1000
    cv2 = np.array([10.0, 5.5, 4.9, 4.8])
    assert bd.choose_k(cv2, many, gain_frac=0.0) == 4
    assert bd.choose_k(cv2, many) == 3
    assert bd.choose_k(np.array([3.0, np.inf]), np.ones((2, 1))) == 1


def test_small_region_splits_voxels_for_the_cv() -> None:
    true = np.array([0.5])
    col, m, f, xh, w, sigma2 = region(
        true, np.array([[0.0], [2.0]]), n_cols=3, per_col=200, noise=0.3
    )
    K, Z, _ = bd.bands_for_region(
        col, m, f, xh, w, sigma2, 3, 1 / 50, 8, 0.05, np.random.default_rng(0)
    )
    assert K == 2 and abs(Z[0] - 0.5) <= 1 / 50 + 1e-9


def test_fixed_bands_merge_sparse_edges() -> None:
    m = ((np.arange(30) + 0.5) / 30) * 0.92  # the top layers are sparse or empty
    m = np.tile(m, 4)
    Z = bd.fixed_bands(m, 25, 0.25)
    assert Z[0] == pytest.approx(1 / 25) and Z[-1] == pytest.approx(22 / 25)


def test_band_of() -> None:
    z = np.array([0.25, 0.5])
    m = np.array([0.0, 0.25, 0.3, 0.5, 0.9])
    assert bd.band_of(m, z).tolist() == [0, 1, 1, 2, 2]
    zz = np.array([[0.1, 0.2], [0.5, 0.6], [0.0, 0.95], [0.3, 0.4], [0.9, 0.95]])
    assert bd.band_of(m, zz).tolist() == [0, 0, 1, 2, 1]


def test_smooth_trend_does_not_take_every_band() -> None:
    """A large region whose profile has two steps and a smooth trend: the
    standard error alone takes K_max, the gain term stops near the steps."""
    rng = np.random.default_rng(1)
    n_cols, per_col = 300, 100
    col = np.repeat(np.arange(n_cols), per_col)
    m = np.tile((np.arange(per_col) + 0.5) / per_col, n_cols)
    prof = np.where(m < 0.35, 0.0, np.where(m < 0.7, 1.5, 0.5)) + 0.6 * m
    f = 100 + 10 * (prof + 0.3 * rng.standard_normal(len(m)))
    a = np.full((n_cols, 1), 100.0)
    s = np.full((n_cols, 1), 10.0)
    xh = ((f[:, None] - a[col]) / s[col]).astype(np.float64)
    sigma2 = np.full((n_cols, 1), 9.0)
    w = s[col] ** 2 / sigma2[col]
    args = (col, m, f[:, None], xh, w, sigma2, n_cols, 1 / 50, 8, 0.05)
    K0, _, _ = bd.bands_for_region(*args, np.random.default_rng(0), gain_frac=0.0)
    K, Z, _ = bd.bands_for_region(*args, np.random.default_rng(0))
    assert K0 == 8
    assert K < 8
    for step in (0.35, 0.7):
        assert np.min(np.abs(Z - step)) <= 1 / 50 + 1e-9
