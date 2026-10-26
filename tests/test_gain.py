"""Tests for baillarger.gain: plan test 2 (the limits of κ), and the templates
leave the model unchanged."""

import numpy as np
import numpy.typing as npt

from baillarger import gain as gn
from baillarger.bands import fit_affine

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def setup(
    E: int = 3, n: int = 12, per: int = 40, seed: int = 3
) -> tuple[I64, F64, F64, F64, I64, F64]:
    """Columns in a chain (one connected graph), a template per voxel, and
    echoes with a gain and an offset per column."""
    rng = np.random.default_rng(seed)
    col = np.repeat(np.arange(n), per)
    t = np.cos(5 * np.tile(np.linspace(0, 1, per), n))[:, None] * np.ones((1, E))
    a = rng.uniform(50, 150, (n, E))
    s = rng.uniform(5, 20, (n, E))
    f = a[col] + s[col] * t + rng.normal(0, 1, (len(col), E))
    sigma2 = rng.uniform(0.5, 2, (n, E))
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], 1).astype(np.int64)
    ew = np.full(n - 1, 10.0)
    return col, f, t, sigma2, edges, ew


def test_kappa_zero_is_per_column_least_squares() -> None:
    """Plan test 2, κ = 0."""
    col, f, t, sigma2, edges, ew = setup()
    n, E = sigma2.shape
    a, s = gn.solve_gain(
        col, f, t, sigma2, edges, ew, 0.0, np.zeros((n, E)), np.ones((n, E))
    )
    for e in range(E):
        ae, se, _ = fit_affine(f[:, e], t[:, e], 1.0 / sigma2[col, e], col, n)
        assert np.allclose(a[:, e], ae, atol=1e-4)
        assert np.allclose(s[:, e], se, atol=1e-4)


def test_large_kappa_is_one_global_gain_and_offset() -> None:
    """Plan test 2, κ → ∞."""
    col, f, t, sigma2, edges, ew = setup()
    n, E = sigma2.shape
    a, s = gn.solve_gain(
        col, f, t, sigma2, edges, ew, 1e9, np.zeros((n, E)), np.ones((n, E))
    )
    for e in range(E):
        w = 1.0 / sigma2[col, e]
        X = np.stack([np.ones(len(col)), t[:, e]], 1) * np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(X, f[:, e] * np.sqrt(w), rcond=None)[0]
        assert np.allclose(a[:, e], coef[0], atol=1e-3)
        assert np.allclose(s[:, e], coef[1], atol=1e-3)


def test_echoes_are_independent() -> None:
    col, f, t, sigma2, edges, ew = setup()
    n, E = sigma2.shape
    a0, s0 = np.zeros((n, E)), np.ones((n, E))
    a, s = gn.solve_gain(col, f, t, sigma2, edges, ew, 3.0, a0, s0)
    for e in range(E):
        k = slice(e, e + 1)
        ae, se = gn.solve_gain(
            col, f[:, k], t[:, k], sigma2[:, k], edges, ew, 3.0, a0[:, k], s0[:, k]
        )
        assert np.allclose(a[:, e], ae[:, 0]) and np.allclose(s[:, e], se[:, 0])


def test_a_column_with_nothing_to_fit_keeps_its_values() -> None:
    col, f, t, sigma2, edges, ew = setup(E=1)
    keep = col != 4
    n = sigma2.shape[0]
    a0 = np.full((n, 1), 7.0)
    s0 = np.full((n, 1), 3.0)
    a, s = gn.solve_gain(col[keep], f[keep], t[keep], sigma2, edges, ew, 0.0, a0, s0)
    assert np.isclose(a[4, 0], 7.0) and np.isclose(s[4, 0], 3.0)


def test_templates_are_normalized_and_leave_the_model_unchanged() -> None:
    rng = np.random.default_rng(1)
    n, per, E = 10, 60, 3
    col = np.repeat(np.arange(n), per)
    reg = np.array([0] * 5 + [1] * 3 + [2] * 2)
    K = {0: 4, 1: 2, 2: 1}
    bandv = rng.integers(0, 4, len(col)) % np.array([K[r] for r in reg])[col]
    f = rng.normal(100, 10, (len(col), E))
    sigma2 = rng.uniform(1, 4, (n, E))
    a = rng.normal(100, 3, (n, E))
    s = rng.uniform(5, 10, (n, E))
    mus = gn.update_templates(col, reg, bandv, f, a.copy(), s.copy(), sigma2, K)
    a2, s2 = a.copy(), s.copy()
    mus2 = gn.update_templates(col, reg, bandv, f, a2, s2, sigma2, K)
    for R, k in K.items():
        assert mus[R].shape == (k, E)
        assert np.allclose(mus[R], mus2[R])
    # the fitted model a + s mu, before and after the normalization
    for R, k in K.items():
        sel = reg[col] == R
        c, b = col[sel], bandv[sel]
        w = 1.0 / sigma2[c]
        idx = b
        for e in range(E):
            # the raw least-squares template of the region and band
            sn = np.bincount(idx, w[:, e] * s[c, e] * (f[sel, e] - a[c, e]), k)
            sd = np.bincount(idx, w[:, e] * s[c, e] ** 2, k)
            raw = sn / sd
            before = a[c, e] + s[c, e] * raw[b]
            after = a2[c, e] + s2[c, e] * mus2[R][b, e]
            assert np.allclose(before, after)
            wt = np.bincount(idx, w[:, e], k)
            if k == 1:
                assert np.all(mus2[R][:, e] == 0)
            else:
                mean = np.sum(wt * mus2[R][:, e]) / wt.sum()
                rms = np.sqrt(np.sum(wt * mus2[R][:, e] ** 2) / wt.sum())
                assert abs(mean) < 1e-9 and abs(rms - 1) < 1e-9
