"""Tests for orient_phantoms: the phantoms' stated patterns, labels and
noise statistics."""

import numpy as np
import orient_phantoms as op

SIGMA = np.array([1.0, 0.8, 0.6])
R3 = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.4], [0.2, 0.4, 1.0]])


def test_plane_pattern() -> None:
    shape = (8, 9, 10)
    vox = (0.5, 0.3, 0.2)
    normal = (1.0, 2.0, -2.0)
    amp = [0.2, 0.15, 0.1]
    base = [100.0, 80.0, 60.0]
    im = op.laminae_plane(shape, vox, normal, 1.7, amp, base)
    assert im.shape == (3, *shape)
    n = np.array(normal) / 3.0
    for idx in [(0, 0, 0), (3, 4, 5), (7, 8, 9), (2, 7, 1)]:
        x = np.array(idx) * np.array(vox)
        s = np.sin(2 * np.pi * (n @ x) / 1.7)
        for e in range(3):
            assert np.isclose(np.log(im[(e, *idx)]), np.log(base[e]) + amp[e] * s)


def test_sphere_labels_and_cortex() -> None:
    shape = (30, 30, 30)
    vox = (0.5, 0.5, 0.5)
    c = (7.0, 7.0, 7.0)
    amp = [0.3, 0.2, 0.1]
    base = [100.0, 80.0, 60.0]
    ech, lab = op.laminae_sphere(shape, vox, c, 3.0, 5.0, 1.2, amp, base)
    assert lab.dtype == np.int16 and ech.shape == (3, *shape)
    ii = np.indices(shape).astype(float) * 0.5
    r = np.sqrt(sum((ii[a] - c[a]) ** 2 for a in range(3)))
    assert np.all(lab[r < 3.0] == 41)
    assert np.all(lab[(r >= 3.0) & (r < 5.0)] == 42)
    assert np.all(lab[r >= 5.0] == 0)
    cx = lab == 42
    assert cx.any() and (lab == 41).any() and (lab == 0).any()
    for e in range(3):
        want = np.log(base[e]) + amp[e] * np.sin(2 * np.pi * (r - 3.0) / 1.2)
        assert np.allclose(np.log(ech[e][cx]), want[cx])
        assert np.allclose(ech[e][lab == 41], base[e] * 1.3)
        assert np.all(ech[e][lab == 0] == 1.0)


def test_log_noise_covariance() -> None:
    # Each entry within 3 standard errors; se^2 = (C_ii C_jj + C_ij^2) / n.
    rng = np.random.default_rng(1)
    C = np.array([[1.0, 0.3, 0.1], [0.3, 0.5, 0.2], [0.1, 0.2, 0.4]])
    n = 200_000
    x = op.log_noise(rng, (n,), C)
    assert x.shape == (3, n)
    S = np.cov(x)
    se = np.sqrt((np.outer(np.diag(C), np.diag(C)) + C**2) / n)
    assert np.all(np.abs(S - C) < 3 * se)


def test_chi_moments_rayleigh() -> None:
    mean, var = op.chi_moments(np.array([2.0]), 1)
    assert np.isclose(mean[0], 2.0 * np.sqrt(np.pi / 2))
    assert np.isclose(var[0], (4 - np.pi) / 2 * 4.0)


def test_background_moments() -> None:
    n = 200_000
    for ch in (1, 4):
        rng = np.random.default_rng(10 + ch)
        m = op.background(rng, (n,), SIGMA, R3, ch)
        mean, var = op.chi_moments(SIGMA, ch)
        assert np.allclose(m.mean(axis=1), mean, rtol=0.01)
        assert np.allclose(m.var(axis=1), var, rtol=0.01)


def test_background_identity_uncorrelated() -> None:
    rng = np.random.default_rng(3)
    m = op.background(rng, (200_000,), SIGMA, np.eye(3), 2)
    c = np.asarray(np.corrcoef(m))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert abs(c[i, j]) < 0.01


def test_signal_with_noise_high_snr() -> None:
    rng = np.random.default_rng(4)
    n = 200_000
    # signal_e = 100 * sigma_e, so SNR = 100 in every echo.
    signal = np.tile((100.0 * SIGMA)[:, None], (1, n))
    mag = op.signal_with_noise(rng, signal, SIGMA, R3)
    noise = np.log(mag) - np.log(signal)
    assert np.allclose(noise.std(axis=1), SIGMA / signal[:, 0], rtol=0.02)
    c = np.corrcoef(noise)
    assert np.allclose(c, R3, atol=0.02)
