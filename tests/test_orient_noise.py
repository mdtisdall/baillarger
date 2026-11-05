"""orient, Part B: the measurement of the noise covariance C
(docs/formats/orient.md, "The noise file")."""

import math
from pathlib import Path

import numpy as np
import orient_phantoms as op
import pytest

from baillarger import orient as O
from baillarger import orient_noise as N
from baillarger.bids import read_sidecar

DOC = Path(__file__).resolve().parents[1] / "docs" / "formats" / "orient.md"


@pytest.mark.parametrize("channels", [1, 2, 8, 32])
def test_the_chi_fit_inverts_the_moments(channels: int) -> None:
    mean, var = op.chi_moments(np.array([31.0]), channels)
    n, s = N.fit_chi(float(mean[0]), float(var[0]))
    assert n == pytest.approx(channels, rel=1e-6)
    assert s == pytest.approx(31.0, rel=1e-6)


def test_the_chi_ratio() -> None:
    assert N.chi_ratio(1.0) == pytest.approx(math.pi / 4)
    assert N.chi_ratio(1000.0) > 0.999


@pytest.mark.parametrize("channels", [1, 4])
def test_the_thermal_part_of_synthetic_background(channels: int) -> None:
    """Background with known sigma_e, channels and R: the measurement
    recovers sigma within 1 %, the channels within 5 %, and R within 0.03
    (its sampling error: about 1 / sqrt(voxels) on c = R^2, so about 0.003 /
    (2 R) on R, larger near R = 0; 0.03 holds for R >= 0.2)."""
    rng = np.random.default_rng(7)
    sigma = np.array([35.0, 32.0, 31.0])
    R = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.6], [0.3, 0.6, 1.0]])
    halo = O.radius(N.BG_SCALE)
    stats = []
    for b in range(12):
        x = op.background(rng, (32 + 2 * halo,) * 3, sigma, R, channels)
        stats.append(N.block_stats(x, halo, (b, 0, 0)))
    th = N.thermal_from_blocks(stats)
    np.testing.assert_allclose(th["sigma"], sigma, rtol=0.01)
    assert th["channels"] == pytest.approx(channels, rel=0.05)
    np.testing.assert_allclose(th["R"], R, atol=0.03)
    assert th["signal_fraction"] == 0.0


def test_the_rest_recovers_a_known_covariance() -> None:
    """Laminae on a plane (three echoes, different amplitudes), plus white
    noise in the log with a known covariance C_known and no thermal part
    (sigma = 0): the rest is g C_known within 8 % on the diagonal and 0.08 g
    max(C) off it. The laminar direction is removed, so the laminae add
    nothing.

    The estimate is low by 2 to 5 % (measured on 104^3 voxels, three
    seeds: 0.95, 0.96 and 0.98 of the diagonal for these amplitudes, the
    same at every signal scale): n is fitted in the same window as the
    noise, so it follows part of the noise, and the in-plane residual loses
    about that part, most in the echo with the largest share of J. The rest
    of the tolerance is the sampling error of 32,768 correlated voxels
    (about 3 %)."""
    rng = np.random.default_rng(11)
    shape = (56, 56, 56)
    vox = (0.16, 0.16, 0.16)
    ech = op.laminae_plane(
        shape, vox, (1.0, 2.0, 2.0), 0.8, (0.4, 0.3, 0.2), (600.0, 350.0, 200.0)
    )
    C = np.array([[4.0, 2.4, 1.5], [2.4, 6.0, 3.0], [1.5, 3.0, 9.0]]) * 1e-4
    logs = np.log(ech) + op.log_noise(rng, shape, C)
    ctx = np.ones(shape, bool)
    lv = O.local_level(np.exp(logs), ctx, 0.64, vox)
    det = logs - lv
    keep = np.zeros(shape, bool)
    keep[12:-12, 12:-12, 12:-12] = True
    sums = N.rest_sums(det, ctx, lv, np.zeros(3), np.eye(3), O.gradient_gain(), keep)
    c0 = N.COHERENCE[0]
    assert sums.voxels[c0] > 20000
    rest = sums.total[c0] / sums.voxels[c0] - sums.thermal[c0] / sums.voxels[c0]
    want = O.gradient_gain() * C
    np.testing.assert_allclose(np.diag(rest), np.diag(want), rtol=0.08)
    assert np.max(np.abs(rest - want)) < 0.08 * want.max()
    assert np.all(np.diag(rest) < np.diag(want))


def test_the_thermal_c_model() -> None:
    lv = np.log(np.array([[400.0, 800.0], [200.0, 100.0]]))  # (E=2, 2 voxels)
    sigma = np.array([30.0, 20.0])
    R = np.array([[1.0, 0.25], [0.25, 1.0]])
    c = N.thermal_c(lv, sigma, R, 2.0)
    assert c.shape == (2, 2, 2)
    assert c[0, 0, 0] == pytest.approx(2.0 * (30 / 400) ** 2)
    assert c[0, 1, 1] == pytest.approx(2.0 * 0.25 * (30 / 800) * (20 / 100))


def test_the_spec_lists_the_noise_keys() -> None:
    text = DOC.read_text()
    assert f"SpecVersion `{N.NOISE_VERSION}`" in text
    for k in N.NOISE_SPEC.keys:
        assert f"| `{k.name}` |" in text, k.name


def test_the_noise_file_round_trip(tmp_path: Path) -> None:
    """A noise file written from sums reads back as the model."""
    from test_orient import write_files

    paths, dseg = write_files(tmp_path)
    s = O.OrientSampler(paths, dseg, cells=8)
    th = dict(
        sigma=np.array([3.0, 2.0, 1.0]),
        channels=1.0,
        R=np.eye(3),
        sqcorr=np.eye(3),
        region=dict(Kind="background"),
    )
    sums = N.measure_rest(s, th, level=1)
    d = N.noise_file(s, th, sums, 1, O.SIGMA)
    path = N.noise_path(tmp_path / "deriv", paths[0])
    assert path.name == "sub-T01_ses-S1_desc-orientnoise_noise.json"
    from baillarger.bids import write_sidecar

    write_sidecar(path, N.NOISE_SPEC, d)
    assert read_sidecar(path, N.NOISE_SPEC)["Echoes"] == 3
    model = N.read_noise(path)
    np.testing.assert_allclose(model.sigma, [3.0, 2.0, 1.0])
    lv = np.log(np.full((3, 4), 100.0))
    c = model.c(lv)
    assert c.shape == (3, 3, 4)
    np.testing.assert_allclose(
        c[..., 0], model.gain * np.diag([9.0, 4.0, 1.0]) / 1e4 + model.rest
    )
