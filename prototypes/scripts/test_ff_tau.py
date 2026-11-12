"""Tests of ff_tau: the τ phantom's labels, profile, image, depth, speeds, offsets and stop
distance, with exact answers.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_tau.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_tau as FT  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

SHAPE = (32, 48, 16)  # small: local x across axis 0, u along axis 1
H = FP.H
_R = (np.arange(50) + 0.5) / 50


def _synthetic_profile():
    """A profile shaped like the site's: a small bright band near 0.18, a dark band near 0.36, a
    large bright band near 0.81 and a dark top near 0.96 (in the site's depth, before the flat
    band's map)."""
    g = lambda c, w: np.exp(-(((_R - c) / w) ** 2))  # noqa: E731
    return list(0.1 * g(0.18, 0.05) - 0.08 * g(0.36, 0.05) + 0.25 * g(0.81, 0.08) - 0.6 * g(0.96, 0.03))


FEATS = {e: dict(profile=_synthetic_profile(), wm_level=-0.3, fluid_level=-2.0, hp_stat=0.0)
         for e in ("echo2", "echo3")}
FEATS_NOISY = {e: dict(FEATS[e], hp_stat=0.05) for e in FEATS}


def _local(X):
    """Voxel positions to local mm, from the definition (independent of Fold.to_local)."""
    R = FP.R_DEFAULT
    c = (np.array(SHAPE) - 1) / 2.0
    origin = -0.5 * 2.4 * R[:, 0]
    return ((np.asarray(X, float) - c) * H - origin) @ R


def _t(u):
    return 2.4 * (1 + 0.25 * np.sin(2 * np.pi * u / 6.0))


def _dt(u):
    return 2.4 * 0.25 * (2 * np.pi / 6.0) * np.cos(2 * np.pi * u / 6.0)


@pytest.fixture(scope="module")
def ph():
    return FT.tau(FEATS, shape=SHAPE)


# ---------------------------------------------------------------- 1. labels


def test_labels_match_closed_form_at_every_voxel_centre(ph):
    X = np.indices(SHAPE).reshape(3, -1).T.astype(float)
    Y = _local(X)
    x, t = Y[:, 0], _t(Y[:, 1])
    lab = np.where(x < 0, 2, np.where(x > t, 1, 0)).reshape(SHAPE)
    assert np.array_equal(ph.labels, lab)
    assert set(np.unique(lab)) == {0, 1, 2}
    # the box centre is the local point (t0 / 2, 0, 0)
    assert np.allclose(_local((np.array(SHAPE) - 1) / 2.0), [1.2, 0.0, 0.0])
    assert np.allclose(ph.geometry.to_local(X[:100]), Y[:100])
    g = ph.geometry
    assert np.array_equal(g.stopped(Y), g.label(Y) == T.MEDIUM)
    assert np.all(g.bank(Y) == 0)


# ---------------------------------------------------------------- 2. the flat band and the render


def test_profile_constant_over_the_flat_band_and_continuous():
    prof = np.array(FEATS["echo2"]["profile"])
    c = FP.profile_at(prof, 0.40)
    r = np.linspace(0.40, 0.55, 301)
    for u in (-1.0, 1.0):
        p = FT.profile_tau(prof, r, np.full_like(r, u))
        assert np.all(p == c)
        # continuous at both edges: the change over eps is at most (largest slope) * eps
        slope = np.abs(np.diff(prof)).max() * 50 * (1 - 0.40) / (1 - 0.55)
        for edge in (0.40, 0.55):
            for eps in (1e-6, 1e-4):
                q = FT.profile_tau(prof, np.array([edge - eps, edge + eps]), np.full(2, u))
                assert abs(q[0] - c) <= slope * eps + 1e-15
                assert abs(q[1] - c) <= slope * eps + 1e-15


def test_render_reproduces_profile_tau_and_the_flat_band(ph):
    geo = ph.geometry
    labels, imgs = FT.render_tau(geo, FEATS)
    X = np.indices(SHAPE).reshape(3, -1).T.astype(float)
    offs = (np.arange(3) + 0.5) / 3 - 0.5
    O = np.array([[a, b, c] for a in offs for b in offs for c in offs])
    phis = np.stack([geo.phi(geo.to_local(X + o)) for o in O], 1)
    us = np.stack([geo.to_local(X + o)[:, 1] for o in O], 1)
    inside = np.all((phis >= 0) & (phis <= 1), 1)
    band = np.all((phis >= 0.40) & (phis <= 0.55), 1)
    one_side = np.all(us > 0, 1) | np.all(us <= 0, 1)
    assert inside.sum() > 1000 and (band & one_side).sum() > 100
    for k, e in enumerate(("echo2", "echo3")):
        prof = np.array(FEATS[e]["profile"])
        img = imgs[k].ravel()
        ref = np.mean([FT.profile_tau(prof, phis[:, j], us[:, j]) for j in range(len(O))], 0)
        assert np.max(np.abs(img[inside] - ref[inside])) < 1e-12
        assert np.max(np.abs(img[band & one_side] - FP.profile_at(prof, 0.40))) < 1e-12


def test_plain_case_gives_the_arrays_of_fold_phantoms_make(ph):
    geo = ph.geometry
    a = FT.make_tau(geo, FEATS_NOISY, noise=1.0, seed=3, band=None, scale=1.0)
    b = FP.make(geo, None, noise=1.0, seed=3, feats=FEATS_NOISY)
    assert a["noise_sd"] == b["noise_sd"] and a["noise_sd"][0] > 0
    for k in ("ctx", "wm", "rim", "miss", "vox", "dface", "labels", "lo"):
        assert np.array_equal(a[k], b[k]), k
    for k in (1, 2):
        assert np.array_equal(a["det"][k], b["det"][k])


# ---------------------------------------------------------------- 3. the areal border


def test_no_band_moves_at_the_areal_border():
    r = np.linspace(0.0, 1.0, 20001)
    for e in ("echo2", "echo3"):
        prof = np.array(FEATS[e]["profile"])
        a, b = FT.trend_tau(prof)
        left = FT.profile_tau(prof, r, np.full_like(r, -1.0))
        right = FT.profile_tau(prof, r, np.full_like(r, 1.0))
        assert np.array_equal(FT.profile_tau(prof, r, np.zeros_like(r)), left)  # u = 0 is the left side
        B = FT.bands_tau(prof, r)
        assert B.shape == (2, len(r))
        # the two sides differ by exactly -0.5 times the same bands
        assert np.max(np.abs((right - left) + 0.5 * B.sum(0))) < 1e-15
        res_l, res_r = left - (a + b * r), right - (a + b * r)
        out = np.ones_like(r, bool)
        for k, (w0, c0, c1, w1) in enumerate(FT.WINDOWS):
            win = (r >= w0) & (r <= w1)
            out &= ~win
            # the band's extremum is in the core, and the band is real
            ib = np.argmax(np.abs(B[k]))
            assert c0 < r[ib] < c1 and abs(B[k, ib]) > 0.01
            # in the core, the image's own extremum (after the trend) is at the same depth on the
            # two sides (one sample of 5e-5), with half the amplitude
            core = np.flatnonzero((r >= c0) & (r <= c1))
            i_l = core[np.argmax(np.abs(res_l[core]))]
            i_r = core[np.argmax(np.abs(res_r[core]))]
            assert i_l == i_r == ib
            assert res_r[i_r] / res_l[i_l] == pytest.approx(0.5, abs=1e-12)
        assert np.array_equal(left[out], right[out])  # nothing changes outside the windows


# ---------------------------------------------------------------- 4. depth, speeds, offsets


def test_phi_and_grad_phi(ph):
    geo = ph.geometry
    Y = np.random.default_rng(0).uniform([-3, -8, -2], [4, 8, 2], (300, 3))
    assert np.allclose(geo.phi(Y), Y[:, 0] / _t(Y[:, 1]), rtol=0, atol=1e-14)
    G = geo.grad_phi(Y)
    eps = 1e-6
    for a in range(3):
        d = np.zeros(3)
        d[a] = eps
        fd = (geo.phi(Y + d) - geo.phi(Y - d)) / (2 * eps)
        assert np.max(np.abs(fd - G[:, a])) < 1e-7
    assert np.all(G[:, 2] == 0)


def test_true_speed_and_offset(ph):
    geo = ph.geometry
    # where t'(u) = 0 (u = +-P/4, the sine's extrema), the speed is t(u) / tau_scale
    for u, t in ((1.5, 3.0), (-1.5, 1.8)):
        Y = np.array([[x, u, z] for x in (-1.0, 0.0, 0.7, 1.5) for z in (-0.5, 0.5)])
        v = FT.true_speed_at(ph, geo.to_vox(Y))
        assert np.allclose(v, t / 2.4, rtol=1e-12, atol=0)
    # everywhere: 1 / (tau_scale |grad phi|), |grad phi| from the closed form
    Y = np.random.default_rng(1).uniform([-3, -8, -2], [4, 8, 2], (300, 3))
    x, u = Y[:, 0], Y[:, 1]
    g = np.sqrt(1 / _t(u) ** 2 + x**2 * _dt(u) ** 2 / _t(u) ** 4)
    assert np.allclose(FT.true_speed_at(ph, geo.to_vox(Y)), 1 / (2.4 * g), rtol=1e-12, atol=0)
    # offsets: the true tau at root - 2 mm along +x
    u = np.linspace(-6, 6, 41)
    roots = geo.to_vox(np.stack([np.zeros_like(u), u, 0.3 * np.ones_like(u)], 1))
    assert np.allclose(FT.true_offset_at(ph, roots), (-2.0 / _t(u) - 1.0) * 2.4, rtol=0, atol=1e-12)
    assert np.allclose(T.true_tau(geo, ph.tau_scale, roots), -2.4, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- 5. stop distance


def test_stop_sdf_zero_on_the_surface_and_exact(ph):
    geo = ph.geometry
    rng = np.random.default_rng(2)
    u = rng.uniform(-8, 8, 200)
    S = np.stack([_t(u), u, rng.uniform(-2, 2, 200)], 1)
    assert np.max(np.abs(geo.stop_sdf(S))) < 1e-12
    n = 120
    u = rng.uniform(-8, 8, n)
    x = _t(u) + rng.uniform(-3.5, 1.5, n)
    Y = np.stack([x, u, rng.uniform(-2, 2, n)], 1)
    d = geo.stop_sdf(Y)
    w = np.linspace(-6.0, 6.0, 1_200_001)  # 1e-5 mm steps
    for i in range(n):
        uu = u[i] + w
        brute = np.sqrt(((x[i] - _t(uu)) ** 2 + w**2).min())
        assert abs(abs(d[i]) - brute) < 1e-6, (x[i], u[i])
        assert np.sign(d[i]) == np.sign(x[i] - _t(u[i]))


# ---------------------------------------------------------------- 6. the phantom


def test_phantom_fields(ph):
    assert ph.name == "tau" and ph.tau_scale == 2.4 and ph.site == "synthetic"
    assert ph.noise == 0.0 and ph.seed == 0 and ph.shape == SHAPE
    assert isinstance(ph.geometry, T.Geometry)
    rec = json.loads(json.dumps(ph.record()))
    p = rec["params"]
    assert p["shape"] == list(SHAPE) and p["R"] == "R_DEFAULT"
    assert p["period_mm"] == 6.0 and p["amp"] == 0.25 and p["t0"] == 2.4
    assert p["band"] == [0.40, 0.55] and p["border_u_mm"] == 0.0 and p["border_scale"] == 0.5
    assert len(p["windows"]) == 2
    assert set(ph.det) == {1, 2}
    # the default box: 96 voxels on the voxel axis nearest the local y (u)
    assert FT.SHAPE == (64, 96, 48) and int(np.argmax(np.abs(FP.R_DEFAULT[:, 1]))) == 1
