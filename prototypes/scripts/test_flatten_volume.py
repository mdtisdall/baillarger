"""Tests of flatten_lib with exact answers.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_flatten_volume.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import flatten_lib as FL  # noqa: E402

STEP = 0.04
N_S = 151  # samples: 6 mm


def pattern(t):
    """A laminar pattern in relative depth (mm): bands 0.3-0.5 mm wide, alternating sign."""
    t = np.asarray(t, float)
    c = [(0.3, 1.0, 0.10), (0.8, -1.2, 0.12), (1.3, 0.8, 0.10), (1.8, -0.9, 0.14), (2.4, 0.7, 0.12),
         (3.0, -0.8, 0.12), (3.6, 0.9, 0.13), (4.3, -0.7, 0.12)]
    return sum(a * np.exp(-0.5 * ((t - m) / s) ** 2) for m, a, s in c)


def two_feature(f):
    return np.stack([f, 0.7 * f])


def test_constant_shift():
    z = np.arange(N_S) * STEP
    s0 = 0.10
    Fv = two_feature(pattern(z))[:, None]
    Fw = two_feature(pattern(z - s0))[:, None]
    M = np.ones((1, N_S), bool)
    s, ncc, en, valid = FL.shifts(Fv, Fw, M, M, STEP)
    w = FL.pair_weights(ncc, en, valid, np.median(en[valid]))
    good = (w[0] > 0.3) & (z > 0.4) & (z < 4.5)
    assert good.sum() > 50
    assert np.max(np.abs(s[0, good] - s0)) < 0.005


def test_depth_varying_shift():
    """w is v stretched: phi(z) = 1.04 z + 0.05, so s(z) = phi(z) - z."""
    z = np.arange(N_S) * STEP
    Fv = two_feature(pattern(z))[:, None]
    Fw = two_feature(pattern((z - 0.05) / 1.04))[:, None]
    M = np.ones((1, N_S), bool)
    s, ncc, en, valid = FL.shifts(Fv, Fw, M, M, STEP)
    w = FL.pair_weights(ncc, en, valid, np.median(en[valid]))
    good = (w[0] > 0.3) & (z > 0.4) & (z < 4.0)
    assert good.sum() > 50
    assert np.median(np.abs(s[0, good] - (0.04 * z[good] + 0.05))) < 0.005
    assert np.max(np.abs(s[0, good] - (0.04 * z[good] + 0.05))) < 0.015


def grid_volume(nx=16, ny=12, sp=0.16, alpha=0.2, beta=0.12, top=4.6, noise=0.0, seed=0):
    """Traces on a plane grid. True horizons z_v(tau) = tau (1 + alpha x) + beta y sin(pi tau / top),
    x, y centred in [-1, 1] so that mean_v z_v(tau) = tau. Each trace ends at z_v(top)."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    x, y = xs.ravel(), ys.ravel()
    N = x.size
    tf = np.linspace(0, top, 2000)
    zt = tf[None] * (1 + alpha * x[:, None]) + beta * y[:, None] * np.sin(np.pi * tf[None] / top)
    zs = np.arange(N_S) * STEP
    F = np.full((2, N, N_S), np.nan)
    ns = np.zeros(N, int)
    rng = np.random.default_rng(seed)
    for i in range(N):
        n = int(np.floor(zt[i, -1] / STEP)) + 1
        ns[i] = n
        t_of_z = np.interp(zs[:n], zt[i], tf)
        f = pattern(t_of_z) + noise * rng.standard_normal(n)
        F[:, i, :n] = two_feature(f)
    P = np.c_[xs.ravel(), ys.ravel()]
    pos = np.c_[np.repeat(np.arange(nx), ny), np.tile(np.arange(ny), nx)] * sp
    from scipy.spatial import cKDTree

    pairs = np.array(sorted(cKDTree(pos).query_pairs(0.25)))
    return F, ns, pairs, x, y, (alpha, beta, top), P


def run_fit(F, ns, pairs, eps=0.2, n_outer=4):
    M = np.isfinite(F[0])
    Fz = np.nan_to_num(F)
    v, w = pairs[:, 0], pairs[:, 1]
    s, ncc, en, valid = FL.shifts(Fz[:, v], Fz[:, w], M[v], M[w], STEP)
    W = FL.pair_weights(ncc, en, valid, np.median(en[valid]))
    tau, r, info = FL.fit_warp(pairs, s, W, F.shape[1], STEP, tau_max=4.4, eps=eps, n_outer=n_outer)
    return tau, r, info


def truth(x, y, tau, prm):
    alpha, beta, top = prm
    return tau[None] * (1 + alpha * x[:, None]) + beta * y[:, None] * np.sin(np.pi * tau[None] / top)


def test_fit_recovers_warp():
    F, ns, pairs, x, y, prm, _ = grid_volume()
    tau, r, info = run_fit(F, ns, pairs)
    z = tau[None] + r
    zt = truth(x, y, tau, prm)
    inner = (tau > 0.3) & (tau < 4.2)
    err = np.abs(z - zt)[:, inner]
    assert np.median(err) < 0.01
    assert np.percentile(err, 95) < 0.03


def test_gauge_mean_depth():
    """tau is the mean depth of its horizon: mean_v r_v(tau) = 0 at every level."""
    F, ns, pairs, x, y, prm, _ = grid_volume()
    tau, r, _ = run_fit(F, ns, pairs)
    assert np.max(np.abs(r.mean(0))) < 1e-4


def test_flattened_traces_align():
    """Resampled at constant tau, every trace matches the pattern."""
    F, ns, pairs, x, y, prm, _ = grid_volume()
    tau, r, _ = run_fit(F, ns, pairs)
    G = FL.resample(F, ns, tau[None] + r, STEP)
    ref = pattern(tau)
    inner = (tau > 0.3) & (tau < 4.2)
    d = np.abs(G[0][:, inner] - ref[inner])
    assert np.nanmedian(d) < 0.02


def test_noise():
    F, ns, pairs, x, y, prm, _ = grid_volume(noise=0.2)
    tau, r, _ = run_fit(F, ns, pairs)
    zt = truth(x, y, tau, prm)
    inner = (tau > 0.3) & (tau < 4.2)
    err = np.abs(tau[None] + r - zt)[:, inner]
    assert np.median(err) < 0.03


def test_tau_of_inverts():
    F, ns, pairs, x, y, prm, _ = grid_volume()
    tau = np.arange(0, 4.41, 0.08)
    z = truth(x, y, tau, prm)
    t = FL.tau_of(z[:, 20], z, tau)
    assert np.max(np.abs(t - tau[20])) < 1e-9
