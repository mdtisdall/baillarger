"""Tests of baillarger.flatten with exact answers."""

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from baillarger import flatten as FL

type Float = npt.NDArray[np.float64]

STEP = 0.04
N_S = 151  # samples: 6 mm


def pattern(t: Any) -> Float:
    """A laminar pattern in relative depth (mm): bands 0.3-0.5 mm wide, alternating
    sign."""
    t = np.asarray(t, float)
    c = [
        (0.3, 1.0, 0.10),
        (0.8, -1.2, 0.12),
        (1.3, 0.8, 0.10),
        (1.8, -0.9, 0.14),
        (2.4, 0.7, 0.12),
        (3.0, -0.8, 0.12),
        (3.6, 0.9, 0.13),
        (4.3, -0.7, 0.12),
    ]
    out: Float = sum(
        (a * np.exp(-0.5 * ((t - m) / s) ** 2) for m, a, s in c), np.zeros_like(t)
    )
    return out


def two_feature(f: Float) -> Float:
    return np.stack([f, 0.7 * f])


def test_constant_shift() -> None:
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


def test_depth_varying_shift() -> None:
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


def grid_volume(
    nx: int = 16,
    ny: int = 12,
    sp: float = 0.16,
    alpha: float = 0.2,
    beta: float = 0.12,
    top: float = 4.6,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[Any, ...]:
    """Traces on a plane grid. True horizons z_v(tau) = tau (1 + alpha x) + beta y
    sin(pi tau / top), x, y centred in [-1, 1] so that mean_v z_v(tau) = tau. Each trace
    ends at z_v(top)."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    x, y = xs.ravel(), ys.ravel()
    N = x.size
    tf = np.linspace(0, top, 2000)
    zt = tf[None] * (1 + alpha * x[:, None]) + beta * y[:, None] * np.sin(
        np.pi * tf[None] / top
    )
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
    pairs = np.array(sorted(cKDTree(pos).query_pairs(0.25)))
    return F, ns, pairs, x, y, (alpha, beta, top), P


def run_fit(
    F: Float, ns: Any, pairs: Any, eps: float = 0.2, n_outer: int = 4
) -> tuple[Any, ...]:
    M = np.isfinite(F[0])
    Fz = np.nan_to_num(F)
    v, w = pairs[:, 0], pairs[:, 1]
    s, ncc, en, valid = FL.shifts(Fz[:, v], Fz[:, w], M[v], M[w], STEP)
    W = FL.pair_weights(ncc, en, valid, np.median(en[valid]))
    tau, r, info = FL.fit_warp(
        pairs, s, W, F.shape[1], STEP, tau_max=4.4, eps=eps, n_outer=n_outer
    )
    return tau, r, info


def truth(x: Float, y: Float, tau: Float, prm: Any) -> Float:
    alpha, beta, top = prm
    out: Float = tau[None] * (1 + alpha * x[:, None]) + beta * y[:, None] * np.sin(
        np.pi * tau[None] / top
    )
    return out


def test_fit_recovers_warp() -> None:
    F, ns, pairs, x, y, prm, _ = grid_volume()
    tau, r, info = run_fit(F, ns, pairs)
    z = tau[None] + r
    zt = truth(x, y, tau, prm)
    inner = (tau > 0.3) & (tau < 4.2)
    err = np.abs(z - zt)[:, inner]
    assert np.median(err) < 0.01
    assert np.percentile(err, 95) < 0.03


def test_gauge_mean_depth() -> None:
    """tau is the mean depth of its horizon: mean_v r_v(tau) = 0 at every level."""
    F, ns, pairs, x, y, prm, _ = grid_volume()
    tau, r, _ = run_fit(F, ns, pairs)
    assert np.max(np.abs(r.mean(0))) < 1e-4


def test_flattened_traces_align() -> None:
    """Resampled at constant tau, every trace matches the pattern."""
    F, ns, pairs, x, y, prm, _ = grid_volume()
    tau, r, _ = run_fit(F, ns, pairs)
    G = FL.resample(F, ns, tau[None] + r, STEP)
    ref = pattern(tau)
    inner = (tau > 0.3) & (tau < 4.2)
    d = np.abs(G[0][:, inner] - ref[inner])
    assert np.nanmedian(d) < 0.02


def test_noise() -> None:
    F, ns, pairs, x, y, prm, _ = grid_volume(noise=0.2)
    tau, r, _ = run_fit(F, ns, pairs)
    zt = truth(x, y, tau, prm)
    inner = (tau > 0.3) & (tau < 4.2)
    err = np.abs(tau[None] + r - zt)[:, inner]
    assert np.median(err) < 0.03


def test_tau_of_inverts() -> None:
    F, ns, pairs, x, y, prm, _ = grid_volume()
    tau = np.arange(0, 4.41, 0.08)
    z = truth(x, y, tau, prm)
    t = FL.tau_of(z[:, 20], z, tau)
    assert np.max(np.abs(t - tau[20])) < 1e-9


# Dynamic warping (dp_lags, dynamic_shifts): shifts beyond the gradient estimator's
# reach.


def brute_lags(E: Float, m: int) -> tuple[float, Any]:
    """The minimum of sum_i E[i, l_i] over every path with |l_{i+1} - l_i| <= m, by
    enumeration."""
    n, nl = E.shape
    best, arg = np.inf, None
    for p in itertools.product(range(nl), repeat=n):
        if all(abs(p[i + 1] - p[i]) <= m for i in range(n - 1)):
            c = sum(E[i, p[i]] for i in range(n))
            if c < best:
                best, arg = c, p
    return best, np.array(arg)


def test_dp_lags_exact() -> None:
    """The accumulation and backtrack give a minimum-cost path, against enumeration."""
    rng = np.random.default_rng(1)
    for m in (0, 1, 2):
        for _ in range(20):
            E = rng.random((3, 6, 5))  # 3 pairs, 6 samples, 5 lags
            lags = FL.dp_lags(E, m)
            for k in range(3):
                best, _ = brute_lags(E[k], m)
                path = lags[k]
                assert np.all(np.abs(np.diff(path)) <= m)
                assert abs(E[k][np.arange(6), path].sum() - best) < 1e-12


def test_large_constant_shift() -> None:
    """A shift of 0.6 mm, past half the spacing of same-sign bands (about 1.1 mm): the
    gradient estimator, even with its bound opened to 0.8 mm, slides to the wrong band
    (on this pattern it holds to about 0.5 mm); dynamic warping recovers it."""
    z = np.arange(N_S) * STEP
    s0 = 0.6
    Fv = two_feature(pattern(z))[:, None]
    Fw = two_feature(pattern(z - s0))[:, None]
    M = np.ones((1, N_S), bool)
    good = (z > 0.6) & (z < 4.2)
    s_lk, *_ = FL.shifts(Fv, Fw, M, M, STEP, smax=0.8)
    assert np.median(np.abs(s_lk[0, good] - s0)) > 0.05
    s, ncc, en, valid = FL.dynamic_shifts(Fv, Fw, M, M, STEP, smax=0.8)
    w = FL.pair_weights(ncc, en, valid, np.median(en[valid]))
    ok = good & (w[0] > 0.3)
    assert ok.sum() > 50
    assert np.max(np.abs(s[0, ok] - s0)) < 0.005


def test_large_depth_varying_shift() -> None:
    """w is v stretched by 10 % and moved 0.2 mm: s(z) = 0.1 z + 0.2, up to 0.6 mm."""
    z = np.arange(N_S) * STEP
    Fv = two_feature(pattern(z))[:, None]
    Fw = two_feature(pattern((z - 0.2) / 1.1))[:, None]
    M = np.ones((1, N_S), bool)
    s, ncc, en, valid = FL.dynamic_shifts(Fv, Fw, M, M, STEP, smax=0.8)
    w = FL.pair_weights(ncc, en, valid, np.median(en[valid]))
    true = 0.1 * z + 0.2
    ok = (z > 0.4) & (z < 3.6) & (w[0] > 0.3)
    assert ok.sum() > 50
    assert np.median(np.abs(s[0, ok] - true[ok])) < 0.005
    assert np.max(np.abs(s[0, ok] - true[ok])) < 0.015


def test_dynamic_strain_bound() -> None:
    """Without the gradient polish the shift's slope never exceeds the strain bound."""
    z = np.arange(N_S) * STEP
    rng = np.random.default_rng(2)
    Fv = rng.standard_normal((2, 4, N_S))
    Fw = rng.standard_normal((2, 4, N_S))
    M = np.ones((4, N_S), bool)
    s, *_ = FL.dynamic_shifts(Fv, Fw, M, M, STEP, smax=0.5, strain=0.15, polish=0)
    assert np.max(np.abs(np.diff(s, axis=1))) / STEP <= 0.15 + 1e-9
    assert np.max(np.abs(s)) <= 0.5 + 1e-9
    del z


def test_dynamic_small_shifts_agree() -> None:
    """Where the gradient estimator works (0.10 mm), dynamic warping agrees with it."""
    z = np.arange(N_S) * STEP
    Fv = two_feature(pattern(z))[:, None]
    Fw = two_feature(pattern(z - 0.10))[:, None]
    M = np.ones((1, N_S), bool)
    s_lk, *_ = FL.shifts(Fv, Fw, M, M, STEP)
    s, ncc, en, valid = FL.dynamic_shifts(Fv, Fw, M, M, STEP)
    good = (z > 0.4) & (z < 4.5)
    assert np.max(np.abs(s[0, good] - s_lk[0, good])) < 0.003


def test_dynamic_short_trace() -> None:
    """w ends early (nan past 2.5 mm): the shift on v's samples with a partner is still
    right, and samples whose partner is past w's end are not valid. Near the partner's
    end (z = 2.2) the gradient iterations' window (`shifts`, also the polish here)
    biases both estimators: measured 0.007 mm at 0.7 mm from it, 0.01 mm at 0.5 mm, 0.06
    mm at 0.2 mm. Checked as two bounds, not hidden."""
    z = np.arange(N_S) * STEP
    s0 = 0.3
    Fv = two_feature(pattern(z))[:, None]
    Fw = two_feature(pattern(z - s0))[:, None]
    Mv = np.ones((1, N_S), bool)
    Mw = (z < 2.5)[None]
    s, ncc, en, valid = FL.dynamic_shifts(
        Fv, np.where(Mw[None], Fw, 0.0), Mv, Mw, STEP, smax=0.6
    )
    w = FL.pair_weights(ncc, en, valid, np.median(en[valid]))
    ok = (z > 0.6) & (z < 1.7) & (w[0] > 0.3)
    assert ok.sum() > 20
    far = ok & (z < 1.2)
    assert np.max(np.abs(s[0, far] - s0)) < 0.005
    assert np.max(np.abs(s[0, ok] - s0)) < 0.015
    assert not valid[0, z > 2.5 - s0 + 0.05].any()


def grid_shifted() -> tuple[Any, ...]:
    """grid_volume with fewer, wider-spaced columns in x and a strong sideways thickness
    change: neighbours in x differ by 0.35 * 2 / 7 * tau, 0.36 mm at the top (tau 3.6),
    strain up to 0.15."""
    return grid_volume(nx=8, alpha=0.35, beta=0.12, top=3.6)


def run_fit_dyn(F: Float, ns: Any, pairs: Any, smax: float = 0.6) -> tuple[Any, ...]:
    M = np.isfinite(F[0])
    Fz = np.nan_to_num(F)
    v, w = pairs[:, 0], pairs[:, 1]
    s, ncc, en, valid = FL.dynamic_shifts(
        Fz[:, v], Fz[:, w], M[v], M[w], STEP, smax=smax
    )
    W = FL.pair_weights(ncc, en, valid, np.median(en[valid]))
    return FL.fit_warp(pairs, s, W, F.shape[1], STEP, tau_max=3.6, eps=0.2, n_outer=4)


def test_fit_recovers_large_warp() -> None:
    """The fit from dynamic shifts recovers a warp whose neighbour shifts reach 0.36
    mm."""
    F, ns, pairs, x, y, prm, _ = grid_shifted()
    tau, r, _ = run_fit_dyn(F, ns, pairs)
    zt = truth(x, y, tau, prm)
    inner = (tau > 0.3) & (tau < 3.4)
    err = np.abs(tau[None] + r - zt)[:, inner]
    assert np.median(err) < 0.01
    assert np.percentile(err, 95) < 0.03


def test_dynamic_matches_gradient_on_a_smooth_small_shift() -> None:
    """s(z) = 0.03 z, within the gradient estimator's reach: dynamic warping gives the
    same shift. The gradient polish only fixes the shift's average over its window, so
    the lag path's ripple (kinks every `coarse` mm) must be smoothed away before it."""
    z = np.arange(N_S) * STEP
    Fv = two_feature(pattern(z))[:, None]
    Fw = two_feature(pattern(z / 1.03))[:, None]
    M = np.ones((1, N_S), bool)
    true = 0.03 * z
    s_lk, *_ = FL.shifts(Fv, Fw, M, M, STEP)
    s, *_ = FL.dynamic_shifts(Fv, Fw, M, M, STEP)
    good = (z > 0.4) & (z < 4.0)
    assert np.median(np.abs(s_lk[0, good] - true[good])) < 0.0005
    assert np.median(np.abs(s[0, good] - true[good])) < 0.0005
