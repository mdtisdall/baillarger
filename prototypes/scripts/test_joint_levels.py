"""Tests of joint_levels' plumbing with exact answers.
    nix develop --command uv run --with thinqpbo pytest -q prototypes/scripts/test_joint_levels.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import joint_levels as JL  # noqa: E402


def data(n, top, pairs=np.zeros((0, 2), int), s=0.0, w=1.0, m=200):
    P = len(pairs)
    return JL.Data(step=0.04, ns=np.full(n, m), top=np.asarray(top, float), mu=np.zeros(n),
                   pairs=np.asarray(pairs, int).reshape(-1, 2), S=np.full((P, m), s), W=np.full((P, m), w))


def test_window_recentres_to_an_anchor_outside_it():
    """One open column, top 2.4 mm; the pia's window starts at 1.6 +- 0.32. Re-centring at the edge
    walks it to the anchor."""
    D = data(1, [2.4])
    start = np.array([[0.53], [1.07], [1.6]])
    z, cert, r, hist = JL.solve_pass(D, 0.08, start, np.full(start.shape, 0.32), recentre=6)
    assert abs(z[-1, 0] - 2.4) <= 0.08 + 1e-9
    assert hist[-1]["edge_hits"] == 0
    assert np.all(np.diff(z[:, 0]) >= 0.1 - 1e-9)


def test_conservation_carries_a_shift():
    """Two columns, v open (top 2.0), w with no anchor; at every depth s_vw = +0.2 mm with weight 1.
    The inner levels of w sit 0.2 mm deeper than v's."""
    D = data(2, [2.0, np.nan], pairs=[[0, 1]], s=0.2)
    start = np.array([[0.67, 0.67], [1.33, 1.33], [2.0, 2.0]])
    z, cert, r, hist = JL.solve_pass(D, 0.04, start, np.full(start.shape, 0.5))
    assert abs(z[-1, 0] - 2.0) <= 0.04 + 1e-9
    for l in (0, 1):
        assert abs((z[l, 1] - z[l, 0]) - 0.2) <= 0.04 + 1e-9


def test_tie_places_an_unanchored_pia():
    """Columns a (open, top 2.0) and b (no anchor), facing: a tie a + b = 4.6 mm puts b's pia at 2.6."""
    D = data(2, [2.0, np.nan])
    D.ties = np.array([[0, 1]]); D.tie_L = np.array([4.6]); D.tie_w = np.array([1.0])
    start = np.array([[0.67, 0.8], [1.33, 1.6], [2.0, 2.4]])
    z, cert, r, hist = JL.solve_pass(D, 0.04, start, np.full(start.shape, 0.6))
    assert abs(z[-1, 0] - 2.0) <= 0.04 + 1e-9
    assert abs(z[-1, 1] - 2.6) <= 0.04 + 1e-9


def test_recentring_never_raises_the_energy():
    """Random shifts on a chain of columns, narrow windows: each re-centred window contains the last
    solution and the energy does not depend on the windows, so it never rises."""
    rng = np.random.default_rng(0)
    n, m = 12, 200
    pairs = np.array([(i, i + 1) for i in range(n - 1)])
    top = np.where(rng.random(n) < 0.3, rng.uniform(1.8, 2.6, n), np.nan)
    D = JL.Data(step=0.04, ns=np.full(n, m), top=top, mu=np.zeros(n), pairs=pairs,
                S=rng.uniform(-0.15, 0.15, (len(pairs), m)), W=rng.uniform(0, 1, (len(pairs), m)))
    D.c0 = np.full(n, 2.2)
    start = np.stack([t * np.full(n, 2.2) for t in JL.TAU])
    z, cert, r, hist = JL.solve_pass(D, 0.04, start, np.full(start.shape, 0.12), recentre=15)
    e = [h["energy"] for h in hist]
    assert all(e[i + 1] <= e[i] + 1e-9 for i in range(len(e) - 1))


def test_fill_recovers_a_known_warp_and_ignores_the_far_stack():
    """Traces on a grid: true horizon u at depth t_v phi_v(u), phi_v monotone with phi_v(0) = 0,
    phi_v(1) = 1, thickness t_v varying sideways. Past each pia, a far stack (the pattern mirrored,
    at another thickness), as on crossing traces. Given the true levels, each fitted level set of
    tau is one true horizon across all columns. (Measured: the same errors with the taper below the
    pia switched off, median 0.002 mm, p95 0.015; here the far stack reaches the fill only through
    the shift estimator's window near the pia.) (Between levels tau is only a
    parametrization: a reparametrization shared by all columns changes no equation, so alignment,
    not z_v(tau) itself, is what the data fix.)"""
    import flatten_lib as FL
    from scipy.spatial import cKDTree

    def pattern(t):
        """A laminar pattern in relative depth (mm), as tests/test_flatten.py."""
        t = np.asarray(t, float)
        c = [(0.3, 1.0, 0.10), (0.8, -1.2, 0.12), (1.3, 0.8, 0.10), (1.8, -0.9, 0.14), (2.4, 0.7, 0.12),
             (3.0, -0.8, 0.12), (3.6, 0.9, 0.13), (4.3, -0.7, 0.12)]
        return sum(a * np.exp(-0.5 * ((t - m) / s) ** 2) for m, a, s in c)

    nx, ny, step, m = 14, 10, 0.04, 200
    xs, ys = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    x, y = xs.ravel(), ys.ravel()
    n = x.size
    t = 2.2 + 0.4 * x  # pia (mm)
    phi = lambda u: u + 0.12 * np.sin(np.pi * u) * (1 + 0.5 * y[:, None])  # noqa: E731
    uf = np.linspace(0, 1, 2001)[None]
    zt = t[:, None] * phi(uf)  # (n, 2001)
    zs = np.arange(m) * step
    F = np.zeros((2, n, m))
    for i in range(n):
        u = np.interp(zs, zt[i], uf[0], right=np.nan)
        near = pattern(2.4 * u)  # the bank's own bands in tau (scaled to mm-like spacing)
        far_u = (zs - t[i]) / (1.6 + 0.3 * y[i])  # the far bank, mirrored, another thickness
        far = pattern(2.4 * (1 - far_u))
        f = np.where(np.isfinite(u), near, np.where(far_u <= 1, far, 0.0))
        F[:, i] = np.stack([f, 0.7 * f])
    pos = np.c_[np.repeat(np.arange(nx), ny), np.tile(np.arange(ny), nx)] * 0.16
    pairs = np.array(sorted(cKDTree(pos).query_pairs(0.25)))
    M_ = np.ones((n, m), bool)
    s, ncc, en, va = FL.shifts(F[:, pairs[:, 0]], F[:, pairs[:, 1]], M_[pairs[:, 0]], M_[pairs[:, 1]], step)
    W = FL.pair_weights(ncc, en, va, np.median(en[va]))
    levels = np.stack([t * phi(np.full((1, 1), tt))[:, 0] for tt in JL.TAU])
    tg, z, info = JL.fill_warp(pairs, s, W, step, levels)
    # the true horizon each fitted depth lies on, per column; then its mean over columns
    u = np.stack([np.interp(z[i], zt[i], uf[0]) for i in range(n)])
    ubar = u.mean(0)
    aligned = t[:, None] * phi(ubar[None])  # where the common horizon ubar lies on each column
    inner = (tg > 0.05) & (tg < 0.95)
    err = np.abs(z - aligned)[:, inner]
    assert np.median(err) < 0.01
    assert np.percentile(err, 95) < 0.03
    for l, tt in enumerate(JL.TAU):  # and the levels are kept
        j = int(round(tt * 24))
        assert np.max(np.abs(z[:, j] - levels[l])) < 0.02


def test_band_cost_places_its_level():
    """One open column (top 2.0 mm), levels at tau 0.4 and 1; a dark band's evidence at 1.0 mm
    (a dip in the cost), read through the straight warp. The vertical prior alone puts the level
    at 0.8 mm; with the band's evidence it moves to the dip."""
    D = data(1, [2.0])
    D.tau = (0.4, 1.0)
    tf = np.linspace(0, 1, 101)
    D.band_tf = tf
    D.warp_tg = np.linspace(0, 1, 25)
    D.warp_z = 2.0 * D.warp_tg[None]
    D.band_cost = [(-np.exp(-0.5 * ((tf * 2.0 - 1.0) / 0.08) ** 2))[None], None]
    start = np.array([[0.8], [2.0]])
    z0, *_ = JL.solve_pass(D, 0.04, start, np.full(start.shape, 0.4),
                           prm=JL.Params(w_band=0.0))
    z1, *_ = JL.solve_pass(D, 0.04, start, np.full(start.shape, 0.4), prm=JL.Params(w_band=5.0))
    assert abs(z0[0, 0] - 0.8) <= 0.04 + 1e-9
    assert abs(z1[0, 0] - 1.0) <= 0.04 + 1e-9


def test_interp_rows_matches_np_interp():
    rng = np.random.default_rng(3)
    n, m, k = 50, 30, 12
    xp = np.cumsum(rng.uniform(0.01, 0.2, (n, m)), axis=1)
    fp = rng.normal(size=(n, m))
    x = rng.uniform(-0.5, xp.max() + 0.5, (n, k))
    ref = np.stack([np.interp(x[i], xp[i], fp[i]) for i in range(n)])
    assert np.max(np.abs(JL.interp_rows(x, xp, fp) - ref)) < 1e-12


def test_band_cost_vectorized_matches_the_loop():
    """The band unary built in one batch equals the per-trace loop (monotone warps)."""
    rng = np.random.default_rng(4)
    n = 30
    D = data(n, np.full(n, np.nan))
    D.tau = (0.4, 1.0)
    D.band_tf = np.linspace(0, 1, 101)
    D.warp_tg = np.linspace(0, 1, 25)
    D.warp_z = np.cumsum(rng.uniform(0.05, 0.15, (n, 25)), axis=1) - 0.05
    D.warp_z[:, 0] = 0.0
    D.band_cost = [rng.normal(size=(n, 101)), None]
    centre = np.stack([rng.uniform(0.6, 1.2, n), rng.uniform(2.0, 2.6, n)])
    half = np.full(centre.shape, 0.3)
    M, o = JL.build(D, 0.04, centre, half, JL.Params(w_band=0.7))
    M0, _ = JL.build(D, 0.04, centre, half, JL.Params(w_band=0.0))
    for v in range(n):
        d = 0.04 * (o[0, v] + np.arange(M.K[v]))
        ref = 0.7 * np.interp(np.interp(d, D.warp_z[v], D.warp_tg), D.band_tf, D.band_cost[0][v])
        assert np.max(np.abs((M.unary[v] - M0.unary[v]) - ref)) < 1e-12


def test_unconverged_pass_is_reported():
    """An anchor outside the window and no re-centring allowed: the pass says it did not converge."""
    D = data(1, [2.4])
    start = np.array([[0.53], [1.07], [1.6]])
    z, cert, r, hist = JL.solve_pass(D, 0.08, start, np.full(start.shape, 0.32), recentre=0)
    assert hist[-1]["converged"] is False
    z, cert, r, hist = JL.solve_pass(D, 0.08, start, np.full(start.shape, 0.32), recentre=6)
    assert hist[-1]["converged"] is True
