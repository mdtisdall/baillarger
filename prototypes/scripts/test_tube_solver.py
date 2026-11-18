"""Unit tests of the tube solver's parts on small synthetic voxel arrays
(exact answers where possible). Solver behaviour on data is judged on real
site crops, not here.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_tube_solver.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import tube_energy as TE  # noqa: E402
import tube_kernels as K  # noqa: E402
import tube_prep as TP  # noqa: E402

H = 0.16


def grid_problem(tau3: np.ndarray, members_owner: int = 0):
    """All voxels of tau3 (finite) are band voxels of one tube; one big
    local problem with every non-padded voxel as unknown."""
    shp = np.array(tau3.shape) + 2 * K.PAD
    strides = np.array([shp[1] * shp[2], shp[2], 1], np.int64)
    tau = np.pad(tau3, K.PAD, constant_values=np.nan).ravel()
    owner = np.where(np.isfinite(tau), members_owner, -1).astype(np.int64)
    inside = np.zeros(tau.size, bool)
    b = np.flatnonzero(owner >= 0)
    tvptr = np.array([0, len(b)], np.int64)
    sptr = np.array([0, 1], np.int64)
    sidx = np.array([0], np.int32)
    roots = np.tile(b[:1], (1, 8))
    rootw = np.full((1, 8), 1 / 8)
    out = K.build_problem(
        np.array([0]), tvptr, b, tau, owner, inside, np.inf, sptr, sidx, strides,
        K.offsets_l1(2, strides), -np.ones(tau.size, np.int64), roots, rootw,
    )
    sl, un, ev, nb, nbv, valid, rslot, rw, hasroot = out
    S = len(sl) + 1
    P = {
        "x0": np.r_[tau[sl], 0.0], "unk": un, "umask": np.ones(len(un)),
        "evalw": np.r_[ev, 0].astype(float), "nb": np.concatenate([nb, np.full((1, 3, 2), S - 1)]),
        "nbv": np.concatenate([nbv, np.zeros((1, 3, 2), bool)]), "valid": np.r_[valid, 0].astype(float), "inner": np.zeros(S),
        "c": np.zeros(S), "coh": np.zeros(S), "nhat": np.zeros((S, 3)), "wq": np.ones(S),
        "h": np.full(3, H), "htau": np.float64(H), "rslot": rslot, "rw": rw, "hasroot": hasroot * 0,
        "logmed": np.zeros(1), "rj": np.zeros((1, 4)), "sj": np.ones((1, 4)), "mj": np.zeros((1, 4)),
    }
    return P, sl, tau, strides


def interior(P, sl, shp_inner, strides, margin=2):
    """Slots whose voxel is at least margin voxels from the array faces."""
    sp = np.array(shp_inner) + 2 * K.PAD
    xyz = np.stack(np.unravel_index(sl, sp), 1) - K.PAD
    return np.all((xyz >= margin) & (xyz < np.array(shp_inner) - margin), 1)


def test_linear_tau_terms_vanish():
    x = np.indices((9, 9, 9))[2] * H - 1.0  # τ = z - 1 mm, |∇τ| = 1 along z
    P, sl, _, st = grid_problem(x.astype(float))
    P["coh"][:] = 1.0
    P["nhat"][:, 2] = 1.0
    xs = P["x0"]
    g0 = np.asarray(TE.g0_of(xs, P))
    vt = {k: np.asarray(v) for k, v in TE.voxel_terms(xs, P, g0).items()}
    m = np.r_[interior(P, sl, (9, 9, 9), st), False]
    g = np.asarray(TE.gradient(xs, P))
    assert np.allclose(g[m], [0, 0, 1], atol=1e-12)
    for k in ("align", "smooth", "equi", "pull"):
        assert np.abs(vt[k][m]).max() < 1e-7, k  # EPS = 1e-9 in |∇τ|²


def test_one_sided_at_faces():
    x = np.indices((5, 5, 5))[0] * H * 2.0
    P, sl, _, _ = grid_problem(x.astype(float))
    g = np.asarray(TE.gradient(P["x0"], P))[:-1]
    assert np.allclose(g[:, 0], 2.0) and np.allclose(g[:, 1:], 0.0)


def test_equivolume_sphere():
    n = 41
    c = (n - 1) / 2
    r = np.sqrt(((np.indices((n,) * 3) - c) ** 2).sum(0)) * H
    P, sl, _, st = grid_problem(r - 2.5)
    xs = P["x0"]
    g0 = np.asarray(TE.g0_of(xs, P))
    vt = np.asarray(TE.voxel_terms(xs, P, g0)["equi"])[:-1]
    sp = np.array((n,) * 3) + 2 * K.PAD
    xyz = np.stack(np.unravel_index(sl, sp), 1) - K.PAD
    rr = np.sqrt(((xyz - c) ** 2).sum(1)) * H
    sel = (rr > 2.5) & (rr < 3.0)
    f = np.clip(0.5 - (rr - 2.5) / H, 0, 1)
    sel &= f == 0  # beyond the top f = 0: the term vanishes there
    assert np.all(vt[sel] == 0)
    sel = (rr > 1.6) & (rr < 2.0)  # τ < -h/2: f = 1, div r̂ = 2 / r
    assert np.allclose(vt[sel], (2 / rr[sel]) ** 2, rtol=0.03)


def test_newton_decreases_energy():
    rng = np.random.default_rng(0)
    x = np.indices((7, 7, 9))[2] * H - 0.8 + rng.normal(0, 0.03, (7, 7, 9))
    P, sl, _, _ = grid_problem(x)
    P["c"][:] = -TE.LOG_ODDS
    P["coh"][:] = 0.5
    P["nhat"][:, 2] = 1.0
    PB = {k: np.asarray(v)[None] for k, v in P.items()}
    u, e0, e1, it, p0, p1 = (np.asarray(a) for a in TE.newton_batch(PB, iters=10))
    assert e1[0] < e0[0]
    assert np.isclose(p1[0].sum(), e1[0])


def test_simple_point_blocks_handle():
    """Closing a ring creates a handle: rejected, and the object is unchanged;
    a voxel that extends the open ring is accepted."""
    sp = np.array([7, 7, 7])
    st = np.array([49, 7, 1])
    inside = np.zeros(343, bool)
    ring = np.zeros((7, 7, 7), np.uint8)
    ring[1:6, 1:6, 3] = 1
    ring[2:5, 2:5, 3] = 0
    ring[3, 1, 3] = 0  # open the ring at one voxel
    before = ring.copy()
    v = np.array([np.array([3, 1, 3]) @ st])
    ok = K.apply_signs(ring, sp, v, np.array([1.0]), np.array([-1.0]), inside)
    assert not ok and np.array_equal(ring, before)
    v = np.array([np.array([1, 0, 3]) @ st])
    ok = K.apply_signs(ring, sp, v, np.array([1.0]), np.array([-1.0]), inside)
    assert ok and ring[1, 0, 3] == 1


def test_shell_lowest_extrapolation_wins():
    sp = np.array([10, 10, 10])
    st = np.array([100, 10, 1], np.int64)
    N = 1000
    tau = np.full(N, np.nan)
    owner = -np.ones(N, np.int64)
    a, b, y = 5 * 100 + 5 * 10 + 3, 5 * 100 + 5 * 10 + 5, 5 * 100 + 5 * 10 + 4
    tau[a], owner[a] = -0.05, 0
    tau[b], owner[b] = -0.10, 1
    allowed = np.zeros(N, bool)
    allowed[y] = True
    o26 = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
    K.rebuild_shell(tau, owner, np.zeros(N, bool), allowed, np.array([a, b]), o26 @ st,
                    np.linalg.norm(o26 * H, axis=1), 1e-3)
    assert owner[y] == 1 and np.isclose(tau[y], -0.10 + H)


def test_same_bank_and_colouring():
    ptr = np.array([0, 2, 3, 5], np.int64)
    idx = np.array([1, 2, 0, 0, 1], np.int32)
    assert K.same_bank(0, 2, ptr, idx) and K.same_bank(1, 1, ptr, idx) and not K.same_bank(1, 2, ptr, idx)
    rng = np.random.default_rng(1)
    n = 50
    pairs = {(int(i), int(j)) for i, j in rng.integers(0, n, (200, 2)) if i != j}
    pairs |= {(j, i) for i, j in pairs}
    pa = np.array(sorted(pairs))
    p = np.searchsorted(pa[:, 0], np.arange(n + 1))
    col = K.greedy_colour(p, pa[:, 1].copy(), np.arange(n))
    assert np.all(col[pa[:, 0]] != col[pa[:, 1]])


def test_fps_covers():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 5, (3000, 3))
    seeds = TP.fps(p, 0.5)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(p[seeds]).query(p)
    assert d.max() <= 0.5 + 1e-12
    ds, _ = cKDTree(p[seeds]).query(p[seeds], k=2)
    assert ds[:, 1].min() > 0.5 - 1e-12


@pytest.mark.parametrize("pad_ok", [True])
def test_offsets(pad_ok):
    st = np.array([100, 10, 1], np.int64)
    assert len(K.offsets_l1(2, st)) == 25 and len(K.offsets_l1(4, st)) == 129


def test_inner_voxels_carry_pull_only():
    rng = np.random.default_rng(3)
    x = np.indices((7, 7, 7))[2] * H * 1.3 - 1.0 + rng.normal(0, 0.02, (7, 7, 7))
    P, sl, _, _ = grid_problem(x)
    P["c"][:] = 2.0
    P["coh"][:] = 1.0
    P["nhat"][:, 0] = 1.0
    g0 = np.asarray(TE.g0_of(P["x0"], P))
    out = TE.voxel_terms(P["x0"], P, g0)
    P["inner"][:] = 1.0
    inn = TE.voxel_terms(P["x0"], P, g0)
    for k in ("stop", "align", "smooth", "equi"):
        assert np.all(np.asarray(inn[k]) == 0), k
    assert np.allclose(inn["pull"], out["pull"]) and np.asarray(out["pull"]).max() > 0


def test_ls_residuals_match_energy_where_f_is_constant():
    """Deep tissue (f = 1): Σ r² = E - stop - log t terms; gradients equal."""
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(4)
    x = np.indices((7, 7, 9))[2] * H * 1.2 - 3.0 + rng.normal(0, 0.03, (7, 7, 9))
    P, sl, _, _ = grid_problem(x)
    P["c"][:] = 0.0
    P["coh"][:] = 0.7
    P["nhat"][:, 0] = 0.6
    P["nhat"][:, 2] = 0.8
    P = {k: jnp.asarray(v) for k, v in P.items()}
    g0 = TE.g0_of(P["x0"], P)
    fw, tw = TE.irls_weights(P["x0"], P)
    assert float(jnp.min(fw[:-1])) == 1.0

    def e_ls(x):
        parts = TE.energy_parts(x, P, g0)
        return sum(parts[k] for k in ("align", "smooth", "equi", "pull"))

    rs = lambda x: jnp.sum(TE.ls_residuals(x, P, g0, fw, tw) ** 2)  # noqa: E731
    assert np.isclose(float(rs(P["x0"])), float(e_ls(P["x0"])), rtol=1e-8)  # EPS: |ĝ| < 1
    assert np.allclose(jax.grad(rs)(P["x0"]), jax.grad(e_ls)(P["x0"]), atol=1e-8)
