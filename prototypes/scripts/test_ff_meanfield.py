"""Tests of ff_meanfield: the plan's exact test "the free energy does not rise in any
sweep on the slab", a loose sanity check of the solved slab, the pruning, the
determinism and the carry-over of the re-centring.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_meanfield.py

The slab run uses a box of 24³ voxels and a grid spacing of 0.32 mm (13³ = 2197 nodes)
so that the whole file runs in well under 60 s on a laptop CPU in float64. At 0.32 mm
the cortex (2.4 mm) still has more than 7 nodes across, and the box holds WM, cortex
and the outside medium. The full box (64 × 64 × 48 voxels) at 0.16 mm has about
2·10⁵ nodes, which is a run, not a test.

Every tolerance here was fixed when the test was written.
"""

from __future__ import annotations

import functools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as FG  # noqa: E402
import ff_meanfield as MF  # noqa: E402
import ff_oracle as FO  # noqa: E402
import ff_terms as FTM  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402

FEATS = {
    e: dict(
        profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))),
        wm_level=-0.3,
        fluid_level=-2.0,
        hp_stat=3.0,
    )
    for e in ("echo2", "echo3")
}  # synthetic features (CI has no site data); hp_stat > 0 gives a positive noise SD
SHAPE = (24, 24, 24)
SPACING_MM = 0.32
RISE_REL = 1e-9  # the float64 rounding allowance of the plan's test
AWAY_MM = 0.2  # the sanity check uses nodes this far from the true stop surface
MEDIAN_ANGLE_DEG = 5.0  # loose: the posterior-mean direction in the cortex
MAX_ANGLE_DEG = 30.0

Solved = tuple[T.Phantom, FG.UniformGrid, T.Field, dict[str, Any], dict[str, Any]]


@functools.cache
def slab_run() -> Solved:
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=SHAPE)
    grid = FG.UniformGrid(ph.shape, SPACING_MM)
    field, extras, log = MF.solve(grid, FO.oracle_inputs(ph), ph.image["noise_sd"])
    return ph, grid, field, extras, log


# ---------------------------------------------------------------- the free energy


def test_free_energy_does_not_rise_on_the_slab() -> None:
    _, _, _, _, log = slab_run()
    assert log["free_energy_rises"] == []
    assert log["defects"] == []
    assert len(log["free_energy"]) == len(T.DEFAULTS.widths)
    for fl, pr, ns in zip(
        log["free_energy"], log["pruning"]["per_width"], log["sweeps"], strict=True
    ):
        assert len(fl) == ns >= 1
        # after the first sweep the reference is F after the pruning
        seq = [pr["free_energy_after"], *fl[1:]]
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            assert b - a <= RISE_REL * abs(a), (a, b)
    json.dumps(log, allow_nan=True)  # JSON-able for run.json


# ---------------------------------------------------------------- sanity of the slab


def test_slab_states_and_directions() -> None:
    ph, grid, field, extras, _ = slab_run()
    geo = ph.geometry
    assert geo is not None
    Y = geo.to_local(grid.nodes)
    sdf = geo.stop_sdf(Y)
    phi = geo.phi(Y)
    medium = sdf > AWAY_MM
    cortex = (sdf < -AWAY_MM) & (phi > 0.0) & (phi < 1.0)
    assert medium.sum() > 20 and cortex.sum() > 100
    assert np.all(field.q_stop[medium] > 0.5)
    assert np.all(field.q_stop[cortex] < 0.5)
    d = field.direction_mean()[cortex]
    t = T.true_direction(geo, grid.nodes[cortex])
    ang = np.degrees(np.arccos(np.clip(np.sum(d * t, 1), -1.0, 1.0)))
    assert np.median(ang) < MEDIAN_ANGLE_DEG
    assert ang.max() < MAX_ANGLE_DEG
    for k in ("u", "speed", "p_jump"):
        assert extras[k].shape == (grid.n_nodes,)
        assert np.all(np.isfinite(extras[k]))


def test_field_layout() -> None:
    _, grid, field, extras, _ = slab_run()
    K = T.DEFAULTS.prune_max
    assert field.tau.shape == (grid.n_nodes, K)
    assert np.allclose(field.q_stop + field.q.sum(1), 1.0, atol=1e-12)
    slot = np.arange(K)[None, :]
    kept = slot < field.n_kept[:, None]
    assert np.all(np.isfinite(field.tau[kept]))
    assert np.all(np.isnan(field.tau[~kept]))
    assert np.all(field.direction[~kept] == 0.0)
    assert np.all(field.q[~kept] == 0.0)
    assert np.allclose(np.linalg.norm(field.direction[kept], axis=1), 1.0)
    assert np.all(field.tau[kept] <= T.TAU_TOP)
    assert np.all((extras["cand_index"] >= 0) == kept)


# ---------------------------------------------------------------- pruning


def test_pruning_counts_in_the_run() -> None:
    _, _, field, _, log = slab_run()
    s = T.DEFAULTS
    for pr in log["pruning"]["per_width"]:
        assert s.prune_min <= pr["kept"]["min"] <= pr["kept"]["max"] <= s.prune_max
    assert np.all((field.n_kept >= s.prune_min) & (field.n_kept <= s.prune_max))
    fin = log["pruning"]["final"]
    for k in ("tau_at_kept_edge", "tau_at_set_edge", "dir_at_ring", "tau_single"):
        assert 0 <= fin[k] <= fin["moving_nodes"]


def _random_state(n: int, K: int, rng: np.random.Generator) -> FTM.TermState:
    valid = rng.random((n, K)) < 0.8
    valid[:, :5] = True
    # mixed shapes: peaked, flat and nearly empty moving mass
    logits = rng.normal(0.0, 1.0, (n, K)) * rng.choice([0.1, 3.0, 20.0], (n, 1))
    q = np.where(valid, np.exp(logits), 0.0)
    q /= q.sum(1, keepdims=True)
    q_stop = rng.choice(np.array([0.0, 0.3, 0.999999]), n)
    q *= (1.0 - q_stop)[:, None]
    d = rng.normal(size=(n, K, 3))
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return FTM.TermState(
        cand_tau=-rng.random((n, K)),
        cand_dir=d,
        valid=valid,
        q=q,
        q_stop=q_stop,
        u=np.zeros(n),
        s_ref=np.ones(n),
        profiles=rng.normal(size=(n, K, FTM.N_LEVELS, 2)),
        level_base=np.zeros(n, np.int64),
        stop_log_odds=np.zeros(n),
        tensor_dir=np.tile([1.0, 0.0, 0.0], (n, 1)),
        coherence=np.ones(n),
    )


def test_prune_keeps_the_mass_rule() -> None:
    s = T.DEFAULTS
    rng = np.random.default_rng(5)
    n, K = 300, 63
    st = _random_state(n, K, rng)
    q0, qs0, prof0 = st.q.copy(), st.q_stop.copy(), st.profiles.copy()
    ci0 = np.broadcast_to(np.arange(K, dtype=np.int16), (n, K)).copy()
    res = MF.prune_choice(st.q, st.valid, s)
    nk = res.n_keep
    assert np.all((nk >= s.prune_min) & (nk <= s.prune_max))
    for i in range(n):
        kept = res.order[i, : nk[i]]
        assert np.all(st.valid[i, kept])
        p = q0[i].sum()
        mass = q0[i, kept].sum()
        # the kept ones are the largest
        rest = np.setdiff1d(np.flatnonzero(st.valid[i]), kept)
        if len(rest):
            assert q0[i, kept].min() >= q0[i, rest].max()
        if nk[i] < s.prune_max and p > 0:
            assert mass >= s.prune_mass * p * (1 - 1e-12)
        if nk[i] > s.prune_min:
            # one fewer would not hold the mass
            smaller = np.sort(q0[i, kept])[::-1][: nk[i] - 1].sum()
            assert smaller < s.prune_mass * p
    new, ci, res2 = MF.prune(st, ci0, s)
    assert new.K == s.prune_max
    assert np.array_equal(res2.n_keep, nk)
    assert np.allclose(new.q_stop + new.q.sum(1), 1.0, atol=1e-12)
    for i in range(n):
        kept = np.sort(res.order[i, : nk[i]])
        assert np.array_equal(ci[i, : nk[i]], kept)
        assert np.all(ci[i, nk[i] :] == -1)
        assert np.all(new.valid[i, : nk[i]]) and not np.any(new.valid[i, nk[i] :])
        z = qs0[i] + q0[i, kept].sum()
        assert np.allclose(new.q[i, : nk[i]], q0[i, kept] / z, rtol=1e-12, atol=0)
        assert math.isclose(new.q_stop[i], qs0[i] / z, rel_tol=1e-12, abs_tol=1e-300)
        assert np.array_equal(new.profiles[i, : nk[i]], prof0[i, kept])


# ---------------------------------------------------------------- determinism


def test_two_solves_give_identical_bits() -> None:
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=SHAPE)
    grid = FG.UniformGrid(ph.shape, SPACING_MM)
    s = replace(T.DEFAULTS, max_sweeps=3)
    out = [MF.solve(grid, FO.oracle_inputs(ph), ph.image["noise_sd"], s) for _ in "ab"]
    (f1, e1, l1), (f2, e2, l2) = out
    for k in ("tau", "direction", "q", "q_stop", "n_kept"):
        assert getattr(f1, k).tobytes() == getattr(f2, k).tobytes(), k
    for k in e1:
        assert e1[k].tobytes() == e2[k].tobytes(), k
    assert l1["free_energy"] == l2["free_energy"]


# ---------------------------------------------------------------- carry-over


def test_carry_over_moves_a_point_mass_to_the_nearest_candidate() -> None:
    s = T.DEFAULTS
    p = np.array([[0.3, -0.5, 0.81]])
    p /= np.linalg.norm(p)
    tau_pred = np.array([-0.5])
    new_tau, new_dir, new_valid = FTM.moving_candidates(tau_pred, p, s)
    target = 6 * s.n_dir + 3  # τ index 6, ring direction 3
    # the point mass: 0.3 Δτ from the target's τ, 2° from its direction
    t0 = new_tau[0, target] + 0.3 * s.dtau
    d_t = new_dir[0, target]
    ax = np.cross(d_t, [1.0, 0.0, 0.0])
    ax /= np.linalg.norm(ax)
    th = math.radians(2.0)
    d0 = d_t * math.cos(th) + np.cross(ax, d_t) * math.sin(th)
    mass = 0.7
    q = MF.carry_over(
        np.array([[t0]]),
        d0[None, None, :],
        np.array([[mass]]),
        np.ones((1, 1), bool),
        new_tau,
        new_dir,
        new_valid,
        s.dtau,
        s.dir_cone_deg,
    )
    assert int(np.argmax(q[0])) == target
    assert math.isclose(q.sum(), mass, rel_tol=1e-12)
    # near the top: no mass on candidates that are not states
    tau_top = np.array([-0.05])
    nt, nd, nv = FTM.moving_candidates(tau_top, p, s)
    q2 = MF.carry_over(
        np.array([[0.0]]),
        p[:, None, :],
        np.array([[mass]]),
        np.ones((1, 1), bool),
        nt,
        nd,
        nv,
        s.dtau,
        s.dir_cone_deg,
    )
    assert np.all(q2[~nv] == 0.0)
    assert math.isclose(q2.sum(), mass, rel_tol=1e-12)
    top = int(np.argmax(np.where(nv, nt, -np.inf)[0]))  # the largest valid τ, m = 0
    assert int(np.argmax(q2[0])) == top


# ---------------------------------------------------------------- the inputs


class _HoleTensor:
    """A tensor with no direction at the first two points (as the oracle tensor on
    the axis of a crown)."""

    source: T.InputSource = "oracle"

    def at(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.tile([0.0, 0.0, 2.0], (len(X), 1))
        d[0] = np.nan
        d[1] = 0.0
        return d, np.ones(len(X))


def test_tensor_without_a_direction_gets_zero_coherence() -> None:
    X = np.zeros((4, 3))
    d, c, bad = MF.tensor_at_nodes(_HoleTensor(), X)
    assert bad == 2
    assert np.array_equal(c, [0.0, 0.0, 1.0, 1.0])
    assert np.array_equal(d[:2], [MF.NO_DIRECTION] * 2)
    assert np.array_equal(d[2:], [[0.0, 0.0, 1.0]] * 2)


def test_initial_from_a_field_on_another_grid() -> None:
    coarse = FG.UniformGrid((9, 9, 9), 0.64)  # 3³ nodes
    fine = FG.UniformGrid((9, 9, 9), 0.32)  # 5³ nodes
    n = coarse.n_nodes
    K = T.DEFAULTS.prune_max
    d0 = np.array([0.0, 0.6, 0.8])
    tau = np.full((n, K), np.nan)
    tau[:, :2] = [-0.3, -0.5]
    direction = np.zeros((n, K, 3))
    direction[:, :2] = d0
    q = np.zeros((n, K))
    q[:, :2] = [0.3, 0.3]  # τ̄ = −0.4 where moving
    q_stop = np.full(n, 0.4)
    q_stop[0] = 1.0  # one node stopped: no weight in τ and d
    q[0] = 0.0
    f = T.Field(coarse, tau, direction, q, q_stop, np.full(n, 2, np.int64))
    t, d, qs, has = MF.initial_from_field(f, fine.nodes)
    # fine node 0 is on the stopped coarse node 0: no moving weight there
    assert not has[0] and np.all(has[1:])
    assert np.allclose(t[1:], -0.4, atol=1e-12)
    assert np.allclose(d[1:], d0, atol=1e-12)
    assert math.isclose(qs[0], 1.0) and np.all((qs >= 0.4 - 1e-12) & (qs <= 1.0))


# ---------------------------------------------------------------- chunks


def test_chunked_steps_give_the_same_bits() -> None:
    """The node chunks of the colour update, the pruning rule, the posterior means and
    the entropy give the result of one chunk (the colour update and the rule bit for
    bit; the entropy to rounding)."""
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=(12, 12, 12))
    grid = FG.UniformGrid(ph.shape, SPACING_MM)
    s = T.DEFAULTS
    ga = MF.grid_arrays(grid, s)
    rng = np.random.default_rng(23)
    n, K = grid.n_nodes, 20
    st0 = _random_state(n, K, rng)
    st0.cand_tau[:] = -rng.random((n, K)) * 2.0
    st0.u[:] = rng.normal(0.0, 0.2, n)
    tw = FTM.term_weights(SPACING_MM, [1.0, 1.0], 2, s)
    sel = np.flatnonzero(np.asarray(grid.colour) == 3)
    out = []
    for chunk in (10**6, 5):
        st = replace(st0, q=st0.q.copy(), q_stop=st0.q_stop.copy())
        mom = FTM.node_moments(st)
        dq = MF._update_colour(st, ga, tw, sel, mom, chunk=chunk)
        out.append((dq, st.q.tobytes(), st.q_stop.tobytes(), mom.m2.tobytes()))
    assert out[0] == out[1]
    r1 = MF.prune_choice(st0.q, st0.valid, s, chunk=10**6)
    r2 = MF.prune_choice(st0.q, st0.valid, s, k_out=s.prune_max, chunk=7)
    assert np.array_equal(r1.order[:, : s.prune_max], r2.order)
    assert np.array_equal(r1.n_keep, r2.n_keep)
    assert r1.kept_mass.tobytes() == r2.kept_mass.tobytes()
    for a, b in zip(
        MF.posterior_means(st0, chunk=10**6),
        MF.posterior_means(st0, chunk=3),
        strict=True,
    ):
        assert a.tobytes() == b.tobytes()
    assert math.isclose(
        MF.entropy(st0, chunk=10**6), MF.entropy(st0, chunk=3), rel_tol=1e-13
    )
    ci = np.where(st0.valid, np.arange(K, dtype=np.int16)[None, :], -1).astype(np.int16)
    ec = MF.edge_counts(st0, ci, s, chunk=10**6)
    assert ec == MF.edge_counts(st0, ci, s, chunk=7)
    assert ec["moving_nodes"] > 0
