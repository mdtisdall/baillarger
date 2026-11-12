"""Tests of ff_meanfield: the plan's exact test "the free energy does not rise in any
sweep on the slab", a loose sanity check of the solved slab, the pruning, the
determinism, the carry-over of the re-centring and the option CD-13 (re-centring at
the edge).

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
import inspect
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as FG  # noqa: E402
import ff_integrate as FI  # noqa: E402
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


# ---------------------------------------------------------------- the profile lines


def test_pruned_slots_read_their_lines() -> None:
    """After the pruning, the kept slots of the line form read the same profiles as the
    direct form, compacted (bit for bit); the lines themselves are not changed."""
    s = T.DEFAULTS
    rng = np.random.default_rng(31)
    n, K = 200, s.n_tau * s.n_dir
    st = _random_state(n, K, rng)
    lines = rng.normal(size=(n, s.n_dir, FTM.N_LEVELS + s.n_tau - 1, 2))
    line, off = FTM.line_slots(s.n_tau, s.n_dir)
    st_line = replace(
        st,
        profiles=lines.copy(),
        prof_line=np.broadcast_to(line, (n, K)).copy(),
        prof_off=np.broadcast_to(off, (n, K)).copy(),
    )
    rows, slots = np.arange(n)[:, None], np.arange(K)[None, :]
    direct = FTM.candidate_profile_rows(st_line, rows, slots)
    ci0 = np.broadcast_to(np.arange(K, dtype=np.int16), (n, K)).copy()
    # prune compacts the arrays of its state in place: each form gets its own copies
    own = dict(
        cand_tau=st.cand_tau.copy(),
        cand_dir=st.cand_dir.copy(),
        valid=st.valid.copy(),
        q=st.q.copy(),
        q_stop=st.q_stop.copy(),
    )
    st_direct = replace(st, profiles=direct.copy(), **own)
    a, _, _ = MF.prune(st_line, ci0.copy(), s)
    b, _, _ = MF.prune(st_direct, ci0.copy(), s)
    assert a.profiles.shape == lines.shape and np.array_equal(a.profiles, lines)
    rows, slots = np.arange(n)[:, None], np.arange(a.K)[None, :]
    pa = FTM.candidate_profile_rows(a, rows, slots)
    pb = FTM.candidate_profile_rows(b, rows, slots)
    assert pa.tobytes() == pb.tobytes()


# ---------------------------------------------------------------- CD-13


TINY_SHAPE = (12, 12, 12)  # 7³ = 343 nodes at 0.32 mm
TINY_MAX_SWEEPS = 60  # the first pass ends by a stop rule, a repeat reaches the cap


def test_recentre_at_edge_is_off_by_default() -> None:
    """The option CD-13 is off by default: the plan's schedule, no repeats."""
    p = inspect.signature(MF.solve).parameters["recentre_at_edge"]
    assert p.default is False
    _, _, _, _, log = slab_run()
    assert log["run"]["recentre_at_edge"] is False
    assert log["edge_repeats"] == []


@functools.cache
def tiny_run_at_edge() -> dict[str, Any]:
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=TINY_SHAPE)
    grid = FG.UniformGrid(ph.shape, SPACING_MM)
    s = replace(T.DEFAULTS, max_sweeps=TINY_MAX_SWEEPS)
    _, _, log = MF.solve(
        grid, FO.oracle_inputs(ph), ph.image["noise_sd"], s, recentre_at_edge=True
    )
    return log


def test_recentre_at_edge_ends_without_edge_or_at_the_cap() -> None:
    """With the option on, each width ends with no final choice at the edge, or with
    max_sweeps sweeps used over its passes. The passes add up to the width's sweeps."""
    log = tiny_run_at_edge()
    assert log["run"]["recentre_at_edge"] is True
    reps = log["edge_repeats"]
    assert any(r["repeat"] > 0 for r in reps)  # the case does repeat
    for i, w in enumerate(T.DEFAULTS.widths):
        rw = [r for r in reps if r["width"] == float(w)]
        assert [r["repeat"] for r in rw] == list(range(len(rw)))
        assert sum(r["sweeps"] for r in rw) == log["sweeps"][i] <= TINY_MAX_SWEEPS
        assert [f for r in rw for f in r["free_energy"]] == log["free_energy"][i]
        for r in rw[:-1]:
            assert r["at_edge"] and r["next"] == "repeat"
        last = rw[-1]
        assert last["next"] == "stop"
        ec = last["edge_counts"]
        no_edge = ec["tau_at_kept_edge"] == 0 and ec["dir_at_ring"] == 0
        assert no_edge == (not last["at_edge"])
        assert no_edge or last["sweeps_of_width"] == TINY_MAX_SWEEPS
    fin = {k: v for k, v in log["pruning"]["final"].items() if k != "kept"}
    assert fin == reps[-1]["edge_counts"]


def test_recentre_at_edge_free_energy_does_not_rise_in_a_pass() -> None:
    """With the option on, F does not rise in any sweep of a pass (the plan's rise
    rule; after the first sweep the reference is F after the pruning)."""
    log = tiny_run_at_edge()
    assert log["free_energy_rises"] == []
    assert log["defects"] == []
    for r in log["edge_repeats"]:
        fr = r["free_energy"]
        assert len(fr) == r["sweeps"] >= 1
        seq = [r["pruning"]["free_energy_after"], *fr[1:]]
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            assert b - a <= RISE_REL * abs(a), (r["width"], r["repeat"], a, b)
    json.dumps(log, allow_nan=True)  # JSON-able for run.json


# ---------------------------------------------------------------- the origin
#
# The plan's exact test *Origin* (docs/plans/flow-field-phantoms.md, *Exact tests*):
# "with an anchor column, moving the anchor changes every τ by one constant (to 1e-6)
# and nothing else. With the top as the origin, every complete column's stop is at
# τ = 0 (to one Δτ)." The tolerances below are the plan's, fixed before the first run.

ORIGIN_ABS = 1e-6  # the plan's: τ after one constant, q_stop, directions, speeds
ORIGIN_SPACING_MM = 0.48  # the anchor runs: 9³ = 729 nodes on the 24³ slab box
RUN_SHAPE = (44, 20, 20)  # holds the start points 2 mm behind the white surface


def _anchor_pair(
    grid: FG.UniformGrid, lo: np.ndarray, tau0: np.ndarray
) -> tuple[int, int]:
    """Two moving nodes whose initial τ differ by a non-zero exact multiple of Δτ (to
    1e-9 Δτ): of all such pairs, the pair nearest the box centre (the smallest sum of
    the two distances; ties: the lower node ids)."""
    moving = np.flatnonzero(lo < 0.0)
    k = (tau0[moving][:, None] - tau0[moving][None, :]) / T.DEFAULTS.dtau
    ok = (np.abs(k - np.rint(k)) < 1e-9) & (np.rint(k) != 0)
    r = np.linalg.norm(grid.nodes[moving] - grid.nodes.mean(0), axis=1)
    cost = np.where(ok, r[:, None] + r[None, :], np.inf)
    ia, ib = np.unravel_index(int(np.argmin(cost)), cost.shape)
    assert np.isfinite(cost[ia, ib])
    a, b = sorted((int(moving[ia]), int(moving[ib])))
    return a, b


ANCHOR_XFAIL = (
    "Finding (coordinator decision CD-12). Measured on this case: the far "
    "field moves by the initial offset (C = -12 Δτ), but near the anchors the "
    "solved field differs by more than 1e-6: max |Δτ - C| 0.066 (on) and 0.063 "
    "(off), q_stop 0.014 (on) and 0.16 (off), speed 0.056 (on) and 0.10 (off). The "
    "anchor is a point mass at one node, so moving it changes the conditioning of "
    "the discrete problem, not only its gauge."
)


@pytest.mark.xfail(strict=True, reason=ANCHOR_XFAIL)
@pytest.mark.parametrize("contrast", ["on", "off"])
def test_origin_moving_the_anchor_shifts_tau_only(contrast: str) -> None:
    """Origin (i). Contrast "off": noise_sd = inf gives the contrast term the weight 0
    exactly (1 / σ_e² = 0). Then the energy depends on τ only by differences. Contrast
    "on": the profiles are on the absolute levels ℓ Δτ, so the two anchors are chosen
    with initial τ that differ by an exact multiple of Δτ (`_anchor_pair`)."""
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=SHAPE)
    grid = FG.UniformGrid(ph.shape, ORIGIN_SPACING_MM)
    inputs = FO.oracle_inputs(ph)
    noise = ph.image["noise_sd"] if contrast == "on" else [math.inf, math.inf]
    ga = MF.grid_arrays(grid)
    lo = np.asarray(inputs.stop.log_odds(grid.nodes), np.float64)
    tdir, _, _ = MF.tensor_at_nodes(inputs.tensor, grid.nodes)
    tau0, _, _ = MF.initial_field(ga, lo, tdir)
    a, b = _anchor_pair(grid, lo, tau0)
    k = (tau0[a] - tau0[b]) / T.DEFAULTS.dtau
    assert a != b and abs(k - round(k)) < 1e-9 and round(k) != 0
    runs = [
        MF.solve(grid, inputs, noise, origin="anchor", anchor_node=i) for i in (a, b)
    ]
    (f1, e1, _), (f2, e2, _) = runs
    for f, i in ((f1, a), (f2, b)):  # the anchor: moving, τ a point mass at 0
        assert f.q_stop[i] == 0.0
        assert np.all(f.tau[i, : f.n_kept[i]] == 0.0)
    both = (f1.p_moving > 0) & (f2.p_moving > 0)
    dt = f2.tau_mean()[both] - f1.tau_mean()[both]
    C = float(np.median(dt))
    err = dict(
        tau=float(np.abs(dt - C).max()),
        q_stop=float(np.abs(f1.q_stop - f2.q_stop).max()),
        direction=float(np.abs(f1.direction_mean() - f2.direction_mean()).max()),
        speed=float(np.abs(e1["speed"] - e2["speed"]).max()),
    )
    assert all(v <= ORIGIN_ABS for v in err.values()), (C, err)


TOP_XFAIL = (
    "Finding (coordinator decision CD-12). Measured on this case (13 STOPPED "
    "columns, grid 0.32 mm): |τ_arrival| from 0.052 to 0.24, all above Δτ = 0.04. "
    "τ_arrival is the masked interpolation of τ̄ over the moving corners only, and "
    "a moving node can lie up to one spacing (0.32 mm, τ about -0.32 at speed 1) "
    "before the stop without a jump (spec item 8)."
)


@pytest.mark.xfail(strict=True, reason=TOP_XFAIL)
def test_origin_top_every_complete_column_stops_at_zero() -> None:
    """Origin (ii): with the top as the origin, the integrated columns from the slab's
    start points (2 mm behind the white surface) that end STOPPED have |τ_arrival| ≤
    Δτ."""
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=RUN_SHAPE)
    geo, ts = FTR.geometry_of("slab", RUN_SHAPE)
    truth = FTR.truth(geo, ts, "slab", root_spacing_mm=0.48, margin_mm=0.0)
    assert truth.n >= 4
    grid = FG.UniformGrid(ph.shape, SPACING_MM)
    field, _, log = MF.solve(grid, FO.oracle_inputs(ph), ph.image["noise_sd"])
    assert log["run"]["origin"] == "top" and log["run"]["anchor_node"] is None
    cols, _ = FI.integrate(field, truth.starts, truth.roots)
    stopped = cols.end == T.EndReason.STOPPED
    assert stopped.sum() >= 4
    assert np.all(np.abs(cols.tau_arrival[stopped]) <= T.DEFAULTS.dtau), np.abs(
        cols.tau_arrival[stopped]
    ).max()


def test_anchor_origin_contract() -> None:
    """The anchor origin: the arguments are checked; the anchor is moving with τ = 0;
    τ may be positive; the jump term is 0 and p_jump is nan."""
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=SHAPE)
    grid = FG.UniformGrid(ph.shape, ORIGIN_SPACING_MM)
    inputs = FO.oracle_inputs(ph)
    nsd = ph.image["noise_sd"]
    lo = np.asarray(inputs.stop.log_odds(grid.nodes), np.float64)
    s = replace(T.DEFAULTS, max_sweeps=3)
    with pytest.raises(ValueError):
        MF.solve(grid, inputs, nsd, s, origin="anchor")
    with pytest.raises(ValueError):
        MF.solve(grid, inputs, nsd, s, anchor_node=0)
    with pytest.raises(ValueError):
        MF.solve(grid, inputs, nsd, s, origin="bottom")
    with pytest.raises(ValueError):  # an anchor in the stopped region
        MF.solve(grid, inputs, nsd, s, origin="anchor", anchor_node=int(np.argmax(lo)))
    a = int(np.argmin(lo))
    f, e, log = MF.solve(grid, inputs, nsd, s, origin="anchor", anchor_node=a)
    assert log["run"]["origin"] == "anchor" and log["run"]["anchor_node"] == a
    assert log["run"]["anchor_position_vox"] == grid.nodes[a].tolist()
    assert f.q_stop[a] == 0.0 and math.isclose(float(f.q[a].sum()), 1.0, rel_tol=1e-12)
    assert np.all(f.tau[a, : f.n_kept[a]] == 0.0)
    assert np.nanmax(f.tau) > 0.0  # no upper bound away from the anchor
    assert all(t["jump"] == 0.0 for t in log["terms"])
    assert np.all(np.isnan(e["p_jump"]))


# ---------------------------------------------------------------- the R2 change: the
# jump at the stop crossing (solve(jump_at="crossing"), integrate(jump_at="crossing"))


@functools.cache
def _crossing_run() -> tuple[Any, ...]:
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=RUN_SHAPE)
    geo, ts = FTR.geometry_of("slab", RUN_SHAPE)
    truth = FTR.truth(geo, ts, "slab", root_spacing_mm=0.48, margin_mm=0.0)
    grid = FG.UniformGrid(ph.shape, SPACING_MM)
    inputs = FO.oracle_inputs(ph)
    field, extras, log = MF.solve(
        grid, inputs, ph.image["noise_sd"], jump_at="crossing"
    )
    cols, _ = FI.integrate(
        field, truth.starts, truth.roots, jump_at="crossing", u=extras["u"]
    )
    return grid, inputs, truth, field, extras, log, cols


def test_crossing_every_complete_column_stops_at_zero() -> None:
    """Origin (ii) with the R2 change: on the small slab (oracle, noise 0, 0.32 mm),
    the columns that end STOPPED have |τ_arrival| ≤ Δτ, and their jumps (all on
    intact cortex) are 0 to within Δτ. Tolerance Δτ, fixed before the first run."""
    _, _, truth, _, _, log, cols = _crossing_run()
    assert log["run"]["jump_at"] == "crossing"
    stopped = cols.end == T.EndReason.STOPPED
    assert stopped.sum() >= 4
    ta = np.abs(cols.tau_arrival[stopped])
    assert np.all(ta <= T.DEFAULTS.dtau), (ta.max(), np.median(ta))
    assert np.all(cols.jump[stopped] <= T.DEFAULTS.dtau), cols.jump[stopped].max()


def test_crossing_edge_jumps_on_intact_cortex() -> None:
    """With the R2 change, the jump of the solved field on each edge from a moving node
    to a stopped node (the posterior-mean τ, direction and slowness, measured at the
    stop crossing) is 0 to within Δτ on the small slab (all intact). Tolerance Δτ,
    fixed before the first run."""
    grid, inputs, _, field, extras, _, _ = _crossing_run()
    lo = np.asarray(inputs.stop.log_odds(grid.nodes), np.float64)
    mov = field.q_stop < T.DEFAULTS.stop_level
    tm = np.nan_to_num(field.tau_mean())
    dm = field.direction_mean()
    ga = MF.grid_arrays(grid, T.DEFAULTS, False)
    i, j = ga.edges[:, 0], ga.edges[:, 1]
    J = []
    for a, b, sgn in ((i, j, 1.0), (j, i, -1.0)):
        m = mov[a] & ~mov[b]
        frac = FTM.stop_crossing(lo[a][m], lo[b][m])
        _, Jm = FTM.jump_energy(
            tm[a][m], dm[a][m], np.exp(-extras["u"][a][m]),
            sgn * ga.edge_vec_mm[m], T.DEFAULTS, frac,
        )  # fmt: skip
        J.append(Jm)
    Jall = np.concatenate(J)
    assert len(Jall) > 0
    assert np.all(Jall <= T.DEFAULTS.dtau), (Jall.max(), np.median(Jall), len(Jall))
