"""Tests of ff_sample.py: the draw of a field and the sampled columns. Synthetic fields
on a
small uniform grid (the minimal grid of test_ff_integrate.py); no site data. Every
tolerance here was fixed when the test was written."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_sample as FSA  # noqa: E402
import ff_types as T  # noqa: E402
from test_ff_integrate import Grid, make_field  # noqa: E402

H = T.VOXEL_MM
X_STOP = 8.0  # voxels: the stop plane
TAU_SCALE = X_STOP * H  # mm


class Slab:
    """The parts of a Geometry that ff_sample reads: φ = x_mm / TAU_SCALE (0 at x = 0,
    1 on the stop plane at x = X_STOP voxels), local mm = voxels · VOXEL_MM."""

    def to_local(self, X: T.Array) -> T.Array:
        return np.asarray(X, float) * H

    def phi(self, Y: T.Array) -> T.Array:
        return np.asarray(Y, float)[:, 0] / TAU_SCALE

    def grad_phi(self, Y: T.Array) -> T.Array:
        g = np.zeros((len(Y), 3))
        g[:, 0] = 1.0 / TAU_SCALE
        return g

    def stop_sdf(self, Y: T.Array) -> T.Array:
        return np.asarray(Y, float)[:, 0] - TAU_SCALE


def truth_of(roots: T.Array) -> T.Truth:
    n = len(roots)
    z = np.zeros(n)
    return T.Truth(
        phantom="synthetic", roots=roots, normals=np.tile([1.0, 0, 0], (n, 1)),
        starts=roots, paths=roots[:, None, :], counts=np.ones(n, np.int64), stops=roots,
        stop_kind=np.zeros(n, np.int8), bank=np.zeros(n, np.int8), tau_arrival=z,
        jump=z, offset=z, tau_scale=TAU_SCALE, step_mm=0.01, root_spacing_mm=0.1,
    )  # fmt: skip


def roots(n: int = 12) -> T.Array:
    rng = np.random.default_rng(1)
    r = np.zeros((n, 3))
    r[:, 0] = 0.5
    r[:, 1:] = 2.0 + 7.0 * rng.random((n, 2))
    return r


def field(onehot: bool, seed: int = 3) -> T.Field:
    """Flow along +x on 12^3 nodes, τ true, a stopped region at x > X_STOP. With onehot
    the stop probability is 0 or 1; else a sigmoid whose width makes the samples differ,
    and two slots at the transition."""
    g = Grid((12, 12, 12))
    X = g.nodes
    rng = np.random.default_rng(seed)
    d = np.concatenate(
        [np.ones((len(X), 1)), 0.05 * rng.standard_normal((len(X), 2))], 1
    )
    tau = X[:, 0] * H - TAU_SCALE
    q_stop = (
        (X[:, 0] > X_STOP).astype(float)
        if onehot
        else T.sigmoid(1.5 * (X[:, 0] - X_STOP))
    )
    return make_field(g, d, tau, q_stop)


def run(f: T.Field, n: int = 10, seed: int = 0) -> FSA.Samples:
    return FSA.sampled_columns(
        f,
        truth_of(roots()),
        T.DEFAULTS,
        n,
        seed,
        geo=Slab(),
        tau_scale=TAU_SCALE,  # type: ignore[arg-type]
    )


def test_one_hot_field_gives_the_same_field_every_draw() -> None:
    f = field(True)
    a = FSA.sample_field(f, np.random.default_rng(1))
    b = FSA.sample_field(f, np.random.default_rng(2))
    for k in ("tau", "direction", "q", "q_stop", "n_kept"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    np.testing.assert_array_equal(a.q_stop, f.q_stop)
    mov = f.q_stop == 0
    np.testing.assert_array_equal(a.tau[mov, 0], f.tau[mov, 0])


def test_one_hot_field_gives_zero_spread() -> None:
    s = run(field(True), n=10)
    cal = FSA.calibrate(s, min_n=8)
    assert (s.end == int(T.EndReason.STOPPED)).all()
    np.testing.assert_array_equal(cal.stop.sd, 0.0)
    assert cal.stop.zero_sd.all() and not np.isfinite(cal.stop.z).any()
    found = np.isfinite(s.level_dev)
    assert found.any()
    np.testing.assert_array_equal(cal.level.sd[cal.level.n >= 8], 0.0)
    assert (cal.level.n[cal.level.n > 0] == 10).all()


def test_draw_frequencies_match_q() -> None:
    n, K = 20000, 4
    q = np.array([0.2, 0.3, 0.1, 0.0])  # slot 3 is padding
    f = T.Field(
        grid=Grid((2, 2, 2)),  # type: ignore[arg-type]
        tau=np.tile([-1.0, -2.0, -3.0, np.nan], (n, 1)),
        direction=np.zeros((n, K, 3)) + np.array([1.0, 0, 0]),
        q=np.tile(q, (n, 1)),
        q_stop=np.full(n, 0.4),
        n_kept=np.full(n, 3, np.int64),
    )
    d = FSA.sample_field(f, np.random.default_rng(5))
    assert np.allclose(d.q.sum(1) + d.q_stop, 1.0)
    stopped = d.q_stop == 1.0
    assert not np.isnan(d.tau[~stopped]).any()
    for p, mask in zip(
        [0.2, 0.3, 0.1, 0.4],
        [d.tau[:, 0] == -1.0, d.tau[:, 0] == -2.0, d.tau[:, 0] == -3.0, stopped],
        strict=True,
    ):
        se = np.sqrt(p * (1 - p) / n)
        assert abs(mask.mean() - p) < 4 * se


def test_hanging_node_is_drawn_independently() -> None:
    """CD-10: a hanging node has its own free state, so its draw is its own."""
    g = Grid((2, 2, 2))
    g.hanging = (
        np.array([1]),
        np.array([[0, 2, 0, 0]]),
        np.array([[0.3, 0.7, 0.0, 0.0]]),
    )
    n = 3
    f = T.Field(
        grid=g,  # type: ignore[arg-type]
        tau=np.array([[-1.0], [-9.0], [-2.0]]),
        direction=np.tile([1.0, 0, 0], (n, 1, 1)),
        q=np.ones((n, 1)),
        q_stop=np.zeros(n),
        n_kept=np.ones(n, np.int64),
    )
    d = FSA.sample_field(f, np.random.default_rng(0))
    assert d.tau[1, 0] == -9.0


def test_same_seed_same_result_bit_for_bit() -> None:
    f = field(False)
    a, b, c = run(f, 6, 7), run(f, 6, 7), run(f, 6, 8)
    for k in ("stop_sdf", "level_dev", "end", "stops", "tau_arrival"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    assert not np.array_equal(a.stops, c.stops)
    x = FSA.sample_field(f, np.random.default_rng([7, 0]))
    y = FSA.sample_field(f, np.random.default_rng([7, 0]))
    np.testing.assert_array_equal(x.tau, y.tau)
    np.testing.assert_array_equal(x.q_stop, y.q_stop)


def test_noisy_field_has_spread_and_z() -> None:
    s = run(field(False), n=16)
    cal = FSA.calibrate(s)
    assert np.isfinite(cal.stop.z).any()
    assert (cal.stop.sd[np.isfinite(cal.stop.sd)] > 0).any()
    r = FSA.report(s, cal)
    assert r["stops"]["n_columns_used"] + r["stops"]["n_excluded_few"] + r["stops"][
        "n_excluded_zero_sd"
    ] == len(roots())
    cols = FSA.with_spread(s.mean_columns, s, cal)
    assert cols.level_sd.shape == (len(roots()), len(FSA.FR.LEVEL_PHI))


def test_moments_constant_values_have_zero_sd() -> None:
    x = np.full((30, 2), 0.1)
    x[:, 1] = np.nan
    x[:5, 1] = 1.0  # only 5 values
    m = FSA.moments(x, 8)
    assert m.sd[0] == 0.0 and m.zero_sd[0] and np.isnan(m.z[0])
    assert m.few[1] and np.isnan(m.sd[1])
