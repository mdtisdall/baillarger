"""Tests of ff_integrate.py and ff_checks.py: the plan's exact tests *Masked
interpolation* and *Integrator* (docs/plans/flow-field-phantoms.md), the stop bisection,
and each topology check on a small synthetic case with a known count.

The grid here is a minimal uniform grid that implements the `ff_types.Grid` protocol
(not ff_grid.py). Every tolerance here was fixed when the test was written, before any
run."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_checks as FC  # noqa: E402
import ff_integrate as FI  # noqa: E402
import ff_types as T  # noqa: E402

H_VOX = T.DEFAULTS.rk4_step_mm / T.VOXEL_MM  # one RK4 step in voxels


class Grid:
    """A uniform grid with nodes at (i, j, k) · h voxels, dims (ni, nj, nk) nodes."""

    kind: Literal["uniform", "octree"] = "uniform"

    def __init__(self, dims: tuple[int, int, int], h: float = 1.0) -> None:
        self.dims = dims
        self.h = h
        ni, nj, nk = dims
        self.cdims = (ni - 1, nj - 1, nk - 1)
        IJK = np.stack(np.meshgrid(*(np.arange(n) for n in dims), indexing="ij"), -1)
        self.n_nodes = ni * nj * nk
        self.nodes = IJK.reshape(-1, 3).astype(float) * h
        C = np.stack(
            np.meshgrid(*(np.arange(n) for n in self.cdims), indexing="ij"), -1
        )
        C = C.reshape(-1, 3)
        cells = np.zeros((len(C), 8), np.int64)
        for c in range(8):
            off = np.array([(c >> a) & 1 for a in range(3)])
            cells[:, c] = np.ravel_multi_index(tuple((C + off).T), dims)
        self.cells = cells
        self.cell_origin = C.astype(float) * h
        self.cell_size_mm = np.full(len(C), h * T.VOXEL_MM)
        self.prior_shaped = np.zeros(len(C), bool)
        self.node_level = np.zeros(self.n_nodes, np.int64)
        edges = []
        for a in range(3):
            sel = IJK.reshape(-1, 3)[:, a] < dims[a] - 1
            ids = np.flatnonzero(sel)
            stride = int(np.prod(dims[a + 1 :]))
            edges.append(np.stack([ids, ids + stride], 1))
        E = np.concatenate(edges)
        self.edges = E[np.lexsort((E[:, 1], E[:, 0]))]
        self.hanging = (
            np.zeros(0, np.int64),
            np.zeros((0, 4), np.int64),
            np.zeros((0, 4)),
        )
        self.colour = (IJK.reshape(-1, 3) % 2) @ np.array([1, 2, 4])
        self.n_colours = 8

    def locate(self, X: T.Array) -> tuple[T.IntArray, T.Array]:
        t = np.asarray(X, float).reshape(-1, 3) / self.h
        top = np.array(self.cdims, float)
        inside = np.all((t >= 0.0) & (t <= top), axis=1)
        c = np.clip(np.floor(t), 0, top - 1).astype(np.int64)
        local = np.where(inside[:, None], t - c, 0.0)
        cid = np.ravel_multi_index(tuple(c.T), self.cdims)
        return np.where(inside, cid, -1).astype(np.int64), local

    def morton(self) -> T.IntArray:
        return np.arange(len(self.cells), dtype=np.int64)

    def arrays(self) -> dict[str, Any]:
        return {}


def make_field(
    grid: Grid, direction: T.Array, tau: T.Array, q_stop: T.Array
) -> T.Field:
    """A field with one moving candidate per node (q = 1 − q_stop)."""
    n = grid.n_nodes
    d = np.asarray(direction, float).reshape(n, 3)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    return T.Field(
        grid=grid,
        tau=np.asarray(tau, float).reshape(n, 1),
        direction=d[:, None, :],
        q=(1.0 - np.asarray(q_stop, float)).reshape(n, 1),
        q_stop=np.asarray(q_stop, float).reshape(n),
        n_kept=np.ones(n, np.int64),
    )


def test_grid_is_a_grid() -> None:
    assert isinstance(Grid((3, 2, 2)), T.Grid)


# ---------------------------------------------------------------- masked interpolation


def two_cells(stopped_dir: T.Array, stopped_tau: float) -> T.Field:
    """Two cells along x (nodes x = 0, 1, 2). Nodes at x = 2 are stopped (q_stop 0.7)
    but keep a direction and τ of their own."""
    g = Grid((3, 2, 2))
    x = g.nodes[:, 0]
    d = np.where((x == 0)[:, None], [1.0, 0.0, 0.0], [1.0, 0.2, 0.0])
    d = np.where((x == 2)[:, None], stopped_dir, d)
    tau = np.where(x == 0, -2.0, np.where(x == 1, -1.0, stopped_tau))
    return make_field(g, d, tau, np.where(x == 2, 0.7, 0.1))


def test_masked_interpolation_takes_no_weight_from_a_stopped_node() -> None:
    X = np.array([[1.5, 0.3, 0.7], [1.9, 0.5, 0.5], [0.4, 0.2, 0.9], [1.0, 1.0, 1.0]])
    f1 = two_cells(np.array([0.0, 1.0, 0.0]), 99.0)
    f2 = two_cells(np.array([-1.0, -3.0, 2.0]), -1e6)
    D1, t1, p1, c1, u1 = FI.masked_sample(f1, X)
    D2, t2, p2, c2, u2 = FI.masked_sample(f2, X)
    # the values of the stopped nodes do not change the result, to the last bit
    for a, b in ((D1, D2), (t1, t2), (p1, p2), (c1, c2), (u1, u2)):
        np.testing.assert_array_equal(a, b)
    d1 = np.array([1.0, 0.2, 0.0]) / np.hypot(1.0, 0.2)
    np.testing.assert_allclose(D1[:2], np.tile(d1, (2, 1)), atol=1e-15)
    np.testing.assert_allclose(t1[:2], -1.0, atol=1e-15)
    # the stop probability: trilinear over all 8 corners
    np.testing.assert_allclose(
        p1[:2], [0.1 * 0.5 + 0.7 * 0.5, 0.1 * 0.1 + 0.7 * 0.9], atol=1e-15
    )
    # cell 0: the usual trilinear interpolation of the two moving layers
    np.testing.assert_allclose(t1[2], 0.6 * -2.0 + 0.4 * -1.0, atol=1e-15)
    assert list(c1) == [1, 1, 0, 1]
    assert not u1.any()


def three_cells_sheet() -> T.Field:
    """Three cells along x: nodes x = 0, 1 point +x, nodes x = 2, 3 point −x, all
    moving. A stopped sheet between x = 1 and x = 2 is not resolved: cell 1 is
    unresolved."""
    g = Grid((4, 2, 2))
    x = g.nodes[:, 0]
    d = np.where((x <= 1)[:, None], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    return make_field(g, d, np.where(x <= 1, x - 3.0, -x), np.full(g.n_nodes, 0.1))


def test_unresolved_contact_cell_ends_the_column() -> None:
    f = three_cells_sheet()
    assert list(np.flatnonzero(FI.unresolved_cells(f))) == [1]
    starts = np.array([[0.3, 0.5, 0.5], [2.7, 0.4, 0.6], [0.2, 0.1, 0.2]])
    cols, info = FI.integrate(f, starts, starts)
    assert list(cols.end) == [int(T.EndReason.UNRESOLVED_CONTACT)] * 3
    assert info["ends"]["UNRESOLVED_CONTACT"] == 3
    assert info["unresolved_cells_met"] == [1]
    assert info["n_unresolved_cells"] == 1
    for i, side in ((0, -1), (1, 1), (2, -1)):
        P = cols.paths[i, : cols.counts[i]]
        # no sample in the cell between the sheet's sides: no weight from the far side
        assert np.all(P[:, 0] < 1.0) if side < 0 else np.all(P[:, 0] > 2.0)
        # the column ended within one RK4 step of the cell
        assert abs(P[-1, 0] - (1.0 if side < 0 else 2.0)) < H_VOX


def test_converging_flow_is_not_a_contact() -> None:
    """A flow that converges on a focus line beyond the grid (the concave side of a
    fold) has no unresolved cell. Without the CONTACT_COS condition (the spec's rule),
    the band of cells that straddle the flow's plane of symmetry (y = 2.5) is
    unresolved."""
    g = Grid((5, 5, 5))
    d = np.array([12.0, 2.5, 0.0]) - g.nodes
    d[:, 2] = 0.0
    f = make_field(g, d, -np.ones(g.n_nodes), np.full(g.n_nodes, 0.1))
    assert not FI.unresolved_cells(f).any()
    spec = FI.unresolved_cells(f, max_cos=2.0)
    band = g.cell_origin[:, 1] == 2.0
    np.testing.assert_array_equal(spec, band)
    assert spec.sum() == 16


# ---------------------------------------------------------------- the integrator


def wavy_field() -> T.Field:
    """A smooth bent field with noise, a stopped region at x > ~8 and a sink-free
    interior."""
    g = Grid((12, 12, 12))
    X = g.nodes
    rng = np.random.default_rng(3)
    d = np.stack(
        [np.ones(len(X)), 0.3 * np.sin(X[:, 1] / 2), 0.2 * np.cos(X[:, 2] / 3)], 1
    )
    d = d + 0.05 * rng.standard_normal(d.shape)
    q_stop = T.sigmoid(2.0 * (X[:, 0] - 8.0 + 0.5 * np.sin(X[:, 1] / 2)))
    return make_field(g, d, X[:, 0] - 8.0, q_stop)


def test_same_input_same_columns_in_any_order() -> None:
    f = wavy_field()
    rng = np.random.default_rng(7)
    starts = np.c_[
        rng.uniform(1, 3, 60), rng.uniform(1, 10, 60), rng.uniform(1, 10, 60)
    ]
    roots = starts + [1.0, 0.0, 0.0]
    c1, i1 = FI.integrate(f, starts, roots)
    c2, i2 = FI.integrate(f, starts, roots)
    assert i1 == i2
    assert i1["ends"]["STOPPED"] > 50
    for perm in (np.arange(60)[::-1], rng.permutation(60)):
        c3, i3 = FI.integrate(f, starts[perm], roots[perm])
        assert i3 == i1
        for k in ("paths", "tau_path", "counts", "stops", "end", "tau_arrival", "jump"):
            a, b, c = getattr(c1, k), getattr(c2, k), getattr(c3, k)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a[perm], c)
    # a column does not depend on the other columns of the batch
    sub = np.arange(0, 60, 7)
    c4, _ = FI.integrate(f, starts[sub], roots[sub])
    m = int(c4.counts.max())
    np.testing.assert_array_equal(c1.paths[sub, :m], c4.paths)
    np.testing.assert_array_equal(c1.tau_path[sub, :m], c4.tau_path)
    np.testing.assert_array_equal(c1.end[sub], c4.end)


def test_rk4_on_a_constant_field_is_a_straight_line() -> None:
    g = Grid((11, 11, 11))
    d = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])
    f = make_field(
        g, np.tile(d, (g.n_nodes, 1)), np.zeros(g.n_nodes), np.full(g.n_nodes, 0.1)
    )
    starts = np.array([[1.0, 0.5, 2.0], [0.2, 0.3, 0.4], [3.3, 1.1, 7.7]])
    cols, _ = FI.integrate(f, starts, starts)
    assert (cols.end == int(T.EndReason.LEFT_BOX)).all()
    for i in range(3):
        P = cols.paths[i, : cols.counts[i]]
        k = np.arange(len(P))[:, None]
        np.testing.assert_allclose(P, starts[i] + k * H_VOX * d, rtol=0, atol=1e-9)
        assert cols.counts[i] > 50


@pytest.mark.parametrize("d", [[1.0, 0.0, 0.0], [1.0, 0.7, -0.4]])
def test_stop_lies_on_the_plane_to_six_bisections(d: list[float]) -> None:
    """q_stop is linear in x and crosses 0.5 at x0, so its trilinear interpolation
    crosses 0.5 on the plane x = x0 (to rounding)."""
    g = Grid((11, 11, 11))
    du = np.array(d) / np.linalg.norm(d)
    x0 = 5.37
    q_stop = 0.5 + 0.08 * (g.nodes[:, 0] - x0)
    f = make_field(g, np.tile(du, (g.n_nodes, 1)), g.nodes[:, 0] - x0, q_stop)
    starts = np.array([[1.0, 1.0, 5.0], [1.3, 1.9, 6.2], [2.0, 2.5, 7.1]])
    cols, info = FI.integrate(f, starts, starts)
    assert info["ends"]["STOPPED"] == 3
    res = H_VOX * du[0] / 2**T.DEFAULTS.stop_bisections  # one step / 2⁶, along x
    gap = x0 - cols.stops[:, 0]
    assert np.all(gap >= -1e-12) and np.all(gap <= res + 1e-12)
    # τ at the stop is the masked interpolation: only the moving corners (x = 5, τ = 5 −
    # x0)
    # enter, so τ is held at their value up to the stop
    np.testing.assert_allclose(cols.tau_arrival, 5.0 - x0, atol=1e-12)
    np.testing.assert_allclose(cols.jump, -cols.tau_arrival, atol=1e-15)
    # the stop is on the trajectory: the straight line
    for i in range(3):
        P = cols.paths[i, : cols.counts[i]]
        v = P - starts[i]
        np.testing.assert_allclose(v - (v @ du)[:, None] * du, 0.0, atol=1e-9)


def test_max_length_and_sink() -> None:
    g = Grid((11, 11, 11))
    d = np.tile([1.0, 0.0, 0.0], (g.n_nodes, 1))
    f = make_field(g, d, np.zeros(g.n_nodes), np.full(g.n_nodes, 0.1))
    s = T.SolverSettings(max_column_mm=0.5)
    cols, info = FI.integrate(f, np.array([[1.0, 1.0, 1.0]]), np.zeros((1, 3)), s)
    assert info["ends"]["MAX_LENGTH"] == 1 and cols.counts[0] == 51
    assert np.isnan(cols.jump[0])
    # the field reverses at x = 5: the column turns back, a sink
    d = np.where((g.nodes[:, 0] < 5)[:, None], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    f = make_field(g, d, np.zeros(g.n_nodes), np.full(g.n_nodes, 0.1))
    cols, info = FI.integrate(
        f, np.array([[1.0, 1.0, 1.0]]), np.zeros((1, 3)), max_cos=-2.0
    )  # no unresolved cells: test the sink rule alone
    assert info["ends"]["SINK"] == 1
    assert 4.0 < cols.stops[0, 0] < 5.0


# ---------------------------------------------------------------- the checks


def make_columns(
    paths: list[T.Array], roots: T.Array, end: list[int] | None = None
) -> T.Columns:
    n = len(paths)
    m = max(len(p) for p in paths)
    P = np.full((n, m, 3), np.nan)
    for i, p in enumerate(paths):
        P[i, : len(p)] = p
    counts = np.array([len(p) for p in paths], np.int64)
    nan = np.full(n, np.nan)
    return T.Columns(
        roots=np.asarray(roots, float),
        paths=P,
        counts=counts,
        tau_path=np.full((n, m), np.nan),
        stops=P[np.arange(n), counts - 1],
        end=np.array(end if end is not None else [0] * n, np.int8),
        tau_arrival=nan,
        jump=nan,
        p_jump=nan,
        stop_sd=nan,
        level_sd=np.zeros((n, 0)),
        levels=np.zeros(0),
    )


def line(a: list[float], b: list[float]) -> T.Array:
    """Samples from a to b (voxels) at the path step."""
    a_, b_ = np.array(a), np.array(b)
    L = np.linalg.norm(b_ - a_)
    k = np.arange(int(np.floor(L / H_VOX + 1e-9)) + 1)
    return np.asarray(a_ + (k * H_VOX / L)[:, None] * (b_ - a_))


class SlabGeometry:
    """Local mm = voxel × VOXEL_MM. The stopped region is the slab 5 ≤ x_vox ≤ 6."""

    shape = (11, 11, 11)
    lo, hi = 5 * T.VOXEL_MM, 6 * T.VOXEL_MM

    def to_local(self, X: T.Array) -> T.Array:
        return np.asarray(X, float) * T.VOXEL_MM

    def to_vox(self, Y: T.Array) -> T.Array:
        return np.asarray(Y, float) / T.VOXEL_MM

    def dir_to_vox(self, U: T.Array) -> T.Array:
        return U

    def label(self, Y: T.Array) -> Any:
        return np.where(self.stopped(Y), T.MEDIUM, T.CORTEX).astype(np.int8)

    def phi(self, Y: T.Array) -> T.Array:
        return np.asarray(Y[:, 0] / self.lo)

    def grad_phi(self, Y: T.Array) -> T.Array:
        return np.tile([1.0 / self.lo, 0.0, 0.0], (len(Y), 1))

    def stopped(self, Y: T.Array) -> T.BoolArray:
        return np.asarray((Y[:, 0] >= self.lo) & (Y[:, 0] <= self.hi))

    def stop_sdf(self, Y: T.Array) -> T.Array:
        return np.asarray(np.minimum(Y[:, 0] - self.lo, self.hi - Y[:, 0]))

    def bank(self, Y: T.Array) -> Any:
        return np.zeros(len(Y), np.int8)


def sheet_field() -> T.Field:
    """All +x, stopped node layer at x = 5 (q_stop 0.9), else 0.1."""
    g = Grid((11, 11, 11))
    x = g.nodes[:, 0]
    return make_field(
        g, np.tile([1.0, 0.0, 0.0], (g.n_nodes, 1)), x - 5.0, np.where(x == 5, 0.9, 0.1)
    )


def test_check_stop_crossings() -> None:
    f = sheet_field()
    paths = [
        line([1, 2, 2], [8, 2, 2]),  # through the sheet and back into tissue
        line([1, 4, 4], [4.4, 4, 4]),  # clean
        line([1, 6, 6], [4.6, 6, 6]),  # stops on the moving side: last sample p > 0.5
        line([1, 8, 8], [5.5, 8, 8]),
    ]  # overshoots the true stop (for the true check)
    cols = make_columns(paths, np.array([p[0] for p in paths]))
    r = FC.solved_stop_crossings(f, cols)
    assert r["count"] == 2 and list(r["columns"]) == [0, 3]
    rt = FC.true_stop_crossings(SlabGeometry(), cols)
    assert rt["count"] == 1 and list(rt["columns"]) == [0]
    assert rt["overshoot_count"] == 1 and list(rt["overshoot_columns"]) == [3]
    assert abs(rt["places"][0, 0] - 5.0) < 2 * FC.TRUE_STOP_TOL_MM / T.VOXEL_MM


def test_check_overlap_and_focal_point() -> None:
    # overlap: roots 5·√2 voxels (1.13 mm) apart; the paths cross at (5, 2, 2)
    a = line([0, 2, 2], [9, 2, 2])
    b = line([5, -3, 2], [5, 7, 2])
    r = FC.column_meetings(make_columns([a, b], np.array([a[0], b[0]])))
    assert r["overlap"]["count"] == 1 and r["focal"]["count"] == 0
    np.testing.assert_allclose(r["overlap"]["places"][0], [5, 2, 2], atol=0.05)
    # focal point: roots 2 voxels (0.32 mm) apart; the paths meet at (5, 2, 2)
    c = line([0, 4, 2], [5, 2, 2])
    c = np.r_[c, line([5, 2, 2], [9, 0.4, 2])[1:]]
    r = FC.column_meetings(make_columns([a, c], np.array([a[0], c[0]])))
    assert r["focal"]["count"] == 1 and r["overlap"]["count"] == 0
    # with the geometry: only meetings in the cortex count (the slab is outside medium)
    rg = FC.column_meetings(
        make_columns([a, b], np.array([a[0], b[0]])), SlabGeometry()
    )
    assert rg["overlap"]["count"] == 0 and rg["overlap"]["count_all"] == 1


def test_check_clean_case() -> None:
    f = sheet_field()
    # parallel columns, two voxels apart, that stop before the sheet (p < 0.5 for x <
    # 4.5)
    paths = [line([1, y, 2], [4.4, y, 2]) for y in (2, 4, 6, 8)]
    cols = make_columns(paths, np.array([p[0] for p in paths]))
    rep = FC.topology(f, cols, geometry=SlabGeometry())
    c = FC.counts(rep)
    assert set(c.values()) == {0}, c


def test_check_contact_is_not_overlap() -> None:
    """Two columns with roots 1.6 mm apart end head-on on a shared sheet (ends 0.02
    voxel apart): no overlap, because the samples near each stop are left out."""
    a = line([0, 6, 2], [4.99, 6, 2])
    b = line([10, 6, 2], [5.01, 6, 2])
    r = FC.column_meetings(make_columns([a, b], np.array([a[0], b[0]])))
    assert r["overlap"]["count"] == 0 and r["focal"]["count"] == 0
    # the same pair, run on past each other: an overlap
    a2 = line([0, 6, 2], [7, 6, 2.01])
    r = FC.column_meetings(make_columns([a2, b], np.array([a2[0], b[0]])))
    assert r["overlap"]["count"] == 1


def test_check_sinks() -> None:
    g = Grid((5, 5, 5))
    n = g.n_nodes
    centre = int(np.ravel_multi_index((2, 2, 2), g.dims))
    D = np.tile([1.0, 0.0, 0.0], (n, 1))
    # one node with two opposite candidates of equal probability: resultant 0
    q = np.zeros((n, 2))
    q[:, 0] = 0.9
    q[centre] = [0.45, 0.45]
    dirs = np.stack([D, D], 1)
    dirs[centre, 1] = [-1.0, 0.0, 0.0]
    tau = np.zeros((n, 2))
    f = T.Field(
        grid=g,
        tau=tau,
        direction=dirs,
        q=q,
        q_stop=np.full(n, 0.1),
        n_kept=np.full(n, 2, np.int64),
    )
    r = FC.field_sinks(f)
    assert r["low_resultant"]["count"] == 1 and list(r["low_resultant"]["nodes"]) == [
        centre
    ]
    assert r["reversed_edges"]["count"] == 0
    # one node reversed: 6 reversed edges, one converging (with the −x neighbour), one
    # diverging
    D2 = D.copy()
    D2[centre] = [-1.0, 0.0, 0.0]
    f2 = make_field(g, D2, np.zeros(n), np.full(n, 0.1))
    r2 = FC.field_sinks(f2)
    assert r2["low_resultant"]["count"] == 0
    e = r2["reversed_edges"]
    assert e["count"] == 6 and e["converging"] == 1 and e["diverging"] == 1
    # the cells that hold the centre and its −x neighbours are unresolved: the 4 on the
    # −x side
    assert int(FI.unresolved_cells(f2).sum()) == 4
    # the SINK columns
    a = line([0, 2, 2], [3, 2, 2])
    cols = make_columns(
        [a, a + [0, 2, 0]], np.zeros((2, 3)), end=[int(T.EndReason.SINK), 0]
    )
    assert FC.topology(f2, cols)["sink_columns"]["count"] == 1
