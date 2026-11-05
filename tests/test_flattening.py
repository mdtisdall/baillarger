"""Tests of baillarger.flattening with exact answers, on the phantoms of
flattening_phantoms."""

import numpy as np
import numpy.typing as npt
import pytest
from flattening_phantoms import (
    ALPHA,
    STEP,
    TOP,
    Interval,
    Plane,
    build_rays,
    grid_edges,
    grid_xy,
    horizon,
    plane_grid,
    sample_plane,
    warped_volumes,
)

from baillarger import flatten as FL
from baillarger import flattening as FG
from baillarger import rays as R
from baillarger.column_volume import ColumnVolume

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


@pytest.fixture(autouse=True)
def numpy_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAILLARGER_DEVICE", "numpy")


# --- extents ---


def hand_rays(intervals: list[list[Interval]]) -> R.Rays:
    n = len(intervals)
    return build_rays(np.zeros((n, 3)), np.tile([0, 0, 1.0], (n, 1)), intervals, 0, 0)


def test_extents_from_label_intervals() -> None:
    """Dyadic values: every answer is exact."""
    W, C, OUT = R.WM, R.CORTEX, R.OUTSIDE
    iv: list[list[Interval]] = [
        # the usual column: cortex from 0.25 to 2.125
        [(-2.0, 0.25, W), (0.25, 2.125, C), (2.125, 8.0, OUT)],
        # cortex starts behind the root
        [(-2.0, -0.5, W), (-0.5, 1.5, C), (1.5, 8.0, OUT)],
        # no cortex
        [(-2.0, 8.0, W)],
        # cortex over both caps: clipped
        [(-2.0, 7.75, C), (7.75, 8.0, OUT)],
        # a first cortex interval ends behind the root: the second one counts
        [
            (-2.0, -1.0, C),
            (-1.0, 0.25, W),
            (0.25, 1.5, C),
            (1.5, 2.0, W),
            (2.0, 3.0, C),
            (3.0, 8.0, OUT),
        ],
        # the cortex ends at the root (s1 = 0): not a hit
        [(-2.0, 0.0, C), (0.0, 8.0, OUT)],
        # the cortex starts further out than the margin: e_in is clipped to 0
        [(-2.0, 1.0, W), (1.0, 2.0, C), (2.0, 8.0, OUT)],
    ]
    e_in, e_out = FG.extents(hand_rays(iv))
    assert e_in.tolist() == [0.25, 1.0, 0.5, 2.0, 0.25, 0.5, 0.0]
    assert e_out.tolist() == [2.625, 2.0, 0.5, 8.0, 2.0, 0.5, 2.5]


def test_extents_with_another_margin() -> None:
    iv: list[list[Interval]] = [
        [(-2.0, 0.25, R.WM), (0.25, 2.0, R.CORTEX), (2.0, 8.0, R.OUTSIDE)],
        [(-2.0, 8.0, R.OUTSIDE)],
    ]
    e_in, e_out = FG.extents(hand_rays(iv), margin=0.125)
    assert e_in.tolist() == [0.0, 0.125]
    assert e_out.tolist() == [2.125, 0.125]


def test_extents_on_the_plane_follow_its_intervals() -> None:
    sw = np.array([0.0, 0.25, -0.5, 0.125, 0.0, 0.0])
    st = np.array([3.0, 1.0, 2.0, 7.75, 0.5, 5.0])
    pl = plane_grid(3, 2, s_wm=sw, s_top=st)
    e_in, e_out = FG.extents(pl.rays)
    assert e_in.tolist() == np.clip(0.5 - sw, 0, 2).tolist()
    assert e_out.tolist() == np.clip(st + 0.5, 0, 8).tolist()
    # the plane's own columns have those extents, floored to the step
    assert (pl.rays.k_in == np.floor(e_in / STEP + 1e-6)).all()
    assert (pl.rays.k_out == np.floor(e_out / STEP + 1e-6)).all()


# --- grid_start ---


@pytest.mark.parametrize("dtau", [0.08, 0.04, 0.1])
def test_grid_start_is_a_level_below_the_extent(dtau: float) -> None:
    for e in (0.0, 0.01, 0.08, 0.5, 0.56, 0.6, 1.0, 1.99, 2.0):
        s = FG.grid_start(e, dtau)
        j = round(-s / dtau)
        assert abs(s + j * dtau) < 1e-12  # a multiple of dtau: the root is level j
        assert s <= -e + 1e-9
        assert s > -e - dtau + 1e-9 or abs(s + e) < 1e-9  # the first one below


def test_grid_start_known_values() -> None:
    assert FG.grid_start(0.5) == pytest.approx(-0.56)
    assert FG.grid_start(0.56) == pytest.approx(-0.56)
    assert FG.grid_start(0.0) == 0.0
    assert FG.grid_start(2.0) == pytest.approx(-2.0)


# --- pad ---


def expected_pad(
    cv: ColumnVolume, echoes: tuple[int, ...], s_lo: float, n: int
) -> tuple[F64, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Padded arrays from the definition: sample k of column c goes to index
    k - s_lo / step; cortex is s0 <= s < s1 of a CORTEX interval (float32)."""
    rays = cv.rays
    k_lo = round(s_lo / rays.step)
    m = len(rays.k_in)
    echo = np.full((len(echoes), m, n), np.nan)
    valid = np.zeros((m, n), bool)
    cortex = np.zeros((m, n), bool)
    for c in range(m):
        a = int(rays.sample_start[c])
        q0, q1 = int(rays.interval_start[c]), int(rays.interval_start[c + 1])
        for k in range(-int(rays.k_in[c]), int(rays.k_out[c]) + 1):
            row = a + k + int(rays.k_in[c])
            assert abs(float(rays.s[row]) - k * rays.step) < 1e-6
            i = k - k_lo
            if not 0 <= i < n or not cv.valid[row]:
                continue
            valid[c, i] = True
            for e, ech in enumerate(echoes):
                echo[e, c, i] = float(cv.echo[row, ech])
            s = rays.s[row]
            cortex[c, i] = any(
                rays.interval_class[q] == R.CORTEX
                and rays.interval_s0[q] <= s < rays.interval_s1[q]
                for q in range(q0, q1)
            )
    return echo, valid, cortex


def pad_case() -> tuple[Plane, ColumnVolume]:
    """Columns of different extents on warped volumes that end at s = 3.5."""
    nx, ny = 5, 4
    x, y = grid_xy(nx, ny)
    rng = np.random.default_rng(3)
    sw = rng.uniform(-0.3, 0.3, nx * ny)
    sw[0], sw[1] = 0.2, 0.0  # a boundary on a sample, one on the root
    st = rng.uniform(1.5, 4.5, nx * ny)
    st[0] = 1.0
    pl = plane_grid(nx, ny, s_wm=sw, s_top=st)
    return pl, sample_plane(pl, warped_volumes(pl, x, y, z_max=3.5))


def test_pad_equals_the_sampler_bit_for_bit() -> None:
    pl, cv = pad_case()
    assert len(set(pl.rays.k_in.tolist())) > 2
    assert len(set(pl.rays.k_out.tolist())) > 2
    assert not cv.valid.all()
    assert cv.valid.any()
    p = FG.pad(cv)
    k_lo = round(p.s_lo / STEP)
    assert p.s_lo == FG.grid_start(pl.rays.k_in.max() * STEP)
    assert k_lo <= -int(pl.rays.k_in.max())
    assert p.step == STEP
    assert p.echo.shape == (2, pl.n, int(pl.rays.k_out.max()) - k_lo + 1)
    echo, valid, cortex = expected_pad(cv, FG.FEATURE_ECHOES, p.s_lo, p.valid.shape[1])
    assert np.array_equal(p.echo, echo, equal_nan=True)
    assert np.array_equal(p.valid, valid)
    assert np.array_equal(p.cortex, cortex)
    assert p.valid.sum() == cv.valid.sum()
    # NaN exactly where there is no valid sample
    assert np.array_equal(np.isnan(p.echo[0]), ~p.valid)


def test_pad_with_given_start_length_and_echoes() -> None:
    pl, cv = pad_case()
    p = FG.pad(cv, echoes=(0,), s_lo=-0.2, n=60)
    assert p.echo.shape == (1, pl.n, 60)
    echo, valid, cortex = expected_pad(cv, (0,), -0.2, 60)
    assert np.array_equal(p.echo, echo, equal_nan=True)
    assert np.array_equal(p.valid, valid)
    assert np.array_equal(p.cortex, cortex)
    assert p.s_lo == -0.2


def test_pad_refuses_a_start_off_the_step() -> None:
    _, cv = pad_case()
    with pytest.raises(ValueError, match="multiple"):
        FG.pad(cv, s_lo=-0.55)


def test_pad_cortex_counts_on_a_clean_column() -> None:
    """Cortex from 0.1 to 1.0: the samples k = 3 .. 24 (0.12 to 0.96). The
    sample at 1.0 is at the end of the interval and is out."""
    pl = plane_grid(2, 2, s_wm=0.1, s_top=1.0)
    assert float(pl.rays.s[pl.rays.k_in[0] + 25]) == 1.0
    vol = np.full((2, 2, 251), 5.0, np.float32)
    cv = sample_plane(pl, [vol, vol, vol])
    p = FG.pad(cv)
    k_lo = round(p.s_lo / STEP)
    idx = np.flatnonzero(p.cortex[0])
    assert (idx + k_lo).tolist() == list(range(3, 25))
    assert p.valid[0].all()


def test_pad_cortex_includes_the_start_of_the_interval() -> None:
    pl = plane_grid(2, 2, s_wm=0.2, s_top=1.0)
    assert float(pl.rays.s[pl.rays.k_in[0] + 5]) == float(np.float32(0.2))
    vol = np.full((2, 2, 251), 5.0, np.float32)
    p = FG.pad(sample_plane(pl, [vol, vol, vol]))
    k_lo = round(p.s_lo / STEP)
    assert (np.flatnonzero(p.cortex[0]) + k_lo).tolist() == list(range(5, 25))


# --- features ---


def padded_case(rng: np.random.Generator, m: int = 6, n: int = 140) -> FG.Padded:
    """Random echoes (a quarter of them below 1), runs of cortex, a few
    invalid samples (NaN)."""
    echo = np.exp(rng.normal(1.0, 1.5, (2, m, n)))
    valid = rng.random((m, n)) > 0.05
    cortex = np.zeros((m, n), bool)
    for c in range(m):
        a = int(rng.integers(5, 40))
        cortex[c, a : a + int(rng.integers(30, 60))] = True
        b = int(rng.integers(90, 100))
        cortex[c, b : b + 30] = True
    cortex &= valid
    echo[:, ~valid] = np.nan
    return FG.Padded(-0.56, STEP, echo, valid, cortex)


def trend_by_hand(L: F64, mask: npt.NDArray[np.bool_], sigma: float) -> F64:
    """The Gaussian of sigma samples, cut at int(4 sigma + 0.5), normalised
    over the mask, at the masked samples."""
    n = len(L)
    rad = int(4 * sigma + 0.5)
    out = np.zeros(n)
    for i in map(int, np.flatnonzero(mask)):
        j = np.arange(max(0, i - rad), min(n, i + rad + 1))
        w = np.exp(-0.5 * ((j - i) / sigma) ** 2) * mask[j]
        out[i] = (w * L[j]).sum() / w.sum()
    return out


def test_raw_features_are_the_log_minus_the_local_cortex_level() -> None:
    p = padded_case(np.random.default_rng(0))
    assert (p.echo[np.isfinite(p.echo)] < 1).any()
    raw = FG.raw_features(p)
    sigma = FG.TREND_MM / STEP
    for e in range(2):
        for c in range(p.valid.shape[0]):
            L = np.log(np.maximum(np.nan_to_num(p.echo[e, c], nan=1.0), 1.0))
            want = np.where(p.cortex[c], L - trend_by_hand(L, p.cortex[c], sigma), 0.0)
            np.testing.assert_allclose(raw[e, c], want, rtol=0, atol=1e-12)


def test_the_cortex_level_ignores_samples_outside_the_cortex() -> None:
    p = padded_case(np.random.default_rng(1))
    raw = FG.raw_features(p)
    echo = np.where(p.cortex[None], p.echo, 1e6)
    raw2 = FG.raw_features(FG.Padded(p.s_lo, p.step, echo, p.valid, p.cortex))
    assert np.array_equal(raw, raw2)


def test_features_are_zero_outside_the_cortex() -> None:
    p = padded_case(np.random.default_rng(2))
    F, M, _ = FG.features(p)
    assert np.isfinite(F).all()
    assert (F[:, ~p.cortex] == 0).all()
    assert (F[:, p.cortex] != 0).any()
    assert np.array_equal(M, p.valid)


def test_features_with_given_scales_are_raw_over_scales() -> None:
    p = padded_case(np.random.default_rng(3))
    sc = np.array([2.0, 0.5])
    F, _, used = FG.features(p, scales=sc)
    assert used.tolist() == [2.0, 0.5]
    raw = FG.raw_features(p)
    assert np.array_equal(F, raw / sc[:, None, None])


def test_default_scales_give_unit_sd_over_the_cortex() -> None:
    p = padded_case(np.random.default_rng(4))
    F, _, sc = FG.features(p)
    raw = FG.raw_features(p)
    for e in range(2):
        assert sc[e] == pytest.approx(np.std(raw[e][p.cortex]), rel=1e-12)
        assert np.std(F[e][p.cortex]) == pytest.approx(1.0, abs=1e-12)
    assert FG.scales_of(raw, p.cortex).tolist() == sc.tolist()


def test_scales_of_a_set_of_columns() -> None:
    p = padded_case(np.random.default_rng(5))
    raw = FG.raw_features(p)
    rows = np.array([True, False, True, False, False, True])
    got = FG.scales_of(raw, p.cortex, rows)
    for e in range(2):
        want = np.std(raw[e][p.cortex & rows[:, None]])
        assert got[e] == pytest.approx(want, rel=1e-12)
    assert not np.allclose(got, FG.scales_of(raw, p.cortex))


def test_a_column_does_not_depend_on_the_call_when_the_scales_are_given() -> None:
    p = padded_case(np.random.default_rng(6))
    sc = np.array([1.3, 0.7])
    F, _, _ = FG.features(p, scales=sc)
    c = 3
    one = FG.Padded(
        p.s_lo, p.step, p.echo[:, c : c + 1], p.valid[c : c + 1], p.cortex[c : c + 1]
    )
    F1, _, _ = FG.features(one, scales=sc)
    np.testing.assert_allclose(F1[:, 0], F[:, c], rtol=0, atol=1e-12)
    # with the default scales the column does depend on the call
    G, _, _ = FG.features(p)
    G1, _, _ = FG.features(one)
    assert not np.allclose(G1[:, 0], G[:, c])


# --- pairs ---


def expected_subset_pairs(nx: int, ny: int, cols: npt.NDArray[np.int64]) -> I64:
    pos = {int(c): k for k, c in enumerate(cols)}
    out = sorted(
        (min(pos[a], pos[b]), max(pos[a], pos[b]))
        for a, b in grid_edges(nx, ny).tolist()
        if a in pos and b in pos
    )
    return np.array(out, np.int64).reshape(-1, 2)


def test_the_phantom_mesh_has_the_edges_of_the_grid() -> None:
    """A check of the phantom: 15 * 12 + 16 * 11 + 15 * 11 edges."""
    assert len(grid_edges(16, 12)) == 521
    pl = plane_grid(16, 12)
    assert len(pl.faces) == 2 * 15 * 11
    assert sorted(pl.vertex.tolist()) == list(range(192))


@pytest.mark.parametrize("blocks", [1, 3])
def test_mesh_edge_pairs_are_the_mesh_edges(blocks: int) -> None:
    nx, ny = 7, 5
    pl = plane_grid(nx, ny, blocks=blocks)
    got = FG.mesh_edge_pairs(pl.surf, pl.rays.roots.vertex, pl.voxel)
    want = grid_edges(nx, ny)
    assert np.array_equal(got, want)
    assert len(want) == 6 * 5 + 7 * 4 + 6 * 4


@pytest.mark.parametrize("blocks", [1, 3])
def test_mesh_edge_pairs_of_a_shuffled_subset(blocks: int) -> None:
    nx, ny = 9, 6
    pl = plane_grid(nx, ny, blocks=blocks)
    rng = np.random.default_rng(11)
    cols = rng.permutation(nx * ny)[:30].astype(np.int64)
    got = FG.mesh_edge_pairs(pl.surf, pl.vertex[cols], pl.voxel[cols])
    want = expected_subset_pairs(nx, ny, cols)
    assert 5 < len(want) < len(grid_edges(nx, ny))
    assert np.array_equal(got, want)
    # a subset in one corner, in the order of the grid
    cols = np.array([i * ny + j for i in range(3) for j in range(3)], np.int64)
    got = FG.mesh_edge_pairs(pl.surf, pl.vertex[cols], pl.voxel[cols])
    assert np.array_equal(got, expected_subset_pairs(nx, ny, cols))
    assert len(got) == 12 + 4  # 3 x 3 nodes: 12 along the axes, 4 diagonals
    # columns that are apart share no edge; none, one, or none given
    far = np.array([0, nx * ny - 1], np.int64)
    out = FG.mesh_edge_pairs(pl.surf, pl.vertex[far], pl.voxel[far])
    assert out.shape == (0, 2)
    one = FG.mesh_edge_pairs(pl.surf, pl.vertex[:1], pl.voxel[:1])
    assert one.shape == (0, 2)
    none = FG.mesh_edge_pairs(pl.surf, pl.vertex[:0], pl.voxel[:0])
    assert none.shape == (0, 2)


def test_mesh_edge_pairs_use_the_vertex_not_the_column_index() -> None:
    pl = plane_grid(4, 3)
    assert not np.array_equal(pl.vertex, np.arange(12))
    rev = np.arange(12)[::-1].copy()
    got = FG.mesh_edge_pairs(pl.surf, pl.vertex[rev], pl.voxel[rev])
    assert np.array_equal(got, expected_subset_pairs(4, 3, rev))


def test_proximity_pairs_are_the_eight_neighbours() -> None:
    nx, ny = 5, 4
    pl = plane_grid(nx, ny)
    nrm = pl.rays.roots.normal
    got = FG.proximity_pairs(pl.root_mm, nrm)
    want = sorted(
        (a * ny + b, c * ny + d)
        for a in range(nx)
        for b in range(ny)
        for c in range(nx)
        for d in range(ny)
        if a * ny + b < c * ny + d and max(abs(a - c), abs(b - d)) == 1
    )
    assert got.tolist() == [list(t) for t in want]
    # the distances: 0.16 along an axis, 0.226 along a diagonal, 0.32 next
    d = np.linalg.norm(pl.root_mm[got[:, 0]] - pl.root_mm[got[:, 1]], axis=1)
    assert d.max() < 0.25
    assert d.min() == pytest.approx(0.16, abs=1e-6)
    assert d.max() == pytest.approx(0.16 * np.sqrt(2), abs=1e-6)


def test_proximity_pairs_drop_pairs_with_normals_that_disagree() -> None:
    nx, ny = 5, 4
    pl = plane_grid(nx, ny)
    nrm = pl.rays.roots.normal.astype(np.float64).copy()
    full = FG.proximity_pairs(pl.root_mm, nrm)
    flipped, tilted_ok, tilted_bad = 6, 13, 19
    nrm[flipped] = [0, 0, -1]
    nrm[tilted_ok] = [0.6, 0, 0.8]  # dot 0.8 with +z: kept
    nrm[tilted_bad] = [0.8, 0, 0.6]  # dot 0.6 with +z: dropped
    got = FG.proximity_pairs(pl.root_mm, nrm)
    # the same pairs of the grid, minus those with a column of a bad normal
    bad = (full == flipped).any(1) | (full == tilted_bad).any(1)
    assert not (full == tilted_ok).all(1).any()
    assert np.array_equal(got, full[~bad])
    assert (got == tilted_ok).any()
    assert bad.sum() == 8 + 3  # column 6 has 8 neighbours, column 19 has 3


# --- pair_shifts ---


def test_pair_shifts_do_not_depend_on_the_chunks_and_keep_the_reference() -> None:
    rng = np.random.default_rng(8)
    n = 120
    base = rng.standard_normal(n + 6)
    base = np.convolve(base, np.ones(5) / 5, mode="same")
    F = np.stack(
        [np.stack([base[k : k + n] for k in range(4)]) for _ in range(2)]
    )  # (2, 4, n)
    M = np.ones((4, n), bool)
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], np.int64)
    a = FG.pair_shifts(F, M, pairs, STEP)
    b = FG.pair_shifts(F, M, pairs, STEP, chunk=3)
    assert np.array_equal(a.pairs, pairs)
    assert a.shift.shape == (4, n)
    assert np.array_equal(a.shift, b.shift)
    assert np.array_equal(a.weight, b.weight)
    assert a.energy_ref == b.energy_ref
    c = FG.pair_shifts(F, M, pairs, STEP, energy_ref=2 * a.energy_ref)
    assert c.energy_ref == 2 * a.energy_ref
    assert not np.array_equal(c.weight, a.weight)


# --- end to end ---

NX, NY = 16, 12


def test_the_flattening_recovers_the_known_warp() -> None:
    """The phantom of test_flatten.grid_volume (alpha 0.2, beta 0.12, top
    4.6) on a plane of 16 x 12 columns, through the real sampler, pad,
    features, the mesh-edge pairs, pair_shifts and fit.

    The pattern depth u is at s = Z_c(u) on column c. Level j is at
    s = s_lo + tau_j before the warp, which is the pattern depth
    u_j = s_lo + tau_j. So the true warp is r_c(tau_j) = Z_c(u_j) - u_j. It
    is 0 at u = 0, which is the anchor level (the root, and the start of
    the cortex).

    Tolerances. Measured on this phantom: median error 0.0017 mm, 95th
    percentile 0.0079 mm, largest 0.020 mm, over the levels with
    0.3 < u < 4.2. The bounds are about 3 times the measured values for the
    median and the percentile, and 2 times for the largest. They are far
    under the error of no warp at all (median 0.19 mm, 95th percentile 0.65
    mm), and under the bounds of test_flatten.test_fit_recovers_warp (0.01,
    0.03)"""
    x, y = grid_xy(NX, NY)
    pl = plane_grid(NX, NY, s_wm=0.0, s_top=TOP * (1 + ALPHA * x))
    cv = sample_plane(pl, warped_volumes(pl, x, y))
    assert cv.valid.all()
    p = FG.pad(cv)
    F, M, _ = FG.features(p)
    pairs = FG.mesh_edge_pairs(pl.surf, pl.rays.roots.vertex, pl.voxel)
    assert np.array_equal(pairs, grid_edges(NX, NY))
    ps = FG.pair_shifts(F, M, pairs, p.step)
    w = FG.fit(ps, pl.n, p.s_lo, p.step)
    assert w.s_lo == p.s_lo
    assert w.anchor_level == 6  # k_in = 12 samples: s_lo = -0.48
    assert w.tau[w.anchor_level] + w.s_lo == pytest.approx(0.0, abs=1e-12)
    assert w.r.shape == (pl.n, len(w.tau))
    u = w.s_lo + w.tau
    true_r = horizon(x, y, u) - u[None]
    assert abs(true_r[:, w.anchor_level]).max() < 1e-12
    assert abs(w.r[:, w.anchor_level]).max() < 5e-3
    inner = (u > 0.3) & (u < 4.2)
    err = np.abs(w.r - true_r)[:, inner]
    assert np.median(err) < 0.005
    assert np.percentile(err, 95) < 0.025
    assert err.max() < 0.04


# ---------------------------------------------------------------- orient


def test_orient_puts_the_smaller_key_first_whatever_the_order() -> None:
    key = np.array([50, 10, 40, 20, 30], np.int64)
    pairs = np.array([[0, 1], [2, 3], [1, 4], [3, 4]], np.int64)
    got = FG.orient(pairs, key)
    assert np.all(key[got[:, 0]] < key[got[:, 1]])
    # the same pairs in another column order give the same key pairs
    perm = np.array([3, 0, 4, 1, 2])
    inv = np.argsort(perm)
    got2 = FG.orient(inv[pairs][::-1], key[perm])
    a = {tuple(x) for x in key[got].tolist()}
    b = {tuple(x) for x in key[perm][got2].tolist()}
    assert a == b


# ---------------------------------------------------------------- solver


def test_the_line_preconditioner_inverts_the_tridiagonal_blocks() -> None:
    rng = np.random.default_rng(3)
    N, K, off = 5, 9, -0.04
    d = 0.2 + rng.random((N, K))
    P = FL.line_preconditioner(d, off)
    b = rng.standard_normal(N * K)
    x = P.matvec(b).reshape(N, K)
    for v in range(N):
        T = np.diag(d[v]) + off * (np.eye(K, k=1) + np.eye(K, k=-1))
        np.testing.assert_allclose(T @ x[v], b.reshape(N, K)[v], atol=1e-12)


def test_fit_warp_gives_the_same_warp_with_either_preconditioner() -> None:
    from test_flatten import grid_volume

    F, ns, pairs, *_ = grid_volume()
    M = np.isfinite(F[0])
    Fz = np.nan_to_num(F)
    v, w = pairs[:, 0], pairs[:, 1]
    s, ncc, en, va = FL.shifts(Fz[:, v], Fz[:, w], M[v], M[w], 0.04)
    W = FL.pair_weights(ncc, en, va, float(np.median(en[va])))
    out = [
        FL.fit_warp(pairs, s, W, F.shape[1], 0.04, tau_max=4.4, precond=pc)[1]
        for pc in ("jacobi", "line")
    ]
    assert np.abs(out[0] - out[1]).max() < 1e-5


# ---------------------------------------------------------------- the gauge


def hand_results(
    rng: np.random.Generator, n: int = 60, B: int = 4, K: int = 12
) -> tuple[list[FG.Flattened], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A warp r (n, K) with mean 0 over the columns at each level (the global
    gauge), split into B superblocks of consecutive columns, each with a halo
    of 3 columns on each side; each superblock's warp is r plus its own
    constant per level (its gauge). Returns the results, r and the gauges."""
    r = rng.standard_normal((n, K))
    r -= r.mean(0)
    gauge = rng.standard_normal((B, K))
    keys = np.arange(n, dtype=np.int64) * 7 + 3
    out = []
    bounds = np.linspace(0, n, B + 1).astype(int)
    for b in range(B):
        a, e = bounds[b], bounds[b + 1]
        halo = np.r_[np.arange(max(0, a - 3), a), np.arange(e, min(n, e + 3))]
        out.append(
            FG.Flattened(
                edge_key=keys[a:e],
                s_lo=-0.16,
                step=0.04,
                tau=np.arange(K, dtype=np.float64) * 0.08,
                r=r[a:e] + gauge[b],
                anchor_level=2,
                pair_key=np.zeros((0, 2), np.int64),
                shift=np.zeros((0, 1), np.float32),
                weight=np.zeros((0, 1), np.float32),
                scales=np.ones(2),
                energy_ref=1.0,
                info=[],
                n_columns=0,
                n_pairs=0,
                misfit=np.zeros(e - a),
                vertex=keys[a:e],
                halo_key=keys[halo],
                halo_r=r[halo] + gauge[b],
            )
        )
    return out, r, gauge


def test_gauge_offsets_remove_the_gauges_exactly() -> None:
    out, r, gauge = hand_results(np.random.default_rng(5))
    lev, c = FG.gauge_offsets(out)
    np.testing.assert_array_equal(lev, np.arange(-2, 10))
    got = np.concatenate([FG.in_gauge(f, c[b], lev) for b, f in enumerate(out)])
    np.testing.assert_allclose(got, r, atol=1e-10)
    np.testing.assert_allclose(c, -gauge, atol=1e-10)


def test_gauge_offsets_per_connected_set() -> None:
    """Two sets of superblocks with no overlap between them: each set gets
    the gauge with the sum 0 over its own columns."""
    rng = np.random.default_rng(6)
    a, ra, _ = hand_results(rng, n=40, B=2)
    b, rb, _ = hand_results(rng, n=40, B=2)
    b = [
        FG.Flattened(
            **{
                **f.__dict__,
                "edge_key": f.edge_key + 1000,
                "halo_key": f.halo_key + 1000,
            }
        )
        for f in b
    ]
    lev, c = FG.gauge_offsets(a + b)
    got = [FG.in_gauge(f, c[i], lev) for i, f in enumerate(a + b)]
    np.testing.assert_allclose(np.concatenate(got[:2]), ra, atol=1e-10)
    np.testing.assert_allclose(np.concatenate(got[2:]), rb, atol=1e-10)


def test_common_levels_extend_the_first_and_last_levels() -> None:
    out, r, gauge = hand_results(np.random.default_rng(7), B=1)
    f = out[0]
    lev = np.arange(-4, 13)
    got = FG.common_levels(f, f.r, lev)
    np.testing.assert_array_equal(got[:, 2:14], f.r)
    np.testing.assert_array_equal(got[:, 0], f.r[:, 0])
    np.testing.assert_array_equal(got[:, -1], f.r[:, -1])
