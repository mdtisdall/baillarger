"""Tests of baillarger.viewer.section (the profile section that follows the
surface), with exact answers, on phantom meshes: a flat sheet, a thin fold (two
parallel sheets 0.16 mm apart joined by a half cylinder), and two separate
sheets. The columns at each step are checked against distances on the whole
graph; the order of the section against the sheets' own order."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from baillarger.mesh_graph import edge_graph
from baillarger.viewer import section as VSec

type Float = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

H = 0.1  # the phantoms' vertex spacing, mm
GAP = 0.16  # the distance between the fold's sheets, mm (one voxel)
VOXEL = 0.16


def mean_spacing(roots: Float) -> float:
    """The phantoms' step: the mean distance from each root to its nearest
    other root, mm (the grid spacing on a regular sheet)."""
    d, _ = cKDTree(roots).query(roots, k=2)
    return float(d[:, 1].mean())


def grid(t: Float, y: Float) -> tuple[I64, Float, Float]:
    """The faces of a grid of len(t) x len(y) vertices (vertex i * len(y) + j
    at (t[i], y[j])), two triangles a cell, and the grid's coordinates."""
    nt, ny = len(t), len(y)
    i, j = np.meshgrid(np.arange(nt - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    b, c, d = a + ny, a + 1, a + ny + 1
    f = np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])
    tt, yy = np.meshgrid(t, y, indexing="ij")
    return f, tt.ravel(), yy.ravel()


def sheet(half: float = 2.0, z: float = 0.0) -> tuple[Float, I64]:
    """A flat sheet z = const, x, y in [-half, half], spacing H."""
    s = np.round(np.arange(-half, half + H / 2, H), 9)
    f, x, y = grid(s, s)
    return np.c_[x, y, np.full(len(x), z)], f


def fold() -> tuple[Float, I64]:
    """Sheet A (z = 0) and sheet B (z = GAP) for x in [-2, 0], y in [-1, 1],
    joined at x = 0 by a half cylinder of radius GAP / 2 (x >= 0): one strip,
    parametrised by the arc length t of its profile."""
    r = GAP / 2
    half_turn = np.pi * r
    t_a = np.round(np.arange(0.0, 2.0, H), 9)
    t_turn = 2.0 + np.linspace(0.0, half_turn, 7)
    t_b = 2.0 + half_turn + np.round(np.arange(0.0, 2.0 + H / 2, H), 9)
    t = np.concatenate([t_a, t_turn, t_b[1:]])
    y = np.round(np.arange(-1.0, 1.0 + H / 2, H), 9)
    f, tt, yy = grid(t, y)
    x = np.where(
        tt <= 2.0,
        tt - 2.0,
        np.where(
            tt <= 2.0 + half_turn,
            r * np.sin((tt - 2.0) / r),
            -(tt - 2.0 - half_turn),
        ),
    )
    z = np.where(
        tt <= 2.0,
        0.0,
        np.where(tt <= 2.0 + half_turn, r - r * np.cos((tt - 2.0) / r), GAP),
    )
    return np.c_[x, yy, z], f


def two_sheets() -> tuple[Float, I64]:
    """Sheet A (z = 0) and sheet B (z = GAP), not joined."""
    pa, fa = sheet()
    pb, fb = sheet(z=GAP)
    return np.concatenate([pa, pb]), np.concatenate([fa, fb + len(pa)])


def vertex_at(pts: Float, xyz: list[float]) -> int:
    d = np.linalg.norm(pts - np.asarray(xyz), axis=1)
    i = int(np.argmin(d))
    assert d[i] < 1e-9
    return i


def section(
    pts: Float,
    f: I64,
    origin: int,
    normal: list[float],
    u: list[float],
    radius: float,
    mode: str = "nearest",
    column_of: I64 | None = None,
) -> tuple[VSec.Curve, Float, Float, I64, list[list[int]], Float, csr_matrix]:
    g = edge_graph(pts, f, [1.0, 1.0, 1.0])
    cols = np.arange(len(pts)) if column_of is None else column_of
    curve = VSec.plane_cut(pts, f, pts[origin], normal, u, radius)
    on = cols >= 0
    step = mean_spacing(pts[on])
    arc, p, fi = VSec.steps(curve, step)
    out, dist = VSec.step_columns(
        p, fi, pts, f, g, cKDTree(pts), cols, mode, VOXEL, radius
    )
    return curve, arc, p, fi, out, dist, g


def brute(p: Float, seeds: I64, pts: Float, g: csr_matrix) -> npt.NDArray[np.float64]:
    """The distances along the surface from p (in the face `seeds`) to every
    vertex, on the whole graph."""
    n = len(pts)
    c = g.tocoo()
    off = np.linalg.norm(pts[seeds] - p, axis=1)
    off = np.where(off > 0, off, np.finfo(np.float64).tiny)
    rows = np.concatenate([c.row, np.full(len(seeds), n)])
    cols = np.concatenate([c.col, seeds])
    aug = coo_matrix(
        (np.concatenate([c.data, off]), (rows, cols)), shape=(n + 1, n + 1)
    ).tocsr()
    d: npt.NDArray[np.float64] = dijkstra(aug, directed=True, indices=n)[:n]
    d[d == np.finfo(np.float64).tiny] = 0.0
    return d


def check_against_brute(
    pts: Float,
    f: I64,
    p: Float,
    fi: I64,
    out: list[list[int]],
    dist: Float,
    g: csr_matrix,
    column_of: I64,
    mode: str,
    limit: float,
) -> None:
    for k in range(len(p)):
        d = brute(p[k], f[fi[k]], pts, g)
        on = np.flatnonzero(column_of >= 0)
        if mode == "mean":
            want = sorted(int(column_of[v]) for v in on if d[v] <= VOXEL)
            assert out[k] == want, k
        else:
            dd, cc = d[on], column_of[on]
            if not np.isfinite(dd).any() or dd.min() > limit:
                assert out[k] == [], k
                continue
            j = np.lexsort((cc, dd))[0]
            # the search may stop at limit; within it, the nearest is exact
            assert out[k] == [int(cc[j])], k
            assert dist[k] == pytest.approx(float(dd[j]), abs=1e-12)


# ---------------------------------------------------------------- the curve


@pytest.mark.parametrize("angle", [0.0, 7.0, 45.0, 90.0])
def test_on_a_flat_sheet_the_curve_is_the_line_along_u(angle: float) -> None:
    pts, f = sheet()
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    a = np.deg2rad(angle)
    u = [np.cos(a), np.sin(a), 0.3]  # not normal to n: made so
    curve, arc, p, fi, out, *_ = section(pts, f, o, [0, 0, 1], u, 1.5)
    ua = np.array([np.cos(a), np.sin(a), 0.0])
    assert curve.u == pytest.approx(ua)
    assert np.abs(curve.point @ curve.w).max() < 1e-12
    assert curve.point @ ua == pytest.approx(curve.arc, abs=1e-12)
    assert curve.arc[0] == pytest.approx(-1.5) and curve.arc[-1] == pytest.approx(1.5)
    assert p @ ua == pytest.approx(arc, abs=1e-12)
    # the steps: every mean spacing from 0 (the spacing of the grid: H)
    assert np.diff(arc) == pytest.approx(H)
    assert 0.0 in arc.round(12)


@pytest.mark.parametrize("u", [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
def test_on_a_flat_sheet_the_order_is_the_order_by_u(u: list[float]) -> None:
    pts, f = sheet()
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    _, arc, _, _, out, dist, _ = section(pts, f, o, [0, 0, 1], u, 1.5)
    cols = [c[0] for c in out]
    # the line of vertices through the origin along u, sorted by u
    q = (pts - pts[o]) @ np.asarray(u)
    row = np.flatnonzero(
        (np.abs((pts - pts[o]) @ np.cross([0, 0, 1], u)) < 1e-9)
        & (np.abs(q) <= 1.5 + 1e-9)
    )
    assert cols == row[np.argsort(q[row])].tolist()
    assert dist == pytest.approx(0.0, abs=1e-12)  # each step is at a vertex


def test_columns_against_the_whole_graph_on_a_flat_sheet() -> None:
    pts, f = sheet(1.0)
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    u = [np.cos(0.4), np.sin(0.4), 0.0]
    for mode in VSec.MODES:
        _, _, p, fi, out, dist, g = section(pts, f, o, [0, 0, 1], u, 0.9, mode)
        check_against_brute(pts, f, p, fi, out, dist, g, np.arange(len(pts)), mode, 0.9)


def test_sparse_columns_are_found_by_the_widening_search() -> None:
    """With a column at every 7th vertex only (as with `every`), the nearest
    column is often several edges away: the search widens until it is found."""
    pts, f = sheet(1.0)
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    cols = -np.ones(len(pts), np.int64)
    cols[::7] = np.arange(len(cols[::7]))
    u = [np.cos(0.4), np.sin(0.4), 0.0]
    _, _, p, fi, out, dist, g = section(pts, f, o, [0, 0, 1], u, 0.9, column_of=cols)
    check_against_brute(pts, f, p, fi, out, dist, g, cols, "nearest", 0.9)
    assert all(len(c) == 1 for c in out)


# ---------------------------------------------------------------- the thin fold


def sheet_of(x: Float) -> int:
    """0: sheet A (z = 0, x <= 0), 1: the turn (x > 0), 2: sheet B."""
    if x[0] > 1e-9:
        return 1
    return 0 if abs(x[2]) < 1e-9 else 2


@pytest.mark.parametrize("mode", VSec.MODES)
def test_a_thin_fold_root_sheet_then_turn_then_other_sheet(mode: str) -> None:
    pts, f = fold()
    o = vertex_at(pts, [-0.5, 0.0, 0.0])
    curve, arc, p, fi, out, dist, g = section(
        pts, f, o, [0, 0, -1], [1.0, 0.0, 0.0], 2.0, mode
    )
    # the curve goes round the turn: 0.5 on A, the half circle, the rest on B
    turn = np.pi * GAP / 2
    assert curve.arc[-1] == pytest.approx(2.0)
    assert curve.arc[0] == pytest.approx(-1.5)
    on_b = curve.point[curve.arc > 0.5 + turn + 1e-6]
    assert np.allclose(on_b[:, 2], GAP) and np.all(on_b[:, 0] <= 1e-9)
    # the columns: A, then the turn, then B; no column of one sheet comes
    # between two columns of the other
    sets = [{sheet_of(pts[c]) for c in cs} for cs in out]
    if mode == "nearest":
        assert all(len(cs) == 1 for cs in out)
    steps_of = {s: [k for k, ls in enumerate(sets) if s in ls] for s in (0, 1, 2)}
    assert all(steps_of.values())
    # every A step before every B step; a step holds both only on the turn,
    # where one voxel along the surface reaches round it to both sheets
    assert max(steps_of[0]) <= min(steps_of[2])
    both = [k for k, ls in enumerate(sets) if {0, 2} <= ls]
    assert len(both) <= 1 and all(p[k][0] > 0 for k in both)
    assert min(steps_of[1]) > min(steps_of[0]) and max(steps_of[1]) < max(steps_of[2])
    # in order of arc length the sheets never go back: A, turn, B
    lo, hi = [min(ls) for ls in sets], [max(ls) for ls in sets]
    assert lo == sorted(lo) and hi == sorted(hi)
    check_against_brute(pts, f, p, fi, out, dist, g, np.arange(len(pts)), mode, 2.0)


# ---------------------------------------------------------------- a separate sheet


@pytest.mark.parametrize("mode", VSec.MODES)
def test_a_separate_sheet_near_the_plane_is_not_in_the_section(mode: str) -> None:
    pts, f = two_sheets()
    n_a = len(pts) // 2
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    curve, arc, p, fi, out, *_ = section(
        pts, f, o, [0, 0, 1], [1.0, 0.0, 0.0], 1.5, mode
    )
    assert np.allclose(curve.point[:, 2], 0.0)
    cols = [c for cs in out for c in cs]
    assert cols and all(c < n_a for c in cols)
    # sheet B is within one voxel in space of every step
    d, _ = cKDTree(pts[n_a:]).query(p)
    assert np.all(d <= VOXEL + 1e-12)


# ---------------------------------------------------------------- no alternation


def lag_corr(v: Float, lag: int) -> float:
    return float(np.corrcoef(v[:-lag], v[lag:])[0, 1])


def strip(pts: Float, o: int, u: Float, w: Float, half: float, r: float) -> I64:
    """The previous section (columns.js before this change): the columns within
    `half` of the plane, sorted by u."""
    q = pts - pts[o]
    keep = np.flatnonzero((np.abs(q @ w) <= half) & (np.abs(q @ u) <= r))
    out: I64 = keep[np.argsort(q[keep] @ u, kind="stable")]
    return out


def test_no_alternation_where_the_image_changes_across_the_curve() -> None:
    """The image changes across the curve (it is the distance from the plane)
    and not along it. The section's columns change smoothly with arc length:
    the correlation at lag 1 is not less than at lag 2. The previous section (a
    strip of half width 0.25 mm sorted by u) puts columns of different lines
    next to each other: its neighbours are less alike."""
    pts, f = sheet()
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    a = np.deg2rad(7.0)
    u = [np.cos(a), np.sin(a), 0.0]
    curve, arc, p, fi, out, *_ = section(pts, f, o, [0, 0, 1], u, 1.9)
    image = (pts - pts[o]) @ curve.w
    v = np.array([image[cs[0]] for cs in out])
    assert np.ptp(v) > 0
    r1, r2 = lag_corr(v, 1), lag_corr(v, 2)
    assert r1 >= r2, (r1, r2)
    # the previous section on the same sheet: its neighbours are less alike
    # (lag 1) and differ more (the mean squared step over the variance)
    old = image[strip(pts, o, curve.u, curve.w, 0.25, 1.9)]
    assert lag_corr(old, 1) < r1

    def jump(x: Float) -> float:
        return float(np.mean(np.diff(x) ** 2) / np.var(x))

    assert jump(old) > jump(v)


def test_bad_input() -> None:
    pts, f = sheet(0.5)
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="parallel"):
        VSec.plane_cut(pts, f, pts[o], [0, 0, 1], [0, 0, 2], 1.0)
    with pytest.raises(ValueError, match="mode"):
        VSec.step_columns(
            pts[:1],
            np.zeros(1, np.int64),
            pts,
            f,
            edge_graph(pts, f, [1, 1, 1]),
            cKDTree(pts),
            np.arange(len(pts)),
            "median",
            VOXEL,
            1.0,
        )
    with pytest.raises(ValueError, match="cut"):
        VSec.plane_cut(pts, f, [0, 0, 5.0], [0, 0, 1], [1, 0, 0], 1.0)


# ---------------------------------------------------------------- a slice plane


def dome(radius: float = 1.0, half: float = 1.0) -> tuple[Float, I64]:
    """A cap z = -(x^2 + y^2) / (2 radius), x, y in [-half, half], spacing H:
    its top (the origin) has the normal +z, and every other vertex is below."""
    pts, f = sheet(half)
    pts[:, 2] = -(pts[:, 0] ** 2 + pts[:, 1] ** 2) / (2 * radius)
    return pts, f


def slice_section(
    pts: Float,
    f: I64,
    origin: int,
    normal: list[float],
    u: list[float],
    radius: float,
    plane: list[float] | None,
    mode: str = "nearest",
) -> tuple[VSec.Curve, str | None, Float, list[list[int]]]:
    """The section as Columns.section makes it, with a slice plane."""
    g = edge_graph(pts, f, [1.0, 1.0, 1.0])
    step = mean_spacing(pts)
    curve, why = VSec.section_curve(pts, f, pts[origin], normal, u, radius, step, plane)
    arc, p, fi = VSec.steps(curve, step)
    cols = np.arange(len(pts))
    out, _ = VSec.step_columns(
        p, fi, pts, f, g, cKDTree(pts), cols, mode, VOXEL, radius
    )
    return curve, why, arc, out


@pytest.mark.parametrize("mode", VSec.MODES)
@pytest.mark.parametrize("plane", [[0.0, 1.0, 0.0], [0.0, -3.0, 0.0]])
def test_a_slice_that_holds_the_normal_is_the_plane_of_n_and_u(
    plane: list[float], mode: str
) -> None:
    """On the fold, from sheet A: the slice y = 0 holds the normal (+z) and u
    (+x); its section is the section of the plane of n and u, round the turn."""
    pts, f = fold()
    o = vertex_at(pts, [-1.0, 0.0, 0.0])
    want = slice_section(pts, f, o, [0, 0, 1], [1, 0, 0], 3.0, None, mode)
    got = slice_section(pts, f, o, [0, 0, 1], [1, 0, 0], 3.0, plane, mode)
    assert got[1] is None
    np.testing.assert_array_equal(got[0].point, want[0].point)
    np.testing.assert_array_equal(got[0].arc, want[0].arc)
    np.testing.assert_array_equal(got[0].face, want[0].face)
    np.testing.assert_array_equal(got[2], want[2])
    assert got[3] == want[3]
    assert got[0].point[:, 2].max() == pytest.approx(GAP)  # round the turn


def test_a_slanted_slice_holds_the_normal_and_u_at_any_u() -> None:
    """On the dome, u at 30 degrees: the slice n x u gives the same curve."""
    pts, f = dome()
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    u = [np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0]
    w = np.cross([0.0, 0.0, 1.0], u)
    want, _, arc_w, cols_w = slice_section(pts, f, o, [0, 0, 1], u, 0.8, None)
    got, why, arc_g, cols_g = slice_section(pts, f, o, [0, 0, 1], u, 0.8, list(w))
    assert why is None
    np.testing.assert_allclose(got.point, want.point, atol=1e-12)
    np.testing.assert_array_equal(got.face, want.face)
    np.testing.assert_allclose(arc_g, arc_w, atol=1e-12)
    assert cols_g == cols_w


@pytest.mark.parametrize("mode", VSec.MODES)
def test_a_slanted_slice_gives_a_curve_in_the_slice(mode: str) -> None:
    """On the dome, a slice that does not hold the normal: every point of the
    curve lies in the slice plane through the root, the curve goes through the
    root, and each step's columns are the nearest along the surface."""
    pts, f = dome()
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    plane = [0.2, 1.0, 0.6]
    curve, why, arc, out = slice_section(
        pts, f, o, [0, 0, 1], [1, 0, 0], 0.8, plane, mode
    )
    assert why is None
    w = np.asarray(plane) / np.linalg.norm(plane)
    assert np.abs((curve.point - pts[o]) @ w).max() < 1e-12
    assert curve.point[np.argmin(np.abs(curve.arc))] == pytest.approx(pts[o])
    assert -curve.arc[0] == pytest.approx(0.8) and curve.arc[-1] == pytest.approx(0.8)
    # +u: u made to lie in the slice
    k = int(np.argmin(np.abs(curve.arc - 0.05)))
    assert (curve.point[k] - pts[o]) @ [1, 0, 0] > 0
    g = edge_graph(pts, f, [1.0, 1.0, 1.0])
    step = mean_spacing(pts)
    _, p, fi = VSec.steps(curve, step)
    _, dist = VSec.step_columns(
        p, fi, pts, f, g, cKDTree(pts), np.arange(len(pts)), mode, VOXEL, 0.8
    )
    check_against_brute(pts, f, p, fi, out, dist, g, np.arange(len(pts)), mode, 0.8)


@pytest.mark.parametrize(
    ("phantom", "reason"),
    [(sheet, "does not cut"), (dome, "ends within one step")],
)
def test_a_slice_parallel_to_the_surface_falls_back(
    phantom: object, reason: str
) -> None:
    """The slice z = 0 at the root: on the flat sheet it does not cut the
    surface; on the dome it only touches it at the root. Both give the plane
    of n and u, and say why."""
    pts, f = phantom()  # type: ignore[operator]
    o = vertex_at(pts, [0.0, 0.0, 0.0])
    want, _, arc_w, cols_w = slice_section(pts, f, o, [0, 0, 1], [1, 0, 0], 0.8, None)
    got, why, arc_g, cols_g = slice_section(
        pts, f, o, [0, 0, 1], [1, 0, 0], 0.8, [0.0, 0.0, 2.0]
    )
    assert why is not None and reason in why
    np.testing.assert_array_equal(got.point, want.point)
    np.testing.assert_array_equal(arc_g, arc_w)
    assert cols_g == cols_w


def test_slice_axes() -> None:
    with pytest.raises(ValueError, match="zero"):
        VSec.slice_axes([0, 0, 1], [1, 0, 0], [0, 0, 0])
    # u normal to the slice: the axis is plane x normal
    a, w = VSec.slice_axes([0, 0, 1], [0, 1, 0], [0, 1, 0])
    np.testing.assert_allclose(np.abs(a), [1, 0, 0])
    # the slice normal to n: an axis in the slice
    a, w = VSec.slice_axes([0, 0, 1], [0, 0, 1], [0, 0, 1])
    assert abs(a @ w) < 1e-12 and np.linalg.norm(a) == pytest.approx(1)
