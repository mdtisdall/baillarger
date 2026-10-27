"""Correctness tests for baillarger.fronts: exact eikonal solutions, labels,
arc length, and invariants. Grids are small (at most 61^3)."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from baillarger.fronts import march, riemann_from_direction, selling

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def point_seed(n: int, c: tuple[int, int, int] | None = None) -> Int:
    lab = -np.ones((n, n, n), np.int64)
    c = c or (n // 2, n // 2, n // 2)
    lab[c] = 0
    return lab


def grid(n: int, c: tuple[int, int, int], h: Float | None = None) -> Float:
    """Positions relative to c, in the units of h (default voxels)."""
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).astype(float)
    g -= np.array(c, float)
    return g if h is None else g * h


def gradient_angles(u: Float, v: Float, keep: npt.NDArray[np.bool_]) -> Float:
    """Angle (degrees) between the grid gradients of u and v where keep."""
    gu = np.stack([np.asarray(d, np.float64) for d in np.gradient(u)], -1)
    gv = np.stack([np.asarray(d, np.float64) for d in np.gradient(v)], -1)
    ok = keep & np.all(np.isfinite(gu), -1)
    cos = np.sum(gu[ok] * gv[ok], -1) / (
        np.linalg.norm(gu[ok], axis=-1) * np.linalg.norm(gv[ok], axis=-1)
    )
    out: Float = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out


def rotated_metric(ratio: float) -> Float:
    v = np.array([1.0, 2.0, 0.5])
    return riemann_from_direction(v, ratio)


# ---------------------------------------------------------------- Selling


@pytest.mark.parametrize("ratio", [1.0, 3.0, 10.0, 30.0])
def test_selling_reconstructs(ratio: float) -> None:
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((100, 3, 3)))
    ev = np.stack(
        [np.ones(100), np.full(100, ratio**2), rng.uniform(1, ratio**2, 100)], 1
    )
    d = q @ (ev[:, :, None] * np.eye(3)) @ q.transpose(0, 2, 1)
    off, w = selling(d)
    assert off.dtype == np.int8
    assert np.all(w >= 0)
    rec = np.einsum("np,npi,npj->nij", w, off.astype(float), off.astype(float))
    assert np.max(np.abs(rec - d)) <= 1e-10 * np.max(np.abs(d))


def test_selling_rejects_bad_matrices() -> None:
    with pytest.raises(ValueError):
        selling(np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        selling(np.diag([1.0, np.nan, 1.0]))
    with pytest.raises(ValueError):
        selling(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


# ---------------------------------------------------------------- arrival


def test_plane_source_is_exact() -> None:
    """A plane source and a metric with unit cost along its normal: arrival =
    distance to the plane, exactly, and length = arrival."""
    n = 21
    lab = -np.ones((n, n, n), np.int64)
    lab[0] = 0
    h = np.array([0.16028, 0.16, 0.16028])
    for metric in (np.eye(3), riemann_from_direction(np.array([1.0, 0.0, 0.0]), 5.0)):
        f = march(metric, lab, spacing=h)
        x = np.arange(n)[:, None, None] * h[0] * np.ones((n, n, n))
        assert np.allclose(f.arrival, x, atol=1e-12)
        assert np.allclose(f.length, x, atol=1e-12)


def test_isotropic_point_source() -> None:
    n, c = 41, (20, 20, 20)
    f = march(np.eye(3), point_seed(n, c))
    ex = np.linalg.norm(grid(n, c), axis=-1)
    far = ex > 5
    rel = np.abs(f.arrival[far] - ex[far]) / ex[far]
    assert np.median(rel) < 0.06
    assert np.max(rel) < 0.25
    # isotropic: the flow is the gradient, so the length equals the arrival
    assert np.allclose(f.length, f.arrival, rtol=1e-10, atol=1e-12)


def test_isotropic_converges() -> None:
    """The error at a fixed physical radius falls when the grid is refined."""
    errs = []
    for n in (21, 41):
        c = (n // 2,) * 3
        h = np.full(3, 2.0 / (n - 1))
        f = march(np.eye(3), point_seed(n, c), spacing=h)
        ex = np.linalg.norm(grid(n, c, h), axis=-1)
        shell = (ex > 0.5) & (ex < 0.9)
        errs.append(float(np.mean(np.abs(f.arrival[shell] - ex[shell]))))
    assert errs[1] < 0.75 * errs[0]


@pytest.mark.parametrize("ratio", [3.0, 10.0])
def test_constant_anisotropic_metric(ratio: float) -> None:
    """Exact arrival sqrt(x^T M x); the arrival's direction is not biased by
    the grid (the voxel shortest path was about 29 degrees off)."""
    n, c = 41, (20, 20, 20)
    m = rotated_metric(ratio)
    f = march(m, point_seed(n, c))
    g = grid(n, c)
    ex = np.sqrt(np.einsum("...i,ij,...j->...", g, m, g))
    far = np.linalg.norm(g, axis=-1) > 5
    rel = np.abs(f.arrival[far] - ex[far]) / ex[far]
    assert np.median(rel) < 0.08
    ang = gradient_angles(f.arrival, ex, far)
    assert np.median(ang) < 2.0
    assert np.percentile(ang, 95) < 6.0


def oblique_plane_length_error(ratio: float) -> tuple[float, float]:
    """Arc length from the plane source x = 0 under a constant metric oblique
    to it: the characteristics are straight, along V = M^-1 e_x, so the exact
    length is x |V| / V_x. Returns the median and 95th percentile of the
    relative error where the characteristic from the plane stays in the box."""
    n = 41
    m = rotated_metric(ratio)
    lab = -np.ones((n, n, n), np.int64)
    lab[0] = 0
    f = march(m, lab)
    v = np.linalg.inv(m) @ np.array([1.0, 0.0, 0.0])
    x, y, z = np.meshgrid(*[np.arange(n, dtype=float)] * 3, indexing="ij")
    exact = x * np.linalg.norm(v) / v[0]
    y0 = y - v[1] / v[0] * x
    z0 = z - v[2] / v[0] * x
    keep = (x >= 5) & (y0 >= 0) & (y0 <= n - 1) & (z0 >= 0) & (z0 <= n - 1)
    rel = np.abs(f.length[keep] - exact[keep]) / exact[keep]
    return float(np.median(rel)), float(np.percentile(rel, 95))


@pytest.mark.parametrize("ratio", [3.0, 10.0])
def test_length_from_an_oblique_plane_source(ratio: float) -> None:
    """The arc length is carried by the transport equation V . grad S = |H V|
    on the arrival's stencil. With a metric oblique to the source the
    characteristics are not along grad u; the length must still be exact, as
    it is for any flat front."""
    med, p95 = oblique_plane_length_error(ratio)
    assert med < 0.01
    assert p95 < 0.05


def ball_length_error(n: int, ratio: float) -> tuple[float, float]:
    """Arc length from a ball of source voxels (radius 6, in a box of side 40
    with n points per axis) under a constant metric. Geodesics are straight,
    so the exact length at x is the Euclidean distance to the source voxel
    nearest x in the metric. Returns the median and 95th percentile of the
    relative error at 3-12 from the ball's surface."""
    m = rotated_metric(ratio)
    h = 40.0 / (n - 1)
    g = grid(n, (0, 0, 0)) * h
    r = np.linalg.norm(g - 20.0, axis=-1)
    ball = r <= 6.0
    f = march(m, np.where(ball, 0, -1).astype(np.int64), spacing=(h, h, h))
    chol = np.linalg.cholesky(m)  # metric distance = |chol^T (x - y)|
    _, idx = cKDTree(g[ball] @ chol).query(g.reshape(-1, 3) @ chol)
    exact = np.linalg.norm(g.reshape(-1, 3) - g[ball][idx], axis=-1).reshape(r.shape)
    keep = (r > 9.0) & (r < 18.0)
    rel = np.abs(f.length[keep] - exact[keep]) / exact[keep]
    return float(np.median(rel)), float(np.percentile(rel, 95))


def test_length_from_a_curved_source_converges() -> None:
    """On curved fronts under a strongly anisotropic metric the arc length is
    only first order (about 35 % off at 41^3), but it must converge: both the
    median and the worst errors fall when the grid is refined. An earlier,
    inconsistent discretization of the transport left the 95th percentile at
    about 95 % whatever the resolution."""
    coarse = ball_length_error(41, 10.0)
    fine = ball_length_error(81, 10.0)
    assert fine[0] < 0.75 * coarse[0]
    assert fine[1] < 0.8 * coarse[1]


def test_anisotropic_spacing() -> None:
    """Voxel sizes differ by axis: arrival is the physical distance."""
    n, c = 41, (20, 20, 20)
    h = np.array([0.16028, 0.16, 0.2])
    f = march(np.eye(3), point_seed(n, c), spacing=h)
    ex = np.linalg.norm(grid(n, c, h), axis=-1)
    far = ex > 1.0
    assert np.median(np.abs(f.arrival[far] - ex[far]) / ex[far]) < 0.06


def manufactured(n: int, a: float = 0.12) -> tuple[float, float]:
    """Median relative error and median gradient angle (degrees) for the metric
    M = Dphi^T Dphi, phi(x) = x + a/k (sin k y, sin k z, sin k x), on a box of
    side 40 with n points per axis and a point source at its centre. The exact
    arrival is |phi(x) - phi(s)| where the straight segment in phi's image
    stays inside it; phi bends mildly, so compare in a shell round the source."""
    c = (n // 2,) * 3
    h = 40.0 / (n - 1)
    x = grid(n, (0, 0, 0)) * h
    k = np.pi / 40.0
    phi = x + (a / k) * np.stack(
        [np.sin(k * x[..., 1]), np.sin(k * x[..., 2]), np.sin(k * x[..., 0])], -1
    )
    jac = np.zeros((n, n, n, 3, 3))
    jac[..., 0, 0] = jac[..., 1, 1] = jac[..., 2, 2] = 1.0
    jac[..., 0, 1] = a * np.cos(k * x[..., 1])
    jac[..., 1, 2] = a * np.cos(k * x[..., 2])
    jac[..., 2, 0] = a * np.cos(k * x[..., 0])
    m = np.einsum("...ki,...kj->...ij", jac, jac)
    f = march(m, point_seed(n, c), spacing=(h, h, h))
    ex = np.linalg.norm(phi - phi[c], axis=-1)
    r = np.linalg.norm(x - x[c], axis=-1)
    keep = (r > 5) & (r < 15)
    rel = np.abs(f.arrival[keep] - ex[keep]) / ex[keep]
    return float(np.median(rel)), float(np.median(gradient_angles(f.arrival, ex, keep)))


def test_manufactured_metric_converges() -> None:
    """A smooth, varying metric with an exact solution: first-order accuracy
    from a point source (the source's singularity dominates), and both the
    value and the direction errors fall when the grid is refined."""
    coarse = manufactured(41)
    fine = manufactured(81)
    assert coarse[0] < 0.10 and coarse[1] < 8.0
    assert fine[0] < 0.75 * coarse[0]
    assert fine[1] < 0.8 * coarse[1]


# ---------------------------------------------------------------- labels


def test_two_sources_split_at_the_bisector() -> None:
    n = 41
    lab = -np.ones((n, n, n), np.int64)
    a, b = (10, 20, 20), (30, 22, 18)
    lab[a] = 0
    lab[b] = 1
    m = rotated_metric(3.0)
    f = march(m, lab)
    ga, gb = grid(n, a), grid(n, b)
    da = np.sqrt(np.einsum("...i,ij,...j->...", ga, m, ga))
    db = np.sqrt(np.einsum("...i,ij,...j->...", gb, m, gb))
    clear = np.abs(da - db) > 0.15 * np.minimum(da, db) + 2.0
    expect = np.where(da < db, 0, 1)
    assert np.mean(f.label[clear] == expect[clear]) > 0.999
    # every voxel reached, and the competing label recorded near the meeting
    assert np.all(f.label >= 0)
    near = np.abs(da - db) < 1.0
    assert np.mean(f.second_label[near] >= 0) > 0.9


def test_label_regions_are_connected() -> None:
    rng = np.random.default_rng(1)
    n = 31
    lab = -np.ones((n, n, n), np.int64)
    pts = rng.integers(3, n - 3, size=(6, 3))
    for i, p in enumerate(pts):
        lab[tuple(p)] = i
    f = march(rotated_metric(3.0), lab)
    for i, p in enumerate(pts):
        comp, _ = ndi.label(f.label == i, structure=np.ones((3, 3, 3)))
        assert comp[tuple(p)] > 0
        assert comp.max() == 1


# ---------------------------------------------------------------- invariants


def test_acceptance_order_is_by_arrival() -> None:
    f = march(rotated_metric(10.0), point_seed(31))
    reached = f.order >= 0
    by_order = f.arrival[reached][np.argsort(f.order[reached])]
    assert np.all(np.diff(by_order) >= -1e-12)


def test_mirror_symmetry() -> None:
    """A problem symmetric under x -> -x gives a mirrored arrival."""
    n = 31
    m = riemann_from_direction(np.array([0.0, 1.0, 0.3]), 4.0)  # even in x
    f = march(m, point_seed(n))
    assert np.allclose(f.arrival, f.arrival[::-1], rtol=1e-9, atol=1e-9)


def test_wall_is_not_jumped() -> None:
    """A one-voxel wall across the domain stops the front, even with stencils
    several voxels long; a hole in it lets the front through, round it."""
    n = 31
    mask = np.ones((n, n, n), bool)
    mask[:, 15, :] = False
    lab = -np.ones((n, n, n), np.int64)
    lab[15, 5, 15] = 0
    m = riemann_from_direction(np.array([0.3, 1.0, 0.2]), 10.0)
    f = march(m, lab, mask=mask)
    assert np.all(np.isinf(f.arrival[:, 16:, :]))
    assert np.all(f.label[:, 16:, :] == -1)
    assert np.all(np.isnan(f.length[:, 16:, :]))
    mask[3:6, 15, 3:6] = True  # a hole far from the source
    f = march(np.eye(3), lab, mask=mask)
    behind = (15, 20, 15)
    direct = float(np.linalg.norm(np.array(behind) - np.array([15, 5, 15])))
    assert np.isfinite(f.arrival[behind])
    assert f.arrival[behind] > direct + 5.0


def test_oblique_thin_wall_is_not_jumped() -> None:
    """A wall one voxel thick on the diagonal plane x = y is connected only
    through voxel edges; diagonal stencil steps must not pass between them."""
    n = 31
    i = np.arange(n)
    mask = np.ones((n, n, n), bool)
    mask[i, i, :] = False
    lab = -np.ones((n, n, n), np.int64)
    lab[20, 10, 15] = 0  # on the x > y side
    m = riemann_from_direction(np.array([1.0, -1.0, 0.2]), 5.0)
    f = march(m, lab, mask=mask)
    x, y = np.meshgrid(i, i, indexing="ij")
    other = np.broadcast_to((x < y)[:, :, None], (n, n, n))
    assert np.all(np.isinf(f.arrival[other]))
    assert np.all(
        np.isfinite(f.arrival[np.broadcast_to((x > y)[:, :, None], (n, n, n))])
    )


def test_input_checks() -> None:
    lab = point_seed(5)
    with pytest.raises(ValueError):
        march(np.eye(3), lab, mask=np.zeros((5, 5, 5), bool))
    with pytest.raises(ValueError):
        march(np.eye(3), lab.astype(float))
    with pytest.raises(ValueError):
        march(np.eye(2), lab)
    with pytest.raises(ValueError):
        march(np.eye(3), lab, spacing=(1.0, 0.0, 1.0))
    bad = np.broadcast_to(np.eye(3), (5, 5, 5, 3, 3)).copy()
    bad[2, 2, 3] = np.nan
    with pytest.raises(ValueError):
        march(bad, lab)


# ---------------------------------------------------------------- weights


def metric_coords(m: Float) -> Float:
    """C with |(x - y) @ C| the metric distance under the constant metric m."""
    out: Float = np.linalg.cholesky(m)
    return out


def two_seeds(n: int, a: tuple[int, int, int], b: tuple[int, int, int]) -> Int:
    lab = -np.ones((n, n, n), np.int64)
    lab[a] = 0
    lab[b] = 1
    return lab


def test_single_source_start_and_speed_are_exact_on_a_plane() -> None:
    """A plane source with start t0 and speed c: arrival t0 + x / c, exactly,
    and the length x, whatever the speed."""
    n = 21
    lab = -np.ones((n, n, n), np.int64)
    lab[0] = 0
    m = riemann_from_direction(np.array([1.0, 0.0, 0.0]), 5.0)
    f = march(m, lab, start_time=[-1.5], speed=[0.4])
    x = np.arange(n)[:, None, None] * np.ones((n, n, n))
    assert np.allclose(f.arrival, -1.5 + x / 0.4, atol=1e-12)
    assert np.allclose(f.length, x, atol=1e-12)


@pytest.mark.parametrize("ratio", [1.0, 10.0])
def test_single_source_weights_scale_the_arrival(ratio: float) -> None:
    """From a point source under any metric, start t0 and speed c give
    t0 + (the unweighted arrival) / c, and the same arc length and labels:
    the transport of the length is invariant under the speed."""
    n = 31
    m = rotated_metric(ratio)
    plain = march(m, point_seed(n))
    w = march(m, point_seed(n), start_time=[2.5], speed=[0.3])
    assert np.allclose(w.arrival, 2.5 + plain.arrival / 0.3, rtol=1e-10, atol=1e-10)
    assert np.allclose(w.length, plain.length, rtol=1e-10, atol=1e-12)
    assert np.array_equal(w.label, plain.label)


@pytest.mark.parametrize("ratio", [1.0, 3.0])
def test_start_times_give_hyperboloid_boundaries(ratio: float) -> None:
    """Additively weighted sources: arrival min_i (t_i + d_i), and the regions
    meet on the hyperboloid sheet d_0 - d_1 = t_1 - t_0. Additively weighted
    regions are star-shaped from their sources, so crystal growth and the
    pure weighted Voronoi partition agree."""
    n = 41
    a, b = (10, 20, 20), (30, 21, 19)
    t = np.array([0.0, 6.0])
    m = rotated_metric(ratio)
    f = march(m, two_seeds(n, a, b), start_time=t)
    ga, gb = grid(n, a), grid(n, b)
    da = np.sqrt(np.einsum("...i,ij,...j->...", ga, m, ga)) + t[0]
    db = np.sqrt(np.einsum("...i,ij,...j->...", gb, m, gb)) + t[1]
    exact = np.minimum(da, db)
    far = (np.linalg.norm(ga, axis=-1) > 5) & (np.linalg.norm(gb, axis=-1) > 5)
    rel = np.abs(f.arrival[far] - exact[far]) / exact[far]
    assert np.median(rel) < 0.08
    clear = np.abs(da - db) > 0.15 * np.minimum(da, db) + 2.0
    expect = np.where(da < db, 0, 1)
    assert np.mean(f.label[clear] == expect[clear]) > 0.999
    # the labels' boundary lies on the hyperboloid: at voxels with a
    # differently labelled axis neighbour, |d_0 - d_1| is small
    edge = np.zeros((n, n, n), bool)
    for ax in range(3):
        d = np.diff(f.label, axis=ax) != 0
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = slice(0, n - 1)
        hi[ax] = slice(1, n)
        edge[tuple(lo)] |= d
        edge[tuple(hi)] |= d
    gap = np.abs(da - db)[edge]
    assert edge.sum() > 100
    assert np.median(gap) < 1.0 * np.sqrt(ratio)
    assert np.percentile(gap, 95) < 0.1 * np.median(exact[edge]) + 2.0 * ratio


def shadow_case(
    ratio: float, a: tuple[int, int, int], b: tuple[int, int, int]
) -> tuple[Float, Float, Float, Float, Int]:
    """A fast source a (speed 1) and a slow one b (speed 1/2), n = 41, under
    the constant metric rotated_metric(ratio). Returns, in metric coordinates,
    the distance of each voxel from the Apollonius ball's centre (with the
    ball's radius subtracted), the distance from the ball's centre to the
    segment from a to the voxel (minus the radius), the slow source's pure
    arrival, the shortest fast path that goes round the ball, and the labels."""
    n = 41
    m = rotated_metric(ratio)
    f = march(m, two_seeds(n, a, b), speed=[1.0, 0.5])
    chol = metric_coords(m)
    y = grid(n, (0, 0, 0)) @ chol
    ya = np.array(a, float) @ chol
    yb = np.array(b, float) @ chol
    k = 0.5
    dist = float(np.linalg.norm(yb - ya))
    c = (yb - k**2 * ya) / (1 - k**2)
    r = k * dist / (1 - k**2)
    rel = y - c
    xr = np.linalg.norm(rel, axis=-1)
    # distance from c to the segment [ya, y]
    seg = y - ya
    s = np.clip(
        np.sum((c - ya) * seg, -1) / np.maximum(np.sum(seg * seg, -1), 1e-12), 0, 1
    )
    to_seg = np.linalg.norm(ya + s[..., None] * seg - c, axis=-1)
    slow = np.linalg.norm(y - yb, axis=-1) / k
    # round the ball: tangent from a, arc, tangent to the voxel
    big_a = float(np.linalg.norm(ya - c))
    cos = np.sum((ya - c) * rel, -1) / (big_a * np.maximum(xr, 1e-12))
    theta = np.arccos(np.clip(cos, -1, 1))
    xo = np.maximum(xr, r)
    around = (
        np.sqrt(big_a**2 - r**2)
        + np.sqrt(xo**2 - r**2)
        + r * (theta - np.arccos(r / big_a) - np.arccos(r / xo))
    )
    return xr - r, to_seg - r, slow, around, f.label


@pytest.mark.parametrize(("ratio", "a"), [(1.0, (8, 20, 20)), (3.0, (14, 20, 20))])
def test_speeds_give_apollonius_ball_and_a_tail(
    ratio: float, a: tuple[int, int, int]
) -> None:
    """Multiplicatively weighted sources (speeds 1 and 1/2). The slow source's
    pure region is the Apollonius ball |y - b| = |y - a| / 2. Under crystal
    growth the ball lies inside its region; outside the ball, the slow region
    is only in the ball's shadow seen from the fast source (where the fast
    front must go round it), and there it has a tail: wherever the slow
    source's straight arrival beats the shortest fast path round the ball."""
    b = (20, 20, 20)
    out, seg, slow, around, label = shadow_case(ratio, a, b)
    margin = 1.5 * ratio
    inside = out < -margin
    assert inside.sum() > 50
    assert np.all(label[inside] == 1)
    lit = (out > margin) & (seg > margin)  # outside the ball and its shadow
    assert np.all(label[lit] == 0)
    # the fast front's true arrival is at least `around` (it cannot cross the
    # ball) and first-order errors are about 5 %
    tail = (out > 1.0) & (seg < 0) & (slow < 0.95 * around)
    assert tail.sum() > 10
    assert np.all(label[tail] == 1)
    # the slow region outside the ball is all in the shadow
    assert np.all(seg[(label == 1) & (out > margin)] < margin)


def test_regions_are_connected_under_random_weights() -> None:
    """Crystal growth: every label's region is connected and contains its
    source, or is empty (a source reached by another front before its own
    start). The pure weighted Voronoi regions need not be connected."""
    rng = np.random.default_rng(2)
    n = 31
    m = rotated_metric(3.0)
    for trial in range(4):
        k = 8
        lab = -np.ones((n, n, n), np.int64)
        pts = rng.integers(3, n - 3, size=(k, 3))
        for i, p in enumerate(pts):
            lab[tuple(p)] = i
        t = rng.uniform(-3, 3, k)
        c = rng.uniform(0.4, 2.5, k)
        f = march(m, lab, start_time=t, speed=c)
        assert np.all(f.label >= 0)
        for i, p in enumerate(pts):
            own = f.label == i
            if not own.any():
                assert f.label[tuple(p)] != i
                continue
            comp, ncomp = ndi.label(own, structure=np.ones((3, 3, 3)))
            assert ncomp == 1, (trial, i)
            assert comp[tuple(p)] == 1


def test_a_late_source_is_swallowed() -> None:
    """A source whose start time comes after another front has reached it
    gets no voxels."""
    n = 21
    a, b = (5, 10, 10), (15, 10, 10)
    f = march(np.eye(3), two_seeds(n, a, b), start_time=[0.0, 12.0])
    assert np.all(f.label == 0)
    assert f.arrival[b] == pytest.approx(10.0)


def test_weight_input_checks() -> None:
    lab = point_seed(5)
    lab[0, 0, 0] = 2
    good = np.ones(3)
    for bad in (
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 1.0],
        [1.0, np.inf, 1.0],
        [np.nan, 1, 1],
        [1.0, 1e-160, 1.0],  # 1 / speed^2 overflows
    ):
        with pytest.raises(ValueError, match="speed"):
            march(np.eye(3), lab, speed=bad)
    for bad in ([0.0, np.nan, 0.0], [0.0, 0.0, np.inf]):
        with pytest.raises(ValueError, match="start_time"):
            march(np.eye(3), lab, start_time=bad)
    with pytest.raises(ValueError, match="speed"):
        march(np.eye(3), lab, speed=[1.0, 1.0])  # labels run to 2
    with pytest.raises(ValueError, match="start_time"):
        march(np.eye(3), lab, start_time=np.zeros((3, 1)))
    f = march(np.eye(3), lab, start_time=np.zeros(5), speed=good)  # longer is fine
    assert np.all(f.label >= 0)


def test_weights_with_spacing_and_a_wall() -> None:
    """Speeds act in physical units (anisotropic voxels), and a fast front
    does not jump a wall: behind a one-voxel wall with a hole, a slow source
    owns the cortex near it until the fast front has gone round."""
    n = 21
    h = np.array([0.2, 0.16, 0.16])
    lab = -np.ones((n, n, n), np.int64)
    lab[0] = 0
    f = march(np.eye(3), lab, spacing=h, start_time=[1.0], speed=[2.5])
    x = np.arange(n)[:, None, None] * h[0] * np.ones((n, n, n))
    assert np.allclose(f.arrival, 1.0 + x / 2.5, atol=1e-12)
    assert np.allclose(f.length, x, atol=1e-12)
    n = 31
    mask = np.ones((n, n, n), bool)
    mask[:, 15, :] = False
    mask[2:4, 15, 2:4] = True  # a hole far from both sources
    lab = -np.ones((n, n, n), np.int64)
    lab[15, 5, 15] = 0  # fast, in front of the wall
    lab[15, 25, 15] = 1  # slow, behind it
    m = riemann_from_direction(np.array([0.3, 1.0, 0.2]), 5.0)
    f = march(m, lab, mask=mask, speed=[10.0, 1.0])
    assert np.all(f.label[:, :15, :][mask[:, :15, :]] == 0)
    # straight through the wall, the fast front would win at (15, 20, 15)
    assert f.label[15, 20, 15] == 1
    comp, ncomp = ndi.label(f.label == 1, structure=np.ones((3, 3, 3)))
    assert ncomp == 1
