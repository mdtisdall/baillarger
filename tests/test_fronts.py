"""Correctness tests for baillarger.fronts: exact eikonal solutions, labels,
arc length, and invariants. Grids are small (at most 61^3)."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy import ndimage as ndi

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
