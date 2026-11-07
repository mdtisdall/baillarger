"""Tests of bent_texture: the eigenvalues, the fibre measure, the laminar
coherence, the sampling at points, the split of the columns into banks and
the first outside on a ray.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_bent_texture.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import bent_texture as BT  # noqa: E402

from baillarger import orient as O  # noqa: E402

type F64 = npt.NDArray[np.float64]

#: the voxel size of the synthetic images, mm (isotropic)
VOX = 0.16
#: the side of a synthetic image, voxels
N = 56
#: the thermal sigma of the synthetic images
SIGMA = 5.0


@dataclass(frozen=True)
class Noise:
    """A simple noise model with the fields of orient.NoiseModel."""

    sigma: F64
    R: F64
    gain: float
    rest: F64


def _noise() -> Noise:
    return Noise(
        sigma=np.full(1, SIGMA),
        R=np.eye(1),
        gain=O.gradient_gain(O.SD),
        rest=np.zeros((1, 1)),
    )


def _unit(v: npt.ArrayLike) -> F64:
    a = np.asarray(v, np.float64)
    out: F64 = a / np.linalg.norm(a)
    return out


def _frame(t: F64) -> tuple[F64, F64]:
    """Two unit vectors perpendicular to t and to each other."""
    a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(t, a))
    return u, np.cross(t, u)


def _grid() -> F64:
    """(N, N, N, 3) the voxel coordinates, centred on the image."""
    i = np.arange(N, dtype=np.float64) - (N - 1) / 2
    out: F64 = np.stack(np.meshgrid(i, i, i, indexing="ij"), -1)
    return out


def _texture_of(image: F64, seed: int) -> BT.Texture:
    """orient_arrays on one echo with white noise, the whole image as
    cortex; the texture at the voxels that hold the whole reach (NaN
    elsewhere)."""
    rng = np.random.default_rng(seed)
    img = image + rng.normal(0.0, SIGMA, image.shape)
    mask = np.ones(image.shape, bool)
    vs = np.full(3, VOX)
    J = O.orient_arrays(img[None], mask, _noise(), vs)["tensor"]
    J = np.where(BT.valid_mask(image.shape, O.orient_reach(vs, O.SIGMA)), J, np.nan)
    return BT.texture(J)


def _lines(t: F64) -> F64:
    """An image that is a function of the distance to a square lattice of
    parallel lines along t (period 6 voxels, a Gaussian of 1.2 voxels
    round each line)."""
    u, w = _frame(t)
    x = _grid()
    lam = 6.0
    a = (x @ u + lam / 2) % lam - lam / 2
    b = (x @ w + lam / 2) % lam - lam / 2
    d2 = a * a + b * b
    out: F64 = 1000.0 + 300.0 * np.exp(-d2 / (2 * 1.2**2))
    return out


def _planes(n: F64) -> F64:
    """An image of parallel planes normal to n (period 8 voxels, contrast
    3 %). At a contrast of 30 % the log makes harmonics. Near the grid's
    Nyquist frequency, the central differences of an oblique harmonic do not
    point along n, and the median of f was 0.35 (0.14 at 3 %)."""
    x = _grid()
    out: F64 = 1000.0 + 30.0 * np.cos(2 * np.pi * (x @ n) / 8.0)
    return out


def _noise_only() -> F64:
    return np.full((N, N, N), 1000.0)


def _angle(v: F64, t: F64) -> F64:
    """The angle in degrees between the axes v (..., 3) and t (3,)."""
    out: F64 = np.degrees(np.arccos(np.clip(np.abs(v @ t), 0.0, 1.0)))
    return out


T_LINE = _unit([1.0, 2.0, 3.0])
N_PLANE = _unit([3.0, -1.0, 2.0])


def test_lines_give_fibre_measure_near_1_along_the_lines() -> None:
    """Thresholds: the median of f > 0.9, its 5th percentile > 0.75; the
    95th percentile of the angle between the fibre direction and the lines
    < 5 degrees."""
    tx = _texture_of(_lines(T_LINE), seed=1)
    ok = np.isfinite(tx.f)
    assert ok.sum() > 1000
    f = tx.f[ok]
    assert np.median(f) > 0.9
    assert np.percentile(f, 5) > 0.75
    assert np.percentile(_angle(tx.fibre[ok], T_LINE), 95) < 5.0


def test_planes_give_fibre_measure_of_noise_and_coherence_near_1() -> None:
    """With noise, l2 and l3 of a plane image come from the noise alone, so
    f has the distribution that the noise alone gives, not exactly 0.
    Thresholds: the median of f is at most the noise-only median + 0.05;
    the median of c > 0.95; the 95th percentile of the angle between the
    leading eigenvector and the planes' normal < 5 degrees."""
    tp = _texture_of(_planes(N_PLANE), seed=2)
    tn = _texture_of(_noise_only(), seed=3)
    ok = np.isfinite(tp.f)
    assert ok.sum() > 1000
    assert np.median(tp.f[ok]) <= np.nanmedian(tn.f) + 0.05
    assert np.median(tp.c[ok]) > 0.95
    assert np.percentile(_angle(tp.leading[ok], N_PLANE), 95) < 5.0


def test_lines_are_not_laminar() -> None:
    """For lines, l1 is about l2: the median of c < 0.5."""
    tx = _texture_of(_lines(T_LINE), seed=4)
    assert np.nanmedian(tx.c) < 0.5


def test_exact_tensors() -> None:
    """Tensors built from their eigenvalues give f and c exactly."""
    t = T_LINE
    u, w = _frame(t)
    # a fibre along t: l1 = 4, l2 = 3, l3 = 1
    M = 4 * np.outer(u, u) + 3 * np.outer(w, w) + 1 * np.outer(t, t)
    # a lamina normal to u: l1 = 5, l2 = l3 = 0.5
    P = 5 * np.outer(u, u) + 0.5 * (np.outer(w, w) + np.outer(t, t))
    J = np.stack([[M[i, j], P[i, j]] for i, j in O.PAIRS])  # (6, 2)
    tx = BT.texture(J)
    np.testing.assert_allclose(tx.values[:, 0], [4, 3, 1], atol=1e-12)
    np.testing.assert_allclose(tx.f, [0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(tx.c, [1 / 7, 9 / 11], atol=1e-12)
    np.testing.assert_allclose(tx.strength, [4.0, 1.0], atol=1e-12)
    assert _angle(tx.fibre[0], t) < 1e-6
    assert _angle(tx.leading[1], u) < 1e-6


def _random_tensors(shape: tuple[int, ...], seed: int) -> F64:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(*shape, 3, 4))
    m = np.einsum("...ik,...jk->...ij", a, a)
    J = np.stack([m[..., i, j] for i, j in O.PAIRS])
    J[:, 0, 0, 0] = np.nan
    J[2, 1, 0, 0] = np.nan
    return J


def test_leading_equals_orient_leading() -> None:
    J = _random_tensors((4, 5, 6), seed=5)
    tx = BT.texture(J)
    v, c = O.leading(J)
    ok = np.isfinite(c)
    assert not ok[0, 0, 0] and not ok[1, 0, 0]
    assert np.array_equal(ok, np.isfinite(tx.c))
    dot = np.abs(np.sum(tx.leading[ok] * v[ok], -1))
    np.testing.assert_allclose(dot, 1.0, atol=1e-10)
    np.testing.assert_allclose(tx.c[ok], c[ok], atol=1e-12)
    assert np.all(np.isnan(tx.f[~ok]))
    assert np.all(np.isnan(tx.fibre[~ok]))


def test_valid_mask() -> None:
    m = BT.valid_mask((10, 12, 14), [2, 3, 4])
    assert m.sum() == 6 * 6 * 6
    assert m[2, 3, 4] and m[7, 8, 9]
    assert not m[1, 3, 4] and not m[8, 8, 9]


def test_sample_trilinear_interpolates_the_tensor() -> None:
    J = _random_tensors((4, 4, 4), seed=6)
    J[:, 0, 0, 0] = 1.0  # no NaN near the points below
    J[2, 1, 0, 0] = 1.0
    p = np.array([[1.0, 2.0, 1.0], [1.5, 2.0, 1.0], [2.5, 2.5, 2.5]])
    got = BT.sample_tensor(J, p, order=1)
    np.testing.assert_allclose(got[:, 0], J[:, 1, 2, 1], atol=1e-12)
    np.testing.assert_allclose(got[:, 1], (J[:, 1, 2, 1] + J[:, 2, 2, 1]) / 2)
    cube = J[:, 2:4, 2:4, 2:4].reshape(6, -1).mean(1)
    np.testing.assert_allclose(got[:, 2], cube, atol=1e-12)
    near = BT.sample_tensor(J, np.array([[1.4, 2.4, 0.6]]), order=0)
    np.testing.assert_allclose(near[:, 0], J[:, 1, 2, 1])
    # a NaN neighbour makes a trilinear value NaN
    J[:, 3, 3, 3] = np.nan
    assert np.all(np.isnan(BT.sample_tensor(J, p[2:], order=1)))


def test_to_scanner() -> None:
    A = np.diag([0.2, 0.1, 0.4, 1.0])
    A[:3, 3] = [5, 6, 7]
    vs = np.array([0.2, 0.1, 0.4])
    v = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    np.testing.assert_allclose(BT.to_scanner(v, A, vs), v)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    B = np.eye(4)
    B[:3, :3] = R @ np.diag(vs)
    np.testing.assert_allclose(BT.to_scanner(v, B, vs), v @ R.T, atol=1e-12)


def test_split_banks() -> None:
    """Columns along a fundus axis x: normals tilted by +-60 degrees toward
    +-z are banks, normals within 10 degrees of the vertex normal are
    fundus."""
    rng = np.random.default_rng(7)
    k = 200
    ang = np.concatenate(
        [rng.uniform(40, 80, k), rng.uniform(-80, -40, k), rng.uniform(-10, 10, k)]
    )
    a = np.radians(ang)
    nrm = np.stack([np.zeros_like(a), np.cos(a), np.sin(a)], -1)
    nrm[:, 0] = rng.normal(0, 0.05, len(a))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm[0] = [0.0, 1.0, 0.0]  # the vertex
    part, u = BT.split_banks(nrm, vertex_index=0)
    assert abs(abs(u[2]) - 1.0) < 0.05
    s = np.sign(u[2])
    assert np.all(part[1:k] == (BT.BANK_P if s > 0 else BT.BANK_N))
    assert np.all(part[k : 2 * k] == (BT.BANK_N if s > 0 else BT.BANK_P))
    assert np.all(part[2 * k :] == BT.FUNDUS)


def test_first_outside() -> None:
    start = np.array([0, 3, 5, 7])
    s0 = np.array([-2.0, 0.0, 1.5, -2.0, 0.0, -2.0, -1.0], np.float32)
    s1 = np.array([0.0, 1.5, 8.0, 0.0, 8.0, -1.0, 8.0], np.float32)
    cls = np.array([1, 2, 0, 1, 2, 0, 2], np.uint8)
    got = BT.first_outside(start, s0, s1, cls)
    np.testing.assert_allclose(got[:1], [1.5])
    assert np.isnan(got[1])
    # an outside interval before the root does not count
    assert np.isnan(got[2])


@pytest.mark.parametrize("seed", [8])
def test_noise_image_has_the_model_covariance(seed: int) -> None:
    sig = np.array([10.0, 20.0])
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    level = np.log(np.full((2, 30, 30, 30), 500.0))
    img = BT.noise_image(level, sig, R, np.random.default_rng(seed))
    r = (img - 500.0).reshape(2, -1)
    C = np.cov(r)
    np.testing.assert_allclose(C, np.outer(sig, sig) * R, rtol=0.05)
