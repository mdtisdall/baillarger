"""Tests of bent_sample: trilinear sampling in torch, with gradients.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_bent_sample.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent))
import bent_sample as BS  # noqa: E402

from baillarger import streamlines  # noqa: E402

SHAPE = (7, 8, 9)


def _volume(c: int | None = None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = SHAPE if c is None else (c, *SHAPE)
    return rng.uniform(100.0, 1000.0, shape).astype(np.float32)


def _points(n: int, lo: float, hi: float, seed: int = 1) -> np.ndarray:
    """n random points; each coordinate is a fraction in [lo, hi] of its axis."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (n, 3)) * (np.array(SHAPE, float) - 1.0)


def test_matches_scipy_and_streamlines_inside() -> None:
    vol = _volume()
    pts = _points(500, 0.0, 1.0).astype(np.float32)
    got = BS.trilinear(torch.from_numpy(vol), torch.from_numpy(pts)).numpy()
    ref = map_coordinates(vol.astype(np.float64), pts.T.astype(float), order=1)
    ref2 = streamlines.sample(vol, pts[None])[0]
    # float32 rounding: a few ulp of the volume's scale (1000).
    tol = 1e-5 * float(vol.max())
    assert np.abs(got - ref).max() < tol
    assert np.abs(got - ref2).max() < tol


def test_matches_scipy_outside_nearest() -> None:
    vol = _volume()
    rng = np.random.default_rng(3)
    pts = rng.uniform(-3.0, 11.0, (500, 3)).astype(np.float32)
    got = BS.trilinear(torch.from_numpy(vol), torch.from_numpy(pts)).numpy()
    ref = map_coordinates(
        vol.astype(np.float64), pts.T.astype(float), order=1, mode="nearest"
    )
    assert np.abs(got - ref).max() < 1e-5 * float(vol.max())


def test_integer_points_exact() -> None:
    vol = _volume()
    ii = np.stack(np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij"), axis=-1)
    got = BS.trilinear(torch.from_numpy(vol), torch.from_numpy(ii.astype(np.float32)))
    assert got.shape == SHAPE
    assert torch.equal(got, torch.from_numpy(vol))


def test_axis_order() -> None:
    vol = torch.zeros(SHAPE)
    vol[2, 5, 6] = 1.0
    out = BS.trilinear(vol, torch.tensor([[2.0, 5.0, 6.0], [6.0, 5.0, 2.0]]))
    assert out.tolist() == [1.0, 0.0]


def test_gradient_matches_central_difference() -> None:
    vol = torch.from_numpy(_volume()).double()
    pts = torch.from_numpy(_points(50, 0.02, 0.96)).double()
    # Keep the points off voxel boundaries (distance > 0.05 from integers).
    frac = pts - pts.floor()
    keep = ((frac > 0.05) & (frac < 0.95)).all(dim=1)
    pts = pts[keep].clone().requires_grad_(True)
    assert pts.shape[0] > 5
    BS.trilinear(vol, pts).sum().backward()
    assert pts.grad is not None
    h = 1e-4
    num = torch.zeros_like(pts)
    for k in range(3):
        d = torch.zeros(3, dtype=torch.float64)
        d[k] = h
        num[:, k] = (
            BS.trilinear(vol, pts.detach() + d) - BS.trilinear(vol, pts.detach() - d)
        ) / (2 * h)
    assert torch.allclose(pts.grad, num, rtol=1e-6, atol=1e-6)


def test_gradient_float32_runs() -> None:
    vol = torch.from_numpy(_volume())
    pts = torch.tensor([[1.3, 2.4, 3.6]], requires_grad=True)
    BS.trilinear(vol, pts).sum().backward()
    assert pts.grad is not None and pts.grad.dtype == torch.float32


def test_gradient_zero_outside() -> None:
    vol = torch.from_numpy(_volume())
    pts = torch.tensor([[-2.0, 3.5, 3.5]], requires_grad=True)
    BS.trilinear(vol, pts).sum().backward()
    assert pts.grad is not None and float(pts.grad[0, 0]) == 0.0


def test_channels_equal_single() -> None:
    vol = _volume(3)
    pts = torch.from_numpy(_points(40, 0.0, 1.0).astype(np.float32)).reshape(4, 10, 3)
    both = BS.trilinear(torch.from_numpy(vol), pts)
    assert both.shape == (3, 4, 10)
    for c in range(3):
        one = BS.trilinear(torch.from_numpy(vol[c]), pts)
        assert one.shape == (4, 10)
        assert torch.equal(both[c], one)


def test_inside_mask() -> None:
    pts = torch.tensor(
        [[0.0, 0.0, 0.0], [6.0, 7.0, 8.0], [6.1, 0.0, 0.0], [-0.1, 1, 1]]
    )
    assert BS.inside(SHAPE, pts).tolist() == [True, True, False, False]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")
def test_cpu_and_mps_agree() -> None:
    vol = _volume(3)
    rng = np.random.default_rng(5)
    pts = rng.uniform(-1.0, 9.0, (300, 3)).astype(np.float32)
    v, p = torch.from_numpy(vol), torch.from_numpy(pts).requires_grad_(True)
    a = BS.trilinear(v, p)
    a.sum().backward()
    pm = torch.from_numpy(pts).to("mps").requires_grad_(True)
    b = BS.trilinear(v.to("mps"), pm)
    b.sum().backward()
    assert b.device.type == "mps"
    assert np.abs(a.detach().numpy() - b.detach().cpu().numpy()).max() < 1e-5 * 1000
    assert p.grad is not None and pm.grad is not None
    assert np.abs(p.grad.numpy() - pm.grad.cpu().numpy()).max() < 1e-2
