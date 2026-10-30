"""Tests for baillarger.mesh_torch: each PyTorch device (CPU, and MPS or
CUDA when present) against the numpy reference, for marching tetrahedra
and the smoothing of the white mesh."""

import numpy as np
import numpy.typing as npt
import pytest

from baillarger import backend
from baillarger import mesh_smoothing as S
from baillarger.mesh import marching_tetrahedra_blocks, topology
from baillarger.mesh_torch import MeshOps

DEVICES = [d for d in backend.available() if d != "numpy"]
EPS32 = float(np.finfo(np.float32).eps)


def tolerance(x: npt.NDArray[np.floating], ulps: float) -> float:
    """ulps float32 units at the largest coordinate of x: rounding grows
    with the size of the coordinates, so tolerances scale with it."""
    return ulps * EPS32 * float(np.abs(x).max())


def bumpy_ball() -> npt.NDArray[np.uint8]:
    x = np.indices((48, 44, 40)).astype(float)
    r = np.sqrt(((x - np.array([23.3, 21.6, 19.8])[:, None, None, None]) ** 2).sum(0))
    out: npt.NDArray[np.uint8] = (
        r + 1.5 * np.sin(x[0] / 2.1) * np.cos(x[2] / 1.7) < 15
    ).view(np.uint8)
    return out


@pytest.mark.parametrize("name", DEVICES)
def test_marching_on_a_device_matches_numpy(name: str) -> None:
    dev = backend.device(name)
    rng = np.random.default_rng(5)
    x = np.indices((30, 30, 30)).astype(float)
    F = np.sqrt(((x - 14.6) ** 2).sum(0)) + 0.8 * rng.standard_normal(x.shape[1:])
    for G, lv in ((bumpy_ball(), 0.5), (F, 9.0), (np.round(F), 9.0)):
        v0, f0, b0 = marching_tetrahedra_blocks(G, lv, 8, np.float32)
        v1, f1, b1 = marching_tetrahedra_blocks(G, lv, 8, np.float32, dev)
        # the same triangles, in the same order, and the same blocks
        assert np.array_equal(f1, f0)
        assert np.array_equal(b1.origin, b0.origin)
        assert np.array_equal(b1.vertex_stop, b0.vertex_stop)
        assert np.array_equal(b1.face_stop, b0.face_stop)
        assert v1.dtype == np.float32
        # the CPU does numpy's arithmetic; a GPU may round divisions
        # differently (a few units)
        assert np.abs(v1 - v0).max() <= (0 if name == "cpu" else tolerance(v0, 4))


def test_a_device_works_in_float32() -> None:
    with pytest.raises(ValueError, match="float32"):
        marching_tetrahedra_blocks(
            bumpy_ball(), 0.5, 8, np.float64, backend.device("cpu")
        )


@pytest.mark.parametrize("name", DEVICES)
@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_smoothing_on_a_device_matches_numpy(name: str, offset: float) -> None:
    """Only the order of the sums differs from numpy: float32 rounding, which
    grows with the coordinates (offset: coordinates near 1,000, as in the
    hemisphere). A normal's tolerance is the position's over an edge length.
    Normals are compared where they are well conditioned (the
    averaged normal keeps at least 0.9 of its length): elsewhere a rounding
    difference can turn them far (left to the column stage)."""
    v0, f, _ = marching_tetrahedra_blocks(bumpy_ball(), 0.5, 8, np.float32)
    v0 = (v0 + np.float32(offset)).astype(np.float32)
    f = f[:, [0, 2, 1]].copy()
    ref = S.twice(S.relax(v0, f), f, 2.0)
    nref = S.smooth_normals(ref, f)
    ops = MeshOps(f, len(v0), backend.device(name).torch)  # type: ignore[arg-type]
    v = ops.smooth(v0, 2.0)
    n = ops.normals(v)
    assert v.dtype == n.dtype == np.float32
    assert np.abs(v - ref).max() <= tolerance(ref, 64)
    A, deg = S.umbrella(nref.astype(np.float64), f)
    nr = nref.astype(np.float64)
    cond = np.linalg.norm(0.5 * nr + 0.5 * (A @ nr) / deg[:, None], axis=1)
    good = cond >= 0.9
    assert good.mean() > 0.99
    # a normal is turned by the position error over an edge
    E = S.edges(f)
    edge = float(np.linalg.norm(ref[E[:, 0]] - ref[E[:, 1]], axis=1).mean())
    assert np.abs(n - nref)[good].max() <= tolerance(ref, 64) / edge
    # and the same every run
    assert np.array_equal(ops.smooth(v0, 2.0), v)


@pytest.mark.parametrize("name", DEVICES)
def test_smoothing_keeps_an_open_sheet_s_boundary(name: str) -> None:
    """An open sheet: its boundary vertices stay put, as in numpy."""
    i, j = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    v0 = np.c_[i.ravel(), j.ravel(), 0.3 * np.sin(i.ravel())].astype(np.float32)
    q = (i[:-1, :-1] * 12 + j[:-1, :-1]).ravel()
    f = np.r_[np.c_[q, q + 12, q + 1], np.c_[q + 1, q + 12, q + 13]].astype(np.int64)
    assert topology(f, len(v0))["boundary_loops"] == 1
    fixed = S.boundary_vertices(f, len(v0))
    ref = S.twice(S.relax(v0, f), f, 1.0)
    v = MeshOps(f, len(v0), backend.device(name).torch).smooth(v0, 1.0)  # type: ignore[arg-type]
    assert np.array_equal(v[fixed], v0[fixed])
    assert np.abs(v - ref).max() <= tolerance(ref, 64)
