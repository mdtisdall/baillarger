"""Tests of bent_terms on synthetic volumes in voxels of 0.16 mm (an affine
of diag(1 / 0.16) from mm to voxels), with exact answers where possible.
The gradient tests are in float64 on the cpu.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_bent_terms.py
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bent_terms as BT  # noqa: E402

from baillarger import rays as R  # noqa: E402

VOX = 0.16
MM_TO_VOX = np.diag([1 / VOX, 1 / VOX, 1 / VOX, 1.0])
F64 = torch.float64

# ---------------------------------------------------------------- helpers


def grid_mesh(nx: int, ny: int) -> np.ndarray:
    """The faces of a regular grid of nx * ny vertices (index i * ny + j),
    each square cut by the same diagonal."""
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = (
                i * ny + j,
                (i + 1) * ny + j,
                (i + 1) * ny + j + 1,
                i * ny + j + 1,
            )
            f += [(a, b, c), (a, c, d)]
    return np.asarray(f, np.int64).reshape(-1, 3)


def make(
    roots_vox: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    outside: np.ndarray,
    wm: np.ndarray,
    cortex: np.ndarray,
    echoes: np.ndarray | None = None,
    moving: np.ndarray | None = None,
    mesh_edges: np.ndarray | None = None,
    align: np.ndarray | None = None,
) -> BT.Inputs:
    shape = outside.shape
    if echoes is None:
        echoes = np.full((2, *shape), 100.0)
    return BT.Inputs.from_arrays(
        roots=np.asarray(roots_vox, float) * VOX,
        normals=normals,
        faces=faces,
        mm_to_vox=MM_TO_VOX,
        outside=outside,
        wm=wm,
        cortex=cortex,
        echoes=echoes,
        scales=np.array([0.3, 0.2]),
        moving=moving,
        mesh_edges=mesh_edges,
        align=align,
        device="cpu",
        dtype=F64,
    )


def slab(shape: tuple[int, int, int], top: int, wm_top: int) -> tuple[np.ndarray, ...]:
    """outside for z >= top, WM for z <= wm_top, cortex between (voxel
    indices on the last axis)."""
    z = np.arange(shape[2])[None, None, :] * np.ones(shape)
    outside = z >= top
    wm = z <= wm_top
    return outside, wm, ~outside & ~wm


def levels(inp: BT.Inputs, s_lo: float, tau: np.ndarray, r: np.ndarray) -> BT.Levels:
    return BT.Levels.make(inp, s_lo, tau, r)


def zeros(inp: BT.Inputs) -> torch.Tensor:
    return torch.zeros((inp.n, 2, inp.K), dtype=inp.dtype)


# The rich synthetic box of the gradient tests: a 4 x 4 grid of roots, 0.4 mm
# apart, normals tilted at random by up to 0.03 rad, a WM below the roots, a
# WM island in the cortex on one side, a top that is tilted along x, and two
# layered echoes, and a laminar field that varies in space (NaN in the outside
# medium).


def rich_align(shape: tuple[int, int, int], outside: np.ndarray) -> np.ndarray:
    x, y, z = np.meshgrid(*[np.arange(s, dtype=float) for s in shape], indexing="ij")
    v = np.stack(
        [0.4 * np.sin(x / 5 + z / 9), 0.3 * np.cos(y / 7 - z / 11), np.ones(shape)], -1
    )
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    c = 0.4 + 0.3 * np.sin(z / 6 + x / 8)
    c[outside] = np.nan
    return BT.align_field(v, c, np.eye(4), np.ones(3))


def rich() -> tuple[BT.Inputs, BT.Levels]:
    rng = np.random.default_rng(3)
    shape = (24, 24, 80)
    x, y, z = np.meshgrid(*[np.arange(s, dtype=float) for s in shape], indexing="ij")
    outside = z > 28 + 0.3 * x
    wm = (z <= 10) | ((z >= 18) & (z <= 20) & (x >= 10) & (x <= 16))
    cortex = ~outside & ~wm
    e1 = 200 + 40 * np.sin(2 * np.pi * z / 4.3) + 3 * x + 2 * y
    e2 = 150 + 30 * np.cos(2 * np.pi * z / 6.1) + 2 * x - y
    nx = ny = 4
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    roots = np.stack(
        [6.37 + 2.5 * gi.ravel(), 6.21 + 2.5 * gj.ravel(), np.full(16, 10.43)], 1
    )
    t = rng.uniform(-0.03, 0.03, (16, 2))
    normals = np.c_[t, np.ones(16)]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    moving = np.ones(16, bool)
    moving[0] = False
    inp = make(
        roots,
        normals,
        grid_mesh(nx, ny),
        outside,
        wm,
        cortex,
        np.stack([e1, e2]),
        moving,
        align=rich_align(shape, outside),
    )
    tau = np.arange(0.0, 6.4, 0.08)
    r = 0.03 * rng.standard_normal((16, len(tau)))
    return inp, levels(inp, -0.4, tau, r)


def check_grad(fn, p0: torch.Tensor, seed: int = 0, h: float = 1e-6) -> None:
    """The gradient of fn at p0 against a central finite difference along
    three random directions. Tolerance: 1e-6 relative to the larger of 1 and
    the size of the derivative."""
    p = p0.clone().requires_grad_(True)
    v = fn(p)
    assert torch.isfinite(v)
    (g,) = torch.autograd.grad(v, p)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(3):
        d = torch.randn(p0.shape, generator=gen, dtype=p0.dtype)
        with torch.no_grad():
            fd = (fn(p0 + h * d) - fn(p0 - h * d)) / (2 * h)
        an = (g * d).sum()
        assert abs(float(fd - an)) <= 1e-6 * max(1.0, abs(float(an))), (
            float(fd),
            float(an),
        )


# ---------------------------------------------------------------- the path


def test_frame_is_orthonormal() -> None:
    rng = np.random.default_rng(0)
    n = rng.standard_normal((50, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[:3] = np.eye(3)
    e1, e2 = BT.frame(n)
    for a, b in ((n, e1), (n, e2), (e1, e2)):
        assert np.abs((a * b).sum(1)).max() < 1e-12
    assert np.allclose(np.linalg.norm(e1, axis=1), 1, atol=1e-12)
    assert np.allclose(np.cross(n, e1), e2, atol=1e-12)


def test_basis_value_zero_at_root() -> None:
    B0, B1, B2 = BT.basis()
    assert B0.shape == (BT.NS, BT.K) and B1.shape == B0.shape and B2.shape == B0.shape
    assert np.all(B0[0] == 0)
    # the slope at the root is free
    assert np.abs(B1[0]).max() > 0.1


def test_zero_shear_gives_the_straight_samples() -> None:
    rng = np.random.default_rng(1)
    n = 7
    roots = rng.uniform(30, 31, (n, 3)).astype(np.float32)
    nor = rng.standard_normal((n, 3)).astype(np.float32)
    nor /= np.linalg.norm(nor, axis=1, keepdims=True)
    shape = (16, 16, 16)
    o, w, c = slab(shape, 100, -1)
    inp = BT.Inputs.from_arrays(
        roots=roots,
        normals=nor,
        faces=np.zeros((0, 3), np.int64),
        mm_to_vox=MM_TO_VOX,
        outside=o,
        wm=w,
        cortex=c,
        echoes=np.ones((2, *shape)),
        scales=np.ones(2),
        device="cpu",
        dtype=F64,
    )
    p = BT.paths(zeros(inp), inp)
    k_out = np.full(n, BT.NS - 1, np.int32)
    _, s, pos = R._samples(
        roots.astype(np.float64),
        nor.astype(np.float64),
        np.zeros(n, np.int32),
        k_out,
        R.STEP,
    )
    x = p.x.numpy()
    # float64: the sampler's formula in float64, exactly
    ref = (
        roots.astype(np.float64)[:, None]
        + (np.arange(BT.NS) * R.STEP)[None, :, None] * nor.astype(np.float64)[:, None]
    )
    assert np.array_equal(x, ref)
    # and the sampler's own float32 positions, exactly
    assert np.array_equal(x.astype(np.float32).reshape(-1, 3), pos)
    assert np.array_equal(p.s.numpy().astype(np.float32), s[: BT.NS])
    # a random shear keeps the root
    q = BT.paths(0.1 * torch.randn(inp.n, 2, inp.K, dtype=F64), inp)
    assert torch.equal(q.x[:, 0], torch.as_tensor(roots, dtype=F64))
    # float32 inputs: within float32 rounding of the same positions
    inp32 = BT.Inputs.from_arrays(
        roots=roots,
        normals=nor,
        faces=np.zeros((0, 3), np.int64),
        mm_to_vox=MM_TO_VOX,
        outside=o,
        wm=w,
        cortex=c,
        echoes=np.ones((2, *shape)),
        scales=np.ones(2),
        device="cpu",
        dtype=torch.float32,
    )
    x32 = BT.paths(torch.zeros(n, 2, inp.K), inp32).x.numpy()
    assert np.abs(x32 - ref).max() < 1e-5


# ---------------------------------------------------------------- soft top


def test_slab_length_and_mask() -> None:
    # Outside medium at voxels z >= 30: the top is the face z = 29.5. The
    # root is at voxel (7.3, 8.6, 10.3) and the normal is +z, so the top is
    # at h = (29.5 - 10.3) * 0.16 = 3.072 mm along the path, between the
    # samples 3.04 and 3.08 mm and not on a voxel centre.
    shape = (16, 16, 80)
    o, w, c = slab(shape, 30, 9)
    inp = make(
        np.array([[7.3, 8.6, 10.3]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.zeros((0, 3), np.int64),
        o,
        w,
        c,
    )
    p = BT.paths(zeros(inp), inp)
    top = BT.soft_top(p, inp)
    h = (29.5 - 10.3) * VOX
    m = top.mask[0].numpy()
    s = p.s.numpy()
    k = int(np.flatnonzero(m < 0.5)[0])
    cross = s[k - 1] + (m[k - 1] - 0.5) / (m[k - 1] - m[k]) * R.STEP
    assert abs(cross - h) < R.STEP
    assert abs(float(top.length[0]) - h) < R.STEP
    assert not bool(top.censored[0])
    assert np.all(np.diff(m) <= 0)


def test_censored_without_outside() -> None:
    shape = (16, 16, 80)
    o, w, c = slab(shape, 1000, 9)
    inp = make(
        np.array([[7.3, 8.6, 10.3]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.zeros((0, 3), np.int64),
        o,
        w,
        c,
    )
    top = BT.soft_top(BT.paths(zeros(inp), inp), inp)
    assert bool(top.censored[0])
    assert abs(float(top.length[0]) - BT.S_MAX) < 1e-9


# ---------------------------------------------------------------- cost 1


def two_columns() -> BT.Inputs:
    shape = (16, 16, 80)
    o, w, c = slab(shape, 1000, -1)
    roots = np.array([[7.3, 8.6, 10.3], [8.3, 8.6, 10.3]])
    nor = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    return make(
        roots, nor, np.zeros((0, 3), np.int64), o, w, c, mesh_edges=np.array([[0, 1]])
    )


def test_agreement_zero_and_grows_with_shift() -> None:
    inp = two_columns()
    s = torch.as_tensor(np.arange(BT.NS) * R.STEP)
    tau = np.arange(0.0, 6.0, 0.08)
    lv = levels(inp, 0.0, tau, np.zeros((2, len(tau))))
    mask = torch.ones(2, BT.NS, dtype=F64)

    def cost(shift: float) -> float:
        f0 = torch.stack(
            [torch.sin(2 * math.pi * s / 1.3), torch.cos(2 * math.pi * s / 0.9)]
        )
        f1 = torch.stack(
            [
                torch.sin(2 * math.pi * (s + shift) / 1.3),
                torch.cos(2 * math.pi * (s + shift) / 0.9),
            ]
        )
        feat = torch.stack([f0, f1], 1)  # (E, n, NS)
        return float(BT.cost_agreement(feat, mask, lv, inp))

    assert cost(0.0) == 0.0
    seq = [cost(d) for d in (0.0, 0.04, 0.08, 0.12, 0.16)]
    assert all(b > a for a, b in zip(seq, seq[1:], strict=False))


def test_agreement_normalize_divides_by_weights() -> None:
    inp = two_columns()
    tau = np.arange(0.0, 6.0, 0.08)
    lv = levels(inp, 0.0, tau, np.zeros((2, len(tau))))
    feat = torch.zeros(2, 2, BT.NS, dtype=F64)
    feat[0, 1] = 1.0
    mask = torch.ones(2, BT.NS, dtype=F64)
    raw = float(BT.cost_agreement(feat, mask, lv, inp))
    norm = float(BT.cost_agreement(feat, mask, lv, inp, normalize=True))
    assert raw == pytest.approx(len(tau))
    assert norm == pytest.approx(1.0)


# ---------------------------------------------------------------- costs 2, 3


def flat_grid(nx: int = 5, ny: int = 4) -> BT.Inputs:
    shape = (40, 40, 80)
    o, w, c = slab(shape, 40, 9)
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    roots = np.stack(
        [5.3 + 2.5 * gi.ravel(), 5.2 + 2.5 * gj.ravel(), np.full(nx * ny, 10.3)], 1
    )
    nor = np.tile([0.0, 0.0, 1.0], (nx * ny, 1))
    return make(roots, nor, grid_mesh(nx, ny), o, w, c)


def test_length_cost() -> None:
    L = torch.tensor([1.0, 2.0, 3.5], dtype=F64)
    cen = torch.tensor([False, True, False])
    assert float(BT.cost_length(L, cen)) == 4.5


def test_length_agreement_forms() -> None:
    nx, ny = 5, 4
    inp = flat_grid(nx, ny)
    cen = torch.zeros(inp.n, dtype=torch.bool)
    free = torch.zeros(inp.n, dtype=F64)
    eq = torch.full((inp.n,), 2.5, dtype=F64)
    assert float(BT.cost_length_agreement(eq, cen, free, inp, "A")) == 0.0
    i = torch.as_tensor(np.repeat(np.arange(nx), ny), dtype=F64)
    lin = 2.0 + 0.125 * i  # linear along each row of the grid
    assert float(BT.cost_length_agreement(lin, cen, free, inp, "A")) > 0
    assert float(BT.cost_length_agreement(lin, cen, free, inp, "B")) == 0.0
    quad = 2.0 + 0.125 * i**2
    assert float(BT.cost_length_agreement(quad, cen, free, inp, "B")) > 0
    # a censored column takes the free length
    cen2 = cen.clone()
    cen2[7] = True
    free2 = free.clone()
    free2[7] = 2.5
    bad = eq.clone()
    bad[7] = 7.0
    assert float(BT.cost_length_agreement(bad, cen2, free2, inp, "A")) == 0.0


# ---------------------------------------------------------------- cost 4


def test_shear_costs() -> None:
    inp = flat_grid()
    p = BT.paths(zeros(inp), inp)
    m = BT.soft_top(p, inp).mask
    assert float(BT.cost_bending(p, m, inp)) == 0.0
    assert float(BT.cost_shear(p, m, inp)) == 0.0
    # the same shear on every column (equal normals, so equal frames)
    c = 0.05 * torch.randn(1, 2, inp.K, dtype=F64).expand(inp.n, 2, inp.K)
    q = BT.paths(c.contiguous(), inp)
    mq = BT.soft_top(q, inp).mask
    assert float(BT.cost_shear(q, mq, inp)) == 0.0
    assert float(BT.cost_bending(q, mq, inp)) > 0.0


# ---------------------------------------------------------------- cost 5


def test_crossing_parallel_is_zero() -> None:
    inp = flat_grid()
    p = BT.paths(zeros(inp), inp)
    m = BT.soft_top(p, inp).mask
    tau = np.arange(0.0, 6.0, 0.08)
    lv = levels(inp, 0.0, tau, np.zeros((inp.n, len(tau))))
    bar, crowd = BT.cost_crossing(p, m, lv, inp)
    assert abs(float(bar)) < 1e-12 and abs(float(crowd)) < 1e-20


def tilted_face() -> BT.Inputs:
    """One face: roots (0, 0, 0), (1, 0, 0), (0, 1, 0) mm from the corner
    voxel (2.37, 3.21, 5.43); normals +z, +z and (0, -1, 3)/sqrt(10). The
    third point reaches the line of the other two at s* = sqrt(10) mm, and
    the area is a0 (1 - s / s*) at equal s, with a0 = 1/2 mm^2."""
    shape = (24, 24, 70)
    o, w, c = slab(shape, 1000, -1)
    base = np.array([2.37, 3.21, 5.43])
    roots = base + np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) / VOX
    nor = np.array([[0, 0, 1], [0, 0, 1], [0, -1 / math.sqrt(10), 3 / math.sqrt(10)]])
    return make(roots, nor, np.array([[0, 1, 2]]), o, w, c)


def test_crossing_barrier_grows_without_bound() -> None:
    inp = tilted_face()
    p = BT.paths(zeros(inp), inp)
    m = BT.soft_top(p, inp).mask
    star = math.sqrt(10)
    seq = []
    for eps in (0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-6):
        q = star * (1 - eps)
        lv = levels(inp, 0.0, np.array([0.0, q]), np.zeros((3, 2)))
        bar, _ = BT.cost_crossing(p, m, lv, inp)
        seq.append(float(bar))
        assert float(bar) == pytest.approx(-math.log(eps), rel=1e-6)
    assert all(b > a for a, b in zip(seq, seq[1:], strict=False))
    # past the crossing: +inf with the pure barrier, finite with extend
    q = star + 0.1
    lv = levels(inp, 0.0, np.array([0.0, q]), np.zeros((3, 2)))
    assert float(BT.cost_crossing(p, m, lv, inp)[0]) == math.inf
    delta = 0.1
    x = 1 - q / star
    want = -math.log(delta) - (x - delta) / delta + (x - delta) ** 2 / (2 * delta**2)
    got = float(BT.cost_crossing(p, m, lv, inp, extend=delta)[0])
    assert got == pytest.approx(want, rel=1e-9)
    # and its gradient there
    lv2 = levels(inp, 0.0, np.array([0.0, star - 0.3, star + 0.1]), np.zeros((3, 3)))

    def fn(c: torch.Tensor) -> torch.Tensor:
        pp = BT.paths(c, inp)
        return BT.cost_crossing(pp, BT.soft_top(pp, inp).mask, lv2, inp, extend=delta)[
            0
        ]

    check_grad(fn, 0.01 * torch.randn(3, 2, inp.K, dtype=F64))


def test_crowding_positive_when_a_face_shrinks() -> None:
    inp = flat_grid()
    c = torch.zeros(inp.n, 2, inp.K, dtype=F64)
    c[7] = 0.05  # one column moves; its faces change area against the others
    p = BT.paths(c, inp)
    m = BT.soft_top(p, inp).mask
    tau = np.arange(0.0, 4.0, 0.08)
    lv = levels(inp, 0.0, tau, np.zeros((inp.n, len(tau))))
    bar, crowd = BT.cost_crossing(p, m, lv, inp)
    assert float(crowd) > 0 and math.isfinite(float(bar))


# ---------------------------------------------------------------- cost 6


def test_return_zero_for_one_exit_positive_for_reentry() -> None:
    shape = (16, 16, 80)
    o, w, c = slab(shape, 40, 9)
    inp = make(
        np.array([[7.3, 8.6, 9.4]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.zeros((0, 3), np.int64),
        o,
        w,
        c,
    )
    p = BT.paths(zeros(inp), inp)
    m = BT.soft_top(p, inp).mask
    assert float(BT.cost_return(p, m, inp)) == 0.0
    w2 = w.copy()
    w2[:, :, 20:23] = True  # a WM island in the cortex, below the top
    inp2 = make(
        np.array([[7.3, 8.6, 9.4]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.zeros((0, 3), np.int64),
        o & ~w2,
        w2,
        c & ~w2,
    )
    p2 = BT.paths(zeros(inp2), inp2)
    assert float(BT.cost_return(p2, BT.soft_top(p2, inp2).mask, inp2)) > 0.01


# ---------------------------------------------------------------- tools


def test_gather_gradient() -> None:
    idx = np.array([[3, 0], [3, 3], [1, 0]])
    g = BT.Gather(idx, 5, torch.device("cpu"))
    x = torch.randn(5, 4, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: g(t), (x,))
    assert torch.equal(g(x), x[torch.as_tensor(idx)])
    # a pad row: its entries get no gradient, the other rows are exact
    gp = BT.Gather(idx, 5, torch.device("cpu"), pad=3)
    (gr,) = torch.autograd.grad(gp(x).sum(), x)
    want = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0], dtype=F64)[:, None].expand(5, 4)
    assert torch.equal(gr, want)


def test_running_max() -> None:
    x = torch.randn(4, 30, dtype=F64, requires_grad=True)
    v = BT.running_max(x)
    assert torch.equal(v, torch.cummax(x, -1).values)
    assert torch.autograd.gradcheck(BT.running_max, (x,))
    # ties: plateaus give a gradient to one element of each tie only
    y = torch.tensor([[0.0, 1.0, 1.0, 0.5, 2.0, 2.0]], requires_grad=True)
    BT.running_max(y).sum().backward()
    assert y.grad is not None and float(y.grad.sum()) == 6.0


def test_from_box_maps_the_labels() -> None:
    shape = (12, 12, 30)
    lab = np.zeros(shape, np.int32)
    lab[:, :, :10] = 2
    lab[:, :, 10:20] = 3
    lab[:, :, 5:7] = 41
    box = SimpleNamespace(
        roots=np.array([[1.0, 1.0, 1.4], [1.3, 1.0, 1.4], [1.0, 1.3, 1.4]], np.float32),
        normals=np.tile(np.array([0, 0, 1], np.float32), (3, 1)),
        faces=np.array([[0, 1, 2]]),
        edges=np.array([[0, 1], [0, 2], [1, 2]]),
        inner=np.array([True, False, True]),
        labels=lab,
        filled=np.array([2, 41]),
        cortex=3,
        echoes=np.ones((3, *shape), np.float32),
        mm_to_box=MM_TO_VOX,
    )
    inp = BT.Inputs.from_box(box, "cpu", scales=np.ones(2), dtype=F64)
    assert inp.moving.tolist() == [True, False, True]
    v = inp.volumes
    # deep in each region the soft maps are 0 or 1 (smoothing: 1 voxel)
    assert float(v[0, 6, 6, 25]) == pytest.approx(1.0)  # outside
    assert float(v[1, 6, 6, 2]) == pytest.approx(1.0)  # WM (label 2)
    assert float(v[2, 6, 6, 15]) == pytest.approx(1.0)  # cortex
    assert v.shape[0] == 3 + 2  # three maps and the two feature echoes


# ---------------------------------------------------------------- gradients


def _rich_parts() -> tuple[BT.Inputs, BT.Levels, torch.Tensor]:
    inp, lv = rich()
    gen = torch.Generator().manual_seed(5)
    c0 = 0.03 * torch.randn(inp.n, 2, inp.K, generator=gen, dtype=F64)
    return inp, lv, c0


def test_rich_box_is_not_degenerate() -> None:
    inp, lv, c0 = _rich_parts()
    p = BT.paths(c0, inp)
    top = BT.soft_top(p, inp)
    assert not bool(top.censored.any())
    L = top.length
    assert float(L.max() - L.min()) > 0.2  # the tilted top
    assert float(BT.cost_return(p, top.mask, inp)) > 0  # the island
    bar, crowd = BT.cost_crossing(p, top.mask, lv, inp)
    assert math.isfinite(float(bar)) and float(bar) != 0 and float(crowd) > 0


@pytest.mark.parametrize(
    "term",
    [
        "agreement",
        "agreement_normalized",
        "length",
        "length_agreement_A",
        "length_agreement_B",
        "bending",
        "shear",
        "barrier",
        "crowding",
        "return",
        "alignment",
    ],
)
def test_gradients(term: str) -> None:
    inp, lv, c0 = _rich_parts()
    cen = torch.zeros(inp.n, dtype=torch.bool)
    cen[5] = True  # one censored column, with a free length
    free = torch.full((inp.n,), 3.0, dtype=F64)

    def fn(c: torch.Tensor) -> torch.Tensor:
        p = BT.paths(c, inp)
        top = BT.soft_top(p, inp)
        m = top.mask
        if term.startswith("agreement"):
            f = BT.features(p, m, inp)
            return BT.cost_agreement(
                f, m, lv, inp, normalize=term.endswith("normalized")
            )
        if term == "length":
            return BT.cost_length(top.length, cen)
        if term.startswith("length_agreement"):
            return BT.cost_length_agreement(top.length, cen, free, inp, term[-1])
        if term == "bending":
            return BT.cost_bending(p, m, inp)
        if term == "shear":
            return BT.cost_shear(p, m, inp)
        if term == "barrier":
            return BT.cost_crossing(p, m, lv, inp)[0]
        if term == "crowding":
            return BT.cost_crossing(p, m, lv, inp)[1]
        if term == "alignment":
            return BT.cost_alignment(p, m, inp)
        return BT.cost_return(p, m, inp)

    check_grad(fn, c0)
    # the fixed column gets no gradient
    c = c0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(fn(c), c)
    assert float(g[0].abs().max()) == 0.0


def test_free_length_gradient() -> None:
    inp, lv, c0 = _rich_parts()
    cen = torch.zeros(inp.n, dtype=torch.bool)
    cen[5] = True
    L = BT.soft_top(BT.paths(c0, inp), inp).length.detach()
    for form in ("A", "B"):
        check_grad(
            lambda f, form=form: BT.cost_length_agreement(L, cen, f, inp, form),
            torch.full((inp.n,), 3.0, dtype=F64),
        )


def test_total() -> None:
    inp, lv, c0 = _rich_parts()
    w = {k: float(i + 1) for i, k in enumerate(BT.TERMS)}
    out = BT.total(c0, inp, w, lv, "B")
    s = sum(w[k] * out[k] for k in BT.TERMS)
    assert float(out["total"]) == pytest.approx(float(s), rel=1e-12)
    # a zero weight leaves the term out of the sum, but it is still reported
    w0 = dict(w, barrier=0.0)
    out0 = BT.total(c0, inp, w0, lv, "A")
    assert math.isfinite(float(out0["barrier"]))
    with pytest.raises(ValueError):
        BT.total(c0, inp, {"nothing": 1.0}, lv, "A")
    assert "alignment" in out and float(out["alignment"]) > 0


def test_total_without_a_laminar_field() -> None:
    inp = flat_grid()
    tau = np.arange(0.0, 4.0, 0.08)
    lv = levels(inp, 0.0, tau, np.zeros((inp.n, len(tau))))
    c = zeros(inp)
    out = BT.total(c, inp, {"length": 1.0}, lv)
    assert "alignment" not in out
    assert float(out["total"]) == float(out["length"])
    out0 = BT.total(c, inp, {"length": 1.0, "alignment": 0.0}, lv)
    assert "alignment" not in out0
    with pytest.raises(ValueError, match="alignment"):
        BT.total(c, inp, {"alignment": 1.0}, lv)
    with pytest.raises(ValueError, match="align"):
        BT.cost_alignment(
            BT.paths(c, inp), BT.soft_top(BT.paths(c, inp), inp).mask, inp
        )


def test_trend_matrix_is_scipy_gaussian() -> None:
    from scipy.ndimage import gaussian_filter1d

    x = np.random.default_rng(2).standard_normal((3, BT.NS))
    g = BT._gauss_matrix(BT.TREND_MM / BT.STEP, BT.NS)
    want = gaussian_filter1d(x, BT.TREND_MM / BT.STEP, axis=-1, mode="constant")
    assert np.abs(x @ g.T - want).max() < 1e-12


# ---------------------------------------------------------------- cost 7


def sym(a: np.ndarray) -> np.ndarray:
    """(6, ...) in the order of orient.PAIRS -> (..., 3, 3)."""
    out = np.zeros((*a.shape[1:], 3, 3))
    for q, (i, j) in enumerate(BT.PAIRS):
        out[..., i, j] = out[..., j, i] = a[q]
    return out


def test_align_field_projects_out_the_normal() -> None:
    rng = np.random.default_rng(4)
    v = np.array([0.3, -0.5, 0.8])
    v /= np.linalg.norm(v)
    lead = np.tile(v, (2, 3, 2, 1))
    c = np.full((2, 3, 2), 0.7)
    c[1, 2, 1] = np.nan
    lead[0, 1, 0] = np.nan
    a = BT.align_field(lead, c, np.eye(4), np.ones(3))
    assert a.shape == (6, 2, 3, 2) and a.dtype == np.float32
    M = sym(a.astype(np.float64))
    t = rng.standard_normal((20, 3))
    want = 0.7 * (t - (t @ v)[:, None] * v)
    got = t @ M[0, 0, 1]  # M is symmetric
    assert np.abs(got - want).max() < 1e-6  # float32 storage
    # no valid tensor -> 0
    assert np.all(a[:, 1, 2, 1] == 0) and np.all(a[:, 0, 1, 0] == 0)
    # the sign of v does not matter
    assert np.array_equal(BT.align_field(-lead, c, np.eye(4), np.ones(3)), a)


def test_align_field_scanner_frame() -> None:
    # affine[:3, :3] = Rot @ Perm @ diag(vox): a voxel-frame direction u (mm
    # along each voxel axis) is Rot @ Perm @ u in scanner space.
    vox = np.array([0.16, 0.2, 0.3])
    th = 0.4
    rot = np.array(
        [[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]]
    )
    perm = np.eye(3)[[2, 0, 1]]
    aff = np.eye(4)
    aff[:3, :3] = rot @ perm @ np.diag(vox)
    aff[:3, 3] = [5.0, -3.0, 1.0]
    u = np.array([0.6, 0.0, 0.8])
    w = rot @ perm @ u
    w /= np.linalg.norm(w)
    a = BT.align_field(np.tile(u, (1, 1, 1, 1)), np.full((1, 1, 1), 0.5), aff, vox)
    M = sym(a.astype(np.float64))[0, 0, 0]
    assert np.abs(M - 0.5 * (np.eye(3) - np.outer(w, w))).max() < 1e-6
    # the same map as bent_texture.to_scanner
    import bent_texture

    assert np.allclose(bent_texture.to_scanner(u, aff, vox), w, atol=1e-12)


def uniform_align(shape: tuple[int, ...], v: np.ndarray, c: float) -> np.ndarray:
    return BT.align_field(
        np.broadcast_to(v, (*shape, 3)), np.full(shape, c), np.eye(4), np.ones(3)
    )


def angle_grid(v: np.ndarray, c: float) -> BT.Inputs:
    """flat_grid's columns (normals +z, a flat top) with a uniform field."""
    shape = (40, 40, 80)
    o, w, cx = slab(shape, 40, 9)
    nx, ny = 5, 4
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    roots = np.stack(
        [5.3 + 2.5 * gi.ravel(), 5.2 + 2.5 * gj.ravel(), np.full(nx * ny, 10.3)], 1
    )
    nor = np.tile([0.0, 0.0, 1.0], (nx * ny, 1))
    return make(
        roots, nor, grid_mesh(nx, ny), o, w, cx, align=uniform_align(shape, v, c)
    )


def test_alignment_zero_along_v() -> None:
    inp = angle_grid(np.array([0.0, 0.0, 1.0]), 0.8)
    p = BT.paths(zeros(inp), inp)
    m = BT.soft_top(p, inp).mask
    assert float(BT.cost_alignment(p, m, inp)) == 0.0


def test_alignment_at_a_known_angle() -> None:
    # Straight paths along +z, v at theta to +z in the x-z plane. The
    # field is stored in float32: the cost is A_zz sum m w_k to float64
    # rounding, and A_zz is c sin^2 theta to float32 rounding (1e-6).
    c = 0.6
    for theta in (0.2, 0.7, math.pi / 2):
        v = np.array([math.sin(theta), 0.0, math.cos(theta)])
        inp = angle_grid(v, c)
        p = BT.paths(zeros(inp), inp)
        m = BT.soft_top(p, inp).mask
        got = float(BT.cost_alignment(p, m, inp))
        assert inp.align is not None
        azz = float(inp.align[BT.PAIRS.index((2, 2)), 0, 0, 0])
        sm = float((m * inp.ds).sum())
        assert got == pytest.approx(azz * sm, rel=1e-12)
        assert got == pytest.approx(c * math.sin(theta) ** 2 * sm, rel=1e-6)
        # the sign of v does not matter
        inp2 = angle_grid(-v, c)
        assert float(BT.cost_alignment(p, m, inp2)) == got


def test_align_field_must_match_the_grid() -> None:
    shape = (16, 16, 80)
    o, w, c = slab(shape, 30, 9)
    with pytest.raises(ValueError, match="align"):
        make(
            np.array([[7.3, 8.6, 10.3]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.zeros((0, 3), np.int64),
            o,
            w,
            c,
            align=uniform_align((16, 16, 79), np.array([0.0, 0.0, 1.0]), 0.5),
        )


def test_tangent_is_unit_and_along_the_derivative() -> None:
    inp, _, c0 = _rich_parts()
    p = BT.paths(c0, inp)
    t = p.tangent
    assert torch.allclose(t.norm(dim=-1), torch.ones_like(p.speed), atol=1e-12)
    # against a central difference of the positions along s
    dx = (p.y[:, 2:] - p.y[:, :-2]) / (2 * BT.STEP)
    tf = dx / dx.norm(dim=-1, keepdim=True)
    assert float((t[:, 1:-1] - tf).abs().max()) < 1e-3


# ---------------------------------------------------------------- closest


def test_closest_outside_reaches_the_top() -> None:
    # Outside medium at z >= 30; the path along +z from z = 10.3 reaches
    # z = 60.3 > 31, so some samples have all eight corners outside: the
    # minimum is exactly 0.
    shape = (16, 16, 80)
    o, w, c = slab(shape, 30, 9)
    inp = make(
        np.array([[7.3, 8.6, 10.3]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.zeros((0, 3), np.int64),
        o,
        w,
        c,
    )
    d = BT.closest_outside(BT.paths(zeros(inp), inp), inp)
    assert d.shape == (1,) and float(d[0]) == 0.0


def test_closest_outside_parallel_to_the_top() -> None:
    # A path along +x at z0 below a flat outside at z >= 30 (top face at
    # 29.5). The distance map at voxel centre z = k < 30 is (30 - k) VOX
    # (to the centre of the nearest outside voxel), linear in z between
    # centres, so the trilinear value is (30 - z0) VOX exactly (to
    # rounding). With d = (29.5 - z0) VOX the distance to the face, that is
    # d + VOX / 2: within half a voxel of d.
    shape = (64, 16, 40)
    o, w, c = slab(shape, 30, 2)
    for z0 in (20.0, 24.3, 28.9):
        inp = make(
            np.array([[3.3, 8.6, z0]]),
            np.array([[1.0, 0.0, 0.0]]),
            np.zeros((0, 3), np.int64),
            o,
            w,
            c,
        )
        got = float(BT.closest_outside(BT.paths(zeros(inp), inp), inp)[0])
        assert got == pytest.approx((30 - z0) * VOX, abs=1e-12)
        d = (29.5 - z0) * VOX
        assert abs(got - d) <= VOX / 2 + 1e-12
