"""Tests of bent_fit on small synthetic volumes in voxels of 0.16 mm.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_bent_fit.py
"""

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bent_fit as BF  # noqa: E402
import bent_terms as BT  # noqa: E402

FP: Any = importlib.import_module("fold_phantoms")

VOX = 0.16
MM_TO_VOX = np.diag([1 / VOX, 1 / VOX, 1 / VOX, 1.0])


def grid_mesh(nx: int, ny: int) -> np.ndarray:
    """The faces of a regular grid of nx * ny vertices (index i * ny + j)."""
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
    return np.asarray(f, np.int64)


def grid_roots(
    nx: int, ny: int, x0: float, y0: float, z: float, d: float
) -> np.ndarray:
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return np.stack([x0 + d * gi.ravel(), y0 + d * gj.ravel(), np.full(nx * ny, z)], 1)


def interior(nx: int, ny: int) -> np.ndarray:
    """Moving: the grid's vertices off its outer ring (the ring is a fixed
    halo, as the box's halo)."""
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return ((gi > 0) & (gi < nx - 1) & (gj > 0) & (gj < ny - 1)).ravel()


def flat_problem() -> BF.Problem:
    """A slab: WM for z <= 10, cortex to z = 28, outside above; laminae
    parallel to the slab (echoes that vary with z only); normals along z;
    a laminar field along z (c = 0.5)."""
    shape = (24, 24, 50)
    z = np.arange(shape[2], dtype=float)[None, None, :] * np.ones(shape)
    outside, wm = z >= 28, z <= 10
    e1 = 200 + 40 * np.sin(2 * np.pi * z / 4.3)
    e2 = 150 + 30 * np.cos(2 * np.pi * z / 6.1)
    nx = ny = 5
    roots = grid_roots(nx, ny, 6.37, 6.21, 10.43, 2.5) * VOX
    normals = np.tile([0.0, 0.0, 1.0], (nx * ny, 1))
    v = np.zeros((*shape, 3))
    v[..., 2] = 1.0
    c = np.full(shape, 0.5)
    c[outside] = np.nan
    inp = BT.Inputs.from_arrays(
        roots,
        normals,
        grid_mesh(nx, ny),
        MM_TO_VOX,
        outside,
        wm,
        ~outside & ~wm,
        np.stack([e1, e2]),
        np.ones(2),
        moving=interior(nx, ny),
        align=BT.align_field(v, c, np.eye(4), np.ones(3)),
    )
    return BF.make_problem(inp)


def rich_problem() -> BF.Problem:
    """A box with a top tilted along x, layered echoes that vary along x
    and y too, a WM island in the cortex, and normals tilted at random."""
    rng = np.random.default_rng(3)
    shape = (28, 28, 60)
    x, y, z = np.meshgrid(*[np.arange(s, dtype=float) for s in shape], indexing="ij")
    outside = z > 26 + 0.3 * x
    wm = (z <= 10) | ((z >= 18) & (z <= 20) & (x >= 10) & (x <= 16))
    e1 = 200 + 40 * np.sin(2 * np.pi * (z - 0.1 * x) / 4.3) + 3 * x + 2 * y
    e2 = 150 + 30 * np.cos(2 * np.pi * (z - 0.1 * x) / 6.1) + 2 * x - y
    nx = ny = 5
    roots = grid_roots(nx, ny, 6.37, 6.21, 10.43, 3.0) * VOX
    t = rng.uniform(-0.05, 0.05, (nx * ny, 2))
    normals = np.c_[t, np.ones(nx * ny)]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    vv = np.stack([0.3 * np.sin(x / 5), np.zeros(shape), np.ones(shape)], -1)
    vv /= np.linalg.norm(vv, axis=-1, keepdims=True)
    c = 0.4 + 0.3 * np.sin(z / 6 + x / 8)
    c[outside] = np.nan
    inp = BT.Inputs.from_arrays(
        roots,
        normals,
        grid_mesh(nx, ny),
        MM_TO_VOX,
        outside,
        wm,
        ~outside & ~wm,
        np.stack([e1, e2]),
        np.ones(2),
        moving=interior(nx, ny),
        align=BT.align_field(vv, c, np.eye(4), np.ones(3)),
    )
    return BF.make_problem(inp)


@pytest.mark.parametrize("form", ["A", "B"])
@pytest.mark.parametrize("alignment", [True, False])
def test_flat_slab_stays_straight(form: str, alignment: bool) -> None:
    prob = flat_problem()
    out = BF.fit(prob, form, alignment, rounds=2, steps=10)
    assert BF.move_mm(BF.zeros(prob.inputs), out.params, prob.inputs) < 0.01
    assert BF.shear_mm(out.params, prob.inputs).max() < 0.01


def test_total_does_not_increase_between_rounds() -> None:
    """The compared value: the total (weighted sum) of each accepted round,
    at that round's warp and censored set, after its descent ("after").
    Also: within a round, after <= before."""
    prob = rich_problem()
    out = BF.fit(prob, "A", True, rounds=3, steps=8, tol=0.0)
    rec = [h for h in out.history[1:] if h["accepted"]]
    assert len(rec) >= 2
    tot = [h["after"]["total"] for h in rec]
    assert all(b <= a for a, b in zip(tot, tot[1:], strict=False))
    for h in out.history[1:]:
        assert h["after"]["total"] <= h["before"]["total"]
    # the paths moved: the test is not of a fit that did nothing
    assert BF.move_mm(BF.zeros(prob.inputs), out.params, prob.inputs) > 1e-3


def test_same_seed_same_paths() -> None:
    a = BF.fit(rich_problem(), "B", True, rounds=2, steps=5, seed=4)
    b = BF.fit(rich_problem(), "B", True, rounds=2, steps=5, seed=4)
    assert torch.equal(a.params, b.params)
    assert a.weights == b.weights


def test_weights_equalize_the_start() -> None:
    """Each term that is not 0 at the straight rays has weighted value +-1
    there; bending and shear (0 at the start) have it at p_ref; a term that
    is 0 at both has weight 1."""
    prob = rich_problem()
    p0 = BF.zeros(prob.inputs)
    lv = BT.Levels.from_warp(prob.inputs, BF.fit_warp(prob, p0))
    w, rec = BF.start_weights(prob, lv, "A", 0)
    assert rec["bending"]["by"] == "reference" and rec["shear"]["by"] == "reference"
    for k, r in rec.items():
        if r["by"] == "start":
            assert abs(w[k] * r["start"]) == pytest.approx(1.0)
        elif r["by"] == "reference":
            assert abs(r["start"]) < BF.ZERO
            assert abs(w[k] * r["reference"]) == pytest.approx(1.0)
        else:
            assert abs(r["start"]) < BF.ZERO and abs(r["reference"]) < BF.ZERO
            assert w[k] == 1.0


def test_reference_perturbation_is_a_constant_bend() -> None:
    prob = rich_problem()
    inp = prob.inputs
    p = BT.paths(BF.reference_params(inp, 0), inp)
    m = inp.moving.bool()
    s = inp.s
    d = torch.linalg.norm(p.d, dim=-1)[m]
    assert torch.allclose(d, BF.KAPPA_REF * s**2 / 2, atol=1e-6)
    assert torch.allclose(
        p.curv2[m], torch.full_like(p.curv2[m], BF.KAPPA_REF**2), atol=1e-6
    )


def test_straight_profiles_are_the_straight_rays() -> None:
    prob = flat_problem()
    inp = prob.inputs
    pr = BF.sample_profiles(BF.zeros(inp), inp, BF.S_LO)
    k = int(round(-BF.S_LO / BT.STEP))
    s = BF.S_LO + BT.STEP * np.arange(pr.vox.shape[1])
    want = inp.root_vox.double().numpy()[:, None] + s[None, :, None] * (
        inp.normal.double().numpy()[:, None] @ inp.lin.double().numpy().T
    )
    assert np.allclose(pr.vox, want, atol=1e-4)
    assert pr.vox.shape[1] == k + BT.NS


def test_phantom_image_matches_make() -> None:
    feats = {
        e: dict(
            profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))),
            wm_level=-0.3,
            fluid_level=-2.0,
            hp_stat=0.1,
        )
        for e in ("echo2", "echo3")
    }
    fold = FP.Fold("fundus", (40, 40, 40), dict(g=0.16, t=1.6))
    labels, L, sds = BF.phantom_image(fold, feats, 1.0, 3)
    B = FP.make(fold, None, noise=1.0, seed=3, feats=feats)
    assert np.array_equal(labels, B["labels"])
    assert sds == B["noise_sd"]
    ctx = labels == 0
    for k in range(2):
        assert np.array_equal(FP.detrend(L[k], ctx), B["det"][k + 1])
