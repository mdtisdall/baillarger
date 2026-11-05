"""Tests of baillarger.flattening.flatten_superblock: the local solve on the
real samplers (ray-sampler/1, column-volume/1), on a slab phantom made by
the dseg and the mesh commands.

The phantom: a slab of WM in a box of cortex, with a laminar echo above the
top face of the WM. The bands are warped across the slab, so the warp is
not zero. The tests compare the warp of the same columns from different
calls: one call for the whole phantom (the global solve), calls for the two
superblocks at level 1 that meet at i = 16 and for two superblocks at
level 0 (local solves, with the halo HALO_MM).
"""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from test_flatten import pattern
from test_rays import PHANTOM_AFF, VS

from baillarger import flattening as FG
from baillarger import rays as R
from baillarger import white_dseg, white_surf
from baillarger.blocks import ArraySource
from baillarger.column_volume import ColumnVolumeSampler

#: the grid of the phantom, and the top face of its WM (voxel k)
N_IJ, N_K = 32, 30
K_TOP = 11.5
#: the two superblocks of the local solves (level 1: 16 voxels, 2.56 mm),
#: on each side of the face i = 16
LEVEL = 1
SUPERBLOCKS = ((0, 1, 0), (1, 1, 0))
#: the local solve agrees with the global solve to this, mm. Measured on
#: this phantom with the halo of 2 mm: 0.0032 at most (99 % below 0.0002);
#: with 4 mm: below 1e-13.
TOL_MM = 0.005


def labels(shift: int = 0) -> npt.NDArray[np.int16]:
    """Cortex (42) in the box [2, 30) x [2, 30) x [2, 28), WM (41) in
    [4, 28) x [4, 28) x [6, 12), moved by `shift` voxels along i and j."""
    s = shift
    a = np.zeros((N_IJ + s, N_IJ + s, N_K), np.int16)
    a[2 + s : 30 + s, 2 + s : 30 + s, 2:28] = 42
    a[4 + s : 28 + s, 4 + s : 28 + s, 6:12] = 41
    return a


def echoes(shift: int = 0) -> list[npt.NDArray[np.float32]]:
    """Three echoes: the laminar pattern in depth above the top face of the
    WM, warped along i and j (the bands are 6 % deeper at one side, and
    bulge by up to 0.1 mm along j). In the WM and below, a constant."""
    s = shift
    i, j, k = np.indices((N_IJ + s, N_IJ + s, N_K), float)
    x = (i - s - 16.0) / 12.0
    y = (j - s - 16.0) / 12.0
    d = (k - K_TOP) * VS
    u = d / (1.0 + 0.06 * x) - 0.1 * np.cos(np.pi * y / 2) * np.clip(d / 3.0, 0, 1)
    p = np.where(d > 0, pattern(u), 0.0)
    return [
        (sc * np.exp(pw * p)).astype(np.float32)
        for sc, pw in ((600.0, 1.0), (400.0, 1.0), (250.0, 0.8))
    ]


def write_phantom(
    d: Path, a: npt.NDArray[np.int16], aff: npt.NDArray[np.float64]
) -> tuple[Path, Path]:
    """Labels a (a right hemisphere) through the dseg and the mesh commands
    (8-cell blocks), as test_rays.write_phantom, with its own affine."""
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.nifti1.save(nib.nifti1.Nifti1Image(a, aff), src)
    out = d / "deriv"
    white_dseg.main([str(src), "--out", str(out)])
    anat = out / "sub-T01/ses-S1/anat"
    dseg = anat / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    white_surf.main([str(dseg), "--out", str(out), "--block-cells", "8"])
    return anat / "sub-T01_ses-S1_hemi-R_white.surf.gii", dseg


Phantom = tuple[R.RaySampler, ColumnVolumeSampler, FG.SurfaceFaces]


def make(tmp: Path, shift: int = 0) -> Phantom:
    """The samplers of the phantom on a grid moved by `shift` voxels, with
    the affine moved back: the same scanner positions."""
    aff = PHANTOM_AFF.copy()
    aff[:2, 3] -= shift * VS
    surf, dseg = write_phantom(tmp, labels(shift), aff)
    rs = R.RaySampler(surf, dseg)
    smp = ColumnVolumeSampler(
        [ArraySource(v) for v in echoes(shift)],
        rs.affine,
        [rs.affine] * 3,
        (rs.shape[0], rs.shape[1], rs.shape[2]),
    )
    return rs, smp, FG.SurfaceFaces.load(surf)


def levels_by_root(
    rs: R.RaySampler, out: list[FG.Flattened]
) -> dict[tuple[float, ...], dict[int, float]]:
    """For each owned column, keyed by its root (scanner mm, rounded to
    1 um): the s of each level's horizon, keyed by the level's s before the
    warp (in units of 0.08 mm), so that calls with another grid start
    compare."""
    roots = rs.roots((0, 0, 0), 4)
    pos = {
        int(k): tuple(np.round(x.astype(float), 3).tolist())
        for k, x in zip(roots.edge_key, roots.root, strict=True)
    }
    d: dict[tuple[float, ...], dict[int, float]] = {}
    for f in out:
        lev = np.rint((f.s_lo + f.tau) / 0.08).astype(int).tolist()
        hz = f.s_lo + f.tau[None] + f.r
        for k, row in zip(f.edge_key.tolist(), hz, strict=True):
            d[pos[k]] = dict(zip(lev, row.tolist(), strict=True))
    return d


def largest_difference(
    a: dict[Any, dict[int, float]], b: dict[Any, dict[int, float]]
) -> float:
    """The largest difference of a horizon between a and b, over the columns
    of a and the levels both have in the cortex (s from 0 to 2.5 mm)."""
    worst = 0.0
    for k, la in a.items():
        lb = b[k]
        for lev in set(la) & set(lb):
            if 0 <= lev <= round(2.5 / 0.08):
                worst = max(worst, abs(la[lev] - lb[lev]))
    return worst


def local(p: Phantom, scales: Any, eref: float) -> list[FG.Flattened]:
    rs, smp, faces = p
    return [
        FG.flatten_superblock(rs, smp, faces, sb, LEVEL, scales=scales, energy_ref=eref)
        for sb in SUPERBLOCKS
    ]


@pytest.fixture(scope="module")
def phantom(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    p = make(tmp_path_factory.mktemp("slab"))
    rs, smp, faces = p
    glob = FG.flatten_superblock(rs, smp, faces, (0, 0, 0), 4, halo_mm=0.0)
    loc = local(p, glob.scales, glob.energy_ref)
    return dict(p=p, glob=glob, loc=loc)


def test_the_phantom_has_a_warp(phantom: dict[str, Any]) -> None:
    """The warp in the cortex spans many times the tolerance."""
    g = phantom["glob"]
    top = g.r[:, g.anchor_level + 20]  # 1.6 mm above the root
    assert np.ptp(top) > 20 * TOL_MM
    assert sum(len(f.edge_key) for f in phantom["loc"]) > 500


def test_each_column_is_owned_once(phantom: dict[str, Any]) -> None:
    keys = np.concatenate([f.edge_key for f in phantom["loc"]])
    assert len(np.unique(keys)) == len(keys)
    assert set(keys.tolist()) <= set(phantom["glob"].edge_key.tolist())


def test_the_local_solve_equals_the_global_solve(phantom: dict[str, Any]) -> None:
    rs = phantom["p"][0]
    loc = levels_by_root(rs, phantom["loc"])
    glo = levels_by_root(rs, [phantom["glob"]])
    assert largest_difference(loc, glo) < TOL_MM


def test_the_result_does_not_depend_on_where_the_blocks_fall(
    phantom: dict[str, Any],
) -> None:
    """Level 0 (8 voxels): the superblock faces now fall at i = 8, 16 and 24,
    one block from the faces of level 1. The columns of the two level-0
    superblocks get the same warp as in the global solve and at level 1,
    within the tolerance.

    (A grid moved by one block is not used: the mesh commands do not give
    the same surface on a moved grid; its vertices move by up to 0.075 mm.)"""
    rs, smp, faces = phantom["p"]
    g = phantom["glob"]
    out = [
        FG.flatten_superblock(
            rs, smp, faces, sb, 0, scales=g.scales, energy_ref=g.energy_ref
        )
        for sb in ((1, 2, 1), (2, 2, 1))
    ]
    a = levels_by_root(rs, out)
    assert len(a) > 200
    assert largest_difference(a, levels_by_root(rs, [g])) < TOL_MM
    one = levels_by_root(rs, phantom["loc"])
    both = {k: v for k, v in a.items() if k in one}
    assert len(both) > 100
    assert largest_difference(both, one) < TOL_MM


def test_the_pairs_are_oriented_by_edge_key(phantom: dict[str, Any]) -> None:
    for f in phantom["loc"]:
        assert np.all(f.pair_key[:, 0] < f.pair_key[:, 1])


# ---------------------------------------------------------------- the gauge


def on_levels(
    rs: R.RaySampler, out: list[FG.Flattened], lev: Any, c: Any
) -> dict[tuple[float, ...], dict[int, float]]:
    """As levels_by_root, for warps on the common levels with offsets c."""
    roots = rs.roots((0, 0, 0), 4)
    pos = {
        int(k): tuple(np.round(x.astype(float), 3).tolist())
        for k, x in zip(roots.edge_key, roots.root, strict=True)
    }
    d: dict[tuple[float, ...], dict[int, float]] = {}
    for b, f in enumerate(out):
        hz = lev[None] * 0.08 + FG.in_gauge(f, c[b], lev)
        for k, row in zip(f.edge_key.tolist(), hz, strict=True):
            d[pos[k]] = dict(zip(lev.tolist(), row.tolist(), strict=True))
    return d


@pytest.fixture(scope="module")
def small_halo(phantom: dict[str, Any]) -> list[FG.Flattened]:
    """Local solves with the halo HALO_MM on every superblock of level 1
    that owns columns."""
    rs, smp, faces = phantom["p"]
    g = phantom["glob"]
    out = [
        FG.flatten_superblock(
            rs, smp, faces, (a, b, 0), LEVEL, scales=g.scales, energy_ref=g.energy_ref
        )
        for a in (0, 1)
        for b in (0, 1)
    ]
    return [f for f in out if len(f.edge_key)]


def test_local_solves_with_offsets_equal_the_global_solve(
    phantom: dict[str, Any], small_halo: list[FG.Flattened]
) -> None:
    rs, g = phantom["p"][0], phantom["glob"]
    keys = np.concatenate([f.edge_key for f in small_halo])
    assert set(keys.tolist()) == set(g.edge_key.tolist())
    lev, c = FG.gauge_offsets(small_halo)
    got = on_levels(rs, small_halo, lev, c)
    assert largest_difference(got, levels_by_root(rs, [g])) < TOL_MM


def test_the_offsets_of_one_solve_are_zero(phantom: dict[str, Any]) -> None:
    """One superblock that holds every column: its gauge is the global one.

    (On this phantom a halo of 2 mm reaches most of the slab, so the gauges
    of the local solves differ little; tests/test_flattening.py tests the
    offsets exactly on gauges made by hand.)"""
    lev, c = FG.gauge_offsets([phantom["glob"]])
    np.testing.assert_allclose(c, 0.0, atol=1e-6)  # CG to rtol 1e-7
