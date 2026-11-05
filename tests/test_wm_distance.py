"""The WM-distance sampler (docs/formats/wm-distance.md): exact tests on
phantoms, a brute-force reference, and the spec against the code."""

import dataclasses
import inspect
import itertools
import typing
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
import skfmm
from wm_distance_ref import (
    geodesic_reference,
    oblique_plane_phantom,
    obstacle_label_phantom,
    plane_exact,
    plane_phantom,
    reference_bound,
    sulcus_phantom,
)

from baillarger import white_dseg
from baillarger import wm_distance as W

DOC = Path(__file__).resolve().parents[1] / "docs" / "formats" / "wm-distance.md"
ISO = (0.16, 0.16, 0.16)
ANISO = (0.16, 0.24, 0.12)
VOXELS = [ISO, ANISO]


# ------------------------------------------------------------------- the spec


def doc_rows(heading: str) -> list[list[str]]:
    """The cells of the table rows (first cell in backticks) under heading."""
    lines = DOC.read_text().splitlines()
    i = lines.index(heading)
    out: list[list[str]] = []
    for line in lines[i + 1 :]:
        if line.startswith("#") and out:
            break
        if line.startswith("| `"):
            out.append([c.strip().strip("`") for c in line.strip("|").split("|")])
    return out


def test_the_spec_gives_the_constants() -> None:
    text = DOC.read_text()
    assert f"SpecVersion `{W.VERSION}`" in text
    assert f"`CAP` = {W.CAP}" in text
    assert f"`REACH` = {W.REACH}" in text
    assert "`MARGIN` = √3" in text and W.MARGIN == np.sqrt(3.0)
    assert "`TOLERANCE` = 1e-5 mm" in text and W.TOLERANCE == 1e-5


def test_the_spec_lists_the_parameters() -> None:
    sig = inspect.signature(W.WMDistanceSampler.sample)
    names = [p for p in sig.parameters if p != "self"]
    assert [row[0] for row in doc_rows("### Parameters")] == names


def test_the_spec_lists_the_output_fields() -> None:
    hints = typing.get_type_hints(W.WMDistance)
    got = [row[0] for row in doc_rows("### `WMDistance`")]
    assert got == [f.name for f in dataclasses.fields(W.WMDistance)]
    assert set(hints) == set(got)


# ------------------------------------------------------- no obstacle: a plane


@pytest.mark.parametrize("vox", VOXELS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_plane_on_an_axis_is_exact(
    vox: tuple[float, float, float], axis: int
) -> None:
    """An interface on a grid plane: second-order fast marching is exact in
    1-D, so the distance is (i - level + 0.5) * vox[axis] to float64
    rounding, on both sides."""
    shape = (30, 30, 30)
    wm, ctx = plane_phantom(shape, axis, 12)
    d = W.wm_distance(wm, ctx, vox)
    want = plane_exact(shape, axis, 12, vox)
    assert np.all(np.isfinite(d))
    np.testing.assert_allclose(d, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("vox", VOXELS)
def test_an_oblique_plane_is_within_half_a_voxel(
    vox: tuple[float, float, float],
) -> None:
    """An oblique plane, n = (1, 2, 3)/|.|: the labels make the interface a
    staircase within half a voxel of the plane, and the march adds a
    second-order error. The bound: |D - n . x + c| <= max(vox) / 2 + 1e-9,
    within 2 mm of the plane, where the plane's nearest point is in the
    array."""
    shape = (40, 40, 40)
    n = np.array((1.0, 2.0, 3.0)) / np.linalg.norm((1.0, 2.0, 3.0))
    v = np.asarray(vox)
    off = float(19.5 * (v * n).sum())
    wm, ctx = oblique_plane_phantom(shape, (1.0, 2.0, 3.0), off, vox)
    d = W.wm_distance(wm, ctx, vox)
    x = np.indices(shape).astype(np.float64) * v[:, None, None, None]
    s = np.tensordot(n, x, axes=1) - off
    foot = x - s * n[:, None, None, None]
    ext = ((np.asarray(shape) - 1) * v)[:, None, None, None]
    ok = np.all((foot >= 0) & (foot <= ext), axis=0) & (np.abs(s) < 2.0)
    assert ok.sum() > 30000
    assert np.max(np.abs(d - s)[ok]) <= max(vox) / 2 + 1e-9


# ------------------------------------------------------------------ obstacles


@pytest.mark.parametrize("vox", VOXELS)
def test_the_path_goes_round_a_sulcus(vox: tuple[float, float, float]) -> None:
    """A wall of outside beside the WM, open at the top: the far bank's
    cortex reaches the WM over the wall's edge. Against Dijkstra on a graph
    of 3^3 sub-voxels with a radius-2 stencil: -reference_bound(d) <=
    D - ref <= 1.5 max(vox). The upper bound is the march's: it knows the
    voxel centres only, so it cannot cut the corner of the wall's cubes as
    the reference can (measured: 1.27 and 1.37 voxels)."""
    wm, ctx, detour = sulcus_phantom((32, 32, 6))
    d = W.wm_distance(wm, ctx, vox)
    ref = geodesic_reference(wm, ctx, vox)
    assert np.array_equal(np.isnan(d), ~(wm | ctx))
    assert np.all(np.isfinite(d[ref < W.CAP - 0.5]))
    free = np.isfinite(d) & np.isfinite(ref)
    assert free.sum() > 5000
    e = np.abs(d[free]) - ref[free]
    lo = np.array([reference_bound(r, vox, 3, 2) for r in ref[free]])
    assert np.all(e >= -lo)
    assert np.all(e <= 1.5 * max(vox))
    # the detour is longer than the straight line to the interface plane
    x = np.indices(wm.shape)[0]
    assert np.all(d[detour] > (x[detour] - 5.5) * vox[0] + 0.1)


@pytest.mark.parametrize("vox", VOXELS)
def test_an_obstacle_label_blocks(vox: tuple[float, float, float]) -> None:
    """A block of another label between cortex and WM: the cortex above it
    goes round it. NaN in the block; against the reference as above."""
    wm, ctx, blocked = obstacle_label_phantom((24, 24, 24))
    d = W.wm_distance(wm, ctx, vox)
    ref = geodesic_reference(wm, ctx, vox)
    assert np.array_equal(np.isnan(d), ~(wm | ctx))
    free = wm | ctx
    assert np.all(np.isfinite(d[free]))
    e = np.abs(d[free]) - ref[free]
    lo = np.array([reference_bound(r, vox, 3, 2) for r in ref[free]])
    assert np.all(e >= -lo) and np.all(e <= 1.5 * max(vox))
    x = np.indices(wm.shape)[0]
    straight = (x - 5.5) * vox[0]
    assert np.all(d[blocked] > straight[blocked])
    assert np.max((d - straight)[blocked]) > 0.1


def test_tissue_with_no_path_is_infinite() -> None:
    """Cortex shut off from the WM by outside, and WM with no cortex: ±inf,
    never NaN. NaN only at the obstacles."""
    a = np.zeros((20, 20, 20), np.int16)
    a[:, :, :8] = 41
    a[:, :, 8:14] = 42
    a[:, :, 16:] = 42  # behind a slab of outside at k = 14, 15
    wm, ctx = a == 41, a == 42
    d = W.wm_distance(wm, ctx, ISO)
    assert np.all(d[:, :, 16:] == np.inf)
    assert np.all(np.isnan(d[:, :, 14:16]))
    assert np.all(np.isfinite(d[:, :, :14]))
    only_wm = W.wm_distance(wm, np.zeros_like(wm), ISO)
    assert np.all(only_wm[wm] == -np.inf) and np.all(np.isnan(only_wm[~wm]))


# ------------------------------------------------------------------- the cap


@pytest.mark.parametrize("cap", [1.0, 2.5, W.CAP])
def test_values_beyond_the_cap_are_infinite(cap: float) -> None:
    """|D| <= cap is the value; beyond it +inf in cortex, -inf in the WM."""
    shape = (40, 12, 70)
    wm, ctx = plane_phantom(shape, 2, 20)
    d = W.wm_distance(wm, ctx, ISO, cap)
    want = plane_exact(shape, 2, 20, ISO)
    inside = np.abs(want) <= cap - 1e-9
    beyond = np.abs(want) > cap + 1e-9
    np.testing.assert_allclose(d[inside], want[inside], rtol=0, atol=1e-12)
    assert np.all(d[beyond & ctx] == np.inf)
    assert np.all(d[beyond & wm] == -np.inf)


def test_narrow_does_not_change_the_values_in_the_cap() -> None:
    """skfmm's narrow band (narrow = cap) gives the same float64 values, bit
    for bit, as the full march, where |D| <= cap."""
    wm, ctx, _ = sulcus_phantom((32, 32, 6))
    cap = 2.0
    d = W.wm_distance(wm, ctx, ISO, cap)
    phi = np.ma.MaskedArray(np.where(wm, -1.0, 1.0), mask=~(wm | ctx))
    full = np.ma.filled(skfmm.distance(phi, dx=ISO, order=2), np.nan)
    inside = np.abs(full) <= cap
    assert inside.sum() > 1000
    assert np.array_equal(d[inside], full[inside])
    assert np.all(np.isinf(d[(wm | ctx) & ~inside]))


def test_orient_stays_clear_of_the_cap() -> None:
    """orient asks for a cap of 5 mm and uses the sign in cortex up to about
    4.5 mm from its WM. On a 4.5 mm cortex (28 voxels of 0.16 mm) over a
    curved WM (a ball), every cortex value is finite and at least 0.4 mm
    below the cap: the finite / inf jump at the cap is not met."""
    shape = (90, 90, 90)
    x = np.indices(shape).astype(np.float64)
    r = np.sqrt(((x - 44.6) ** 2).sum(0)) * 0.16
    wm = r < 2.0
    ctx = (r >= 2.0) & (r < 6.5)
    d = W.wm_distance(wm, ctx, ISO, 5.0)
    assert np.all(np.isfinite(d[ctx]))
    assert np.max(d[ctx]) < 5.0 - 0.4


# ---------------------------------------------- the sampler on the files

VS = 0.16
AFF = np.diag([VS, 0.2, 0.12, 1.0])
AFF[:3, 3] = [-3.0, 2.0, 5.0]


def ball_labels() -> npt.NDArray[np.int16]:
    """A right hemisphere on 48^3 voxels: WM ball, cortex shell, outside;
    and a slab of the amygdala (54, an obstacle) through part of the shell.
    Oblique fronts everywhere."""
    x = np.indices((48, 48, 48)).astype(np.float64)
    c = np.array([23.6, 24.2, 23.9])[:, None, None, None]
    r = np.sqrt(
        (((x - c) * np.array([VS, 0.2, 0.12])[:, None, None, None]) ** 2).sum(0)
    )
    a = np.zeros((48, 48, 48), np.int16)
    a[r < 4.2] = 42
    a[r < 2.0] = 41
    a[30:34, 10:30, 20:40][a[30:34, 10:30, 20:40] == 42] = 54
    return a


@pytest.fixture(scope="module")
def sampler(tmp_path_factory: pytest.TempPathFactory) -> W.WMDistanceSampler:
    d = tmp_path_factory.mktemp("ball")
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(ball_labels(), AFF), src)
    out = d / "deriv"
    white_dseg.main([str(src), "--out", str(out)])
    dseg = out / "sub-T01/ses-S1/anat/sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    return W.WMDistanceSampler(dseg, cells=8)


def test_the_sampler_matches_the_whole_array(sampler: W.WMDistanceSampler) -> None:
    """A call's values are the whole array's values, capped: within
    TOLERANCE and 1 float32 ulp.
    The cap is small (1 mm), so that the read box is smaller than the grid."""
    cap = 1.0
    lab = sampler.labels.read(tuple(slice(0, n) for n in sampler.shape))
    whole = W.wm_distance(
        np.isin(lab, sampler.filled), lab == sampler.cortex, sampler.voxel_mm, cap
    ).astype(np.float32)
    for i, j, k in itertools.product(range(6), repeat=3):
        r = sampler.sample((i, j, k), 0, halo_mm=0.5, cap=cap)
        assert r.distance.dtype == np.float32
        want = whole[r.box]
        got = r.distance
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.array_equal(np.isinf(got), np.isinf(want))
        f = np.isfinite(want)
        tol = W.TOLERANCE + np.spacing(np.abs(want[f]))
        assert np.all(np.abs(got[f] - want[f]) <= tol)


def test_two_superblocks_agree(sampler: W.WMDistanceSampler) -> None:
    """A voxel in the interior of one superblock and the halo of the next:
    float64 within TOLERANCE, float32 within TOLERANCE and 1 ulp; and the
    same NaN, and the same inf except within TOLERANCE of the cap. The
    counts of differences are printed (pytest -s)."""
    cap = 1.0
    n32 = n64 = 0
    worst = 0.0
    for i0, j0, k0 in itertools.product(range(1, 5), repeat=3):
        a = (i0, j0, k0)
        for b in ((i0 + 1, j0, k0), (i0, j0 + 1, k0), (i0, j0, k0 + 1)):
            da, ba = sampler.sample64(a, 0, 1.0, cap)
            db, bb = sampler.sample64(b, 0, 1.0, cap)
            lo = [max(p.start, q.start) for p, q in zip(ba, bb, strict=True)]
            hi = [min(p.stop, q.stop) for p, q in zip(ba, bb, strict=True)]
            sa = tuple(
                slice(i - s.start, j - s.start)
                for i, j, s in zip(lo, hi, ba, strict=True)
            )
            sb_ = tuple(
                slice(i - s.start, j - s.start)
                for i, j, s in zip(lo, hi, bb, strict=True)
            )
            x, y = da[sa], db[sb_]
            assert x.size > 0
            assert np.array_equal(np.isnan(x), np.isnan(y))
            f = np.isfinite(x) & np.isfinite(y)
            near = (
                np.abs(
                    np.abs(np.where(np.isfinite(x), x, np.where(np.isfinite(y), y, 0)))
                    - cap
                )
                <= W.TOLERANCE
            )
            assert np.array_equal(np.isinf(x) & ~near, np.isinf(y) & ~near)
            diff = np.abs(x[f] - y[f])
            assert np.all(diff <= W.TOLERANCE)
            x32, y32 = x[f].astype(np.float32), y[f].astype(np.float32)
            tol = np.float32(W.TOLERANCE) + np.spacing(np.abs(x32))
            assert np.all(np.abs(x32 - y32) <= tol)
            n32 += int(np.sum(x32 != y32))
            n64 += int(np.sum(diff > 0))
            worst = max(worst, float(diff.max(initial=0.0)))
    print(f"two superblocks: {n64} float64, {n32} float32 differ; max {worst:.3g} mm")


def test_the_boxes(sampler: W.WMDistanceSampler) -> None:
    """The output box is blocks.superblock_box; the read box adds the cap,
    rounded outward per axis, and REACH voxels, clipped to the grid."""
    out, read = sampler.boxes((2, 3, 1), 0, halo_mm=0.5, cap=1.0)
    vs = sampler.voxel_mm
    for axis, i in enumerate((2, 3, 1)):
        h = int(np.ceil(0.5 / vs[axis] - 1e-9))
        assert out[axis] == slice(i * 8 - h, (i + 1) * 8 + 1 + h)
        w = int(np.ceil((0.5 + W.MARGIN) / vs[axis] - 1e-9)) + W.REACH
        assert read[axis] == slice(max(i * 8 - w, 0), min((i + 1) * 8 + 1 + w, 48))


def test_the_call_refuses_bad_parameters(sampler: W.WMDistanceSampler) -> None:
    for kw in (
        {"cap": 0.0},
        {"cap": W.CAP + 0.1},
        {"halo_mm": -1.0},
        {"halo_mm": float("inf")},
    ):
        with pytest.raises(ValueError):
            sampler.sample((0, 0, 0), 0, **kw)
    with pytest.raises(ValueError):
        sampler.sample((0, -1, 0), 0)
    with pytest.raises(ValueError):
        sampler.sample((0, 0, 0), -1)


def test_the_read_factor() -> None:
    assert W.read_factor(10.0, 0.0) == 1.0
    assert W.read_factor(10.0, 5.0) == pytest.approx(8.0)
