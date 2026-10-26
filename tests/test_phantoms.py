"""Tests for baillarger.phantoms: labels, topology of the WM and the CSF, and
the expected masks."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from scipy import ndimage as ndi

from baillarger import phantoms as ph
from baillarger.topology import euler_26_6

type Bool3 = npt.NDArray[np.bool_]
type PhantomFn = Callable[[], ph.Phantom]

S26 = np.ones((3, 3, 3), bool)
KEYS = ("seam_required", "seam_forbidden_mask", "seam_target_mask")
ALL = [
    ph.p1_kissing_banks,
    ph.p2_sulcus_end,
    ph.p3_stria_band,
    ph.p4_kissing_dark_line,
    ph.p5_wm_handle,
]
KISS = [ph.p1_kissing_banks, ph.p4_kissing_dark_line]
NO_SEAM = [ph.p2_sulcus_end, ph.p3_stria_band, ph.p5_wm_handle]
BOTTOM_FREE_Z = 24  # above the cortex that covers the U bottom (z < 24)


def n_components(mask: Bool3, structure: Bool3 = S26) -> int:
    return ndi.label(mask, structure=structure)[1]


@pytest.mark.parametrize("f", ALL)
def test_types_labels_and_size(f: PhantomFn) -> None:
    rim, flash, exp = f()
    assert rim.dtype == np.int16 and flash.dtype == np.float32
    assert rim.shape == flash.shape and rim.ndim == 3
    assert max(rim.shape) < 100
    assert set(np.unique(rim)) <= {0, 1, 2, 3}
    assert {1, 2, 3} <= set(np.unique(rim))
    assert ph.VOXEL_MM == 0.16


@pytest.mark.parametrize("f", ALL)
def test_wm_is_one_26_component(f: PhantomFn) -> None:
    rim, _, _ = f()
    assert n_components(rim == 2) == 1


@pytest.mark.parametrize("f", ALL)
def test_expected_keys_and_masks(f: PhantomFn) -> None:
    rim, _, exp = f()
    for k in KEYS:
        assert k in exp
    assert isinstance(exp["seam_required"], bool)
    for m in (exp["seam_forbidden_mask"], exp["seam_target_mask"]):
        assert m.shape == rim.shape and m.dtype == bool
    cortex = rim == 3
    assert not (exp["seam_forbidden_mask"] & ~cortex).any()
    assert not (exp["seam_target_mask"] & ~cortex).any()
    assert not (exp["seam_forbidden_mask"] & exp["seam_target_mask"]).any()


@pytest.mark.parametrize("f", ALL)
def test_reproducible(f: PhantomFn) -> None:
    a, b = f(), f()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    ea: dict[str, Any] = dict(a[2])
    eb: dict[str, Any] = dict(b[2])
    assert ea.keys() == eb.keys()
    for k, v in ea.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, eb[k])
        else:
            assert v == eb[k]


@pytest.mark.parametrize("f", ALL)
def test_flash_intensities(f: PhantomFn) -> None:
    rim, flash, _ = f()
    # the dark band / dark sheet voxels (flash < 0.7) are excluded from the cortex
    for lab, mean in ((1, 0.1), (2, 0.8), (3, 1.0)):
        m = (rim == lab) & (flash > 0.7 if lab == 3 else True)
        assert abs(flash[m].mean() - mean) < 0.01
        assert abs(flash[m].std() - 0.05) < 0.01


@pytest.mark.parametrize("f", KISS)
def test_csf_is_6_connected_to_the_border(f: PhantomFn) -> None:
    rim, _, exp = f()
    csf = rim == 1
    lab, n = ndi.label(csf)  # default structure is 6-connectivity
    border = np.zeros_like(csf)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    border[:, :, 0] = border[:, :, -1] = True
    touching = np.unique(lab[border & csf])
    assert set(range(1, n + 1)) == set(touching[touching > 0])
    # the gap, deep inside the U and away from the touching patch, is open at both ends
    xs = np.where(csf[:, 5, 30])[0]
    mid = xs[(xs > 20) & (xs < rim.shape[0] - 20)]
    assert len(mid) > 5
    assert lab[mid[0], 5, 30] == lab[mid[0], 0, 30] == lab[mid[0], rim.shape[1] - 1, 30]


@pytest.mark.parametrize("f", KISS)
def test_touching_patch_has_no_csf(f: PhantomFn) -> None:
    rim, _, exp = f()
    patch = exp["patch_mask"]
    assert patch.any()
    assert (rim[patch] == 3).all()
    # bounding box of the patch: no CSF between the banks at any (y, z) of the patch
    xs, ys, zs = (np.where(patch.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1)))
    box = rim[xs.min() : xs.max() + 1, ys.min() : ys.max() + 1, zs.min() : zs.max() + 1]
    assert not (box == 1).any()


@pytest.mark.parametrize("f", KISS)
def test_bank_masks_partition_the_cortex(f: PhantomFn) -> None:
    rim, _, exp = f()
    a, b = exp["bank_a_mask"], exp["bank_b_mask"]
    assert a.dtype == bool and b.dtype == bool and a.shape == rim.shape
    assert not (a & b).any()
    assert np.array_equal(a | b, rim == 3)
    assert a.any() and b.any()


@pytest.mark.parametrize("f", KISS)
def test_banks_are_joined_by_the_touching_patch(f: PhantomFn) -> None:
    rim, _, exp = f()
    a, b, patch = exp["bank_a_mask"], exp["bank_b_mask"], exp["patch_mask"]
    cortex = rim == 3
    up = np.zeros_like(cortex)
    up[:, :, BOTTOM_FREE_Z:] = True
    # above the U bottom the two banks are 26-connected only through the patch
    assert n_components(cortex & up) == 1
    assert n_components(cortex & up & ~patch) == 2
    # the two components are the two banks
    lab, _ = ndi.label(cortex & up & ~patch, structure=S26)
    for bank in (a, b):
        assert len(np.unique(lab[bank & up & ~patch])) == 1
    # with the whole cortex, the banks are also joined through the cortex on top
    # of the U bottom
    assert n_components(cortex & ~patch) == 1
    # the banks touch in the patch: a voxel of A has a 26-neighbour in B
    assert (ndi.binary_dilation(a, S26) & b & patch).any()


@pytest.mark.parametrize("f", KISS)
def test_seam_expectation_for_kissing_banks(f: PhantomFn) -> None:
    rim, _, exp = f()
    tgt, forb, patch = (
        exp["seam_target_mask"],
        exp["seam_forbidden_mask"],
        exp["patch_mask"],
    )
    assert exp["seam_required"] is True
    assert tgt.any() and forb.any()
    assert (tgt & ~patch).sum() == 0
    d = ndi.distance_transform_edt(~tgt)
    assert np.array_equal(forb, (rim == 3) & (d > 3))
    # the sheet separates the banks: no 26-path between them through cortex minus
    # the sheet, above the bottom
    cortex = rim == 3
    up = np.zeros_like(cortex)
    up[:, :, BOTTOM_FREE_Z:] = True
    assert n_components(cortex & up & ~tgt) >= 2
    # sheet thickness along x is 1-2 voxels in every column of the patch
    per_col = tgt.sum(axis=0)
    cols = per_col[per_col > 0]
    assert cols.min() >= 1 and cols.max() <= 2


def test_p1_target_is_midway_between_the_wm_plates() -> None:
    rim, _, exp = ph.p1_kissing_banks()
    xs = np.where(exp["seam_target_mask"].any(axis=(1, 2)))[0]
    wm_x = np.where((rim == 2).any(axis=(1, 2)))[0]
    left = wm_x[wm_x < rim.shape[0] // 2].max() + 1
    right = wm_x[wm_x >= rim.shape[0] // 2].min()
    assert abs(xs.mean() + 0.5 - (left + right) / 2) <= 1.0
    assert len(xs) == 2


def test_p4_banks_differ_by_1p5_and_the_sheet_is_dark_and_off_centre() -> None:
    rim, flash, exp = ph.p4_kissing_dark_line()
    ta = exp["bank_a_mask"][:, 2, 30].sum()
    tb = (exp["bank_b_mask"][:, 2, 30]).sum()
    assert tb == pytest.approx(1.5 * ta)
    tgt = exp["seam_target_mask"]
    assert (rim[tgt] == 3).all()
    assert abs(flash[tgt].mean() - 0.3) < 0.03
    assert len(np.unique(np.where(tgt)[0])) == 1  # one voxel thick
    # not midway between the WM plates
    wm_x = np.where((rim == 2).any(axis=(1, 2)))[0]
    left = wm_x[wm_x < rim.shape[0] // 2].max() + 1
    right = wm_x[wm_x >= rim.shape[0] // 2].min()
    x = np.where(tgt)[0][0]
    assert abs(x + 0.5 - (left + right) / 2) > 3
    # the banks are cut at the sheet
    assert not (exp["bank_b_mask"] & tgt).any() or not (exp["bank_a_mask"] & tgt).any()


def test_p1_has_no_dark_voxels() -> None:
    rim, flash, _ = ph.p1_kissing_banks()
    assert flash[rim == 3].min() > 0.7


@pytest.mark.parametrize("f", NO_SEAM)
def test_no_seam_phantoms(f: PhantomFn) -> None:
    rim, _, exp = f()
    assert exp["seam_required"] is False
    assert np.array_equal(exp["seam_forbidden_mask"], rim == 3)
    assert not exp["seam_target_mask"].any()


def test_p2_pocket_stops_inside_the_cortex() -> None:
    rim, _, _ = ph.p2_sulcus_end()
    wm_top = np.where((rim == 2).any(axis=(0, 1)))[0].max()
    cortex_z = np.where((rim == 3).any(axis=(0, 1)))[0]
    top = cortex_z.max()
    pocket = rim == 1
    pocket[:, :, top + 1 :] = False  # remove the outside above the slab
    pocket[:, :, : wm_top + 1] = False  # and the outside below the WM
    assert pocket.any()
    zp = np.where(pocket.any(axis=(0, 1)))[0]
    assert zp.max() == top  # enters from the outer surface
    assert zp.min() > wm_top + 1  # stops above the WM: does not reach the far face
    # every pocket column has cortex under it, down to the WM
    cols = pocket.any(axis=2)
    below = (rim == 3)[cols]
    assert below[:, wm_top + 1 : zp.min()].all()
    # the pocket is wholly surrounded by cortex sideways too: no pocket voxel on
    # the volume border
    assert not (
        pocket[0].any() or pocket[-1].any() or pocket[:, 0].any() or pocket[:, -1].any()
    )


def test_p3_dark_band_is_2_voxels_at_60_percent_depth() -> None:
    rim, flash, _ = ph.p3_stria_band()
    c = rim == 3
    z = np.where(c.any(axis=(0, 1)))[0]
    z0, z1 = z.min(), z.max() + 1
    dark_z = np.where(((flash < 0.7) & c).any(axis=(0, 1)))[0]
    assert len(dark_z) == 2 and dark_z[1] - dark_z[0] == 1
    assert (dark_z[0] - z0, dark_z[1] + 1 - z0) == (
        pytest.approx(0.6 * (z1 - z0) - 1),
        pytest.approx(0.6 * (z1 - z0) + 1),
    )
    band = np.zeros_like(c)
    band[:, :, dark_z] = True
    assert (rim[band & c] == 3).all()
    assert abs(flash[band & c].mean() - 0.5) < 0.01
    assert (flash[c & ~band] > 0.7).all()


def test_p5_torus_topology() -> None:
    rim, _, exp = ph.p5_wm_handle()
    chi = euler_26_6(rim == 2)
    assert chi == 0 == exp["wm_euler"]
    assert exp["genus"] == 1 == 1 - chi
    # the cortex shell leaves the hole open: the outside inside the hole joins the
    # outside above
    assert n_components(rim == 1, structure=ndi.generate_binary_structure(3, 1)) == 1
    # the cortex is one shell
    assert n_components(rim == 3) == 1
    assert euler_26_6(rim == 3) == 0


# ---------------------------------------------------------------- P6-P9


def test_p6_band_is_in_the_thick_bank_and_the_line_is_gone() -> None:
    rim, flash, e, band, mid, banks = ph.p6_fused_stria()
    assert flash.dtype == np.float32
    assert band.any() and not (band & ~e["bank_b_mask"]).any()
    assert len(np.unique(np.nonzero(band)[0])) == 3  # 3 voxels thick
    assert abs(flash[band].mean() - 0.75) < 0.01
    assert flash[e["seam_target_mask"]].min() > 0.7  # no dark line
    assert not (banks[0] & banks[1]).any()
    assert np.array_equal(banks[0] | banks[1], rim == 3)  # mid is not on a voxel
    assert mid % 1 == 0.5


def test_p7_slit_is_cortex_and_dark() -> None:
    rim, flash, slit = ph.p7_visible_slit()
    assert rim.dtype == np.int16 and flash.dtype == np.float32
    assert slit.any() and (rim[slit] == 3).all()
    assert abs(flash[slit].mean() - 0.3) < 0.01
    assert (flash[(rim == 3) & ~slit] > 0.7).all()
    # no outside label inside the slab: the slit is visible only in the image
    z = np.nonzero((rim == 3).any(axis=(0, 1)))[0]
    assert not (rim[:, :, z.min() : z.max() + 1] == 1).any()


def test_p8_contact_is_off_the_midpoint() -> None:
    rim, flash, e, contact, mid = ph.p8_natural_contact()
    assert e["seam_required"] is True
    assert contact - mid == -3.0
    patch = e["patch_mask"]
    xs = np.unique(np.nonzero(patch)[0])
    assert len(xs) == 2 and xs.mean() == contact  # the filled 2 voxel gap
    assert flash[rim == 3].min() > 0.7  # no dark line


def test_p9_holes_are_on_the_line_and_bright() -> None:
    rim, flash, e, holes = ph.p9_dark_line_with_holes()
    line = e["seam_target_mask"]
    assert holes.any() and not (holes & ~line).any()
    assert 0.25 < holes.sum() / line.sum() < 0.55
    assert abs(flash[holes].mean() - 1.0) < 0.01
    assert abs(flash[line & ~holes].mean() - 0.3) < 0.01
