"""Tests for baillarger.growth on the phantoms of baillarger.phantoms."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy import ndimage as ndi

from baillarger import phantoms as ph
from baillarger.growth import (
    grow_competing,
    grow_topological,
    mincut_seam,
    seam_priority,
    sheetness,
    thin_seam,
    visible_sheets,
)
from baillarger.topology import CONN_26_6, euler_26_6

type Bool3 = npt.NDArray[np.bool_]
type Banks = dict[str, Bool3]

S26 = np.ones((3, 3, 3), bool)


def run(
    rim: ph.Rim, flash: ph.Flash, dark_weight: float = 30.0
) -> tuple[Bool3, Bool3, npt.NDArray[np.int64]]:
    """grow_topological from the WM into the cortex, with seam_priority."""
    pri = seam_priority(rim, flash, ph.VOXEL_MM, dark_weight)
    return grow_topological(rim == 2, rim == 3, pri, CONN_26_6)


def banks_separated_above_bottom(
    rim: ph.Rim, e: ph.Expected | Banks, obj: Bool3, z_min: int = 24
) -> bool:
    """The two banks' object voxels above z_min are in different 26-components."""
    m = obj.copy()
    m[:, :, :z_min] = False
    m &= rim == 3
    lab, _ = ndi.label(m, S26)
    la = set(np.unique(lab[e["bank_a_mask"] & m])) - {0}
    lb = set(np.unique(lab[e["bank_b_mask"] & m])) - {0}
    return la.isdisjoint(lb)


def far_banks(e: ph.Expected, margin: int = 6) -> Banks:
    """The parts of the two banks farther than margin voxels from the touching
    patch, so that a separation test does not depend on where in the patch the
    cut is."""
    far = ~ndi.binary_dilation(e["patch_mask"], S26, iterations=margin)
    return dict(bank_a_mask=e["bank_a_mask"] & far, bank_b_mask=e["bank_b_mask"] & far)


# ---------------------------------------------------------------- grow_topological


def test_p1_kissing_banks_are_separated() -> None:
    rim, flash, e = ph.p1_kissing_banks()
    obj, seam, _ = run(rim, flash)
    assert seam.any()
    assert banks_separated_above_bottom(rim, e, obj)
    # the seam lies in the touching patch, not elsewhere
    assert not (seam & e["seam_forbidden_mask"]).any()


def test_p2_no_seam_at_sulcus_end() -> None:
    rim, flash, _ = ph.p2_sulcus_end()
    obj, seam, _ = run(rim, flash)
    assert not seam.any()
    assert obj[rim == 3].all()


def test_p3_no_seam_on_stria() -> None:
    rim, flash, _ = ph.p3_stria_band()
    obj, seam, _ = run(rim, flash)
    assert not seam.any()
    assert obj[rim == 3].all()


def test_p4_seam_on_the_dark_line() -> None:
    rim, flash, e = ph.p4_kissing_dark_line()
    obj, seam, _ = run(rim, flash)
    assert seam.any()
    assert banks_separated_above_bottom(rim, e, obj)
    near = ndi.binary_dilation(e["seam_target_mask"], S26)
    assert (seam & near).sum() >= 0.8 * seam.sum()


def test_p5_keeps_the_wm_handle() -> None:
    rim, flash, e = ph.p5_wm_handle()
    obj, seam, _ = run(rim, flash)
    assert euler_26_6(obj) == euler_26_6(rim == 2) == e["wm_euler"]
    assert not seam.any()


def test_euler_characteristic_never_changes() -> None:
    for phantom in (ph.p1_kissing_banks, ph.p4_kissing_dark_line, ph.p5_wm_handle):
        rim, flash, _ = phantom()
        obj, _, order = run(rim, flash)
        cur = rim == 2
        chi0 = euler_26_6(cur)
        flat = cur.ravel()
        for k in range(0, len(order), 1000):
            flat[order[k : k + 1000]] = True
            assert euler_26_6(cur) == chi0
        assert np.array_equal(cur, obj)


def test_p4_without_darkness_reports_seam_location() -> None:
    """Report only: where the seam goes when the image is not used."""
    rim, flash, e = ph.p4_kissing_dark_line()
    _, seam, _ = run(rim, flash, dark_weight=0.0)
    xs = np.nonzero(seam)[0]
    tx = np.nonzero(e["seam_target_mask"])[0]
    print(
        f"\nP4 without darkness: seam x median {np.median(xs) if len(xs) else None}, "
        f"dark line x {np.median(tx)}, seam voxels {seam.sum()}"
    )


def test_p6_fused_without_line_seam_not_on_stria() -> None:
    rim, flash, e, band, mid, banks = ph.p6_fused_stria()
    obj, seam, _ = run(rim, flash)
    assert seam.any()
    split = dict(bank_a_mask=banks[0], bank_b_mask=banks[1])
    assert banks_separated_above_bottom(rim, split, obj)
    assert not (seam & ndi.binary_dilation(band, S26)).any()
    xs = np.nonzero(seam)[0]
    print(
        f"\nP6: seam x median {np.median(xs)}, midpoint {mid}, "
        f"stria x {np.unique(np.nonzero(band)[0])}"
    )
    assert abs(np.median(xs) - mid) <= 3


def test_p6_band_is_like_a_real_stria() -> None:
    """The V1V2 stria has sheetness below 0.11 for 99 % of its voxels; the band
    of P6 must be in that range, or P6 tests a harder case than the data has."""
    rim, flash, _, band, _, _ = ph.p6_fused_stria()
    assert np.percentile(sheetness(flash, rim == 3)[band], 99) < 0.12


def test_p7_visible_slit_stays_open() -> None:
    rim, flash, slit = ph.p7_visible_slit()
    forbid = visible_sheets(rim, flash, ph.VOXEL_MM)
    pri = seam_priority(rim, flash, ph.VOXEL_MM)
    obj, seam, _ = grow_topological(rim == 2, (rim == 3) & ~forbid, pri, CONN_26_6)
    seam = seam | forbid
    # below the partial-volume edge
    deep = slit & (np.arange(rim.shape[2])[None, None, :] < 24)
    assert (seam & deep).sum() >= 0.9 * deep.sum()
    assert not (seam & ~ndi.binary_dilation(slit, S26, iterations=2)).any()
    assert euler_26_6(obj) == euler_26_6(rim == 2)


# ---------------------------------------------------------------- grow_competing


def test_p1_fused_banks_get_a_seam_at_the_midpoint() -> None:
    rim, flash, e = ph.p1_kissing_banks()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    assert fluid.any()
    assert banks_separated_above_bottom(rim, e, tissue)
    assert not (fluid & e["seam_forbidden_mask"]).any()


def test_p2_p3_p5_no_seam() -> None:
    for phantom in (ph.p2_sulcus_end, ph.p3_stria_band, ph.p5_wm_handle):
        rim, flash, _ = phantom()
        tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
        assert not fluid.any(), phantom.__name__
        assert euler_26_6(tissue) == euler_26_6(rim == 2)


def test_p4_fluid_follows_the_dark_line() -> None:
    rim, flash, e = ph.p4_kissing_dark_line()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    assert banks_separated_above_bottom(rim, e, tissue)
    near = ndi.binary_dilation(e["seam_target_mask"], S26)
    assert (fluid & near).sum() >= 0.9 * fluid.sum()


def test_p6_stria_in_a_fused_bank_does_not_draw_the_seam() -> None:
    # where the cut goes in a fused patch with no line depends on the thickness
    # model (P6's patch makes one bank about 5 mm thick there); the test asks only
    # that the banks are separated inside the patch and that the stria is not
    # touched
    rim, flash, e, band, _, _ = ph.p6_fused_stria()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    assert banks_separated_above_bottom(rim, far_banks(e), tissue)
    assert not (fluid & ~e["patch_mask"]).any()
    assert not (fluid & ndi.binary_dilation(band, S26)).any()


def test_p7_fluid_fills_a_visible_slit_without_csf() -> None:
    rim, flash, slit = ph.p7_visible_slit()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    # the slit's interior: below the partial-volume edge, and one voxel in from
    # its lateral ends and its tip, where the sheet measure fades (the fluid
    # stops about one voxel short there)
    y = np.arange(rim.shape[1])[None, :, None]
    z = np.arange(rim.shape[2])[None, None, :]
    deep = slit & (z < 24) & (z > 16) & (y > 10) & (y < 37)
    assert (fluid & deep).sum() >= 0.9 * deep.sum()
    assert not (fluid & ~ndi.binary_dilation(slit, S26, iterations=2)).any()
    assert euler_26_6(tissue) == euler_26_6(rim == 2)


def test_tissue_topology_is_kept() -> None:
    for phantom in (ph.p1_kissing_banks, ph.p4_kissing_dark_line):
        rim, flash, _ = phantom()
        tissue, _, _ = grow_competing(rim, flash, ph.VOXEL_MM)
        assert euler_26_6(tissue) == euler_26_6(rim == 2)


@pytest.mark.xfail(
    reason="open problem: where the cut goes in a fused stretch with no visible "
    "line. Thickness-normalized and too-thick costs did not move it to the "
    "contact.",
    strict=False,
)
def test_p8_cut_goes_to_the_natural_contact_not_the_midpoint() -> None:
    rim, flash, e, contact, mid = ph.p8_natural_contact()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    assert banks_separated_above_bottom(rim, far_banks(e), tissue)
    x = np.median(np.nonzero(fluid)[0])
    print(f"\nP8: contact x {contact}, midpoint x {mid}, cut x {x}")
    assert abs(x - contact) <= 1


# ---------------------------------------------------------------- thin_seam


def test_thin_seam_shrinks_a_thick_cut_and_keeps_the_separation() -> None:
    rim, flash, e = ph.p1_kissing_banks()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    # make the cut 7 voxels thick inside the patch, as the growth can when fronts
    # meet broadly
    thick = (
        ndi.binary_dilation(fluid, np.ones((7, 1, 1), bool))
        & e["patch_mask"]
        & (rim == 3)
    )
    t2, s2 = tissue & ~thick, fluid | thick
    t3, s3 = thin_seam(t2, s2, flash, rim == 3)
    # back to about the size of the cut the growth made; where in the patch the
    # thin sheet lies is free, so the banks are judged far from the patch
    assert s3.sum() <= 1.5 * fluid.sum()
    assert banks_separated_above_bottom(rim, far_banks(e), t3)
    assert euler_26_6(t3) == euler_26_6(t2)


def test_thin_seam_keeps_dark_lines_and_slits() -> None:
    rim, flash, e = ph.p4_kissing_dark_line()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    t3, s3 = thin_seam(tissue, fluid, flash, rim == 3)
    target = e["seam_target_mask"]
    assert (s3 & target).sum() >= 0.9 * (fluid & target).sum()
    assert banks_separated_above_bottom(rim, e, t3)
    rim, flash, slit = ph.p7_visible_slit()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    t3, s3 = thin_seam(tissue, fluid, flash, rim == 3)
    assert (s3 & slit).sum() >= 0.9 * (fluid & slit).sum()


def test_thin_seam_moves_a_cut_onto_a_faint_line() -> None:
    """A thick cut over a faint line (0.75, too faint for the fluid) thins onto
    the line."""
    rim, flash, e = ph.p1_kissing_banks()
    flash = flash.copy()
    x = np.arange(rim.shape[0])[:, None, None]
    line = e["patch_mask"] & (x == 29)
    flash[line] = 0.75
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    thick = (
        ndi.binary_dilation(fluid | line, np.ones((9, 1, 1), bool))
        & e["patch_mask"]
        & (rim == 3)
    )
    t3, s3 = thin_seam(tissue & ~thick, fluid | thick, flash, rim == 3)
    assert banks_separated_above_bottom(rim, e, t3) or banks_separated_above_bottom(
        rim, far_banks(e), t3
    )
    assert (s3 & line).sum() >= 0.8 * s3.sum()
    assert euler_26_6(t3) == euler_26_6(tissue)


# ---------------------------------------------------------------- mincut_seam


def test_mincut_moves_cuts_onto_the_line_through_holes() -> None:
    rim, flash, e, holes = ph.p9_dark_line_with_holes()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    line = e["seam_target_mask"]
    near = ndi.binary_dilation(line, S26)
    before = (fluid & near).sum() / fluid.sum()
    t3, s3, rep = mincut_seam(tissue, fluid, flash, rim == 3)
    after = (s3 & near).sum() / s3.sum()
    print(
        f"\nP9 min-cut: seam within 1 voxel of the line: before {before:.2f} "
        f"({fluid.sum()} voxels), after {after:.2f} ({s3.sum()}); holes on the "
        f"seam {(s3 & holes).sum()} of {holes.sum()}; {rep}"
    )
    assert after >= 0.9
    # the holes are completed: the line is the cut through them too
    assert (s3 & holes).sum() >= 0.8 * holes.sum()
    assert banks_separated_above_bottom(rim, far_banks(e), t3)
    assert euler_26_6(t3) == euler_26_6(tissue)


def test_mincut_leaves_clean_cases_alone() -> None:
    for phantom in (ph.p2_sulcus_end, ph.p3_stria_band, ph.p5_wm_handle):
        rim, flash, _ = phantom()
        tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
        t3, s3, _ = mincut_seam(tissue, fluid, flash, rim == 3)
        assert not s3.any(), phantom.__name__
        assert np.array_equal(t3, tissue), phantom.__name__


def test_mincut_keeps_a_full_dark_line_and_the_fused_cut() -> None:
    rim, flash, e = ph.p4_kissing_dark_line()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    t3, s3, _ = mincut_seam(tissue, fluid, flash, rim == 3)
    near = ndi.binary_dilation(e["seam_target_mask"], S26)
    assert (s3 & near).sum() >= 0.9 * s3.sum()
    assert banks_separated_above_bottom(rim, far_banks(e), t3)
    assert euler_26_6(t3) == euler_26_6(tissue)
    rim, flash, e = ph.p1_kissing_banks()  # no line: the cut stays one thin sheet
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    t3, s3, _ = mincut_seam(tissue, fluid, flash, rim == 3)
    assert banks_separated_above_bottom(rim, far_banks(e), t3)
    assert s3.sum() <= 1.5 * fluid.sum()
    assert euler_26_6(t3) == euler_26_6(tissue)


def test_mincut_does_not_put_the_cut_on_the_stria() -> None:
    rim, flash, _, band, _, banks = ph.p6_fused_stria()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    t3, s3, _ = mincut_seam(tissue, fluid, flash, rim == 3)
    assert not (s3 & ndi.binary_dilation(band, S26)).any()
    split = dict(bank_a_mask=banks[0], bank_b_mask=banks[1])
    assert banks_separated_above_bottom(rim, split, t3)
    assert euler_26_6(t3) == euler_26_6(tissue)


def test_mincut_in_small_windows_matches_one_cut() -> None:
    """A large seam is cut in windows; on P9 small windows give the same cut as
    one."""
    rim, flash, _, _ = ph.p9_dark_line_with_holes()
    tissue, fluid, _ = grow_competing(rim, flash, ph.VOXEL_MM)
    _, s1, r1 = mincut_seam(tissue, fluid, flash, rim == 3)
    t5, s5, r5 = mincut_seam(tissue, fluid, flash, rim == 3, window=5)
    assert r1["cut"] == 1 and r5["cut"] > 1
    assert np.array_equal(s1, s5)
    assert euler_26_6(t5) == euler_26_6(tissue)
