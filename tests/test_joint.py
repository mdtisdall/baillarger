"""Tests for baillarger.joint (step 0: the starting seams) on the phantoms."""

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from baillarger import columns as cl
from baillarger import frame as fr
from baillarger import joint as jt
from baillarger import phantoms as ph

VOX = (0.16, 0.16, 0.16)
S26 = np.ones((3, 3, 3), bool)


def test_seam_on_the_dark_line() -> None:
    rim, flash, e = ph.p4_kissing_dark_line()
    seams, rep = jt.initial_seams(rim, flash, VOX)
    assert seams.any() and rep["seam_voxels"] == seams.sum()
    assert not (seams & (rim != 3)).any()
    near = ndi.binary_dilation(e["seam_target_mask"], S26)
    assert (seams & near).sum() >= 0.9 * seams.sum()
    # the banks are cut apart above the U bottom
    rim2 = jt.with_seams(rim, seams)
    up = np.zeros(rim.shape, bool)
    up[:, :, 24:] = True
    lab, _ = ndi.label((rim2 == 3) & up, S26)
    a = set(np.unique(lab[e["bank_a_mask"] & (rim2 == 3) & up]).tolist()) - {0}
    b = set(np.unique(lab[e["bank_b_mask"] & (rim2 == 3) & up]).tolist()) - {0}
    assert a and b and not a & b


def test_no_seams_where_none_is_needed() -> None:
    for phantom in (ph.p2_sulcus_end, ph.p3_stria_band, ph.p5_wm_handle):
        rim, flash, _ = phantom()
        seams, _ = jt.initial_seams(rim, flash, VOX)
        assert not seams.any(), phantom.__name__


def test_seams_keep_out_of_missing_tissue() -> None:
    rim, flash, e = ph.p4_kissing_dark_line()
    missing = np.zeros(rim.shape, bool)
    missing[:, 14:20, 30:50] = True  # a block across the touching patch
    missing &= (rim == 2) | (rim == 3)
    seams, _ = jt.initial_seams(rim, flash, VOX, missing)
    assert seams.any() and not (seams & missing).any()


def test_seams_bring_fused_columns_into_the_model() -> None:
    """In P1's touching patch the streamlines meet inside the cortex and have
    no pial depth. With the seams as fluid, a bank whose streamlines cross the
    seam before the ridge exits there and joins the model. P1 has no dark
    line, and the cut lies at x = 28, off the ridge at x = 31.5: bank B's
    streamlines reach the ridge first and still do not exit (the seams as
    barriers, a frame change, are for step 7)."""
    rim, flash, e = ph.p1_kissing_banks()
    F = fr.build_frame(rim, VOX)
    seams, _ = jt.initial_seams(rim, flash, VOX)
    seam_x = np.unique(np.nonzero(seams)[0])
    assert seam_x.tolist() == [28]
    x = np.arange(rim.shape[0])[:, None, None]
    patch = e["patch_mask"] & ~seams
    side_a, side_b = patch & (x < 28), patch & (x > 28)

    def covered(rim_x: npt.NDArray[np.int16]) -> npt.NDArray[np.bool_]:
        D, _ = cl.column_data(F, rim_x, flash)
        inm = np.zeros(rim.size, bool)
        inm[D.index[D.ctx]] = True
        return inm.reshape(rim.shape)

    before = covered(rim.astype(np.int16))
    after = covered(jt.with_seams(rim, seams))
    assert before[patch].mean() < 0.2
    assert after[side_a].mean() > 0.8
    assert after[side_b].mean() < 0.5
    # the frame itself is the same with the seams (cortex and fluid alike)
    F2 = fr.build_frame(jt.with_seams(rim, seams), VOX)
    assert np.array_equal(F.depth, F2.depth, equal_nan=True)


def test_with_seams() -> None:
    rim = np.array([[[3, 3, 2, 1, 0]]], np.int16)
    seams = np.array([[[True, False, False, False, False]]])
    assert jt.with_seams(rim, seams).tolist() == [[[1, 3, 2, 1, 0]]]
    assert rim.tolist() == [[[3, 3, 2, 1, 0]]]  # unchanged
