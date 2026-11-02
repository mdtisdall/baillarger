"""Tests for the dseg command (baillarger.white.wm_topo, write_wm_topo): the
segmentation with its WM mask made cavity- and handle-free, on a synthetic
aseg."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import bids, white
from baillarger import handles as H
from baillarger.handles import kuhn_betti

SHAPE = (40, 36, 30)
VOX = (0.2, 0.16, 0.2)
FILLED = white.FILLED_LABELS["R"]


def synthetic_aseg() -> npt.NDArray[np.int16]:
    """Right hemisphere: a WM block drilled by a 2 x 2 tunnel of cortex (a
    fill of 4), holding a ventricle and an enclosed pocket of background (a
    cavity of 8); a WM ring 5 wide closed by a one-voxel rod across a slit of
    background (a cut of 1); a patch of hippocampus."""
    a = np.zeros(SHAPE, np.int16)
    a[8:24, 6:22, 4:20] = 42
    a[10:22, 8:20, 6:18] = 41
    a[15:17, 13:15, 6:18] = 42  # the tunnel
    a[11:13, 9:11, 8:10] = 43  # a ventricle inside the WM
    a[19:21, 9:11, 12:14] = 0  # the enclosed pocket
    a[8:10, 6:10, 4:8] = 53
    a[25:37, 3:33, 22:28] = 41  # the ring round a 2 x 20 hole
    a[30:32, 8:28, 22:28] = 0
    a[25:30, 16:18, 22:28] = 0  # the slit
    a[27, 16:18, 25] = 41  # the rod across it
    return a


def test_wm_topo_of_a_synthetic_aseg() -> None:
    a = synthetic_aseg()
    t = white.wm_topo(a, VOX)
    assert t.hemi == "R"
    assert t.betti_before == (2, 2, 1) and t.betti_after == (2, 0, 0)
    assert kuhn_betti(np.isin(t.dseg, FILLED)) == (2, 0, 0)
    cav, fill, cut = (np.argwhere(t.fixes == k) for k in (3, 2, 1))
    # the pocket, set to WM
    assert len(cav) == 8 and (t.dseg[19:21, 9:11, 12:14] == 41).all()
    # the plug of the tunnel: 2 x 2 voxels in one slice
    assert len(fill) == 4 and len(np.unique(fill[:, 2])) == 1
    assert {tuple(p[:2]) for p in fill} == {(15, 13), (15, 14), (16, 13), (16, 14)}
    assert all(t.dseg[tuple(p)] == 41 for p in fill)
    # one voxel of the rod, given the label of the slit round it
    assert len(cut) == 1 and cut[0][0] == 27 and cut[0][2] == 25
    assert t.dseg[tuple(cut[0])] == 0 and t.cut_labels == {0: 1}
    # nothing else changed: the ventricle and the hippocampus are kept
    assert np.array_equal(t.dseg[t.fixes == 0], a[t.fixes == 0])
    assert (t.dseg == 43).sum() == 8 and (t.dseg == 53).sum() == 32
    assert sorted(m["Kind"] for m in t.moves) == ["cut", "fill"]
    assert sum(m["Voxels"] for m in t.moves) == 5


def test_one_hemisphere_only() -> None:
    a = synthetic_aseg()
    a[0, 0, 0] = 2  # the left WM too
    with pytest.raises(ValueError, match="exactly one"):
        white.wm_topo(a, VOX)
    with pytest.raises(ValueError, match="exactly one"):
        white.hemisphere(np.full((3, 3, 3), 42))


def test_a_fill_that_joins_two_pieces_into_a_handle_is_refused() -> None:
    """remove_handles fixes each piece of the mask with the others as background,
    so a fill of one piece can touch another. Here a WM plate has a 7 x 7 hole
    (cheaper to fill than to cut), and a second piece of WM, a U, passes both
    legs through it: the plug touches both legs, and plate, plug and U make a
    handle. wm_topo refuses the result rather than write it."""
    a = np.zeros((40, 40, 40), np.int16)
    a[2:38, 2:38, 18:22] = 41  # the plate, 4 thick
    a[16:23, 16:23, 18:22] = 42  # the hole, cortex
    a[18, 19, 14:26] = 41  # the U's legs, 2 voxels from the hole's walls
    a[20, 19, 14:26] = 41
    a[18:21, 19, 25] = 41  # joined 3 voxels above the plate
    assert kuhn_betti(np.isin(a, FILLED)) == (2, 1, 0)
    with pytest.raises(RuntimeError, match=r"Betti numbers \(1, 1, 0\)"):
        white.wm_topo(a, VOX)


def test_cuts_take_the_nearest_donor_in_mm() -> None:
    labels = np.full((30, 30, 30), 41, np.int16)
    donor = np.zeros(labels.shape, bool)
    cut = np.zeros(labels.shape, bool)
    cut[15, 15, 15] = True
    # 3 voxels along x (0.3 mm) against 2 along y (0.4 mm)
    labels[18, 15, 15], donor[18, 15, 15] = 7, True
    labels[15, 13, 15], donor[15, 13, 15] = 8, True
    white.relabel_cuts(labels, cut, donor, (0.1, 0.2, 0.2), margin=1)
    assert labels[15, 15, 15] == 7
    # a donor far beyond the first box: the box is widened
    labels = np.full((30, 30, 30), 41, np.int16)
    donor = np.zeros(labels.shape, bool)
    labels[0, 0, 0], donor[0, 0, 0] = 9, True
    white.relabel_cuts(labels, cut, donor, (1.0, 1.0, 1.0), margin=1)
    assert labels[15, 15, 15] == 9


def test_the_dseg_command_writes_the_bids_files(tmp_path: Path) -> None:
    src = tmp_path / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir()
    nib.save(nib.nifti1.Nifti1Image(synthetic_aseg(), np.diag([*VOX, 1.0])), src)
    out = tmp_path / "deriv"
    white.main(["dseg", str(src), "--out", str(out)])
    anat = out / "sub-T01" / "ses-S1" / "anat"
    stem = "sub-T01_ses-S1_desc-wmtopo_dseg"
    fx = "sub-T01_ses-S1_desc-wmtopofixes_dseg"
    assert (out / "dataset_description.json").exists()
    img = nib.load(anat / f"{stem}.nii.gz")
    assert isinstance(img, nib.nifti1.Nifti1Image)
    d = np.asarray(img.dataobj)
    assert d.dtype == np.int16 and np.allclose(img.affine, np.diag([*VOX, 1.0]))
    assert kuhn_betti(np.isin(d, FILLED)) == (2, 0, 0)
    side = bids.read_sidecar(anat / f"{stem}.json", white.DSEG_SPEC)
    assert side["BettiBefore"] == [2, 2, 1] and side["BettiAfter"] == [2, 0, 0]
    assert side["CavityVoxels"] == 8 and side["HandleFillVoxels"] == 4
    assert side["HandleCutVoxels"] == 1 and side["CutLabels"] == {"0": 1}
    assert side["Sources"] == [str(src.resolve())]
    assert side["SourcesSHA256"][str(src.resolve())] == bids.file_sha256(src)
    tsv = (anat / f"{stem}.tsv").read_text().splitlines()
    assert tsv[0] == "index\tname" and "41\tRight-Cerebral-White-Matter" in tsv
    assert "53\tRight-Hippocampus" in tsv
    fimg = nib.load(anat / f"{fx}.nii.gz")
    assert isinstance(fimg, nib.nifti1.Nifti1Image)
    f = np.asarray(fimg.dataobj)
    assert [(f == k).sum() for k in (1, 2, 3)] == [1, 4, 8]
    bids.read_sidecar(anat / f"{fx}.json", white.FIXES_SPEC)
    assert json.loads((anat / f"{fx}.json").read_text())["SpecVersion"] == (
        "wm-dseg-fixes/1"
    )


def test_a_cavity_holding_cortex_is_refused() -> None:
    a = synthetic_aseg()
    a[19:21, 9:11, 12:14] = 42  # the enclosed pocket is cortex
    with pytest.raises(
        ValueError, match=r"hold cortex \(1; voxels, centre\): 8, \[20, 10, 12\]"
    ):
        white.wm_topo(a, VOX)


def test_thickening_adds_wm_and_keeps_the_topology(tmp_path: Path) -> None:
    a = synthetic_aseg()
    t = white.wm_topo(a, VOX, scale_mm=0.4)
    assert t.betti_after == (2, 0, 0)
    assert kuhn_betti(np.isin(t.dseg, FILLED)) == (2, 0, 0)
    assert t.thickened_voxels == (t.fixes == 5).sum() > 0 and t.thicken_passes > 0
    assert (t.dseg[t.fixes == 5] == 41).all() and (a[t.fixes == 5] != 41).all()
    assert sum(x["Voxels"] for x in t.exceptions) == (t.fixes == 6).sum()
    assert (t.fixes == 2).sum() == 4  # the tunnel's plug, as without thickening
    src = tmp_path / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir()
    nib.save(nib.nifti1.Nifti1Image(a, np.diag([*VOX, 1.0])), src)
    out = tmp_path / "d"
    white.main(["dseg", str(src), "--out", str(out), "--scale-mm", "0.4"])
    anat = out / "sub-T01" / "ses-S1" / "anat"
    side = bids.read_sidecar(
        anat / "sub-T01_ses-S1_desc-wmtopo_dseg.json", white.DSEG_SPEC
    )
    assert side["ScaleMM"] == 0.4 and side["ThickenedVoxels"] == t.thickened_voxels
    rows = (anat / side["ExceptionsFile"]).read_text().splitlines()
    assert rows[0] == "index\tkind\tvoxels\ti\tj\tk"
    assert len(rows) == 1 + side["Exceptions"] == 1 + len(t.exceptions)


def test_well_composed_removes_the_diagonal_contacts() -> None:
    a = synthetic_aseg()
    a[22, 20, 18] = 41  # touches the block's WM corner (21, 19, 17) only there
    t = white.wm_topo(a, VOX, well_composed=True)
    assert t.betti_after == (2, 0, 0)
    assert not H.pinched(np.isin(t.dseg, FILLED)).any()
    assert t.composed_added == (t.fixes == 7).sum()
    assert t.composed_removed == (t.fixes == 8).sum()
    assert t.composed_added + t.composed_removed > 0
