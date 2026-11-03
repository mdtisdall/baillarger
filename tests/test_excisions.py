"""Tests for baillarger.excisions: the detector of excised tissue, the
relabel to outside, and the command's files."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import bids, excisions, white_dseg
from baillarger import frame as fr
from baillarger import phantoms as ph

VOX = (0.16, 0.16, 0.16)
type Labels = npt.NDArray[np.int16]
type F32 = npt.NDArray[np.float32]
type Bool = npt.NDArray[np.bool_]


def excised_slab() -> tuple[Labels, F32, Bool]:
    """P3's slab as a right-hemisphere aseg (WM 41, cortex 42, outside 0),
    with a block 14 x 14 x 18 where the FLASH is black (an excision through
    the WM and the cortex), and a black spot of 8 voxels: too small to be an
    excision."""
    rim, flash, _ = ph.p3_stria_band()
    aseg = np.zeros(rim.shape, np.int16)
    aseg[rim == 2] = 41
    aseg[rim == 3] = 42
    flash = flash.astype(np.float32)
    block = np.zeros(rim.shape, bool)
    block[16:30, 16:30, 4:22] = True
    flash[block] = 0.02
    flash[5:7, 40:42, 14:16] = 0.02
    return aseg, flash, block


def test_detect_finds_the_excision_only() -> None:
    aseg, flash, block = excised_slab()
    miss = excisions.detect(aseg, flash, VOX, "R")
    tissue = (aseg == 41) | (aseg == 42)
    core = block.copy()
    core[16:18] = core[28:30] = False  # smoothing softens the block's edges
    core[:, 16:18] = core[:, 28:30] = False
    core[:, :, 4:6] = core[:, :, 20:22] = False
    assert miss[core & tissue].all()
    assert not miss[~tissue].any()
    # within the margin round the block, and not at the small spot
    far = np.ones(aseg.shape, bool)
    far[12:34, 12:34, :] = False
    assert not miss[far].any()


def test_detect_reads_the_labels_of_the_hemisphere() -> None:
    """The same slab with left-hemisphere labels (WM 2, cortex 3) gives the
    same map with hemi L, and nothing with hemi R (no tissue of R)."""
    aseg, flash, _ = excised_slab()
    left = np.select([aseg == 41, aseg == 42], [2, 3], 0).astype(np.int16)
    want = excisions.detect(aseg, flash, VOX, "R")
    assert np.array_equal(excisions.detect(left, flash, VOX, "L"), want)
    with pytest.raises(ValueError, match="WM label of hemisphere R"):
        excisions.detect(left, flash, VOX, "R")


def test_frame_missing_tissue_is_the_detector() -> None:
    """frame.missing_tissue (rim labels: 2 WM, 3 cortex) gives the map of
    excisions.detect on the matching aseg."""
    aseg, flash, _ = excised_slab()
    rim = np.select([aseg == 41, aseg == 42], [2, 3], 1).astype(np.int16)
    assert np.array_equal(
        fr.missing_tissue(rim, flash, VOX), excisions.detect(aseg, flash, VOX, "R")
    )


def test_frame_missing_tissue_keeps_its_old_behaviour() -> None:
    """The old callers get what they got before the port: a box without
    cortex gives an empty mask (the old median was NaN), and a box without
    WM is not refused (the detector works on the cortex)."""
    aseg, flash, _ = excised_slab()
    rim = np.select([aseg == 41, aseg == 42], [2, 3], 1).astype(np.int16)
    no_cortex = np.where(rim == 3, 1, rim).astype(np.int16)
    assert not fr.missing_tissue(no_cortex, flash, VOX).any()
    no_wm = np.where(rim == 2, 1, rim).astype(np.int16)
    want = old_missing_tissue(no_wm, flash, VOX)
    assert want.any()
    assert np.array_equal(fr.missing_tissue(no_wm, flash, VOX), want)


def test_relabel_sets_the_mask_to_outside_in_a_copy() -> None:
    aseg, _, block = excised_slab()
    before = aseg.copy()
    out = excisions.relabel(aseg, block)
    assert np.array_equal(aseg, before)  # the input is not changed
    assert out.dtype == aseg.dtype
    assert (out[block] == 0).all()
    assert np.array_equal(out[~block], aseg[~block])


def old_missing_tissue(
    rim: npt.NDArray[np.int16],
    flash: npt.ArrayLike,
    voxel_mm: tuple[float, float, float],
) -> Bool:
    """The detector as frame.missing_tissue had it before the port (float64,
    the default structures of ndi.label and ndi.binary_dilation), with its
    default parameters: the reference for the port."""
    from scipy import ndimage as ndi

    f = np.asarray(flash, np.float64)
    tis = (rim == 2) | (rim == 3)
    gm_med = np.median(f[rim == 3])
    blk = tis & (ndi.gaussian_filter(f, 2.0) < 0.25 * gm_med)
    lab, n = ndi.label(blk)
    big = np.zeros(n + 1, bool)
    big[1:] = np.bincount(lab.ravel(), minlength=n + 1)[1:] >= 250
    miss: Bool = big[lab]
    it = int(round(0.3 / min(voxel_mm)))
    if it:
        miss = ndi.binary_dilation(miss, iterations=it) & tis
    return miss


def test_the_port_gives_the_old_mask() -> None:
    """On the excised slab, the port (float32) gives the same mask as the old
    detector (float64), voxel for voxel."""
    aseg, flash, _ = excised_slab()
    rim = np.select([aseg == 41, aseg == 42], [2, 3], 1).astype(np.int16)
    want = old_missing_tissue(rim, flash, VOX)
    assert want.sum() > 2000
    assert np.array_equal(excisions.detect(aseg, flash, VOX, "R"), want)


def test_the_excised_slab_closes_at_the_cut() -> None:
    """The excision is outside after relabel: the WM mask of the excised
    slab has a pit where the block was, and needs no fix (Betti (1, 0, 0))."""
    aseg, flash, _ = excised_slab()
    miss = excisions.detect(aseg, flash, VOX, "R")
    t = white_dseg.wm_topo(excisions.relabel(aseg, miss), VOX)
    assert t.betti_before == t.betti_after == (1, 0, 0)
    assert not t.fixes.any()
    assert not np.isin(t.dseg[miss], white_dseg.FILLED_LABELS["R"]).any()


def read(p: Path) -> npt.NDArray[np.generic]:
    img = nib.load(p)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    return np.asarray(img.dataobj)


def write_inputs(d: Path) -> tuple[Path, Path, Labels, Bool]:
    aseg, flash, _ = excised_slab()
    aff = np.diag([*VOX, 1.0])
    d.mkdir(parents=True, exist_ok=True)
    a = d / "sub-T01_ses-S1_aseg.nii.gz"
    f = d / "sub-T01_ses-S1_echo-2_FLASH.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(aseg, aff), a)
    nib.save(nib.nifti1.Nifti1Image(flash.astype(np.float32), aff), f)
    return a, f, aseg, excisions.detect(aseg, flash, VOX, "R")


def test_the_command_writes_the_bids_files(tmp_path: Path) -> None:
    a, f, aseg, miss = write_inputs(tmp_path / "in")
    out = tmp_path / "deriv"
    excisions.main([str(a), str(f), "--hemi", "R", "--out", str(out)])
    anat = out / "sub-T01/ses-S1/anat"
    dseg = anat / "sub-T01_ses-S1_desc-excised_dseg.nii.gz"
    mask = anat / "sub-T01_ses-S1_desc-excisions_mask.nii.gz"
    x, m = read(dseg), read(mask)
    assert x.dtype == aseg.dtype and m.dtype == np.uint8
    assert np.array_equal(m.astype(bool), miss)
    assert np.array_equal(x, excisions.relabel(aseg, miss))
    tsv = (anat / "sub-T01_ses-S1_desc-excised_dseg.tsv").read_text().split("\n")
    assert tsv[0] == "index\tname" and "41\tRight-Cerebral-White-Matter" in tsv
    spec = excisions.EXCISIONS_SPEC
    sides = [bids.read_sidecar(bids.sidecar_path(p), spec) for p in (dseg, mask)]
    for side in sides:
        assert side["Hemisphere"] == "R" and side["TissueLabels"] == [41, 42]
        assert side["ExcisedVoxels"] == miss.sum()
        assert sum(side["ExcisedByLabel"].values()) == miss.sum()
        assert set(side["ExcisedByLabel"]) <= {"41", "42"}
        assert side["CortexVoxels"] == (aseg == 42).sum()
        assert side["Components"] == 1
        assert side["Detector"] == dict(
            Frac=0.25, Sigma=2.0, MinVoxels=250, MarginMM=0.3, DilationSteps=2
        )
        assert side["CortexMedian"] == pytest.approx(
            float(np.median(read(f)[aseg == 42]))
        )
        assert len(side["Sources"]) == 2
    assert {k: v for k, v in sides[0].items() if k != "Description"} == {
        k: v for k, v in sides[1].items() if k != "Description"
    }
    assert (
        json.loads((out / "dataset_description.json").read_text())["DatasetType"]
        == "derivative"
    )


def test_the_command_refuses_bad_inputs(tmp_path: Path) -> None:
    a, f, aseg, _ = write_inputs(tmp_path / "in")
    out = tmp_path / "deriv"
    # the hemisphere's WM label is not in the aseg
    with pytest.raises(ValueError, match="WM label of hemisphere L"):
        excisions.write_excisions(a, f, "L", out)
    # a FLASH on another grid
    g = tmp_path / "in" / "sub-T01_ses-S1_echo-2_acq-x_FLASH.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), g)
    with pytest.raises(ValueError, match="grid"):
        excisions.write_excisions(a, g, "R", out)
    assert not out.exists()
