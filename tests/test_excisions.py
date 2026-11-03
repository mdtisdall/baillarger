"""Tests for baillarger.excisions: the detector of excised tissue, the
fragments and crumbs, the relabel to outside, and the command's files."""

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


# ---------------------------------------------------------------- fragments, crumbs

#: the volume of a voxel of VOX, mm3
V1 = 0.16**3


def test_fragments_below_the_limit_are_dropped() -> None:
    """At 0.004096 mm3 a voxel, 24 voxels (0.098 mm3) are below 0.1 mm3 and
    25 voxels (0.102 mm3) are not."""
    assert 24 * V1 < excisions.FRAGMENT_MM3 <= 25 * V1
    miss = np.zeros((30, 30, 30), bool)
    miss[2:8, 2:6, 2:3] = True  # 24 voxels
    miss[2:7, 10:15, 2:3] = True  # 25 voxels
    miss[15:25, 15:25, 15:25] = True  # 1000 voxels
    miss[2, 20, 20] = True  # 1 voxel
    out, pieces, voxels = excisions.drop_fragments(miss, VOX)
    want = miss.copy()
    want[2:8, 2:6, 2:3] = False
    want[2, 20, 20] = False
    assert np.array_equal(out, want)
    assert (pieces, voxels) == (2, 25)


def test_fragments_are_counted_with_face_adjacency() -> None:
    """A voxel that touches a large piece only at an edge is a piece of its
    own, and is dropped."""
    miss = np.zeros((30, 30, 30), bool)
    miss[5:15, 5:15, 5:15] = True
    miss[15, 15, 10] = True  # shares an edge with the block
    out, pieces, voxels = excisions.drop_fragments(miss, VOX)
    assert not out[15, 15, 10] and out[5:15, 5:15, 5:15].all()
    assert (pieces, voxels) == (1, 1)


def test_small_pieces_that_touch_the_excision_are_crumbs() -> None:
    tis = np.zeros((40, 40, 30), bool)
    tis[2:38, 2:38, 2:12] = True  # large, below the slab
    tis[2:6, 2:6, 14:18] = True  # 64 voxels, touches the slab
    tis[10:28, 10:28, 14:20] = True  # 1944 voxels, touches the slab
    tis[32:36, 2:33, 14:15] = True  # 4 x 31 = 124 voxels, touches the slab
    island = np.zeros_like(tis)
    island[2:5, 30:33, 24:27] = True  # 27 voxels, away from the slab
    tis |= island
    miss = np.zeros_like(tis)
    miss[2:38, 2:38, 12:14] = True
    tis &= ~miss
    crumb, n = excisions.cut_off(tis, miss, VOX)
    want = np.zeros_like(tis)
    want[2:6, 2:6, 14:18] = True
    want[32:36, 2:33, 14:15] = True
    assert np.array_equal(crumb, want)
    assert n == 2
    assert not crumb[island].any()


def test_the_crumb_limit_is_exact() -> None:
    """244 voxels (0.9994 mm3) are a crumb, 245 voxels (1.0035 mm3) are not."""
    assert 244 * V1 < excisions.CRUMB_MM3 <= 245 * V1
    tis = np.zeros((70, 30, 10), bool)
    tis[2:63, 2:6, 4:5] = True  # 61 x 4 = 244 voxels
    tis[2:51, 10:15, 4:5] = True  # 49 x 5 = 245 voxels
    miss = np.zeros_like(tis)
    miss[:, :, 3] = True
    crumb, n = excisions.cut_off(tis, miss, VOX)
    want = np.zeros_like(tis)
    want[2:63, 2:6, 4:5] = True
    assert np.array_equal(crumb, want) and n == 1


def test_excise_drops_fragments_then_sets_crumbs_to_outside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """excise with a known detector result: the fragment keeps its label,
    the excision and the crumb it cuts off become outside, the rest is not
    changed, and the map is the voxels set to 0."""
    aseg = np.zeros((40, 40, 30), np.int16)
    aseg[2:38, 2:38, 2:20] = 42
    aseg[6:34, 6:34, 6:16] = 41
    found = np.zeros(aseg.shape, bool)
    found[2:38, 2:38, 10:13] = True  # the excision: cuts the block in two
    found[20, 20, 14] = True  # a fragment of 1 voxel
    # a small piece of cortex on the excision (x, y 2-5, z 13-15), cut off
    # from the upper half by a moat of outside
    aseg[2:6, 2:6, 16:20] = 0
    aseg[6, 2:7, 13:16] = 0
    aseg[2:7, 6, 13:16] = 0
    monkeypatch.setattr(excisions, "detect", lambda *a, **k: found)
    ex = excisions.excise(aseg, np.ones(aseg.shape, np.float32), VOX, "R")
    crumb = np.zeros(aseg.shape, bool)
    crumb[2:6, 2:6, 13:16] = True  # 48 voxels
    exc = found.copy()
    exc[20, 20, 14] = False
    assert np.array_equal(ex.mask, exc | crumb)
    want = aseg.copy()
    want[exc | crumb] = 0
    assert np.array_equal(ex.dseg, want)
    assert ex.dseg[20, 20, 14] == aseg[20, 20, 14] == 41
    assert ex.fragments == dict(Pieces=1, Voxels=1)
    assert ex.crumbs == dict(Pieces=1, Voxels=48)
    assert ex.wm_islands == dict(Pieces=0, Voxels=0)


def test_excise_gives_cut_off_wm_islands_the_cortex_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small WM piece that the excision cuts off inside cortex, which is
    still attached, becomes cortex. A small WM island away from the
    excision, and the large WM, keep their label."""
    aseg = np.zeros((40, 40, 30), np.int16)
    aseg[2:38, 2:38, 2:20] = 42
    aseg[6:34, 6:34, 6:12] = 41  # the large WM
    aseg[8:11, 8:11, 14:17] = 41  # 27 voxels of WM on the excision
    aseg[26:29, 26:29, 15:18] = 41  # 27 voxels of WM away from it
    found = np.zeros(aseg.shape, bool)
    found[6:20, 6:20, 12:14] = True  # cuts the small WM from the large
    monkeypatch.setattr(excisions, "detect", lambda *a, **k: found)
    ex = excisions.excise(aseg, np.ones(aseg.shape, np.float32), VOX, "R")
    want = aseg.copy()
    want[found] = 0
    want[8:11, 8:11, 14:17] = 42
    assert np.array_equal(ex.dseg, want)
    assert np.array_equal(ex.mask, found)
    assert ex.crumbs == dict(Pieces=0, Voxels=0)
    assert ex.wm_islands == dict(Pieces=1, Voxels=27)


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
    return a, f, aseg, excisions.excise(aseg, flash, VOX, "R").mask


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
    # the slab's excision has no fragment and cuts off no crumb
    assert np.array_equal(miss, excisions.detect(read(a), read(f), VOX, "R"))
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
            Frac=0.25,
            Sigma=2.0,
            MinVoxels=250,
            MarginMM=0.3,
            DilationSteps=2,
            FragmentMM3=0.1,
            CrumbMM3=1.0,
        )
        assert side["Fragments"] == dict(Pieces=0, Voxels=0)
        assert side["Crumbs"] == dict(Pieces=0, Voxels=0)
        assert side["WMIslands"] == dict(Pieces=0, Voxels=0)
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
