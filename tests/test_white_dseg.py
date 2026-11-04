"""Tests for the dseg command (baillarger.white_dseg.wm_topo, write_wm_topo): the
segmentation with its WM mask made cavity- and handle-free, on a synthetic
aseg, and on excisions set to outside (cut faces)."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from scipy import ndimage as ndi

from baillarger import bids, white_dseg, white_surf
from baillarger.handles import KUHN_OFFSETS, kuhn_betti, kuhn_structure
from baillarger.mesh import marching_tetrahedra

SHAPE = (40, 36, 30)
VOX = (0.2, 0.16, 0.2)
FILLED = white_dseg.FILLED_LABELS["R"]


def synthetic_aseg() -> npt.NDArray[np.int16]:
    """Right hemisphere: a WM block drilled by a 2 x 2 tunnel of cortex (a
    fill of 4), holding a thalamus and an enclosed pocket of cortex (a
    cavity of 8); a WM ring 5 wide closed by a one-voxel rod across a slit of
    background (a cut of 1); a patch of hippocampus."""
    a = np.zeros(SHAPE, np.int16)
    a[8:24, 6:22, 4:20] = 42
    a[10:22, 8:20, 6:18] = 41
    a[15:17, 13:15, 6:18] = 42  # the tunnel
    a[11:13, 9:11, 8:10] = 49  # a thalamus inside the WM
    a[19:21, 9:11, 12:14] = 42  # the enclosed pocket
    a[8:10, 6:10, 4:8] = 53
    a[25:37, 3:33, 22:28] = 41  # the ring round a 2 x 20 hole
    a[30:32, 8:28, 22:28] = 0
    a[25:30, 16:18, 22:28] = 0  # the slit
    a[27, 16:18, 25] = 41  # the rod across it
    return a


def test_wm_topo_of_a_synthetic_aseg() -> None:
    a = synthetic_aseg()
    t = white_dseg.wm_topo(a, VOX)
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
    # nothing else changed: the thalamus and the hippocampus are kept
    assert np.array_equal(t.dseg[t.fixes == 0], a[t.fixes == 0])
    assert (t.dseg == 49).sum() == 8 and (t.dseg == 53).sum() == 32
    assert sorted(m["Kind"] for m in t.moves) == ["cut", "fill"]
    assert sum(m["Voxels"] for m in t.moves) == 5


def test_one_hemisphere_only() -> None:
    a = synthetic_aseg()
    a[0, 0, 0] = 2  # the left WM too
    with pytest.raises(ValueError, match="exactly one"):
        white_dseg.wm_topo(a, VOX)
    with pytest.raises(ValueError, match="exactly one"):
        white_dseg.hemisphere(np.full((3, 3, 3), 42))


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
        white_dseg.wm_topo(a, VOX)


def test_cuts_take_the_nearest_donor_in_mm() -> None:
    labels = np.full((30, 30, 30), 41, np.int16)
    donor = np.zeros(labels.shape, bool)
    cut = np.zeros(labels.shape, bool)
    cut[15, 15, 15] = True
    # 3 voxels along x (0.3 mm) against 2 along y (0.4 mm)
    labels[18, 15, 15], donor[18, 15, 15] = 7, True
    labels[15, 13, 15], donor[15, 13, 15] = 8, True
    white_dseg.relabel_cuts(labels, cut, donor, (0.1, 0.2, 0.2), margin=1)
    assert labels[15, 15, 15] == 7
    # a donor far beyond the first box: the box is widened
    labels = np.full((30, 30, 30), 41, np.int16)
    donor = np.zeros(labels.shape, bool)
    labels[0, 0, 0], donor[0, 0, 0] = 9, True
    white_dseg.relabel_cuts(labels, cut, donor, (1.0, 1.0, 1.0), margin=1)
    assert labels[15, 15, 15] == 9


def test_the_dseg_command_writes_the_bids_files(tmp_path: Path) -> None:
    src = tmp_path / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir()
    nib.save(nib.nifti1.Nifti1Image(synthetic_aseg(), np.diag([*VOX, 1.0])), src)
    out = tmp_path / "deriv"
    white_dseg.main([str(src), "--out", str(out)])
    anat = out / "sub-T01" / "ses-S1" / "anat"
    stem = "sub-T01_ses-S1_desc-wmtopo_dseg"
    fx = "sub-T01_ses-S1_desc-wmtopofixes_dseg"
    assert (out / "dataset_description.json").exists()
    img = nib.load(anat / f"{stem}.nii.gz")
    assert isinstance(img, nib.nifti1.Nifti1Image)
    d = np.asarray(img.dataobj)
    assert d.dtype == np.int16 and np.allclose(img.affine, np.diag([*VOX, 1.0]))
    assert kuhn_betti(np.isin(d, FILLED)) == (2, 0, 0)
    side = bids.read_sidecar(anat / f"{stem}.json", white_dseg.DSEG_SPEC)
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
    assert side["SmallPieces"] == 0 and side["SmallPieceVoxels"] == 0
    fside = bids.read_sidecar(anat / f"{fx}.json", white_dseg.FIXES_SPEC)
    assert "5 small_piece" in fside["Description"]
    rows = (anat / f"{fx}.tsv").read_text().splitlines()
    assert rows[1:] == [f"{k}\t{v}" for k, v in sorted(white_dseg.FIXES.items())]
    assert json.loads((anat / f"{fx}.json").read_text())["SpecVersion"] == (
        "wm-dseg-fixes/1"
    )


# ---------------------------------------------------------------- cut faces


def excised_block() -> tuple[npt.NDArray[np.int16], npt.NDArray[np.bool_]]:
    """Right hemisphere: a WM block 12 voxels a side in a cortex shell 2
    thick. The excision is a 2 x 2 bore along k, through the shell and the
    WM, from one face to the other."""
    a = np.zeros(SHAPE, np.int16)
    a[8:24, 8:24, 6:22] = 42
    a[10:22, 10:22, 8:20] = 41
    cut = np.zeros(SHAPE, bool)
    cut[15:17, 15:17, 6:22] = True
    return a, cut


def test_an_excision_that_leaves_a_tunnel_is_plugged() -> None:
    """With the excision set to outside, the WM mask is a block with a
    tunnel (b1 = 1). The cheapest fix is a plug of 2 x 2 voxels in one
    slice of the tunnel, not a cut through the 12 x 12 block. The plug is
    in the excision: it is an ordinary fill into outside."""
    a, cut = excised_block()
    x = np.where(cut, 0, a).astype(a.dtype)
    assert kuhn_betti(np.isin(x, FILLED)) == (1, 1, 0)
    t = white_dseg.wm_topo(x, VOX)
    assert t.betti_before == (1, 1, 0) and t.betti_after == (1, 0, 0)
    fill = np.argwhere(t.fixes == 2)
    assert len(fill) == 4 and len(np.unique(fill[:, 2])) == 1
    assert {tuple(p[:2]) for p in fill} == {(15, 15), (15, 16), (16, 15), (16, 16)}
    assert 10 <= fill[0, 2] < 20  # inside the WM part of the tunnel
    assert all(t.dseg[tuple(p)] == 41 for p in fill)
    assert not (t.fixes == 1).any() and not (t.fixes == 3).any()
    # every other voxel of the excision stays outside
    rest = cut & (t.fixes == 0)
    assert (t.dseg[rest] == 0).all() and rest.sum() == cut.sum() - 4


def test_an_excision_that_cuts_a_piece_off_adds_a_piece() -> None:
    """The excision is a slab 2 voxels thick across the whole block, through
    the shell and the WM. The WM mask then has two pieces (b0 = 2), and the
    correction changes nothing. The surface of each piece is closed and has
    genus 0."""
    a, _ = excised_block()
    cut = np.zeros(SHAPE, bool)
    cut[8:24, 8:24, 13:15] = True
    x = np.where(cut, 0, a).astype(a.dtype)
    t = white_dseg.wm_topo(x, VOX)
    assert t.betti_before == (2, 0, 0) and t.betti_after == (2, 0, 0)
    assert not t.fixes.any() and np.array_equal(t.dseg, x)
    v, f = marching_tetrahedra(np.isin(t.dseg, FILLED).astype(np.float64), 0.5)
    pieces = white_surf.mesh_pieces(v, f, VOX)
    assert len(pieces) == 2
    assert all(p["genus"] == 0 and p["boundary_loops"] == 0 for p in pieces)


# ---------------------------------------------------------------- small pieces

V1 = float(np.prod(VOX))  # mm3 of one voxel


def test_the_crumb_size_falls_between_156_and_157_voxels() -> None:
    assert 156 * V1 < white_dseg.CRUMB_MM3 <= 157 * V1


def test_a_speck_that_touches_a_piece_only_across_a_diagonal_is_removed() -> None:
    """A WM voxel in the cortex shell of the block touches a WM voxel of the
    block across an edge diagonal that the Kuhn adjacency does not join, and
    across two corners. It is a piece of its own: it is removed, not joined,
    and takes the label of its other neighbours (cortex)."""
    a, _ = excised_block()
    speck = (9, 22, 14)  # the block's WM is [10:22, 10:22, 8:20]
    a[speck] = 41
    touch = [
        np.subtract(q, speck)
        for q in np.argwhere(a == 41)
        if np.abs(np.subtract(q, speck)).max() == 1
    ]
    assert sorted(sorted(np.abs(o).tolist()) for o in touch) == [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert not any((o == k).all() for o in touch for k in KUHN_OFFSETS)
    m = np.isin(a, FILLED)
    assert kuhn_betti(m) == (2, 0, 0)
    assert ndi.label(m, np.ones((3, 3, 3), bool))[1] == 1  # 26-adjacency joins
    t = white_dseg.wm_topo(a, VOX)
    assert t.betti_before == (2, 0, 0) and t.betti_after == (1, 0, 0)
    assert t.dseg[speck] == 42 and t.small_pieces == 1
    assert np.argwhere(t.fixes == 5).tolist() == [list(speck)]
    assert not (t.fixes % 5).any()  # no other fix
    assert np.array_equal(t.dseg[t.fixes == 0], a[t.fixes == 0])
    assert t.moves == [dict(Kind="small_piece", Voxels=1, Centre=list(speck))]


def test_an_isolated_two_voxel_speck_becomes_cortex() -> None:
    a, _ = excised_block()
    a[28:36, 26:34, 22:28] = 42
    a[31:33, 29, 24] = 41
    t = white_dseg.wm_topo(a, VOX)
    assert t.betti_before == (2, 0, 0) and t.betti_after == (1, 0, 0)
    assert (t.dseg[31:33, 29, 24] == 42).all()
    assert np.argwhere(t.fixes == 5).tolist() == [[31, 29, 24], [32, 29, 24]]
    assert (t.fixes != 0).sum() == 2
    assert t.moves == [dict(Kind="small_piece", Voxels=2, Centre=[32, 29, 24])]


@pytest.mark.parametrize("extra, kept", [(False, False), (True, True)])
def test_a_piece_is_kept_from_one_mm3(extra: bool, kept: bool) -> None:
    """A WM sheet of 12 x 13 voxels (156 voxels, 0.9984 mm3) in cortex is
    removed; with one more voxel (157, 1.0048 mm3) it is kept."""
    a, _ = excised_block()
    a[26:40, 0:15, 22:25] = 42
    a[27:39, 1:14, 23] = 41
    if extra:
        a[27, 1, 24] = 41
    t = white_dseg.wm_topo(a, VOX)
    n = 157 if extra else 156
    assert (n * V1 >= white_dseg.CRUMB_MM3) == kept
    assert t.betti_before == (2, 0, 0)
    assert t.betti_after == ((2, 0, 0) if kept else (1, 0, 0))
    assert t.small_pieces == (0 if kept else 1)
    assert (t.fixes == 5).sum() == (0 if kept else n)
    sheet = t.dseg[27:39, 1:14, 23]
    assert (sheet == (41 if kept else 42)).all()


@pytest.mark.parametrize("first, second", [(53, 42), (42, 53)])
def test_a_tie_goes_to_the_smaller_label(first: int, second: int) -> None:
    """A WM voxel with 13 neighbours of one label and 13 of another takes
    the smaller label, whichever label comes first in the offsets."""
    a, _ = excised_block()
    c = np.array([33, 30, 25])
    for k, o in enumerate(white_dseg.NEIGHBOURS_26):
        a[tuple(c + o)] = first if k < 13 else second
    a[tuple(c)] = 41
    t = white_dseg.wm_topo(a, VOX)
    assert t.dseg[tuple(c)] == 42 and t.small_pieces == 1


def test_a_voxel_with_no_neighbour_outside_the_mask_is_refused() -> None:
    """The rule gives no label to a voxel whose 26 neighbours are all in the
    mask: a 3 x 3 x 3 cube of WM (0.17 mm3) has one at its centre."""
    a, _ = excised_block()
    a[30:35, 27:32, 22:27] = 42
    a[31:34, 28:31, 23:26] = 41
    with pytest.raises(RuntimeError, match=r"\[32, 29, 24\] of a small piece"):
        white_dseg.wm_topo(a, VOX)


def test_small_pieces_counts_with_the_kuhn_adjacency() -> None:
    obj = np.zeros((6, 6, 6), bool)
    obj[1, 1, 1] = obj[2, 2, 2] = True  # a Kuhn neighbour: one piece
    obj[4, 1, 1] = obj[3, 2, 1] = True  # (1, -1, 0) is not: two pieces
    assert ndi.label(obj, kuhn_structure())[1] == 3
    idx, piece = white_dseg.small_pieces(obj, (1.0, 1.0, 1.0), 2.5)
    assert sorted(np.bincount(piece)[1:].tolist()) == [1, 1, 2]
    idx, piece = white_dseg.small_pieces(obj, (1.0, 1.0, 1.0), 1.5)
    assert len(piece) == 2 and set(piece.tolist()) == {1, 2}
