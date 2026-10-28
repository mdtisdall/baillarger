"""Tests for baillarger.white: the handle-free WM label of a small synthetic
aseg, its changes on the full grid, and boxes cropped from it."""

from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import data, white
from baillarger.handles import kuhn_betti
from baillarger.mesh import marching_tetrahedra, topology

SHAPE = (40, 36, 30)


def write_aseg(d: Path) -> npt.NDArray[np.int16]:
    """An aseg with two WM components: a block drilled by a 2 x 2 tunnel (a
    fill of 4 voxels), and a ring 5 voxels wide closed by a one-voxel rod (a
    cut of 1)."""
    a = np.zeros(SHAPE, np.int16)
    a[8:24, 6:22, 4:20] = data.ASEG_CORTEX
    a[10:22, 8:20, 6:18] = data.ASEG_WM
    a[15:17, 13:15, 6:18] = data.ASEG_CORTEX
    a[25:37, 3:33, 22:28] = data.ASEG_WM  # 5 wide round a 2 x 20 hole
    a[30:32, 8:28, 22:28] = 0
    a[25:30, 16:18, 22:28] = 0
    a[27, 16:18, 25] = data.ASEG_WM
    nib.save(
        nib.nifti1.Nifti1Image(a, np.diag([0.2, 0.16, 0.2, 1.0])), data.aseg_path(d)
    )
    return a


def test_changes_on_a_phantom_aseg(tmp_path: Path) -> None:
    a = write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    assert ch.betti_before == (2, 2, 0) and ch.betti_after == (2, 0, 0)
    fill = ch.voxels[ch.kind == white.FILL]
    cut = ch.voxels[ch.kind == white.CUT]
    assert len(ch.voxels) == 5 and not (ch.kind == white.STRANDED).any()
    # the plug of the tunnel: 2 x 2 voxels in one slice, full-grid coordinates
    assert len(fill) == 4 and len(np.unique(fill[:, 2])) == 1
    assert {tuple(p[:2]) for p in fill} == {(15, 13), (15, 14), (16, 13), (16, 14)}
    assert all(a[tuple(p)] == data.ASEG_CORTEX for p in fill)
    # one voxel of the rod
    assert len(cut) == 1 and cut[0][0] == 27 and cut[0][2] == 25
    assert a[tuple(cut[0])] == data.ASEG_WM
    assert len(np.unique(ch.move)) == 2


def test_boxes_apply_the_changes(tmp_path: Path) -> None:
    a = write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    wm = white.white_matter_box((0, 0, 0), SHAPE, ch, tmp_path)
    expected = a == data.ASEG_WM
    expected[tuple(ch.voxels[ch.kind == white.FILL].T)] = True
    expected[tuple(ch.voxels[ch.kind != white.FILL].T)] = False
    assert np.array_equal(wm, expected)
    assert kuhn_betti(wm) == (2, 0, 0)
    v, f = marching_tetrahedra(wm.astype(np.float64), 0.5)
    assert topology(f, len(v))["genus"] == 0
    for lo, hi in (((12, 10, 4), (20, 30, 25)), ((26, 0, 20), (40, 36, 30))):
        box = white.white_matter_box(lo, hi, ch, tmp_path)
        assert np.array_equal(
            box, expected[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        )


def test_save_and_load(tmp_path: Path) -> None:
    write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    p = tmp_path / "wm_changes.npz"
    ch.save(p)
    back = white.WMChanges.load(p)
    assert back.betti_before == ch.betti_before and back.betti_after == ch.betti_after
    for k in ("voxels", "kind", "move"):
        assert np.array_equal(getattr(back, k), getattr(ch, k))


def test_box_outside_the_grid_raises(tmp_path: Path) -> None:
    write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    with pytest.raises(ValueError):
        white.white_matter_box((30, 0, 0), (41, 10, 10), ch, tmp_path)
