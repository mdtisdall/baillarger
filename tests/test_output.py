"""Tests for baillarger.output on the two-area patch of test_refine."""

import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from test_refine import two_areas

from baillarger import output as out
from baillarger.refine import Refiner


def load(path: Path) -> npt.NDArray[Any]:
    img = nib.load(path)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    return np.asarray(img.dataobj)


def test_write_level(tmp_path: Path) -> None:
    data, truth = two_areas()
    _, levels = Refiner(data).run([2])
    lv = levels[2]
    out.write_level(tmp_path, data, lv, np.eye(4))
    reg = load(tmp_path / "regions.nii.gz").ravel()
    # every voxel of the table is in region 1 or 2, as its column
    assert set(np.unique(reg[data.index]).tolist()) == {1, 2}
    assert np.array_equal(reg[data.index], lv.reg[data.col] + 1)
    band = load(tmp_path / "bands.nii.gz").ravel()
    cx = data.ctx
    assert band[data.index[cx]].min() == 1
    assert band[data.index[cx]].max() == max(lv.K.values())
    # the WM extension is -1 down to each region's bottom (here all of it)
    assert (band[data.index[~cx]] == -1).all()
    filt = load(tmp_path / "filtered.nii.gz")
    assert filt.shape[-1] == 3
    with open(tmp_path / "regions.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["K"]) for r in rows] == [lv.K[0], lv.K[1]]
    assert sum(int(r["columns"]) for r in rows) == data.n
    with open(tmp_path / "columns.csv") as fh:
        cols = list(csv.DictReader(fh))
    assert len(cols) == data.n
    assert all(len(r["z"].split()) == lv.K[lv.reg[i]] - 1 for i, r in enumerate(cols))
