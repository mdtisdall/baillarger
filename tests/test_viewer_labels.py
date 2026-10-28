"""Tests for baillarger.viewer.labels: decisions and the brush are saved to the change
file at once, the first change keeps the original, and the overlay shows them."""

import gzip
from pathlib import Path

import nibabel as nib
import numpy as np

from baillarger import pial_labels as PL
from baillarger.data import RIM_CORTEX, RIM_OUTSIDE
from baillarger.viewer.labels import LabelReview


def review(tmp_path: Path) -> LabelReview:
    ch = PL.LabelChanges(
        lo=(10, 20, 30),
        hi=(18, 28, 38),
        voxels=np.array([[11, 21, 31], [12, 22, 32]], np.int32),
        label=np.full(2, RIM_OUTSIDE, np.int8),
        component=np.array([0, 1], np.int32),
        status=np.array([PL.PROPOSED, PL.PROPOSED], np.int8),
        components=[
            dict(id=i, rank=1 - i, auto="outer", kind="tube", voxels=1, columns=i,
                 crest_hits=0, touches_fluid=True, frac_median=0.8,
                 betti_local_before=[1, 0, 0], betti_local_after=[1, 0, 0],
                 at_itksnap=[12 + i, 22 + i, 32 + i], at_box0=[1 + i, 1 + i, 1 + i])
            for i in range(2)
        ],
    )  # fmt: skip
    path = tmp_path / "pial_labels_box.npz"
    ch.save(path)
    aff = np.diag([0.5, 0.5, 0.5, 1.0])
    return LabelReview(path, (10, 20, 30), (8, 8, 8), aff)


def test_rows_are_ranked_with_world_points(tmp_path: Path) -> None:
    R = review(tmp_path).rows()
    assert [r["id"] for r in R["rows"]] == [1, 0]
    assert R["rows"][0]["world_mm"] == [1.0, 1.0, 1.0]
    assert R["counts"] == dict(proposed=2, accepted=0, rejected=0)


def test_decide_and_brush_are_saved(tmp_path: Path) -> None:
    rv = review(tmp_path)
    rv.decide(0, "accepted")
    out = rv.brush([1.5, 1.5, 1.5], 1, "cortex")  # box voxel (3, 3, 3), a ball of 7
    assert out["voxels"] == 7 and out["centre_box0"] == [3, 3, 3]
    assert (tmp_path / "pial_labels_box.orig.npz").exists()
    ch = PL.LabelChanges.load(tmp_path / "pial_labels_box.npz")
    assert ch.status.tolist() == [PL.ACCEPTED, PL.PROPOSED]
    assert (ch.label[ch.component < 0] == RIM_CORTEX).all()
    orig = PL.LabelChanges.load(tmp_path / "pial_labels_box.orig.npz")
    assert orig.status.tolist() == [PL.PROPOSED, PL.PROPOSED]
    ov = nib.Nifti1Image.from_bytes(gzip.decompress(rv.overlay_bytes())).get_fdata()
    assert ov[1, 1, 1] == 2 and ov[2, 2, 2] == 1 and ov[3, 3, 3] == PL.OV_BRUSH_CORTEX
