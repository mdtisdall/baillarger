"""Tests for baillarger.wm_edits (manual label edits of the aseg) and their use by
the dseg command (baillarger.white.wm_topo, write_wm_topo)."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from test_white_dseg import FILLED, VOX, synthetic_aseg

from baillarger import bids, white
from baillarger import wm_edits as E
from baillarger.handles import kuhn_betti

ROD = [(27, 16, 25), (27, 17, 25)]  # the rod across the slit: the aseg's one cut
DRILL = [(18, 16, z) for z in range(6, 18)]  # a 1 x 1 tunnel of cortex through WM


def remove_rod() -> E.Edits:
    """Answer 1: remove the rod (WM to background); answer 2: leave a site."""
    return (
        E.Edits()
        .with_answer(
            E.Answer(1, ROD[0], "remove-strand", "ab" * 32), ROD, [41, 41], [0, 0]
        )
        .with_answer(E.Answer(2, (15, 15, 10), "leave"))
    )


def save_aseg(tmp_path: Path) -> Path:
    src = tmp_path / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(exist_ok=True)
    nib.save(nib.nifti1.Nifti1Image(synthetic_aseg(), np.diag([*VOX, 1.0])), src)
    return src


def test_edits_round_trip(tmp_path: Path) -> None:
    src = save_aseg(tmp_path)
    e = remove_rod()
    path = E.edits_path(tmp_path / "deriv", bids.entities(src))
    assert path.name == "sub-T01_ses-S1_desc-wmedits_edits.tsv"
    E.write_edits(path, src, e)
    got, side = E.read_edits(path)
    assert np.array_equal(got.voxels, e.voxels) and np.array_equal(got.old, e.old)
    assert np.array_equal(got.new, e.new) and np.array_equal(got.answer, e.answer)
    assert got.answers == e.answers  # the leave answer is kept, with no rows
    assert path.read_text().splitlines()[0] == "i\tj\tk\told\tnew\tanswer"
    assert side["Sources"] == [str(src.resolve())] and E.sources_match(side, src)
    a = json.loads(bids.sidecar_path(path).read_text())["Answers"][0]
    assert a["Choice"] == "remove-strand" and a["Site"] == list(ROD[0])


def test_without_reverts_one_answer() -> None:
    e = remove_rod().without(1)
    assert len(e.voxels) == 0 and [a.id for a in e.answers] == [2]


def test_edits_that_do_not_fit_are_refused(tmp_path: Path) -> None:
    a = synthetic_aseg()
    site = E.Answer(1, (0, 0, 0), "x")
    with pytest.raises(ValueError, match="old label"):
        E.Edits().with_answer(site, ROD, [42, 42], [0, 0]).check(a)
    with pytest.raises(ValueError, match="outside the grid"):
        E.Edits().with_answer(site, [(99, 0, 0)], [0], [41]).check(a)
    with pytest.raises(ValueError, match="twice"):
        E.Edits().with_answer(site, [ROD[0], ROD[0]], [41, 41], [0, 0]).check(a)
    with pytest.raises(ValueError, match="new label is its old"):
        E.Edits().with_answer(site, ROD, [41, 41], [41, 0]).check(a)
    with pytest.raises(ValueError, match="there already"):
        remove_rod().with_answer(E.Answer(2, (0, 0, 0), "x"))
    with pytest.raises(ValueError, match="answers that are not there"):
        E.Edits(np.array([ROD[0]]), np.array([41]), np.array([0]), np.array([7])).check(
            a
        )
    with pytest.raises(ValueError, match="old label"):
        E.write_edits(tmp_path / "x.tsv", save_aseg(tmp_path), remove_rod(), a * 0)


def test_an_edit_replaces_the_automatic_cut() -> None:
    """Removing the rod by hand leaves the ring without its handle: the automatic
    correction no longer cuts it, and the fixes show the edit."""
    t = white.wm_topo(synthetic_aseg(), VOX, remove_rod())
    assert t.betti_before == (2, 1, 1) and t.betti_after == (2, 0, 0)
    assert kuhn_betti(np.isin(t.dseg, FILLED)) == (2, 0, 0)
    assert all(t.dseg[v] == 0 and t.fixes[v] == 4 for v in ROD)
    assert t.manual_voxels == 2 and (t.fixes == 1).sum() == 0
    assert t.moves[0] == dict(Kind="manual", Voxels=2, Centre=[27, 16, 25], Answer=1)
    assert sorted(m["Kind"] for m in t.moves) == ["fill", "manual"]


def test_no_edit_can_leave_a_handle() -> None:
    """An edit that drills a tunnel through the WM makes a handle; the automatic
    correction then removes it, and the voxel it fills shows the correction."""
    e = E.Edits().with_answer(
        E.Answer(1, DRILL[0], "drill"), DRILL, [41] * len(DRILL), [42] * len(DRILL)
    )
    t = white.wm_topo(synthetic_aseg(), VOX, e)
    assert t.betti_before == (2, 3, 1) and t.betti_after == (2, 0, 0)
    assert kuhn_betti(np.isin(t.dseg, FILLED)) == (2, 0, 0)
    drilled = np.zeros(t.fixes.shape, bool)
    drilled[tuple(np.array(DRILL).T)] = True
    assert set(np.unique(t.fixes[drilled]).tolist()) <= {2, 4}
    assert (t.fixes[drilled] == 4).sum() + (t.fixes[drilled] == 2).sum() == len(DRILL)
    assert t.manual_voxels == len(DRILL)


def test_the_dseg_command_takes_edits(tmp_path: Path) -> None:
    src = save_aseg(tmp_path)
    out = tmp_path / "deriv"
    path = E.edits_path(out, bids.entities(src))
    E.write_edits(path, src, remove_rod())
    white.main(["dseg", str(src), "--edits", str(path), "--out", str(out)])
    anat = out / "sub-T01" / "ses-S1" / "anat"
    side = bids.read_sidecar(
        anat / "sub-T01_ses-S1_desc-wmtopo_dseg.json", white.DSEG_SPEC
    )
    want = [str(p.resolve()) for p in (src, path, bids.sidecar_path(path))]
    assert side["Sources"] == want and set(side["SourcesSHA256"]) == set(want)
    assert side["ManualVoxels"] == 2 and side["HandleCutVoxels"] == 0
    assert side["Moves"][0]["Kind"] == "manual" and side["Moves"][0]["Answer"] == 1
    fimg = nib.load(anat / "sub-T01_ses-S1_desc-wmtopofixes_dseg.nii.gz")
    assert isinstance(fimg, nib.nifti1.Nifti1Image)
    f = np.asarray(fimg.dataobj)
    assert (f == 4).sum() == 2
    tsv = (anat / "sub-T01_ses-S1_desc-wmtopofixes_dseg.tsv").read_text()
    assert "4\tmanual_edit" in tsv
    # without edits: ManualVoxels 0 and the aseg alone in Sources
    white.main(["dseg", str(src), "--out", str(tmp_path / "plain")])
    plain = bids.read_sidecar(
        tmp_path / "plain/sub-T01/ses-S1/anat/sub-T01_ses-S1_desc-wmtopo_dseg.json",
        white.DSEG_SPEC,
    )
    assert plain["ManualVoxels"] == 0 and plain["Sources"] == [str(src.resolve())]


def test_edits_against_another_aseg_are_refused(tmp_path: Path) -> None:
    src = save_aseg(tmp_path)
    path = E.edits_path(tmp_path / "deriv", bids.entities(src))
    E.write_edits(path, src, remove_rod())
    a = synthetic_aseg()
    a[0, 0, 0] = 42  # the aseg changed after the edits were made
    nib.save(nib.nifti1.Nifti1Image(a, np.diag([*VOX, 1.0])), src)
    with pytest.raises(ValueError, match="not against"):
        white.write_wm_topo(src, tmp_path / "deriv", path)
