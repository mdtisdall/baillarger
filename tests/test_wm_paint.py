"""Tests for the crop of the WM labels (baillarger.wm_paint): the export, the
changes that the import reads back, the answer that the import appends, and
the command line."""

from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import bids
from baillarger import wm_edits as E
from baillarger import wm_paint as P
from baillarger.wm_choices import box_round

SHAPE = (30, 20, 16)
SHA = "ab" * 32
AFFINE: npt.NDArray[np.float64] = np.diag([0.2, 0.16, 0.2, 1.0])
AFFINE[:3, 3] = [-5.0, 3.0, 2.0]
SITE = (15, 10, 8)
HALF = 4


def load(path: Path) -> nib.Nifti1Image:
    img = nib.load(path)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    return img


def phantom() -> npt.NDArray[np.int16]:
    a = np.full(SHAPE, 42, np.int16)
    a[5:20, 4:14, 4:12] = 41
    return a


@pytest.fixture
def dseg(tmp_path: Path) -> Path:
    p = tmp_path / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(phantom(), AFFINE), p)
    return p


def echo(
    tmp_path: Path,
    name: str,
    shape: tuple[int, int, int] = SHAPE,
    affine: npt.NDArray[np.float64] = AFFINE,
) -> Path:
    p = tmp_path / name
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    nib.save(nib.nifti1.Nifti1Image(data, affine), p)
    return p


def make_log(tmp_path: Path, dseg: Path) -> E.EditLog:
    p = E.edits_paths(tmp_path / "deriv", dict(sub="T01", ses="S1"))
    return E.EditLog.create(p["log"], dseg, "R")


def session(log: E.EditLog) -> int:
    s = log.start_session(0.0, None, SHA, SHA)
    log.shown(s, SITE, "misfacing", 1)
    return s


def test_export_writes_the_files_and_the_offset(tmp_path: Path, dseg: Path) -> None:
    out = tmp_path / "crop"
    e = echo(tmp_path, "echo-src.nii.gz")
    assert P.export(dseg, SITE, out, [e], HALF) == out
    for name in (P.LABELS, P.EXPORTED, "echo-1.nii.gz", P.SIDECAR):
        assert (out / name).is_file()
    side = bids.read_sidecar(out / P.SIDECAR, P.PAINT_SPEC)
    box = box_round(SHAPE, SITE, HALF)
    assert side["Offset"] == [s.start for s in box]
    assert side["Shape"] == list(SHAPE) and side["Site"] == list(SITE)
    assert side["Sources"] == [str(dseg.resolve()), str(e.resolve())]
    assert side["DsegSHA256"] == bids.file_sha256(dseg)
    crop = load(out / P.LABELS)
    assert crop.shape == tuple(s.stop - s.start for s in box)
    assert load(out / "echo-1.nii.gz").shape == crop.shape
    # the crop's voxel [0, 0, 0] is the grid voxel Offset, in the same world point
    world = crop.affine @ [0, 0, 0, 1]
    assert np.allclose(world, AFFINE @ [*side["Offset"], 1])


def test_export_then_import_with_no_change_gives_no_voxels(
    tmp_path: Path, dseg: Path
) -> None:
    log = make_log(tmp_path, dseg)
    s = session(log)
    out = P.export(dseg, SITE, tmp_path / "crop", (), HALF)
    a = P.import_paint(out, log.path, s, 5.0)
    assert log.read().voxels(a)[0].shape == (0, 3)


def test_one_painted_voxel_is_one_voxel_at_its_grid_position(
    tmp_path: Path, dseg: Path
) -> None:
    log = make_log(tmp_path, dseg)
    s = session(log)
    out = P.export(dseg, SITE, tmp_path / "crop", (), HALF)
    labels_path = out / P.LABELS
    img = load(labels_path)
    data = np.asanyarray(img.dataobj).copy()
    data[1, 1, 1] = 0
    nib.save(nib.nifti1.Nifti1Image(data, img.affine), labels_path)
    a = P.import_paint(out, log.path, s, 5.0)
    vox, lab, _ = log.read().voxels(a)
    assert vox.tolist() == [[12, 7, 5]] and lab.tolist() == [0]
    # the grid voxel and the painted crop voxel are the same world point
    assert np.allclose(AFFINE @ [*vox[0].tolist(), 1], img.affine @ [1, 1, 1, 1])


def test_an_echo_on_another_grid_is_refused(tmp_path: Path, dseg: Path) -> None:
    shifted = AFFINE.copy()
    shifted[0, 3] += 1.0
    bad = echo(tmp_path, "echo-bad.nii.gz", affine=shifted)
    with pytest.raises(ValueError, match="grid"):
        P.export(dseg, SITE, tmp_path / "crop", [bad], HALF)


def test_import_refuses_a_log_on_another_grid(tmp_path: Path, dseg: Path) -> None:
    out = P.export(dseg, SITE, tmp_path / "crop", (), HALF)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = other_dir / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(phantom()[:-1], AFFINE), other)
    log = make_log(other_dir, other)
    s = session(log)
    with pytest.raises(ValueError, match="grid"):
        P.import_paint(out, log.path, s, 1.0)


def test_the_command_line_runs_export_and_import(
    tmp_path: Path, dseg: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = make_log(tmp_path, dseg)
    s = session(log)
    e = echo(tmp_path, "echo-src.nii.gz")
    crop = tmp_path / "crop"
    P.main(
        [
            "export",
            str(dseg),
            "--site",
            *map(str, SITE),
            "--out",
            str(crop),
            "--echo",
            str(e),
            "--half",
            str(HALF),
        ]
    )
    assert (crop / P.LABELS).is_file() and (crop / "echo-1.nii.gz").is_file()
    P.main(
        [
            "import",
            str(crop),
            "--log",
            str(log.path),
            "--session",
            str(s),
            "--seconds",
            "5",
        ]
    )
    a = int(capsys.readouterr().out)
    assert log.read().answer(a)["Choice"] == "manual-paint"
    assert log.read().voxels(a)[0].shape == (0, 3)
