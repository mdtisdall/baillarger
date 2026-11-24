"""The regeneration from the command line (docs/plans/label-editing.md, Part 7):
the review pane gives the regenerate command as text, and after the command
replaced the outputs each write is refused until the viewer restarts. On the
phantom review of tests/test_viewer_review.py with its outputs in a
derivatives root and its input labels in another. No regenerate runs here."""

import shlex
import shutil
import sys
from pathlib import Path

import pytest
from edits_helpers import write_dseg_sidecar
from test_viewer_review import diff, files, make, paint_strokes, replace_on_disk

from baillarger import label_edits as LE
from baillarger.bids import sidecar_path
from baillarger.viewer import review as VR

__all__ = ["files"]  # the fixture, used by name

ANAT = Path("sub-T01/ses-S1/anat")


@pytest.fixture
def bids_files(files: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    """The phantom review's outputs (dseg, queue, fix diff, the dseg's copy
    of the edits) moved to the anat directory of a derivatives root, where
    regenerate writes them, and the input labels to the anat directory of
    another root beside it (so they have a standard edits file); the edits
    given by path stay outside both."""
    root = tmp_path / "derivatives" / "root"
    anat = root / ANAT
    anat.mkdir(parents=True)
    out = dict(files, root=root)
    for k in ("dseg", "queue", "fixdiff", "copy"):
        out[k] = anat / files[k].name
        shutil.move(files[k], out[k])
    for k in ("queue", "copy"):  # their sidecars
        shutil.move(sidecar_path(files[k]), sidecar_path(out[k]))
    inp_anat = tmp_path / "derivatives" / "inputs" / ANAT
    inp_anat.mkdir(parents=True)
    out["input"] = inp_anat / files["input"].name
    shutil.move(files["input"], out["input"])
    write_dseg_sidecar(out["dseg"], out["input"])
    out["standard"] = LE.standard_path(out["input"], "R")
    return out


def base(f: dict[str, Path]) -> list[str]:
    return [
        *VR.REGENERATE,
        str(f["input"].resolve()),
        "--root",
        str(f["root"].resolve()),
    ]


def test_regenerate_is_not_a_route(bids_files: dict[str, Path]) -> None:
    rv = make(bids_files)
    with pytest.raises(KeyError):
        rv.get("/review/regenerate", {})
    with pytest.raises(KeyError):
        rv.post("/review/regenerate", {})
    assert not hasattr(VR, "Regeneration") and not hasattr(VR, "regen_cap_gb")
    rv.close()


def test_the_default_command_is_the_regenerate_module() -> None:
    assert VR.REGENERATE == (sys.executable, "-m", "baillarger.regenerate")


def test_the_command_has_no_edits_for_the_standard_file(
    bids_files: dict[str, Path],
) -> None:
    f = bids_files
    rv = make(dict(f, edits=f["standard"]))
    assert rv.queue()["regenerate_command"] == shlex.join(base(f))
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    assert f["standard"].exists()
    q = rv.queue()
    assert q["regenerate_command"] == shlex.join(base(f))
    assert q["regenerate_off"] is None
    rv.close()


def test_the_command_names_another_edits_file(bids_files: dict[str, Path]) -> None:
    f = bids_files
    rv = make(f)  # files["edits"]: not the standard file, not there yet
    assert rv.queue()["regenerate_command"] == shlex.join([*base(f), "--no-edits"])
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    assert rv.queue()["regenerate_command"] == shlex.join(
        [*base(f), "--edits", str(f["edits"].resolve())]
    )
    rv.close()


def test_a_dseg_outside_a_derivatives_root_has_no_command(
    files: dict[str, Path],
) -> None:
    rv = make(files)  # the dseg is not in <root>/sub-T01/ses-S1/anat
    with pytest.raises(ValueError, match="not the dseg that regenerate writes"):
        rv.regenerate_argv()
    q = rv.queue()
    assert q["regenerate_command"] is None
    assert "not the dseg that regenerate writes" in q["regenerate_off"]
    rv.close()


def test_replaced_outputs_refuse_each_write_until_a_new_viewer(
    bids_files: dict[str, Path],
) -> None:
    f = bids_files
    rv = make(f)
    rv.start(0, None)
    rv.next()
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    assert rv.queue()["outputs_changed"] == []
    before = f["edits"].read_bytes(), diff(f), len(rv.undo)
    outs = ("dseg", "queue", "fixdiff", "copy")
    for k in outs:
        replace_on_disk(f[k])
    assert sorted(rv.queue()["outputs_changed"]) == sorted(f[k].name for k in outs)
    match = "restart the viewer"
    with pytest.raises(ValueError, match=match):
        rv.post("/review/paint", dict(strokes=paint_strokes((0, [(6, 4, 4)]))))
    with pytest.raises(ValueError, match=match):
        rv.answer("5,5,5", "thicken", {}, 1.0)
    with pytest.raises(ValueError, match=match):
        rv.answer("5,5,5", VR.ACCEPT, {}, 1.0)
    with pytest.raises(ValueError, match=match):
        rv.post("/review/undo", {})
    with pytest.raises(ValueError, match=match):
        rv.get("/review/pending", {})
    assert (f["edits"].read_bytes(), diff(f), len(rv.undo)) == before
    rv.close()
    # a new viewer reads the new outputs and writes
    rv = make(f)
    out = rv.post("/review/paint", dict(strokes=paint_strokes((0, [(6, 4, 4)]))))
    assert out == dict(voxels=1, undo_depth=1)
    rv.close()
