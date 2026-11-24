"""The editing switch of the review and its standard edits file
(docs/plans/label-editing.md, Part 7): a viewer without --edits finds the
standard edits file of the dseg's input labels and starts with editing off;
POST /review/editing switches it. On the phantom review of
tests/test_viewer_review.py in derivatives roots (bids_files of
tests/test_viewer_regenerate.py)."""

import argparse
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from test_viewer_regenerate import bids_files
from test_viewer_review import A, Fake, diff, files, paint_strokes, randoms

from baillarger.bids import sidecar_path
from baillarger.viewer import review as VR
from baillarger.wm_edits import read_log

__all__ = ["bids_files", "files"]  # the fixtures, used by name


def parse(*argv: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    VR.add_arguments(ap)
    return ap.parse_args(list(argv))


def labels(f: dict[str, Path]) -> Callable[[], Any]:
    lab = np.asarray(nib.nifti1.load(f["dseg"]).dataobj)
    return lambda: lab


@pytest.fixture
def off(bids_files: dict[str, Path]) -> Iterator[tuple[VR.Review, dict[str, Path]]]:
    """The queue review on the standard edits file with editing off (as
    for_white with --sites and no --edits gives it, with stand-in
    choices)."""
    f = bids_files
    rv = VR.Review(
        f["queue"],
        f["log"],
        f["standard"],
        f["dseg"],
        labels(f),
        A,
        randoms,
        Fake(),
        inputs=f["input"],
        editing=False,
    )
    try:
        yield rv, f
    finally:
        rv.close()


def done(rv: VR.Review, out: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
    """A preview's done state: a slow preview answers as a job, polled."""
    t0 = time.monotonic()
    while out.get("status") == "running" and time.monotonic() - t0 < timeout:
        time.sleep(0.05)
        out = rv.get("/review/preview", dict(job=out["job"]))
    return out


def labeledits_root(f: dict[str, Path]) -> Path:
    """The labeledits root: the directory above sub-T01/ses-S1/anat."""
    return f["standard"].parents[3]


def test_without_edits_the_viewer_finds_the_standard_file_with_editing_off(
    bids_files: dict[str, Path],
) -> None:
    f = bids_files
    assert f["standard"].name == "sub-T01_ses-S1_hemi-R_desc-labeledits_edits.tsv"
    assert labeledits_root(f) == f["root"].parent / "labeledits"
    rv = VR.for_white(parse(), Path("s"), f["dseg"], labels(f), A)
    assert rv is not None and rv.paint_only and not rv.editing
    assert rv.edits == f["standard"]
    q = rv.queue()
    assert q["editing"] is False and q["edits"] == str(f["standard"])
    rv.close()


def test_without_edits_a_paint_an_answer_and_an_undo_are_refused(
    off: tuple[VR.Review, dict[str, Path]],
) -> None:
    rv, f = off
    rv.start(0, None)
    rv.next()
    n_log = len(read_log(f["log"]).records)
    calls: list[Callable[[], Any]] = [
        lambda: rv.post("/review/paint", dict(strokes=paint_strokes((0, [(4, 4, 4)])))),
        lambda: rv.answer("5,5,5", "thicken", {}, 1.0),
        lambda: rv.answer("5,5,5", VR.ACCEPT, {}, 1.0),
        lambda: rv.answer("5,5,5", VR.MANUAL, {}, 1.0, paint_strokes((0, [(4, 4, 4)]))),
        lambda: rv.post("/review/undo", {}),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="editing is off"):
            call()
    # nothing is made: no edits file, no root, no answer in the log
    assert not labeledits_root(f).exists()
    assert len(read_log(f["log"]).records) == n_log and rv.undo == []


def test_the_reads_work_with_editing_off(
    off: tuple[VR.Review, dict[str, Path]],
) -> None:
    rv, f = off
    rv.start(0, None)
    rv.next()
    assert rv.get("/review/pending", {}) == dict(voxels=[], labels=[])
    p = rv.post("/review/preview", dict(site="5,5,5", choice="thicken", params={}))
    p = done(rv, p)
    assert p["status"] == "done" and p["answer"]
    p = rv.post(
        "/review/preview",
        dict(site="5,5,5", choice=VR.MANUAL, strokes=paint_strokes((0, [(4, 4, 4)]))),
    )
    assert done(rv, p)["status"] == "done"
    assert not labeledits_root(f).exists()


def test_after_the_switch_a_paint_writes_into_the_standard_file(
    off: tuple[VR.Review, dict[str, Path]],
) -> None:
    rv, f = off
    assert rv.post("/review/editing", dict(on=True)) == dict(
        editing=True, edits=str(f["standard"])
    )
    assert rv.queue()["editing"] is True
    out = rv.post("/review/paint", dict(strokes=paint_strokes((0, [(4, 4, 4)]))))
    assert out == dict(voxels=1, undo_depth=1)
    assert diff(dict(f, edits=f["standard"])) == {(4, 4, 4): 0}
    assert (labeledits_root(f) / "dataset_description.json").is_file()
    # off again: refused, nothing changes
    assert rv.post("/review/editing", dict(on=False))["editing"] is False
    before = f["standard"].read_bytes()
    with pytest.raises(ValueError, match="editing is off"):
        rv.post("/review/undo", {})
    assert f["standard"].read_bytes() == before and len(rv.undo) == 1


def test_with_edits_editing_is_on_at_start_and_makes_no_standard_file(
    bids_files: dict[str, Path],
) -> None:
    f = bids_files
    a = parse("--edits", str(f["edits"]))
    rv = VR.for_white(a, Path("s"), f["dseg"], labels(f), A)
    assert rv is not None and rv.editing and rv.queue()["editing"] is True
    rv.post("/review/paint", dict(strokes=paint_strokes((0, [(4, 4, 4)]))))
    assert diff(f) == {(4, 4, 4): 0}
    assert not labeledits_root(f).exists()
    rv.close()


def test_the_switch_is_refused_without_a_readable_copy_of_the_edits(
    off: tuple[VR.Review, dict[str, Path]],
) -> None:
    rv, f = off
    f["copy"].unlink()  # an output set with no copy that the viewer can read
    sidecar_path(f["copy"]).unlink()
    with pytest.raises(ValueError, match="needs a regeneration"):
        rv.post("/review/editing", dict(on=True))
    assert rv.editing is False and rv.queue()["editing"] is False
    with pytest.raises(ValueError, match="true or false"):
        rv.post("/review/editing", dict(on="yes"))


def test_the_sites_need_the_log_and_the_edits_are_optional(
    bids_files: dict[str, Path],
) -> None:
    f = bids_files
    for argv in (["--sites", "q.tsv"], ["--log", "l.jsonl"]):
        with pytest.raises(ValueError, match="needs"):
            VR.for_white(parse(*argv), Path("s"), f["dseg"], labels(f), A)
    a = parse("--sites", str(f["queue"]), "--log", str(f["log"]))
    rv = VR.for_white(a, Path("s"), f["dseg"], labels(f), A)
    assert rv is not None and not rv.paint_only and not rv.editing
    assert rv.edits == f["standard"]
    rv.close()
    # a dseg with no standard edits file: no review without --sites, an
    # error with them
    assert VR.for_white(parse(), Path("s"), Path("d"), labels(f), A) is None
    with pytest.raises(ValueError, match="give --edits"):
        VR.for_white(a, Path("s"), Path("d"), labels(f), A)
