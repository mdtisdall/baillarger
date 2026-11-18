"""Tests of the grow brush's route in the review (docs/plans/label-editing.md,
Part 6)."""

import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from test_viewer_review import SHAPE, A, diff, files, paint_only, paint_strokes

from baillarger.viewer import review as VR

__all__ = ["files"]  # the fixture

X_TRUE = 12  # the true WM ends at x = 12; the dseg's WM runs to x = 16


def write_echoes(tmp: Path) -> list[Path]:
    """Echoes 2 and 3: bright in the true WM (x < X_TRUE inside the dseg's WM
    box), dark elsewhere, with noise."""
    rng = np.random.default_rng(0)
    true = np.zeros(SHAPE, bool)
    true[3:X_TRUE, 3:15, 3:13] = True
    out = []
    for k, (hi, lo) in enumerate(((2000.0, 900.0), (1500.0, 800.0)), start=2):
        e = np.where(true, hi, lo) * np.exp(0.03 * rng.standard_normal(SHAPE))
        p = tmp / f"echo-{k}.nii"
        nib.save(nib.nifti1.Nifti1Image(e.astype(np.float32), A), p)
        out.append(p)
    return out


def grow_review(files: dict[str, Path], echoes: list[Path] | None) -> VR.Review:
    rv = paint_only(files)
    rv.echo_paths = echoes
    return rv


def strokes() -> Any:
    """Outside (0) strokes on z = 5, 8, 11 over the wrong part of the WM."""
    vs = []
    for z in (5, 8, 11):
        vs.append((0, [(x, y, z) for x in (13, 14) for y in range(5, 13)]))
    return paint_strokes(*vs)


def wait(rv: VR.Review, out: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
    t0 = time.time()
    while out["status"] == "running" and time.time() - t0 < timeout:
        time.sleep(0.05)
        out = rv.get("/review/grow", dict(job=out["job"]))
    return out


def test_grow_proposes_and_writes_nothing(
    files: dict[str, Path], tmp_path: Path
) -> None:
    rv = grow_review(files, write_echoes(tmp_path))
    q = rv.queue()
    assert q["grow"] is True and q["grow_off"] is None
    out = wait(rv, rv.post("/review/grow", dict(strokes=strokes())))
    assert out["status"] == "done", out
    assert not files["edits"].exists()  # nothing written
    v = np.asarray(out["voxels"])
    lab = np.asarray(out["labels"])
    assert len(v) and set(lab.tolist()) <= {0, 41}
    # the wrong part of the WM (x 13 and 14 under the strokes' slices) goes
    assert np.all(lab[(v[:, 0] >= 13) & (v[:, 0] <= 14)] == 0)
    # nothing of the true WM changes, away from its boundary
    assert not np.any(v[:, 0] < X_TRUE - 1)
    assert out["settings"] == dict(band=3, beta=130.0, sigma=1.0, zone=4)


def test_an_accepted_proposal_is_one_paint_and_one_undo(
    files: dict[str, Path], tmp_path: Path
) -> None:
    rv = grow_review(files, write_echoes(tmp_path))
    out = wait(rv, rv.post("/review/grow", dict(strokes=strokes())))
    by: dict[int, list[Any]] = {}
    for v, lab in zip(out["voxels"], out["labels"], strict=True):
        by.setdefault(lab, []).append(v)
    acc = [dict(label=lab, voxels=vs) for lab, vs in by.items()]
    r = rv.post("/review/paint", dict(strokes=acc))
    assert r["undo_depth"] == 1 and r["voxels"] == len(out["voxels"])
    got = diff(files)
    assert got == {
        tuple(v): lab for v, lab in zip(out["voxels"], out["labels"], strict=True)
    }
    assert rv.post("/review/undo", {})["undo_depth"] == 0
    assert diff(files) == {}


def test_grow_is_off_without_echoes_and_refuses_the_eraser(
    files: dict[str, Path], tmp_path: Path
) -> None:
    rv = grow_review(files, None)
    q = rv.queue()
    assert q["grow"] is False and "echoes 2 and 3" in q["grow_off"]
    with pytest.raises(ValueError, match="echoes 2 and 3"):
        rv.post("/review/grow", dict(strokes=strokes()))
    rv = grow_review(files, write_echoes(tmp_path))
    with pytest.raises(ValueError, match="eraser"):
        rv.post("/review/grow", dict(strokes=paint_strokes(("input", [(13, 6, 5)]))))
    with pytest.raises(ValueError, match="paint menu"):
        rv.post("/review/grow", dict(strokes=paint_strokes((7, [(13, 6, 5)]))))
