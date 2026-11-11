"""The regenerate run of the review pane (baillarger.viewer.review, Regenerate):
a stand-in command under the real scripts/memguard, on the phantom review of
tests/test_viewer_review.py with its outputs in a derivatives root. No real
regenerate runs here."""

import json
import os
import shutil
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_viewer_review import diff, files, make, paint_strokes

from baillarger import regenerate as RG
from baillarger.viewer import review as VR

__all__ = ["files"]  # the fixture, used by name

ANAT = Path("sub-T01/ses-S1/anat")
GB = VR.GB

#: the stand-in command: its first argument is the mode, then the arguments
#: of the regenerate command
STAND_IN = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path

    mode, args = sys.argv[1], sys.argv[2:]
    root = Path(args[args.index("--root") + 1])
    out = root.parent
    (out / "argv.json").write_text(json.dumps(args))
    print("regenerate: step 1 of 4, the dseg (white_dseg)", file=sys.stderr)
    if mode == "wait":
        t0 = time.monotonic()
        while not (out / "go").exists() and time.monotonic() - t0 < 60:
            time.sleep(0.05)
    if mode == "fail":
        print("ValueError: boom", file=sys.stderr)
        sys.exit(1)
    if mode == "cap":
        sys.exit(137)
    anat = root / "sub-T01" / "ses-S1" / "anat"
    names = sorted(p.name for p in anat.iterdir())
    if mode == "partial":
        names = [n for n in names if n.endswith("_dseg.nii.gz")]
    for n in names:  # replaced as regenerate replaces them: a new inode
        new = anat / (n + ".new")
        new.write_bytes((anat / n).read_bytes())
        os.replace(new, anat / n)
        print(f"written: {anat / n}")
    if mode == "partial":
        sys.exit(1)
    from baillarger.regenerate import override_line  # slow: only here

    row = dict(id=7, kind="cavity", i=4, j=5, k=6, overrides=3)
    print(override_line(row))
    print("overrides: 1")
    """
)


@pytest.fixture
def bids_files(files: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    """The phantom review's outputs (dseg, queue, fix diff, the dseg's copy
    of the edits) moved to the anat directory of a derivatives root, where
    regenerate writes them; the input labels and the edits stay outside
    it."""
    root = tmp_path / "root"
    anat = root / ANAT
    anat.mkdir(parents=True)
    out = dict(files, root=root)
    for k in ("dseg", "queue", "fixdiff", "copy"):
        out[k] = anat / files[k].name
        shutil.move(files[k], out[k])
    for k in ("queue", "copy"):  # their sidecars
        shutil.move(files[k].with_suffix(".json"), out[k].with_suffix(".json"))
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN)
    out["script"] = script
    return out


@pytest.fixture
def big_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 64 GB machine, as the laptop (the CI runners have less)."""
    monkeypatch.setattr(VR, "physical_bytes", lambda: 64 * GB)


def review(f: dict[str, Path], mode: str) -> VR.Review:
    rv = make(f)
    rv.regenerate_command = [sys.executable, str(f["script"]), mode]
    return rv


@pytest.fixture
def rv_wait(bids_files: dict[str, Path], big_machine: None) -> Iterator[VR.Review]:
    rv = review(bids_files, "wait")
    try:
        yield rv
    finally:
        (bids_files["root"].parent / "go").touch()
        rv.close()


def until_end(rv: VR.Review, timeout: float = 60) -> dict[str, Any]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = rv.get("/review/regenerate", {})
        if st["status"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("the run did not end")


def until_started(f: dict[str, Path], timeout: float = 30) -> None:
    """Wait until the stand-in has read its arguments."""
    t0 = time.monotonic()
    while not (f["root"].parent / "argv.json").exists():
        assert time.monotonic() - t0 < timeout, "the stand-in did not start"
        time.sleep(0.05)


def refused_writes(rv: VR.Review, match: str) -> None:
    """Each write into the edits is refused with match."""
    with pytest.raises(ValueError, match=match):
        rv.post("/review/paint", dict(strokes=paint_strokes((0, [(6, 4, 4)]))))
    with pytest.raises(ValueError, match=match):
        rv.answer("5,5,5", "thicken", {}, 1.0)
    with pytest.raises(ValueError, match=match):
        rv.answer("5,5,5", VR.ACCEPT, {}, 1.0)
    with pytest.raises(ValueError, match=match):
        rv.post("/review/undo", {})


# ------------------------------------------------------- the arguments


def test_the_arguments_name_the_input_the_root_and_the_edits(
    bids_files: dict[str, Path],
) -> None:
    f = bids_files
    rv = review(f, "ok")
    head = [sys.executable, str(f["script"]), "ok"]
    base = [*head, str(f["input"].resolve()), "--root", str(f["root"].resolve())]
    # no edits yet: no --edits
    assert rv.regenerate_argv() == base
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    assert rv.regenerate_argv() == [*base, "--edits", str(f["edits"].resolve())]
    rv.close()


def test_a_dseg_outside_a_derivatives_root_is_refused(
    files: dict[str, Path],
) -> None:
    rv = make(files)  # the dseg is not in <root>/sub-T01/ses-S1/anat
    with pytest.raises(ValueError, match="not the dseg that regenerate writes"):
        rv.regenerate_argv()
    rv.close()


def test_the_default_command_is_the_regenerate_module() -> None:
    assert VR.REGENERATE == (sys.executable, "-m", "baillarger.regenerate")
    assert VR.MEMGUARD.is_file() and os.access(VR.MEMGUARD, os.X_OK)


# ------------------------------------------------------- the memory cap


def test_the_cap_is_one_and_a_half_peaks_less_the_viewer_memory() -> None:
    assert VR.REGEN_CAP_GB == round(1.5 * VR.REGEN_PEAK_GB)
    # the laptop (64 GB) with a small viewer: the full cap
    assert VR.regen_cap_gb(10 * GB, 64 * GB) == VR.REGEN_CAP_GB
    # a large viewer: 64 - 30 - 8 = 26 GB
    assert VR.regen_cap_gb(30 * GB, 64 * GB) == 26
    # too little left for the peak
    with pytest.raises(ValueError, match="stop the viewer"):
        VR.regen_cap_gb(40 * GB, 64 * GB)
    with pytest.raises(ValueError, match="needs about 22 GB"):
        VR.regen_cap_gb(1 * GB, 16 * GB)


def test_the_viewer_memory_counts_its_own_process() -> None:
    assert VR.tree_rss_bytes(os.getpid()) > 10 * 1024**2


def test_the_memory_too_small_refuses_the_start(
    bids_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(VR, "physical_bytes", lambda: 16 * GB)
    rv = review(bids_files, "ok")
    with pytest.raises(ValueError, match="needs about 22 GB"):
        rv.post("/review/regenerate", {})
    assert rv.get("/review/regenerate", {})["status"] == "idle"
    assert not (bids_files["root"].parent / "argv.json").exists()
    rv.close()


# ------------------------------------------------------- the override lines


def test_an_override_line_reads_back() -> None:
    row = dict(id=12, kind="handle-fill", i=3, j=40, k=511, overrides=9)
    line = RG.override_line(row)
    assert VR.parse_override(line) == dict(
        fix=12, kind="handle-fill", voxel=[3, 40, 511], voxels=9
    )
    assert VR.parse_override("overrides: 1") is None
    assert VR.parse_override("written: /x/y.nii.gz") is None


# ------------------------------------------------------- a run


def test_a_run_closes_the_paint_mode_then_asks_for_a_restart(
    bids_files: dict[str, Path], rv_wait: VR.Review
) -> None:
    f, rv = bids_files, rv_wait
    rv.start(0, None)
    rv.next()
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    assert rv.get("/review/pending", {})["voxels"] == [[4, 4, 4]]
    edits_before = f["edits"].read_bytes(), diff(f), len(rv.undo)
    st = rv.post("/review/regenerate", {})
    assert st["status"] == "running" and st["cap_gb"] == VR.REGEN_CAP_GB
    assert st["command"].split()[:2] == [str(VR.MEMGUARD), str(VR.REGEN_CAP_GB)]
    until_started(f)
    args = json.loads((f["root"].parent / "argv.json").read_text())
    assert args[-2:] == ["--edits", str(f["edits"].resolve())]
    # while it runs: each write into the edits is refused, nothing is written
    refused_writes(rv, "regenerate runs")
    assert (f["edits"].read_bytes(), diff(f), len(rv.undo)) == edits_before
    with pytest.raises(ValueError, match="in progress"):
        rv.post("/review/regenerate", {})
    # the log lines arrive while it runs
    t0 = time.monotonic()
    while not rv.get("/review/regenerate", {})["log"]:
        assert time.monotonic() - t0 < 30
        time.sleep(0.05)
    assert "step 1 of 4" in rv.get("/review/regenerate", {})["log"][0]
    (f["root"].parent / "go").touch()
    st = until_end(rv)
    assert st["status"] == "done", st
    assert st["overrides"] == [dict(fix=7, kind="cavity", voxel=[4, 5, 6], voxels=3)]
    assert st["error"] is None and st["seconds"] > 0
    assert any(x.startswith("memguard: peak") for x in st["log"])
    n_out = len(list((f["root"] / ANAT).iterdir()))
    assert sum(x.startswith("written: ") for x in st["log"]) == n_out == 6
    # until the restart: the identity check refuses each write, the pending
    # edits, and a new run
    refused_writes(rv, "restart the viewer")
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.get("/review/pending", {})
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.post("/review/regenerate", {})
    assert (f["edits"].read_bytes(), diff(f), len(rv.undo)) == edits_before


def test_a_failed_run_keeps_the_outputs_and_opens_the_paint_mode(
    bids_files: dict[str, Path], big_machine: None
) -> None:
    f = bids_files
    rv = review(f, "fail")
    ids = {k: VR.identity(f[k]) for k in ("dseg", "queue", "fixdiff")}
    rv.post("/review/regenerate", {})
    st = until_end(rv)
    assert st["status"] == "failed"
    assert st["error"] == "exit code 1: ValueError: boom"
    assert st["outputs_changed"] is False and st["overrides"] == []
    assert {k: VR.identity(f[k]) for k in ids} == ids
    # the writes are open again
    out = rv.post("/review/paint", dict(strokes=paint_strokes((0, [(4, 4, 4)]))))
    assert out == dict(voxels=1, undo_depth=1)
    rv.close()


def test_a_run_stopped_by_the_memory_cap_says_so(
    bids_files: dict[str, Path], big_machine: None
) -> None:
    rv = review(bids_files, "cap")
    rv.post("/review/regenerate", {})
    st = until_end(rv)
    assert st["status"] == "failed"
    assert st["error"] == (
        f"exit code 137: the memory cap ({VR.REGEN_CAP_GB} GB) stopped the run"
    )
    rv.close()


def test_a_run_that_failed_during_its_moves_keeps_the_writes_refused(
    bids_files: dict[str, Path], big_machine: None
) -> None:
    rv = review(bids_files, "partial")
    rv.post("/review/regenerate", {})
    st = until_end(rv)
    assert st["status"] == "failed" and st["outputs_changed"] is True
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.paint(paint_strokes((0, [(4, 4, 4)])))
    rv.close()


def test_close_stops_a_run(bids_files: dict[str, Path], big_machine: None) -> None:
    f = bids_files
    rv = review(f, "wait")
    rv.post("/review/regenerate", {})
    until_started(f)
    proc = rv.regen.proc
    assert proc is not None
    rv.close()
    assert proc.poll() is not None
    st = rv.get("/review/regenerate", {})
    assert st["status"] == "failed"
    assert VR.identity(f["dseg"]) == rv._outputs[f["dseg"]]
