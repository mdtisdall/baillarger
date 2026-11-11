"""The regenerate command (baillarger.regenerate): the four white-surface
commands in one, written into a new directory, then moved over the old
outputs."""

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from baillarger import (
    regenerate,
    white_dseg,
    white_overview,
    white_surf,
    wm_sites,
)
from baillarger.fix_diff import write_table

ANAT = Path("sub-T01/ses-S1/anat")
NAME = "sub-T01_ses-S1_aseg.nii.gz"
BLOCK = ["--block-cells", "8"]


def sphere(path: Path, wm_radius: float = 9.0) -> Path:
    """A synthetic aseg at path: a WM ball in a cortex shell (right
    hemisphere labels)."""
    x = np.indices((34, 34, 30)).astype(float)
    r = np.sqrt(((x - np.array([16.6, 16.3, 14.8])[:, None, None, None]) ** 2).sum(0))
    a = np.zeros(x.shape[1:], np.int16)
    a[r < 13] = 42
    a[r < wm_radius] = 41
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.nifti1.Nifti1Image(a, np.diag([0.16, 0.16, 0.16, 1.0])), path)
    return path


def by_hand(inp: Path, root: Path) -> None:
    """The four commands, run one after another."""
    white_dseg.main([str(inp), "--out", str(root)])
    dseg = root / ANAT / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    white_surf.main([str(dseg), "--out", str(root), *BLOCK])
    surf = root / ANAT / "sub-T01_ses-S1_hemi-R_white.surf.gii"
    white_overview.main([str(surf)])
    wm_sites.main([str(surf)])


def files(root: Path) -> dict[str, bytes]:
    """Each file in root, by its path relative to root."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


#: sidecar keys that hold a time: of creation, and the seconds of a step
TIMES = ("Created", "seconds")


def _norm(x: Any) -> Any:
    """A sidecar without its times."""
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items() if k not in TIMES}
    if isinstance(x, list):
        return [_norm(v) for v in x]
    return x


def assert_same(a: Path, b: Path) -> None:
    """The files in roots a and b are the same: the same names, the same
    bytes, and the same sidecars apart from their times and the root's
    path."""
    fa, fb = files(a), files(b)
    assert sorted(fa) == sorted(fb)
    ra, rb = str(a.resolve()), str(b.resolve())
    for name in fa:
        if name.endswith(".json"):
            ja = json.loads(fa[name].decode().replace(ra, rb))
            jb = json.loads(fb[name].decode())
            assert _norm(ja) == _norm(jb), name
        else:
            assert fa[name] == fb[name], name


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The four commands by hand into one root, the regenerate command into
    another, on the same input."""
    d = tmp_path_factory.mktemp("regen")
    inp = sphere(d / "in" / NAME)
    by_hand(inp, d / "hand")
    regenerate.main([str(inp), "--root", str(d / "regen"), *BLOCK])
    return dict(inp=inp, hand=d / "hand", regen=d / "regen", dir=d)


def test_the_command_gives_the_files_of_the_four_commands(
    runs: dict[str, Path],
) -> None:
    assert_same(runs["hand"], runs["regen"])
    # no new directory is left in the root, and no file names it
    regen = runs["regen"]
    assert not list(regen.glob(regenerate.STAGE_PREFIX + "*"))
    for name, data in files(regen).items():
        assert regenerate.STAGE_PREFIX.encode() not in data, name
    # the sidecars name the files at their final paths
    surf = regen / ANAT / "sub-T01_ses-S1_hemi-R_white.surf.gii"
    side = json.loads(
        surf.with_name(surf.name.replace(".surf.gii", ".json")).read_text()
    )
    assert all(Path(s).exists() for s in side["Sources"])


def test_a_second_run_replaces_the_outputs(
    runs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    d = runs["dir"]
    inp = sphere(d / "in2" / NAME, wm_radius=7.0)
    root = d / "again"
    regenerate.main([str(runs["inp"]), "--root", str(root), *BLOCK])
    capsys.readouterr()
    regenerate.main([str(inp), "--root", str(root), *BLOCK])
    out = capsys.readouterr().out
    assert "overrides: 0" in out
    # the hand run of the first input into the same root, then of the second
    hand = d / "hand2"
    by_hand(runs["inp"], hand)
    by_hand(inp, hand)
    assert_same(hand, root)


def test_a_failing_step_leaves_the_old_outputs_unchanged(
    runs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    d = runs["dir"]
    inp = sphere(d / "in3" / NAME, wm_radius=7.0)
    before = files(runs["regen"])

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("step 4 fails")

    monkeypatch.setattr(wm_sites, "write_sites", fail)
    with pytest.raises(RuntimeError, match="step 4 fails"):
        regenerate.regenerate(inp, None, runs["regen"], block=8)
    assert files(runs["regen"]) == before
    assert not list(runs["regen"].glob(regenerate.STAGE_PREFIX + "*"))


def test_an_output_does_not_replace_the_input(tmp_path: Path) -> None:
    # the input has the name of the dseg that step 1 writes
    root = tmp_path / "deriv"
    inp = sphere(root / ANAT / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz")
    data = inp.read_bytes()
    with pytest.raises(ValueError, match="replace an input"):
        regenerate.regenerate(inp, None, root, block=8)
    assert inp.read_bytes() == data
    assert sorted(files(root)) == [str(inp.relative_to(root))]


def test_the_overrides_are_the_fixes_that_change_edited_voxels(
    tmp_path: Path,
) -> None:
    rows = [
        dict(id=1, kind="handle_cut", voxels=3, i=1, j=2, k=3),
        dict(id=2, kind="handle_fill", voxels=5, i=4, j=5, k=6, overrides=2),
        dict(id=3, kind="manual_edit", voxels=0),
    ]
    table = tmp_path / "fixes.tsv"
    write_table(table, rows)
    over = regenerate.overrides(table)
    assert [r["id"] for r in over] == [2]
    assert regenerate.override_line(over[0]) == (
        "override: fix 2 (handle_fill) at voxel 4,5,6 changes 2 edited voxels"
    )


# ------------------------------------------------------------ with label edits


def ring_edits(tmp_path: Path) -> tuple[Path, Path]:
    """The synthetic aseg of test_white_dseg and a label-edits/1 diff: the
    ring and its hole with their input labels (so the correction must change
    edited voxels: an override), and the rod painted 2 voxels lower
    (label_edits.paint). Returns the input labels and the diff."""
    from test_white_dseg import VOX, synthetic_aseg

    from baillarger import label_edits as LE

    a = synthetic_aseg()
    A = np.diag([*VOX, 1.0])
    inp = tmp_path / "in" / NAME
    inp.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(a, A), inp)
    box = np.zeros(a.shape, bool)
    box[25:37, 3:33, 22:28] = True
    idx = np.flatnonzero(box).astype(np.int64)
    e = LE.Edits(idx, a.reshape(-1)[idx].astype(np.int32))
    v = [(27, 16, 25), (27, 17, 25), (27, 16, 23), (27, 17, 23)]
    e, _ = LE.paint(e, a, v, [0, 0, 41, 41])
    edits = LE.bids_path(tmp_path / "edits", dict(sub="T01", ses="S1"), "R")
    LE.write(edits, e, hemisphere="R", shape=a.shape, affine=A)
    return inp, edits


def test_the_command_with_edits_gives_the_files_of_the_four_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inp, edits = ring_edits(tmp_path)
    hand = tmp_path / "hand"
    white_dseg.main([str(inp), "--edits", str(edits), "--out", str(hand)])
    dseg = hand / ANAT / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    white_surf.main([str(dseg), "--out", str(hand), *BLOCK])
    surf = hand / ANAT / "sub-T01_ses-S1_hemi-R_white.surf.gii"
    white_overview.main([str(surf)])
    wm_sites.main([str(surf)])
    capsys.readouterr()
    regen = tmp_path / "regen"
    regenerate.main([str(inp), "--edits", str(edits), "--root", str(regen), *BLOCK])
    out = capsys.readouterr().out
    assert_same(hand, regen)
    # the dseg's copy of the diff is among the outputs, equal to the diff
    copy = regen / ANAT / "sub-T01_ses-S1_desc-wmtopo_edits.npy"
    assert copy.read_bytes() == edits.read_bytes()
    # the edit that the correction must change (the moved rod) is printed
    lines = [x for x in out.splitlines() if x.startswith("override: ")]
    assert len(lines) == 1 and " at voxel 27," in lines[0] and ",23 " in lines[0]
    assert "overrides: 1" in out


# ------------------------------------------- a stopped move, the kept fix diffs


#: the name of a kept run of another fix diff
OTHER = "0" * 64

#: the move into the root that fails in the stopped run (1 = the first)
STOP_AT = 8


def _stat(p: Path) -> tuple[int, int]:
    """The inode and the modification time of a file."""
    s = p.stat()
    return s.st_ino, s.st_mtime_ns


@pytest.fixture(scope="module")
def second(
    runs: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Two copies of the root of the first input's run: the second input
    regenerated into one (ref), and into the other with the STOP_AT-th move
    into the root failing (stopped). With the stats and bytes of the files
    before the runs."""
    import os
    import shutil

    d = tmp_path_factory.mktemp("second")
    inp = sphere(d / "in" / NAME, wm_radius=7.0)
    ref, stopped = d / "ref", d / "stopped"
    shutil.copytree(runs["regen"], ref)
    shutil.copytree(runs["regen"], stopped)
    # a kept run of another fix diff (the sphere has no fixes, so both inputs
    # give the same, empty diff): a copy of the kept run under another name
    for root in (ref, stopped):
        hist = root / "sub-T01/ses-S1" / white_dseg.HISTORY
        for p in sorted(hist.iterdir()):
            shutil.copyfile(p, hist / (OTHER + p.suffix))
    before = files(stopped)
    stats = {str(p.relative_to(ref)): _stat(p) for p in ref.rglob("*") if p.is_file()}
    regenerate.regenerate(inp, None, ref, block=8)

    real = os.replace
    moves = [0]

    def replace(src: Any, dst: Any) -> None:
        # count only the moves into the root (not the steps' renames in the
        # new directory or in a temporary directory)
        if str(dst).startswith(str(stopped)) and regenerate.STAGE_PREFIX not in str(
            dst
        ):
            moves[0] += 1
            if moves[0] == STOP_AT:
                raise KeyboardInterrupt("stopped between two moves")
        real(src, dst)

    mp = pytest.MonkeyPatch()
    mp.setattr(os, "replace", replace)
    try:
        with pytest.raises(KeyboardInterrupt):
            regenerate.regenerate(inp, None, stopped, block=8)
    finally:
        mp.undo()
    return dict(ref=ref, stopped=stopped, before=before, stats=stats)


def test_a_stop_between_two_moves_leaves_old_and_new_files_mixed(
    second: dict[str, Any],
) -> None:
    # the window that the swap cannot close: some outputs are new, others old
    now, ref, before = files(second["stopped"]), files(second["ref"]), second["before"]
    # (the images, surfaces and tables: the sidecars hold their root and time)
    data = [n for n in before if not n.endswith(".json")]
    new = [n for n in data if now.get(n) == ref.get(n) != before[n]]
    old = [n for n in data if now.get(n) == before[n] != ref.get(n)]
    assert new and old


def test_finish_completes_a_stopped_swap(second: dict[str, Any]) -> None:
    stopped = second["stopped"]
    assert len(list(stopped.glob(regenerate.STAGE_PREFIX + "*"))) == 1
    regenerate.finish(stopped)
    assert_same(stopped, second["ref"])
    assert not list(stopped.glob(regenerate.STAGE_PREFIX + "*"))


def test_a_run_does_not_rewrite_the_files_that_it_only_reads(
    second: dict[str, Any],
) -> None:
    # the kept run of another fix diff and dataset_description.json are read,
    # not written: they are the same files after the second run
    ref, stats = second["ref"], second["stats"]
    kept = [n for n in stats if OTHER in n] + [regenerate.DESCRIPTION]
    assert len(kept) == 4
    for n in kept:
        assert _stat(ref / n) == stats[n], n


def _stage(root: Path, done: bool) -> Path:
    """A new directory in root with one file, a/x.txt; with the marker of a
    finished run if done."""
    stage = root / (regenerate.STAGE_PREFIX + "test")
    (stage / "a").mkdir(parents=True)
    (stage / "a" / "x.txt").write_text("new\n")
    if done:
        (stage / regenerate.MARKER).write_text("a/x.txt\n")
    return stage


def test_finish_moves_the_files_of_a_finished_run(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("old\n")
    _stage(tmp_path, done=True)
    assert regenerate.finish(tmp_path) == [tmp_path / "a" / "x.txt"]
    assert (tmp_path / "a" / "x.txt").read_text() == "new\n"
    assert sorted(files(tmp_path)) == ["a/x.txt"]


def test_finish_deletes_an_unfinished_run(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("old\n")
    _stage(tmp_path, done=False)
    assert regenerate.finish(tmp_path) == []
    assert sorted(files(tmp_path)) == ["a/x.txt"]
    assert (tmp_path / "a" / "x.txt").read_text() == "old\n"


def test_a_run_first_finishes_a_stopped_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, done=True)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("step 1 fails")

    monkeypatch.setattr(regenerate, "_steps", fail)
    with pytest.raises(RuntimeError, match="step 1 fails"):
        regenerate.regenerate(tmp_path / NAME, None, tmp_path, block=8)
    assert sorted(files(tmp_path)) == ["a/x.txt"]


def test_a_second_command_in_the_same_root_stops(tmp_path: Path) -> None:
    with regenerate.locked(tmp_path):
        with pytest.raises(RuntimeError, match="another regenerate"):
            regenerate.regenerate(tmp_path / NAME, None, tmp_path, block=8)
