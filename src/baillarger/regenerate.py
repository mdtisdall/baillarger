"""Regenerate the outputs of the label edits: the four white-surface commands
in one command (docs/plans/label-editing.md, Part 5).

    python -m baillarger.regenerate INPUT [--edits EDITS] --root ROOT
        [--block-cells N] [--device DEVICE] [--workers N]

INPUT is the input labels (the excised dseg). EDITS is the diff of the label
edits. The command calls, in this order:

1. baillarger.white_dseg.write_wm_topo (INPUT, with EDITS);
2. baillarger.white_surf.write_white_surface (the dseg of step 1);
3. baillarger.white_overview.write_white_overview (the surface of step 2);
4. baillarger.wm_sites.write_sites (the surface of step 2).

The steps write into a new directory in ROOT (a derivatives root of its own,
STAGE_PREFIX + a random name). If a step fails, the command deletes the new
directory, and the old outputs are unchanged. If all the steps succeed, the
command writes the list of the files to move (MARKER) into the new directory,
then moves each file to the same path in ROOT, over the old file, one at a
time (os.replace), then deletes the new directory. The old outputs are not
kept.

The move of all the files is not atomic: a process that stops during the
moves leaves some new files and some old files in ROOT. The marker makes this
state recoverable (roll forward): finish(ROOT) moves the files that the
marker lists and that are still in the new directory, and deletes a new
directory that has no marker (a run that stopped before its moves; the old
outputs are unchanged). Each run calls finish first. A run holds an exclusive
lock (fcntl.flock) on ROOT, so a second run in the same ROOT stops with an
error, and finish never touches the new directory of a running command.
The diff (EDITS) must not be in ROOT itself: its reader locks the diff's
directory, and that lock would wait for the run's lock on ROOT.
This covers a stopped process (an error, a signal, the memory cap); it does
not cover a crash of the machine or of the file system.

Before the steps, the command copies into the new directory the files of ROOT
that the steps read: dataset_description.json, and the sidecars (.json) of the
kept fix diffs (baillarger.white_dseg.history_dir), so that the check of a
reproduced fix diff (white_dseg.check_reproduced) sees the kept runs. The
kept diffs and tables (.npy, .tsv) are not read, so they are not copied. A
copied file that the steps leave the same is not moved back.

The sidecars name their Sources by absolute path. Before the move, the command
changes the path of the new directory in each sidecar to the path of ROOT, so
the sidecars name the files at their final paths.

At the end, the command prints the overrides: each fix that changes edited
voxels (the fix table's overrides column), its voxel and the number of edited
voxels that it changes.
"""

import argparse
import filecmp
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baillarger import backend, white_dseg, white_overview, white_surf, wm_sites
from baillarger.bids import derivative_path, entities
from baillarger.fileops import flock, write_atomic
from baillarger.fix_diff import read_table
from baillarger.mesh import BLOCK
from baillarger.timing import log, log_to_stderr

#: the name prefix of the new directory in ROOT
STAGE_PREFIX = ".regenerate-"

#: the file that baillarger.bids.dataset_description writes in a root
DESCRIPTION = "dataset_description.json"

#: the list of the files to move (one path relative to the root on each
#: line), in the new directory: written when all the steps have succeeded
MARKER = "moves.txt"


@dataclass(frozen=True)
class Regenerated:
    """The result of a regeneration: the files written (their final paths in
    the root, sorted) and the fix table's rows that have overrides."""

    files: list[Path]
    overrides: list[dict[str, Any]]


def _history(root: Path, inp: Path) -> Path:
    """The history directory of the fix diffs in root, for the input labels
    inp (white_dseg.history_dir of the dseg's path)."""
    dseg = derivative_path(root, entities(inp), "dseg", ".nii.gz", desc="wmtopo")
    return white_dseg.history_dir(dseg)


def _seed(stage: Path, root: Path, inp: Path) -> list[Path]:
    """Copy into stage the files of root that the steps read. Returns their
    paths relative to root."""
    found = [root / DESCRIPTION] if (root / DESCRIPTION).is_file() else []
    old = _history(root, inp)
    if old.is_dir():
        found += sorted(old.glob("*.json"))
    rel = [p.relative_to(root) for p in found]
    for r in rel:
        (stage / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / r, stage / r)
    return rel


@contextmanager
def locked(root: Path) -> Iterator[None]:
    """An exclusive lock (fcntl.flock) on the directory root, for one
    regeneration. Raises RuntimeError at once if another process holds it."""
    with ExitStack() as stack:
        try:
            stack.enter_context(flock(root, exclusive=True, wait=False))
        except BlockingIOError:
            raise RuntimeError(f"another regenerate runs in {root}") from None
        yield


def _mark(stage: Path, rel: list[Path]) -> None:
    """Write the marker: a new file, synced, then renamed to MARKER."""
    write_atomic(stage / MARKER, "".join(f"{r.as_posix()}\n" for r in rel).encode())


def _move(stage: Path, root: Path) -> list[Path]:
    """Move the files that the marker of stage lists, and that are still in
    stage, to the same paths in root; then delete stage. Returns the final
    paths of all the files that the marker lists."""
    rel = [Path(x) for x in (stage / MARKER).read_text().splitlines()]
    log.info(f"regenerate: move {len(rel)} files into {root}")
    for r in rel:
        if (stage / r).is_file():
            (root / r).parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / r, root / r)
    shutil.rmtree(stage)
    return [root / r for r in rel]


def _finish(root: Path) -> list[Path]:
    """finish, with the lock held."""
    moved: list[Path] = []
    for stage in sorted(root.glob(STAGE_PREFIX + "*")):
        if not stage.is_dir():
            continue
        if (stage / MARKER).is_file():
            log.info(f"regenerate: finish the moves of {stage.name}")
            moved += _move(stage, root)
        else:
            log.info(f"regenerate: delete {stage.name}, a run without its moves")
            shutil.rmtree(stage)
    return moved


def finish(root: Path) -> list[Path]:
    """Finish the moves of a run in root that stopped during its moves (a new
    directory with the marker), and delete each new directory without the
    marker (a run that stopped before its moves). Returns the final paths of
    the files that the finished runs move. Raises RuntimeError if a
    regeneration runs in root."""
    with locked(root):
        return _finish(root)


def _relocate(stage: Path, root: Path) -> None:
    """In each JSON file in stage, change the path of stage to the path of
    root (the Sources of the sidecars)."""
    old = json.dumps(str(stage.resolve()))[1:-1]
    new = json.dumps(str(root.resolve()))[1:-1]
    for p in sorted(stage.rglob("*.json")):
        text = p.read_text()
        if old in text:
            p.write_text(text.replace(old, new))


def _steps(
    inp: Path,
    edits: Path | None,
    stage: Path,
    block: int,
    device: str | None,
    workers: int | None,
) -> Path:
    """The four steps, written into stage. Returns the fix table's path."""
    log.info("regenerate: step 1 of 4, the dseg (white_dseg)")
    d = white_dseg.write_wm_topo(inp, stage, edits)
    log.info("regenerate: step 2 of 4, the white surface (white_surf)")
    s = white_surf.write_white_surface(d["dseg"], stage, block, device)
    surf = s.paths["surf"]
    log.info("regenerate: step 3 of 4, the overview (white_overview)")
    white_overview.write_white_overview(surf, workers=workers)
    log.info("regenerate: step 4 of 4, the sites (wm_sites)")
    wm_sites.write_sites(surf)
    return d["table"]


def overrides(table: Path) -> list[dict[str, Any]]:
    """The rows of the fix table at table whose fix changes edited voxels."""
    return [r for r in read_table(table) if r["overrides"] is not None]


def regenerate(
    inp: Path,
    edits: Path | None,
    root: Path,
    block: int = BLOCK,
    device: str | None = None,
    workers: int | None = 4,
) -> Regenerated:
    """Run the four steps on the input labels inp with the diff edits into a
    new directory in root, then move their files over the old files in root
    (the module's docstring). First finishes a stopped run (finish). If a
    step fails, the new directory is deleted, the old outputs are unchanged,
    and the error is raised. Raises ValueError if edits is in root itself,
    or before the move if an output would replace inp or edits, and
    RuntimeError if another regeneration runs in root."""
    if edits is not None and edits.resolve().parent == root.resolve():
        raise ValueError(
            f"{edits.name}: the diff is in ROOT ({root}) itself; the run's lock "
            "on ROOT would block the diff's lock. Put the diff in a "
            "subdirectory (label_edits.bids_path puts it in anat/)"
        )
    root.mkdir(parents=True, exist_ok=True)
    keep = {inp.resolve()} | (set() if edits is None else {edits.resolve()})
    with locked(root):
        _finish(root)
        stage = Path(tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=root))
        try:
            seeded = _seed(stage, root, inp)
            table = _steps(inp, edits, stage, block, device, workers)
            _relocate(stage, root)
            rel = sorted(
                r
                for r in (p.relative_to(stage) for p in stage.rglob("*") if p.is_file())
                if not (r in seeded and filecmp.cmp(stage / r, root / r, shallow=False))
            )
            clash = [r for r in rel if (root / r).resolve() in keep]
            if clash:
                raise ValueError(f"an output would replace an input: {clash[0]}")
            _mark(stage, rel)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        written = _move(stage, root)
    final = root / table.relative_to(stage)
    return Regenerated(written, overrides(final))


def override_line(row: dict[str, Any]) -> str:
    """One printed line for a fix with overrides."""
    return (
        f"override: fix {row['id']} ({row['kind']}) at voxel "
        f"{row['i']},{row['j']},{row['k']} changes {row['overrides']} edited voxels"
    )


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(
        description="regenerate the dseg, the white surface, its overview and "
        "the sites from the input labels and the label edits"
    )
    ap.add_argument("input", type=Path, help="the input labels (the excised dseg)")
    ap.add_argument("--edits", type=Path, help="the diff of the label edits")
    ap.add_argument("--root", required=True, type=Path, help="BIDS derivatives root")
    ap.add_argument("--block-cells", type=int, default=BLOCK, help="block size")
    ap.add_argument(
        "--device",
        choices=backend.NAMES,
        help="where the mesh steps run (default: $BAILLARGER_DEVICE, else auto)",
    )
    ap.add_argument("--workers", type=int, default=4, help="processes to decimate")
    a = ap.parse_args(argv)
    r = regenerate(a.input, a.edits, a.root, a.block_cells, a.device, a.workers)
    for p in r.files:
        print(f"written: {p}")
    for row in r.overrides:
        print(override_line(row))
    print(f"overrides: {len(r.overrides)}")


if __name__ == "__main__":
    main()
