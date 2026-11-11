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
STAGE_PREFIX + a random name). If all the steps succeed, the command moves
each file that they wrote to the same path in ROOT, over the old file. The
old outputs are not kept. If a step fails, the command deletes the new
directory, and the old outputs are unchanged.

Before the steps, the command copies into the new directory the files of ROOT
that the steps read: dataset_description.json, and the history of the fix
diffs (baillarger.white_dseg.history_dir), so that the check of a reproduced
fix diff sees the kept runs.

The sidecars name their Sources by absolute path. Before the move, the command
changes the path of the new directory in each sidecar to the path of ROOT, so
the sidecars name the files at their final paths.

At the end, the command prints the overrides: each fix that changes edited
voxels (the fix table's overrides column), its voxel and the number of edited
voxels that it changes.
"""

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baillarger import backend, white_dseg, white_overview, white_surf, wm_sites
from baillarger.bids import derivative_path, entities
from baillarger.fix_diff import read_table
from baillarger.mesh import BLOCK
from baillarger.timing import log, log_to_stderr

#: the name prefix of the new directory in ROOT
STAGE_PREFIX = ".regenerate-"

#: the file that baillarger.bids.dataset_description writes in a root
DESCRIPTION = "dataset_description.json"


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


def _seed(stage: Path, root: Path, inp: Path) -> None:
    """Copy into stage the files of root that the steps read."""
    desc = root / DESCRIPTION
    if desc.is_file():
        shutil.copyfile(desc, stage / DESCRIPTION)
    old = _history(root, inp)
    if old.is_dir():
        shutil.copytree(old, _history(stage, inp))


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
    (the module's docstring). If a step fails, the new directory is deleted,
    the old outputs are unchanged, and the error is raised. Raises ValueError
    before the move if an output would replace inp or edits."""
    root.mkdir(parents=True, exist_ok=True)
    keep = {inp.resolve()} | (set() if edits is None else {edits.resolve()})
    stage = Path(tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=root))
    try:
        _seed(stage, root, inp)
        table = _steps(inp, edits, stage, block, device, workers)
        _relocate(stage, root)
        rel = sorted(p.relative_to(stage) for p in stage.rglob("*") if p.is_file())
        clash = [r for r in rel if (root / r).resolve() in keep]
        if clash:
            raise ValueError(f"an output would replace an input: {clash[0]}")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    log.info(f"regenerate: move {len(rel)} files into {root}")
    for r in rel:
        (root / r).parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / r, root / r)
    shutil.rmtree(stage)
    final = root / table.relative_to(stage)
    return Regenerated([root / r for r in rel], overrides(final))


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
