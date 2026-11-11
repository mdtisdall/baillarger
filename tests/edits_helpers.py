"""Helpers for the tests that build, read and write label edits
(baillarger.label_edits): one copy for every test module."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from baillarger import bids, white_dseg
from baillarger import label_edits as LE

Voxel = tuple[int, int, int]

#: the rod across the slit of test_white_dseg.synthetic_aseg (its one cut),
#: and the same rod 2 voxels lower
ROD = [(27, 16, 25), (27, 17, 25)]
ROD_LOWER = [(27, 16, 23), (27, 17, 23)]


def edits(pairs: dict[Voxel, int], shape: tuple[int, ...]) -> LE.Edits:
    """A diff from {voxel: label} on a grid of this shape."""
    idx = [int(np.ravel_multi_index(v, shape)) for v in pairs]
    order = np.argsort(idx)
    return LE.Edits(
        np.asarray(idx, np.int64)[order],
        np.asarray(list(pairs.values()), np.int32)[order],
    )


def as_dict(e: LE.Edits, shape: tuple[int, ...]) -> dict[tuple[int, ...], int]:
    """A diff on a grid of this shape as {voxel: label}."""
    v = np.stack(np.unravel_index(e.index, shape), 1).tolist()
    return {tuple(x): int(y) for x, y in zip(v, e.label.tolist(), strict=True)}


def ring_edits(aseg: npt.NDArray[np.integer]) -> LE.Edits:
    """The whole ring of test_white_dseg.synthetic_aseg and its hole, rod
    included, with the labels that they have in aseg: the handle can be fixed
    only through an edited voxel (an override)."""
    box = np.zeros(aseg.shape, bool)
    box[25:37, 3:33, 22:28] = True
    idx = np.flatnonzero(box).astype(np.int64)
    return LE.Edits(idx, aseg.reshape(-1)[idx].astype(np.int32))


def move_rod(e: LE.Edits, aseg: npt.NDArray[np.integer]) -> LE.Edits:
    """The diff e with the rod painted 2 voxels lower (label_edits.paint):
    outside at the rod, WM below it."""
    out, _ = LE.paint(e, aseg, ROD + ROD_LOWER, [0, 0, 41, 41])
    return out


def write_edits(
    path: Path,
    e: LE.Edits,
    shape: tuple[int, ...],
    affine: npt.ArrayLike,
    hemisphere: str = "R",
) -> Path:
    """The diff e written at path (label_edits.write). Returns path."""
    LE.write(path, e, hemisphere=hemisphere, shape=shape, affine=affine)
    return path


def write_dseg_sidecar(dseg: Path, inp: Path) -> None:
    """A wm-dseg/1 sidecar for the dseg at dseg, with the input labels inp as
    its first Source, of the right hemisphere."""
    bids.write_sidecar(
        bids.sidecar_path(dseg),
        white_dseg.DSEG_SPEC,
        dict(
            SpecVersion=white_dseg.DSEG_SPEC.version,
            Description="a test dseg",
            **bids.sources(inp),
            Provenance=bids.provenance("tests"),
            Hemisphere="R",
            ToOutside={},
            FilledLabels=[41],
            WMLabel=41,
            BettiBefore=[1, 0, 0],
            BettiAfter=[1, 0, 0],
            CavityVoxels=0,
            HandleFillVoxels=0,
            HandleCutVoxels=0,
            CutLabels={},
            SmallPieces=0,
            SmallPieceVoxels=0,
            ManualVoxels=0,
            Moves=[],
        ),
    )
