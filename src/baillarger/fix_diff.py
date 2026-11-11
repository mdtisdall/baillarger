"""The fixes of the WM topology correction as a table and a sparse diff
(docs/formats/wm-dseg.md, wm-dseg-fixes/1).

dseg reads the input labels, applies the label edits to them, then corrects
the topology. The diff is the change from the labels with the edits
applied ("old") to the labels that dseg writes ("new"): one row per changed
voxel, with the id of the fix that changed it last. Each fix has one row in
the table. The ids are in a fixed order (baillarger.white_dseg.wm_topo), so
the same input gives the same ids.

The diff is a NumPy .npy file of DIFF_DTYPE records (16 bytes a voxel), in
Morton order of the 64^3 block grid (baillarger.blocks.morton), and in C
order of the linear index within a block. apply_diff rebuilds the corrected
labels from the old labels and the diff; revert_fix undoes one fix, for
inspection.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger.blocks import CHUNK, morton

type Int = npt.NDArray[np.int64]
type Rows = npt.NDArray[np.void]

#: one changed voxel: its linear index (C order of the grid), the id of the
#: fix that changed it last, and its label before and after the correction
DIFF_DTYPE = np.dtype(
    [("index", "<i8"), ("fix", "<i4"), ("old", "<i2"), ("new", "<i2")]
)

#: the columns of the fix table
TABLE_COLUMNS = (
    "id",
    "kind",
    "voxels",
    "i",
    "j",
    "k",
    "x",
    "y",
    "z",
    "x_min",
    "y_min",
    "z_min",
    "x_max",
    "y_max",
    "z_max",
    "wall",
    "overrides",
)
_INT_COLUMNS = ("id", "voxels", "i", "j", "k", "wall", "overrides")


def morton_key(index: npt.ArrayLike, shape: tuple[int, ...]) -> Int:
    """The sort key of linear indices on a grid of this shape: the Morton
    code of the voxel's 64^3 block, then the linear index."""
    idx = np.asarray(index, np.int64)
    ijk = np.stack(np.unravel_index(idx, shape), -1)
    code = morton(ijk // CHUNK)
    order = np.lexsort((idx, code))
    key = np.empty(len(idx), np.int64)
    key[order] = np.arange(len(idx))
    return key


def make_diff(
    old: npt.ArrayLike,
    new: npt.ArrayLike,
    index: npt.ArrayLike,
    fix: npt.ArrayLike,
) -> Rows:
    """The diff rows of the voxels (linear indices) changed from the labels
    old to the labels new, with the fix id of each, in Morton order. Voxels
    whose label does not change are left out. Raises ValueError if a voxel
    appears twice, or if a label does not fit int16."""
    o, n = np.asarray(old).reshape(-1), np.asarray(new).reshape(-1)
    shape = np.asarray(old).shape
    idx = np.asarray(index, np.int64).ravel()
    fx = np.asarray(fix, np.int64).ravel()
    if len(np.unique(idx)) != len(idx):
        raise ValueError("a voxel appears twice in the diff")
    keep = o[idx] != n[idx]
    idx, fx = idx[keep], fx[keep]
    rows = np.zeros(len(idx), DIFF_DTYPE)
    for k, src in (("old", o), ("new", n)):
        v = src[idx].astype(np.int64)
        if np.any((v < -(1 << 15)) | (v >= 1 << 15)):
            raise ValueError("a label does not fit int16")
        rows[k] = v
    rows["index"], rows["fix"] = idx, fx
    order = np.argsort(morton_key(idx, shape), kind="stable")
    out: Rows = rows[order]
    return out


def check_diff(rows: Rows, shape: tuple[int, ...]) -> None:
    """Raises ValueError unless rows is a diff on a grid of this shape:
    DIFF_DTYPE, indices in the grid, each once, in Morton order, and new not
    old."""
    if rows.dtype != DIFF_DTYPE:
        raise ValueError(f"the diff has dtype {rows.dtype}, not {DIFF_DTYPE}")
    idx = rows["index"].astype(np.int64)
    size = int(np.prod(shape))
    if len(idx) and (idx.min() < 0 or idx.max() >= size):
        raise ValueError("a voxel of the diff is outside the grid")
    if len(np.unique(idx)) != len(idx):
        raise ValueError("a voxel appears twice in the diff")
    if np.any(morton_key(idx, shape) != np.arange(len(idx))):
        raise ValueError("the diff is not in Morton order")
    if np.any(rows["old"] == rows["new"]):
        raise ValueError("a row of the diff does not change its voxel")


def write_diff(path: Path, rows: Rows) -> None:
    """Write the diff rows as a .npy file (no pickle)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.save(fh, rows, allow_pickle=False)


def read_diff(path: Path, shape: tuple[int, ...]) -> Rows:
    """The diff at path, checked (check_diff) on a grid of this shape."""
    rows: Rows = np.load(path, allow_pickle=False)
    check_diff(rows, shape)
    return rows


def apply_diff(old: npt.ArrayLike, rows: Rows) -> npt.NDArray[Any]:
    """The corrected labels: a copy of the labels old (the input with the
    label edits applied) with the diff written in. Raises ValueError if a
    voxel's label is not the row's old label."""
    a = np.asarray(old)
    flat = a.reshape(-1)
    idx = rows["index"].astype(np.int64)
    if np.any(flat[idx] != rows["old"]):
        raise ValueError("a voxel does not have the old label of the diff")
    out = a.copy()
    out.reshape(-1)[idx] = rows["new"]
    return out


def revert_fix(new: npt.ArrayLike, rows: Rows, fix_id: int) -> npt.NDArray[Any]:
    """A copy of the corrected labels new with one fix undone: its voxels get
    their old labels. Only the voxels of that fix change. For inspection: the
    result can have a handle or a cavity."""
    a = np.asarray(new)
    sel = rows[rows["fix"] == fix_id]
    flat = a.reshape(-1)
    idx = sel["index"].astype(np.int64)
    if np.any(flat[idx] != sel["new"]):
        raise ValueError("a voxel does not have the new label of the diff")
    out = a.copy()
    out.reshape(-1)[idx] = sel["old"]
    return out


def fix_voxels(rows: Rows, fix_id: int) -> tuple[Int, Int]:
    """The linear indices (sorted) of one fix's voxels in the diff, and their
    new labels."""
    sel = rows[rows["fix"] == fix_id]
    order = np.argsort(sel["index"], kind="stable")
    return (
        sel["index"][order].astype(np.int64),
        sel["new"][order].astype(np.int64),
    )


# ------------------------------------------------------------------ the table


@dataclass(frozen=True)
class Fix:
    """One fix: its id, kind (a value of baillarger.white_dseg.FIXES), the
    linear indices of the voxels that it changed last, the size of its wall
    (voxels claimed by the other front; None if the fix has no wall) and the
    number of edited voxels that it changes (None if none)."""

    id: int
    kind: str
    index: Int
    wall: int | None = None
    overrides: int | None = None


def table_rows(
    fixes: list[Fix], shape: tuple[int, ...], affine: npt.ArrayLike
) -> list[dict[str, Any]]:
    """The table's rows: per fix, its voxel count, the voxel of the fix
    nearest its centroid (ties: the lowest linear index), and its centroid
    and bounding box (of the voxel centres) in mm. A fix with no voxel has no
    centre: n/a."""
    A = np.asarray(affine, np.float64)
    out = []
    for f in fixes:
        r: dict[str, Any] = dict(id=f.id, kind=f.kind, voxels=len(f.index))
        idx = np.sort(np.asarray(f.index, np.int64))
        if len(idx):
            ijk = np.stack(np.unravel_index(idx, shape), 1).astype(np.float64)
            mm = ijk @ A[:3, :3].T + A[:3, 3]
            c = ijk.mean(0)
            near = ijk[int(np.argmin(((ijk - c) ** 2).sum(1)))].astype(int)
            r.update(i=int(near[0]), j=int(near[1]), k=int(near[2]))
            cm, lo, hi = mm.mean(0), mm.min(0), mm.max(0)
            for a, n in enumerate("xyz"):
                r[n] = round(float(cm[a]), 4)
                r[f"{n}_min"] = round(float(lo[a]), 4)
                r[f"{n}_max"] = round(float(hi[a]), 4)
        r["wall"] = f.wall
        r["overrides"] = f.overrides
        out.append(r)
    return out


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the fix table (TSV, a header row; n/a for a missing value)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(TABLE_COLUMNS)
        for r in rows:
            w.writerow(["n/a" if r.get(c) is None else r[c] for c in TABLE_COLUMNS])


def read_table(path: Path) -> list[dict[str, Any]]:
    """The fix table at path, typed: integers, numbers (mm) and None for
    n/a."""
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if not rows or tuple(rows[0]) != TABLE_COLUMNS:
        raise ValueError(f"{path.name}: the columns are not {TABLE_COLUMNS}")
    out = []
    for row in rows[1:]:
        d: dict[str, Any] = {}
        for c, v in zip(TABLE_COLUMNS, row, strict=True):
            if v == "n/a":
                d[c] = None
            elif c == "kind":
                d[c] = v
            elif c in _INT_COLUMNS:
                d[c] = int(v)
            else:
                d[c] = float(v)
        out.append(d)
    return out
