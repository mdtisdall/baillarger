"""The label edits of a hemisphere: one sparse diff (docs/formats/label-edits.md).

The diff holds the current state of the edits: each edited voxel and its
label. It holds no history. The dseg command (baillarger.white_dseg --edits)
applies the diff to the input labels before the automatic correction, and
the correction changes the edited voxels last. Nothing else applies edits.

Two files: the records (.npy, EDITS_DTYPE, sorted by index, each index once)
and the sidecar (.json). write replaces both files under an exclusive lock
(fcntl.flock on their directory); read holds a shared lock and checks the
sidecar's SHA256. So a reader sees the old diff or the new diff, never a mix.

paint and restore change a diff in memory; differ compares two diffs;
changed gives the voxels where two label arrays differ.
"""

import fcntl
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger.bids import (
    Key,
    SidecarSpec,
    derivative_path,
    provenance,
    read_sidecar,
    sidecar_path,
)
from baillarger.labels import LUT_NAMES

type Int = npt.NDArray[np.int64]
type Int32 = npt.NDArray[np.int32]

#: docs/formats/label-edits.md
EDITS_SPEC = SidecarSpec(
    "label-edits/1",
    (
        Key("SpecVersion", "string", True, "format and version of the spec"),
        Key("Description", "string", True, "what the file is"),
        Key("Hemisphere", "string", True, "R or L"),
        Key("Shape", "array of integers", True, "the grid's shape"),
        Key("Affine", "array of numbers", True, "voxel to mm, 4 x 4, by rows"),
        Key("Count", "integer", True, "number of edited voxels"),
        Key("SHA256", "string", True, "SHA-256 of the .npy file"),
        Key("Provenance", "object", True, "Created, Function, Version, Commit, Dirty"),
    ),
)

#: one edited voxel: its linear index (C order of the grid) and its label
EDITS_DTYPE = np.dtype([("index", "<i8"), ("label", "<i4")])

#: the label of "no entry" in an undo, and of the eraser in paint: the voxel
#: takes its input label
INPUT = -1


@dataclass(frozen=True)
class Edits:
    """A diff: the linear indices of the edited voxels (int64, sorted, each
    once) and their labels (int32). An undo (paint, restore) has the same
    form, with the label INPUT where the voxel had no entry."""

    index: Int
    label: Int32

    def __len__(self) -> int:
        return len(self.index)

    def check(self, shape: tuple[int, ...] | None = None, undo: bool = False) -> None:
        """Raises ValueError unless the indices are int64, sorted and unique
        (and in a grid of this shape, if given), and the labels int32 and in
        label-set/1 (or INPUT, for an undo)."""
        if self.index.dtype != np.int64 or self.label.dtype != np.int32:
            raise ValueError("the index is not int64 or the label is not int32")
        if self.index.shape != self.label.shape or self.index.ndim != 1:
            raise ValueError("the index and the label differ in shape")
        if len(self.index) and np.any(np.diff(self.index) <= 0):
            raise ValueError("the index is not sorted, or a voxel appears twice")
        if shape is not None and len(self.index):
            size = int(np.prod(shape))
            if self.index[0] < 0 or self.index[-1] >= size:
                raise ValueError("an edited voxel is outside the grid")
        allowed = set(LUT_NAMES) | ({INPUT} if undo else set())
        bad = sorted(set(np.unique(self.label).tolist()) - allowed)
        if bad:
            raise ValueError(f"labels {bad} are not in label-set/1")

    def records(self) -> npt.NDArray[np.void]:
        """The diff as EDITS_DTYPE records (the .npy file's content)."""
        out = np.zeros(len(self), EDITS_DTYPE)
        out["index"], out["label"] = self.index, self.label
        return out


def _edits(index: npt.ArrayLike, label: npt.ArrayLike) -> Edits:
    return Edits(np.asarray(index, np.int64), np.asarray(label, np.int32))


def empty() -> Edits:
    """The diff with no edited voxel."""
    return _edits(np.zeros(0, np.int64), np.zeros(0, np.int32))


# ------------------------------------------------------------ the files


def bids_path(root: Path, ents: dict[str, str], hemi: str) -> Path:
    """The diff of a hemisphere of the input with entities ents, in the
    derivatives root: sub-<sub>_ses-<ses>_hemi-<h>_desc-labeledits_edits.npy."""
    return derivative_path(root, ents, "edits", ".npy", desc="labeledits", hemi=hemi)


def edits_paths(path: Path) -> dict[str, Path]:
    """The two files of the diff at path (a .npy): records and sidecar."""
    if not path.name.endswith(".npy"):
        raise ValueError(f"{path.name}: the diff is a .npy file")
    return dict(records=path, sidecar=sidecar_path(path))


@contextmanager
def _locked(directory: Path, op: int) -> Iterator[None]:
    """A lock (fcntl.flock) on the directory: LOCK_EX for a write, LOCK_SH for
    a read. The files are replaced by rename, so the lock is not on them."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, op)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _npy_bytes(edits: Edits) -> bytes:
    raw = io.BytesIO()
    np.save(raw, edits.records(), allow_pickle=False)
    return raw.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    """Write data to a new file in path's directory, then rename it over
    path."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write(
    path: Path,
    edits: Edits,
    *,
    hemisphere: str,
    shape: tuple[int, ...],
    affine: npt.ArrayLike,
    description: str = "The label edits of a hemisphere: each edited voxel and "
    "its label.",
) -> None:
    """Write the diff at path (a .npy) and its sidecar (label-edits/1). Each
    file is written as a new file in the same directory, then renamed over the
    old one (the records first, then the sidecar), under an exclusive lock.
    Raises ValueError for a diff that is not valid on the grid (Edits.check)
    or a hemisphere that is not R or L."""
    p = edits_paths(path)
    edits.check(tuple(shape))
    if hemisphere not in ("R", "L"):
        raise ValueError(f"hemisphere {hemisphere!r} is not R or L")
    A = np.asarray(affine, np.float64)
    if A.shape != (4, 4) or len(shape) != 3:
        raise ValueError("the grid has 3 axes and a 4 x 4 affine")
    data = _npy_bytes(edits)
    side = dict(
        SpecVersion=EDITS_SPEC.version,
        Description=description,
        Hemisphere=hemisphere,
        Shape=[int(s) for s in shape],
        Affine=[float(x) for x in A.ravel()],
        Count=len(edits),
        SHA256=hashlib.sha256(data).hexdigest(),
        Provenance=provenance("baillarger.label_edits.write"),
    )
    EDITS_SPEC.check(side)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path.parent, fcntl.LOCK_EX):
        _write_atomic(p["records"], data)
        _write_atomic(p["sidecar"], (json.dumps(side, indent=1) + "\n").encode())


def read_with_sidecar(path: Path) -> tuple[Edits, dict[str, Any], str]:
    """The diff at path, its sidecar and the SHA-256 of its .npy file, read
    under a shared lock. Raises ValueError if the .npy file does not have the
    sidecar's SHA256 or Count, or is not a valid diff on the sidecar's grid
    (Edits.check)."""
    p = edits_paths(path)
    with _locked(path.parent, fcntl.LOCK_SH):
        side = read_sidecar(p["sidecar"], EDITS_SPEC)
        data = p["records"].read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != side["SHA256"]:
        raise ValueError(f"{path.name}: the SHA-256 does not agree with the sidecar")
    rec = np.load(io.BytesIO(data), allow_pickle=False)
    if rec.dtype != EDITS_DTYPE or rec.ndim != 1:
        raise ValueError(f"{path.name}: the records are not {EDITS_DTYPE}")
    if len(rec) != side["Count"]:
        raise ValueError(
            f"{path.name}: {len(rec)} records, the sidecar has Count {side['Count']}"
        )
    e = _edits(rec["index"], rec["label"])
    try:
        e.check(tuple(side["Shape"]))
    except ValueError as err:
        raise ValueError(f"{path.name}: {err}") from err
    return e, side, sha


def read(path: Path) -> Edits:
    """The diff at path (read_with_sidecar)."""
    return read_with_sidecar(path)[0]


# ------------------------------------------------------------ the changes


def apply(labels: npt.ArrayLike, edits: Edits) -> npt.NDArray[Any]:
    """A copy of the labels with the diff's labels written at its voxels.
    Raises ValueError if a voxel is outside the labels or a label does not
    fit the labels' type."""
    a = np.asarray(labels)
    edits.check(a.shape)
    lab = edits.label.astype(np.int64)
    if np.any(lab != lab.astype(a.dtype)):
        raise ValueError(f"a label of the diff does not fit {a.dtype}")
    out = a.copy()
    out.reshape(-1)[edits.index] = edits.label
    return out


def _set(edits: Edits, index: Int, label: npt.ArrayLike) -> Edits:
    """The diff with the voxels at index (unique) set to label: INPUT removes
    the voxel's entry."""
    lab = np.asarray(label, np.int32)
    keep = ~np.isin(edits.index, index)
    add = lab != INPUT
    idx = np.concatenate([edits.index[keep], index[add]])
    out = np.concatenate([edits.label[keep], lab[add]])
    order = np.argsort(idx, kind="stable")
    return _edits(idx[order], out[order])


def _entries(edits: Edits, index: Int) -> Int32:
    """The diff's label at each voxel of index, INPUT where it has none."""
    out = np.full(len(index), INPUT, np.int32)
    if not len(edits):
        return out
    at = np.minimum(np.searchsorted(edits.index, index), len(edits) - 1)
    hit = edits.index[at] == index
    out[hit] = edits.label[at[hit]]
    return out


def paint(
    edits: Edits,
    labels: npt.ArrayLike,
    voxels: npt.ArrayLike,
    new: npt.ArrayLike,
    *,
    dseg: npt.ArrayLike | None = None,
    offset: npt.ArrayLike = (0, 0, 0),
    shape: tuple[int, ...] | None = None,
) -> tuple[Edits, Edits]:
    """Paint voxels (N, 3) of the grid with new (a label per voxel, or INPUT
    for the eraser). labels are the input labels and dseg the dseg's labels
    (the labels that the user sees: the input labels after the automatic
    correction), both of one shape: the whole grid, or a crop of the grid of
    this shape whose voxel [0, 0, 0] is the grid voxel offset. dseg None
    means that the dseg is the input labels (no correction).

    Only the voxels whose label changes are written. A paint compares with
    the voxel's current label: its entry in the diff if it has one, else its
    dseg label. A painted voxel whose new label is not its current label gets
    an entry with its new label, also when the new label is its input label
    (a paint back to the input label, against the correction). The eraser
    removes the entry of a voxel that has one, and changes no other voxel.
    Returns the new diff and the undo: the earlier entries of the changed
    voxels (INPUT for no entry). Raises ValueError for a voxel outside the
    grid or the labels, labels and dseg of different shapes, a voxel given
    twice, or a label not in label-set/1."""
    a = np.asarray(labels)
    d = a if dseg is None else np.asarray(dseg)
    if d.shape != a.shape:
        raise ValueError(
            f"the input labels {a.shape} and the dseg {d.shape} differ in shape"
        )
    grid = a.shape if shape is None else tuple(shape)
    off = np.asarray(offset, np.int64)
    v = np.asarray(voxels, np.int64).reshape(-1, 3)
    nl = np.asarray(new, np.int64).reshape(-1)
    if len(v) != len(nl):
        raise ValueError("voxels and labels differ in length")
    if len(v) and not np.all((v >= 0) & (v < grid)):
        raise ValueError("a voxel is outside the grid")
    if len(v) and not np.all((v >= off) & (v - off < a.shape)):
        raise ValueError("a voxel is outside the labels")
    bad = sorted(set(nl.tolist()) - set(LUT_NAMES) - {INPUT})
    if bad:
        raise ValueError(f"labels {bad} are not in label-set/1")
    idx: Int = np.zeros(0, np.int64)
    if len(v):
        idx = np.asarray(np.ravel_multi_index(tuple(v.T), grid), np.int64)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("a voxel is given twice")
    shown = d[tuple((v - off).T)].astype(np.int64) if len(v) else idx
    earlier = _entries(edits, idx)
    current = np.where(earlier == INPUT, shown, earlier)
    ch = np.where(nl == INPUT, earlier != INPUT, nl != current)
    idx, nl, earlier = idx[ch], nl[ch], earlier[ch]
    order = np.argsort(idx)
    idx, nl, earlier = idx[order], nl[order], earlier[order]
    return _set(edits, idx, nl), _edits(idx, earlier)


def restore(edits: Edits, undo: Edits) -> Edits:
    """The diff with the entries of an undo (paint) written back: a voxel
    with the label INPUT loses its entry."""
    undo.check(undo=True)
    return _set(edits, undo.index, undo.label)


def differ(a: Edits, b: Edits) -> tuple[Int, Int32]:
    """The voxels (linear indices, sorted) where the diffs a and b differ: an
    entry in one only, or different labels. With each, its label in a
    (INPUT where a has no entry)."""
    idx = np.union1d(a.index, b.index).astype(np.int64)
    la, lb = _entries(a, idx), _entries(b, idx)
    d = la != lb
    return idx[d], la[d]


def changed(
    before: npt.ArrayLike, after: npt.ArrayLike, offset: npt.ArrayLike = (0, 0, 0)
) -> tuple[Int, Int]:
    """The voxels (N, 3, C order) where two label arrays of one shape differ,
    plus offset (the grid voxel of their voxel [0, 0, 0], for a crop), and
    their labels in after. Raises ValueError if the shapes differ."""
    b, a = np.asarray(before), np.asarray(after)
    if b.shape != a.shape:
        raise ValueError(f"the shapes {b.shape} and {a.shape} differ")
    mask = b != a
    pts = np.argwhere(mask).astype(np.int64).reshape(-1, 3)
    return pts + np.asarray(offset, np.int64), a[mask].astype(np.int64).reshape(-1)
