"""The label edits of a hemisphere: one sparse diff (docs/formats/label-edits.md).

The diff holds the current state of the edits: each edited voxel and its
label. It holds no history. The dseg command (baillarger.white_dseg --edits)
applies the diff to the input labels before the automatic correction, and
the correction changes the edited voxels last. Nothing else applies edits.

Two files: the records (.npy, EDITS_DTYPE, sorted by index, each index once)
and the sidecar (.json). write replaces both files under an exclusive lock
(fileops.flock on their directory): the records first, then the sidecar.
read holds a shared lock and checks the records against the sidecar's SHA256
and Count, and the diff against the grid, affine and hemisphere that the
caller gives. A write that stopped between its two renames leaves a sidecar
older than the records: a reader checks the records on the sidecar's grid
and writes a new sidecar (the repair). update holds the exclusive lock
across one read, change and write, so that two writers do not lose each
other's voxels.

paint and restore change a diff in memory; differ compares two diffs;
changed gives the voxels where two label arrays differ.
"""

import hashlib
import io
import json
import logging
from collections.abc import Callable
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
from baillarger.fileops import flock, write_atomic
from baillarger.labels import LUT_NAMES

log = logging.getLogger(__name__)

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

    def check(
        self,
        shape: tuple[int, ...] | None = None,
        undo: bool = False,
        labels: bool = True,
    ) -> None:
        """Raises ValueError unless the indices are int64, sorted and unique
        (and in a grid of this shape, if given), and the labels int32 and in
        label-set/1 (or INPUT, for an undo). labels False skips the last
        check (the caller has checked the labels)."""
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
        if not labels:
            return
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


def _npy_bytes(edits: Edits) -> bytes:
    raw = io.BytesIO()
    np.save(raw, edits.records(), allow_pickle=False)
    return raw.getvalue()


def _stat(path: Path) -> tuple[int, int, int, int]:
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


@dataclass(frozen=True)
class Current:
    """A diff as one process last read or wrote it: its edits, the SHA-256
    of its records (None: no file) and the identity of its records file
    (device, inode, size, mtime; None: no file). update uses it as the
    memory copy: if the sidecar's SHA256 and the records file's identity
    agree, the records are not read again."""

    edits: Edits
    sha: str | None
    ident: tuple[int, int, int, int] | None


def _read_records(path: Path) -> bytes:
    """The bytes of a records file (a function, so that a test can count
    the reads)."""
    return path.read_bytes()


def _check_grid(
    path: Path,
    side: dict[str, Any],
    shape: tuple[int, ...] | None,
    affine: npt.ArrayLike | None,
    hemisphere: str | None,
) -> None:
    grid = tuple(int(s) for s in side["Shape"])
    if (shape is not None and grid != tuple(int(s) for s in shape)) or (
        affine is not None
        and not np.allclose(np.reshape(side["Affine"], (4, 4)), np.asarray(affine))
    ):
        want = "" if shape is None else f" {tuple(int(s) for s in shape)}"
        raise ValueError(
            f"{path.name}: the diff's grid {grid} (Shape and Affine) is not the "
            f"grid{want} that it is used on"
        )
    if hemisphere is not None and side["Hemisphere"] != hemisphere:
        raise ValueError(
            f"{path.name}: the diff is of hemisphere {side['Hemisphere']}, "
            f"not of hemisphere {hemisphere}"
        )


class _Stale(Exception):
    """The sidecar does not agree with the records (a write that stopped
    between its two renames): a repair needs the exclusive lock."""


def _sidecar(
    edits: Edits,
    sha: str,
    *,
    hemisphere: str,
    shape: tuple[int, ...],
    affine: npt.ArrayLike,
    description: str,
    function: str,
) -> dict[str, Any]:
    if hemisphere not in ("R", "L"):
        raise ValueError(f"hemisphere {hemisphere!r} is not R or L")
    A = np.asarray(affine, np.float64)
    if A.shape != (4, 4) or len(shape) != 3:
        raise ValueError("the grid has 3 axes and a 4 x 4 affine")
    side = dict(
        SpecVersion=EDITS_SPEC.version,
        Description=description,
        Hemisphere=hemisphere,
        Shape=[int(s) for s in shape],
        Affine=[float(x) for x in A.ravel()],
        Count=len(edits),
        SHA256=sha,
        Provenance=provenance(function),
    )
    EDITS_SPEC.check(side)
    return side


def _side_bytes(side: dict[str, Any]) -> bytes:
    return (json.dumps(side, indent=1) + "\n").encode()


def _repair(path: Path, side: dict[str, Any], data: bytes, sha: str) -> Edits:
    """The records of a write that stopped between its two renames, checked
    on the sidecar's grid (Edits.check), and a new sidecar for them: the old
    sidecar's Description, Hemisphere, Shape and Affine, with the records'
    Count and SHA256. Call with the exclusive lock held. Raises ValueError
    if the records are not a valid diff on that grid."""
    try:
        rec = np.load(io.BytesIO(data), allow_pickle=False)
        if rec.dtype != EDITS_DTYPE or rec.ndim != 1:
            raise ValueError(f"the records are not {EDITS_DTYPE}")
        e = _edits(rec["index"], rec["label"])
        e.check(tuple(side["Shape"]))
    except ValueError as err:
        raise ValueError(
            f"{path.name}: the records do not agree with the sidecar's SHA-256 "
            f"or Count, and they are not a valid diff on its grid: {err}"
        ) from err
    new = _sidecar(
        e,
        sha,
        hemisphere=side["Hemisphere"],
        shape=tuple(side["Shape"]),
        affine=np.reshape(side["Affine"], (4, 4)),
        description=side["Description"],
        function="baillarger.label_edits.repair",
    )
    write_atomic(sidecar_path(path), _side_bytes(new))
    log.warning(
        f"{path.name}: the sidecar was older than the records (a write that "
        f"stopped between its two renames); a new sidecar is written for the "
        f"{len(e)} records"
    )
    return e


def _load(
    path: Path, known: Current | None, repair: bool
) -> tuple[Current, dict[str, Any]]:
    """The diff at path and its sidecar. Call with a lock held: the
    exclusive lock if repair, else the shared lock. If known agrees with the
    sidecar's SHA256 and the records file's identity, the records are not
    read. A records file that does not agree with its sidecar is repaired
    (_repair) if repair, else _Stale is raised."""
    p = edits_paths(path)
    side = read_sidecar(p["sidecar"], EDITS_SPEC)
    ident = _stat(p["records"])
    if known is not None and known.sha == side["SHA256"] and known.ident == ident:
        return known, side
    data = _read_records(p["records"])
    sha = hashlib.sha256(data).hexdigest()
    rec = np.load(io.BytesIO(data), allow_pickle=False)
    if sha != side["SHA256"] or rec.ndim != 1 or len(rec) != side["Count"]:
        if not repair:
            raise _Stale
        e = _repair(path, side, data, sha)
        return Current(e, sha, ident), read_sidecar(p["sidecar"], EDITS_SPEC)
    if rec.dtype != EDITS_DTYPE:
        raise ValueError(f"{path.name}: the records are not {EDITS_DTYPE}")
    e = _edits(rec["index"], rec["label"])
    try:
        e.check(tuple(side["Shape"]))
    except ValueError as err:
        raise ValueError(f"{path.name}: {err}") from err
    return Current(e, sha, ident), side


def _write_locked(
    path: Path,
    edits: Edits,
    *,
    hemisphere: str,
    shape: tuple[int, ...],
    affine: npt.ArrayLike,
    description: str,
    labels_checked: bool = False,
) -> Current:
    """write, with the exclusive lock held."""
    p = edits_paths(path)
    edits.check(tuple(shape), labels=not labels_checked)
    data = _npy_bytes(edits)
    sha = hashlib.sha256(data).hexdigest()
    side = _sidecar(
        edits,
        sha,
        hemisphere=hemisphere,
        shape=shape,
        affine=affine,
        description=description,
        function="baillarger.label_edits.write",
    )
    write_atomic(p["records"], data)
    write_atomic(p["sidecar"], _side_bytes(side))
    return Current(edits, sha, _stat(p["records"]))


DESCRIPTION = "The label edits of a hemisphere: each edited voxel and its label."


def write(
    path: Path,
    edits: Edits,
    *,
    hemisphere: str,
    shape: tuple[int, ...],
    affine: npt.ArrayLike,
    description: str = DESCRIPTION,
) -> Current:
    """Write the diff at path (a .npy) and its sidecar (label-edits/1). Each
    file is written as a new file in the same directory, then renamed over the
    old one (the records first, then the sidecar; fileops.write_atomic), under
    an exclusive lock on the directory. Raises ValueError for a diff that is
    not valid on the grid (Edits.check) or a hemisphere that is not R or L.
    Returns the diff as written (Current)."""
    edits_paths(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with flock(path.parent, exclusive=True):
        return _write_locked(
            path,
            edits,
            hemisphere=hemisphere,
            shape=shape,
            affine=affine,
            description=description,
        )


def read_with_sidecar(
    path: Path,
    *,
    shape: tuple[int, ...] | None = None,
    affine: npt.ArrayLike | None = None,
    hemisphere: str | None = None,
) -> tuple[Edits, dict[str, Any], str]:
    """The diff at path, its sidecar and the SHA-256 of its .npy file, read
    under a shared lock. If the records do not agree with the sidecar's
    SHA256 or Count (a write that stopped between its two renames), the
    sidecar is older than the records: the records are checked on the
    sidecar's grid and a new sidecar is written, under the exclusive lock,
    with a log line. Raises ValueError if the records are not a valid diff on
    the sidecar's grid (Edits.check), or if the diff is not on the grid of
    this shape and affine, or not of this hemisphere (each when given)."""
    edits_paths(path)
    try:
        with flock(path.parent, exclusive=False):
            cur, side = _load(path, None, repair=False)
    except _Stale:
        with flock(path.parent, exclusive=True):
            cur, side = _load(path, None, repair=True)
    _check_grid(path, side, shape, affine, hemisphere)
    assert cur.sha is not None
    return cur.edits, side, cur.sha


def read(
    path: Path,
    *,
    shape: tuple[int, ...] | None = None,
    affine: npt.ArrayLike | None = None,
    hemisphere: str | None = None,
) -> Edits:
    """The diff at path (read_with_sidecar, with its checks)."""
    return read_with_sidecar(path, shape=shape, affine=affine, hemisphere=hemisphere)[0]


def update(
    path: Path,
    fn: Callable[[Current], Edits | None],
    *,
    hemisphere: str,
    shape: tuple[int, ...],
    affine: npt.ArrayLike,
    known: Current | None = None,
    labels_checked: bool = False,
) -> Current:
    """One read-change-write of the diff at path, under the exclusive lock
    on its directory for all of it, so that no other writer (in this or
    another process) writes between the read and the write. Reads the diff
    (the empty diff if there is no file; a stopped write is repaired as in
    read_with_sidecar), checks that it is on this grid and of this
    hemisphere, calls fn with it, and writes what fn returns (nothing if fn
    returns None). known is the caller's memory copy: if it agrees with the
    sidecar, only the sidecar is read. labels_checked skips the check of the
    labels against label-set/1 (the caller, label_edits.paint, has checked
    them). An error from fn propagates, and nothing is written. Returns the
    diff on disk after the update."""
    p = edits_paths(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with flock(path.parent, exclusive=True):
        if p["sidecar"].exists():
            cur, side = _load(path, known, repair=True)
            _check_grid(path, side, shape, affine, hemisphere)
        elif p["records"].exists():
            # the first write stopped between its two renames: the grid is
            # this update's
            data = _read_records(p["records"])
            stub = dict(
                Description=DESCRIPTION,
                Hemisphere=hemisphere,
                Shape=[int(s) for s in shape],
                Affine=np.asarray(affine, np.float64).ravel().tolist(),
            )
            sha = hashlib.sha256(data).hexdigest()
            e = _repair(path, stub, data, sha)
            cur = Current(e, sha, _stat(p["records"]))
        else:
            cur = Current(empty(), None, None)
        new = fn(cur)
        if new is None:
            return cur
        return _write_locked(
            path,
            new,
            hemisphere=hemisphere,
            shape=shape,
            affine=affine,
            description=DESCRIPTION,
            labels_checked=labels_checked,
        )


# ------------------------------------------------------------ the changes


def apply(
    labels: npt.ArrayLike,
    edits: Edits,
    *,
    offset: npt.ArrayLike = (0, 0, 0),
    shape: tuple[int, ...] | None = None,
) -> npt.NDArray[Any]:
    """A copy of the labels with the diff's labels written at its voxels.
    labels are the whole grid, or a crop of the grid of this shape whose
    voxel [0, 0, 0] is the grid voxel offset (as in paint); the diff's voxels
    outside the crop are left out. Raises ValueError if a voxel is outside
    the grid or a label does not fit the labels' type."""
    a = np.asarray(labels)
    grid = a.shape if shape is None else tuple(int(n) for n in shape)
    off = np.asarray(offset, np.int64)
    edits.check(grid)
    lab = edits.label.astype(np.int64)
    if np.any(lab != lab.astype(a.dtype)):
        raise ValueError(f"a label of the diff does not fit {a.dtype}")
    out = a.copy()
    if grid == a.shape and not off.any():
        out.reshape(-1)[edits.index] = edits.label
        return out
    q = np.stack(np.unravel_index(edits.index, grid), 1).astype(np.int64) - off
    ok = np.all((q >= 0) & (q < np.asarray(a.shape)), axis=1)
    out[tuple(q[ok].T)] = edits.label[ok]
    return out


def _set(edits: Edits, index: Int, label: npt.ArrayLike) -> Edits:
    """The diff with the voxels at index (sorted, unique) set to label: INPUT
    removes the voxel's entry. A merge of two sorted arrays (searchsorted):
    O(N log E + E) for N voxels and E entries, with no sort of the diff."""
    lab = np.asarray(label, np.int32)
    if not len(index):
        return edits
    pos = np.searchsorted(edits.index, index)
    hit = pos < len(edits)
    hit[hit] = edits.index[pos[hit]] == index[hit]
    keep = np.ones(len(edits), bool)
    keep[pos[hit]] = False
    kept_idx, kept_lab = edits.index[keep], edits.label[keep]
    add = lab != INPUT
    at = np.searchsorted(kept_idx, index[add])
    return _edits(
        np.insert(kept_idx, at, index[add]), np.insert(kept_lab, at, lab[add])
    )


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
