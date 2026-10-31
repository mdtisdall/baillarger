"""Manual label edits for the WM topology correction (docs/formats/wm-edits.md).

The edits are diffs against one input segmentation (the aseg), which is never
changed: per voxel its old and new label and the answer it belongs to, and per
answer the site, the choice and when. The dseg command
(baillarger.white.write_wm_topo) applies them before its automatic correction,
so no edit can leave a handle, and they apply again when the correction
changes.

Edits is immutable: with_answer adds an answer and its voxels, without removes
one (to revert it). write_edits checks the edits against their segmentation
and writes the table and its sidecar; read_edits reads and checks them;
apply_edits gives the segmentation with the edits made.
"""

import csv
import datetime
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt

from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    derivative_path,
    file_sha256,
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)

type Int = npt.NDArray[np.int64]

#: docs/formats/wm-edits.md
EDITS_SPEC = SidecarSpec(
    "wm-edits/1",
    COMMON_KEYS + (Key("Answers", "array of objects", True, "the answers"),),
)

COLUMNS = ("i", "j", "k", "old", "new", "answer")


def edits_path(root: Path, ents: dict[str, str]) -> Path:
    """The edits table of an input with entities ents, in the derivatives
    root."""
    return derivative_path(root, ents, "edits", ".tsv", desc="wmedits")


@dataclass(frozen=True)
class Answer:
    """One answer: Id (unique), the site voxel, the choice, the SHA-256 of the
    dseg looked at (None if none) and when (ISO 8601)."""

    id: int
    site: tuple[int, int, int]
    choice: str
    decided_against: str | None = None
    time: str = field(
        default_factory=lambda: (
            datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        )
    )

    def to_json(self) -> dict[str, Any]:
        return dict(
            Id=self.id,
            Site=list(self.site),
            Choice=self.choice,
            DecidedAgainst=self.decided_against,
            Time=self.time,
        )

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Answer":
        if set(d) != {"Id", "Site", "Choice", "DecidedAgainst", "Time"}:
            raise ValueError(f"an answer has the fields {sorted(d)}")
        site = d["Site"]
        ok = (
            _is_int(d["Id"])
            and isinstance(site, list)
            and len(site) == 3
            and all(_is_int(t) for t in site)
            and isinstance(d["Choice"], str)
            and (d["DecidedAgainst"] is None or isinstance(d["DecidedAgainst"], str))
            and isinstance(d["Time"], str)
        )
        if not ok:
            raise ValueError(f"answer {d.get('Id')!r}: a field has the wrong type")
        return Answer(
            d["Id"],
            (site[0], site[1], site[2]),
            d["Choice"],
            d["DecidedAgainst"],
            d["Time"],
        )


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


@dataclass(frozen=True)
class Edits:
    """The rows (voxel (N, 3), old, new, answer: one per changed voxel) and
    the answers."""

    voxels: Int = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    old: Int = field(default_factory=lambda: np.zeros(0, np.int64))
    new: Int = field(default_factory=lambda: np.zeros(0, np.int64))
    answer: Int = field(default_factory=lambda: np.zeros(0, np.int64))
    answers: tuple[Answer, ...] = ()

    def with_answer(
        self,
        a: Answer,
        voxels: npt.ArrayLike = (),
        old: npt.ArrayLike = (),
        new: npt.ArrayLike = (),
    ) -> "Edits":
        """These edits and answer a with its rows (none for an answer that
        changes nothing). Raises ValueError for an Id already used."""
        if any(b.id == a.id for b in self.answers):
            raise ValueError(f"answer {a.id} is there already")
        v = np.asarray(voxels, np.int64).reshape(-1, 3)
        o = np.asarray(old, np.int64).ravel()
        n = np.asarray(new, np.int64).ravel()
        if not len(v) == len(o) == len(n):
            raise ValueError("voxels, old and new differ in length")
        return Edits(
            np.concatenate([self.voxels, v]),
            np.concatenate([self.old, o]),
            np.concatenate([self.new, n]),
            np.concatenate([self.answer, np.full(len(v), a.id, np.int64)]),
            self.answers + (a,),
        )

    def without(self, answer_id: int) -> "Edits":
        """These edits without one answer and its rows."""
        keep = self.answer != answer_id
        return replace(
            self,
            voxels=self.voxels[keep],
            old=self.old[keep],
            new=self.new[keep],
            answer=self.answer[keep],
            answers=tuple(a for a in self.answers if a.id != answer_id),
        )

    def check(self, labels: npt.ArrayLike) -> None:
        """Raises ValueError unless the edits are consistent and fit the
        segmentation labels: unique answer Ids, rows naming them, voxels inside
        the grid and each at most once, old the label there, new not old."""
        a = np.asarray(labels)
        ids = [b.id for b in self.answers]
        problems = []
        if len(set(ids)) != len(ids):
            problems.append("answer Ids are not unique")
        if not np.isin(self.answer, ids).all():
            problems.append("rows name answers that are not there")
        inside = np.all((self.voxels >= 0) & (self.voxels < a.shape), axis=1)
        if not inside.all():
            problems.append(f"{int((~inside).sum())} voxels outside the grid")
        elif len(self.voxels):
            if len(np.unique(self.voxels, axis=0)) != len(self.voxels):
                problems.append("a voxel appears twice")
            got = a[tuple(self.voxels.T)]
            wrong = got != self.old
            if wrong.any():
                first = int(np.argmax(wrong))
                v, g, o = self.voxels[first].tolist(), got[first], self.old[first]
                problems.append(
                    f"{int(wrong.sum())} voxels do not have their old label "
                    f"(the first: {v} has {g}, not {o})"
                )
        if (self.new == self.old).any():
            problems.append("a row's new label is its old label")
        if problems:
            raise ValueError("; ".join(problems))


def apply_edits(
    labels: npt.ArrayLike, edits: Edits
) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
    """A copy of the segmentation labels with the edits made, and the mask of
    the voxels they changed. Raises ValueError for edits that do not fit it
    (Edits.check)."""
    a = np.asarray(labels)
    edits.check(a)
    out = a.copy()
    changed = np.zeros(a.shape, bool)
    if len(edits.voxels):
        ix = tuple(edits.voxels.T)
        new = edits.new
        if np.any(new != new.astype(a.dtype)):
            raise ValueError(f"a new label does not fit the labels' type {a.dtype}")
        out[ix] = new
        changed[ix] = True
    return out, changed


def write_edits(
    path: Path, labels_path: Path, edits: Edits, labels: npt.ArrayLike | None = None
) -> Path:
    """Write the edits table at path (edits_path) and its sidecar, after
    checking the edits against the segmentation at labels_path (labels: its
    voxels, if already read). Returns the sidecar's path."""
    if labels is None:
        img = nib.load(labels_path)
        if not isinstance(img, nib.nifti1.Nifti1Image):
            raise ValueError(f"{labels_path.name}: not a NIfTI-1 image")
        labels = np.asarray(img.dataobj)
    edits.check(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for (i, j, k), o, n, b in zip(
            edits.voxels.tolist(),
            edits.old.tolist(),
            edits.new.tolist(),
            edits.answer.tolist(),
            strict=True,
        ):
            w.writerow((i, j, k, o, n, b))
    side = sidecar_path(path)
    write_sidecar(
        side,
        EDITS_SPEC,
        dict(
            SpecVersion=EDITS_SPEC.version,
            Description="Manual label edits of the input segmentation, applied "
            "by dseg before the WM topology correction.",
            **sources(labels_path),
            Provenance=provenance("baillarger.wm_edits.write_edits"),
            Answers=[a.to_json() for a in edits.answers],
        ),
    )
    return side


def read_edits(path: Path) -> tuple[Edits, dict[str, Any]]:
    """The edits in the table at path and its sidecar, checked against
    wm-edits/1 (not against a segmentation: Edits.check, or apply_edits)."""
    side = read_sidecar(sidecar_path(path), EDITS_SPEC)
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if not rows or tuple(rows[0]) != COLUMNS:
        raise ValueError(f"{path.name}: the columns are not {COLUMNS}")
    try:
        t = np.array(rows[1:], np.int64).reshape(-1, len(COLUMNS))
    except ValueError as e:
        raise ValueError(f"{path.name}: a value is not an integer") from e
    answers = tuple(Answer.from_json(d) for d in side["Answers"])
    edits = Edits(t[:, :3], t[:, 3], t[:, 4], t[:, 5], answers)
    return edits, side


def sources_match(side: dict[str, Any], labels_path: Path) -> bool:
    """Whether the edits (their sidecar) are against the segmentation at
    labels_path: the same SHA-256."""
    return list(side["SourcesSHA256"].values()) == [file_sha256(labels_path)]
