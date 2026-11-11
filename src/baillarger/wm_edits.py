"""The answer log of the WM review (docs/formats/wm-edits.md).

The log is append-only. It holds the answers in the order that they were
made, and the records of each review session: when it started, its random
share and seed, and every site shown or skipped. It holds raw facts only, and
no measure.

The log is the review queue's own record. The dseg command does not read it:
the labels change by a write into the label edits (baillarger.label_edits,
one diff), and the log records the answer. There are two kinds of answer:

- a site answer (a choice at a site of the queue): its voxels and labels;
- a scribble (strokes, each with one class).

A correction is a new answer. Nothing in the log is changed or removed.
Log.without gives a log without one answer, in memory only, for inspection.

Three files: the log (.jsonl, one JSON object per line), the voxels (.bin,
VOXEL_DTYPE records) and the sidecar (.json, written one time). EditLog
appends to them; read_log reads them.

read_log reads and checks the whole log. EditLog does that one time, when it
is made, and keeps the records in memory. Each later call reads and checks
only the lines that another writer appended after that, and their voxels;
EditLog.check_log reads and checks the whole log again.
"""

import datetime
import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.fileops import flock
from baillarger.labels import LUT_NAMES

type Int = npt.NDArray[np.int64]
type Site = tuple[int, int, int]

#: docs/formats/wm-edits.md
EDITS_SPEC = SidecarSpec(
    "wm-edits/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("Shape", "array of integers", True, "the grid's shape"),
        Key("Affine", "array of numbers", True, "voxel to mm, 4 x 4, by rows"),
    ),
)

#: one voxel of an answer in the .bin file: its linear index (C order of the
#: grid), its label (the new label, or the stroke's class) and its stroke (-1
#: for a site answer)
VOXEL_DTYPE = np.dtype([("index", "<i8"), ("label", "<i4"), ("stroke", "<i4")])

#: the choices of a site answer (scribbles have the choice "scribble")
SITE_CHOICES = (
    "remove-strand",
    "thicken",
    "fill-gap",
    "widen-gap",
    "leave",
    "manual-paint",
    "accept-fix",
)
#: how a site was chosen
HOW = ("ranked", "random", "none")


def edits_paths(
    root: Path, ents: dict[str, str], desc: str = "wmedits"
) -> dict[str, Path]:
    """The three files of the log of an input with entities ents, in the
    derivatives root: log (.jsonl), voxels (.bin) and sidecar (.json). desc
    "wmeditstest" names a test log."""
    log = derivative_path(root, ents, "edits", ".jsonl", desc=desc)
    return dict(
        log=log, voxels=log.with_suffix(".bin"), sidecar=log.with_suffix(".json")
    )


def _paths(log: Path) -> dict[str, Path]:
    if log.suffix != ".jsonl":
        raise ValueError(f"{log.name}: the log is a .jsonl file")
    return dict(log=log, voxels=log.with_suffix(".bin"), sidecar=sidecar_path(log))


def now() -> str:
    """The time now, ISO 8601 with the time zone, to the second."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _site(x: Any) -> Site | None:
    if x is None:
        return None
    t = tuple(int(c) for c in x)
    if len(t) != 3:
        raise ValueError(f"a site has 3 coordinates, not {len(t)}")
    return (t[0], t[1], t[2])


# ----------------------------------------------------------------- the reader


@dataclass(frozen=True)
class Log:
    """A log, read: its sidecar, its records in order (dicts, as in the
    file) and its voxels (VOXEL_DTYPE: the whole .bin file from read_log;
    from EditLog, the part of it that the records name, mapped and not
    read)."""

    side: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    data: npt.NDArray[np.void]

    @property
    def shape(self) -> tuple[int, int, int]:
        s = self.side["Shape"]
        return (int(s[0]), int(s[1]), int(s[2]))

    def answers(self) -> list[dict[str, Any]]:
        """The answer records (site answers and scribbles), in log order."""
        return [r for r in self.records if r["Record"] == "answer"]

    def answer(self, answer_id: int) -> dict[str, Any]:
        """The answer record with this Id. Raises KeyError if there is none."""
        for r in self.answers():
            if r["Id"] == answer_id:
                return r
        raise KeyError(f"no answer {answer_id}")

    def rows(self, answer_id: int) -> npt.NDArray[np.void]:
        """The VOXEL_DTYPE rows of an answer."""
        v = self.answer(answer_id)["Voxels"]
        out: npt.NDArray[np.void] = self.data[v["Offset"] : v["Offset"] + v["Count"]]
        return out

    def voxels(self, answer_id: int) -> tuple[Int, Int, Int]:
        """An answer's voxels (N, 3) on the grid, their labels and their
        strokes (-1 for a site answer)."""
        r = self.rows(answer_id)
        ijk = np.stack(np.unravel_index(r["index"].astype(np.int64), self.shape), 1)
        return (
            ijk.astype(np.int64),
            r["label"].astype(np.int64),
            r["stroke"].astype(np.int64),
        )

    def sessions(self) -> list[dict[str, Any]]:
        """The session records, in log order."""
        return [r for r in self.records if r["Record"] == "session"]

    def shown(self, session: int) -> list[dict[str, Any]]:
        """The shown records of a session, in log order."""
        return [
            r
            for r in self.records
            if r["Record"] == "shown" and r["Session"] == session
        ]

    def answered(self, session: int) -> set[Site]:
        """The sites that an answer of this session names."""
        out = set()
        for r in self.answers():
            s = _site(r["Site"])
            if r["Session"] == session and s is not None:
                out.add(s)
        return out

    def without(self, answer_id: int) -> "Log":
        """This log without one answer, for inspection. The file does not
        change. Raises KeyError if there is no such answer."""
        self.answer(answer_id)
        recs = tuple(
            r
            for r in self.records
            if not (r["Record"] == "answer" and r["Id"] == answer_id)
        )
        return replace(self, records=recs)


def _check_record(r: dict[str, Any], seen: dict[str, Any]) -> None:
    """Raises ValueError unless r is a valid record after the records seen
    (sessions: set of ids; shown: {(session, site)}; ids: set of answer ids)."""
    kind = r.get("Record")
    if kind == "session":
        if not (_is_int(r["Session"]) and r["Session"] not in seen["sessions"]):
            raise ValueError(f"session {r.get('Session')!r}: not a new integer")
        if not (_is_num(r["RandomShare"]) and 0 <= r["RandomShare"] <= 1):
            raise ValueError("RandomShare is not a number in [0, 1]")
        if not (r["RandomSeed"] is None or _is_int(r["RandomSeed"])):
            raise ValueError("RandomSeed is not an integer or null")
        seen["sessions"].add(r["Session"])
        return
    if kind not in ("shown", "skipped", "answer"):
        raise ValueError(f"unknown record {kind!r}")
    if r["Session"] not in seen["sessions"]:
        raise ValueError(f"{kind}: session {r['Session']!r} has not started")
    site = _site(r["Site"])
    if kind == "shown":
        if r["How"] not in HOW:
            raise ValueError(f"How {r['How']!r} is not one of {HOW}")
        assert site is not None
        seen["shown"][(r["Session"], site)] = r
        return
    if kind == "skipped":
        if (r["Session"], site) not in seen["shown"]:
            raise ValueError(f"skipped: site {site} was not shown in the session")
        return
    if not (_is_int(r["Id"]) and r["Id"] not in seen["ids"]):
        raise ValueError(f"answer {r.get('Id')!r}: not a new integer Id")
    if r["Kind"] == "site":
        if r["Choice"] not in SITE_CHOICES:
            raise ValueError(f"answer {r['Id']}: unknown choice {r['Choice']!r}")
        if site is None:
            raise ValueError(f"answer {r['Id']}: a site answer has a site")
        if r["Choice"] == "leave" and r["Voxels"]["Count"]:
            raise ValueError(f"answer {r['Id']}: leave has no voxels")
    elif r["Kind"] == "scribble":
        if r["Choice"] != "scribble":
            raise ValueError(f"answer {r['Id']}: a scribble has choice scribble")
    else:
        raise ValueError(f"answer {r['Id']}: unknown kind {r['Kind']!r}")
    if site is not None and (r["Session"], site) not in seen["shown"]:
        raise ValueError(f"answer {r['Id']}: site {site} was not shown in the session")
    bad = [s for s in r["Supersedes"] if s not in seen["ids"]]
    if bad:
        raise ValueError(f"answer {r['Id']}: supersedes unknown answers {bad}")
    seen["ids"].add(r["Id"])


def _new_seen() -> dict[str, Any]:
    return dict(sessions=set(), shown={}, ids=set())


@dataclass
class _State:
    """A log, read and checked up to a point: its records, the records seen
    (_check_record), the bytes of the .jsonl file in complete lines read
    (log_end), the VOXEL_DTYPE records of the .bin file that a record names
    (bin_end), and the device and inode of the .jsonl file."""

    side: dict[str, Any]
    ident: tuple[int, int]
    records: list[dict[str, Any]] = field(default_factory=list)
    seen: dict[str, Any] = field(default_factory=_new_seen)
    log_end: int = 0
    bin_end: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        s = self.side["Shape"]
        return (int(s[0]), int(s[1]), int(s[2]))

    def feed(
        self,
        name: str,
        text: bytes,
        rows: Callable[[int, int], npt.NDArray[np.void]],
    ) -> None:
        """Check and add the complete lines of text: the bytes of the .jsonl
        file after log_end. rows(offset, count) gives the voxels of an
        answer, and raises ValueError if they are not in the .bin file. The
        part after the last line end (empty, or an append that did not
        finish) is not read."""
        size = int(np.prod(self.shape))
        for line in text[: text.rfind(b"\n") + 1].split(b"\n")[:-1]:
            try:
                r = json.loads(line)
                _check_record(r, self.seen)
                if r["Record"] == "answer":
                    v = r["Voxels"]
                    x = rows(v["Offset"], v["Count"])
                    if hashlib.sha256(x.tobytes()).hexdigest() != v["SHA256"]:
                        raise ValueError("the SHA-256 of its voxels does not agree")
                    if len(x) and (x["index"].min() < 0 or x["index"].max() >= size):
                        raise ValueError("a voxel is outside the grid")
                    self.bin_end = max(self.bin_end, v["Offset"] + v["Count"])
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"{name}, line {len(self.records) + 1}: {e}") from e
            self.records.append(r)
            self.log_end += len(line) + 1


def _ident(fh: Any) -> tuple[int, int]:
    st = os.fstat(fh.fileno())
    return (st.st_dev, st.st_ino)


def _read_all(path: Path) -> tuple[_State, npt.NDArray[np.void]]:
    """The whole log at path, read and checked (read_log), and the whole
    .bin file."""
    p = _paths(path)
    side = read_sidecar(p["sidecar"], EDITS_SPEC)
    # the lines first: the voxels of a complete line are in the .bin file
    # before the line is written
    with p["log"].open("rb") as fh:
        st = _State(side, _ident(fh))
        text = fh.read()
    raw = p["voxels"].read_bytes()
    n = len(raw) // VOXEL_DTYPE.itemsize
    data = np.frombuffer(raw[: n * VOXEL_DTYPE.itemsize], VOXEL_DTYPE)

    def rows(offset: int, count: int) -> npt.NDArray[np.void]:
        if offset < 0 or offset + count > n:
            raise ValueError("its voxels are not in the .bin file")
        out: npt.NDArray[np.void] = data[offset : offset + count]
        return out

    st.feed(path.name, text, rows)
    return st, data


def read_log(path: Path) -> Log:
    """The log at path (.jsonl), with its sidecar and voxels, checked against
    wm-edits/1: each record valid after the ones before it, and each answer's
    voxels inside the .bin file with their SHA-256. A last line with no line
    end (an append that did not finish) is ignored, and so are the bytes of
    the .bin file that no record names."""
    st, data = _read_all(path)
    return Log(st.side, tuple(st.records), data)


# ----------------------------------------------------------------- the writer


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    with flock(path, exclusive=True):
        yield


class EditLog:
    """The writer of a log. Each call appends; nothing is changed or removed.
    Each append holds an exclusive lock (fcntl.flock) on the log, appends the
    voxels to the .bin file first, then the record to the .jsonl file, and
    flushes both to the disk.

    The log is read and checked in full one time, when the EditLog is made
    (read_log). The records stay in memory, and a new record is checked
    against them. Each call reads only the lines that another writer
    appended since, and their voxels, and checks them. check_log reads and
    checks the whole log again."""

    def __init__(self, path: Path) -> None:
        self.paths = _paths(path)
        self.path = self.paths["log"]
        self._mutex = threading.Lock()  # the threads of this process
        self._state: _State | None = None
        with self._held():
            st = self._current()
        self.shape = st.shape

    @contextmanager
    def _held(self) -> Iterator[None]:
        with self._mutex, _locked(self.path):
            yield

    def _current(self) -> _State:
        """The memory copy, with the lines that another writer appended
        since. A .jsonl file that is replaced or shorter is read again in
        full. Call with the locks held."""
        st = self._state
        self._state = None  # until it is current: an error leaves no copy
        if st is not None:
            with self.path.open("rb") as fh:
                if _ident(fh) == st.ident and os.fstat(fh.fileno()).st_size >= (
                    st.log_end
                ):
                    fh.seek(st.log_end)
                    text = fh.read()
                    if b"\n" in text:
                        with self.paths["voxels"].open("rb") as vh:

                            def rows(offset: int, count: int) -> npt.NDArray[np.void]:
                                if offset < 0:
                                    raise ValueError(
                                        "its voxels are not in the .bin file"
                                    )
                                vh.seek(offset * VOXEL_DTYPE.itemsize)
                                b = vh.read(count * VOXEL_DTYPE.itemsize)
                                if len(b) != count * VOXEL_DTYPE.itemsize:
                                    raise ValueError(
                                        "its voxels are not in the .bin file"
                                    )
                                return np.frombuffer(b, VOXEL_DTYPE)

                            st.feed(self.path.name, text, rows)
                    self._state = st
                    return st
        st = _read_all(self.path)[0]
        self._state = st
        return st

    def _log(self, st: _State) -> Log:
        data: npt.NDArray[np.void] = np.zeros(0, VOXEL_DTYPE)
        if st.bin_end:
            data = np.memmap(self.paths["voxels"], VOXEL_DTYPE, "r", shape=st.bin_end)
        return Log(st.side, tuple(st.records), data)

    @staticmethod
    def create(path: Path, dseg: Path, hemi: str) -> "EditLog":
        """Make a new, empty log at path (.jsonl) for the grid of the
        segmentation at dseg (its shape and affine), and its sidecar. Raises
        FileExistsError if a log is there."""
        p = _paths(path)
        if any(q.exists() for q in p.values()):
            raise FileExistsError(f"{path.name}: a log is there already")
        img = nib.load(dseg)
        if not isinstance(img, nib.nifti1.Nifti1Image):
            raise ValueError(f"{dseg.name}: not a NIfTI-1 image")
        write_sidecar(
            p["sidecar"],
            EDITS_SPEC,
            dict(
                SpecVersion=EDITS_SPEC.version,
                Description="The answer log of the WM review: answers and "
                "session records, append-only.",
                **sources(dseg),
                Provenance=provenance("baillarger.wm_edits.EditLog.create"),
                Hemisphere=hemi,
                Shape=[int(s) for s in img.shape[:3]],
                Affine=[float(x) for x in np.asarray(img.affine, float).ravel()],
            ),
        )
        p["log"].touch()
        p["voxels"].touch()
        return EditLog(path)

    def read(self) -> Log:
        """The log as it is now. Its voxels are mapped, not read."""
        with self._held():
            return self._log(self._current())

    def check_log(self) -> Log:
        """The log as it is now, read and checked in full again (read_log).
        It becomes the memory copy. Raises ValueError as read_log does."""
        with self._held():
            self._state = None
            st, data = _read_all(self.path)
            self._state = st
            return Log(st.side, tuple(st.records), data)

    def sessions(self) -> list[dict[str, Any]]:
        """The session records."""
        return self.read().sessions()

    def answered(self, session: int) -> set[Site]:
        """The sites that an answer of this session names."""
        return self.read().answered(session)

    def _append(
        self, rec: dict[str, Any], rows: npt.NDArray[np.void] | None = None
    ) -> dict[str, Any]:
        with self._held():
            st = self._current()
            seen = st.seen
            if rec["Record"] == "session" and rec["Session"] is None:
                rec["Session"] = 1 + max(seen["sessions"], default=0)
            if rec["Record"] == "answer":
                rec["Id"] = 1 + max(seen["ids"], default=0)
                site = _site(rec["Site"])
                if site is not None:
                    hit = seen["shown"].get((rec["Session"], site))
                    if hit is None:
                        raise ValueError(
                            f"site {site} was not shown in session {rec['Session']}"
                        )
                    for k in ("SiteKind", "How", "Rank", "Score"):
                        rec[k] = hit[k]
                else:
                    rec.update(SiteKind=None, How="none", Rank=None, Score=None)
                assert rows is not None
                # the end of the voxels that a record names: bytes after it are
                # from an append that did not finish
                rec["Voxels"] = dict(
                    Offset=int(st.bin_end),
                    Count=len(rows),
                    SHA256=hashlib.sha256(rows.tobytes()).hexdigest(),
                )
            line = (json.dumps(rec, separators=(",", ":")) + "\n").encode()
            parsed = json.loads(line)  # the record as a reader sees it
            # changes seen only if the record is valid; after that, a failed
            # write leaves no memory copy, and the next call reads the log
            _check_record(parsed, seen)
            self._state = None
            if rows is not None:
                with self.paths["voxels"].open("r+b") as fh:
                    # bytes that no record names (an append that did not
                    # finish) are written over
                    fh.seek(rec["Voxels"]["Offset"] * VOXEL_DTYPE.itemsize)
                    fh.write(rows.tobytes())
                    fh.truncate()
                    fh.flush()
                    os.fsync(fh.fileno())
            with self.path.open("r+b") as fh:
                # bytes after the last complete line (an append that did not
                # finish) are written over
                fh.seek(st.log_end)
                fh.truncate()
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            st.records.append(parsed)
            st.log_end += len(line)
            if rows is not None:
                st.bin_end = rec["Voxels"]["Offset"] + len(rows)
            self._state = st
        return rec

    def start_session(
        self,
        random_share: float,
        random_seed: int | None,
        dseg_sha256: str,
        queue_sha256: str,
        time: str | None = None,
    ) -> int:
        """Start a review session: the share of random sites that the reviewer
        chose (0 to 1), their seed (None if the share is 0), the SHA-256 of the
        dseg and of the queue file shown. Returns the session's id (1, 2,
        ...)."""
        r = self._append(
            dict(
                Record="session",
                Session=None,
                Time=time or now(),
                RandomShare=random_share,
                RandomSeed=random_seed,
                DsegSHA256=dseg_sha256,
                QueueSHA256=queue_sha256,
            )
        )
        return int(r["Session"])

    def shown(
        self,
        session: int,
        site: Sequence[int],
        site_kind: str,
        rank: int | None,
        score: float | None = None,
        time: str | None = None,
    ) -> None:
        """Record that a site was shown in a session: its kind (wm-sites/1,
        for example misfacing or random), its rank in the queue (None for a
        random site) and its score (None if the queue has none). A random site
        has How "random"; any other site has How "ranked"."""
        self._append(
            dict(
                Record="shown",
                Session=session,
                Time=time or now(),
                Site=list(_site(site) or ()),
                SiteKind=site_kind,
                How="random" if site_kind == "random" else "ranked",
                Rank=rank,
                Score=score,
            )
        )

    def skipped(
        self, session: int, site: Sequence[int], time: str | None = None
    ) -> None:
        """Record that the reviewer skipped a site shown in the session."""
        self._append(
            dict(
                Record="skipped",
                Session=session,
                Time=time or now(),
                Site=list(_site(site) or ()),
            )
        )

    def _rows(
        self, voxels: npt.ArrayLike, labels: npt.ArrayLike, strokes: npt.ArrayLike
    ) -> npt.NDArray[np.void]:
        shape = self.shape
        v = np.asarray(voxels, np.int64).reshape(-1, 3)
        lab = np.asarray(labels, np.int64).ravel()
        st = np.asarray(strokes, np.int64).ravel()
        if not len(v) == len(lab) == len(st):
            raise ValueError("voxels, labels and strokes differ in length")
        if len(v) and not np.all((v >= 0) & (v < shape)):
            raise ValueError("a voxel is outside the grid")
        bad = sorted(set(lab.tolist()) - set(LUT_NAMES))
        if bad:
            raise ValueError(f"labels {bad} are not in the label set")
        index: Int = np.zeros(0, np.int64)
        if len(v):
            index = np.asarray(np.ravel_multi_index(tuple(v.T), shape), np.int64)
        if len(np.unique(index)) != len(index):
            raise ValueError("a voxel appears twice in one answer")
        rows = np.zeros(len(v), VOXEL_DTYPE)
        rows["index"], rows["label"], rows["stroke"] = index, lab, st
        return rows

    def answer(
        self,
        session: int,
        site: Sequence[int],
        choice: str,
        voxels: npt.ArrayLike,
        labels: npt.ArrayLike,
        decided_against: str | None,
        reviewer_seconds: float,
        params: dict[str, Any] | None = None,
        supersedes: Sequence[int] = (),
        time: str | None = None,
    ) -> int:
        """Append a site answer: at a site shown in the session (ValueError
        if it was not), the choice (SITE_CHOICES), its voxels (N, 3) on the
        grid and their new labels, the SHA-256 of the dseg looked at, the
        reviewer time (seconds, from opening the site to submitting the
        answer), the choice's parameters and the Ids of the answers it
        supersedes. How the site was chosen comes from its shown record.
        Returns the answer's Id."""
        rows = self._rows(voxels, labels, np.full(len(np.asarray(labels).ravel()), -1))
        r = self._append(
            dict(
                Record="answer",
                Id=None,
                Session=session,
                Kind="site",
                Site=list(_site(site) or ()),
                Choice=choice,
                Params=dict(params or {}),
                Supersedes=[int(s) for s in supersedes],
                DecidedAgainst=decided_against,
                Time=time or now(),
                ReviewerSeconds=float(reviewer_seconds),
                Strokes=None,
            ),
            rows,
        )
        return int(r["Id"])

    def scribble(
        self,
        session: int,
        strokes: Sequence[tuple[int, npt.ArrayLike]],
        decided_against: str | None,
        reviewer_seconds: float,
        site: Sequence[int] | None = None,
        supersedes: Sequence[int] = (),
        time: str | None = None,
    ) -> int:
        """Append a scribble: strokes, each (class, voxels (N, 3) on the
        grid), made in the session, at a site shown in it or at no site
        (None). A voxel is in at most one stroke. Returns the answer's Id."""
        vox = [np.asarray(v, np.int64).reshape(-1, 3) for _, v in strokes]
        cls = [int(c) for c, _ in strokes]
        rows = self._rows(
            np.concatenate(vox) if vox else np.zeros((0, 3), np.int64),
            np.concatenate([np.full(len(v), c) for v, c in zip(vox, cls, strict=True)])
            if vox
            else np.zeros(0, np.int64),
            np.concatenate([np.full(len(v), k) for k, v in enumerate(vox)])
            if vox
            else np.zeros(0, np.int64),
        )
        r = self._append(
            dict(
                Record="answer",
                Id=None,
                Session=session,
                Kind="scribble",
                Site=None if site is None else list(_site(site) or ()),
                Choice="scribble",
                Params={},
                Supersedes=[int(s) for s in supersedes],
                DecidedAgainst=decided_against,
                Time=time or now(),
                ReviewerSeconds=float(reviewer_seconds),
                Strokes=[dict(Stroke=k, Class=c) for k, c in enumerate(cls)],
            ),
            rows,
        )
        return int(r["Id"])
