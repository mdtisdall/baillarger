"""The stream 1 review pane's server side (phase 4.2): the queue of sites
(wm-sites/1), review sessions with random sites, the site answers with their
previews, and scribbles, written to the answer log (wm-edits/1); and the
paint mode with its undo. A site answer and a paint change the labels: they
write into the label edits (label-edits/1, baillarger.label_edits), the one
diff that the dseg command reads (white_dseg --edits). The pane holds no
stream 1 logic: it calls phase 1.3's code (baillarger.wm_sites,
baillarger.wm_edits, baillarger.wm_choices, baillarger.fix_diff) and
baillarger.label_edits, and writes nothing else.

Sites. A site is its voxel (i, j, k) on the segmentation's grid; the page names
it "i,j,k". The ranked sites are the rows of the queue file, in its order (rank
1, 2, ...).

Sessions. A session starts with a random share s (0 to 1, given by the
reviewer; there is no default) and a seed. Its sites are the N ranked sites and
n random sites (wm_sites.read_random_sites with the seed), with
n = round(N s / (1 - s)), so that random sites are the share s of the session's
sites (s = 1: N random sites and no ranked one). The random sites take
positions among the ranked ones drawn with the same seed
(numpy.random.default_rng(seed).permutation); the ranked sites keep their order.
After a restart, the last session of the log is resumed: its sites are drawn
again from its share and seed, and the log gives the state of each.

State of a site in the session: answered (an answer of the session names it),
skipped (a skip record and no answer), shown (a shown record only) or
not shown. A random site's kind is "random"; the page gets kind null for it
until it is answered, so that its kind does not change how the reviewer looks
(a coordinator decision for review). Its shown record has its real kind.

Routes (Review.get, Review.post), JSON:
- GET /review/queue: the session's sites (or the ranked sites before a
  session), each {id, kind, rank, score, state}; the session; the label
  classes of a scribble; the grid's affine; the log's path.
- POST /review/session {random_share, seed} or {resume: true}.
- GET /review/next[?site=i,j,k]: the next site not shown (or the site named);
  the log records it as shown. {site: {id, kind, rank, voxel, mm}} or
  {site: null} when the session has no site left.
- POST /review/skip {site}.
- POST /review/preview {site, choice, params}: the answer's voxels, the
  correction's voxels and whether the correction overrides the answer
  (wm_choices.preview), in grid voxels; a preview that takes more than
  WAIT_S runs on as a job: {status: "running", job}, then GET
  /review/preview?job=... until {status: "done", ...} (or {status:
  "failed", error}). The previews of the choices are jobs of
  baillarger.viewer.jobs.Jobs, keyed by the site, the choice, the params and
  the SHA-256 of the edits, and the last PREVIEWS_KEPT finished previews stay
  in memory for polling. One runs at a time, in one worker process (the
  "spawn" context): the request reads the box round the site and sends only
  the box and the choice's arguments. The newest request wins: a request
  for another key while a preview runs stops the worker (terminate, join),
  its job fails with "replaced by a newer preview", and the new preview
  starts in a new worker. A request for the key that runs gets {status:
  "running", job}. A worker that ends without a result (the memory cap
  killed it) fails its job. The worker is a daemon process: it stops when
  the viewer ends (and at Review.close). The previews of manual-paint and
  accept-fix are computed in the request.
- POST /review/answer {site, choice, params, reviewer_seconds}: the choice's
  voxels and labels at the site, written into the label edits
  (label_edits.paint: only the voxels whose label changes from the dseg's
  label with the edits applied), then appended to the log. If the log refuses
  the answer, the edits are written back as they were. {answer, kind, how, rank, voxels,
  edited}: edited is the number of voxels written into the edits.
- POST /review/scribble {strokes: [{class, voxels}], reviewer_seconds}:
  strokes appended to the log as one scribble. A voxel in two strokes stays in
  the later one only (the log takes a voxel in at most one stroke).
- POST /review/paint {strokes: [{label, voxels}]}: the strokes written into
  the label edits (Paint, below). {voxels, undo_depth}: voxels is the number
  of voxels whose label changed.
- POST /review/undo: the last write into the label edits of this viewer run
  undone (Undo, below). {voxels, undo_depth}, or {error} if there is nothing
  to undo.

Outputs replaced. At start, the review records the identity (os.stat:
device, inode, mtime) of the dseg, the queue and the dseg's fix diff. If one
of them changes while the viewer runs (regenerate replaced the outputs), a
paint, an answer, an accept-fix and an undo are refused with an error that
says to restart the viewer, and nothing is written: the review's labels, its
queue and the fix ids of its rows are those of the old outputs. The check is
made under the edits' file lock, before each write into the edits.

Each write holds the review's lock and is flushed to the disk before the
response (EditLog appends and fsyncs; label_edits.update replaces the files of
the edits). A write into the edits is one label_edits.update: it holds the
edits' file lock across the read, the change and the write, so that a
second viewer process on the same edits does not lose this one's voxels.
An answer's log append is the step after its write, under the same file
lock: if the log refuses the answer, the edits are written back before
another writer can write.
The review keeps the edits that it last read or wrote in memory
(label_edits.Current), so that a write reads only the sidecar while no other
writer changed the edits. A scribble is a record of the log only: it does not change the
edits. A choice and its preview read a box
round the site (Choices.half), so no response scales with the volume.

The input labels are the dseg's source: the first of its sidecar's Sources
(the excised dseg). An answer reads only the box of its voxels from them.

accept-fix (ACCEPT) is a choice of an override site only (kind override): its
voxels and labels are those of the site's fix (the row's fix id) in the dseg's
fix diff (desc-wmtopofixes_diff.npy, baillarger.fix_diff). Written into the
edits, the fix's labels are edits, so the next dseg has no override there.
Its preview gives the fix's voxels and no correction.

Paint. The strokes of the paint mode, each a label of the menu (menu: outside,
WM and cortex of the hemisphere, and "input", the eraser:
label_edits.INPUT) and its voxels. The strokes become voxels with the same
function as a scribble's (stroke_voxels: a voxel in two strokes keeps the
later one). A label that is not in the menu, or a voxel off the grid, refuses
the whole paint, and nothing is written. Only the voxels whose label changes
are written (label_edits.paint: from the dseg's labels, the labels that the
user sees, with the edits applied). A paint back to a voxel's input label,
against the automatic correction, is an edit.

Undo. Every write into the edits of this viewer run (a paint, and a site
answer of any choice: accept-fix and manual-paint too) pushes the earlier
entries of the voxels that it changes on one undo stack. /review/undo writes
the top entries back (label_edits.restore) and pops them, so an undo always
reverses the last write and never brings back labels under a later one. An
undo is refused if the edits on disk are not the ones that the stack's top
write left (another writer changed them since): it would write the earlier
entries over the other writer's changes. A write that changes no voxel
pushes nothing. The stack is in memory only, for
this viewer run. An undo does not change the log: an undone answer stays in
it.

manual-paint (MANUAL) at a queue site: the page opens the paint mode, and its
submit is /review/answer {site, choice: manual-paint, strokes,
reviewer_seconds}. It writes into the edits as a paint, and the log records
the answer: the voxels of the strokes and the label each takes (the eraser's
voxels: their input label). Its preview has no voxels.

Paint only. With no queue and no log (sites and log None; the viewer's
--edits alone), the review has the paint routes only; the hemisphere comes
from the dseg's sidecar (wm-dseg/1).
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from baillarger import blocks, label_edits, white_dseg
from baillarger.bids import file_sha256
from baillarger.labels import CORTEX_LABEL, LUT_NAMES, WM_LABEL
from baillarger.viewer.jobs import Jobs
from baillarger.wm_edits import EditLog, read_log
from baillarger.wm_sites import read_random_sites, read_sites

type Arr = npt.NDArray[Any]
type Site = tuple[int, int, int]

#: the choice whose answer is the strokes of the paint mode
MANUAL = "manual-paint"
#: the eraser of the paint menu: the voxel takes its input label again
ERASER = "input"
#: the choice of an override site that writes its fix's labels into the edits
ACCEPT = "accept-fix"
#: seconds a preview request waits before it answers as a job
WAIT_S = 1.5
#: the finished previews that the review keeps for polling (jobs.Jobs)
PREVIEWS_KEPT = 8

#: a file's identity (device, inode, mtime in ns), or None if it is not there
type Identity = tuple[int, int, int] | None


def identity(path: Path) -> Identity:
    """The identity of the file at path (os.stat), or None if there is none."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino, st.st_mtime_ns)


def site_id(s: Sequence[int]) -> str:
    return f"{int(s[0])},{int(s[1])},{int(s[2])}"


def parse_site(x: Any) -> Site:
    v = [int(c) for c in (x.split(",") if isinstance(x, str) else x)]
    if len(v) != 3:
        raise ValueError(f"a site is 3 integers, not {x!r}")
    return (v[0], v[1], v[2])


def random_count(n_ranked: int, share: float) -> int:
    """The number of random sites for a share of the session (module
    docstring)."""
    if not 0 <= share <= 1:
        raise ValueError(f"the random share must be from 0 to 1, not {share}")
    if share == 1:
        return n_ranked
    return int(round(n_ranked * share / (1 - share)))


def session_order(
    ranked: list[dict[str, Any]], random: list[dict[str, Any]], seed: int | None
) -> list[dict[str, Any]]:
    """The ranked rows in their order with the random rows at positions drawn
    with the seed."""
    n = len(ranked) + len(random)
    if not random:
        return list(ranked)
    pos = set(np.random.default_rng(seed).permutation(n)[: len(random)].tolist())
    rr, kk = iter(random), iter(ranked)
    return [next(rr) if i in pos else next(kk) for i in range(n)]


def classes(hemi: str) -> list[dict[str, Any]]:
    """The classes of a scribble: the labels of the set for the hemisphere
    (label-set/1), WM and cortex first (keys "wm", "cortex"), then outside."""
    wm, cx = WM_LABEL[hemi], CORTEX_LABEL[hemi]
    other = "Left-" if hemi == "R" else "Right-"
    rest = sorted(
        v
        for v, n in LUT_NAMES.items()
        if v not in (wm, cx, 0) and not n.startswith(other)
    )
    out = [dict(value=wm, name=LUT_NAMES[wm], key="wm")]
    out.append(dict(value=cx, name=LUT_NAMES[cx], key="cortex"))
    out += [dict(value=v, name=LUT_NAMES[v], key="") for v in rest]
    out.append(dict(value=0, name=LUT_NAMES[0], key="outside"))
    return out


def menu(hemi: str) -> list[dict[str, Any]]:
    """The labels of the paint mode: outside, WM and cortex of the hemisphere
    (label-set/1), and the eraser (value "input")."""
    wm, cx = WM_LABEL[hemi], CORTEX_LABEL[hemi]
    return [
        dict(value=0, name="outside", key="outside"),
        dict(value=wm, name=LUT_NAMES[wm], key="wm"),
        dict(value=cx, name=LUT_NAMES[cx], key="cortex"),
        dict(value=ERASER, name="input label (eraser)", key="input"),
    ]


def stroke_voxels(
    strokes: Sequence[dict[str, Any]], key: str
) -> list[tuple[Any, npt.NDArray[np.int64]]]:
    """Strokes, each {key: its label, voxels: [[i, j, k], ...]}, as (label,
    voxels (N, 3)) in stroke order. A voxel in two strokes stays in the later
    one only; a stroke left with no voxel is dropped. The labels are not
    checked here."""
    owner: dict[Site, int] = {}
    for k, st in enumerate(strokes):
        for v in st["voxels"]:
            owner[parse_site(v)] = k
    by: dict[int, list[Site]] = {}
    for v, k in owner.items():
        by.setdefault(k, []).append(v)
    return [
        (strokes[k][key], np.asarray(by[k], np.int64).reshape(-1, 3))
        for k in sorted(by)
    ]


# ------------------------------------------------------------------ the choices


@dataclass(frozen=True)
class Answer:
    """A choice's voxels on the grid and their new labels."""

    voxels: npt.NDArray[np.int64]
    labels: npt.NDArray[np.int64]
    params: dict[str, Any]


@dataclass(frozen=True)
class Preview:
    """A choice's preview on the grid: the voxels the answer changes, the voxels
    the correction then changes, whether the correction overrides the answer,
    and whether a face of the preview's box has WM on it (then the preview is
    approximate)."""

    answer: npt.NDArray[np.int64]
    correction: npt.NDArray[np.int64]
    overrides: bool
    cut_by_box: bool = False


class Choices(Protocol):
    """Phase 1.3's choices at a site: labels is a box of the grid round the
    site (lo: its first voxel on the grid), half(params) voxels each way."""

    def half(self, params: dict[str, Any]) -> int: ...

    def answer(
        self, labels: Arr, lo: Site, site: Site, choice: str, params: dict[str, Any]
    ) -> Answer: ...

    def preview(
        self, labels: Arr, lo: Site, site: Site, choice: str, params: dict[str, Any]
    ) -> Preview: ...


class WmChoices:
    """Choices through baillarger.wm_choices (phase 1.3): run for the answer,
    preview for the preview. Both get a box of the grid, not the whole grid
    (wm_choices.preview copies the labels it is given). The box is one voxel
    larger each way than the box that preview reads (support(radius, reach)
    and the halo), so preview's cut_by_box sees the faces of its own box."""

    def __init__(
        self, hemi: str, voxel_mm: npt.ArrayLike, halo_mm: float = 1.0
    ) -> None:
        self.hemi = hemi
        self.voxel_mm = np.asarray(voxel_mm, np.float64)
        self.halo_mm = float(halo_mm)

    def half(self, params: dict[str, Any]) -> int:
        from baillarger import wm_choices as wc

        halo = int(np.ceil(self.halo_mm / self.voxel_mm - 1e-9).max())
        return wc.support(int(params.get("radius", 1)), int(params.get("reach", 4))) + (
            halo + 1
        )

    def answer(
        self, labels: Arr, lo: Site, site: Site, choice: str, params: dict[str, Any]
    ) -> Answer:
        from baillarger import wm_choices as wc

        local = _local(site, lo)
        ch = wc.run(choice, labels, local, self.hemi, **params)
        v = np.asarray(ch.voxels, np.int64).reshape(-1, 3) + np.asarray(lo)
        return Answer(v, np.asarray(ch.labels, np.int64).ravel(), dict(ch.params))

    def preview(
        self, labels: Arr, lo: Site, site: Site, choice: str, params: dict[str, Any]
    ) -> Preview:
        from baillarger import wm_choices as wc

        p = wc.preview(
            labels,
            _local(site, lo),
            self.hemi,
            choice,
            voxel_mm=self.voxel_mm,
            halo_mm=self.halo_mm,
            **params,
        )
        off = np.asarray(lo)
        return Preview(
            np.asarray(p.answer.voxels, np.int64).reshape(-1, 3) + off,
            np.asarray(p.correction.voxels, np.int64).reshape(-1, 3) + off,
            bool(p.overrides),
            cut_by_box=bool(p.cut_by_box),
        )


def _local(site: Site, lo: Site) -> Site:
    return (site[0] - lo[0], site[1] - lo[1], site[2] - lo[2])


# ------------------------------------------------------------------ the worker


def _preview_worker(conn: Any) -> None:
    """The preview worker process: for each request (choices, box, lo, site,
    choice, params) on conn, ("ok", Preview) or ("error", message). Ends when
    the viewer's end of conn closes."""
    while True:
        try:
            choices, box, lo, site, choice, params = conn.recv()
        except EOFError:
            return
        try:
            out: tuple[str, Any] = (
                "ok",
                choices.preview(box, lo, site, choice, params),
            )
        except Exception as exc:  # reported to the viewer
            out = ("error", f"{type(exc).__name__}: {exc}")
        conn.send(out)


def _preview_dict(p: Preview, t0: float) -> dict[str, Any]:
    """A preview as the page gets it; t0: the time it started."""
    return dict(
        status="done",
        answer=p.answer.tolist(),
        correction=p.correction.tolist(),
        overrides=p.overrides,
        cut_by_box=p.cut_by_box,
        seconds=time.perf_counter() - t0,
    )


class PreviewError(Exception):
    """A preview that the worker did not give: it failed, it was replaced by a
    newer preview, or the worker ended without a result."""


@dataclass
class _Worker:
    proc: Any  # multiprocessing.context.SpawnProcess
    conn: Any  # multiprocessing.connection.Connection
    #: stopped by a newer preview (Review._stop_worker)
    replaced: bool = False


# ------------------------------------------------------------------ the review


class Review:
    """The review on the server: the queue, the answer log at log, the label
    edits at edits (a .npy diff, made at the first answer or paint that
    changes a label), the session, the choices and the undo stack of the
    paints (module docstring). labels() gives the segmentation's voxels (the
    whole grid, read once: the label pyramid's level 0); random(n, seed) the
    random sites (wm_sites.read_random_sites); inputs the input labels
    (default: white_dseg.input_labels(dseg)). With sites and log None, the review is
    paint only (module docstring). Raises ValueError if the edits are on
    another grid or of the other hemisphere, or if only one of sites and log
    is given."""

    def __init__(
        self,
        sites: Path | None,
        log: Path | None,
        edits: Path,
        dseg: Path,
        labels: Callable[[], Arr],
        affine: npt.ArrayLike,
        random: Callable[[int, int], list[dict[str, Any]]] | None,
        choices: Choices | None = None,
        inputs: Path | None = None,
    ) -> None:
        import nibabel as nib

        if (sites is None) != (log is None):
            raise ValueError("a queue needs both its sites and its log")
        self.dseg_sha = file_sha256(dseg)
        #: the outputs that the review reads, and their identities at start
        #: (_check_outputs)
        diff_path = white_dseg.output_paths(Path(dseg))["diff"]
        self._outputs: dict[Path, Identity] = {
            Path(dseg): identity(Path(dseg)),
            diff_path: identity(diff_path),
        }
        if sites is not None:
            self._outputs[Path(sites)] = identity(Path(sites))
        self.rows: list[dict[str, Any]] = []
        self.log_path: Path | None = None
        self._log: EditLog | None = None
        if sites is not None and log is not None:
            self.rows, self.side = read_sites(sites)
            self.hemi = str(self.side["Hemisphere"])
            self.queue_sha = file_sha256(sites)
            self.log_path = Path(log)
            if not self.log_path.exists():
                EditLog.create(self.log_path, dseg, self.hemi)
            self._log = EditLog(self.log_path)
        else:
            self.hemi = str(white_dseg.dseg_sidecar(Path(dseg))["Hemisphere"])
        self.labels, self.random = labels, random
        self.affine = np.asarray(affine, np.float64)
        self.shape = tuple(int(n) for n in nib.nifti1.load(dseg).shape[:3])
        if self._log is not None and tuple(self._log.read().shape) != self.shape:
            raise ValueError(f"{self.log_path}: the log is not on the dseg's grid")
        self.edits = Path(edits)
        #: the edits as this review last read or wrote them (the memory copy
        #: of label_edits.update)
        self._cur: label_edits.Current | None = None
        if label_edits.edits_paths(self.edits)["sidecar"].exists():
            self._read_edits()
        self.inputs = Path(inputs) if inputs is not None else None
        #: the input labels, opened once (_input_labels): the memory-mapped
        #: grid, and its source, kept while the review runs (a temporary
        #: decompressed copy is removed when the source is collected)
        self._inputs_map: Arr | None = None
        self._inputs_src: blocks.NiftiSource | None = None
        self._fix_rows: Arr | None = None
        voxel_mm = np.linalg.norm(self.affine[:3, :3], axis=0)
        self.choices: Choices = choices or WmChoices(self.hemi, voxel_mm)
        self.dseg = Path(dseg)
        self.lock = threading.Lock()  # every write, and the session
        self.session: int | None = None
        self.share: float | None = None
        self.seed: int | None = None
        self.order: list[dict[str, Any]] = []
        #: the previews of the choices (module docstring)
        self.jobs: Jobs[dict[str, Any]] = Jobs(
            log=lambda m: None, keep_in_memory=PREVIEWS_KEPT
        )
        #: the preview worker process, started at the first slow preview
        self._worker: _Worker | None = None
        self._wlock = threading.Lock()  # self._worker
        self._plock = threading.Lock()  # a preview request's stop and start
        #: the number of stops: a job started before a stop does not start
        #: a worker after it
        self._stops = 0
        #: the undo stack: the earlier entries of each write into the edits,
        #: and the SHA-256 of the edits that the write left
        self.undo: list[label_edits.Edits] = []
        self._undo_sha: list[str] = []
        last = self._log.sessions() if self._log is not None else []
        if last:
            self._resume(last[-1])

    @property
    def paint_only(self) -> bool:
        return self._log is None

    @property
    def log(self) -> EditLog:
        """The answer log. Raises ValueError for a review that is paint only."""
        if self._log is None:
            raise ValueError(
                "paint only: the viewer runs with --edits alone (no queue, no log)"
            )
        return self._log

    # -------------------------------------------------------------- sessions
    def _ranked(self) -> list[dict[str, Any]]:
        return [dict(r, how="ranked") for r in self.rows]

    def _draw(self, share: float, seed: int | None) -> list[dict[str, Any]]:
        n = random_count(len(self.rows), share)
        assert self.random is not None
        rnd = (
            [dict(r, how="random", rank=None) for r in self.random(n, seed or 0)]
            if n
            else []
        )
        ranked = [] if share == 1 else self._ranked()
        return session_order(ranked, rnd, seed)

    def _resume(self, rec: dict[str, Any]) -> None:
        self.session = int(rec["Session"])
        self.share, self.seed = float(rec["RandomShare"]), rec["RandomSeed"]
        self.order = self._draw(self.share, self.seed)

    def start(self, share: float, seed: int | None) -> dict[str, Any]:
        """A new session (module docstring)."""
        _ = self.log  # refused when paint only
        share = float(share)
        random_count(len(self.rows), share)  # checks the share
        if share > 0 and seed is None:
            raise ValueError("a seed is needed when the random share is above 0")
        with self.lock:
            order = self._draw(share, seed)
            sid = self.log.start_session(
                share,
                None if share == 0 else int(seed or 0),
                self.dseg_sha,
                self.queue_sha,
            )
            self.session, self.share, self.seed, self.order = sid, share, seed, order
        return self._info()

    def resume(self) -> dict[str, Any]:
        last = self.log.sessions()
        if not last:
            raise ValueError("no session in the log to resume")
        with self.lock:
            self._resume(last[-1])
        return self._info()

    def _info(self) -> dict[str, Any]:
        out = self.session_info()
        assert out is not None
        return out

    def session_info(self) -> dict[str, Any] | None:
        if self.session is None:
            return None
        return dict(
            id=self.session,
            random_share=self.share,
            seed=self.seed,
            n_random=sum(r["how"] == "random" for r in self.order),
            n_ranked=sum(r["how"] == "ranked" for r in self.order),
        )

    def _need(self) -> int:
        _ = self.log  # refused when paint only
        if self.session is None:
            raise ValueError("start a session first")
        return self.session

    # -------------------------------------------------------------- states
    def states(self) -> dict[Site, str]:
        """The state of each site of the session (module docstring)."""
        if self.session is None:
            return {}
        log = self.log.read()
        s = self.session
        out: dict[Site, str] = {}
        for r in log.shown(s):
            out[parse_site(r["Site"])] = "shown"
        for r in log.records:
            if r["Record"] == "skipped" and r["Session"] == s:
                out[parse_site(r["Site"])] = "skipped"
        for site in log.answered(s):
            out[parse_site(site)] = "answered"
        return out

    def _row(self, r: dict[str, Any], state: str) -> dict[str, Any]:
        rnd = r["how"] == "random"
        hide = rnd and state != "answered"
        return dict(
            id=site_id((r["i"], r["j"], r["k"])),
            kind=None if hide else r["kind"],
            rank=None if rnd else r["rank"],
            score=r.get("score"),
            state=state,
        )

    def queue(self) -> dict[str, Any]:
        st = self.states()
        rows = self.order if self.session is not None else self._ranked()
        return dict(
            sites=[
                self._row(r, st.get((r["i"], r["j"], r["k"]), "not shown"))
                for r in rows
            ],
            session=self.session_info(),
            classes=classes(self.hemi),
            menu=menu(self.hemi),
            affine=self.affine.tolist(),
            log=None if self.log_path is None else str(self.log_path),
            edits=str(self.edits),
            hemisphere=self.hemi,
            paint_only=self.paint_only,
            undo_depth=len(self.undo),
        )

    def _find(self, site: Site) -> dict[str, Any]:
        for r in self.order:
            if (r["i"], r["j"], r["k"]) == site:
                return r
        raise ValueError(f"site {site_id(site)} is not in the session")

    def next(self, site: str | None = None) -> dict[str, Any]:
        """The next site not shown, or the site named; recorded as shown."""
        s = self._need()
        with self.lock:
            if site is None:
                st = self.states()
                todo = [r for r in self.order if (r["i"], r["j"], r["k"]) not in st]
                if not todo:
                    return dict(site=None)
                r = todo[0]
            else:
                r = self._find(parse_site(site))
            c = (r["i"], r["j"], r["k"])
            self.log.shown(s, c, r["kind"], r["rank"], r.get("score"))
        mm = (self.affine @ np.r_[np.asarray(c, float), 1.0])[:3]
        rnd = r["how"] == "random"
        return dict(
            site=dict(
                id=site_id(c),
                kind=None if rnd else r["kind"],
                rank=None if rnd else r["rank"],
                voxel=list(c),
                mm=[float(x) for x in mm],
            )
        )

    def skip(self, site: str) -> dict[str, Any]:
        s = self._need()
        c = parse_site(site)
        self._find(c)
        with self.lock:
            self.log.skipped(s, c)
        return dict(site=site_id(c), state="skipped")

    # -------------------------------------------------------------- edits
    def _read_edits(self) -> label_edits.Edits:
        """The current edits (empty if there is no file yet), checked against
        the review's grid, affine and hemisphere (label_edits.read)."""
        if not label_edits.edits_paths(self.edits)["sidecar"].exists():
            return label_edits.empty()
        return label_edits.read(
            self.edits, shape=self.shape, affine=self.affine, hemisphere=self.hemi
        )

    def _check_outputs(self) -> None:
        """Raises ValueError if the dseg, the queue or the fix diff is not the
        file that the review read at start (module docstring, Outputs
        replaced)."""
        changed = [p.name for p, i in self._outputs.items() if identity(p) != i]
        if changed:
            raise ValueError(
                f"the outputs were regenerated while the viewer ran "
                f"({', '.join(changed)} changed): restart the viewer; "
                f"nothing was written"
            )

    def _update(
        self,
        fn: Callable[[label_edits.Current], label_edits.Edits | None],
        after: Callable[[label_edits.Current], None] | None = None,
    ) -> label_edits.Current:
        """One locked read-change-write of the edits (label_edits.update),
        with the memory copy. The labels that fn writes are checked
        (label_edits.paint, label_edits.restore). First, under the edits'
        file lock, refuse if the outputs were replaced (_check_outputs).
        after: the step after the write, under the same file lock (if it
        raises, the edits are written back; label_edits.update). Call with
        the lock held."""

        def checked(cur: label_edits.Current) -> label_edits.Edits | None:
            self._check_outputs()
            return fn(cur)

        self._cur = label_edits.update(
            self.edits,
            checked,
            hemisphere=self.hemi,
            shape=self.shape,
            affine=self.affine,
            known=self._cur,
            labels_checked=True,
            after=after,
        )
        return self._cur

    def _input_labels(self) -> Arr:
        """The input labels of the whole grid, memory-mapped: opened once for
        the review (blocks.open_volume decompresses a .nii.gz once), so that a
        read of voxels by index reads only those voxels. Raises ValueError if
        they are not on the review's grid or have a scaling."""
        if self._inputs_map is None:
            if self.inputs is None:
                self.inputs = white_dseg.input_labels(self.dseg)
            src = blocks.open_volume(self.inputs)
            if tuple(src.shape[:3]) != self.shape:
                raise ValueError(
                    f"{self.inputs.name}: the input labels are not on the grid "
                    f"{self.shape}"
                )
            from nibabel.arrayproxy import ArrayProxy

            proxy = src.img.dataobj
            if not isinstance(proxy, ArrayProxy) or (proxy.slope, proxy.inter) != (
                1.0,
                0.0,
            ):
                raise ValueError(f"{self.inputs.name}: the input labels are scaled")
            m = np.asarray(proxy.get_unscaled())
            self._inputs_map = m.reshape(m.shape[:3])
            self._inputs_src = src
        return self._inputs_map

    def _painted(
        self, edits: label_edits.Edits, voxels: Arr, labels: Arr
    ) -> tuple[label_edits.Edits, label_edits.Edits, Arr]:
        """The edits with the voxels painted with the labels (label_edits.paint,
        against the dseg's labels from labels(); label_edits.INPUT for the
        eraser), the undo: the earlier entries of the voxels whose label
        changes, and the input labels of the voxels (int64). Only the voxels
        are read, not their box. Raises ValueError for a voxel off the grid
        or a label not in label-set/1."""
        v = np.asarray(voxels, np.int64).reshape(-1, 3)
        if not len(v):
            return edits, label_edits.empty(), np.zeros(0, np.int64)
        grid = self._input_labels()
        # the whole grids, unread: paint indexes only the voxels (the input
        # labels are a memory map, the dseg's labels are in memory). paint
        # refuses a voxel off the grid before the input labels are indexed.
        new, undo = label_edits.paint(
            edits,
            grid,
            v,
            np.asarray(labels, np.int64),
            dseg=self.labels(),
            shape=self.shape,
        )
        return new, undo, grid[tuple(v.T)].astype(np.int64)

    def _on_grid(self, v: Arr) -> None:
        if len(v) and not np.all((v >= 0) & (v < self.shape)):
            raise ValueError("a voxel is outside the grid")

    def _commit(
        self,
        voxels: Arr,
        labels: Arr,
        after: Callable[[Arr], None] | None = None,
    ) -> tuple[label_edits.Edits, Arr]:
        """Under the lock: the voxels painted with the labels in one locked
        read-change-write of the edits (_painted, _update), and their earlier
        entries pushed on the undo stack. after, if given, is called with the
        input labels of the voxels after the write, under the edits' file
        lock; if it raises, the edits are written back, nothing is pushed,
        and its error propagates. Returns the undo (no voxel: nothing is
        written or pushed) and the input labels of the voxels."""
        out: dict[str, Any] = {}

        def fn(cur: label_edits.Current) -> label_edits.Edits | None:
            new, undo, inp = self._painted(cur.edits, voxels, labels)
            out.update(undo=undo, inp=inp)
            return new if len(undo) else None

        def step(cur: label_edits.Current) -> None:
            assert after is not None
            after(out["inp"])

        cur = self._update(fn, None if after is None else step)
        undo: label_edits.Edits = out["undo"]
        if len(undo):
            assert cur.sha is not None
            self.undo.append(undo)
            self._undo_sha.append(cur.sha)
        return undo, out["inp"]

    def _pop_undo(self) -> int:
        """Under the lock: the top of the undo stack written back
        (label_edits.restore) and popped. Returns its number of voxels.
        Raises ValueError, and changes nothing, if the edits on disk are not
        the ones that the top write left, or if the outputs were replaced
        (_update)."""
        u, sha = self.undo[-1], self._undo_sha[-1]

        def fn(cur: label_edits.Current) -> label_edits.Edits:
            if cur.sha != sha:
                raise ValueError(
                    "the edits on disk are not the ones that this viewer run last "
                    "wrote (another writer changed them): undo refused"
                )
            return label_edits.restore(cur.edits, u)

        self._update(fn)
        self.undo.pop()
        self._undo_sha.pop()
        return len(u)

    # -------------------------------------------------------------- paint
    def paint_voxels(self, strokes: Sequence[dict[str, Any]]) -> tuple[Arr, Arr]:
        """The strokes of a paint (module docstring) as voxels (N, 3) and their
        labels (label_edits.INPUT for the eraser), a voxel in two strokes in
        the later one only (stroke_voxels). Raises ValueError for no stroke, a
        label not in the menu or a voxel off the grid."""
        if not strokes:
            raise ValueError("a paint needs at least one stroke")
        allowed = {m["value"] for m in menu(self.hemi)}
        vs, labs = [], []
        for lab, v in stroke_voxels(strokes, "label"):
            if isinstance(lab, bool) or lab not in allowed:
                raise ValueError(
                    f"the label {lab!r} is not in the paint menu "
                    f"{sorted(allowed, key=str)}"
                )
            self._on_grid(v)
            vs.append(v)
            labs.append(
                np.full(len(v), label_edits.INPUT if lab == ERASER else int(lab))
            )
        if not vs:
            raise ValueError("a paint needs at least one voxel")
        return np.concatenate(vs).astype(np.int64), np.concatenate(labs).astype(
            np.int64
        )

    def paint(self, strokes: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """A paint (module docstring): {voxels, undo_depth}."""
        v, lab = self.paint_voxels(strokes)
        with self.lock:
            undo, _ = self._commit(v, lab)
            return dict(voxels=len(undo), undo_depth=len(self.undo))

    def undo_last(self) -> dict[str, Any]:
        """The last write into the edits undone (module docstring, Undo): its
        earlier entries written back (label_edits.restore). {voxels,
        undo_depth}. Raises ValueError if the stack is empty, or if another
        writer changed the edits since the top write (_pop_undo)."""
        with self.lock:
            if not self.undo:
                raise ValueError("nothing to undo in this viewer run")
            n = self._pop_undo()
            return dict(voxels=n, undo_depth=len(self.undo))

    def _fix(self, site: Site) -> tuple[int, Arr, Arr]:
        """The fix of the override site: its id, its voxels (N, 3, sorted by
        linear index) and their labels (fix_diff.fix_voxels on the dseg's fix
        diff). Raises ValueError for a site that is not an override site."""
        from baillarger import fix_diff

        r = self._find(site)
        if r["kind"] != "override" or r["how"] != "ranked":
            raise ValueError(
                f"{ACCEPT}: the site {site_id(site)} is not an override site"
            )
        # the fix ids of the rows are those of the fix diff at start
        self._check_outputs()
        if self._fix_rows is None:
            rows = fix_diff.read_diff(
                white_dseg.output_paths(self.dseg)["diff"], self.shape
            )
            self._check_outputs()  # not replaced during the read
            self._fix_rows = rows
        fid = int(r["fix"])
        idx, new = fix_diff.fix_voxels(self._fix_rows, fid)
        if not len(idx):
            raise ValueError(f"{ACCEPT}: fix {fid} has no voxel in the fix diff")
        v = np.stack(np.unravel_index(idx, self.shape), 1).astype(np.int64)
        return fid, v, new

    # -------------------------------------------------------------- choices
    def window(self, site: Site, params: dict[str, Any]) -> tuple[Arr, Site]:
        """The labels (int64) in the box round a site that the choice with
        these params reads (Choices.half), and the box's first voxel."""
        lab = self.labels()
        half = self.choices.half(params)
        c = np.asarray(site, np.int64)
        lo = np.clip(c - half, 0, np.array(lab.shape))
        hi = np.clip(c + half + 1, 0, np.array(lab.shape))
        box = np.array(lab[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], np.int64)
        return box, (int(lo[0]), int(lo[1]), int(lo[2]))

    def _preview(
        self, site: Site, choice: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """The preview of manual-paint or accept-fix, computed here."""
        t0 = time.perf_counter()
        if choice == MANUAL:
            none = np.zeros((0, 3), np.int64)
            p = Preview(none, none, False)
        else:
            _, v, _ = self._fix(site)
            p = Preview(v, np.zeros((0, 3), np.int64), False)
        return _preview_dict(p, t0)

    def _remote(
        self,
        box: Arr,
        lo: Site,
        site: Site,
        choice: str,
        params: dict[str, Any],
        stops: int,
    ) -> dict[str, Any]:
        """The preview of a choice computed by the worker process (started if
        there is none). stops: self._stops when the job started. Raises
        PreviewError if it fails, if a newer preview stops the worker, or if
        the worker ends without a result."""
        t0 = time.perf_counter()
        with self._wlock:
            if self._stops != stops:
                raise PreviewError("replaced by a newer preview")
            w = self._worker
            if w is None or not w.proc.is_alive():
                if w is not None:  # ended while idle
                    w.conn.close()
                w = self._worker = self._start_worker()
        try:
            w.conn.send((self.choices, box, lo, site, choice, params))
            tag, val = w.conn.recv()
        except (EOFError, OSError):
            w.proc.join()
            with self._wlock:
                if self._worker is w:
                    self._worker = None
            w.conn.close()
            if w.replaced:
                raise PreviewError("replaced by a newer preview") from None
            raise PreviewError(
                f"the preview worker ended without a result (exit code "
                f"{w.proc.exitcode})"
            ) from None
        if tag != "ok":
            raise PreviewError(val)
        return _preview_dict(val, t0)

    @staticmethod
    def _start_worker() -> _Worker:
        ctx = multiprocessing.get_context("spawn")
        here, there = ctx.Pipe()
        proc = ctx.Process(
            target=_preview_worker, args=(there,), daemon=True, name="preview"
        )
        proc.start()
        there.close()  # so that recv sees the worker's end (EOFError)
        return _Worker(proc, here)

    def _stop_worker(self, replaced: bool = True) -> None:
        """Stop the worker process (terminate, join); the job that waits on it
        fails (_remote)."""
        with self._wlock:
            w, self._worker = self._worker, None
            self._stops += 1
        if w is not None:
            w.replaced = replaced
            w.proc.terminate()
            w.proc.join()

    def close(self) -> None:
        """Stop the preview worker and wait for its job to end."""
        self._stop_worker(replaced=False)
        th = self.jobs.thread
        if th is not None:
            th.join()

    def preview_key(self, site: Site, choice: str, params: dict[str, Any]) -> str:
        """The job key of a preview: the SHA-256 of its site, choice, params and
        the SHA-256 of the edits as the review last read or wrote them."""
        sha = None if self._cur is None else self._cur.sha
        text = json.dumps([list(site), choice, params, sha], sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    def preview(self, site: str, choice: str, params: dict[str, Any]) -> dict[str, Any]:
        """The preview, or a job that computes it (module docstring): the
        newest request wins."""
        self._need()
        c = parse_site(site)
        self._find(c)
        params = dict(params or {})
        if choice in (MANUAL, ACCEPT):
            return self._preview(c, choice, params)
        key = self.preview_key(c, choice, params)
        with self._plock:
            st = self.jobs.status()
            if st["status"] == "running" and st["key"] == key:
                return dict(status="running", job=key)
            if st["status"] == "running":
                self._stop_worker()
                th = self.jobs.thread
                if th is not None:
                    th.join()
            box, lo = self.window(c, params)
            stops = self._stops
            out = self.jobs.run(
                key,
                f"preview {choice} at {site_id(c)}",
                lambda log: self._remote(box, lo, c, choice, params, stops),
            )
            if out["status"] == "error":
                return dict(status="failed", job=key, error=out["error"])
        return self._outcome(key, self.jobs.wait(key, WAIT_S))

    def job(self, key: str) -> dict[str, Any]:
        """The state of the preview job key: its result when it is done."""
        return self._outcome(key, self.jobs.wait(key, 0))

    def _outcome(self, key: str, out: dict[str, Any]) -> dict[str, Any]:
        if out["status"] == "done":
            try:
                return self.jobs.get(key)
            except KeyError:
                pass  # dropped from memory: reported as no job, below
        elif out["status"] == "error":
            return dict(status="failed", job=key, error=out["error"])
        elif out["status"] == "running":
            return dict(status="running", job=key)
        raise ValueError(f"no preview job {key}")

    def answer(
        self,
        site: str,
        choice: str,
        params: dict[str, Any],
        seconds: float,
        strokes: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """The choice's voxels at the site, written into the edits, then
        appended to the log with the reviewer time (module docstring). manual-paint
        takes the strokes of the paint mode, and no other choice takes
        strokes. Returns the answer's Id, the site's kind and how it was
        chosen (shown to the page only now, for a random site), the number of
        voxels written into the edits, and the depth of the undo stack.
        Nothing is written if the edits refuse the answer. The log append is
        the step after the write into the edits, under the edits' file lock
        (label_edits.update, after), so no other writer writes between the
        two. If the log refuses the answer, the edits are written back under
        that lock and no undo entry is pushed, so the edits and the undo
        stack are as they were, and the error is the log's."""
        s = self._need()
        c = parse_site(site)
        r = self._find(c)
        if (choice == MANUAL) != (strokes is not None):
            raise ValueError(
                "strokes go with the choice manual-paint, and only with it"
            )
        params = dict(params or {})
        if choice == MANUAL:
            assert strokes is not None
            v, lab = self.paint_voxels(strokes)
            a = Answer(v, lab, params)
        elif choice == ACCEPT:
            fid, v, lab = self._fix(c)
            a = Answer(v, lab, {"Fix": fid})
        else:
            box, lo = self.window(c, params)
            a = self.choices.answer(box, lo, c, choice, params)
        ids: list[int] = []

        def append(inp: Arr) -> None:
            # the eraser's voxels: the log records the label they take
            logged = np.where(a.labels == label_edits.INPUT, inp, a.labels)
            ids.append(
                self.log.answer(
                    s,
                    c,
                    choice,
                    a.voxels,
                    logged,
                    self.dseg_sha,
                    float(seconds),
                    params=a.params,
                )
            )

        with self.lock:
            undo, _ = self._commit(a.voxels, a.labels, append)
        return dict(
            answer=ids[0],
            kind=r["kind"],
            how=r["how"],
            rank=r["rank"],
            voxels=len(a.voxels),
            edited=len(undo),
            undo_depth=len(self.undo),
        )

    def scribble(self, strokes: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
        """Strokes, each {class, voxels}, appended as one scribble (module
        docstring: a voxel in two strokes stays in the later one;
        stroke_voxels)."""
        s = self._need()
        if not strokes:
            raise ValueError("a scribble needs at least one stroke")
        out = [(int(c), v) for c, v in stroke_voxels(strokes, "class")]
        with self.lock:
            aid = self.log.scribble(s, out, self.dseg_sha, float(seconds))
        return dict(answer=aid, strokes=len(out), voxels=sum(len(v) for _, v in out))

    # -------------------------------------------------------------- routes
    def get(self, path: str, q: dict[str, str]) -> dict[str, Any]:
        if path == "/review/queue":
            return self.queue()
        if path == "/review/next":
            return self.next(q.get("site"))
        if path == "/review/preview":
            return self.job(q["job"])
        raise KeyError(path)

    def post(self, path: str, req: dict[str, Any]) -> dict[str, Any]:
        if path == "/review/session":
            if req.get("resume"):
                return dict(session=self.resume())
            if "random_share" not in req:
                raise ValueError("give the random share (0 to 1)")
            seed = req.get("seed")
            return dict(
                session=self.start(
                    float(req["random_share"]), None if seed is None else int(seed)
                )
            )
        if path == "/review/skip":
            return self.skip(str(req["site"]))
        if path == "/review/preview":
            return self.preview(
                str(req["site"]), str(req["choice"]), req.get("params") or {}
            )
        if path == "/review/answer":
            st = req.get("strokes")
            return self.answer(
                str(req["site"]),
                str(req["choice"]),
                req.get("params") or {},
                float(req["reviewer_seconds"]),
                None if st is None else list(st),
            )
        if path == "/review/scribble":
            return self.scribble(list(req["strokes"]), float(req["reviewer_seconds"]))
        if path == "/review/paint":
            return self.paint(list(req["strokes"]))
        if path == "/review/undo":
            return self.undo_last()
        raise KeyError(path)


def route(
    h: Any, method: str, path: str, q: dict[str, str], req: dict[str, Any]
) -> bool:
    """Serve a /review/ route for handler h (its `wmreview`); False for any
    other path. 404: no review or no such route; 400: a bad request."""
    if not path.startswith("/review/"):
        return False
    rv: Review | None = getattr(h, "wmreview", None)
    if rv is None:
        h.send_error(404)
        return True

    try:
        out = rv.get(path, q) if method == "GET" else rv.post(path, req)
    except KeyError as exc:
        if exc.args and exc.args[0] == path:
            h.send_error(404)
        else:
            _error(h, f"missing {exc}")
        return True
    except Exception as exc:  # reported to the page; the server keeps serving
        _error(h, f"{type(exc).__name__}: {exc}")
        return True
    h.send(json.dumps(out, default=float).encode(), "application/json")
    return True


def _error(h: Any, msg: str) -> None:
    h.send(json.dumps(dict(error=msg)).encode(), "application/json", 400)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--sites",
        type=Path,
        help="with --white: the review queue (a desc-wmsites_sites.tsv, wm-sites/1); "
        "needs --log and --edits",
    )
    ap.add_argument(
        "--log",
        type=Path,
        help="with --sites: the answer log (.jsonl, wm-edits/1), made if it is not "
        "there; there is no default path",
    )
    ap.add_argument(
        "--edits",
        type=Path,
        help="with --white: the label edits (.npy, label-edits/1) that the answers "
        "and the paint mode write into, made at the first write that changes a "
        "label; there is no default path. Without --sites: paint only",
    )


def for_white(
    a: argparse.Namespace,
    surf: Path,
    dseg: Path,
    labels: Callable[[], Arr],
    affine: Arr,
) -> Review | None:
    """The review of --white mode: with --sites (--log and --edits are then
    needed), or paint only with --edits alone; None with neither."""
    if a.sites is None:
        if a.log is not None:
            raise ValueError("--log needs --sites")
        if a.edits is None:
            return None
        return Review(None, None, a.edits, dseg, labels, affine, None)
    if a.log is None:
        raise ValueError("--sites needs --log: the log's path (there is no default)")
    if a.edits is None:
        raise ValueError(
            "--sites needs --edits: the label edits' path (there is no default)"
        )
    return Review(
        a.sites,
        a.log,
        a.edits,
        dseg,
        labels,
        affine,
        lambda n, seed: read_random_sites(surf, n, seed),
    )


def check_log(path: Path) -> int:
    """The number of records of the log at path, read with phase 1.3's reader
    (which checks every record)."""
    return len(read_log(path).records)
