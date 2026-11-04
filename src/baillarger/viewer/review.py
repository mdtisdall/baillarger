"""The stream 1 review pane's server side (phase 4.2): the queue of sites
(wm-sites/1), review sessions with random sites, the site answers with their
previews, and scribbles, written to the answer log (wm-edits/1). The pane holds
no stream 1 logic: it calls phase 1.3's code (baillarger.wm_sites,
baillarger.wm_edits, baillarger.wm_choices) and writes nothing else.

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
  /review/preview?job=... until {status: "done", ...}.
- POST /review/answer {site, choice, params, reviewer_seconds}: the choice's
  voxels and labels at the site, appended to the log.
- POST /review/scribble {strokes: [{class, voxels}], reviewer_seconds}:
  strokes appended to the log as one scribble. A voxel in two strokes stays in
  the later one only (the log takes a voxel in at most one stroke).

Each write holds the review's lock and is flushed to the disk before the
response (EditLog appends and fsyncs). A choice and its preview read a box
round the site (Choices.half), so no response scales with the volume.

manual-paint is not a choice of the server: its preview gives the commands of
baillarger.wm_paint (export a crop, paint it in ITK-SNAP, import it into the
log), and /review/answer refuses it.
"""

import argparse
import json
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from baillarger.bids import file_sha256
from baillarger.labels import CORTEX_LABEL, LUT_NAMES, WM_LABEL
from baillarger.wm_edits import EditLog, read_log
from baillarger.wm_sites import read_random_sites, read_sites

type Arr = npt.NDArray[Any]
type Site = tuple[int, int, int]

#: the choice whose answer comes from ITK-SNAP through baillarger.wm_paint
MANUAL = "manual-paint"
#: seconds a preview request waits before it answers as a job
WAIT_S = 1.5


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
    the ITK-SNAP command of manual-paint (or None), and whether a face of the
    preview's box has WM on it (then the preview is approximate)."""

    answer: npt.NDArray[np.int64]
    correction: npt.NDArray[np.int64]
    overrides: bool
    command: str | None = None
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


# ------------------------------------------------------------------ the review


@dataclass
class Job:
    status: str = "running"
    result: dict[str, Any] | None = None
    error: str | None = None
    thread: threading.Thread | None = field(default=None, repr=False)


class Review:
    """The review on the server: the queue, the log, the session and the
    choices (module docstring). labels() gives the segmentation's voxels (the
    whole grid, read once: the label pyramid's level 0); random(n, seed) the
    random sites (wm_sites.read_random_sites)."""

    def __init__(
        self,
        sites: Path,
        edits: Path,
        dseg: Path,
        labels: Callable[[], Arr],
        affine: npt.ArrayLike,
        random: Callable[[int, int], list[dict[str, Any]]],
        choices: Choices | None = None,
        echoes: Sequence[Path] = (),
    ) -> None:
        self.rows, self.side = read_sites(sites)
        self.hemi = str(self.side["Hemisphere"])
        self.queue_sha = file_sha256(sites)
        self.dseg_sha = file_sha256(dseg)
        self.edits = Path(edits)
        if not self.edits.exists():
            EditLog.create(self.edits, dseg, self.hemi)
        self.log = EditLog(self.edits)
        self.labels, self.random = labels, random
        self.affine = np.asarray(affine, np.float64)
        voxel_mm = np.linalg.norm(self.affine[:3, :3], axis=0)
        self.choices: Choices = choices or WmChoices(self.hemi, voxel_mm)
        self.dseg = Path(dseg)
        self.echoes = [Path(e) for e in echoes]
        self.lock = threading.Lock()  # every write, and the session
        self.session: int | None = None
        self.share: float | None = None
        self.seed: int | None = None
        self.order: list[dict[str, Any]] = []
        self.jobs: dict[str, Job] = {}
        last = self.log.sessions()
        if last:
            self._resume(last[-1])

    # -------------------------------------------------------------- sessions
    def _ranked(self) -> list[dict[str, Any]]:
        return [dict(r, how="ranked") for r in self.rows]

    def _draw(self, share: float, seed: int | None) -> list[dict[str, Any]]:
        n = random_count(len(self.rows), share)
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
            affine=self.affine.tolist(),
            log=str(self.edits),
            hemisphere=self.hemi,
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

    def paint_commands(self, site: Site) -> str:
        """The shell commands of manual-paint at the site (baillarger.wm_paint;
        docs/formats/wm-paint.md), with absolute paths. The crop goes next to
        the log."""
        i, j, k = site
        log, dseg = self.edits.resolve(), self.dseg.resolve()
        crop = log.parent / f"{log.stem}_paint" / f"site-{i}-{j}-{k}"
        echo = "".join(f" --echo {e.resolve()}" for e in self.echoes)
        snap = (
            f"-g {crop}/echo-1.nii.gz -s {crop}/labels.nii.gz"
            if self.echoes
            else f"-g {crop}/labels.nii.gz"
        )
        return (
            f"python -m baillarger.wm_paint export {dseg} --site {i} {j} {k} "
            f"--out {crop}{echo}\n"
            f"itksnap {snap}\n"
            f"python -m baillarger.wm_paint import {crop} --log {log} "
            f"--session {self.session} --seconds <reviewer seconds>"
        )

    def _preview(
        self, site: Site, choice: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        if choice == MANUAL:
            none = np.zeros((0, 3), np.int64)
            p = Preview(none, none, False, self.paint_commands(site))
        else:
            box, lo = self.window(site, params)
            p = self.choices.preview(box, lo, site, choice, params)
        return dict(
            status="done",
            answer=p.answer.tolist(),
            correction=p.correction.tolist(),
            overrides=p.overrides,
            command=p.command,
            cut_by_box=p.cut_by_box,
            seconds=time.perf_counter() - t0,
        )

    def preview(self, site: str, choice: str, params: dict[str, Any]) -> dict[str, Any]:
        """The preview, or a job that computes it (module docstring)."""
        self._need()
        c = parse_site(site)
        self._find(c)
        job = Job()

        def run() -> None:
            try:
                job.result = self._preview(c, choice, dict(params or {}))
                job.status = "done"
            except Exception as exc:  # reported through the job
                job.error, job.status = f"{type(exc).__name__}: {exc}", "failed"

        key = uuid.uuid4().hex[:12]
        if len(self.jobs) > 32:
            self.jobs = {k: j for k, j in self.jobs.items() if j.status == "running"}
        self.jobs[key] = job
        job.thread = threading.Thread(target=run, daemon=True)
        job.thread.start()
        job.thread.join(WAIT_S)
        return self.job(key)

    def job(self, key: str) -> dict[str, Any]:
        j = self.jobs.get(key)
        if j is None:
            raise ValueError(f"no preview job {key}")
        if j.status == "running":
            return dict(status="running", job=key)
        if j.status == "failed":
            return dict(status="failed", job=key, error=j.error)
        assert j.result is not None
        return j.result

    def answer(
        self, site: str, choice: str, params: dict[str, Any], seconds: float
    ) -> dict[str, Any]:
        """The choice's voxels at the site, appended to the log with the reviewer
        time. Returns the answer's Id, and the site's kind and how it was
        chosen (shown to the page only now, for a random site)."""
        s = self._need()
        c = parse_site(site)
        r = self._find(c)
        if choice == MANUAL:
            raise ValueError("manual-paint: use the commands of its preview")
        params = dict(params or {})
        box, lo = self.window(c, params)
        a = self.choices.answer(box, lo, c, choice, params)
        with self.lock:
            aid = self.log.answer(
                s,
                c,
                choice,
                a.voxels,
                a.labels,
                self.dseg_sha,
                float(seconds),
                params=a.params,
            )
        return dict(
            answer=aid,
            kind=r["kind"],
            how=r["how"],
            rank=r["rank"],
            voxels=len(a.voxels),
        )

    def scribble(self, strokes: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
        """Strokes, each {class, voxels}, appended as one scribble (module
        docstring: a voxel in two strokes stays in the later one)."""
        s = self._need()
        if not strokes:
            raise ValueError("a scribble needs at least one stroke")
        owner: dict[Site, int] = {}
        for k, st in enumerate(strokes):
            for v in st["voxels"]:
                owner[parse_site(v)] = k
        out = []
        for k, st in enumerate(strokes):
            vs = [v for v, o in owner.items() if o == k]
            if vs:
                out.append((int(st["class"]), np.asarray(vs, np.int64)))
        with self.lock:
            aid = self.log.scribble(s, out, self.dseg_sha, float(seconds))
        return dict(answer=aid, strokes=len(out), voxels=len(owner))

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
            return self.answer(
                str(req["site"]),
                str(req["choice"]),
                req.get("params") or {},
                float(req["reviewer_seconds"]),
            )
        if path == "/review/scribble":
            return self.scribble(list(req["strokes"]), float(req["reviewer_seconds"]))
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
        help="with --white: the review queue (a desc-wmsites_sites.tsv, wm-sites/1)",
    )
    ap.add_argument(
        "--edits",
        type=Path,
        help="with --sites: the answer log (.jsonl, wm-edits/1), made if it is not "
        "there; there is no default path",
    )


def for_white(
    a: argparse.Namespace,
    surf: Path,
    dseg: Path,
    labels: Callable[[], Arr],
    affine: Arr,
    echoes: Sequence[Path] = (),
) -> Review | None:
    """The review of --white mode, if --sites is given (--edits is then needed)."""
    if a.sites is None:
        if a.edits is not None:
            raise ValueError("--edits needs --sites")
        return None
    if a.edits is None:
        raise ValueError("--sites needs --edits: the log's path (there is no default)")
    return Review(
        a.sites,
        a.edits,
        dseg,
        labels,
        affine,
        lambda n, seed: read_random_sites(surf, n, seed),
        echoes=echoes,
    )


def check_log(path: Path) -> int:
    """The number of records of the log at path, read with phase 1.3's reader
    (which checks every record)."""
    return len(read_log(path).records)
