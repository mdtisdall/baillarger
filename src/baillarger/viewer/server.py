"""The viewer's local web server (binds 127.0.0.1 only) and its shared state.

Any number of pages (windows, tabs, browsers) can be open at once; each shows the panes
named in its address (static/index.html, ?panes=...). They share one state, kept here:
two points, the crosshair (a world point: the slices of the volume pane) and the WM
selection (a vertex of the true white surface with the shown trace there: the sections,
the streamlines and the markers), the frame of the sections and their rotation about the
normal (degrees, on top of the frame), and the colour layer. A page sends its changes
(POST /state) and every page receives every change as it happens from the event stream
(GET /events, Server-Sent Events), so a click in one window moves the point in all of
them. In each pane a click moves that pane's own point (the slices: the crosshair; the
surface and the map: the WM selection) and a shift-click moves the other one.

Requests:
- GET / (the page), /static/<file>; /meta; /state; /events;
  /layer?name=...; /section?i=&half=[&right=&up=&level=][&rot=] (right, up: 3
  numbers each, comma-separated: the screen's axes on inflated level `level`; rot:
  the turn of both axes about the normal, degrees, the shared state's `rotation`);
  /axes/<k>/<u|v>.tck (the curve that the u or v section of /section answer k
  follows, world mm: its `axes.id`);
  /mesh?level=k[&faces=1] (the white surface for the 3D pane, Site.mesh_payload);
  /locate?mm=x,y,z (the shown trace rooted nearest a world point);
  /snap?mm=x,y,z (the WM selection at the white-mesh vertex nearest a world point);
  /vertex?v=k (the WM selection at vertex k);
  /streamlines?i=&radius= (the traces from and to trace i, by kind and class);
  /tracts/<i>/<radius>/<kind>_<class>.tck (the traces of one group, as .tck);
  /overview/<class>/<every>.tck (every k-th shown trace of a class);
  /echo2.nii.gz, /white.gii, /gm.gii (world mm, for the volume pane);
  /filters (the filter job's status and log, the cached results, the one shown, the
  choices: baillarger.viewer.filters);
  with --labels (baillarger.viewer.labels): /labels/list, /labels.nii.gz,
  /labels/echo<k>.nii.gz (k = 1, 2, 3: the echoes the detector reads).
- Labels only (no column volume; --labels): /meta and /echo2.nii.gz come from the
  review, and only the page, /events, /state and the label routes are served.
- A white surface alone (--white, baillarger.viewer.white_view): /meta, /mesh,
  /vertex, /snap and /white.gii come from its overview, and only those, the page,
  /events and /state are served.
- The volume as bricks (baillarger.viewer.bricks), in every mode that has a volume
  (the column volume's box, the label review's box, or with --white the whole echo 2):
  /volume (its name, without reading it; 404: none), /volume/meta (the levels' shapes
  and affines, the brick size, the dtype, the default window: read on first use),
  /volume/bricks?level=L&b=i,j,k;i,j,k;... (those bricks' bytes, one after another;
  gzip when the page accepts it).
- POST /state {from, patch}: merge patch into the state; every page gets
  {from, seq, patch}.
- POST /filters/run {from, params}: start a filter (one at a time, in a thread) or
  find it cached; /filters/use {from, key}: show a finished result (key null: the raw
  echo 2) in the sections and its layers; every page gets {filter: key}.
- POST /labels/decide {from, id, status}, /labels/brush {from, mm, radius, label}:
  change the pial-label review and save it; every page gets the new labels_seq.
"""

import argparse
import gzip
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import numpy as np

from baillarger.data import echo_path
from baillarger.viewer import filters as VF
from baillarger.viewer import streamlines as SV
from baillarger.viewer.bricks import ArrayPyramid, LazyVolume
from baillarger.viewer.data import Site
from baillarger.viewer.labels import LabelReview
from baillarger.viewer.white_view import WhiteView

STATIC = Path(__file__).parent / "static"
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".css": "text/css",
}
KEEPALIVE_S = 15.0


class Shared:
    """The state that every page shares, and the queues of the open event streams."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = dict(
            crosshair=None, selection=None, frame=None, rotation=0.0, layer=None
        )
        self.seq = 0
        self.lock = threading.Lock()
        self.listeners: list[queue.Queue[str]] = []

    def update(self, sender: str, patch: dict[str, Any]) -> int:
        with self.lock:
            self.state.update(patch)
            self.seq += 1
            msg = json.dumps(dict(sender=sender, seq=self.seq, patch=patch))
            for q in self.listeners:
                q.put(msg)
            return self.seq

    def listen(self) -> "queue.Queue[str]":
        q: queue.Queue[str] = queue.Queue()
        with self.lock:
            self.listeners.append(q)
        return q

    def drop(self, q: "queue.Queue[str]") -> None:
        with self.lock:
            if q in self.listeners:
                self.listeners.remove(q)


def floats(s: str | None) -> list[float] | None:
    return None if not s else [float(x) for x in s.split(",")]


def round3(a: Any) -> list[float | None]:
    return [None if not np.isfinite(x) else round(float(x), 4) for x in a]


# the routes that need no column volume (labels-only mode serves these, /meta and
# /echo2.nii.gz from the review)
# the page and the shared state, which every mode serves and none computes
PAGE = ("/", "/index.html", "/events", "/state")
NO_SITE = (*PAGE, "/labels/list", "/labels.nii.gz")


class Handler(BaseHTTPRequestHandler):
    site: Site | None
    shared: Shared
    labels: LabelReview | None = None
    white: WhiteView | None = None
    volume: LazyVolume | None = None
    lock = threading.Lock()  # one site computation at a time
    groups: dict[tuple[int, float], dict[str, list[int]]] = {}
    axes: dict[int, dict[str, list[list[float]]]] = {}

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, fn: Any) -> None:
        try:
            with self.lock:
                out = fn()
        except Exception as exc:  # report to the page, keep serving
            import traceback

            traceback.print_exc()
            err = json.dumps(dict(error=f"{type(exc).__name__}: {exc}")).encode()
            self.send(err, "application/json", 500)
            return
        self.send(json.dumps(out, default=float).encode(), "application/json")

    def send_json(self, fn: Any) -> None:
        """As json, without the site lock: for calls that only start or read a job."""
        try:
            out = fn()
        except Exception as exc:
            err = json.dumps(dict(error=f"{type(exc).__name__}: {exc}")).encode()
            self.send(err, "application/json", 400)
            return
        self.send(json.dumps(out, default=float).encode(), "application/json")

    def binary(self, fn: Any, ctype: str = "application/octet-stream") -> None:
        with self.lock:
            body = fn()
        self.send(body, ctype)

    def streamline_groups(self, i: int, radius: float) -> dict[str, list[int]]:
        key = (i, radius)
        if key not in self.groups:
            if len(self.groups) > 64:
                self.groups.clear()
            self.groups[key] = cast(Site, self.site).streamlines(i, radius)["_groups"]
        return self.groups[key]

    def streamline_summary(self, i: int, radius: float) -> dict[str, Any]:
        out = cast(Site, self.site).streamlines(i, radius)
        self.groups[(i, radius)] = out.pop("_groups")
        return out

    def section(self, q: dict[str, str]) -> dict[str, Any]:
        out = cast(Site, self.site).section_at(
            int(q["i"]),
            float(q.get("half", 8.0)),
            floats(q.get("right")),
            floats(q.get("up")),
            int(q.get("level", 0)),
            float(q.get("rot", 0.0)),
        )
        k = max(self.axes, default=-1) + 1
        if len(self.axes) > 64:
            self.axes.clear()
        self.axes[k] = out.pop("curves")
        out["axes"] = dict(id=k, points={a: len(c) for a, c in self.axes[k].items()})
        return out

    def events(self) -> None:
        q = self.shared.listen()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            hello = dict(seq=self.shared.seq, state=self.shared.state)
            self.wfile.write(f"event: hello\ndata: {json.dumps(hello)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=KEEPALIVE_S)
                    self.wfile.write(f"data: {msg}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.shared.drop(q)

    def volume_route(self, path: str, q: dict[str, str]) -> None:
        """/volume, /volume/meta and /volume/bricks (not under the site lock: the
        pyramid is read only once built, and LazyVolume builds it once)."""
        v = self.volume
        if v is None:
            self.send_error(404)
        elif path == "/volume":
            self.send(json.dumps(dict(name=v.name)).encode(), "application/json")
        elif path == "/volume/meta":
            self.send_json(v.meta)
        elif path == "/volume/bricks":
            try:
                body = v.bricks(int(q["level"]), q.get("b", ""))
            except (KeyError, ValueError) as exc:
                self.send(str(exc).encode(), "text/plain", 400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                body = gzip.compress(body, compresslevel=1)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/volume" or u.path.startswith("/volume/"):
            self.volume_route(u.path, q)
            return
        if self.white is not None:  # a white surface alone
            w = self.white
            if u.path == "/meta":
                self.json(w.meta)
            elif u.path == "/mesh":
                lv, fc = int(q.get("level", 0)), q.get("faces") == "1"
                self.binary(lambda: w.mesh_payload(lv, fc))
            elif u.path == "/vertex":
                self.json(lambda: w.vertex(int(q["v"])))
            elif u.path == "/snap":
                mm_ = floats(q["mm"])
                self.json(lambda: w.snap(mm_ or []))
            elif u.path == "/white.gii":
                self.binary(w.white_bytes)
            elif u.path not in PAGE and not u.path.startswith("/static/"):
                self.send_error(404)
            if u.path not in PAGE and not u.path.startswith("/static/"):
                return
        if self.site is None:  # labels only
            if u.path == "/meta":
                self.json(lambda: self.review().meta())
                return
            if u.path == "/echo2.nii.gz":
                self.binary(lambda: self.review().echo_bytes(2))
                return
            if u.path not in NO_SITE and not u.path.startswith(
                ("/static/", "/labels/")
            ):
                self.send_error(404)
                return
        # without a site only the NO_SITE (or, for a white surface, PAGE) routes, which
        # do not use it, get here
        s = cast(Site, self.site)
        if u.path in ("/", "/index.html"):
            self.send((STATIC / "index.html").read_bytes(), TYPES[".html"])
        elif u.path.startswith("/static/"):
            p = (STATIC / u.path.removeprefix("/static/")).resolve()
            if STATIC.resolve() not in p.parents or not p.is_file():
                self.send_error(404)
                return
            self.send(p.read_bytes(), TYPES.get(p.suffix, "application/octet-stream"))
        elif u.path == "/events":
            self.events()
        elif u.path == "/state":
            self.json(lambda: dict(seq=self.shared.seq, state=self.shared.state))
        elif u.path == "/meta":
            self.json(s.meta)
        elif u.path == "/filters":
            self.json(
                lambda: dict(
                    status=s.jobs.status(),
                    cached=s.jobs.cached(),
                    active=s.filter_meta(),
                    choices=dict(
                        echo=list(VF.ECHOES),
                        flatten=list(VF.FLATTEN),
                        defaults=VF.Params().as_dict(),
                    ),
                )
            )
        elif u.path == "/layer":
            self.json(
                lambda: dict(name=q["name"], values=round3(s.layers()[q["name"]]))
            )
        elif u.path == "/section":
            self.json(lambda: self.section(q))
        elif u.path.startswith("/axes/") and u.path.endswith(".tck"):
            _, _, k_, a_ = u.path.split("/")
            k, a = int(k_), a_.removesuffix(".tck")
            self.binary(lambda: SV.tck([np.asarray(self.axes[k][a], float)]))
        elif u.path == "/mesh":
            self.binary(
                lambda: s.mesh_payload(int(q.get("level", 0)), q.get("faces") == "1")
            )
        elif u.path == "/locate":
            mm = floats(q["mm"])
            self.json(lambda: dict(trace=s.locate(mm or [])))
        elif u.path == "/snap":
            mm_ = floats(q["mm"])
            self.json(lambda: s.snap(mm_ or []))
        elif u.path == "/vertex":
            self.json(lambda: s.vertex(int(q["v"])))
        elif u.path == "/streamlines":
            self.json(
                lambda: self.streamline_summary(
                    int(q["i"]), float(q.get("radius", 1.0))
                )
            )
        elif u.path.startswith("/tracts/") and u.path.endswith(".tck"):
            # the file name ends in .tck so that NiiVue reads it as tracks
            _, _, i_, r_, g_ = u.path.split("/")
            i, r, g = int(i_), float(r_), g_.removesuffix(".tck")
            self.binary(lambda: s.tracts(self.streamline_groups(i, r)[g]))
        elif u.path.startswith("/overview/") and u.path.endswith(".tck"):
            _, _, c_, e_ = u.path.split("/")
            c, e = int(c_), int(e_.removesuffix(".tck"))
            self.binary(lambda: s.overview_tracts(c, e))
        elif u.path == "/echo2.nii.gz":
            self.binary(s.volume_bytes)
        elif u.path == "/white.gii":
            self.binary(s.white_bytes)
        elif u.path == "/gm.gii":
            self.binary(s.gm_bytes)
        elif u.path == "/labels/list":
            self.json(lambda: self.review().rows())
        elif u.path.startswith("/labels/echo") and u.path.endswith(".nii.gz"):
            k = int(u.path.removeprefix("/labels/echo").removesuffix(".nii.gz"))
            self.binary(lambda: self.review().echo_bytes(k))
        elif u.path == "/labels.nii.gz":
            self.binary(lambda: self.review().overlay_bytes())
        else:
            self.send_error(404)

    def review(self) -> LabelReview:
        if self.labels is None:
            raise ValueError("no label changes: start the viewer with --labels")
        return self.labels

    def label_change(self, fn: Any, sender: str) -> dict[str, Any]:
        out: dict[str, Any] = fn()
        self.shared.update(sender, dict(labels_seq=out["seq"]))
        return out

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n)) if n else {}
        sender = str(req.get("from", ""))
        if self.path == "/labels/decide":
            self.json(
                lambda: self.label_change(
                    lambda: self.review().decide(int(req["id"]), str(req["status"])),
                    sender,
                )
            )
            return
        if self.path == "/labels/brush":
            self.json(
                lambda: self.label_change(
                    lambda: self.review().brush(
                        [float(x) for x in req["mm"]],
                        int(req.get("radius", 1)),
                        str(req["label"]),
                    ),
                    sender,
                )
            )
            return
        if self.path == "/filters/run":
            self.send_json(lambda: cast(Site, self.site).jobs.run(req["params"]))
            return
        if self.path == "/filters/use":
            key = req.get("key")

            def use() -> dict[str, Any]:
                out = cast(Site, self.site).use_filter(
                    None if key is None else str(key)
                )
                self.shared.update(sender, dict(filter=key))
                return out

            self.json(use)
            return
        if self.path != "/state":
            self.send_error(404)
            return
        seq = self.shared.update(sender, dict(req.get("patch", {})))
        self.send(json.dumps(dict(seq=seq)).encode(), "application/json")


def serve(
    site: Site | None,
    port: int,
    labels: LabelReview | None = None,
    white: WhiteView | None = None,
    volume: LazyVolume | None = None,
) -> ThreadingHTTPServer:
    """The server; without a site, labels (labels only) or white (a white surface
    alone) is needed. volume: the bricks of the volume pane (baillarger.viewer.bricks),
    if any."""
    if site is None and labels is None and white is None:
        raise ValueError("give a column volume, label changes or a white surface")
    Handler.site = site
    Handler.white = white
    Handler.volume = volume
    Handler.shared = Shared()
    Handler.groups = {}
    Handler.axes = {}
    Handler.labels = labels
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m baillarger.viewer",
        description="View a site's column data in the browser (127.0.0.1 only).",
    )
    ap.add_argument(
        "column_volume",
        type=Path,
        nargs="?",
        help="column_volume_<name>.npz; without it, labels only (--labels)",
    )
    ap.add_argument("--cache", type=Path, help="flat map and inflation (with columns)")
    ap.add_argument("--joint", type=Path, help="joint_<tag>.npz: the levels")
    ap.add_argument("--pia", type=Path, help="pia_<name>.npz: single-surface horizon")
    ap.add_argument("--priority", type=Path, help="edit_priority_<tag>.npz")
    ap.add_argument(
        "--echo",
        type=Path,
        help="echo 2 of the full grid (default: data); with --white, the volume pane's "
        "image (if the file is there)",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        help="pial_labels_<name>.npz (baillarger.pial_labels), saved in place",
    )
    ap.add_argument(
        "--white",
        type=Path,
        help="a white surface from the mesh command (..._white.surf.gii) with its "
        "overview, alone",
    )
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args(argv)
    t0 = time.time()
    site: Site | None = None
    review: LabelReview | None = None
    white: WhiteView | None = None
    if a.white is not None:
        if a.column_volume is not None or a.labels is not None:
            ap.error("--white is viewed alone: no column volume or --labels")
        if a.cache is None:
            ap.error("--cache is needed with --white")
        white = WhiteView(a.white, a.cache, log=print)
    elif a.column_volume is None:
        if a.labels is None:
            ap.error("give a column volume, or --labels to review the labels alone")
        review = LabelReview.standalone(a.labels, a.echo)
    else:
        if a.cache is None:
            ap.error("--cache is needed with a column volume")
        site = Site(
            a.column_volume, a.cache, a.joint, a.pia, a.priority, a.echo, log=print
        )
        if a.labels:
            review = LabelReview(a.labels, site.lo, site.shape, site.affine)
    volume: LazyVolume | None = None
    if site is not None:
        s_ = site
        volume = LazyVolume(
            lambda: ArrayPyramid.from_nifti(s_.echo_path, s_.lo, s_.shape),
            f"{s_.echo_path.name} (box)",
        )
    elif review is not None:
        r_ = review
        e_ = r_.echo or echo_path(2)
        volume = LazyVolume(
            lambda: ArrayPyramid.from_nifti(e_, r_.lo, r_.shape), f"{e_.name} (box)"
        )
    else:
        e_ = a.echo or echo_path(2)
        if e_.is_file():
            volume = LazyVolume(lambda: ArrayPyramid.from_nifti(e_), e_.name)
    srv = serve(site, a.port, review, white, volume)
    print(
        f"serving on http://127.0.0.1:{a.port}/ ({time.time() - t0:.0f} s)", flush=True
    )
    srv.serve_forever()
