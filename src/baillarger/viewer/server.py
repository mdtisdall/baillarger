"""The viewer's local web server (binds 127.0.0.1 only) and its shared state.

The viewer shows a white surface from baillarger.white_surf (--white, with its overview
from baillarger.white_overview) and the published outputs on it. It reads no prototype
files.

Any number of pages (windows, tabs, browsers) can be open at once; each shows the panes
named in its address (static/index.html, ?panes=...). They share one state, kept here:
two points, the crosshair (a world point: the slices of the volume pane) and the WM
selection (a vertex of the white surface's overview and the true vertex it is), and the
keys that the panes add (for example `columns`). A page sends its changes (POST /state)
and every page receives every change as it happens from the event stream (GET /events,
Server-Sent Events), so a click in one window moves the point in all of them. In each
pane a click moves that pane's own point (the slices: the crosshair; the surface: the
WM selection) and a shift-click moves the other one.

Requests:
- GET / (the page), /static/<file>; /state; /events.
- The white surface (baillarger.viewer.white_view): /meta; /mesh?level=k[&faces=1]
  (the overview for the 3D pane); /vertex?v=k (the WM selection at overview vertex k);
  /vertex/true?v=t (the WM selection at true vertex t; 400 if t is not one);
  /vertex/overview?v=t1,t2,... (the overview vertex of each true vertex, as
  /vertex/true gives it);
  /snap?mm=x,y,z (the WM selection at the overview vertex nearest a world point);
  /white.gii.
- The volume as bricks (baillarger.viewer.bricks), the whole echo 2: /volume (its
  name, without reading it; 404: none), /volume/meta (the levels' shapes and affines,
  the brick size, the dtype, the default window: read on first use),
  /volume/bricks?level=L&b=i,j,k;i,j,k;... (those bricks' bytes, one after another;
  gzip when the page accepts it).
- Named volumes (baillarger.viewer.volumes: echo, echo1 to echo3, labels, input,
  wm, fixes):
  /volume/list, /volume/<name>/meta, /volume/<name>/bricks?level=&b=.
- The stream 1 review (baillarger.viewer.review; with --sites, --log and
  --edits, or paint only with --edits alone):
  GET /review/queue, /review/next, /review/preview?job=, /review/pending;
  POST /review/session, /review/skip, /review/preview, /review/answer,
  /review/scribble, /review/paint, /review/undo.
- Surface outlines (baillarger.viewer.outlines):
  /outline?axis=&mm=&box= (float32 segment ends, world mm).
- The columns pane (baillarger.viewer.columns.route): /columns/meta,
  /columns?v=&radius=..., /columns/profiles?..., /columns/<key>.tck.
- Per-vertex scalar maps (baillarger.viewer.overlay; with --overlay): /overlay/meta,
  /overlay/<file>/<map>?agg=max|min (float32 on the overview vertices),
  /overlay/<file>/<map>/sites?n=&spacing=&order=, /overlay/at?v=1,2,3.
- POST /state {from, patch}: merge patch into the state; every page gets
  {from, seq, patch}.
"""

import argparse
import gzip
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from baillarger.data import echo_path
from baillarger.viewer import columns as VC
from baillarger.viewer import outlines as VO
from baillarger.viewer import overlay as VOL
from baillarger.viewer import review as VR
from baillarger.viewer import volumes as VV
from baillarger.viewer.bricks import ArrayPyramid, LazyVolume, nifti
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


class Handler(BaseHTTPRequestHandler):
    shared: Shared
    white: WhiteView | None = None
    volume: LazyVolume | None = None
    volumes: VV.Volumes | None = None  # named volumes (baillarger.viewer.volumes)
    outlines: VO.Outlines | None = None  # surface outlines (baillarger.viewer.outlines)
    columns: VC.Columns | None = None
    wmreview: VR.Review | None = None  # the stream 1 review (baillarger.viewer.review)
    overlays: VOL.Overlays | None = None  # per-vertex maps (baillarger.viewer.overlay)
    lock = threading.Lock()  # one computation at a time

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
        """fn's answer as JSON, under the lock (one computation at a time)."""
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
        """As json, without the lock: for calls that only read."""
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

    def outline(self, q: dict[str, str]) -> None:
        """/outline (baillarger.viewer.outlines.outline_response)."""
        if self.outlines is None:
            self.send_error(404)
            return
        try:
            body, hdr = VO.outline_response(self.outlines, q)
        except ValueError as exc:
            self.send(str(exc).encode(), "text/plain", 400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        for k, v in hdr.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path.startswith("/volume/") and VV.route(self, u.path, q):
            return
        if u.path == "/volume" or u.path.startswith("/volume/"):
            self.volume_route(u.path, q)
            return
        if u.path == "/outline":
            self.outline(q)
            return
        if u.path == "/columns" or u.path.startswith("/columns/"):
            VC.route(self, u.path, q)
            return
        if VR.route(self, "GET", u.path, q, {}):
            return
        if VOL.route(self, u.path, q):
            return
        w = self.white
        if w is None:
            self.send_error(404)
            return
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
            self.json(w.meta)
        elif u.path == "/mesh":
            lv, fc = int(q.get("level", 0)), q.get("faces") == "1"
            self.binary(lambda: w.mesh_payload(lv, fc))
        elif u.path == "/vertex":
            self.json(lambda: w.vertex(int(q["v"])))
        elif u.path == "/vertex/overview":
            vq = q.get("v", "")
            self.send_json(
                lambda: w.overview_vertices([int(x) for x in vq.split(",") if x])
            )
        elif u.path == "/vertex/true":
            self.send_json(lambda: w.true_vertex(int(q["v"])))
        elif u.path == "/snap":
            mm_ = floats(q["mm"])
            self.json(lambda: w.snap(mm_ or []))
        elif u.path == "/white.gii":
            self.binary(w.white_bytes)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n)) if n else {}
        sender = str(req.get("from", ""))
        if VR.route(self, "POST", self.path, {}, req):
            return
        if self.path != "/state":
            self.send_error(404)
            return
        seq = self.shared.update(sender, dict(req.get("patch", {})))
        self.send(json.dumps(dict(seq=seq)).encode(), "application/json")


def serve(
    site: None,
    port: int,
    *,
    white: WhiteView | None,
    volume: LazyVolume | None = None,
    volumes: VV.Volumes | None = None,
    outlines: VO.Outlines | None = None,
    columns: VC.Columns | None = None,
    wmreview: VR.Review | None = None,
    overlays: VOL.Overlays | None = None,
) -> ThreadingHTTPServer:
    """The server on a white surface (white). volume: the bricks of the volume pane
    (baillarger.viewer.bricks); volumes, outlines, columns: the other panes' data, if
    any. site must be None: the site mode is gone (phase 4.3); the parameter stays so
    that the callers' form serve(None, port, white=...) does not change."""
    if site is not None or white is None:
        raise ValueError("give a white surface (the site mode is gone)")
    Handler.white = white
    Handler.volume = volume
    Handler.volumes = volumes
    Handler.outlines = outlines
    Handler.columns = columns
    Handler.wmreview = wmreview
    Handler.overlays = overlays
    Handler.shared = Shared()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m baillarger.viewer",
        description="View a white surface and the published outputs on it in the "
        "browser (127.0.0.1 only).",
    )
    ap.add_argument(
        "--white",
        type=Path,
        required=True,
        help="a white surface from baillarger.white_surf (..._white.surf.gii) with "
        "its overview from baillarger.white_overview",
    )
    ap.add_argument("--cache", type=Path, required=True, help="the inflation's cache")
    ap.add_argument(
        "--echo",
        type=Path,
        help="the volume pane's default image (default: echo 2 of --echo-dir, if the "
        "file is there)",
    )
    VV.add_arguments(ap)
    ap.add_argument(
        "--echo-dir",
        type=Path,
        help="the directory of echoes 1-3 (.nii, else .nii.gz): the volume pane's "
        "echo choice and the columns pane's profiles (default: data)",
    )
    VR.add_arguments(ap)
    VOL.add_arguments(ap)
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args(argv)
    t0 = time.time()
    white = WhiteView(a.white, a.cache, log=print)
    columns = VC.Columns(
        a.white,
        a.dseg or VC.dseg_for(a.white, white.W.sidecar),
        white.W.v,
        white.W.f,
        VC.echo_files(a.echo_dir or echo_path(1).parent),
        log=print,
    )
    volume: LazyVolume | None = None
    # echoes 1 to 3 in --echo-dir (the .nii, else the .nii.gz); the default volume:
    # --echo, else echo 2 of them
    edir = a.echo_dir or echo_path(1).parent
    echoes = {k: VC.echo_file(k, edir) for k in (1, 2, 3)}
    echoes = {k: p for k, p in echoes.items() if p.is_file()}
    e_ = a.echo or echoes.get(2) or echo_path(2)
    if e_.is_file():
        volume = LazyVolume(lambda: ArrayPyramid.from_nifti(e_), e_.name)
    dt = str(nifti(e_).get_data_dtype()) if volume is not None else ""
    volumes = VV.for_white(
        white.surf,
        white.W.sidecar["Sources"],
        volume,
        dt,
        a.dseg,
        a.fixes,
        filled=white.filled,
        echoes=echoes,
        echo_path=e_,
    )
    outlines = VO.Outlines.from_white_view(white)
    lab = volumes.items["labels"]
    wmreview: VR.Review | None = None
    try:
        wmreview = VR.for_white(
            a,
            white.surf,
            VV.white_paths(white.surf, white.W.sidecar["Sources"], a.dseg, a.fixes)[0],
            lambda: lab.volume.source().arrays[0],  # type: ignore[attr-defined]
            white.W.affine,
        )
    except ValueError as exc:
        ap.error(str(exc))
    overlays: VOL.Overlays | None = None
    try:
        overlays = VOL.for_white(a, white)
    except ValueError as exc:
        ap.error(str(exc))
    srv = serve(
        None,
        a.port,
        white=white,
        volume=volume,
        volumes=volumes,
        outlines=outlines,
        columns=columns,
        wmreview=wmreview,
        overlays=overlays,
    )
    print(
        f"serving on http://127.0.0.1:{a.port}/ ({time.time() - t0:.0f} s)", flush=True
    )
    srv.serve_forever()
