"""The viewer's local web server (binds 127.0.0.1 only) and its shared state.

Any number of pages (windows, tabs, browsers) can be open at once; each shows the panes
named in its address (static/index.html, ?panes=...). They share one state, kept here:
two points, the crosshair (a world point: the slices of the volume pane) and the WM
selection (a vertex of the true white surface with the shown trace there: the
sections, the streamlines and the markers), the frame of the sections, and the colour
layer. A page sends its changes (POST /state) and every page receives every change as
it happens from the event stream (GET /events, Server-Sent Events), so a click in one
window moves the point in all of them. In each pane a click moves that pane's own
point (the slices: the crosshair; the surface and the map: the WM selection) and a
shift-click moves the other one.

Requests:
- GET / (the page), /static/<file>; /meta; /state; /events;
  /layer?name=...; /section?i=&half=[&right=&up=&level=] (right, up: 3 numbers each,
  comma-separated: the screen's axes on inflated level `level`);
  /mesh?level=k[&faces=1] (the white surface for the 3D pane, Site.mesh_payload);
  /locate?mm=x,y,z (the shown trace rooted nearest a world point);
  /snap?mm=x,y,z (the WM selection at the white-mesh vertex nearest a world point);
  /vertex?v=k (the WM selection at vertex k);
  /streamlines?i=&radius= (the traces from and to trace i, by kind and class);
  /tracts/<i>/<radius>/<kind>_<class>.tck (the traces of one group, as .tck);
  /overview/<class>/<every>.tck (every k-th shown trace of a class);
  /echo2.nii.gz, /white.gii, /gm.gii (world mm, for the volume pane);
  with --labels (baillarger.viewer.labels): /labels/list, /labels.nii.gz.
- POST /state {from, patch}: merge patch into the state; every page gets
  {from, seq, patch}.
- POST /labels/decide {from, id, status}, /labels/brush {from, mm, radius, label}:
  change the pial-label review and save it; every page gets the new labels_seq.
"""

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from baillarger.viewer.data import Site
from baillarger.viewer.labels import LabelReview

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
            crosshair=None, selection=None, frame=None, layer=None
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


class Handler(BaseHTTPRequestHandler):
    site: Site
    shared: Shared
    labels: LabelReview | None = None
    lock = threading.Lock()  # one site computation at a time
    groups: dict[tuple[int, float], dict[str, list[int]]] = {}

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

    def binary(self, fn: Any, ctype: str = "application/octet-stream") -> None:
        with self.lock:
            body = fn()
        self.send(body, ctype)

    def streamline_groups(self, i: int, radius: float) -> dict[str, list[int]]:
        key = (i, radius)
        if key not in self.groups:
            if len(self.groups) > 64:
                self.groups.clear()
            self.groups[key] = self.site.streamlines(i, radius)["_groups"]
        return self.groups[key]

    def streamline_summary(self, i: int, radius: float) -> dict[str, Any]:
        out = self.site.streamlines(i, radius)
        self.groups[(i, radius)] = out.pop("_groups")
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

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        s = self.site
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
        elif u.path == "/layer":
            self.json(
                lambda: dict(name=q["name"], values=round3(s.layers()[q["name"]]))
            )
        elif u.path == "/section":
            self.json(
                lambda: s.section_at(
                    int(q["i"]),
                    float(q.get("half", 8.0)),
                    floats(q.get("right")),
                    floats(q.get("up")),
                    int(q.get("level", 0)),
                )
            )
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
        if self.path != "/state":
            self.send_error(404)
            return
        seq = self.shared.update(sender, dict(req.get("patch", {})))
        self.send(json.dumps(dict(seq=seq)).encode(), "application/json")


def serve(
    site: Site, port: int, labels: LabelReview | None = None
) -> ThreadingHTTPServer:
    Handler.site = site
    Handler.shared = Shared()
    Handler.groups = {}
    Handler.labels = labels
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m baillarger.viewer",
        description="View a site's column data in the browser (127.0.0.1 only).",
    )
    ap.add_argument("column_volume", type=Path, help="column_volume_<name>.npz")
    ap.add_argument("--cache", type=Path, required=True, help="flat map and inflation")
    ap.add_argument("--joint", type=Path, help="joint_<tag>.npz: the levels")
    ap.add_argument("--pia", type=Path, help="pia_<name>.npz: single-surface horizon")
    ap.add_argument("--priority", type=Path, help="edit_priority_<tag>.npz")
    ap.add_argument("--echo", type=Path, help="echo 2 of the full grid (default: data)")
    ap.add_argument(
        "--labels",
        type=Path,
        help="pial_labels_<name>.npz (baillarger.pial_labels), saved in place",
    )
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args(argv)
    t0 = time.time()
    site = Site(a.column_volume, a.cache, a.joint, a.pia, a.priority, a.echo, log=print)
    review = None
    if a.labels:
        review = LabelReview(a.labels, site.lo, site.shape, site.affine)
    srv = serve(site, a.port, review)
    print(
        f"serving on http://127.0.0.1:{a.port}/ ({time.time() - t0:.0f} s)", flush=True
    )
    srv.serve_forever()
