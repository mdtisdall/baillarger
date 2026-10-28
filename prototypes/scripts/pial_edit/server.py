"""Pial editor, spike (plan: reports/Pial editing and edit priority.md, §6 step 2).

A local server for the Niivue page in static/: it serves one volume and triangle
meshes (legacy VTK, world mm), and a stand-in for the re-solve: POST /bump moves
the vertices of one mesh within a few mm of a point along their normals by a
Gaussian bump, and returns all vertex positions. The bump only measures the
round trip (server, transfer, Niivue's buffer update); it is not the correction
field.

Binary mesh format (GET /mesh/<i>): uint32 nv, uint32 nt, float32 pts (nv x 3,
world mm), uint32 tris (nt x 3). POST /bump returns float32 pts only.

Usage: server.py <port> <volume.nii.gz> <mesh.vtk> [<mesh.vtk> ...]
Binds 127.0.0.1 only.
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

STATIC = Path(__file__).parent / "static"


def read_vtk(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Points (world mm) and triangles of a legacy ASCII VTK polydata file as
    baillarger.mesh.write_vtk writes it."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("POINTS"))
    nv = int(lines[i].split()[1])
    pts = np.loadtxt(lines[i + 1 : i + 1 + nv], dtype=np.float32).reshape(nv, 3)
    j = next(k for k, ln in enumerate(lines) if ln.startswith("POLYGONS"))
    nt = int(lines[j].split()[1])
    tri = np.loadtxt(lines[j + 1 : j + 1 + nt], dtype=np.int64).reshape(nt, 4)
    assert np.all(tri[:, 0] == 3), "not a triangle mesh"
    return pts, tri[:, 1:].astype(np.uint32)


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]).astype(np.float64)
    n = np.zeros(v.shape, np.float64)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


class Mesh:
    def __init__(self, path: str):
        t = time.time()
        cache = Path(path).with_suffix(".spike.npz")
        if cache.exists() and cache.stat().st_mtime > Path(path).stat().st_mtime:
            z = np.load(cache)
            self.v0, self.f = z["v"], z["f"]
        else:
            self.v0, self.f = read_vtk(path)
            np.savez(cache, v=self.v0, f=self.f)
        self.n = vertex_normals(self.v0, self.f)
        self.v = self.v0.copy()
        self.tree = cKDTree(self.v0)
        self.name = Path(path).name
        print(f"{self.name}: {len(self.v):,} vertices, {len(self.f):,} triangles, "
              f"{time.time() - t:.1f} s", flush=True)

    def payload(self) -> bytes:
        head = np.array([len(self.v), len(self.f)], np.uint32).tobytes()
        return head + self.v.astype(np.float32).tobytes() + self.f.astype(np.uint32).tobytes()

    def bump(self, x: np.ndarray, amp: float, rho: float) -> tuple[float, np.ndarray]:
        t = time.perf_counter()
        idx = np.asarray(self.tree.query_ball_point(x, 3.0 * rho), np.int64)
        if idx.size:
            d2 = np.sum((self.v0[idx] - x) ** 2, 1)
            self.v[idx] += (amp * np.exp(-0.5 * d2 / rho**2))[:, None] * self.n[idx]
        return time.perf_counter() - t, idx


class Handler(BaseHTTPRequestHandler):
    volume: Path
    meshes: list[Mesh]

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, val in (extra or {}).items():
            self.send_header(k, val)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif p == "/volume.nii.gz":
            self._send(self.volume.read_bytes(), "application/octet-stream")
        elif p == "/meshes":
            self._send(json.dumps([m.name for m in self.meshes]).encode(), "application/json")
        elif p.startswith("/mesh/"):
            self._send(self.meshes[int(p.rsplit("/", 1)[1])].payload(), "application/octet-stream")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/bump":
            self.send_error(404)
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        m = self.meshes[int(req.get("mesh", 0))]
        if req.get("reset"):
            m.v[:] = m.v0
            dt, idx = 0.0, np.arange(len(m.v))
        else:
            dt, idx = m.bump(np.array(req["mm"], float), float(req.get("amp", 0.3)), float(req.get("rho", 1.0)))
        t = time.perf_counter()
        if req.get("sparse"):
            # uint32 count, uint32 indices, float32 positions (count x 3)
            body = (np.array([idx.size], np.uint32).tobytes() + idx.astype(np.uint32).tobytes()
                    + m.v[idx].astype(np.float32).tobytes())
        else:
            body = m.v.astype(np.float32).tobytes()
        self._send(body, "application/octet-stream",
                   {"X-Solve-Ms": f"{1e3 * dt:.2f}", "X-Pack-Ms": f"{1e3 * (time.perf_counter() - t):.2f}"})


def main() -> None:
    port, vol, *paths = sys.argv[1:]
    Handler.volume = Path(vol)
    Handler.meshes = [Mesh(p) for p in paths]
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    print(f"serving on http://127.0.0.1:{port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
