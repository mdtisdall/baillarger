"""Σ at the steps of a run, as outlines on the volume pane (2026-11-23).

A directory of surfaces, one per step: files `*_step-<N>.surf.gii` (GIFTI: a
pointset in world mm and its triangles; prototypes/scripts/sigma_steps.py
writes them). The directory is read again at each list, so the steps of a
run that is still going appear as they are written.

Routes (route):
- /sigma/steps: the steps, in order: [{step, file, vertices, faces}].
- /sigma/outline?step=&axis=&mm=&box=: the segments where plane axis = mm
  (world mm) cuts the step's surface, in the box, encoded as /outline's
  (baillarger.viewer.outlines.encode). The surface is small (the overview's
  size), so every face whose bounding box meets the box is cut.

The last few steps' meshes are kept in memory.
"""

import argparse
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt

from baillarger.viewer import outlines as VO

type Float = npt.NDArray[np.float64]

STEP = re.compile(r"_step-(\d+)\.surf\.gii$")
#: meshes kept in memory
KEEP = 6


class SigmaSteps:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        self._mesh: OrderedDict[int, tuple[float, Float, Any, Float, Float]] = (
            OrderedDict()
        )

    def files(self) -> dict[int, Path]:
        out: dict[int, Path] = {}
        for p in sorted(self.root.glob("*_step-*.surf.gii")):
            m = STEP.search(p.name)
            if m:
                out[int(m.group(1))] = p
        return dict(sorted(out.items()))

    def mesh(self, step: int) -> tuple[Float, Any, Float, Float]:
        """Vertices (world mm), faces and the faces' bounding boxes of a step
        (read again if its file changed)."""
        files = self.files()
        if step not in files:
            raise KeyError(step)
        p = files[step]
        mt = p.stat().st_mtime
        with self._lock:
            hit = self._mesh.get(step)
            if hit is not None and hit[0] == mt:
                self._mesh.move_to_end(step)
                return hit[1], hit[2], hit[3], hit[4]
        g: Any = nib.load(p)
        v = np.asarray(g.darrays[0].data, np.float64)
        f = np.asarray(g.darrays[1].data, np.int64)
        c = v[f]
        lo, hi = c.min(1), c.max(1)
        with self._lock:
            self._mesh[step] = (mt, v, f, lo, hi)
            while len(self._mesh) > KEEP:
                self._mesh.popitem(last=False)
        return v, f, lo, hi

    def list(self) -> list[dict[str, Any]]:
        out = []
        for k, p in self.files().items():
            row: dict[str, Any] = dict(step=k, file=p.name)
            with self._lock:
                hit = self._mesh.get(k)
            if hit is not None:
                row.update(vertices=len(hit[1]), faces=len(hit[2]))
            out.append(row)
        return out

    def cut(self, step: int, axis: int, mm: float, lo: Any, hi: Any) -> VO.Outline:
        v, f, flo, fhi = self.mesh(step)
        lo_ = np.asarray(lo, np.float64).copy()
        hi_ = np.asarray(hi, np.float64).copy()
        lo_[axis] = hi_[axis] = mm
        near = np.all(flo <= hi_, axis=1) & np.all(fhi >= lo_, axis=1)
        seg = VO.segments_in_box(VO.plane_cut(v, f[near], axis, mm), axis, lo_, hi_)
        return VO.Outline(seg.astype(np.float32), "sigma", int(near.sum()), len(f))


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--sigma-dir",
        type=Path,
        help="a directory of Σ surfaces by step (*_step-<N>.surf.gii, world mm): "
        "their outlines on the volume pane, with a step slider",
    )


def for_args(a: argparse.Namespace) -> SigmaSteps | None:
    return SigmaSteps(a.sigma_dir) if getattr(a, "sigma_dir", None) else None


def route(h: Any, path: str, q: dict[str, str]) -> bool:
    """Serve a /sigma route for handler h (its `sigma`); False for any other
    path. 404: no directory or no such step; 400: a bad parameter."""
    if not path.startswith("/sigma/"):
        return False
    s: SigmaSteps | None = getattr(h, "sigma", None)
    if s is None:
        h.send_error(404)
        return True
    if path == "/sigma/steps":
        h.json(s.list)
        return True
    if path != "/sigma/outline":
        h.send_error(404)
        return True
    try:
        step = int(q.get("step", ""))
        axis = int(q.get("axis", ""))
        if axis not in (0, 1, 2):
            raise ValueError(f"axis {axis} is not 0, 1 or 2")
        mm = VO.parse_floats(q.get("mm", ""), 1, "mm")[0]
        box = VO.parse_floats(q.get("box", ""), 6, "box")
        o = s.cut(step, axis, mm, box[:3], box[3:])
    except KeyError:
        h.send_error(404)
        return True
    except ValueError as exc:
        h.send(str(exc).encode(), "text/plain", 400)
        return True
    body = VO.encode(o)
    h.send_response(200)
    h.send_header("Content-Type", "application/octet-stream")
    h.send_header("X-Faces-Read", str(o.faces_read))
    h.send_header("X-Segments", str(len(o.segments)))
    h.send_header("Content-Length", str(len(body)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(body)
    return True
