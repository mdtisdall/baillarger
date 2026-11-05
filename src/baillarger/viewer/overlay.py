"""Per-vertex scalar maps on the viewer's overview mesh (GET /overlay/...).

The user gives GIFTI files (.func.gii / .shape.gii) of scalar maps on the white
surface: each data array is one map, one value per TRUE vertex of the surface
(13.8 M for a whole hemisphere). The viewer draws the decimated overview mesh,
whose vertices are true vertices and whose faces each carry a set of true
vertices (baillarger.white_overview). Nothing here knows what a map means (a
normal's angle, a thickness, ...): it is any per-vertex number.

Three things are served, all computed on first use and cached:

- the map on the overview vertices (`aggregate`): for each overview vertex the
  max (or min) of its own value and the values that its faces carry, so a thin
  feature among the true vertices that the decimation dropped still shows. NaN
  is ignored; it stays NaN only where every value is NaN. Raw float32 bytes.
- a ranked list of sites (`sites`): the highest (or lowest) values, at least
  `spacing` mm apart, each with its world point, voxel and an overview vertex
  to put a marker on.
- the values at given true vertices (`at`), for the page's readout.

The maps are memory-mapped and read when a route first needs them, in chunks of
faces or vertices, so no route holds more than a few chunks beyond one map.
"""

import argparse
import json
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiImage

from baillarger.white_overview import WhiteOverview

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]

#: names that the routes use themselves: no file may be called so
RESERVED = ("at", "meta")
#: faces per chunk of the aggregate; candidates per chunk of ranked_sites
FACE_CHUNK = 500_000
SITE_CHUNK = 4096
#: carried entries per chunk when the carrying faces of the sites are looked up
CARRY_CHUNK = 4_000_000


def ranked_sites(
    values: npt.ArrayLike,
    points_mm: npt.NDArray[np.floating[Any]],
    n: int,
    spacing: float,
    order: str = "high",
) -> Int:
    """Greedy sites: the largest finite value (order "low": the smallest) is a
    site; every vertex within `spacing` mm of it (distance <= spacing) is
    dropped; repeat until n sites or no candidate is left. NaN is never a site;
    ties go to the lower vertex id. The finite values are sorted once and walked
    in chunks; each chunk is tested against a KD-tree of the sites kept so far
    (at most n points) and its own conflicts are resolved one by one."""
    from scipy.spatial import cKDTree

    val = np.asarray(values)
    cand = np.flatnonzero(np.isfinite(val))
    key = -val[cand] if order == "high" else val[cand]
    cand = cand[np.argsort(key, kind="stable")]
    kept: list[int] = []
    tree: cKDTree | None = None
    for c0 in range(0, len(cand), SITE_CHUNK):
        if len(kept) >= n:
            break
        ids = cand[c0 : c0 + SITE_CHUNK]
        xyz = np.asarray(points_mm[ids], np.float64)
        if tree is not None:
            near = tree.query_ball_point(xyz, spacing, return_length=True)
            ids, xyz = ids[near == 0], xyz[near == 0]
        new = len(kept)
        mine: list[Float] = []  # the sites of this chunk
        for v, p in zip(ids, xyz, strict=True):
            if mine and np.any(np.linalg.norm(np.asarray(mine) - p, axis=1) <= spacing):
                continue
            kept.append(int(v))
            mine.append(p)
            if len(kept) >= n:
                break
        if len(kept) > new:
            tree = cKDTree(np.asarray(points_mm[np.asarray(kept)], np.float64))
    return np.asarray(kept, np.int64)


def _strip(name: str) -> str:
    for ext in (".func.gii", ".shape.gii", ".gii"):
        if name.endswith(ext):
            return name.removesuffix(ext)
    return name


class Overlays:
    """The maps of the given GIFTI files on a white surface's overview (ov); see
    the module docstring. pts_mm: the surface's vertices (scanner mm; may be
    memory-mapped); n_true: how many true vertices it has."""

    def __init__(
        self,
        paths: list[Path] | list[str],
        ov: WhiteOverview,
        pts_mm: npt.NDArray[np.floating[Any]],
        n_true: int,
    ) -> None:
        self.ov = ov
        self.pts = pts_mm
        self.n_true = int(n_true)
        self.maps: dict[str, dict[str, Any]] = {}  # file -> map name -> GiftiDataArray
        for p in paths:
            p = Path(p)
            key = _strip(p.name)
            if key in RESERVED:
                raise ValueError(
                    f"{p}: the file name {key!r} is reserved by the routes"
                )
            if key in self.maps:
                raise ValueError(f"{p}: two overlay files are called {key!r}")
            try:
                img = GiftiImage.from_filename(str(p), mmap=True)
            except Exception as exc:
                raise ValueError(f"{p}: not a readable GIFTI file ({exc})") from exc
            maps: dict[str, Any] = {}
            for i, d in enumerate(img.darrays):
                name = d.meta.get("Name") or f"map{i}"
                shape = tuple(d.data.shape)
                if shape not in ((self.n_true,), (self.n_true, 1)):
                    raise ValueError(
                        f"{p}: map {name!r} has shape {shape}; the white surface "
                        f"has {self.n_true} vertices, so the shape must be "
                        f"({self.n_true},) or ({self.n_true}, 1)"
                    )
                if name in maps:
                    raise ValueError(f"{p}: two maps are called {name!r}")
                maps[name] = d
            self.maps[key] = maps
        self._meta: list[dict[str, Any]] | None = None
        self._agg: dict[tuple[str, str, str], npt.NDArray[np.float32]] = {}
        self._sites: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._inv = np.linalg.inv(ov.affine)

    def values(self, file: str, name: str) -> npt.NDArray[Any]:
        """A map's values (n_true), read from the memory map."""
        d = self.maps[file][name].data
        return np.asarray(d).reshape(-1)

    def has(self, file: str, name: str) -> bool:
        return name in self.maps.get(file, {})

    def meta(self) -> list[dict[str, Any]]:
        if self._meta is None:
            out = []
            for file, maps in self.maps.items():
                for name in maps:
                    v = self.values(file, name)
                    fin = v[np.isfinite(v)]
                    if len(fin):
                        p1, p50, p99 = (
                            float(x) for x in np.percentile(fin, [1, 50, 99])
                        )
                    else:
                        p1 = p50 = p99 = None  # type: ignore[assignment]
                    out.append(
                        dict(
                            file=file,
                            map=name,
                            p1=p1,
                            p50=p50,
                            p99=p99,
                            finite=int(len(fin)),
                            n=self.n_true,
                        )
                    )
            self._meta = out
        return self._meta

    def aggregate(self, file: str, name: str, agg: str) -> npt.NDArray[np.float32]:
        """The map on the overview vertices: max or min (NaN ignored) of the
        vertex's own value and the values carried by every face that touches it."""
        if agg not in ("max", "min"):
            raise ValueError(f"agg must be max or min, not {agg!r}")
        key = (file, name, agg)
        if key in self._agg:
            return self._agg[key]
        red = np.fmax if agg == "max" else np.fmin
        ov = self.ov
        a = self.values(file, name)
        nf = len(ov.f)
        fval = np.full(nf, np.nan, np.float32)
        for f0 in range(0, nf, FACE_CHUNK):
            f1 = min(f0 + FACE_CHUNK, nf)
            ids, _ = ov.carried(np.arange(f0, f1))  # in the faces' order
            counts = np.diff(ov.ptr[f0 : f1 + 1])
            full = counts > 0
            if not full.any():
                continue
            starts = (np.cumsum(counts) - counts)[full]
            fval[f0:f1][full] = red.reduceat(np.asarray(a[ids], np.float32), starts)
        out = np.asarray(a[ov.vertex_ids], np.float32)
        for k in range(3):
            red.at(out, ov.f[:, k], fval)
        self._agg[key] = out
        return out

    def voxels(self, true_vertices: Int) -> Int:
        """The voxel of the grid nearest each true vertex (as WhiteView.voxel)."""
        x = np.asarray(self.pts[true_vertices], np.float64)
        h = np.c_[x, np.ones(len(x))] @ self._inv.T
        r: Int = np.round(h[:, :3]).astype(np.int64)
        return r

    def markers(self, sites: Int) -> list[int | None]:
        """For each site (a true vertex): the overview vertex to mark it with: the
        one that is the site, else the nearest (mm) corner of a face that carries
        it. The faces are found by one pass over the carry file in chunks; the
        memory is a chunk (CARRY_CHUNK entries) and the sites' lists, not an
        inverse of the carry."""
        ov = self.ov
        srt = np.sort(sites)
        best: dict[int, tuple[float, int]] = {}  # site -> (distance, overview vertex)
        vid = ov.vertex_ids
        byid = np.argsort(vid)
        vsorted = vid[byid]
        pos = np.minimum(np.searchsorted(vsorted, srt), len(vsorted) - 1)
        result: dict[int, int] = {
            int(srt[j]): int(byid[pos[j]]) for j in np.flatnonzero(vsorted[pos] == srt)
        }
        n = len(ov.carried_ids)
        for e0 in range(0, n, CARRY_CHUNK):
            ids = np.asarray(ov.carried_ids[e0 : e0 + CARRY_CHUNK], np.int64)
            pos = np.minimum(np.searchsorted(srt, ids), len(srt) - 1)
            hit = np.flatnonzero(srt[pos] == ids)
            if not len(hit):
                continue
            faces = np.searchsorted(ov.ptr, e0 + hit, side="right") - 1
            for h, fc in zip(hit, faces, strict=True):
                t = int(ids[h])
                if t in result:
                    continue
                corners = ov.f[fc]
                d = np.linalg.norm(
                    np.asarray(self.pts[vid[corners]], np.float64)
                    - np.asarray(self.pts[t], np.float64),
                    axis=1,
                )
                j = int(np.argmin(d))
                if t not in best or d[j] < best[t][0]:
                    best[t] = (float(d[j]), int(corners[j]))
        for t, (_, k) in best.items():
            result.setdefault(t, k)
        return [result.get(int(t)) for t in sites]

    def sites(
        self,
        file: str,
        name: str,
        n: int = 200,
        spacing: float = 1.0,
        order: str = "high",
    ) -> list[dict[str, Any]]:
        key = (file, name, n, spacing, order)
        if key not in self._sites:
            t = ranked_sites(self.values(file, name), self.pts, n, spacing, order)
            val = self.values(file, name)
            vox = self.voxels(t)
            mk = self.markers(t)
            self._sites[key] = [
                dict(
                    rank=i + 1,
                    vertex=int(v),
                    value=float(val[v]),
                    world_mm=[round(float(x), 3) for x in self.pts[v]],
                    voxel=[int(x) for x in vox[i]],
                    overview_vertex=mk[i],
                )
                for i, v in enumerate(t)
            ]
            if len(self._sites) > 32:
                self._sites.pop(next(iter(self._sites)))
        return self._sites[key]

    def check(self, vertices: list[int]) -> Int:
        """The vertices as an array; ValueError if one is not a true vertex."""
        v = np.asarray(vertices, np.int64)
        if len(v) and (v.min() < 0 or v.max() >= self.n_true):
            raise ValueError(f"vertex out of range 0..{self.n_true - 1}")
        return v

    def at(self, vertices: list[int]) -> dict[str, Any]:
        """The value of every map at the true vertices `vertices` (NaN: None)."""
        v = self.check(vertices)
        out: dict[str, dict[str, list[float | None]]] = {}
        for file, maps in self.maps.items():
            out[file] = {}
            for name in maps:
                x = np.asarray(self.values(file, name)[v], np.float64)
                out[file][name] = [float(a) if np.isfinite(a) else None for a in x]
        return dict(v=[int(a) for a in v], values=out)


def _error(h: Any, msg: str, code: int = 400) -> None:
    h.send(json.dumps(dict(error=msg)).encode(), "application/json", code)


def route(h: Any, path: str, q: dict[str, str]) -> bool:
    """Serve an /overlay route for handler h (its `overlays`); False for any
    other path. 404: no overlays or no such file or map; 400: a bad parameter."""
    if not (path == "/overlay" or path.startswith("/overlay/")):
        return False
    ov: Overlays | None = getattr(h, "overlays", None)
    if ov is None:
        h.send_error(404)
        return True
    parts = [urllib.parse.unquote(s) for s in path.split("/")[2:]]
    if parts == ["meta"]:
        h.json(lambda: _meta_json(ov))
    elif parts == ["at"]:
        try:
            vs = [int(x) for x in q.get("v", "").split(",") if x != ""]
            ov.check(vs)
            h.json(lambda: ov.at(vs))
        except ValueError as exc:
            _error(h, str(exc))
    elif len(parts) in (2, 3) and ov.has(parts[0], parts[1]):
        file, name = parts[0], parts[1]
        try:
            if len(parts) == 2:
                agg = q.get("agg", "max")
                if agg not in ("max", "min"):
                    raise ValueError(f"agg must be max or min, not {agg!r}")
                h.binary(lambda: ov.aggregate(file, name, agg).astype("<f4").tobytes())
            elif parts[2] == "sites":
                n = int(q.get("n", 200))
                spacing = float(q.get("spacing", 1.0))
                order = q.get("order", "high")
                if order not in ("high", "low"):
                    raise ValueError(f"order must be high or low, not {order!r}")
                if n < 1 or not spacing >= 0:
                    raise ValueError("n must be at least 1 and spacing 0 or more")
                h.json(lambda: dict(sites=ov.sites(file, name, n, spacing, order)))
            else:
                h.send_error(404)
        except ValueError as exc:
            _error(h, str(exc))
    else:
        h.send_error(404)
    return True


def _meta_json(ov: Overlays) -> dict[str, Any]:
    files: dict[str, list[dict[str, Any]]] = {f: [] for f in ov.maps}
    for m in ov.meta():
        files[m["file"]].append({k: v for k, v in m.items() if k != "file"})
    return dict(files=[dict(file=f, maps=m) for f, m in files.items()])


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--overlay",
        action="append",
        type=Path,
        help="a GIFTI file (.func.gii/.shape.gii) of scalar maps on the white surface, "
        "one value per true vertex; repeatable",
    )


def for_white(a: argparse.Namespace, white: Any) -> Overlays | None:
    """The overlays of --white mode, if --overlay is given."""
    if not a.overlay:
        return None
    return Overlays(a.overlay, white.ov, white.pts, len(white.W.v))
