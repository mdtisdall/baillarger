"""The columns pane's server side: the columns near a vertex of the white
surface, from the published samplers (ray-sampler/1, column-volume/1), for
what is on screen, anywhere on the hemisphere.

A request names a true vertex v of the white surface and a radius r (mm). The
columns of the request are the columns whose roots lie within r of v along the
surface (the graph of baillarger.mesh_graph: the mesh edges and the edges
across neighbouring triangles, in mm), in the order that the ray sampler gives
them (its owned columns, then its halo columns, each in vertex order), every
n-th of them.

How a request is computed (nothing scales with the whole mesh or volume):

1. The superblock S of v at LEVEL (the superblock of the block that owns v).
2. The local graph of S with a halo of MAX_RADIUS (mesh_graph.local_graph):
   only the faces of the blocks near S. Its distances from a root of S are
   exact up to the halo, so a Dijkstra from v, limited to r, gives the roots.
   The graph is kept for the last GRAPHS superblocks. On the hemisphere it
   takes 2.5-4 s to build (about 1.4 M faces of 27 blocks; a halo of 2 mm
   meets the same blocks), so the first request on a superblock runs as a
   job; the next ones take 0.05-0.5 s.
3. One call of the ray sampler on S with a halo of r: every root within r of
   v is in its halo box (v lies in the cells of S, and a root within r along
   the surface is within r in space). The call samples each selected column
   between the requested extents and every other column with extents 0 (one
   sample), so that the samples cost only what the response holds. The
   extents of a column do not change its samples, intervals or class
   (ray-sampler/1, "Samples"), so each column is the sampler's own, bit for
   bit.
4. For the profiles, the column-volume sampler on the selected columns only.

A request that is not in the cache runs as a job (baillarger.viewer.jobs, one
at a time, in a thread). The server waits WAIT_S for it, then answers
"running" and the page asks again. The last CACHE results are kept in memory,
keyed by the request and the sources' SHA-256.

The binary answers (encode): a uint32 header length, a JSON header padded to
8 bytes, then the arrays, each padded to 8 bytes. The header lists each
array's name, dtype, shape and offset (from the start of the body).
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from baillarger import rays as R
from baillarger.bids import file_sha256
from baillarger.blocks import Source, open_volume
from baillarger.column_volume import ColumnVolume, ColumnVolumeSampler
from baillarger.mesh_graph import BlockIndex, LocalGraph, local_graph
from baillarger.viewer import section as VSec
from baillarger.viewer.jobs import Jobs
from baillarger.viewer.tck import tck

type Float = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Arr = npt.NDArray[Any]

#: the superblock level of a request: one block (64 cells, about 10 mm at 160 µm)
LEVEL = 0
#: the largest radius of a request, mm: the halo of the local graphs
MAX_RADIUS = 5.0
#: the most columns in one request. At the full caps and the default step a
#: column has 251 samples of 16 bytes (s and position) and 3 echoes of 4 bytes,
#: so 4000 columns make about 28 MB: the most that a page loads and draws
#: without a pause. Use `every` for a larger radius.
MAX_COLUMNS = 4000
#: how long a request waits for its job before it answers "running", s
WAIT_S = 2.0
#: the results kept in memory, and the local graphs
CACHE = 8
GRAPHS = 4


@dataclass(frozen=True)
class Request:
    """One request of the columns pane (see the module docstring)."""

    vertex: int
    radius: float
    method: str = "normal"
    every: int = 1
    e_in: float = R.CAP_IN
    e_out: float = R.CAP_OUT
    step: float = R.STEP
    #: only these columns (true vertices, comma-separated; "" for all): the
    #: columns of a profile section, when the whole request is over the cap
    cols: str = ""

    @classmethod
    def parse(cls, q: dict[str, str]) -> "Request":
        """A request from the query of a URL: v, radius, and optionally
        method, every, e_in, e_out, step, cols."""
        r = cls(
            vertex=int(q["v"]),
            radius=float(q.get("radius", 2.0)),
            method=str(q.get("method", "normal")),
            every=int(q.get("every", 1)),
            e_in=float(q.get("e_in", R.CAP_IN)),
            e_out=float(q.get("e_out", R.CAP_OUT)),
            step=float(q.get("step", R.STEP)),
            cols=",".join(str(int(x)) for x in q.get("cols", "").split(",") if x),
        )
        r.check()
        return r

    def check(self) -> None:
        if not 0 <= self.radius <= MAX_RADIUS:
            raise ValueError(f"radius {self.radius}: not in [0, {MAX_RADIUS}] mm")
        if self.method not in R.METHODS:
            raise ValueError(f"method {self.method!r}: not one of {R.METHODS}")
        if self.every < 1:
            raise ValueError(f"every {self.every}: not >= 1")
        if not 0 <= self.e_in <= R.CAP_IN or not 0 <= self.e_out <= R.CAP_OUT:
            raise ValueError("e_in, e_out: outside the caps")
        if not self.step > 0:
            raise ValueError(f"step {self.step}: not > 0")

    def only(self) -> I64 | None:
        """The vertices of cols, ascending; None for all."""
        if not self.cols:
            return None
        return np.unique(np.array(self.cols.split(","), np.int64))

    def key(self, sources: str) -> str:
        """The cache key: the request and the sources' SHA-256."""
        text = json.dumps([sources, *self.__dict__.values()], default=float)
        return hashlib.sha1(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class LocalMesh:
    """The faces of a superblock's local graph, for the section: the faces
    (into the graph's vertices), the vertices in scanner mm, and their
    KD-tree."""

    faces: npt.NDArray[np.int32]
    pts: Float
    tree: cKDTree


@dataclass(frozen=True)
class Result:
    """The columns of one request, and when asked for, their echoes."""

    request: Request
    key: str
    rays: R.Rays
    #: the superblock and level of the sampler's call
    superblock: tuple[int, int, int]
    level: int
    #: the distance of each column's root from the vertex along the surface, mm
    distance: npt.NDArray[np.float32]
    #: the time of each step, s
    seconds: dict[str, float]


def select_columns(rays: R.Rays, idx: npt.ArrayLike) -> R.Rays:
    """The columns idx of rays (any order and subset), as a Rays of their own:
    each column with its own samples and intervals, unchanged."""
    c = np.asarray(idx, np.int64)
    r = rays.roots
    roots = replace(
        r,
        edge_key=r.edge_key[c],
        vertex=r.vertex[c],
        owned=r.owned[c],
        root=r.root[c],
        normal=r.normal[c],
        normal_length=r.normal_length[c],
    )

    def rows(start: I64) -> tuple[I64, I64]:
        n = start[c + 1] - start[c]
        out = np.zeros(len(c) + 1, np.int64)
        out[1:] = np.cumsum(n)
        take = (
            np.concatenate([np.arange(start[k], start[k + 1]) for k in c])
            if len(c)
            else np.zeros(0, np.int64)
        )
        return out, take.astype(np.int64)

    ss, si = rows(rays.sample_start)
    qs, qi = rows(rays.interval_start)
    return replace(
        rays,
        roots=roots,
        k_in=rays.k_in[c],
        k_out=rays.k_out[c],
        sample_start=ss,
        s=rays.s[si],
        position=rays.position[si],
        interval_start=qs,
        interval_s0=rays.interval_s0[qi],
        interval_s1=rays.interval_s1[qi],
        interval_label=rays.interval_label[qi],
        interval_class=rays.interval_class[qi],
        derived_class=rays.derived_class[c],
    )


#: the arrays of a /columns answer, in order: (name, the Rays or Roots field)
RAY_ARRAYS = (
    "edge_key",
    "vertex",
    "owned",
    "root",
    "normal",
    "normal_length",
    "k_in",
    "k_out",
    "sample_start",
    "s",
    "position",
    "interval_start",
    "interval_s0",
    "interval_s1",
    "interval_label",
    "interval_class",
    "derived_class",
)


def ray_arrays(rays: R.Rays) -> dict[str, Arr]:
    """The arrays of rays by the names of RAY_ARRAYS."""
    return {
        k: np.asarray(
            getattr(rays.roots, k) if hasattr(rays.roots, k) else getattr(rays, k)
        )
        for k in RAY_ARRAYS
    }


def encode(header: dict[str, Any], arrays: dict[str, Arr]) -> bytes:
    """The binary answer: see the module docstring. Bools go as uint8."""
    parts: list[bytes] = []
    table = []
    off = 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        a = a.astype(a.dtype.newbyteorder("<"))
        b = a.tobytes()
        table.append(
            dict(name=name, dtype=a.dtype.name, shape=list(a.shape), offset=off)
        )
        pad = -len(b) % 8
        parts.append(b + b"\0" * pad)
        off += len(b) + pad
    h = json.dumps(dict(header, arrays=table), default=float).encode()
    h += b" " * (-(len(h) + 4) % 8)
    return np.uint32(len(h)).tobytes() + h + b"".join(parts)


def decode(body: bytes) -> tuple[dict[str, Any], dict[str, Arr]]:
    """The header and the arrays of an answer of encode."""
    n = int(np.frombuffer(body[:4], "<u4")[0])
    head: dict[str, Any] = json.loads(body[4 : 4 + n])
    base = 4 + n
    out: dict[str, Arr] = {}
    for t in head["arrays"]:
        dt = np.dtype(t["dtype"]).newbyteorder("<")
        count = int(np.prod(t["shape"], dtype=np.int64))
        a = np.frombuffer(body, dt, count, base + t["offset"])
        out[t["name"]] = a.reshape(t["shape"])
    return head, out


class Columns:
    """The columns pane's data on one white surface: its ray sampler, its
    column-volume sampler (with echoes), and the mesh for the radius along
    the surface. v (voxel coordinates) and f are the whole white surface (as
    baillarger.white_surf.WhiteSurface holds them); a request reads only the
    rows near its vertex."""

    def __init__(
        self,
        surf: Path,
        dseg: Path,
        v: Float,
        f: npt.NDArray[np.integer[Any]],
        echoes: Sequence[Path] = (),
        log: Callable[[str], object] = print,
    ) -> None:
        self.rays = R.RaySampler(surf, dseg)
        self.v, self.f = v, f
        rs = self.rays
        bt = np.loadtxt(
            rs.surf.with_name(rs.sidecar["BlocksFile"]), skiprows=1, ndmin=2
        ).reshape(-1, 13)
        self.blocks = BlockIndex(
            bt[:, :3].astype(np.int64),
            bt[:, 3].astype(np.int64),
            bt[:, 4].astype(np.int64),
            bt[:, 5].astype(np.int64),
            bt[:, 6].astype(np.int64),
            bt[:, 7:10],
            bt[:, 10:13],
        )
        self.order = np.argsort(self.blocks.vertex_start)
        self.echo_paths = tuple(Path(p) for p in echoes)
        self.volume: ColumnVolumeSampler | None = None
        if self.echo_paths:
            srcs: list[Source] = []
            affs = []
            for p in self.echo_paths:
                src = open_volume(p)
                srcs.append(src)
                affs.append(src.affine)
            self.volume = ColumnVolumeSampler(srcs, rs.affine, affs, rs.shape)  # type: ignore[arg-type]
        side = rs.sidecar
        self.sources = hashlib.sha1(
            json.dumps(
                [side["SourcesSHA256"], side.get("DataSHA256"), file_sha256(dseg)]
                + [_echo_sha(p) for p in self.echo_paths],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16]
        self.vox = rs.voxel_mm
        # scanner mm to the 3D pane's frame (voxel coordinates times voxel size)
        self.to_view = np.diag([*self.vox, 1.0]) @ rs.inverse
        self.jobs: Jobs[Result] = Jobs(log=log, keep_in_memory=CACHE)
        self.profiles: OrderedDict[str, ColumnVolume] = OrderedDict()
        self.graphs: OrderedDict[tuple[int, int, int], LocalGraph] = OrderedDict()
        self.meshes: OrderedDict[tuple[int, int, int], LocalMesh] = OrderedDict()
        self.mesh_lock = threading.Lock()
        self.lock = threading.Lock()  # one sampler call at a time

    # ---------------------------------------------------------------- the columns
    def superblock(self, vertex: int) -> tuple[int, int, int]:
        """The superblock at LEVEL of the block that owns a vertex."""
        b = self.blocks
        k = int(np.searchsorted(b.vertex_start[self.order], vertex, side="right")) - 1
        if k < 0:
            raise ValueError(f"vertex {vertex}: not on the surface")
        row = self.order[k]
        if not b.vertex_start[row] <= vertex < b.vertex_stop[row]:
            raise ValueError(f"vertex {vertex}: owned by no block")
        ijk = b.ijk[row] >> LEVEL
        return int(ijk[0]), int(ijk[1]), int(ijk[2])

    def graph(self, sb: tuple[int, int, int]) -> LocalGraph:
        """The local graph of sb with a halo of MAX_RADIUS."""
        if sb in self.graphs:
            self.graphs.move_to_end(sb)
            return self.graphs[sb]
        g = local_graph(
            self.v,
            self.f,
            self.blocks,
            sb,
            LEVEL,
            self.rays.cells,
            MAX_RADIUS,
            self.vox,
        )
        self.graphs[sb] = g
        while len(self.graphs) > GRAPHS:
            self.graphs.popitem(last=False)
        return g

    def mesh(self, sb: tuple[int, int, int]) -> LocalMesh:
        """The faces of the local graph of sb (the faces of the blocks whose
        boxes meet its halo box, as mesh_graph.local_graph reads them), kept
        for the last GRAPHS superblocks."""
        with self.mesh_lock:
            if sb in self.meshes:
                self.meshes.move_to_end(sb)
                return self.meshes[sb]
            g = self.graph(sb)
            b = self.blocks
            span = 1 << LEVEL
            cells = self.rays.cells
            lo = np.asarray(sb, np.int64) * span * cells
            hi = lo + span * cells
            h = MAX_RADIUS / self.vox
            near = np.all((b.box_max >= lo - h) & (b.box_min <= hi + h), axis=1)
            rows = [
                np.asarray(self.f[a:z], np.int64)
                for a, z in zip(b.face_start[near], b.face_stop[near], strict=True)
                if z > a
            ]
            fl = np.concatenate(rows) if rows else np.zeros((0, 3), np.int64)
            if not np.array_equal(np.unique(fl), g.vertex):
                raise RuntimeError(f"superblock {sb}: faces and graph disagree")
            vx = np.asarray(self.v[g.vertex], np.float64)
            pts = vx @ self.rays.affine[:3, :3].T + self.rays.affine[:3, 3]
            m = LocalMesh(
                np.searchsorted(g.vertex, fl).astype(np.int32), pts, cKDTree(pts)
            )
            self.meshes[sb] = m
            while len(self.meshes) > GRAPHS:
                self.meshes.popitem(last=False)
            return m

    def section(
        self,
        req: Request,
        u: Sequence[float],
        mode: str,
        at: int | None = None,
        plane: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """The profile section of a request (baillarger.viewer.section), without
        sampling its columns, so the cap MAX_COLUMNS does not apply: the curve
        through the root of column `at` (a true vertex), one step for each mean
        spacing of the request's roots, and the columns at each step, as
        indices into the /columns answer of the whole request and as true
        vertices. The centre: `at` if it is a column of the request, else the
        request's vertex if it is one, else the column nearest it along the
        surface; `fallback` says when `at` was not a column. `plane`: the
        normal of a slice plane through the centre's root (scanner mm); the
        curve is that plane's cut unless it is degenerate there
        (VSec.section_curve); `slice` says which, and gives the angle between
        the root's normal and the plane."""
        t0 = time.perf_counter()
        whole = replace(req, cols="")
        sb, roots, chosen, dist = self.candidates(whole)
        if not len(chosen):
            raise ValueError("the request has no columns")
        vertex = roots.vertex[chosen]
        fallback = None
        hit = np.flatnonzero(vertex == (req.vertex if at is None else int(at)))
        if not len(hit) and at is not None:
            fallback = f"vertex {at}: not a column of the request"
            hit = np.flatnonzero(vertex == req.vertex)
        c0 = int(hit[0]) if len(hit) else int(np.argmin(dist))
        g = self.graph(sb)
        m = self.mesh(sb)
        v0 = int(vertex[c0])
        i0 = int(np.searchsorted(g.vertex, v0))
        if i0 >= len(g.vertex) or g.vertex[i0] != v0:
            raise RuntimeError(f"vertex {v0}: not in the local graph")
        normal = roots.normal[chosen[c0]]
        step = VSec.mean_spacing(np.asarray(roots.root[chosen], np.float64))
        curve, why = VSec.section_curve(
            m.pts, m.faces, m.pts[i0], normal, u, req.radius, step, plane
        )
        cut = None
        if plane is not None:
            pn = np.asarray(plane, np.float64)
            pn = pn / np.linalg.norm(pn)
            nn = np.asarray(normal, np.float64)
            nn = nn / np.linalg.norm(nn)
            sin = min(1.0, abs(float(pn @ nn)))
            cut = dict(
                plane=pn.tolist(),
                used=why is None,
                fallback=why,
                # the angle between the root's normal and the slice plane, deg
                angle=float(np.degrees(np.arcsin(sin))),
            )
        arc, pts, faces = VSec.steps(curve, step)
        pos = np.searchsorted(g.vertex, vertex)
        pos = np.minimum(pos, len(g.vertex) - 1)
        column_of = -np.ones(len(g.vertex), np.int64)
        ok = g.vertex[pos] == vertex
        column_of[pos[ok]] = np.flatnonzero(ok)
        voxel = float(np.max(self.vox))
        cols, d = VSec.step_columns(
            pts,
            faces,
            m.pts,
            m.faces,
            g.graph,
            m.tree,
            column_of,
            mode,
            voxel,
            req.radius,
        )
        return dict(
            key=whole.key(self.sources),
            mode=mode,
            n=len(chosen),
            capped=len(chosen) > MAX_COLUMNS,
            max_columns=MAX_COLUMNS,
            column=c0,
            vertex=v0,
            fallback=fallback,
            slice=cut,
            step=step,
            voxel=voxel,
            u=curve.u.tolist(),
            w=curve.w.tolist(),
            arc=arc.tolist(),
            point=pts.round(4).tolist(),
            columns=cols,
            step_vertices=[[int(vertex[c]) for c in cs] for cs in cols],
            # the true vertices of the roots of the section's columns, ascending
            roots=sorted({int(vertex[c]) for cs in cols for c in cs}),
            distance=[None if np.isnan(x) else float(x) for x in d],
            curve_length=[float(-curve.arc[0]), float(curve.arc[-1])],
            seconds=round(time.perf_counter() - t0, 3),
        )

    def candidates(
        self, req: Request
    ) -> tuple[tuple[int, int, int], R.Roots, I64, Float]:
        """The columns of a request without their rays: the superblock, the
        sampler's roots of the call, the chosen roots (indices into them, in
        the order of the /columns answer: within the radius, every n-th, and
        only cols if given), and their distances along the surface."""
        sb = self.superblock(req.vertex)
        verts, dist = self.near(req.vertex, req.radius)
        with self.lock:
            roots = self.rays.roots(sb, LEVEL, req.radius)
        pos = np.searchsorted(verts, roots.vertex)
        pos = np.minimum(pos, len(verts) - 1)
        hit = verts[pos] == roots.vertex
        if hit.sum() != len(verts):
            raise RuntimeError(
                f"vertex {req.vertex}: {len(verts) - int(hit.sum())} roots "
                "within the radius are not in the sampler's halo"
            )
        chosen = np.flatnonzero(hit)[:: req.every]
        only = req.only()
        if only is not None:
            keep = np.isin(roots.vertex[chosen], only)
            missing = np.setdiff1d(only, roots.vertex[chosen[keep]])
            if len(missing):
                raise ValueError(
                    f"cols: vertex {int(missing[0])} is not a column of the request"
                )
            chosen = chosen[keep]
        return sb, roots, chosen, dist[pos[chosen]]

    def near(self, vertex: int, radius: float) -> tuple[I64, Float]:
        """The vertices within radius of vertex along the surface, ascending,
        and their distances (mm)."""
        sb = self.superblock(vertex)
        g = self.graph(sb)
        i = int(np.searchsorted(g.vertex, vertex))
        if i >= len(g.vertex) or g.vertex[i] != vertex:
            return np.array([vertex], np.int64), np.zeros(1)
        d = dijkstra(g.graph, indices=i, limit=radius)
        hit = np.flatnonzero(np.isfinite(d))
        return g.vertex[hit], d[hit]

    def compute(self, req: Request, log: Callable[[str], object] = print) -> Result:
        """The columns of a request (see the module docstring)."""
        t0 = time.perf_counter()
        sb, roots, chosen, d = self.candidates(req)
        t1 = time.perf_counter()
        if len(chosen) > MAX_COLUMNS:
            raise ValueError(
                f"{len(chosen)} columns: more than {MAX_COLUMNS}; "
                "use a smaller radius or a larger `every`"
            )
        with self.lock:
            e_in = np.zeros(len(roots.vertex))
            e_out = np.zeros(len(roots.vertex))
            e_in[chosen], e_out[chosen] = req.e_in, req.e_out
            rays = self.rays.sample(
                sb,
                LEVEL,
                method=req.method,
                e_in=e_in,
                e_out=e_out,
                step=req.step,
                halo_mm=req.radius,
            )
        t2 = time.perf_counter()
        out = select_columns(rays, chosen)
        secs = dict(
            graph=round(t1 - t0, 3),
            sample=round(t2 - t1, 3),
            calls=len(roots.vertex),
        )
        log(
            f"columns at vertex {req.vertex}: {len(chosen)} of "
            f"{len(roots.vertex)} in superblock {sb} ({secs})"
        )
        return Result(
            req, req.key(self.sources), out, sb, LEVEL, d.astype(np.float32), secs
        )

    def get(self, req: Request) -> dict[str, Any]:
        """Start the request as a job, or find it done; wait up to WAIT_S.
        Returns {key, status} ("done", "running" or "busy": another job runs)."""
        key = req.key(self.sources)
        try:
            out = self.jobs.run(
                key,
                f"columns at vertex {req.vertex}",
                lambda log: self.compute(req, log),
            )
        except RuntimeError:
            return dict(key=key, status="busy", job=self.jobs.status())
        if out["status"] != "done" and self.jobs.thread is not None:
            self.jobs.thread.join(WAIT_S)
        st = self.jobs.status()
        if self.jobs.has(key):
            return dict(key=key, status="done")
        if st["key"] == key and st["status"] == "error":
            raise RuntimeError(st.get("error", "the job failed"))
        return dict(key=key, status="running", job=st)

    def result(self, key: str) -> Result:
        return self.jobs.get(key)

    def columns_bytes(self, key: str) -> bytes:
        """The binary answer of /columns for a finished request."""
        r = self.result(key)
        rays = r.rays
        arrays = ray_arrays(rays)
        arrays["distance"] = r.distance
        head = dict(
            key=key,
            spec=R.VERSION,
            request=r.request.__dict__,
            n=len(rays.roots.vertex),
            samples=len(rays.s),
            intervals=len(rays.interval_s0),
            method=rays.method,
            step=rays.step,
            superblock=list(r.superblock),
            level=r.level,
            seconds=r.seconds,
            to_view=self.to_view.tolist(),
            classes=dict(
                interval=dict(
                    OUTSIDE=R.OUTSIDE,
                    WM=R.WM,
                    CORTEX=R.CORTEX,
                    OTHER=R.OTHER,
                    OFF_GRID=R.OFF_GRID,
                ),
                derived=dict(OPEN=R.OPEN, IN_WM=R.IN_WM, NEITHER=R.NEITHER),
            ),
        )
        return encode(head, arrays)

    def profile(self, key: str) -> ColumnVolume:
        """The column volume of a finished request (computed once, kept for
        the last CACHE requests)."""
        if self.volume is None:
            raise ValueError("no echoes: start the viewer with --echo-dir")
        with self.lock:
            if key in self.profiles:
                self.profiles.move_to_end(key)
                return self.profiles[key]
        r = self.result(key)
        t = time.perf_counter()
        cv = self.volume.sample(r.rays, "numpy")
        r.seconds["profiles"] = round(time.perf_counter() - t, 3)
        with self.lock:
            self.profiles[key] = cv
            while len(self.profiles) > CACHE:
                self.profiles.popitem(last=False)
        return cv

    def profiles_bytes(self, key: str) -> bytes:
        """The binary answer of /columns/profiles: each sample's echoes (N, E)
        and validity, in the order of the samples of /columns."""
        cv = self.profile(key)
        head = dict(
            key=key,
            echoes=[p.name for p in self.echo_paths],
            seconds=self.result(key).seconds,
        )
        return encode(head, dict(echo=cv.echo, valid=cv.valid))

    def tck_bytes(self, key: str) -> bytes:
        """The columns of a finished request as .tck (scanner mm)."""
        rays = self.result(key).rays
        st = rays.sample_start
        return tck(
            [
                rays.position[st[c] : st[c + 1]].astype(np.float64)
                for c in range(len(st) - 1)
            ]
        )


def _echo_sha(p: Path) -> str:
    """The SHA-256 of an echo: from the record of its decompressed copy
    (baillarger.blocks.decompress) if there is one, else its file name and
    size (a file of 600 MB is not hashed at each start)."""
    side = p.with_name(p.name + ".source.json")
    if side.exists():
        return str(json.loads(side.read_text()).get("SourceSHA256"))
    return f"{p.name}:{p.stat().st_size}"


# ---------------------------------------------------------------- the routes


def meta(c: Columns) -> dict[str, Any]:
    """GET /columns/meta: the limits and the choices of the pane."""
    return dict(
        spec=R.VERSION,
        methods=list(R.METHODS),
        cap_in=R.CAP_IN,
        cap_out=R.CAP_OUT,
        step=R.STEP,
        max_radius=MAX_RADIUS,
        max_columns=MAX_COLUMNS,
        wait_s=WAIT_S,
        level=LEVEL,
        echoes=[p.name for p in c.echo_paths],
        to_view=c.to_view.tolist(),
    )


def route(h: Any, path: str, q: dict[str, str]) -> None:
    """The /columns routes, for the server's handler h (its send method):

    - GET /columns/meta: meta.
    - GET /columns?v=&radius=[&method=&every=&e_in=&e_out=&step=]: the columns
      (columns_bytes) when they are ready within WAIT_S; else 202 and JSON
      {key, status: "running" or "busy", job}: ask again.
    - GET /columns/profiles?(the same): their echoes (profiles_bytes), or 202
      as above.
    - GET /columns/section?(the same)&u=x,y,z[&mode=nearest|mean&at=
      &plane=x,y,z]: the profile section that follows the surface
      (Columns.section; u and the slice plane's normal in scanner mm). It
      samples no column, so it answers over the cap too; then the page asks
      /columns and /columns/profiles with cols= (the section's columns).
    - GET /columns/<key>.tck: the columns of a finished request, as .tck.
    Errors: 400 with JSON {error}; 404 without columns (no --white)."""
    c: Columns | None = getattr(h, "columns", None)
    if c is None:
        h.send_error(404)
        return

    def js(obj: Any, code: int = 200) -> None:
        h.send(json.dumps(obj, default=float).encode(), "application/json", code)

    try:
        if path == "/columns/meta":
            js(meta(c))
        elif path in ("/columns", "/columns/profiles", "/columns/section"):
            req = Request.parse(q)
            if path == "/columns/section":
                u = [float(x) for x in q["u"].split(",")]
                if len(u) != 3:
                    raise ValueError("u: give three numbers x,y,z")
                mode = str(q.get("mode", "nearest"))
                if mode not in VSec.MODES:
                    raise ValueError(f"mode {mode!r}: not one of {VSec.MODES}")
                at = int(q["at"]) if q.get("at") else None
                plane = None
                if q.get("plane"):
                    plane = [float(x) for x in q["plane"].split(",")]
                    if len(plane) != 3 or not np.linalg.norm(plane) > 0:
                        raise ValueError("plane: give three numbers x,y,z, not all 0")
            if path == "/columns/section":
                js(c.section(req, u, mode, at, plane))
                return
            st = c.get(req)
            if st["status"] != "done":
                js(st, 202)
            elif path == "/columns":
                h.send(c.columns_bytes(st["key"]), "application/octet-stream")
            else:
                h.send(c.profiles_bytes(st["key"]), "application/octet-stream")
        elif path.startswith("/columns/") and path.endswith(".tck"):
            key = path.removeprefix("/columns/").removesuffix(".tck")
            h.send(c.tck_bytes(key), "application/octet-stream")
        else:
            h.send_error(404)
    except (KeyError, ValueError, RuntimeError) as exc:
        js(dict(error=f"{type(exc).__name__}: {exc}"), 400)


def dseg_for(surf: Path, sidecar: dict[str, Any]) -> Path:
    """The surface's segmentation: the _desc-wmtopo_dseg.nii.gz next to the
    surface (with the surface's subject and session), if there is one; else the
    first of the sidecar's Sources. The ray sampler checks its SHA-256."""
    stem = surf.name.split("_hemi-")[0]
    here = surf.with_name(f"{stem}_desc-wmtopo_dseg.nii.gz")
    return here if here.exists() else Path(sidecar["Sources"][0])


def echo_files(directory: Path) -> list[Path]:
    """Echoes 1 to 3 in a directory: the decompressed .nii if it is there,
    else the .nii.gz (baillarger.data.echo_path names). None if one is
    missing: then the pane has no profiles."""
    from baillarger.data import echo_path

    out = []
    for k in (1, 2, 3):
        gz = echo_path(k, directory)
        nii = gz.with_suffix("")
        out.append(nii if nii.exists() else gz)
    return out if all(p.exists() for p in out) else []
