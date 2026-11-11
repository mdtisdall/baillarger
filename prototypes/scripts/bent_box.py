"""The box of the bent-columns trial (docs/plans/bent-columns-trial.md,
phase 1): the straight columns near one vertex of the white surface, the
mesh between their roots, and a voxel box of the echoes and the labels that
holds every column.

box() follows the recipe of baillarger.viewer.columns.Columns.compute, with
no cap on the number of columns:

1. The superblock S (level 0) of the block that owns the vertex.
2. The local graph of S (mesh_graph.local_graph) with a halo of
   radius_mm + halo_mm, and a Dijkstra from the vertex, limited to
   radius_mm + halo_mm. The distances are exact up to the graph's halo.
3. One call of RaySampler.roots and RaySampler.sample on S with a halo of
   radius_mm + halo_mm. The chosen columns get e_in = CAP_IN and
   e_out = CAP_OUT. All other columns get extents 0. The extents of a column
   do not change its samples, intervals or class (ray-sampler/1), so each
   chosen column is the sampler's own, bit for bit (select_columns).

The columns are in the sampler's order: the owned columns of S, then its
halo columns, each in vertex order.

The halo (halo_mm): the box holds the roots within radius_mm along the
surface (the inner columns, Box.inner) and also the roots within
radius_mm + halo_mm (the halo columns). A halo column has the same values as
an inner column. The halo columns give each inner column its neighbours on
all sides: with a halo at least one mesh edge long, every face round an
inner root is in Box.faces. The plan is that a fit moves only the inner
columns and holds the halo columns fixed. With halo_mm = 0 the box has the
columns of Columns.compute(Request(vertex, radius_mm)).

Painted labels (paint): label edits on the dseg's grid (label-edits/1
diffs, baillarger.label_edits; the viewer's paint mode writes them). Their
voxels and labels replace the box's labels where they fall in the box, in
the order given. Only the box's labels change. The straight rays' label
intervals stay those of the dseg (the ray sampler reads the dseg itself), so
a caller that needs the painted labels along a ray samples Box.labels. The
sidecar of the command line lists the diffs.

The voxel box holds the straight rays of all chosen columns from -CAP_IN to
CAP_OUT, widened by MARGIN_MM on each axis, and cut to the grid. A bent
path can leave it: a caller must check.

Run from the repository root (heavy runs under scripts/memguard):
    nix develop --command uv run python prototypes/scripts/bent_box.py \\
        --white W --dseg D --echoes E1 E2 E3 --vertex V --radius 5 --out DIR
It writes DIR/box.npz and DIR/box.json (the inputs and the sizes).
"""

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse.csgraph import dijkstra

from baillarger import label_edits
from baillarger import rays as R
from baillarger.blocks import Source, open_volume
from baillarger.column_volume import ColumnVolumeSampler
from baillarger.mesh_graph import BlockIndex, local_graph
from baillarger.mesh_smoothing import edges
from baillarger.viewer.columns import select_columns
from baillarger.white_surf import WhiteSurface

type Float = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

#: the superblock level of the sampler's call (as the viewer's LEVEL)
LEVEL = 0
#: the margin of the voxel box round the straight rays, mm
MARGIN_MM = 1.0
#: the default halo, mm
HALO_MM = 1.0
#: the file names that the command line writes in its output directory
NPZ = "box.npz"
SIDECAR = "box.json"


@dataclass(frozen=True)
class Box:
    """The columns near one vertex and the voxel box round them (see the
    module docstring). n is the number of columns."""

    #: the true vertex of the white surface at the centre
    vertex: int
    radius_mm: float
    halo_mm: float
    #: the superblock (level LEVEL) of the sampler's call
    superblock: tuple[int, int, int]
    #: the chosen columns, each the sampler's own (straight samples and
    #: label intervals, scanner mm)
    rays: R.Rays
    #: (n,) the distance of each root from the vertex along the surface, mm
    distance: Float
    #: (n,) true for an inner column (distance <= radius_mm), false for a
    #: halo column
    inner: npt.NDArray[np.bool_]
    #: (k, 3) the faces whose three corners are chosen roots, as column
    #: indices 0..n-1, in the order of the sampler's faces
    faces: I64
    #: (m, 2) the edges of the faces (mesh_smoothing.edges)
    edges: I64
    #: (E, X, Y, Z) float32: the echoes in the box, one array for each echo
    echoes: npt.NDArray[np.float32]
    #: (X, Y, Z) int32: the labels of the dseg in the box
    labels: npt.NDArray[np.int32]
    #: (3,) the voxel index of the box's corner (box voxel 0, 0, 0) in the grid
    offset: I64
    #: (4, 4) the grid's voxel to scanner mm affine (VoxelToRAS), and its inverse
    affine: Float
    inverse: Float
    #: (3,) the voxel size, mm
    voxel_mm: Float
    #: the sampler's filled labels (the WM class) and cortex label
    filled: I64
    cortex: int

    @property
    def roots(self) -> npt.NDArray[np.float32]:
        """(n, 3) the roots, scanner mm."""
        return self.rays.roots.root

    @property
    def normals(self) -> npt.NDArray[np.float32]:
        """(n, 3) the unit normals, scanner space."""
        return self.rays.roots.normal

    @property
    def mm_to_box(self) -> Float:
        """(4, 4): scanner mm to box voxel coordinates."""
        t = np.eye(4)
        t[:3, 3] = -self.offset
        out: Float = t @ self.inverse
        return out

    def to_box(self, x: npt.ArrayLike) -> Float:
        """Points (..., 3) in scanner mm to box voxel coordinates (float64).
        Voxel centres are at integer coordinates, as for map_coordinates."""
        p = np.asarray(x, np.float64)
        a = self.inverse
        out = np.stack(
            [
                a[k, 0] * p[..., 0]
                + a[k, 1] * p[..., 1]
                + a[k, 2] * p[..., 2]
                + a[k, 3]
                for k in range(3)
            ],
            axis=-1,
        )
        return out - self.offset


def _block_index(rs: R.RaySampler) -> BlockIndex:
    """The surface's block index as mesh_graph.BlockIndex (as the viewer's
    Columns reads it)."""
    return BlockIndex(
        rs.block_ijk,
        rs.block_vertices[:, 0],
        rs.block_vertices[:, 1],
        rs.block_faces[:, 0],
        rs.block_faces[:, 1],
        rs.block_box[:, :3],
        rs.block_box[:, 3:],
    )


def _superblock(rs: R.RaySampler, vertex: int) -> tuple[int, int, int]:
    """The superblock at LEVEL of the block that owns a vertex."""
    bv = rs.block_vertices
    hit = np.flatnonzero((bv[:, 0] <= vertex) & (vertex < bv[:, 1]))
    if len(hit) != 1:
        raise ValueError(f"vertex {vertex}: owned by {len(hit)} blocks, not 1")
    ijk = rs.block_ijk[hit[0]] >> LEVEL
    return int(ijk[0]), int(ijk[1]), int(ijk[2])


def _echo_sampler(rs: R.RaySampler, echoes: Sequence[Path]) -> list[Source]:
    """The echoes' sources. ColumnVolumeSampler checks that each one is on
    the grid of the dseg, with its affine."""
    srcs: list[Source] = []
    affs = []
    for p in echoes:
        src = open_volume(Path(p))
        srcs.append(src)
        affs.append(src.affine)
    ColumnVolumeSampler(srcs, rs.affine, affs, rs.shape)  # type: ignore[arg-type]
    return srcs


def paint(
    labels: npt.NDArray[np.int32],
    offset: I64,
    diffs: Sequence[Path],
    shape: Sequence[int] | None = None,
) -> tuple[npt.NDArray[np.int32], int]:
    """labels (a box of the grid at offset) with the voxels and labels of
    each label-edits/1 diff applied, in order, and the number of voxels
    changed in the box. With shape (the grid's), a diff on another grid
    raises ValueError."""
    out = np.array(labels, np.int32, copy=True)
    n = 0
    for path in diffs:
        e, side, _ = label_edits.read_with_sidecar(
            Path(path), shape=None if shape is None else tuple(int(s) for s in shape)
        )
        grid = tuple(int(s) for s in side["Shape"])
        pts = np.stack(np.unravel_index(e.index, grid), 1).astype(np.int64)
        vals = e.label.astype(np.int64)
        q = pts - np.asarray(offset, np.int64)
        ok = np.all((q >= 0) & (q < np.asarray(out.shape)), axis=1)
        q, v = q[ok], vals[ok]
        n += int(np.count_nonzero(out[tuple(q.T)] != v))
        out[tuple(q.T)] = v
    return out, n


def box(
    white: Path,
    dseg: Path,
    echoes: Sequence[Path],
    vertex: int,
    radius_mm: float,
    halo_mm: float = HALO_MM,
    margin_mm: float = MARGIN_MM,
    diffs: Sequence[Path] = (),
) -> Box:
    """The box at a vertex: see the module docstring. white is the
    _white.surf.gii; dseg is the segmentation that the surface was made from
    (the sampler checks its SHA-256); echoes are on the dseg's grid; diffs
    are label edits on that grid (paint)."""
    if not radius_mm >= 0 or not halo_mm >= 0 or not margin_mm >= 0:
        raise ValueError("radius_mm, halo_mm and margin_mm must be >= 0")
    rs = R.RaySampler(Path(white), Path(dseg))
    srcs = _echo_sampler(rs, echoes)
    reach = radius_mm + halo_mm
    sb = _superblock(rs, vertex)
    W = WhiteSurface.load(Path(white))
    g = local_graph(W.v, W.f, _block_index(rs), sb, LEVEL, rs.cells, reach, rs.voxel_mm)
    del W
    i = int(np.searchsorted(g.vertex, vertex))
    if i < len(g.vertex) and g.vertex[i] == vertex:
        d = dijkstra(g.graph, indices=i, limit=reach)
        hit = np.flatnonzero(np.isfinite(d))
        verts, dist = g.vertex[hit], d[hit]
    else:  # a vertex that no face uses
        verts, dist = np.array([vertex], np.int64), np.zeros(1)
    roots = rs.roots(sb, LEVEL, reach)
    pos = np.minimum(np.searchsorted(verts, roots.vertex), len(verts) - 1)
    found = verts[pos] == roots.vertex
    if found.sum() != len(verts):
        raise RuntimeError(
            f"vertex {vertex}: {len(verts) - int(found.sum())} roots within "
            "the radius are not in the sampler's halo"
        )
    chosen = np.flatnonzero(found)
    e_in = np.zeros(len(roots.vertex))
    e_out = np.zeros(len(roots.vertex))
    e_in[chosen], e_out[chosen] = R.CAP_IN, R.CAP_OUT
    rays = rs.sample(sb, LEVEL, e_in=e_in, e_out=e_out, halo_mm=reach)
    rays = select_columns(rays, chosen)
    distance = np.asarray(dist[pos[chosen]], np.float64)
    inner = distance <= radius_mm

    # the faces with three chosen corners, as column indices
    f = rs.faces(sb, LEVEL, reach)
    vert = rays.roots.vertex
    order = np.argsort(vert)
    sv = vert[order]
    at = np.minimum(np.searchsorted(sv, f), len(sv) - 1)
    keep = np.all(sv[at] == f, axis=1)
    faces = order[at[keep]].astype(np.int64)
    edge = edges(faces).astype(np.int64) if len(faces) else np.zeros((0, 2), np.int64)

    # the voxel box: both ends of each straight ray at the full caps
    root = rays.roots.root.astype(np.float64)
    normal = rays.roots.normal.astype(np.float64)
    ends = np.concatenate([root - R.CAP_IN * normal, root + R.CAP_OUT * normal])
    v = R._to_voxel(ends, rs.inverse)
    m = margin_mm / rs.voxel_mm
    shape = np.asarray(rs.shape, np.int64)
    lo = np.clip(np.floor(v.min(0) - m).astype(np.int64), 0, shape)
    hi = np.clip(np.ceil(v.max(0) + m).astype(np.int64) + 1, 0, shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    lab = np.ascontiguousarray(rs.labels.read(sl), np.int32)
    echo = np.stack([np.asarray(s.read(sl), np.float32) for s in srcs])
    if diffs:
        lab, _ = paint(lab, lo, diffs, rs.shape)
    return Box(
        vertex=int(vertex),
        radius_mm=float(radius_mm),
        halo_mm=float(halo_mm),
        superblock=sb,
        rays=rays,
        distance=distance,
        inner=inner,
        faces=faces,
        edges=edge,
        echoes=echo,
        labels=lab,
        offset=lo,
        affine=np.asarray(rs.affine, np.float64),
        inverse=np.asarray(rs.inverse, np.float64),
        voxel_mm=np.asarray(rs.voxel_mm, np.float64),
        filled=np.asarray(rs.filled, np.int64),
        cortex=int(rs.cortex),
    )


# ---------------------------------------------------------------- save, load

#: the Box fields that are arrays, saved under their own names
_ARRAYS = (
    "distance",
    "inner",
    "faces",
    "edges",
    "echoes",
    "labels",
    "offset",
    "affine",
    "inverse",
    "voxel_mm",
    "filled",
)


def save(b: Box, path: Path) -> None:
    """Write a box to an .npz (not compressed). The Rays fields are saved
    as "rays.<name>" and the Roots fields as "roots.<name>"."""
    out: dict[str, Any] = {k: getattr(b, k) for k in _ARRAYS}
    for fd in fields(R.Roots):
        out["roots." + fd.name] = getattr(b.rays.roots, fd.name)
    for fd in fields(R.Rays):
        if fd.name != "roots":
            out["rays." + fd.name] = np.asarray(getattr(b.rays, fd.name))
    out["vertex"] = np.int64(b.vertex)
    out["radius_mm"] = np.float64(b.radius_mm)
    out["halo_mm"] = np.float64(b.halo_mm)
    out["superblock"] = np.asarray(b.superblock, np.int64)
    out["cortex"] = np.int64(b.cortex)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def load(path: Path) -> Box:
    """Read a box that save wrote."""
    with np.load(path, allow_pickle=False) as z:
        roots = R.Roots(**{fd.name: z["roots." + fd.name] for fd in fields(R.Roots)})
        kw: dict[str, Any] = {}
        for fd in fields(R.Rays):
            if fd.name != "roots":
                kw[fd.name] = z["rays." + fd.name]
        kw["method"] = str(kw["method"])
        kw["step"] = float(kw["step"])
        rays = R.Rays(roots=roots, **kw)
        sb = z["superblock"]
        return Box(
            vertex=int(z["vertex"]),
            radius_mm=float(z["radius_mm"]),
            halo_mm=float(z["halo_mm"]),
            superblock=(int(sb[0]), int(sb[1]), int(sb[2])),
            rays=rays,
            cortex=int(z["cortex"]),
            **{k: z[k] for k in _ARRAYS},
        )


# ---------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--white", type=Path, required=True, help="the _white.surf.gii")
    p.add_argument("--dseg", type=Path, required=True, help="the surface's dseg")
    p.add_argument("--echoes", type=Path, nargs="+", required=True)
    p.add_argument("--vertex", type=int, required=True, help="true vertex")
    p.add_argument("--radius", type=float, required=True, help="mm")
    p.add_argument("--halo", type=float, default=HALO_MM, help="mm")
    p.add_argument("--margin", type=float, default=MARGIN_MM, help="mm")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument(
        "--paint",
        type=Path,
        action="append",
        default=[],
        help="label edits on the dseg's grid (a label-edits/1 .npy diff, as the "
        "viewer's paint mode writes); repeatable",
    )
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    b = box(a.white, a.dseg, a.echoes, a.vertex, a.radius, a.halo, a.margin, a.paint)
    t1 = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=True)
    save(b, a.out / NPZ)
    t2 = time.perf_counter()
    n = len(b.rays.roots.vertex)
    side = dict(
        white=str(a.white.resolve()),
        dseg=str(a.dseg.resolve()),
        echoes=[str(Path(e).resolve()) for e in a.echoes],
        vertex=a.vertex,
        radius_mm=a.radius,
        halo_mm=a.halo,
        margin_mm=a.margin,
        paint=[str(Path(c).resolve()) for c in a.paint],
        ray_sampler=R.VERSION,
        superblock=list(b.superblock),
        columns=n,
        inner_columns=int(b.inner.sum()),
        samples=len(b.rays.s),
        faces=len(b.faces),
        box_shape=list(b.labels.shape),
        box_offset=b.offset.tolist(),
        seconds=dict(box=round(t1 - t0, 2), save=round(t2 - t1, 2)),
    )
    (a.out / SIDECAR).write_text(json.dumps(side, indent=2) + "\n")
    print(
        f"{n} columns ({int(b.inner.sum())} inner), {len(b.faces)} faces, "
        f"box {tuple(b.labels.shape)} at {tuple(b.offset.tolist())}, "
        f"{t1 - t0:.1f} s (+ {t2 - t1:.1f} s to save)"
    )


if __name__ == "__main__":
    main()
