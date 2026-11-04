"""The outlines of the true white surface on an axis-aligned plane, inside a box, for
the brick pane's overlay.

The plane is world coordinate `axis` equal to `mm`. The box is a world-mm box. The
box is flattened onto the plane: its extent on the plane's axis is replaced by `mm`.

The cut does not touch the whole true mesh. The block index (docs/formats/white-surf.md)
gives, for each block, its face range and the bounds of its vertices in voxel
coordinates. The cut finds the voxel bounding box of the box, selects the blocks whose
bounds meet it (with half a voxel of margin), and adds up their face counts. This is
`faces_in_box`, counted from the table before any face is read.

If `faces_in_box` is at most `limit`, only those face ranges are read and cut, in
mm, from the true vertices. A face that the plane crosses gives one segment between
its two crossing edges. A vertex on the plane counts as on the positive side. A
crossing point on an edge depends only on the edge's two vertices, taken in order of
vertex index, so two faces that share an edge give the same end exactly.

If `faces_in_box` is above `limit`, the cut uses the overview mesh, which is small,
and the outline says so (`source` "overview"; display only).

A segment is kept if at least one of its ends is inside the box on the two in-plane
axes (with EPS_MM of tolerance). Segments are not clipped to the box.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from baillarger.viewer.white_view import WhiteView

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.integer[Any]]

#: the most true faces that one cut reads; above it the overview is cut.
#: Measured (laptop, in-memory arrays, one block, a plane through a sphere mesh):
#: 4.4e7 to 4.6e7 faces/s for 0.33e6 to 5.2e6 faces (0.108 s for 5.2e6 faces). So
#: 4e6 faces take about 0.09 s; the rest of the 0.2 s budget is for reading a memory
#: map and many blocks, which was not measured.
FACE_LIMIT = 4_000_000
#: tolerance of the in-box test, mm
EPS_MM = 1e-4
#: margin of the block test, voxels
MARGIN_VOX = 0.5


@dataclass(frozen=True)
class Outline:
    #: (k, 2, 3) float32, world mm
    segments: npt.NDArray[np.float32]
    #: "true" or "overview"
    source: str
    #: the true faces read, or the overview faces examined
    faces_read: int
    #: the faces of the blocks that meet the box, from the table
    faces_in_box: int


def plane_cut(points: Any, f: Int, axis: int, mm: float) -> Float:
    """The segments (k, 2, 3), world mm, where plane axis = mm cuts the triangles f
    of points (n, 3); points may be a memory map, of which only the vertices of f are
    read."""
    empty = np.zeros((0, 2, 3), np.float64)
    if not len(f):
        return empty
    sd = np.asarray(points[f.ravel(), axis], np.float64).reshape(-1, 3) - mm
    pos = sd >= 0
    n = pos.sum(1)
    hit = (n > 0) & (n < 3)
    if not hit.any():
        return empty
    f, sd, pos, n = f[hit], sd[hit], pos[hit], n[hit]
    r = np.arange(len(f))
    # the vertex alone on its side; the plane crosses its two edges
    odd = np.where(n == 1, pos.argmax(1), (~pos).argmax(1))
    ends = []
    for step in (1, 2):
        j = (odd + step) % 3
        ia, ib = f[r, odd], f[r, j]
        sa, sb = sd[r, odd], sd[r, j]
        swap = ia > ib
        ia, ib = np.where(swap, ib, ia), np.where(swap, f[r, odd], ib)
        sa, sb = np.where(swap, sb, sa), np.where(swap, sd[r, odd], sb)
        pa = np.asarray(points[ia], np.float64)
        pb = np.asarray(points[ib], np.float64)
        w = sa / (sa - sb)
        ends.append(pa + w[:, None] * (pb - pa))
    out: Float = np.stack(ends, 1)
    out[:, :, axis] = mm
    return out


class Outlines:
    """Cuts of a white surface; see the module docstring.

    vertices (n, 3) and faces (m, 3): the true surface, world mm (may be memory maps);
    blocks: its block table (columns block_i block_j block_k vertex_start vertex_stop
    face_start face_stop min_i min_j min_k max_i max_j max_k), bounds in voxel
    coordinates; affine: voxel -> world; ov_vertices (world mm) and ov_faces: the
    overview."""

    def __init__(
        self,
        vertices: npt.ArrayLike,
        faces: Int,
        blocks: npt.ArrayLike,
        affine: npt.ArrayLike,
        ov_vertices: npt.ArrayLike,
        ov_faces: Int,
        limit: int = FACE_LIMIT,
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.blocks = np.asarray(blocks, np.float64).reshape(-1, 13)
        self.inv = np.linalg.inv(np.asarray(affine, np.float64))
        self.ov_vertices = np.asarray(ov_vertices, np.float64)
        self.ov_faces = np.asarray(ov_faces)
        self.limit = int(limit)
        p = self.ov_vertices[self.ov_faces]  # (m, 3, 3)
        self._ov_lo: Float = p.min(1) if len(p) else np.zeros((0, 3))
        self._ov_hi: Float = p.max(1) if len(p) else np.zeros((0, 3))

    @classmethod
    def from_white_view(cls, w: "WhiteView", limit: int = FACE_LIMIT) -> "Outlines":
        """The cuts of w's true surface: its memory maps are kept; the overview is
        converted to mm once."""
        return cls(
            w.pts,
            w.W.f,
            w.blocks,
            w.W.affine,
            w.S.world(w.ov.v),
            w.ov.f,
            limit,
        )

    def cut(
        self,
        axis: int,
        mm: float,
        lo: Sequence[float],
        hi: Sequence[float],
    ) -> Outline:
        """The outline on plane (world coordinate axis = mm) in the world-mm box
        [lo, hi]."""
        if axis not in (0, 1, 2):
            raise ValueError(f"axis {axis} is not 0, 1 or 2")
        lo_ = np.asarray(lo, np.float64).copy()
        hi_ = np.asarray(hi, np.float64).copy()
        lo_[axis] = hi_[axis] = mm
        corners = np.array(
            [[(lo_, hi_)[i >> k & 1][k] for k in range(3)] for i in range(8)]
        )
        vox = corners @ self.inv[:3, :3].T + self.inv[:3, 3]
        vlo, vhi = vox.min(0), vox.max(0)
        b = self.blocks
        meet = np.all(b[:, 7:10] <= vhi + MARGIN_VOX, axis=1) & np.all(
            b[:, 10:13] >= vlo - MARGIN_VOX, axis=1
        )
        rows = b[meet]
        in_box = int((rows[:, 6] - rows[:, 5]).sum())
        if in_box > self.limit:
            return self._overview(axis, mm, lo_, hi_, in_box)
        parts = [self.faces[int(r[5]) : int(r[6])] for r in rows if r[6] > r[5]]
        if not parts:
            return Outline(np.zeros((0, 2, 3), np.float32), "true", 0, in_box)
        f = np.concatenate(parts)
        seg = _in_box(plane_cut(self.vertices, f, axis, mm), axis, lo_, hi_)
        return Outline(seg.astype(np.float32), "true", len(f), in_box)

    def _overview(
        self, axis: int, mm: float, lo: Float, hi: Float, in_box: int
    ) -> Outline:
        """The cut of the overview faces whose vertices' bounding box meets the box."""
        near = np.all(self._ov_lo <= hi, axis=1) & np.all(self._ov_hi >= lo, axis=1)
        f = self.ov_faces[near]
        seg = _in_box(plane_cut(self.ov_vertices, f, axis, mm), axis, lo, hi)
        return Outline(seg.astype(np.float32), "overview", len(f), in_box)


def _in_box(seg: Float, axis: int, lo: Float, hi: Float) -> Float:
    """The segments with an end inside the box on the two in-plane axes."""
    if not len(seg):
        return seg
    ax = [a for a in range(3) if a != axis]
    p = seg[:, :, ax]
    inside = np.all((p >= lo[ax] - EPS_MM) & (p <= hi[ax] + EPS_MM), axis=2)
    out: Float = seg[inside.any(1)]
    return out


def encode(o: Outline) -> bytes:
    """The segments as little-endian float32: per segment end a then end b, x y z."""
    return np.ascontiguousarray(o.segments, dtype="<f4").tobytes()


def _floats(s: str, n: int, name: str) -> list[float]:
    try:
        v = [float(x) for x in s.split(",")]
    except ValueError as e:
        raise ValueError(f"{name}: not numbers: {s!r}") from e
    if len(v) != n or not all(np.isfinite(v)):
        raise ValueError(f"{name}: need {n} finite numbers, got {s!r}")
    return v


def outline_response(
    outlines: Outlines, q: dict[str, str]
) -> tuple[bytes, dict[str, str]]:
    """The body (encode) and headers of the route for query q: axis (0, 1, 2), mm and
    box ("x0,y0,z0,x1,y1,z1", world mm). Raises ValueError for a bad query."""
    for k in ("axis", "mm", "box"):
        if k not in q:
            raise ValueError(f"missing {k}")
    try:
        axis = int(q["axis"])
    except ValueError as e:
        raise ValueError(f"axis: not an integer: {q['axis']!r}") from e
    if axis not in (0, 1, 2):
        raise ValueError(f"axis {axis} is not 0, 1 or 2")
    mm = _floats(q["mm"], 1, "mm")[0]
    box = _floats(q["box"], 6, "box")
    o = outlines.cut(axis, mm, box[:3], box[3:])
    headers = {
        "X-Outline-Source": o.source,
        "X-Faces-Read": str(o.faces_read),
        "X-Faces-In-Box": str(o.faces_in_box),
        "X-Segments": str(len(o.segments)),
    }
    return encode(o), headers
