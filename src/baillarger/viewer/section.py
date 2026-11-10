"""The profile section that follows the surface (GET /columns/section).

The columns pane shows a profile section: columns side by side, the echo along
each. The section here follows the white surface, so that the section itself
makes no bands:

1. The curve (plane_cut). The plane through the origin (the selected root)
   that holds the root's normal n and the section axis u (u is first made
   normal to n). Its cut with the mesh is a set of polylines: one point on
   each mesh edge whose ends lie on different sides of the plane (a vertex on
   the plane counts as on the + side), one segment in each face that has such
   edges. The curve is the piece of the cut that goes through the origin: from
   the point of the cut nearest the origin, along the segments in both
   directions, to an arc length of half_length each way, or to where the piece
   ends. Where the surface turns round a thin edge, the curve goes round the
   turn. Arc length is 0 at that nearest point and positive on the side where
   the curve first goes in the direction of +u.
   A slice plane instead (slice_axes, section_curve): the plane through the
   origin with a given normal (a slice of a volume view). The curve is its cut
   with the mesh, as above; +u is u made to lie in the plane. If the plane's
   cut does not go through the origin, or ends less than one step from it on
   either side (the slice is near the surface's tangent plane at the origin),
   the plane of n and u is used, and the reason is returned.
2. The steps (steps). Points on the curve every `step` mm of arc length, from
   0 both ways, and the face that holds each point.
3. The columns (step_columns). At each step, the distances from the curve
   point along the surface: along the mesh graph (baillarger.mesh_graph: the
   mesh edges and the edges across neighbouring triangles, in mm), from the
   point to the vertices of its face in a straight line. Mode "nearest": the
   column whose root is the nearest along the surface. Mode "mean": every
   column whose root is within one voxel along the surface.

Along the surface, not in space: where the WM is one voxel thick, the roots of
the other side are near in space but far along the surface (round the edge),
so they do not come into the section at that place.

The distances are exact and need only the vertices near the point: a path of
length <= rho from the point visits only vertices within rho of it in space
(each edge is a straight line, no shorter than the distance it spans). So the
graph restricted to the vertices in a ball of radius rho round the point
(with the face's vertices) gives every distance up to rho exactly. "nearest"
searches with rho = the largest distance to a vertex of the face, doubled until
a column is found or rho passes `limit`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

type Float = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

MODES = ("nearest", "mean")
#: the edges of a face, as pairs of its corners
EDGES = ((0, 1), (1, 2), (2, 0))


@dataclass(frozen=True)
class Curve:
    """A piece of a plane's cut with a mesh, in order of arc length."""

    #: (K, 3) the points of the curve, mm
    point: Float
    #: (K,) the arc length of each point from the origin's nearest point, mm
    arc: Float
    #: (K - 1,) the face that holds each segment (point k to point k + 1)
    face: I64
    #: the unit section axis (u made normal to n) and the plane's unit normal
    u: Float
    w: Float


def axes(normal: npt.ArrayLike, u: npt.ArrayLike) -> tuple[Float, Float]:
    """The unit section axis (u without its part along the normal) and the
    plane's unit normal w = n x u."""
    n = np.asarray(normal, np.float64)
    n = n / np.linalg.norm(n)
    a = np.asarray(u, np.float64)
    a = a - (a @ n) * n
    if not np.linalg.norm(a) > 1e-9:
        raise ValueError("u is parallel to the normal")
    a = a / np.linalg.norm(a)
    w = np.cross(n, a)
    return a, w / np.linalg.norm(w)


def slice_axes(
    normal: npt.ArrayLike, u: npt.ArrayLike, plane: npt.ArrayLike
) -> tuple[Float, Float]:
    """For a slice plane with normal `plane`: the unit section axis (u made to
    lie in the plane; if u is normal to it, the direction of plane x normal)
    and the plane's unit normal, signed so that (normal x axis) . w >= 0 (so a
    plane that holds the normal and u gives the axes of `axes`)."""
    n = np.asarray(normal, np.float64)
    n = n / np.linalg.norm(n)
    w = np.asarray(plane, np.float64)
    if not np.linalg.norm(w) > 1e-9:
        raise ValueError("plane: the normal is zero")
    w = w / np.linalg.norm(w)
    a = np.asarray(u, np.float64)
    a = a - (a @ w) * w
    if not np.linalg.norm(a) > 1e-9:
        a = np.cross(w, n)
    if not np.linalg.norm(a) > 1e-9:  # the plane is normal to n: any axis in it
        a = np.cross(w, [1.0, 0.0, 0.0] if abs(w[0]) < 0.9 else [0.0, 1.0, 0.0])
    a = a / np.linalg.norm(a)
    if np.cross(n, a) @ w < 0:
        w = -w
    return a, w


def plane_cut(
    pts: Float,
    faces: npt.NDArray[np.integer[Any]],
    origin: npt.ArrayLike,
    normal: npt.ArrayLike,
    u: npt.ArrayLike,
    half_length: float,
    plane: npt.ArrayLike | None = None,
) -> Curve:
    """The piece of the cut through the origin (see the module docstring), of
    the plane of the normal and u, or of the slice plane with normal `plane`.
    pts: (n, 3) the vertices, mm; faces: (m, 3) into pts."""
    o = np.asarray(origin, np.float64)
    ua, w = axes(normal, u) if plane is None else slice_axes(normal, u, plane)
    f = np.asarray(faces, np.int64)
    d = (np.asarray(pts, np.float64) - o) @ w
    side = d >= 0
    s = side[f]
    cut = s.any(1) & ~s.all(1)
    # only faces that can hold a point within half_length of the origin in space
    # (an arc of half_length stays within half_length in space)
    fc = f[cut]
    p = pts[fc]
    cen = p.mean(1)
    reach = np.linalg.norm(p - cen[:, None], axis=2).max(1)
    near = np.linalg.norm(cen - o, axis=1) <= half_length + reach
    fc = fc[near]
    face_id = np.flatnonzero(cut)[near]
    if not len(fc):
        raise ValueError("the plane does not cut the mesh near the origin")
    # the two crossing edges of each face: their keys and points
    keys = np.zeros((len(fc), 2), np.int64)
    xyz = np.zeros((len(fc), 2, 3))
    k = np.zeros(len(fc), np.int64)
    n_v = len(pts)
    rows = np.arange(len(fc))
    for i, j in EDGES:
        a, b = fc[:, i], fc[:, j]
        x = side[a] != side[b]
        r = rows[x]
        a, b = a[x], b[x]
        t = d[a] / (d[a] - d[b])
        keys[r, k[r]] = np.minimum(a, b) * n_v + np.maximum(a, b)
        xyz[r, k[r]] = pts[a] + t[:, None] * (pts[b] - pts[a])
        k[r] += 1
    assert np.all(k == 2)
    nodes, ends = np.unique(keys, return_inverse=True)
    ends = ends.reshape(-1, 2)
    node_xyz = np.zeros((len(nodes), 3))
    node_xyz[ends.ravel()] = xyz.reshape(-1, 3)
    # the segments at each node
    order = np.argsort(ends.ravel(), kind="stable")
    start = np.searchsorted(ends.ravel()[order], np.arange(len(nodes) + 1))
    seg_at = order // 2
    # the segment nearest the origin, and the nearest point on it
    a, b = node_xyz[ends[:, 0]], node_xyz[ends[:, 1]]
    ab = b - a
    ll = (ab * ab).sum(1)
    t = np.where(ll > 0, ((o - a) * ab).sum(1) / np.where(ll > 0, ll, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[:, None] * ab
    s0 = int(np.argmin(np.linalg.norm(q - o, axis=1)))
    q0 = q[s0]
    used = np.zeros(len(fc), bool)
    used[s0] = True

    def walk(node: int) -> tuple[list[Float], list[int]]:
        """From q0 to node (on segment s0), then on along unused segments, to
        half_length: the points after q0 and the face of each segment."""
        out: list[Float] = []
        fs: list[int] = [int(face_id[s0])]
        last = q0
        length = 0.0
        while True:
            x = node_xyz[node]
            step = float(np.linalg.norm(x - last))
            if length + step >= half_length:
                f_ = (half_length - length) / step if step > 0 else 0.0
                out.append(last + f_ * (x - last))
                return out, fs
            out.append(x)
            length += step
            last = x
            nxt = [int(g) for g in seg_at[start[node] : start[node + 1]] if not used[g]]
            if not nxt:
                return out, fs
            g = nxt[0]
            used[g] = True
            fs.append(int(face_id[g]))
            node = int(ends[g, 1] if ends[g, 0] == node else ends[g, 0])

    pa, fa = walk(int(ends[s0, 0]))
    pb, fb = walk(int(ends[s0, 1]))

    def first_u(points: list[Float]) -> float:
        for x in points:
            if np.linalg.norm(x - q0) > 0:
                return float((x - q0) @ ua)
        return 0.0

    if first_u(pa) > first_u(pb):  # side a is the + side
        pa, fa, pb, fb = pb, fb, pa, fa
    point = np.array([*pa[::-1], q0, *pb])
    seg = np.linalg.norm(np.diff(point, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    arc -= arc[len(pa)]
    face = np.array([*fa[::-1], *fb], np.int64)
    return Curve(point, arc, face, ua, w)


def section_curve(
    pts: Float,
    faces: npt.NDArray[np.integer[Any]],
    origin: npt.ArrayLike,
    normal: npt.ArrayLike,
    u: npt.ArrayLike,
    half_length: float,
    step: float,
    plane: npt.ArrayLike | None,
) -> tuple[Curve, str | None]:
    """The section's curve: the slice plane's cut if `plane` is given and its
    cut goes through the origin and reaches one step on both sides, else the
    cut of the plane of the normal and u (see the module docstring); and the
    reason when a given slice plane was not used."""
    if plane is None:
        return plane_cut(pts, faces, origin, normal, u, half_length), None
    o = np.asarray(origin, np.float64)
    why = None
    try:
        c = plane_cut(pts, faces, origin, normal, u, half_length, plane)
    except ValueError:
        why = "the slice does not cut the surface near the root"
    else:
        q0 = c.point[int(np.argmin(np.abs(c.arc)))]
        reach = min(step, half_length)
        # through the origin: the origin lies on a crossing edge (float error)
        if np.linalg.norm(q0 - o) > 1e-9 * (1.0 + np.linalg.norm(o)):
            why = "the slice's cut does not go through the root"
        elif -c.arc[0] < reach or c.arc[-1] < reach:
            why = "the slice's cut through the root ends within one step"
        else:
            return c, None
    return plane_cut(pts, faces, origin, normal, u, half_length), why


def steps(curve: Curve, step: float) -> tuple[Float, Float, I64]:
    """The points of the curve every `step` of arc length from 0 (both ways):
    their arc lengths, positions and faces."""
    if not step > 0:
        raise ValueError(f"step {step}: not > 0")
    a = curve.arc
    k0 = int(np.ceil(a[0] / step - 1e-9))
    k1 = int(np.floor(a[-1] / step + 1e-9))
    s = np.arange(k0, k1 + 1) * step
    s = np.clip(s, a[0], a[-1])
    j = np.clip(np.searchsorted(a, s, side="right") - 1, 0, len(a) - 2)
    span = a[j + 1] - a[j]
    t = np.where(span > 0, (s - a[j]) / np.where(span > 0, span, 1), 0.0)
    p = curve.point[j] + t[:, None] * (curve.point[j + 1] - curve.point[j])
    return s, p, curve.face[j]


def distances_from(
    p: Float,
    seeds: I64,
    pts: Float,
    graph: csr_matrix,
    tree: cKDTree,
    rho: float,
) -> tuple[I64, Float]:
    """The distances along the surface from point p, which lies in the face
    with vertices `seeds`, to the vertices within rho (see the module
    docstring): the vertices and their distances (inf beyond rho)."""
    ball = np.asarray(tree.query_ball_point(p, rho), np.int64)
    nodes = np.unique(np.concatenate([ball, seeds]))
    sub = graph[nodes][:, nodes].tocoo()
    n = len(nodes)
    off = np.linalg.norm(pts[seeds] - p, axis=1)
    at = np.searchsorted(nodes, seeds)
    rows = np.concatenate([sub.row, np.full(len(seeds), n)])
    cols = np.concatenate([sub.col, at])
    vals = np.concatenate([sub.data, off])
    # a zero offset would be dropped as a missing edge: the point is the vertex
    vals = np.where(vals > 0, vals, np.finfo(np.float64).tiny)
    aug = coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    d = dijkstra(aug, directed=True, indices=n, limit=rho)[:n]
    # tiny + a positive edge is that edge: only the point's own vertex has tiny
    d[d == np.finfo(np.float64).tiny] = 0.0
    return nodes, d


def step_columns(
    points: Float,
    faces_of: I64,
    pts: Float,
    faces: npt.NDArray[np.integer[Any]],
    graph: csr_matrix,
    tree: cKDTree,
    column_of: I64,
    mode: str,
    voxel: float,
    limit: float,
) -> tuple[list[list[int]], Float]:
    """The columns at each step (see the module docstring): for each point, the
    column indices (column_of: the column of each vertex, -1 for none), and the
    distance along the surface (nearest: of its column; mean: of the farthest
    one; NaN: none)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: not one of {MODES}")
    f = np.asarray(faces, np.int64)
    out: list[list[int]] = []
    dist = np.full(len(points), np.nan)
    for i, (p, fi) in enumerate(zip(points, faces_of, strict=True)):
        seeds = f[fi]
        if mode == "mean":
            nodes, d = distances_from(p, seeds, pts, graph, tree, voxel)
            c = column_of[nodes]
            ok = (c >= 0) & (d <= voxel)
            cs = sorted(int(x) for x in c[ok])
            out.append(cs)
            if ok.any():
                dist[i] = float(d[ok].max())
            continue
        rho = float(np.linalg.norm(pts[seeds] - p, axis=1).max())
        rho = max(rho, np.finfo(np.float64).tiny)
        best: list[int] = []
        while True:
            nodes, d = distances_from(p, seeds, pts, graph, tree, rho)
            c = column_of[nodes]
            ok = (c >= 0) & (d <= rho)
            if ok.any():
                k = np.flatnonzero(ok)
                k = k[np.lexsort((c[k], d[k]))][0]
                best = [int(c[k])]
                dist[i] = float(d[k])
                break
            if rho >= limit:
                break
            rho = min(2 * rho, limit)
        out.append(best)
    return out, dist
