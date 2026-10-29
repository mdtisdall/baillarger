"""Sections through the column volume at a trace, cut from the 3D white mesh (plan:
reports/Crossing paths and contacts from laminar profiles.md, *A pia editor on an
inflated surface*).

The flat map (baillarger.viewer.flatmap) only says which way is which: at the
crosshair's root its differential T_v gives the 3D tangent t_u of the map's horizontal
axis; on the inflated surface (baillarger.viewer.inflation) the screen's right and up,
carried to the true mesh, do the same. The u-section is the normal plane through the
root that holds t_u and the surface normal n; the v-section holds t_v = n x t_u
(orthogonal on the true surface by construction; the angle to the carried vertical is
reported). The pair can be turned together about n by an angle (rotate), for example to
bring one axis into a slice plane of the volume pane (snap): the frame that they come
from stays as it was. Each section: the plane's intersection with the true mesh, a
polyline (a plane meets a manifold mesh in curves that never branch), followed both ways
from the root; distances are arc lengths along it, signed (positive along t_u, or t_v).
The traces shown are the subsampled ones rooted within a band of the plane and of that
curve. Ordering a band of vertices along its longest geodesic instead (as
volume_views.section_traces does) mixes two branches where the band forks, measured on
M1S1. The map's or the inflation's area distortion never changes what a section shows:
only the direction comes from them.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.integer[Any]]


def directions(T: Float, n: Any) -> tuple[Float, Float, float]:
    """Unit tangents (t_u, t_v) at a root from the map's differential T (3, 2) and
    the normal n: t_u the map's horizontal axis projected on the tangent plane,
    t_v = n x t_u; and the angle (degrees) between t_v and the map's own vertical
    axis (0 for an exactly conformal map)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = T[:, 0] - (T[:, 0] @ n) * n
    tu = a / np.linalg.norm(a)
    tv = np.cross(n, tu)
    b = T[:, 1] - (T[:, 1] @ n) * n
    if b @ tv < 0:  # the map is mirrored relative to n: keep the map's sense for v
        tv = -tv
    ang = float(np.degrees(np.arccos(np.clip(b @ tv / np.linalg.norm(b), -1, 1))))
    return tu, tv, ang


def rotate(tu: Float, tv: Float, deg: float) -> tuple[Float, Float]:
    """The pair (t_u, t_v) turned by `deg` degrees in its own plane, from t_u toward
    t_v (about n when t_v = n x t_u): 90 gives (t_v, -t_u)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return c * tu + s * tv, -s * tu + c * tv


def snap(
    tu: Float, tv: Float, n: Float, g: Float, deg: float
) -> tuple[float, str, float] | None:
    """The rotation (degrees, in (-180, 180]) nearest `deg` that puts one axis of the
    pair (t_u, t_v), turned by it (rotate), in a plane with normal g: the axis then
    lies in that plane and the section through it is the normal plane nearest to it.
    Returns (rotation, the axis "u" or "v", the tilt left between the section plane
    and that plane: asin |n . g|, degrees); None when g is along n (the tangent
    plane is that plane: every axis lies in it)."""
    g = np.asarray(g, float) / np.linalg.norm(g)
    a, b = float(tu @ g), float(tv @ g)
    if np.hypot(a, b) < 1e-9:
        return None
    th = np.degrees(np.arctan2(-a, b))  # t_u(th) . g = cos th a + sin th b = 0
    k = int(np.round((deg - th) / 90.0))
    r = th + 90.0 * k
    r = float(r - 360.0 * np.ceil((r - 180.0) / 360.0))  # into (-180, 180]
    tilt = float(np.degrees(np.arcsin(min(1.0, abs(float(n @ g)) / np.linalg.norm(n)))))
    return r, ("u" if k % 2 == 0 else "v"), tilt


def section(
    vmm: Float,
    G: csr_matrix,
    root: int,
    x0: Float,
    n: Float,
    t: Float,
    roots_of: Int,
    half: float = 8.0,
    band: float = 0.12,
    f: Int | None = None,
    vf: Any = None,
) -> tuple[Int, Float]:
    """Traces of the normal section through x0 (mm) along tangent t with surface
    normal n: the plane through x0 holding t and n cuts the mesh in polylines (a
    plane meets a manifold mesh in curves that never branch); the one through the
    root is followed both ways from the root, and distances are arc lengths along
    it. The traces shown are those rooted within `band` of the plane and of the
    curve, each at the arc length of its nearest curve point.
    vmm: mesh vertices (mm); G: the mesh graph (mm); f: faces (needed); root: the
    mesh vertex at x0; roots_of: per mesh vertex the trace rooted there (-1: none).
    Returns (traces, signed distances, positive along t), ordered."""
    tr, arc, _ = section_curve(vmm, G, root, x0, n, t, roots_of, half, band, f)
    return tr, arc


def section_curve(
    vmm: Float,
    G: csr_matrix,
    root: int,
    x0: Float,
    n: Float,
    t: Float,
    roots_of: Int,
    half: float = 8.0,
    band: float = 0.12,
    f: Int | None = None,
) -> tuple[Int, Float, Float]:
    """section, and the curve it follows: its points (mm, in order along t) within
    `half` of the start, the crossing nearest x0 (empty when the plane misses).
    Returns (traces, signed distances, curve points)."""
    from scipy.spatial import cKDTree

    none = np.zeros(0, int), np.zeros(0), np.zeros((0, 3))
    m = np.cross(n, t)
    m /= np.linalg.norm(m)
    d = dijkstra(G, indices=root, limit=1.5 * half + 1.0)
    near = np.isfinite(d)
    sd = (vmm - x0) @ m
    # a vertex on the plane counts as on its positive side
    sd = np.where(sd == 0, 1e-12, sd)
    assert f is not None, "section needs the faces f"
    fk = f[near[f].all(1)]
    # crossing edges and their points
    a = np.c_[fk[:, 0], fk[:, 1], fk[:, 2]].ravel()
    b = np.c_[fk[:, 1], fk[:, 2], fk[:, 0]].ravel()
    cr = sd[a] * sd[b] < 0
    if not cr.any():
        return none
    lo_, hi_ = np.minimum(a, b), np.maximum(a, b)
    key = lo_ * len(vmm) + hi_
    ukey, eid = np.unique(key[cr], return_inverse=True)
    ea, eb = ukey // len(vmm), ukey % len(vmm)
    w = sd[ea] / (sd[ea] - sd[eb])
    pts = vmm[ea] + w[:, None] * (vmm[eb] - vmm[ea])
    # each crossing face joins its two crossing edges
    fe = -np.ones(cr.size, int)
    fe[cr] = eid
    fe = fe.reshape(-1, 3)
    seg = np.sort(fe, 1)[:, 1:]
    seg = seg[(seg >= 0).all(1)]
    nb: list[list[int]] = [[] for _ in range(len(pts))]
    for p_, q_ in seg:
        nb[p_].append(q_)
        nb[q_].append(p_)
    start = int(np.argmin(np.linalg.norm(pts - x0, axis=1)))

    def walk(first: int) -> tuple[list[int], list[float]]:
        path, arc, prev, L = [start], [0.0], start, 0.0
        cur: int | None = first
        while cur is not None and L <= half + band:
            L += float(np.linalg.norm(pts[cur] - pts[prev]))
            path.append(cur)
            arc.append(L)
            nxt = [q for q in nb[cur] if q != prev]
            prev, cur = cur, (nxt[0] if nxt and nxt[0] != start else None)
        return path, arc

    ends = nb[start][:2]
    if not ends:
        return none
    pa, aa = walk(ends[0])
    pb, ab = walk(ends[1]) if len(ends) > 1 else ([start], [0.0])
    sgn = 1.0 if (pts[pa[min(1, len(pa) - 1)]] - pts[start]) @ t >= 0 else -1.0
    cpath = np.r_[pb[::-1], pa[1:]]
    carc = np.r_[-sgn * np.array(ab[::-1]), sgn * np.array(aa[1:])]
    o = np.argsort(carc, kind="stable")
    on = np.abs(carc[o]) <= half
    curve = pts[cpath[o][on]]
    # the start point is not at x0 exactly: shift arc so that x0's projection is 0
    cand = np.flatnonzero(near & (roots_of >= 0) & (np.abs(sd) <= band))
    if not cand.size:
        return np.zeros(0, int), np.zeros(0), curve
    tree = cKDTree(pts[cpath])
    dist, k_ = tree.query(vmm[cand])
    k: Int = np.asarray(k_)
    ok = dist <= 1.5 * band + 1e-9
    tr, arc = roots_of[cand[ok]], carc[k[ok]]
    # the root itself at 0, and arc measured from it
    if roots_of[root] >= 0 and roots_of[root] in set(tr.tolist()):
        arc = arc - arc[tr == roots_of[root]][0]
    keep = np.abs(arc) <= half
    o = np.argsort(arc[keep], kind="stable")
    return tr[keep][o], arc[keep][o], curve


def marked_hits(
    pos: Any,
    ns: Any,
    traces: Any,
    P: Float,
    vox: Float,
    within: float = 0.3,
) -> dict[int, list[tuple[int, int]]]:
    """Per trace: (point index, depth index) of the user's points within `within` mm
    of its nearest sample, as edit_data.point_traces (one trace can see several
    points)."""
    out: dict[int, list[tuple[int, int]]] = {}
    if not len(P):
        return out
    Pm = P * vox
    for t in traces:
        p = pos[t, : ns[t]].astype(float) * vox
        d = np.linalg.norm(p[:, None] - Pm[None], axis=2)
        for q in np.flatnonzero(d.min(0) <= within):
            out.setdefault(int(t), []).append((int(q), int(np.argmin(d[:, q]))))
    return out
