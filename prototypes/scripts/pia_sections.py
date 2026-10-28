"""Sections through the column volume at a trace, cut from the 3D white mesh (plan: reports/Crossing
paths and contacts from laminar profiles.md, *A pia editor on an inflated surface*).

The flat map (flat_map.py) only says which way is which: at the crosshair's root its differential
T_v gives the 3D tangent t_u of the map's horizontal axis; on the inflated surface (inflate.py) the
screen's right and up, carried to the true mesh, do the same. The u-section is the normal plane
through the root that holds t_u and the surface normal n; the v-section holds t_v = n x t_u
(orthogonal on the true surface by construction; the angle to the carried vertical is reported).
Each section: the plane's intersection with the true mesh, a polyline (a plane meets a manifold
mesh in curves that never branch), followed both ways from the root; distances are arc lengths
along it, signed (positive along t_u, or t_v). The traces shown are the subsampled ones rooted
within a band of the plane and of that curve. Ordering a band of vertices along its longest
geodesic instead (as volume_views.section_traces does) mixes two branches where the band forks,
measured on M1S1. The map's or the inflation's area distortion never changes what a section shows:
only the direction comes from them.
"""

import numpy as np
from scipy.sparse.csgraph import dijkstra


def directions(T, n):
    """Unit tangents (t_u, t_v) at a root from the map's differential T (3, 2) and the normal n:
    t_u the map's horizontal axis projected on the tangent plane, t_v = n x t_u; and the angle
    (degrees) between t_v and the map's own vertical axis (0 for an exactly conformal map)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = T[:, 0] - (T[:, 0] @ n) * n
    tu = a / np.linalg.norm(a)
    tv = np.cross(n, tu)
    b = T[:, 1] - (T[:, 1] @ n) * n
    if b @ tv < 0:  # the map is mirrored relative to n: keep the map's sense for v
        tv = -tv
    ang = float(np.degrees(np.arccos(np.clip(b @ tv / np.linalg.norm(b), -1, 1))))
    return tu, tv, ang


def section(vmm, G, root, x0, n, t, roots_of, half=8.0, band=0.12, f=None, vf=None):
    """Traces of the normal section through x0 (mm) along tangent t with surface normal n: the
    plane through x0 holding t and n cuts the mesh in polylines (a plane meets a manifold mesh in
    curves that never branch); the one through the root is followed both ways from the root, and
    distances are arc lengths along it. The traces shown are those rooted within `band` of the
    plane and of the curve, each at the arc length of its nearest curve point.
    vmm: mesh vertices (mm); G: the mesh graph (mm); f: faces (needed); root: the mesh vertex at x0;
    roots_of: per mesh vertex the trace rooted there (-1: none). Returns (traces, signed distances,
    positive along t), ordered."""
    from scipy.spatial import cKDTree

    m = np.cross(n, t)
    m /= np.linalg.norm(m)
    d = dijkstra(G, indices=root, limit=1.5 * half + 1.0)
    near = np.isfinite(d)
    sd = (vmm - x0) @ m
    sd = np.where(sd == 0, 1e-12, sd)  # a vertex on the plane counts as on its positive side
    fk = f[near[f].all(1)]
    # crossing edges and their points
    a = np.c_[fk[:, 0], fk[:, 1], fk[:, 2]].ravel()
    b = np.c_[fk[:, 1], fk[:, 2], fk[:, 0]].ravel()
    cr = sd[a] * sd[b] < 0
    if not cr.any():
        return np.zeros(0, int), np.zeros(0)
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
    nb = [[] for _ in range(len(pts))]
    for p_, q_ in seg:
        nb[p_].append(q_)
        nb[q_].append(p_)
    start = int(np.argmin(np.linalg.norm(pts - x0, axis=1)))

    def walk(first):
        path, arc, prev, cur, L = [start], [0.0], start, first, 0.0
        while cur is not None and L <= half + band:
            L += float(np.linalg.norm(pts[cur] - pts[prev]))
            path.append(cur)
            arc.append(L)
            nxt = [q for q in nb[cur] if q != prev]
            prev, cur = cur, (nxt[0] if nxt and nxt[0] != start else None)
        return path, arc

    ends = nb[start][:2]
    if not ends:
        return np.zeros(0, int), np.zeros(0)
    pa, aa = walk(ends[0])
    pb, ab = walk(ends[1]) if len(ends) > 1 else ([start], [0.0])
    sgn = 1.0 if (pts[pa[min(1, len(pa) - 1)]] - pts[start]) @ t >= 0 else -1.0
    cpath = np.r_[pb[::-1], pa[1:]]
    carc = np.r_[-sgn * np.array(ab[::-1]), sgn * np.array(aa[1:])]
    # the start point is not at x0 exactly: shift arc so that x0's projection is 0
    cand = np.flatnonzero(near & (roots_of >= 0) & (np.abs(sd) <= band))
    if not cand.size:
        return np.zeros(0, int), np.zeros(0)
    tree = cKDTree(pts[cpath])
    dist, k = tree.query(vmm[cand])
    ok = dist <= 1.5 * band + 1e-9
    tr, arc = roots_of[cand[ok]], carc[k[ok]]
    # the root itself at 0, and arc measured from it
    if roots_of[root] >= 0 and roots_of[root] in set(tr.tolist()):
        arc = arc - arc[tr == roots_of[root]][0]
    keep = np.abs(arc) <= half
    o = np.argsort(arc[keep], kind="stable")
    return tr[keep][o], arc[keep][o]


def marked_hits(pos, ns, traces, P, vox, within=0.3):
    """Per trace: (point index, depth index) of the user's points within `within` mm of its
    nearest sample, as edit_data.point_traces (one trace can see several points)."""
    out = {}
    if not len(P):
        return out
    Pm = P * vox
    for t in traces:
        p = pos[t, :ns[t]].astype(float) * vox
        d = np.linalg.norm(p[:, None] - Pm[None], axis=2)
        for q in np.flatnonzero(d.min(0) <= within):
            out.setdefault(int(t), []).append((int(q), int(np.argmin(d[:, q]))))
    return out
