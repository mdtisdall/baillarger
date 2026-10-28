"""One conformal flat map of a box's white surface, for the pia editor (plan: reports/Crossing paths
and contacts from laminar profiles.md, *A pia editor on an inflated surface*). Replaces local charts
(chart.py) for navigation only: the editor's sections are cut from the 3D mesh, never from the map.

1. Topology per connected piece of the white mesh: vertices, edges, faces, boundary loops, genus
   (chi = V - E + F = 2 - 2g - b).
2. Handles: a piece of genus g > 0 is cut along a short non-separating loop, g times, each one the
   shortest loop found by the greedy homotopy basis (Erickson & Whittlesey 2005: shortest-path tree
   from a root, the dual of the other edges spanned by a maximum tree on the loop lengths, the
   2g edges left over close the shortest basis loops through the root), the boundary loops filled
   by cones first so that a loop round a hole is not a handle. Roots: a few random vertices, then
   vertices on the best loop so far, until it stops shortening. A thin handle (a white-matter bridge
   in the segmentation, say) gives its neck as the shortest loop; the cut turns it into two holes.
   Optionally (to_disk) the boundary loops are then joined by shortest paths to give a disk.
3. Least-squares conformal map (LSCM; Levy et al. 2002, in the form of Desbrun et al. 2002) of each
   genus-0 piece: minimise the conformal energy E_D - A (Dirichlet energy of the cotangent
   Laplacian minus the signed area from the boundary edges) with two pinned vertices, free
   boundary; one sparse solve. Each piece is then scaled to its 3D area, turned to its principal
   axes and packed in one map (mm).
4. Per face, the map's differential: T (3 x 2), the 3D images of the map's unit axes; its singular
   values s1 >= s2 give the area factor 1 / (s1 s2) and the conformal distortion s1 / s2.

Usage: flat_map.py <figures> <work> <site>     (writes <work>/flat_map_<site>.npz, .json and the
       distortion view <figures>/flat_map_<site>_distortion.png)
"""

import json
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.sparse.linalg import spsolve

CONE_W = 1e6  # length of an edge to a cone vertex (mm): no loop goes through a filled hole


# ---------------------------------------------------------------- topology

def edges_of(f):
    """Undirected edges (m, 2), lower vertex first, and for each half-edge (face k, corner c:
    f[k, c] -> f[k, c+1]) its edge index (nf, 3)."""
    a, b = f, np.roll(f, -1, axis=1)
    lo, hi = np.minimum(a, b).ravel(), np.maximum(a, b).ravel()
    key = lo.astype(np.int64) * (int(f.max()) + 1) + hi
    uk, inv = np.unique(key, return_inverse=True)
    E = np.c_[uk // (int(f.max()) + 1), uk % (int(f.max()) + 1)]
    return E, inv.reshape(f.shape)


def boundary_loops(f):
    """Boundary loops as ordered vertex arrays, each in its faces' half-edge direction."""
    E, he = edges_of(f)
    cnt = np.bincount(he.ravel(), minlength=len(E))
    assert cnt.max() <= 2, "non-manifold edge"
    k, c = np.nonzero(cnt[he] == 1)
    a, b = f[k, c], f[k, (c + 1) % 3]
    nxt = dict(zip(a.tolist(), b.tolist()))
    assert len(nxt) == len(a), "a vertex on two boundary loops (pinched)"
    loops, seen = [], set()
    for s in a.tolist():
        if s in seen:
            continue
        loop = [s]
        seen.add(s)
        x = nxt[s]
        while x != s:
            loop.append(x)
            seen.add(x)
            x = nxt[x]
        loops.append(np.array(loop))
    return loops


def pieces(f, nv):
    """Connected pieces: label per vertex (vertices in no face get -1) and per face."""
    E, _ = edges_of(f)
    G = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(nv, nv))
    _, lab = connected_components(G, directed=False)
    used = np.zeros(nv, bool)
    used[f.ravel()] = True
    _, lab = np.unique(np.where(used, lab, -1), return_inverse=True)
    lab = lab - (0 if used.all() else 1)
    return lab, lab[f[:, 0]]


def topology(f):
    """V, E, F, boundary loops b, chi, genus of one connected oriented manifold mesh."""
    used = np.unique(f)
    E, _ = edges_of(f)
    b = len(boundary_loops(f))
    chi = used.size - len(E) + len(f)
    g2 = 2 - chi - b
    assert g2 % 2 == 0, (chi, b)
    return dict(V=int(used.size), E=int(len(E)), F=int(len(f)), b=b, chi=int(chi), genus=g2 // 2)


def orientable_check(f):
    """Each interior undirected edge used once in each direction."""
    a, b = f.ravel(), np.roll(f, -1, axis=1).ravel()
    key = a.astype(np.int64) * (int(f.max()) + 1) + b
    return np.unique(key).size == key.size


# ---------------------------------------------------------------- cuts

def cut(f, cut_edges):
    """Cut the mesh along undirected edges (k, 2): faces on the two sides of a cut edge no longer
    share its vertices. Returns new faces and orig (new vertex -> old vertex)."""
    nf = len(f)
    E, he = edges_of(f)
    nv1 = int(f.max()) + 1
    ce = np.sort(np.asarray(cut_edges, np.int64).reshape(-1, 2), 1)
    ckey = set((ce[:, 0] * nv1 + ce[:, 1]).tolist())
    is_cut = np.array([int(e0) * nv1 + int(e1) in ckey for e0, e1 in E]) if ckey else np.zeros(len(E), bool)
    assert is_cut.sum() == len(ckey), "a cut edge is not a mesh edge"
    # half-edges by edge: pairs (k1, c1), (k2, c2) with f[k1, c1] = a -> b and f[k2, c2] = b -> a
    h = he.ravel()
    order = np.argsort(h, kind="stable")
    hs = h[order]
    two = np.flatnonzero(hs[1:] == hs[:-1])
    h1, h2 = order[two], order[two + 1]
    keep = ~is_cut[hs[two]]
    h1, h2 = h1[keep], h2[keep]
    k1, c1, k2, c2 = h1 // 3, h1 % 3, h2 // 3, h2 % 3
    # corner ids 3k + c; a -> b in face k1 is corners c1 (a), c1+1 (b); in face k2 b is c2, a is c2+1
    ra = np.r_[3 * k1 + c1, 3 * k1 + (c1 + 1) % 3]
    rb = np.r_[3 * k2 + (c2 + 1) % 3, 3 * k2 + c2]
    G = coo_matrix((np.ones(ra.size), (ra, rb)), shape=(3 * nf, 3 * nf))
    _, lab = connected_components(G, directed=False)
    f2 = lab.reshape(nf, 3)
    orig = np.zeros(lab.max() + 1, np.int64)
    orig[lab] = f.ravel()
    return f2, orig


def _cone_fill(f, nv):
    """Fill each boundary loop by a cone vertex. Returns faces, cone vertex ids."""
    loops = boundary_loops(f)
    extra, cones = [], []
    for i, lp in enumerate(loops):
        c = nv + i
        cones.append(c)
        extra.append(np.c_[np.roll(lp, -1), lp, np.full(lp.size, c)])
    return (np.vstack([f] + extra) if extra else f), np.array(cones, np.int64)


def greedy_loops(v, f, root, cones=()):
    """The shortest homotopy basis loops through root (vertex cycles, closed: first == last not
    repeated) of a closed oriented mesh f (cones: vertices whose edges get length CONE_W).
    Returns loops sorted by length, with their lengths."""
    nv = len(v)
    E, he = edges_of(f)
    w = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1)
    isc = np.zeros(nv, bool)
    isc[list(cones)] = True
    w = np.where(isc[E[:, 0]] | isc[E[:, 1]], CONE_W, w)
    w = np.maximum(w, 1e-9)
    G = csr_matrix((w, (E[:, 0], E[:, 1])), shape=(nv, nv))
    d, pred = dijkstra(G, directed=False, indices=root, return_predecessors=True)
    tree = np.zeros(len(E), bool)
    ch = np.flatnonzero(pred >= 0)
    lo, hi = np.minimum(ch, pred[ch]), np.maximum(ch, pred[ch])
    key = E[:, 0] * nv + E[:, 1]
    srt = np.argsort(key)
    tree[srt[np.searchsorted(key[srt], lo * nv + hi)]] = True
    # dual: faces across each non-tree edge
    h = he.ravel()
    order = np.argsort(h, kind="stable")
    hs = h[order]
    two = np.flatnonzero(hs[1:] == hs[:-1])
    e_of = hs[two]
    fa, fb = order[two] // 3, order[two + 1] // 3
    nt = ~tree[e_of]
    e_nt, fa, fb = e_of[nt], fa[nt], fb[nt]
    ell = d[E[e_nt, 0]] + w[e_nt] + d[E[e_nt, 1]]
    big = ell.max() + 1.0
    D = coo_matrix((big - ell, (fa, fb)), shape=(len(f), len(f))).tocsr()
    assert D.nnz == len(e_nt), "two faces share two edges"
    T = minimum_spanning_tree(D).tocoo()
    in_cot = set(zip(T.row.tolist(), T.col.tolist()))
    gens = [i for i in range(len(e_nt)) if (int(fa[i]), int(fb[i])) not in in_cot
            and (int(fb[i]), int(fa[i])) not in in_cot]
    out = []
    for i in gens:
        a, b = E[e_nt[i]]
        pa, pb = _path(pred, a), _path(pred, b)  # vertex -> root
        sb = set(pb)
        k = next(t for t, x in enumerate(pa) if x in sb)
        common = pa[k]
        # common -> ... -> a, then b -> ... -> the child of common on b's side
        loop = pa[:k + 1][::-1] + pb[:pb.index(common)]
        out.append((float(ell[i] - 2 * d[common]), np.array(loop, np.int64)))
    out.sort(key=lambda t: t[0])
    return [lp for _, lp in out], np.array([L for L, _ in out])


def _path(pred, x):
    p = [int(x)]
    while pred[p[-1]] >= 0:
        p.append(int(pred[p[-1]]))
    return p


def shortest_handle(v, f, cones=(), seed=0, n_random=4, n_along=8, rounds=4):
    """A short non-separating loop of a closed mesh (cones: as greedy_loops): the shortest basis
    loop over random roots, then over roots spread along the best loop so far, until no shorter
    one is found. Returns (loop, length, tried roots)."""
    rng = np.random.default_rng(seed)
    isc = np.zeros(len(v), bool)
    isc[list(cones)] = True
    cand = np.flatnonzero(~isc & np.isin(np.arange(len(v)), f))
    best, bestL, tried = None, np.inf, set()

    def try_roots(roots):
        nonlocal best, bestL
        got = False
        for r in roots:
            if int(r) in tried:
                continue
            tried.add(int(r))
            loops, L = greedy_loops(v, f, int(r), cones)
            if len(loops) and L[0] < bestL - 1e-12:
                best, bestL, got = loops[0], L[0], True
        return got

    try_roots(rng.choice(cand, min(n_random, cand.size), replace=False))
    for _ in range(rounds):
        if best is None or not try_roots(best[np.linspace(0, best.size - 1, n_along).astype(int)]):
            break
    return best, float(bestL), len(tried)


def loop_edges(loop):
    return np.c_[loop, np.roll(loop, -1)]


def cut_handles(v, f, seed=0, log=None):
    """Cut one connected mesh along short non-separating loops until it has genus 0. Returns new
    faces, orig (new vertex -> input vertex) and the loops (input vertex ids, lengths)."""
    orig = np.arange(len(v))
    loops = []
    g = topology(f)["genus"]
    while g > 0:
        nv = int(f.max()) + 1
        fc, cones = _cone_fill(f, nv)
        vv = v[orig[:nv]]
        if cones.size:
            bl = boundary_loops(f)
            vv = np.vstack([vv] + [vv[lp].mean(0, keepdims=True) for lp in bl])
        lp, L, nr = shortest_handle(vv, fc, cones, seed=seed)
        assert lp is not None and not np.isin(lp, cones).any(), "no handle loop off the cones"
        f, o2 = cut(f, loop_edges(lp))
        loops.append((orig[lp], L))
        orig = orig[o2]
        t = topology(f)
        assert t["genus"] == g - 1 and pieces(f, int(f.max()) + 1)[0].max() == 0, t
        if log:
            log(f"handle {len(loops)}: loop of {lp.size} vertices, {L:.2f} mm ({nr} roots); now {t}")
        g = t["genus"]
    return f, orig, loops


def join_boundaries(v, f):
    """Cut shortest interior paths from the first boundary loop to the others until one is left
    (a disk if the mesh has genus 0). Returns new faces and orig."""
    orig = np.arange(int(f.max()) + 1)
    while True:
        bl = boundary_loops(f)
        if len(bl) <= 1:
            return f, orig
        nv = int(f.max()) + 1
        vv = v[orig]
        onb = np.zeros(nv, bool)
        onb[np.concatenate(bl)] = True
        E, _ = edges_of(f)
        E = E[~(onb[E[:, 0]] & onb[E[:, 1]])]
        w = np.maximum(np.linalg.norm(vv[E[:, 0]] - vv[E[:, 1]], axis=1), 1e-9)
        G = csr_matrix((w, (E[:, 0], E[:, 1])), shape=(nv, nv))
        d, pred, _ = dijkstra(G, directed=False, indices=bl[0], min_only=True, return_predecessors=True)
        others = np.concatenate(bl[1:])
        t = others[np.argmin(d[others])]
        assert np.isfinite(d[t]), "boundary loops not joined by an interior path"
        path = _path(pred, t)
        f, o2 = cut(f, np.c_[path[:-1], path[1:]])
        orig = orig[o2]


# ---------------------------------------------------------------- LSCM

def cot_laplacian(v, f):
    """Positive semi-definite: x^T L x = sum over edges (cot a + cot b) / 2 (x_i - x_j)^2."""
    n = len(v)
    I, J, W = [], [], []
    for c in range(3):
        i, j, k = f[:, c], f[:, (c + 1) % 3], f[:, (c + 2) % 3]
        a, b = v[i] - v[k], v[j] - v[k]
        cot = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-15)
        I += [i, j]
        J += [j, i]
        W += [-cot / 2, -cot / 2]
    I, J, W = np.concatenate(I), np.concatenate(J), np.concatenate(W)
    L = coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    return (L - coo_matrix((np.asarray(L.sum(1)).ravel(), (np.arange(n), np.arange(n))), shape=(n, n))).tocsr()


def lscm(v, f, pins=None):
    """Least-squares conformal map of one connected genus-0 mesh with a boundary (mm in, mm out).
    pins: two vertex ids (default: two far-apart vertices), mapped to (0, 0) and (|v_p - v_q|, 0).
    Returns uv (n, 2)."""
    n = len(v)
    Lc = cot_laplacian(v, f)
    bl = boundary_loops(f)
    assert bl, "LSCM needs a boundary"
    a = np.concatenate(bl)
    b = np.concatenate([np.roll(lp, -1) for lp in bl])
    # signed area 1/2 sum (x_a y_b - x_b y_a) = u^T B u
    B = coo_matrix((np.r_[np.full(a.size, 0.5), np.full(a.size, -0.5)], (np.r_[a, b], np.r_[n + b, n + a])),
                   shape=(2 * n, 2 * n)).tocsr()
    from scipy.sparse import block_diag

    Q = (block_diag([Lc, Lc]) - B - B.T).tocsr()
    if pins is None:
        p = int(np.argmax(np.linalg.norm(v - v.mean(0), axis=1)))
        q = int(np.argmax(np.linalg.norm(v - v[p], axis=1)))
    else:
        p, q = (int(x) for x in pins)
    fixed = np.array([p, q, n + p, n + q])
    up = np.array([0.0, np.linalg.norm(v[p] - v[q]), 0.0, 0.0])
    free = np.setdiff1d(np.arange(2 * n), fixed)
    Qff = Q[free][:, free].tocsc()
    rhs = -(Q[free][:, fixed] @ up)
    u = np.zeros(2 * n)
    u[fixed] = up
    u[free] = spsolve(Qff, rhs)
    return np.c_[u[:n], u[n:]]


def tangent_maps(v, f, uv):
    """Per face: T (nf, 3, 2), the 3D images of the map's unit axes (T @ dq = dp on the face), and
    its singular values s1 >= s2, and the signed map area (2D)."""
    P = np.stack([v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]], axis=2)
    Qm = np.stack([uv[f[:, 1]] - uv[f[:, 0]], uv[f[:, 2]] - uv[f[:, 0]]], axis=2)
    det = Qm[:, 0, 0] * Qm[:, 1, 1] - Qm[:, 0, 1] * Qm[:, 1, 0]
    inv = np.stack([np.stack([Qm[:, 1, 1], -Qm[:, 0, 1]], 1), np.stack([-Qm[:, 1, 0], Qm[:, 0, 0]], 1)], 1)
    inv = inv / np.where(np.abs(det) > 0, det, np.nan)[:, None, None]
    T = P @ inv
    s = np.linalg.svd(np.nan_to_num(T), compute_uv=False)
    return T, s[:, 0], s[:, 1], det / 2


def face_areas(v, f):
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def flatten(v, f, to_disk=False, seed=0, gap=2.0, log=None):
    """Flat map of every connected piece of a manifold oriented mesh (v in mm). Pieces of genus > 0
    are cut along handle loops first. Each piece is scaled to its 3D area, turned to its principal
    axes, and the pieces are packed, largest first, in rows under the largest one.
    Returns a dict: f_cut, orig (cut vertex -> input vertex), uv (per cut vertex), piece (per cut
    vertex), pieces (topology before and after, per piece), loops [(input vertex ids, mm)], T, s1,
    s2 (per cut face), uv_v, T_v (per input vertex: its first copy's position and the area-weighted
    mean T of its faces; nan off the mesh)."""
    lab_v, lab_f = pieces(f, len(v))
    npc = lab_v.max() + 1
    out_f, out_o, out_uv, out_pc, info, loops = [], [], [], [], [], []
    base = 0
    for pc in range(npc):
        fk = f[lab_f == pc]
        vid = np.unique(fk)
        loc = -np.ones(len(v), np.int64)
        loc[vid] = np.arange(vid.size)
        fl = loc[fk]
        vl = v[vid]
        t0 = topology(fl)
        fl2, o, lp = cut_handles(vl, fl, seed=seed, log=log) if t0["genus"] > 0 else (fl, np.arange(vid.size), [])
        if to_disk:
            fl2, o3 = join_boundaries(vl[o], fl2)
            o = o[o3]
        loops += [(vid[a], L) for a, L in lp]
        vc = vl[o]
        uv = lscm(vc, fl2)
        A3, A2 = face_areas(vc, fl2).sum(), np.abs(tangent_maps(vc, fl2, uv)[3]).sum()
        uv = (uv - uv.mean(0)) * np.sqrt(A3 / A2)
        w, U = np.linalg.eigh(np.cov(uv.T))
        R = U[:, ::-1]
        if np.linalg.det(R) < 0:
            R[:, 1] *= -1
        uv = uv @ R
        info.append(dict(piece=pc, before=t0, after=topology(fl2), handles=len(lp), area_mm2=round(float(A3), 2)))
        out_f.append(fl2 + base)
        out_o.append(vid[o])
        out_uv.append(uv)
        out_pc.append(np.full(len(o), pc))
        base += len(o)
    # pack: largest piece first, the rest in rows below it
    order = np.argsort([-d["area_mm2"] for d in info])
    width = np.ptp(out_uv[order[0]][:, 0])
    x, y, row_h = 0.0, None, 0.0
    for r, pc in enumerate(order):
        uv = out_uv[pc]
        uv -= uv.min(0)
        if r == 0:
            y0 = uv[:, 1].max() + gap
            x, y = 0.0, y0
            continue
        wv, hv = np.ptp(uv[:, 0]), np.ptp(uv[:, 1])
        if x > 0 and x + wv > width:
            x, y, row_h = 0.0, y + row_h + gap, 0.0
        uv += [x, y]
        x += wv + gap
        row_h = max(row_h, hv)
    fc, orig, uvc = np.vstack(out_f), np.concatenate(out_o), np.vstack(out_uv)
    vc = v[orig]
    T, s1, s2, sa = tangent_maps(vc, fc, uvc)
    u_o, first = np.unique(orig, return_index=True)
    uv_v = np.full((len(v), 2), np.nan)
    uv_v[u_o] = uvc[first]
    a3 = face_areas(vc, fc)
    acc = np.zeros((len(orig), 3, 2))
    wsum = np.zeros(len(orig))
    for c in range(3):
        np.add.at(acc, fc[:, c], np.nan_to_num(T) * a3[:, None, None])
        np.add.at(wsum, fc[:, c], a3)
    Tc = acc / np.maximum(wsum, 1e-30)[:, None, None]
    T_v = np.full((len(v), 3, 2), np.nan)
    T_v[u_o] = Tc[first]
    return dict(f_cut=fc, orig=orig, uv=uvc, piece=np.concatenate(out_pc), pieces=info, loops=loops, T=T, s1=s1,
                s2=s2, signed_area=sa, area3=a3, uv_v=uv_v, T_v=T_v)


# ---------------------------------------------------------------- the sites

def wpct(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return [float(x[o][min(np.searchsorted(c, p / 100), x.size - 1)]) for p in q]


def distortion(out, faces=None):
    """Area-weighted percentiles of log2 area factor (2D / 3D) and of s1/s2 - 1, and folds."""
    k = np.ones(len(out["f_cut"]), bool) if faces is None else faces
    a, s1, s2, sa = out["area3"][k], out["s1"][k], out["s2"][k], out["signed_area"][k]
    ok = np.isfinite(s1) & (s2 > 0)
    la = -np.log2(s1[ok] * s2[ok])
    qc = s1[ok] / s2[ok] - 1
    return dict(log2_area_p1_p5_p50_p95_p99=[round(x, 2) for x in wpct(la, a[ok], [1, 5, 50, 95, 99])],
                conformal_p50_p95_p99=[round(x, 4) for x in wpct(qc, a[ok], [50, 95, 99])],
                folded_faces=int(np.sum(sa <= 0)), folded_area_mm2=round(float(a[sa <= 0].sum()), 3))


def main(fig, work, site, to_disk=False):
    import matplotlib

    matplotlib.use("Agg")

    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    Z = np.load(f"{work}/column_volume_{site}.npz")
    vox, lo = Z["vox"], Z["lo"]
    vb = Z["v"].astype(float)
    v, f = vb * vox, Z["f"].astype(np.int64)
    assert orientable_check(f)
    out = flatten(v, f, to_disk=to_disk, log=say)
    say(f"flattened {len(out['pieces'])} pieces")
    pc_f = out["piece"][out["f_cut"][:, 0]]
    big = int(np.argmax([d["area_mm2"] for d in out["pieces"]]))
    tv = Z["tv"]
    loops = []
    for i, (lp, L) in enumerate(out["loops"]):
        c = vb[lp].mean(0)
        loops.append(dict(handle=i + 1, vertices=int(lp.size), length_mm=round(L, 3),
                          centre_itksnap=[int(x) for x in np.round(c) + 1],
                          centre_full_grid0=[int(x) for x in np.round(c) + lo],
                          extent_mm=[round(float(x), 2) for x in np.ptp(v[lp], 0)],
                          to_box_face_mm=round(float(np.min(np.r_[c, np.array(Z["disp"].shape) - 1 - c]) * vox.min()), 2),
                          traces_rooted_within_1mm=int(np.sum(np.min(np.linalg.norm(
                              v[tv][:, None] - v[lp][None, ::max(1, lp.size // 50)], axis=2), 1) < 1.0))))
    summ = dict(site=site, to_disk=to_disk, pieces=out["pieces"], handles=loops,
                distortion_all=distortion(out), distortion_largest=distortion(out, pc_f == big),
                seconds=round(time.time() - t0, 1))
    tag = "" if not to_disk else "_disk"
    lens = np.array([lp.size for lp, _ in out["loops"]], np.int64)
    np.savez_compressed(f"{work}/flat_map_{site}{tag}.npz", uv_v=out["uv_v"], T_v=out["T_v"], f_cut=out["f_cut"],
                        orig=out["orig"], uv=out["uv"], piece=out["piece"], s1=out["s1"], s2=out["s2"],
                        area3=out["area3"], signed_area=out["signed_area"],
                        loop_vertices=np.concatenate([lp for lp, _ in out["loops"]]) if lens.size else np.zeros(0, int),
                        loop_sizes=lens, loop_mm=np.array([L for _, L in out["loops"]]))
    json.dump(summ, open(f"{work}/flat_map_{site}{tag}.json", "w"), indent=1)
    print(json.dumps({k: v_ for k, v_ in summ.items() if k != "pieces"}, indent=1))
    view(fig, site + tag, out, v, vb)
    say("done")


def view(fig, name, out, v, vb):
    """The map coloured by log2 area factor and by conformal distortion, with the handle cuts;
    histograms; a 3D view of the mesh round each handle loop."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    uv, fc = out["uv"], out["f_cut"]
    la = -np.log2(out["s1"] * out["s2"])
    qc = out["s1"] / out["s2"] - 1
    tri = Triangulation(uv[:, 0], uv[:, 1], fc)
    nl = len(out["loops"])
    W = np.ptp(uv[:, 0])
    H = np.ptp(uv[:, 1])
    fw = 9 * W / max(H, 1e-9)
    figm, ax = plt.subplots(2, 1, figsize=(min(max(fw, 8), 22), 18))
    for a, val, cm, lim, ttl in [(ax[0], la, "RdBu_r", (-3, 3), "log2 area factor (map / surface)"),
                                 (ax[1], qc, "magma", (0, 0.2), "conformal distortion s1/s2 - 1")]:
        im = a.tripcolor(tri, facecolors=np.nan_to_num(val), cmap=cm, vmin=lim[0], vmax=lim[1])
        plt.colorbar(im, ax=a, shrink=0.6)
        for i, (lp, L) in enumerate(out["loops"]):
            cp = np.isin(out["orig"], lp)
            a.plot(uv[cp, 0], uv[cp, 1], ".", c="#00e000", ms=2)
            a.annotate(f"handle {i + 1} ({L:.1f} mm)", uv[cp].mean(0), color="#00a000", fontsize=9)
        a.set_aspect("equal")
        a.set_title(f"{name}: {ttl}; green: handle cuts", fontsize=11)
        a.set_xlabel("map mm")
    figm.tight_layout()
    figm.savefig(f"{fig}/flat_map_{name}_distortion.png", dpi=70)
    plt.close(figm)
    figh, ax = plt.subplots(1, 2 + nl, figsize=(5 * (2 + nl), 4.5), squeeze=False)
    ax = ax[0]
    ok = np.isfinite(la)
    ax[0].hist(la[ok], bins=120, weights=out["area3"][ok], range=(-6, 6))
    ax[0].set_title("log2 area factor (area-weighted)")
    ax[1].hist(qc[ok], bins=120, weights=out["area3"][ok], range=(0, 0.5))
    ax[1].set_title("s1/s2 - 1 (area-weighted)")
    for i, (lp, L) in enumerate(out["loops"]):
        a = figh.add_subplot(1, 2 + nl, 3 + i, projection="3d")
        ax[2 + i].axis("off")
        c = v[lp].mean(0)
        r = max(1.5, 0.75 * np.ptp(v[lp], 0).max())
        near = np.linalg.norm(v - c, axis=1) < r
        fk = out["orig"][fc[near[out["orig"][fc]].all(1)]]
        u, fl = np.unique(fk, return_inverse=True)
        p = v[u]
        a.plot_trisurf(p[:, 0], p[:, 1], p[:, 2], triangles=fl.reshape(-1, 3), color="#c8c8c8", alpha=0.5, lw=0.1,
                       edgecolor="#999")
        a.set_box_aspect(np.ptp(p, 0))
        q = v[np.r_[lp, lp[:1]]]
        a.plot(q[:, 0], q[:, 1], q[:, 2], "-", c="#e00000", lw=2)
        a.set_title(f"handle {i + 1}: {L:.2f} mm round; centre ITK-SNAP {np.round(vb[lp].mean(0)).astype(int) + 1}",
                    fontsize=8)
    figh.tight_layout()
    figh.savefig(f"{fig}/flat_map_{name}_hist_handles.png", dpi=70)
    plt.close(figh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--to-disk" in sys.argv)
