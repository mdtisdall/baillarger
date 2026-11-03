"""An inflated white surface for the viewer's 3D surface pane (plan: reports/Crossing
paths and contacts from laminar profiles.md, *A pia editor on an inflated surface*).
Navigation only: the sections are cut from the true mesh; the inflated one only gives
where to click and which way is which.

1. Inflation: implicit steps of a smoothing with a local metric term, as FreeSurfer's
   mris_inflate pairs a spring term with a distance term (Fischl, Sereno & Dale 1999,
   NeuroImage 9:195-207). Each step minimises

       |x - x_prev|^2_M / (2 tau) + 1/2 sum_e w_e |x_i - x_j|^2
                                  + METRIC/2 sum_e w_e (|x_i - x_j| - d0_e)^2

   over the mesh edges e = (i, j), with d0_e the true edge length, M the lumped
   vertex areas of the true mesh and w_e = 2 A_e / d0_e^2 (A_e: a third of the true
   area of the faces at e). The weights follow the true lengths and areas, not the
   vertex count: on an equilateral mesh they are the cotangent weights, they are
   never negative, and they stay finite on the decimated overview's slivers. With
   the uniform (umbrella) weights the steps followed the sampling instead of the
   shape: a sphere sampled 20 times sparser on one half lost half of that half's
   area share (tested). The step is solved by local-global iterations (each edge's
   target t_e: d0_e along its current direction; then one linear solve with the
   matrix M + tau (1 + METRIC) L, L = B' W B, factored once per level).
   The metric term keeps thin parts: under the smoothing alone a finger or the
   walls of a narrow slot (an excision's crack) lose their area while the rest
   inflates, and the finger is left as a needle (tested).
   The boundary loops (the box faces' cuts) are pinned: the free-boundary version
   collapses an open sheet onto its boundary (tested). A closed piece (no boundary
   loop: the whole white surface, or an island of a box's mesh) has nothing to pin
   and would shrink to its centroid; restoring its area alone is not enough, since
   the slowest mode is left and the piece goes to a needle (tested). So after each
   step it is mapped back, about its area centroid (which the steps keep), to its
   own area-weighted second-moment matrix by the symmetric T = C0^1/2 C^-1/2.
   Levels are saved by their cumulative tau in mm^2 (a diffusion length of about
   sqrt(2 tau) mm): sulci open as the level rises.
2. Sulcal depth: the signed distance from the inflated vertex to its original
   position, along the inflated normal (positive: the original lies outward of the
   inflated surface, i.e. a crown), for shading.
3. Tangent transfer: a tangent vector at a vertex of the inflated mesh, written in
   each incident face's edge basis and carried over to the same face's edges on the
   true mesh (the per-face linear map between the two triangles), area-weighted over
   the faces.

site_inflation runs the inflation on a mesh and returns the levels and the per-level
area summary that the viewer keeps (baillarger.viewer.surface caches them).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu

from baillarger.viewer import flatmap as FM
from baillarger.viewer.flatmap import Log

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.integer[Any]]
type Faces = npt.NDArray[np.integer[Any]]

#: cumulative tau (mm^2): diffusion lengths sqrt(2 tau) of 10, 20 and 40 mm, the
#: lengths that the umbrella levels 100, 400, 1600 reached on the overview's median
#: edge (0.7 mm). Both terms depend on the edge lengths alone, so a far longer flow
#: drifts back towards the folded shape (a ball of radius 9 mm rounds up to a
#: diffusion length of about its radius, then slowly loses it): the levels stay
#: within the size of a hemisphere.
LEVELS = (50.0, 200.0, 800.0)
#: the weight of the metric term against the smoothing. With the same w_e in both
#: terms each is 2 x the area at the true surface, so METRIC is the ratio of their
#: stiffnesses. On a tube of radius r and edges h the smoothing's inward pull per
#: vertex is about A/r and the metric's push back about METRIC h s at a hoop strain
#: s, so s is about h / (METRIC r): 1 (equal stiffness) holds parts down to about
#: one edge thick. FreeSurfer's default ratio (0.1, -dist) is for 1 mm meshes with
#: no part thinner than an edge; here (WM blades of 0.5 mm, edges of 0.7 mm) it
#: kept 7 % of a finger's area (tested). Much more holds the folds as they are and
#: stops the inflation.
METRIC = 1.0
#: local-global iterations per step
ITERATIONS = 4
#: bumped when the inflation changes, so that cached inflations are remade
VERSION = 3


def edge_weights(v: Float, f: Faces) -> tuple[Int, Float, Float]:
    """The edges E (k, 2), their weights w_e = 2 A_e / d0_e^2 and their lengths d0
    on the mesh (v, f)."""
    E, _ = FM.edges_of(f)
    nv = len(v)
    fe = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    ke = E[:, 0] * nv + E[:, 1]
    o = np.argsort(ke)
    pos = o[np.searchsorted(ke[o], fe[:, 0] * nv + fe[:, 1])]
    a = FM.face_areas(v, f)
    Ae = np.bincount(pos, np.tile(a / 3, 3), minlength=len(E))
    d0 = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1)
    w = 2 * Ae / np.maximum(d0, 1e-12) ** 2
    return E, w, d0


def laplacian(v: Float, f: Faces) -> csr_matrix:
    """L = B' W B with the edge weights of edge_weights (B: the edge incidence)."""
    E, w, _ = edge_weights(v, f)
    B = incidence(E, len(v))
    out: csr_matrix = (B.T @ diags(w) @ B).tocsr()
    return out


def incidence(E: Int, nv: int) -> csr_matrix:
    k = np.arange(len(E))
    out: csr_matrix = coo_matrix(
        (
            np.r_[np.ones(len(E)), -np.ones(len(E))],
            (np.r_[k, k], np.r_[E[:, 0], E[:, 1]]),
        ),
        shape=(len(E), nv),
    ).tocsr()
    return out


def vertex_areas(v: Float, f: Faces) -> Float:
    """The lumped vertex areas: a third of the area of each face at the vertex."""
    m = np.zeros(len(v))
    np.add.at(m, f.ravel(), np.repeat(FM.face_areas(v, f) / 3, 3))
    return m


def vertex_normals(v: Float, f: Faces) -> Float:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for c in range(3):
        np.add.at(n, f[:, c], fn)
    out: Float = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return out


def inflate(
    v: Float,
    f: Faces,
    levels: tuple[float, ...] = LEVELS,
    steps_per_level: int = 4,
    metric: float = METRIC,
    log: Log | None = None,
) -> list[Float]:
    """Inflated copies of v (mm) at cumulative tau = levels (mm^2), the boundary
    loops pinned and each closed piece mapped back to its second moments after each
    step. metric = 0: the smoothing alone. Returns a list of (n, 3)."""
    nv = len(v)
    E, w, d0 = edge_weights(v, f)
    B = incidence(E, nv)
    L = (B.T @ diags(w) @ B).tocsr()
    m = vertex_areas(v, f)
    # a vertex whose faces all have zero area would leave its row empty
    m = np.maximum(m, 1e-12 * max(float(m.mean()), 1e-300))
    M = diags(m)
    bl = FM.boundary_loops(f)
    onb = np.zeros(nv, bool)
    if bl:
        onb[np.concatenate(bl)] = True
    used = np.zeros(nv, bool)
    used[f.ravel()] = True
    fr = np.flatnonzero(used & ~onb)
    fixed = np.flatnonzero(~used | onb)
    x = v.astype(float).copy()
    lab_v, lab_f = FM.pieces(f, nv)
    open_piece = np.zeros(int(lab_v.max()) + 1, bool)
    open_piece[lab_v[onb & used]] = True
    closed = np.flatnonzero(~open_piece)
    m0 = _moments(x, m, lab_v, closed, len(open_piece))
    out: list[Float] = []
    done = 0.0
    for lv in levels:
        tau = (lv - done) / steps_per_level
        A = (M + tau * (1 + metric) * L).tocsr()
        lu = splu(A[fr][:, fr].tocsc())
        Afb = A[fr][:, fixed]
        for _ in range(steps_per_level):
            rhs0 = M @ x
            for _ in range(ITERATIONS if metric else 1):
                rhs = rhs0.copy()
                if metric:
                    e = B @ x
                    n = np.linalg.norm(e, axis=1, keepdims=True)
                    t = d0[:, None] * e / np.maximum(n, 1e-30)
                    rhs += tau * metric * (B.T @ (w[:, None] * t))
                rhs_f = rhs[fr] - Afb @ x[fixed]
                for k in range(3):
                    x[fr, k] = lu.solve(np.ascontiguousarray(rhs_f[:, k]))
            _keep_moments(x, m, lab_v, closed, m0)
        done = lv
        out.append(x.copy())
        if log:
            log(f"inflated to tau {lv:g} mm^2")
    return out


def _moments(
    x: Float, wt: Float, lab_v: Int, closed: Int, n: int
) -> tuple[npt.NDArray[np.bool_], Int, Float, Float]:
    """For the vertices of the closed pieces: their mask, piece label, and per
    piece the centroid (n, 3) and second-moment matrix (n, 3, 3), each vertex
    weighted by wt (its true area)."""
    is_closed = np.zeros(n, bool)
    is_closed[closed] = True
    m = (lab_v >= 0) & is_closed[np.maximum(lab_v, 0)]
    lab = lab_v[m]
    wm = wt[m]
    tot = np.bincount(lab, wm, minlength=n)
    tot = np.where(tot > 0, tot, 1.0)
    c = np.stack([np.bincount(lab, wm * x[m, k], minlength=n) for k in range(3)], 1)
    c = c / tot[:, None]
    d = x[m] - c[lab]
    C = np.zeros((n, 3, 3))
    for i in range(3):
        for j in range(3):
            C[:, i, j] = np.bincount(lab, wm * d[:, i] * d[:, j], minlength=n) / tot
    return m, lab, c, C


def _keep_moments(
    x: Float,
    wt: Float,
    lab_v: Int,
    closed: Int,
    m0: tuple[npt.NDArray[np.bool_], Int, Float, Float],
) -> None:
    """Map each closed piece of x in place, about its centroid, to its first
    second-moment matrix C0 by T = C0^1/2 C^-1/2; other vertices are not
    touched."""
    if not closed.size:
        return
    m, lab, _, C0 = m0
    _, _, c, C = _moments(x, wt, lab_v, closed, len(C0))
    T = np.tile(np.eye(3), (len(C0), 1, 1))
    for p in closed:
        w0, U0 = np.linalg.eigh(C0[p])
        w, U = np.linalg.eigh(C[p])
        if w.min() > 1e-12 * w.max() and w0.min() > 0:
            T[p] = (U0 * np.sqrt(w0)) @ U0.T @ (U / np.sqrt(w)) @ U.T
    x[m] = c[lab] + np.einsum("nij,nj->ni", T[lab], x[m] - c[lab])


def sulcal_depth(v: Float, x: Float, f: Faces) -> Float:
    out: Float = np.einsum("ij,ij->i", v - x, vertex_normals(x, f))
    return out


def transfer(
    r: Float,
    vi: int,
    x: Float,
    v: Float,
    f: Faces,
    vert_faces: tuple[Int, Int],
) -> Float:
    """Tangent r (3,) at vertex vi of the inflated mesh x carried to the true mesh v:
    for each face at vi, r projected on the inflated face's plane, written in its
    edge basis (e1, e2) and rebuilt from the true face's edges; area-weighted mean.
    vert_faces: (indptr, faces) as csr."""
    ip, fs = vert_faces
    acc, wsum = np.zeros(3), 0.0
    for k in fs[ip[vi] : ip[vi + 1]]:
        a, b, c = f[k]
        E = np.c_[x[b] - x[a], x[c] - x[a]]
        O = np.c_[v[b] - v[a], v[c] - v[a]]  # noqa: E741
        coef, *_ = np.linalg.lstsq(E, r, rcond=None)
        w = 0.5 * np.linalg.norm(np.cross(O[:, 0], O[:, 1]))
        acc += w * (O @ coef)
        wsum += w
    out: Float = acc / max(wsum, 1e-30)
    return out


def vertex_faces(f: Faces, nv: int) -> tuple[Int, Int]:
    k = np.repeat(np.arange(len(f)), 3)
    M = coo_matrix((np.ones(k.size), (f.ravel(), k)), shape=(nv, len(f))).tocsr()
    return M.indptr, M.indices


def site_inflation(
    v_mm: Float,
    f: Faces,
    levels: tuple[float, ...] = LEVELS,
    log: Log | None = None,
) -> dict[str, Any]:
    """The inflated surface of a site's white mesh at the cumulative tau (mm^2) in
    levels, as the viewer keeps it. v_mm: vertices (mm); f: faces. Returns levels
    (the tau of each level, (k,)), x (the inflated vertices, (k, n, 3), float32) and
    "summary": a dict with levels, one entry per level: tau_mm2,
    log2_area_p1_p5_p50_p95_p99 (area-weighted percentiles, weights the true face
    areas, of the log2 ratio of inflated to true face area) and
    sulcal_depth_mm_p1_p50_p99."""
    X = inflate(v_mm, f, levels=levels, log=log)
    a0 = FM.face_areas(v_mm, f)
    rows: list[dict[str, Any]] = []
    for lv, x in zip(levels, X, strict=True):
        la = np.log2(np.maximum(FM.face_areas(x, f), 1e-30) / np.maximum(a0, 1e-30))
        sd = sulcal_depth(v_mm, x, f)
        q = FM.wpct(la, a0, [1, 5, 50, 95, 99])
        rows.append(
            dict(
                tau_mm2=lv,
                log2_area_p1_p5_p50_p95_p99=[round(t, 2) for t in q],
                sulcal_depth_mm_p1_p50_p99=[
                    round(float(t), 2) for t in np.percentile(sd, [1, 50, 99])
                ],
            )
        )
    return dict(
        levels=np.array(levels),
        x=np.stack(X).astype(np.float32),
        summary=dict(levels=rows),
    )
