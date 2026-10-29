"""An inflated white surface for the viewer's 3D surface pane (plan: reports/Crossing
paths and contacts from laminar profiles.md, *A pia editor on an inflated surface*).
Navigation only: the sections are cut from the true mesh; the inflated one only gives
where to click and which way is which.

1. Inflation: implicit steps of the uniform (umbrella) graph Laplacian,
   x <- (I + lam L)^-1 x, with the boundary loops (the box faces' cuts) pinned: the
   free-boundary version collapses an open sheet onto its boundary (tested), so the
   surface relaxes instead towards the membrane spanned by its cut loops, and area is
   not kept (reported per level). A closed piece (no boundary loop: the whole white
   surface, or an island of a box's mesh) has nothing to pin and would shrink to
   its centroid; restoring its area alone is not enough, since the slowest mode
   is left and the piece goes to a needle (tested). So after each step it is
   mapped back, about its vertex centroid (which the steps keep), to its own
   second-moment matrix by the symmetric T = C0^1/2 C^-1/2; as lam grows it goes
   to its embedding by its three lowest Laplacian modes, at its own size and
   orientation (a round surface for a ball-like piece). Levels are saved by their
   cumulative lam (a diffusion length of about sqrt(2 lam) edges): sulci open as
   the level rises.
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
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import cg

from baillarger.viewer import flatmap as FM
from baillarger.viewer.flatmap import Log

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.integer[Any]]
type Faces = npt.NDArray[np.integer[Any]]

LEVELS = (100.0, 400.0, 1600.0)
#: bumped when the inflation changes, so that cached inflations are remade
VERSION = 2


def umbrella(f: Faces, nv: int) -> csr_matrix:
    E, _ = FM.edges_of(f)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(nv, nv)).tocsr()
    A = (A + A.T).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    return (
        coo_matrix((deg, (np.arange(nv), np.arange(nv))), shape=(nv, nv)) - A
    ).tocsr()


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
    log: Log | None = None,
) -> list[Float]:
    """Inflated copies of v (mm) at cumulative lam = levels, the boundary loops
    pinned and each closed piece mapped back to its second moments after each
    step. Returns a list of (n, 3)."""
    nv = len(v)
    L = umbrella(f, nv)
    bl = FM.boundary_loops(f)
    onb = np.zeros(nv, bool)
    if bl:
        onb[np.concatenate(bl)] = True
    used = np.zeros(nv, bool)
    used[f.ravel()] = True
    fr = np.flatnonzero(used & ~onb)
    x = v.astype(float).copy()
    lab_v, lab_f = FM.pieces(f, nv)
    open_piece = np.zeros(int(lab_v.max()) + 1, bool)
    open_piece[lab_v[onb & used]] = True
    closed = np.flatnonzero(~open_piece)
    m0 = _moments(x, lab_v, closed, len(open_piece))
    out: list[Float] = []
    done = 0.0
    for lv in levels:
        lam = (lv - done) / steps_per_level
        M = (identity(nv, format="csr") + lam * L).tocsr()
        Mff, Mfb = M[fr][:, fr], M[fr][:, ~used[:] | onb]
        for _ in range(steps_per_level):
            xb = x[~used | onb]
            for k in range(3):
                y, info = cg(
                    Mff, x[fr, k] - Mfb @ xb[:, k], x0=x[fr, k], rtol=1e-8, maxiter=5000
                )
                assert info == 0, info
                x[fr, k] = y
            _keep_moments(x, lab_v, closed, m0)
        done = lv
        out.append(x.copy())
        if log:
            log(f"inflated to lam {lv:g}")
    return out


def _moments(
    x: Float, lab_v: Int, closed: Int, n: int
) -> tuple[npt.NDArray[np.bool_], Int, Float, Float]:
    """For the vertices of the closed pieces: their mask, piece label, and per
    piece the vertex centroid (n, 3) and second-moment matrix (n, 3, 3)."""
    is_closed = np.zeros(n, bool)
    is_closed[closed] = True
    m = (lab_v >= 0) & is_closed[np.maximum(lab_v, 0)]
    lab = lab_v[m]
    cnt = np.maximum(np.bincount(lab, minlength=n), 1)
    c = np.stack([np.bincount(lab, x[m, k], minlength=n) for k in range(3)], 1)
    c = c / cnt[:, None]
    d = x[m] - c[lab]
    C = np.zeros((n, 3, 3))
    for i in range(3):
        for j in range(3):
            C[:, i, j] = np.bincount(lab, d[:, i] * d[:, j], minlength=n) / cnt
    return m, lab, c, C


def _keep_moments(
    x: Float,
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
    _, _, c, C = _moments(x, lab_v, closed, len(C0))
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
    """The inflated surface of a site's white mesh at the cumulative lam in levels,
    as the viewer keeps it. v_mm: vertices (mm); f: faces. Returns levels (the
    lam of each level, (k,)), x (the inflated vertices, (k, n, 3), float32) and
    "summary": a dict with levels, one entry per level: lam,
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
                lam=lv,
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
