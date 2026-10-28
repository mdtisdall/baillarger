"""Column traces past T's crest, by one crossing rule, traced independently or
as a growing surface: column_traces.METHODS "traverse" (sigma 0: each vertex
on its own) and "front" (the white mesh advanced outward along the fitted
field with its neighbours coupled). Each vertex's trajectory is its column.

Plan: reports/Crossing paths and contacts from laminar profiles.md (*Columns
as a growing surface*). Positions are voxel index coordinates; vox the voxel
size (mm); lengths in mm; areas mm^2; volumes mm^3. Everything is 3D.

1. Start: every vertex of the white mesh (not only the roots), at its place
   on the WM label's boundary, heading along its mesh normal.
2. Each step (step_mm of arc length per vertex), every moving vertex takes
   the field's direction at its place: the unit gradient of the extended
   fitted field T (streamlines.fitted_field with fluid and wm_band_mm), by
   a midpoint (RK2) step; with blend_mm, blended in from the mesh normal
   over the first blend_mm (the fitted gradient's coherence is 1, so this is
   streamlines.trace's blended start).
3. Past T's crest (ascend, then descend): a vertex climbs +grad T until the
   climb reverses, that is, until the unit gradient at its place points
   against its previous step (T stops rising along its path: the directional
   derivative changes sign). From then on it descends -grad T, into the far
   bank. Across a kink of T between two banks, the two banks' gradients
   point at each other, so -grad T beyond is the continuation of +grad T
   before: the direction field the vertices follow is continuous through the
   crest, where the unsigned rule (streamlines.trace on T's gradient tensor)
   picks, at an oblique crest, the sign that reflects the path back.
   At the kink itself T has no gradient: the sampled one (central
   differences, trilinear) averages the two banks' and, where the crest
   rises (a fused V, whose crest climbs toward the opening), points along
   the crest, so a climb that follows it slides along the crest and never
   reverses. So within the stencil of the crest voxels (kink, given:
   streamlines.crest_voxels, dilated by one voxel along each axis for the
   central differences, and reached by any of a point's eight trilinear
   corners) a vertex keeps its heading, and the reversal test is made where
   it leaves them.
4. Coupling: the moving vertices' directions are smoothed along the mesh by
   a Gaussian of sigma voxels (by default the white mesh's own twicing scale,
   column_traces.MESH_SIGMA: the scale below which the mesh carries no
   shape), with its shrinkage removed by twicing (mesh_smoothing.twice), as
   a normalized convolution over the moving vertices only, then normalized.
   So a vertex follows the field averaged over its mesh neighbourhood and
   cannot wander off alone, and the step does not move the surface across
   the layers more than the field itself varies over that scale. Stopped
   vertices stay where they stopped and do not pull on the others.
5. Stops (streamlines' codes): fluid, the WM after wm_after mm (own bank or
   far bank), missing tissue, other labels, a box face (within face_mm), the
   length cap. Fronts from different banks pass through each other: where
   they meet is not a stop (the reading of the columns places the contact).
6. Bok's measure: per vertex, at every sample, its dual area A (a third of
   its triangles' areas, mm^2) and the volume its column has swept since the
   root, V = sum of N_v . dx over the steps (N_v the vertex's dual area
   vector, a sixth of its triangles' cross products, on the mesh halfway
   through the step; dx the step in mm): the exact flux of the dual cell,
   so V counts the volume between the front's positions whatever the
   front's tilt to the step. The equivolume depth between two cuts s0 < s1
   along a column (the WM's edge and the pia, which step 2 places) is
   (V(s) - V(s0)) / (V(s1) - V(s0)).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy import sparse

from baillarger import streamlines as S
from baillarger.mesh_smoothing import edges, gauss_steps, umbrella

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


@dataclass
class Front:
    """The roots' columns from a growing front."""

    #: (n, nmax, 3) float32 positions, nan past the end; sample i at
    #: i * step_mm along the column
    pos: F32
    #: sample counts and outcomes (streamlines codes)
    ns: I64
    out: I64
    #: (n, nmax) float32: the vertex's dual area (mm^2) and the volume swept
    #: since the root (mm^3) at each sample; nan past the end
    area: F32
    vol: F32
    #: the sample where the column passed T's crest and began to descend, -1
    #: if it never did
    turn_at: I64
    #: grow_equivolume: (n, nmax) float32, the front's time at each sample,
    #: its equivolume coordinate (mm: the volume each column sweeps per unit
    #: root area, while the speed is under its cap); None for grow
    tau: F32 | None = None


def dual_areas(x: F64, f: npt.NDArray[np.integer[Any]], vox: F64) -> tuple[F64, F64]:
    """Per vertex: the dual area (mm^2, a third of its triangles' areas) and
    the dual area vector (mm^2, a sixth of its triangles' cross products,
    pointing the way the faces are oriented)."""
    p = x * vox
    c = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    a = np.linalg.norm(c, axis=1) / 6.0
    return face_sums(f, a, len(x)), face_sums(f, c / 6.0, len(x))


def face_sums(f: npt.NDArray[np.integer[Any]], val: F64, nv: int) -> F64:
    """Per vertex, the sum of val (per face, (nf,) or (nf, k)) over its
    faces."""
    idx = f.ravel()
    if val.ndim == 1:
        out: F64 = np.bincount(idx, np.repeat(val, 3), minlength=nv)
        return out
    rep = np.repeat(val, 3, axis=0)
    return np.stack(
        [np.bincount(idx, rep[:, j], minlength=nv) for j in range(val.shape[1])], 1
    )


def past_crest(g: F64, d: F64, ascending: Bool, coast: Bool) -> tuple[Bool, Bool]:
    """The rule past T's crest (module docstring, 3), per moving vertex: g the
    unit gradient of T at its place, d its previous step (unit, mm),
    ascending whether it still climbs T, coast whether it lies within the
    kink's stencil. Returns which vertices begin to descend now, and which
    keep their heading for this step."""
    switch: Bool = ascending & ~coast & ((g * d).sum(1) < 0)
    return switch, coast


class _Coupling:
    """The masked, twiced Gaussian along the mesh (module docstring, 4)."""

    def __init__(self, v: F64, f: npt.NDArray[np.integer[Any]], sigma: float):
        A, deg = umbrella(v, f)
        self.W: sparse.csr_matrix = sparse.diags(1.0 / deg) @ A
        self.lam = 0.5
        self.n = gauss_steps(v, f, sigma, self.lam)

    def G(self, x: F64) -> F64:
        for _ in range(self.n):
            x = x + self.lam * (self.W @ x - x)
        return x

    def __call__(self, d: F64, m: Bool) -> F64:
        """Smoothed directions (n, 3) from d over the vertices m."""
        mf = m.astype(np.float64)[:, None]
        g1 = self.G(np.c_[d * mf, mf])
        s1 = g1[:, :3] / np.maximum(g1[:, 3:], 1e-12)
        g2 = self.G(np.c_[s1 * mf, mf])
        s2 = g2[:, :3] / np.maximum(g2[:, 3:], 1e-12)
        out: F64 = 2 * s1 - s2
        return out


def grow(
    T: F64,
    labels: npt.NDArray[np.integer[Any]],
    dface: F64,
    v: F64,
    f: npt.NDArray[np.integer[Any]],
    nrm: F64,
    roots: I64,
    vox: npt.ArrayLike,
    nmax: int,
    sigma: float,
    blend_mm: float | None = None,
    kink: Bool | None = None,
    step_mm: float = S.STEP_MM,
    face_mm: float = 2.0,
    wm_after: float = 0.3,
) -> Front:
    """Grow the white mesh (v, f; unit normals nrm out of the WM) along T and
    return the columns of the vertices roots (module docstring). sigma: the
    coupling's scale (voxels; 0 for none). kink: T's crest voxels (None: no
    kinks)."""
    vx = np.asarray(vox, np.float64)
    g = [np.asarray(c, np.float64) for c in S.gradient(T, vx)]  # per mm
    h = step_mm / vx
    nv = len(v)
    couple = _Coupling(v, f, sigma) if sigma > 0 else None

    def unit_grad(xa: F64) -> F64:
        gv = np.stack(
            [ndi.map_coordinates(c, xa.T, order=1, mode="nearest") for c in g], 1
        )
        out: F64 = gv / np.maximum(np.linalg.norm(gv, axis=1, keepdims=True), 1e-12)
        return out

    zone = None
    if kink is not None:
        zone = ndi.binary_dilation(kink).astype(np.float64)

    def in_kink(xa: F64) -> Bool:
        if zone is None:
            return np.zeros(len(xa), bool)
        z = ndi.map_coordinates(zone, xa.T, order=1, mode="nearest")
        out_: Bool = z > 0
        return out_

    nm = nrm * vx  # normals in mm (they transform by the inverse metric)
    nm /= np.linalg.norm(nm, axis=1, keepdims=True)
    x = np.asarray(v, np.float64).copy()
    d = nm.copy()  # each vertex's last step, unit, mm
    sgn = np.ones(nv)
    turned = np.full(nv, -1, np.int64)
    act = np.ones(nv, bool)
    out = np.full(nv, S.RUNNING, np.int64)
    nsamp = np.ones(nv, np.int64)
    vol = np.zeros(nv)
    rix = -np.ones(nv, np.int64)
    rix[roots] = np.arange(len(roots))
    nr = len(roots)
    pos = np.full((nr, nmax, 3), np.nan, np.float32)
    area = np.full((nr, nmax), np.nan, np.float32)
    vols = np.full((nr, nmax), np.nan, np.float32)
    pos[:, 0] = x[roots]
    A0, _ = dual_areas(x, f, vx)
    area[:, 0] = A0[roots]
    vols[:, 0] = 0.0

    def direction(xa: F64, sa: F64, s: float, nma: F64) -> F64:
        u = unit_grad(xa) * sa[:, None]
        if blend_mm is None:
            return u
        b = min(1.0, s / blend_mm)
        dd = b * u + (1 - b) * nma
        out_: F64 = dd / np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-12)
        return out_

    for i in range(1, nmax):
        a = np.flatnonzero(act)
        if not a.size:
            break
        xa = x[a]
        s = (i - 1) * step_mm
        keep = np.zeros(len(a), bool)
        if i > 1:  # ascend, then descend (module docstring, 3)
            back, keep = past_crest(unit_grad(xa), d[a], sgn[a] > 0, in_kink(xa))
            sgn[a[back]] = -1.0
            turned[a[back]] = i - 1
        u1 = direction(xa, sgn[a], s, nm[a])
        u2 = direction(xa + 0.5 * u1 * h, sgn[a], s + 0.5 * step_mm, nm[a])
        u2[keep] = d[a[keep]]
        if couple is not None:
            full = np.zeros((nv, 3))
            full[a] = u2
            sm = couple(full, act)[a]
            ln = np.linalg.norm(sm, axis=1, keepdims=True)
            u2 = np.where(ln > 1e-6, sm / np.maximum(ln, 1e-12), u2)
        x_new = x.copy()
        x_new[a] = xa + u2 * h
        _, N = dual_areas(0.5 * (x + x_new), f, vx)
        vol[a] += (N[a] * (u2 * step_mm)).sum(1)
        x = x_new
        d[a] = u2
        nsamp[a] = i + 1
        code = S._stop_codes(x[a], labels, dface, i * step_mm, face_mm, wm_after)
        if i == nmax - 1:
            code[code == S.RUNNING] = S.LENGTH
        out[a] = code
        act[a[code != S.RUNNING]] = False
        r = rix[a]
        k = r >= 0
        if k.any():
            Ai, _ = dual_areas(x, f, vx)
            pos[r[k], i] = x[a[k]]
            area[r[k], i] = Ai[a[k]]
            vols[r[k], i] = vol[a[k]]
    return Front(pos, nsamp[roots], out[roots], area, vols, turned[roots])


#: The equivolume front's speed cap, a multiple of the root's speed: one
#: voxel per step_mm of the front's time (grow_equivolume, 3).
SPEED_CAP = 4.0
#: The front's time cap, as a multiple of the arc length cap (grow_equivolume,
#: 4): at 2, a column on a cylinder-like crown of radius 0.32 mm (2 voxels,
#: the tightest the white mesh keeps) reaches 2.9 mm of arc, 3.5 mm at 0.5 mm.
TAU_CAP = 2.0


def front_normals(
    x: F64,
    f: npt.NDArray[np.integer[Any]],
    m: Bool,
    x0: F64,
    heading: F64,
    broken: Bool,
    iters: int = 10,
    cap: float = SPEED_CAP,
    E: I64 | None = None,
) -> F64:
    """Unit vertex normals (mm, x in mm), area-weighted, with the vertices
    broken taking their heading instead, then smoothed along the edges
    between the vertices m as mesh_smoothing.smooth_normals smooths the root
    mesh x0 (iters umbrella steps of 1/2), at the same reach in mm: each edge
    weighs (e0 / e)^2, its root length over its length now, up to cap (an
    area ratio, as grow_equivolume's speed cap), so where the front has
    shrunk (toward a focus) its neighbours pull harder, and the steps are
    split so that none moves a normal more than half way to its neighbours'
    mean (at most cap * iters steps). On the root mesh it is
    smooth_normals. E: the mesh's edges (mesh_smoothing.edges), if known."""
    fn = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    n = face_sums(f, fn, len(x))
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n[broken] = heading[broken]
    if E is None:
        E = edges(f)
    E = E[m[E[:, 0]] & m[E[:, 1]]]
    if not len(E):
        return n
    w = np.sum((x0[E[:, 0]] - x0[E[:, 1]]) ** 2, 1) / np.maximum(
        np.sum((x[E[:, 0]] - x[E[:, 1]]) ** 2, 1), 1e-24
    )
    w = np.minimum(w, cap)
    nv = len(x)
    W = sparse.coo_matrix((w, (E[:, 0], E[:, 1])), shape=(nv, nv)).tocsr()
    W = W + W.T
    deg = np.bincount(E.ravel(), minlength=nv).astype(np.float64)
    ws = np.asarray(W.sum(1)).ravel()
    load = float(np.max(ws / np.maximum(deg, 1.0)))
    k = max(iters, int(np.ceil(iters * load)))
    mu = 0.5 * iters / k
    for _ in range(k):
        n = n + mu * ((W @ n) - ws[:, None] * n) / np.maximum(deg, 1.0)[:, None]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def grow_equivolume(
    labels: npt.NDArray[np.integer[Any]],
    dface: F64,
    v: F64,
    f: npt.NDArray[np.integer[Any]],
    roots: I64,
    vox: npt.ArrayLike,
    nmax: int,
    step_mm: float = S.STEP_MM,
    face_mm: float = 2.0,
    wm_after: float = 0.3,
    cap: float = SPEED_CAP,
    tau_cap: float = TAU_CAP,
) -> Front:
    """Grow the white mesh (v, f; faces oriented out of the WM) with no data
    field, at equivolume speed, and return the columns of the vertices roots
    (column_traces.METHODS "front-equivolume"). Positions in voxels.

    1. Direction: each step (a midpoint step, as grow's), the front's own
       normals: area-weighted vertex normals, smoothed along the mesh over
       the moving vertices as mesh_smoothing.smooth_normals smooths the root
       mesh, at the same reach in mm however the front has shrunk or grown
       (front_normals). A normal that points against the vertex's previous
       step (the front has folded over itself past a focus) is reversed, so
       a column never turns back.
    2. Speed: A0 / A, the vertex's dual area at the root over its current
       one. Then every column sweeps the same volume per unit root area,
       A dx = A0 dtau: the fronts are the levels of tau, the equivolume
       layers, and the columns bend to stay normal to them. On a convex
       front (a crown) A grows and the column slows; on a concave one (a
       fundus) A shrinks toward a focus, where A -> 0 and the speed ->
       infinity.
    3. The cap at a focus: the speed is at most cap (default SPEED_CAP = 4)
       times the root's, so a column moves at most one voxel per step_mm of
       tau: within a voxel of a focus the mesh has no normal to follow. Each
       step advances tau by step_mm / max(1, the largest speed among the
       moving vertices), so no vertex moves more than step_mm in one step,
       and the front stays one level of tau. Where the cap binds the column
       lags its level: its swept volume per root area (vol / area at the
       root) falls below tau.
    Where the front has an edge (a vertex with a stopped neighbour: its ring
    of faces is broken) or a fold (a face whose normal points against its
    vertices' headings: near a focus, vertices have crossed), it has no
    normal and no dual area: the vertex goes on straight along its last step
    at its last speed, and its neighbours' smoothing sees that heading.
    4. Samples: as every method, sample i at i * step_mm of arc length along
       the column (linear between the steps, which are at most step_mm), the
       last sample at the stop; tau records each sample's coordinate.
       Columns stop as grow's (fluid, the WM after wm_after mm of arc,
       missing, other, a face within face_mm, the arc length cap
       (nmax - 1) * step_mm), and a cap on tau: the front stops growing at
       tau = tau_cap * (nmax - 1) * step_mm (TAU_CAP), and a column still
       running then (a crown column, slowed below the root's speed) ends
       there as LENGTH, short of its arc cap. Fronts from different banks
       pass through each other.
    5. No tangential relaxation: a column is a vertex's trajectory, and its
       dual area is the column's cross-section; moving vertices along the
       front would trade volume between columns and bend the trajectories
       off the normal. The normals' smoothing keeps the front well shaped
       below the mesh's own scale.
    6. Equivolume: equal swept volume per unit root area is Bok's layer only
       if every column holds the same volume per unit root area from the
       WM to the pia (the same assumption as the "normal" method's depth);
       a relative depth between two cuts s0 < s1 is (tau(s) - tau(s0)) /
       (tau(s1) - tau(s0)). area and vol are recorded as grow's, so the
       flux vol / area[0] checks tau."""
    vx = np.asarray(vox, np.float64)
    nv = len(v)
    f = np.asarray(f)
    x = np.asarray(v, np.float64).copy()
    act = np.ones(nv, bool)
    out = np.full(nv, S.RUNNING, np.int64)
    arc = np.zeros(nv)
    vol = np.zeros(nv)
    rix = -np.ones(nv, np.int64)
    rix[roots] = np.arange(len(roots))
    nr = len(roots)
    pos = np.full((nr, nmax, 3), np.nan, np.float32)
    area = np.full((nr, nmax), np.nan, np.float32)
    vols = np.full((nr, nmax), np.nan, np.float32)
    taus = np.full((nr, nmax), np.nan, np.float32)
    nsamp = np.ones(nr, np.int64)
    Aall, _ = dual_areas(x, f, vx)
    pos[:, 0] = x[roots]
    area[:, 0] = Aall[roots]
    vols[:, 0] = 0.0
    taus[:, 0] = 0.0
    x0mm = x * vx
    A0 = Aall.copy()
    E = edges(f)
    d = front_normals(x0mm, f, act, x0mm, x0mm, np.zeros(nv, bool), E=E)
    last = np.ones(nv)  # each vertex's last speed
    tau, tau_max = 0.0, tau_cap * (nmax - 1) * step_mm
    arc_max = (nmax - 1) * step_mm

    def velocity(xv: F64, broken: Bool) -> tuple[F64, F64]:
        """Unit directions (mm) and speeds at the front xv (1-3 above)."""
        n = front_normals(xv * vx, f, act, x0mm, d, broken, cap=cap, E=E)
        n = np.where(((n * d).sum(1) < 0)[:, None], -n, n)
        n[broken] = d[broken]
        A, _ = dual_areas(xv, f, vx)
        speed = np.where(broken, last, A0 / np.maximum(A, A0 / cap))
        return n, speed

    while act.any():
        xm = x * vx
        fn = np.cross(xm[f[:, 1]] - xm[f[:, 0]], xm[f[:, 2]] - xm[f[:, 0]])
        folded = (fn * d[f].sum(1)).sum(1) <= 0
        broken = np.zeros(nv, bool)
        broken[f[~act[f].all(1) | folded].ravel()] = True
        broken &= act
        n1, s1 = velocity(x, broken)
        dtau = step_mm / max(1.0, float(s1[act].max()))
        # a midpoint (RK2) step, as grow's
        xh = x + np.where(act[:, None], n1 * (s1 * dtau / 2)[:, None], 0.0) / vx
        n, speed = velocity(xh, broken)
        dtau = min(dtau, step_mm / max(1e-12, float(speed[act].max())))
        dx = np.where(act[:, None], n * (speed * dtau)[:, None], 0.0)  # mm
        x_new = x + dx / vx
        _, N = dual_areas(0.5 * (x + x_new), f, vx)
        vol += (N * dx).sum(1)
        arc_new = arc + np.where(act, speed * dtau, 0.0)
        tau += dtau
        A_new, _ = dual_areas(x_new, f, vx)
        a_ = np.flatnonzero(act)
        code = _stop_codes_arc(x_new[a_], labels, dface, arc_new[a_], face_mm, wm_after)
        run = code == S.RUNNING
        code[run & ((arc_new[a_] >= arc_max - 1e-9) | (tau >= tau_max - 1e-9))] = (
            S.LENGTH
        )
        # record the roots: the arc sample crossed in this step, then the stop
        r = rix[a_]
        k = r >= 0
        ri, vi, ci = r[k], a_[k], code[k]
        j = np.floor(arc_new[vi] / step_mm + 1e-9).astype(np.int64)
        cross = (j > np.floor(arc[vi] / step_mm + 1e-9)) & (j < nmax)
        if cross.any():
            rc, vc, jc = ri[cross], vi[cross], j[cross]
            w = (jc * step_mm - arc[vc]) / np.maximum(arc_new[vc] - arc[vc], 1e-12)
            w = np.clip(w, 0.0, 1.0)
            pos[rc, jc] = (1 - w)[:, None] * x[vc] + w[:, None] * x_new[vc]
            area[rc, jc] = (1 - w) * Aall[vc] + w * A_new[vc]
            vols[rc, jc] = (1 - w) * (vol[vc] - (N[vc] * dx[vc]).sum(1)) + w * vol[vc]
            taus[rc, jc] = tau - (1 - w) * dtau
            nsamp[rc] = jc + 1
        stop = ci != S.RUNNING
        if stop.any():
            rs, vs = ri[stop], vi[stop]
            gap = arc_new[vs] - (nsamp[rs] - 1) * step_mm > 1e-6
            put = gap & (nsamp[rs] < nmax)
            rs, vs = rs[put], vs[put]
            pos[rs, nsamp[rs]] = x_new[vs]
            area[rs, nsamp[rs]] = A_new[vs]
            vols[rs, nsamp[rs]] = vol[vs]
            taus[rs, nsamp[rs]] = tau
            nsamp[rs] += 1
        out[a_] = code
        act[a_[code != S.RUNNING]] = False
        x, arc, Aall = x_new, arc_new, A_new
        d = np.where(act[:, None], n, d)
        last = np.where(act, speed, last)
    return Front(pos, nsamp, out[roots], area, vols, np.full(nr, -1, np.int64), taus)


def _stop_codes_arc(
    xa: F64,
    labels: npt.NDArray[np.integer[Any]],
    dface: F64,
    s_mm: F64,
    face_mm: float,
    wm_after: float,
) -> I64:
    """streamlines._stop_codes with each path's own arc length s_mm."""
    early = s_mm <= wm_after
    code = S._stop_codes(xa, labels, dface, np.inf, face_mm, wm_after)
    if early.any():
        code[early] = S._stop_codes(xa[early], labels, dface, 0.0, face_mm, wm_after)
    return code
