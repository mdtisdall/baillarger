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
from baillarger.mesh_smoothing import gauss_steps, umbrella

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


def dual_areas(x: F64, f: npt.NDArray[np.integer[Any]], vox: F64) -> tuple[F64, F64]:
    """Per vertex: the dual area (mm^2, a third of its triangles' areas) and
    the dual area vector (mm^2, a sixth of its triangles' cross products,
    pointing the way the faces are oriented)."""
    p = x * vox
    c = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    a = np.linalg.norm(c, axis=1) / 6.0
    A = np.zeros(len(x))
    N = np.zeros_like(x)
    for k in range(3):
        np.add.at(A, f[:, k], a)
        np.add.at(N, f[:, k], c / 6.0)
    return A, N


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
