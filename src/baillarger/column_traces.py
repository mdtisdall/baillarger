"""The column traces of a box: one streamline per root on the white mesh,
through the cortex to the fluid or across a contact to the far WM, with a
fallback where the streamline fails, and each trace's class.

Plan: reports/Crossing paths and contacts from laminar profiles.md (*The
column volume*, *Column traces at folds*). The column volume adds the echo
features and the depth horizons along these traces
(prototypes/scripts/column_volume.py). Everything is 3D. Positions are voxel
index coordinates of the box; vox the voxel size (mm); lengths in mm.

1. Tissue (load_tissue, tissue): the rim labels of the box, with the WM
   label of choice (the aseg's, or the handle-free label from
   baillarger.white), the missing tissue
   (baillarger.frame.missing_tissue), and echoes 2 and 3 with their trend
   removed (log intensity minus its normalized convolution over the cortex
   at 4 voxels), zero outside the cortex. Labels: 0 cortex, 1 fluid, 2 WM,
   3 missing, 4 other.
2. Roots (white_mesh, root_vertices): the white mesh of the WM label
   (baillarger.white.white_mesh: marching tetrahedra on the binary label,
   tangential relaxation, twicing at 2 voxels) and its smoothed normals
   (baillarger.mesh_smoothing). The binary label keeps the surface on the
   label's own boundary. One root per voxel cell and normal side (dominant
   axis and sign): the vertex nearest the cell's centre, so that roots are
   about a voxel apart. The handle-free label gives genus-0 pieces; the
   aseg's (the default, as in the volumes built before) does not.
3. Primary trace (baillarger.streamlines.trace), by TRACERS:
   - "current": the base structure tensor of the two echoes, from the root's
     first step.
   - "fitted": the fitted field's rank-1 tensor, from the first step.
   - "fitted_blend" (the default): the fitted field, started along the mesh
     normal and blended into the field over BLEND_MM / coherence. On the
     fold phantoms it is the most accurate at tight fundi and at contacts;
     on the two marked sites it gives the most crossing traces and the
     fewest that return to their own bank.
4. Fallback where the primary turned or reached the length cap: ascent of
   the equal-arrival one pass (streamlines.ascend). The one pass
   (one_pass) is an anisotropic fast march (baillarger.fronts) from every
   WM voxel 6-adjacent to the cortex, under a metric with unit cost along
   the data's oriented normal and RATIO across it. Its sources are labelled
   by root patches: geodesic Voronoi cells, PATCH voxels apart, on the graph
   of seed voxels whose outward directions agree (so the two sides of a thin
   WM blade are not joined). The fallback stops at the pass's contact: a
   voxel whose patch is more than SAME_MM along the white surface from the
   path's own root patch.
5. Classes: OPEN (ends in fluid), CROSS (ends in WM at least SAME_MM along
   the white surface from its root), OWN (ends in WM nearer: back on its own
   bank), TOCONTACT (the fallback stopped at the pass's contact), FAILED.
   Roots within FACE_MM of a box face are flagged (near_face).

Prototype entry point: prototypes/scripts/column_volume.py (--tracer).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from baillarger import streamlines as S
from baillarger.data import Patch, load_box
from baillarger.frame import missing_tissue
from baillarger.fronts import Fronts, march, riemann_from_direction
from baillarger.mesh_smoothing import smooth_normals
from baillarger.white import WMChanges, white_matter_box
from baillarger.white import white_mesh as white_surface

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I8 = npt.NDArray[np.int8]
type I64 = npt.NDArray[np.int64]

TRACERS = ("current", "fitted", "fitted_blend")
#: Trace classes.
OPEN, CROSS, OWN, TOCONTACT, FAILED = range(5)
CLASS = ("open", "crossing", "own bank", "to the pass's contact", "failed")
#: Trend scale of the echoes (voxels).
TREND = 4.0
#: Twicing scale of the white mesh (voxels).
MESH_SIGMA = 2.0
#: The blended start's length at coherence 1 (mm).
BLEND_MM = 0.3
#: Root patch spacing (voxels), cost ratio across the normal, the distance
#: along the white surface that separates two banks (mm), and the box-face
#: margin (mm).
PATCH, RATIO, SAME_MM, FACE_MM = 6, 3.0, 5.0, 2.0
#: Samples per trace: 8 mm at STEP_MM.
NMAX = 201


# ---------------------------------------------------------------- tissue


@dataclass
class Tissue:
    """A box's labels and features."""

    patch: Patch
    vox: F64
    rim: npt.NDArray[np.int16]
    miss: Bool
    ctx: Bool
    wm: Bool
    #: echoes 2 and 3 (keys 1, 2), log, trend removed, zero outside the cortex
    det: dict[int, F64]
    #: distance of each voxel to the nearest box face (mm)
    dface: F64
    #: 0 cortex, 1 fluid, 2 WM, 3 missing, 4 other
    labels: npt.NDArray[np.int16]


def face_distance(shape: tuple[int, ...], vox: F64) -> F64:
    """Distance (mm) of every voxel centre to the nearest box face."""
    idx = np.indices(shape)
    d = [np.minimum(idx[a], shape[a] - 1 - idx[a]) * vox[a] for a in range(3)]
    out: F64 = np.minimum(np.minimum(d[0], d[1]), d[2])
    return out


def tissue(p: Patch, wm: Bool | None = None) -> Tissue:
    """Step 1 on a loaded box. wm: the WM label to use (default the aseg's,
    rim == 2). It replaces the aseg's in the rim labels before anything is
    derived from them: voxels it adds become WM, and aseg WM voxels it leaves
    out (a cut across a bridge or a pinhole) become cortex. So the white
    mesh, the fitted field's boundary, the trace stops, the one pass and the
    classes all see the same WM."""
    rim = p.rim.copy()
    if wm is not None:
        if wm.shape != rim.shape:
            raise ValueError(f"WM label {wm.shape} is not on the box {rim.shape}")
        rim[(rim == 2) & ~wm] = 3
        rim[wm] = 2
    vox = np.array(p.voxel_mm)
    miss = missing_tissue(rim, p.flash, p.voxel_mm)
    ctx = (rim == 3) & ~miss
    wm_ = rim == 2
    mb = ctx.astype(float)
    det = {}
    for i in (1, 2):
        L = np.log(np.maximum(p.echoes[..., i].astype(float), 1.0))
        det[i] = (L - S.masked_smooth(L, mb, TREND)) * mb
    labels = np.full(rim.shape, 4, np.int16)
    labels[ctx] = 0
    labels[rim == 1] = 1
    labels[wm_] = 2
    labels[miss] = 3
    return Tissue(
        p, vox, rim, miss, ctx, wm_, det, face_distance(rim.shape, vox), labels
    )


def load_tissue(
    lo: tuple[int, int, int],
    hi: tuple[int, int, int],
    name: str = "box",
    directory: Path | None = None,
    changes: WMChanges | None = None,
) -> Tissue:
    """The box [lo, hi) of the full grid (0-based voxels): step 1. changes:
    the handle-free WM label's changes (baillarger.white.wm_changes); None
    keeps the aseg's WM label."""
    p = load_box(lo, hi, name=name, directory=directory)
    wm = None if changes is None else white_matter_box(lo, hi, changes, directory)
    return tissue(p, wm)


def base_tensor(t: Tissue) -> F64:
    """The structure tensor of echoes 2 and 3 over the cortex, summed."""
    m = t.ctx.astype(float)
    out: F64 = S.structure_tensor(t.det[1], m) + S.structure_tensor(t.det[2], m)
    return out


def data_normal(tensor: F64, ctx: Bool) -> tuple[F64, F64]:
    """The leading unit eigenvector (..., 3) and the coherence (zero outside
    the cortex) of a tensor field, in float64."""
    t = tensor
    Mt = np.stack(
        [
            np.stack([t[0], t[1], t[2]], -1),
            np.stack([t[1], t[3], t[4]], -1),
            np.stack([t[2], t[4], t[5]], -1),
        ],
        -2,
    )
    w, V = np.linalg.eigh(Mt)
    nh: F64 = V[..., :, 2]
    coh = np.clip(
        (w[..., 2] - w[..., 1]) / np.maximum(w[..., 2] + w[..., 1], 1e-12), 0, 1
    )
    return nh, np.where(ctx, coh, 0.0)


# ---------------------------------------------------------------- roots


def white_mesh(wm: Bool, sigma: float = MESH_SIGMA) -> tuple[F64, I64, F64]:
    """Vertices (voxel coordinates), faces (oriented out of the WM) and unit
    normals (out of the WM) of the binary WM label's surface: step 2
    (baillarger.white.white_mesh, and baillarger.mesh_smoothing's normals)."""
    v, f = white_surface(wm, sigma)
    return v, f, smooth_normals(v, f)


def root_vertices(v: F64, n: F64) -> I64:
    """One vertex per voxel cell and normal side (dominant axis and sign): the
    one nearest the cell's centre. Sorted vertex indices."""
    ax = np.argmax(np.abs(n), 1)
    side = ax * 2 + (n[np.arange(len(n)), ax] > 0)
    cell = np.floor(v).astype(np.int64)
    key = np.c_[cell, side]
    dist = np.linalg.norm(v - (cell + 0.5), axis=1)
    o = np.lexsort((dist, *key.T[::-1]))
    k = key[o]
    first = np.r_[True, np.any(k[1:] != k[:-1], 1)]
    out: I64 = np.sort(o[first])
    return out


# ---------------------------------------------------------------- the one pass

# half of the 26-neighbourhood (each unordered pair once)
_N13 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


def _shifted_pairs(a: Bool, b: Bool, off: tuple[int, int, int]) -> tuple[I64, I64]:
    """Flat indices (p, q) with q = p + off, a[p] and b[q]."""
    sl_p, sl_q = [], []
    for o, n in zip(off, a.shape, strict=True):
        sl_p.append(slice(max(0, -o), n - max(0, o)))
        sl_q.append(slice(max(0, o), n - max(0, -o)))
    both = a[tuple(sl_p)] & b[tuple(sl_q)]
    idx = np.nonzero(both)
    p = np.ravel_multi_index(
        tuple(i + s.start for i, s in zip(idx, sl_p, strict=True)), a.shape
    )
    q = np.ravel_multi_index(
        tuple(i + s.start for i, s in zip(idx, sl_q, strict=True)), a.shape
    )
    return p.astype(np.int64), q.astype(np.int64)


def geodesic_patches(graph: csr_matrix, spacing: float) -> tuple[I64, I64]:
    """Centres chosen greedily in index order, each at least `spacing` along
    the graph from every earlier one, then every node assigned to its nearest
    centre (a geodesic Voronoi partition: each cell is connected). Returns
    the centre node of each patch and the patch of each node."""
    n = graph.shape[0]
    covered = np.zeros(n, bool)
    centres = []
    for i in range(n):
        if covered[i]:
            continue
        centres.append(i)
        d = dijkstra(graph, indices=i, limit=spacing)
        covered |= d < spacing
    cen = np.array(centres, np.int64)
    _, _, src = dijkstra(graph, indices=cen, min_only=True, return_predecessors=True)
    lookup = -np.ones(n, np.int64)
    lookup[cen] = np.arange(cen.size)
    return cen, lookup[src]


@dataclass
class Setup:
    """Sources, domain and root patches of the one pass (voxel units)."""

    wm: Bool
    cortex: Bool
    #: WM voxels 6-adjacent to the cortex
    seeds: Bool
    #: cortex | seeds
    domain: Bool
    #: distance to the nearest WM voxel centre (voxels)
    edt: F64
    #: (..., 3) unit direction of the march's metric
    direction: F64
    #: patch id at each seed voxel, -1 elsewhere
    patch: I64
    npatch: int
    #: sorted codes a * npatch + b (a < b) of neighbouring patches
    adjacent: I64
    #: (npatch, 3) position of each patch's centre seed
    centre: F64
    #: seed graph (edge lengths in voxels)
    graph: csr_matrix
    #: per patch, its centre seed (compact id)
    rep: I64
    #: flat index of each seed (compact id order)
    sflat: I64


def setup(
    wm: Bool, cortex: Bool, patch: int = PATCH, direction: F64 | None = None
) -> Setup:
    """Seeds, domain, and root patches. The seed graph joins 26-neighbour
    seeds whose EDT gradients agree. direction: the march's (default: the
    unit EDT gradient)."""
    near_ctx = ndi.binary_dilation(cortex, ndi.generate_binary_structure(3, 1))
    seeds = wm & near_ctx
    domain = cortex | seeds
    edt = ndi.distance_transform_edt(~wm).astype(np.float64)
    g = np.stack(S.gradient(ndi.gaussian_filter(edt, 0.7)), -1)
    nrm = np.linalg.norm(g, axis=-1, keepdims=True)
    unit = np.divide(g, nrm, out=np.zeros_like(g), where=nrm > 1e-6)
    if direction is None:
        direction = unit
    sflat = np.flatnonzero(seeds.ravel())
    sid = -np.ones(seeds.size, np.int64)
    sid[sflat] = np.arange(sflat.size)
    un = unit.reshape(-1, 3)
    rows, cols, lens = [], [], []
    for off in _N13:
        p, q = _shifted_pairs(seeds, seeds, off)
        ok = np.sum(un[p] * un[q], -1) > 0.0
        rows.append(sid[p[ok]])
        cols.append(sid[q[ok]])
        lens.append(np.full(ok.sum(), float(np.linalg.norm(off))))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    ln = np.concatenate(lens)
    graph = coo_matrix((ln, (r, c)), shape=(sflat.size,) * 2).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    rep, comp = geodesic_patches(graph, float(patch))
    npatch = rep.size
    pid = -np.ones(seeds.shape, np.int64)
    pid.ravel()[sflat] = comp
    a, b = comp[r], comp[c]
    diff = a != b
    lo, hi = np.minimum(a[diff], b[diff]), np.maximum(a[diff], b[diff])
    adjacent = np.unique(lo * npatch + hi)
    pos = np.stack(np.unravel_index(sflat, seeds.shape), -1)
    centre = pos[rep].astype(float)
    return Setup(
        wm, cortex, seeds, domain, edt, direction, pid, npatch, adjacent, centre,
        graph, rep, sflat,
    )  # fmt: skip


def run(
    s: Setup,
    mu: F64 | None = None,
    sigma: F64 | None = None,
    ratio: float = RATIO,
) -> Fronts:
    """The one pass. mu, sigma per patch (None: equal arrival; otherwise the
    patches compete by (d - mu) / sigma)."""
    m = riemann_from_direction(s.direction, ratio)
    if mu is None:
        return march(m, s.patch, mask=s.domain)
    assert sigma is not None
    return march(m, s.patch, mask=s.domain, start_time=-mu / sigma, speed=sigma)


@dataclass
class OnePass:
    """The equal-arrival one pass, its root patches and their distances."""

    setup: Setup
    fronts: Fronts
    #: patch centre to patch centre along the white surface (mm), inf beyond
    #: the limit
    D: F32
    #: the arrival, 0 in the WM and its maximum where no front arrived
    u: F64
    tree: cKDTree
    seed_patch: I64

    def patch_at(self, x: npt.ArrayLike) -> I64:
        """The root patch of the nearest seed voxel to each position."""
        out: I64 = self.seed_patch[self.tree.query(np.asarray(x, np.float64))[1]]
        return out


def one_pass(
    t: Tissue, direction: F64, limit_mm: float = 8.0, chunk: int = 200
) -> OnePass:
    """Step 4's one pass along direction (oriented_normal of the data)."""
    wm = t.wm
    h = float(np.mean(t.vox))
    s = setup(wm, t.ctx, direction=direction)
    f_eq = run(s)
    seed_pos = np.stack(np.unravel_index(s.sflat, wm.shape), -1).astype(float)
    D = np.full((s.npatch, s.npatch), np.inf, np.float32)
    for a in range(0, s.npatch, chunk):
        d = dijkstra(s.graph, indices=s.rep[a : a + chunk], limit=limit_mm / h)
        D[a : a + chunk] = d[:, s.rep] * h
    arr = f_eq.arrival
    fin = np.isfinite(arr)
    u = np.where(wm, 0.0, np.where(fin, arr, np.nanmax(arr[fin])))
    return OnePass(s, f_eq, D, u, cKDTree(seed_pos), s.patch.ravel()[s.sflat])


# ---------------------------------------------------------------- traces


@dataclass
class ColumnTraces:
    """The traces of a box and what made them."""

    tracer: str
    #: white mesh: vertices (voxels), faces, unit normals out of the WM
    v: F64
    f: I64
    nrm: F64
    #: the root vertex of each trace
    tv: I64
    #: (n, NMAX, 3) float32 positions, nan past the end; sample i at
    #: i * STEP_MM along the trace
    pos: F32
    ns: I64
    #: the primary's sample count, last position and outcome, and the final
    #: outcome (streamlines outcome codes)
    ns1: I64
    end1: F32
    out1: I64
    outc: I64
    #: 0 primary, 1 fallback
    fallback: I8
    cls: I8
    root_patch: I64
    end_patch: I64
    near_face: Bool
    #: the data's coherence (zero outside the cortex)
    coh: F64
    one_pass: OnePass
    #: the fitted field's solver info, if any
    notes: dict[str, Any]


def primary_tensor(tracer: str, base: F64, t: Tissue) -> tuple[F64, dict[str, Any]]:
    """The tensor field the tracer integrates, and the solver's info."""
    if tracer == "current":
        return base, {}
    if tracer in ("fitted", "fitted_blend"):
        T, info = S.fitted_field(base, t.ctx, t.wm, t.vox)
        return S.gradient_tensor(T, t.vox), info
    raise ValueError(f"tracer {tracer!r}: not one of {TRACERS}")


def classify(
    outc: I64, sep: npt.NDArray[np.floating[Any]], same_mm: float = SAME_MM
) -> I8:
    """Step 5's class from the final outcome and the distance along the white
    surface between the root's patch and the end's (mm)."""
    cls = np.full(len(outc), FAILED, np.int8)
    cls[outc == S.FLUID] = OPEN
    cls[(outc == S.WM) & (sep >= same_mm)] = CROSS
    cls[(outc == S.WM) & (sep < same_mm)] = OWN
    cls[outc == S.CONTACT] = TOCONTACT
    return cls


def trace_columns(t: Tissue, tracer: str = "fitted_blend") -> ColumnTraces:
    """Steps 2-5 on a box."""
    if tracer not in TRACERS:
        raise ValueError(f"tracer {tracer!r}: not one of {TRACERS}")
    base = base_tensor(t)
    nh, coh = data_normal(base, t.ctx)
    direction = S.oriented_normal(nh, coh, S.wm_direction(t.wm, t.vox), 0.1)
    del nh
    v, f, nrm = white_mesh(t.wm)
    tv = root_vertices(v, nrm)
    op = one_pass(t, direction)
    del direction
    root_patch = op.patch_at(v[tv])
    x0 = v[tv]
    field, notes = primary_tensor(tracer, base, t)
    del base
    blend = BLEND_MM if tracer == "fitted_blend" else None
    pos, ns1, out1 = S.trace(field, t.labels, t.dface, x0, nrm[tv], t.vox, NMAX, blend)
    del field
    fb = np.flatnonzero((out1 == S.TURN) | (out1 == S.LENGTH))
    lab_eq = op.fronts.label

    def far(i_paths: I64, q: I64) -> Bool:
        lab = lab_eq[tuple(q.T)]
        rp = root_patch[fb[i_paths]]
        out: Bool = (lab >= 0) & (op.D[rp, np.maximum(lab, 0)] > SAME_MM)
        return out

    end1 = pos[np.arange(len(tv)), ns1 - 1].copy()
    ns, outc = ns1.copy(), out1.copy()
    if fb.size:
        pos2, ns2, out2 = S.ascend(op.u, t.labels, t.dface, x0[fb], t.vox, NMAX, far)
        pos[fb] = pos2
        ns[fb] = ns2
        outc[fb] = out2
    fallback = np.zeros(len(tv), np.int8)
    fallback[fb] = 1
    endp = pos[np.arange(len(tv)), ns - 1].astype(float)
    end_patch = op.patch_at(endp)
    cls = classify(outc, op.D[root_patch, end_patch])
    q = np.clip(np.round(x0).astype(int), 0, np.array(t.wm.shape) - 1)
    near_face = t.dface[tuple(q.T)] < FACE_MM
    return ColumnTraces(
        tracer=tracer,
        v=v,
        f=f,
        nrm=nrm,
        tv=tv,
        pos=pos,
        ns=ns,
        ns1=ns1,
        end1=end1,
        out1=out1,
        outc=outc,
        fallback=fallback,
        cls=cls,
        root_patch=root_patch,
        end_patch=end_patch,
        near_face=near_face,
        coh=coh,
        one_pass=op,
        notes=notes,
    )
