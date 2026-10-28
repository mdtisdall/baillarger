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
   baillarger.white), optionally the pial-side label changes
   (baillarger.pial_labels: cortex relabelled as fluid), the missing tissue
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
4. Where a trace cannot go on, by METHODS:
   - "crest" (the default): the fitted field T extended one voxel into the
     fluid and WM_BAND_MM into the WM (streamlines.fitted_field with fluid
     and wm_band_mm), so that its gradient is T's own up to the pia; no turn
     rule and no fallback: columns are traced through. Where a column meets
     another bank's field, it crosses T's crest (the voxels where T rises
     from two sides into a kink, streamlines.crest_voxels); that is
     recorded (crest_at), not acted on: the crest is the midline between the
     two banks' WM, not the anatomical contact, which the reading of the
     columns has to place. A column that reflects off a crest it cannot
     cross and goes back into its own WM is cut at T's largest value along
     it (cut_at_crest), as CONTACT. With bend_deg, a column whose direction
     changes by more than bend_deg within BEND_MM also ends, at the bend's
     kink (end_at_bend), for comparison.
   - "traverse": every column is a complete ray, through any contact, by
     the crossing rule of baillarger.column_front (climb grad T; within the
     stencil of T's crest voxels keep heading; on leaving it, if T falls
     ahead, descend -grad T for good, into the far bank): the columns traced
     independently (column_front.grow with no coupling). Nothing is cut; the
     reading of the columns chooses where to cut. crest_at as "crest", and
     turn_at the sample where the column began to descend.
   - "normal": no field: each root's column is the straight path of its
     point of the white surface moving along its (smoothed) normal at
     constant speed (streamlines.ray), so the fronts are offsets of the white
     surface. It stops only at fluid, WM (after 0.3 mm), missing or other
     tissue, a face or the length cap. Where it passes the meeting of its
     bank's front with another's (the WM distance through tissue stops rising
     along it; streamlines.first_fall), crest_at records it. vol is the
     equivolume coordinate along it, from the root's curvatures.
   - "one-pass": T as fitted on the cortex alone, the turn rule (60 deg
     within 0.3 mm), and where the primary turned or reached the length cap,
     the fallback: ascent of the equal-arrival one pass (one_pass, an
     anisotropic fast march from every WM voxel 6-adjacent to the cortex,
     baillarger.fronts), stopped at the pass's contact (a voxel whose root
     patch is SAME_MM or more along the white surface from the path's own).
     The method before the crest method, kept for comparison.
   - "front": as "traverse", with the columns linked: the white mesh grown
     outward as one surface (baillarger.column_front), the moving vertices'
     directions smoothed along the mesh at MESH_SIGMA; each root's
     trajectory is its column, and its dual area (area, mm^2) and the
     volume it swept per unit root area (vol, mm: Bok's measure; vol /
     vol[end] is an equivolume relative depth) are recorded per sample.
   - "front-equivolume": no field: the white mesh grown along its own
     smoothed normals at equivolume speed, A0 / A (column_front.
     grow_equivolume), so the fronts are the equivolume layers and the
     columns bend to stay normal to them; tau records each sample's
     equivolume coordinate (mm, the front's time), and area and vol as
     "front" (vol / tau falls below 1 where the speed cap binds). Stops as
     "normal". crest_at is where the column meets another bank's front: its
     first sample in a voxel that a column from SAME_MM or more along the
     white surface reached first, at a smaller tau (meet_at).
5. Patches (mesh_patches): geodesic Voronoi cells PATCH_MM apart on the
   white mesh, and the distances between their centres along it
   (baillarger.mesh_graph). A trace's end patch is the patch of the vertex
   where it crossed the white mesh, on the side it came from, so the two
   sides of a thin WM blade stay apart.
6. Classes: OPEN (ends in fluid), CROSS (ends in WM at least SAME_MM along
   the white surface from its root), OWN (ends in WM nearer: back on its own
   bank), TOCONTACT (ends at a contact: at a bend, or at the one pass's
   contact), FAILED. Roots within FACE_MM of a box face are flagged
   (near_face).

Prototype entry point: prototypes/scripts/column_volume.py (--tracer).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix

from baillarger import column_front as CF
from baillarger import mesh_graph as G
from baillarger import streamlines as S
from baillarger.data import Patch, load_box
from baillarger.frame import mesh_curvatures, missing_tissue
from baillarger.fronts import Fronts, march, riemann_from_direction
from baillarger.white import WhiteMesh, WMChanges, white_matter_box, white_mesh_normals

if TYPE_CHECKING:
    from baillarger.pial_labels import LabelChanges

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I8 = npt.NDArray[np.int8]
type I64 = npt.NDArray[np.int64]

TRACERS = ("current", "fitted", "fitted_blend")
#: Trace classes.
OPEN, CROSS, OWN, TOCONTACT, FAILED = range(5)
CLASS = ("open", "crossing", "own bank", "to the contact", "failed")
#: How a column ends where its trace cannot go on (module docstring, step 4).
METHODS = ("crest", "traverse", "normal", "one-pass", "front", "front-equivolume")
#: The fitted field's band into the WM for the crest method (mm), and the
#: window of its optional bend rule (mm).
WM_BAND_MM, BEND_MM = 0.5, 0.3
#: Trend scale of the echoes (voxels).
TREND = 4.0
#: Twicing scale of the white mesh (voxels).
MESH_SIGMA = 2.0
#: The blended start's length at coherence 1 (mm).
BLEND_MM = 0.3
#: Root patch spacing on the white mesh (mm), the reach of the patch
#: distances (mm), the one pass's seed patch spacing (voxels) and its cost
#: ratio across the normal, the distance along the white surface that
#: separates two banks (mm), and the box-face margin (mm).
PATCH_MM, LIMIT_MM = 0.96, 8.0
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


def tissue(
    p: Patch,
    wm: Bool | None = None,
    pial_changes: "LabelChanges | None" = None,
    reviewed_only: bool = False,
) -> Tissue:
    """Step 1 on a loaded box. wm: the WM label to use (default the aseg's,
    rim == 2). It replaces the aseg's in the rim labels before anything is
    derived from them: voxels it adds become WM, and aseg WM voxels it leaves
    out (a cut across a bridge or a pinhole) become cortex. So the white
    mesh, the fitted field's boundary, the trace stops, the one pass and the
    classes all see the same WM. pial_changes: the pial-side label changes
    (baillarger.pial_labels: cortex relabelled as fluid, accepted in review,
    and brush edits; with proposed, the unreviewed proposals too), applied
    after the WM label and never to a WM voxel."""
    rim = p.rim.copy()
    if wm is not None:
        if wm.shape != rim.shape:
            raise ValueError(f"WM label {wm.shape} is not on the box {rim.shape}")
        rim[(rim == 2) & ~wm] = 3
        rim[wm] = 2
    if pial_changes is not None:
        rim = pial_changes.apply(rim, p.lo, reviewed_only)
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
    pial_changes: "LabelChanges | None" = None,
    reviewed_only: bool = False,
) -> Tissue:
    """The box [lo, hi) of the full grid (0-based voxels): step 1. changes:
    the handle-free WM label's changes (baillarger.white.wm_changes); None
    keeps the aseg's WM label. pial_changes, reviewed_only: the pial-side label changes
    (baillarger.pial_labels.LabelChanges; see tissue)."""
    p = load_box(lo, hi, name=name, directory=directory)
    wm = None if changes is None else white_matter_box(lo, hi, changes, directory)
    return tissue(p, wm, pial_changes, reviewed_only)


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
    return white_mesh_normals(wm, sigma)


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
    rep, comp = G.geodesic_patches(graph, float(patch))
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
    """The equal-arrival one pass, for the fallback."""

    setup: Setup
    fronts: Fronts
    #: the arrival, 0 in the WM and its maximum where no front arrived
    u: F64
    #: the mesh patch of each of the pass's seed patches (by its centre seed)
    to_mesh_patch: I64


def one_pass(
    t: Tissue, direction: F64, v: F64, nrm: F64, patches: "Patches"
) -> OnePass:
    """Step 4's one pass along direction (oriented_normal of the data)."""
    wm = t.wm
    s = setup(wm, t.ctx, direction=direction)
    f_eq = run(s)
    arr = f_eq.arrival
    fin = np.isfinite(arr)
    u = np.where(wm, 0.0, np.where(fin, arr, np.nanmax(arr[fin])))
    centre = np.stack(np.unravel_index(s.sflat[s.rep], wm.shape), -1).astype(float)
    inward = -s.direction.reshape(-1, 3)[s.sflat[s.rep]]
    j = G.crossing_vertex(v, nrm, centre, inward, t.vox)
    return OnePass(s, f_eq, u, patches.of_vertex[j])


# ---------------------------------------------------------------- patches


@dataclass
class Patches:
    """Geodesic Voronoi patches on the white mesh, for step 5's bank rule."""

    graph: csr_matrix
    #: the centre vertex of each patch
    rep: I64
    #: the patch of each vertex
    of_vertex: I64
    #: patch centre to patch centre along the mesh (mm), inf beyond LIMIT_MM
    D: F32


def mesh_patches(v: F64, f: I64, vox: F64) -> Patches:
    """Patches PATCH_MM apart on the white mesh and their distances."""
    g = G.edge_graph(v, f, vox)
    rep, of_vertex = G.geodesic_patches(g, PATCH_MM)
    return Patches(g, rep, of_vertex, G.patch_distances(g, rep, LIMIT_MM))


# ---------------------------------------------------------------- traces


@dataclass
class ColumnTraces:
    """The traces of a box and what made them."""

    tracer: str
    method: str
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
    #: the patches of the root vertex and of the end's vertex (where the
    #: trace crossed the white mesh, by mesh_graph.crossing_vertex)
    root_patch: I64
    end_patch: I64
    near_face: Bool
    #: the data's coherence (zero outside the cortex)
    coh: F64
    patches: Patches
    #: method "crest": the first sample of each column in a crest voxel of T
    #: (where it meets another bank's field; streamlines.crest_voxels), -1 if
    #: none; method "one-pass": -1
    crest_at: I64
    #: the one pass (method "one-pass"; None for "crest")
    one_pass: OnePass | None
    #: the fitted field T (mm, full grid; None for "one-pass" with "current")
    T: F64 | None
    #: the fitted field's solver info, if any
    notes: dict[str, Any]
    #: per sample, the volume the column has swept per unit area of its root
    #: (mm), NaN past the end; a relative equivolume depth is vol / vol at
    #: the end. Method "normal": from the root's principal curvatures, A(s) =
    #: (1 + k1 s)(1 + k2 s); "front" and "front-equivolume": measured, the
    #: flux of the root vertex's dual area vector over the steps over its
    #: dual area at the root. None otherwise
    vol: F32 | None = None
    #: methods "front" and "front-equivolume": per sample, the root vertex's
    #: dual area (mm^2); None otherwise
    area: F32 | None = None
    #: methods "traverse" and "front": the sample where the column began to
    #: descend T (-1 if never); None otherwise
    turn_at: I64 | None = None
    #: method "front-equivolume": per sample, the equivolume coordinate (mm,
    #: the front's time; column_front.grow_equivolume); None otherwise
    tau: F32 | None = None


def level_set_concavity(
    v: F64, f: I64, nrm: F64, T: F64, ctx: Bool, vox: F64, sigma_mm: float = 0.5
) -> F64:
    """Per cortex voxel, the curvature (1/mm) of T's level set there that is
    concave seen from the flow, from the white mesh: the most concave
    principal curvature k2 < 0 (frame.mesh_curvatures, smoothed over
    sigma_mm) of the vertex the descent from the voxel would reach
    (mesh_graph.crossing_vertex along -grad T), carried to the voxel's depth
    d = T (mm) as the level set of an offset surface: |k2| / (1 - |k2| d).
    Past the focus (|k2| d >= 1, where the offsets meet) and anywhere it
    would exceed it, it is capped at 1 / h (h the mean voxel size), the
    sharpest curvature a voxel grid resolves. 0 outside the cortex."""
    _, k2 = mesh_curvatures(v * vox, f, nrm, sigma_mm)
    conc = np.maximum(0.0, -k2)
    pts = np.argwhere(ctx).astype(float)
    gT = np.stack([g_[ctx] for g_ in S.gradient(T, vox)], -1)
    j = G.crossing_vertex(v, nrm, pts, -gT, vox)
    kap = conc[j]
    d = np.maximum(T[ctx], 0.0)
    cap = 1.0 / float(np.mean(vox))
    lev = np.where(kap * d < 1.0, kap / np.maximum(1.0 - kap * d, 1e-12), cap)
    out = np.zeros(T.shape)
    out[ctx] = np.minimum(lev, cap)
    return out


def cut_at_crest(pos: F32, ns: I64, out: I64, T: F64, n0: F64) -> I64:
    """The crest cut: columns that ended in the WM entering it against their
    root's outward normal n0 (n, 3; they reflected off a crest of T and went
    back instead of on) are cut at the sample of T's largest value along
    them and end there as CONTACT. pos, ns and out are updated in place.
    Returns the columns cut."""
    k = np.arange(len(ns))
    d_end = (pos[k, ns - 1] - pos[k, np.maximum(ns - 2, 0)]).astype(float)
    back = np.flatnonzero((out == S.WM) & ((d_end * n0).sum(1) < 0))
    if back.size:
        Ta = S.sample(T, pos[back])
        top = np.argmax(np.where(np.isfinite(Ta), Ta, -np.inf), 1)
        for i, m in zip(back, top, strict=True):
            pos[i, m + 1 :] = np.nan
            ns[i] = m + 1
        out[back] = S.CONTACT
    return back


def first_in(mask: Bool, pos: F32, ns: I64) -> I64:
    """Per path, the index of its first sample in a voxel of mask (rounded
    positions), -1 if none."""
    out = np.full(len(ns), -1, np.int64)
    shape = np.array(mask.shape)
    for i in range(len(ns)):
        q = np.clip(np.round(pos[i, : ns[i]]).astype(int), 0, shape - 1)
        hit = np.flatnonzero(mask[tuple(q.T)])
        if hit.size:
            out[i] = hit[0]
    return out


def equivolume(
    v: F64, f: I64, nrm: F64, tv: I64, ns: I64, vox: F64, sigma_mm: float = 0.5
) -> F32:
    """Per column (n, NMAX), the volume swept per unit area of its root at
    each sample (mm): the integral of A(s) = (1 + k1 s)(1 + k2 s), k1 >= k2
    the white surface's principal curvatures at the root (frame.
    mesh_curvatures, smoothed over sigma_mm; > 0 convex), for a straight
    column along the root's normal (an offset front). A is clipped at 0 past
    a focus. NaN past each column's end."""
    k1, k2 = mesh_curvatures(v * vox, f, nrm, sigma_mm)
    s = np.arange(NMAX) * S.STEP_MM
    A = np.clip((1 + k1[tv, None] * s) * (1 + k2[tv, None] * s), 0.0, None)
    # trapezoidal, from 0 at the root
    out = np.zeros((len(tv), NMAX))
    out[:, 1:] = np.cumsum(0.5 * (A[:, 1:] + A[:, :-1]) * S.STEP_MM, axis=1)
    out[np.arange(NMAX)[None, :] >= ns[:, None]] = np.nan
    return out.astype(np.float32)


def meet_at(
    pos: F32, ns: I64, tau: F32, root_patch: I64, D: F32, same_mm: float = SAME_MM
) -> I64:
    """Per column, its first sample in a voxel (rounded position) that a
    column whose root patch is same_mm or more from its own along the white
    surface reached first (smallest tau): where its bank's front meets
    another's. -1 if none."""
    n, m = tau.shape
    col, i = np.nonzero(np.arange(m)[None, :] < ns[:, None])
    q = np.round(pos[col, i]).astype(np.int64)
    q -= q.min(0)
    e = q.max(0) + 1
    key = (q[:, 0] * e[1] + q[:, 1]) * e[2] + q[:, 2]
    o = np.lexsort((tau[col, i], key))
    first = np.r_[True, key[o][1:] != key[o][:-1]]
    owner = np.empty(len(key), np.int64)
    grp = np.cumsum(first) - 1
    owner[o] = col[o][first][grp]
    far = D[root_patch[col], root_patch[owner]] >= same_mm
    out = np.full(n, m, np.int64)
    np.minimum.at(out, col[far], i[far])
    out[out == m] = -1
    return out


def end_at_bend(pos: F32, ns: I64, out: I64, lag: int) -> I64:
    """Columns whose trace stopped at a bend (TURN) end at the bend itself: the
    sample, within the last lag steps, where the direction changed most from
    one step to the next (the kink). They end there as CONTACT. pos, ns and
    out are updated in place. Returns the columns ended."""
    bent = np.flatnonzero(out == S.TURN)
    for i in bent:
        n = int(ns[i])
        p = pos[i, max(0, n - lag - 2) : n].astype(float)
        d = np.diff(p, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        change = 1.0 - (d[1:] * d[:-1]).sum(1)  # per sample between two steps
        k = max(0, n - lag - 2) + 1 + int(np.argmax(change))
        pos[i, k + 1 :] = np.nan
        ns[i] = k + 1
    out[bent] = S.CONTACT
    return bent


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


def finish(
    t: Tissue,
    v: F64,
    nrm: F64,
    tv: I64,
    P: Patches,
    op: OnePass,
    pos: F32,
    ns1: I64,
    out1: I64,
) -> tuple[F32, I64, I64, I8, I64, I8]:
    """Steps 4 and 5 after the primary trace: the fallback where it turned or
    reached the cap, and the classes. pos is updated in place. Returns pos,
    the sample counts, the final outcomes, the fallback flags, the end
    patches and the classes."""
    root_patch = P.of_vertex[tv]
    x0 = v[tv]
    fb = np.flatnonzero((out1 == S.TURN) | (out1 == S.LENGTH))
    lab_eq = op.fronts.label

    def far(i_paths: I64, q: I64) -> Bool:
        lab = lab_eq[tuple(q.T)]
        rp = root_patch[fb[i_paths]]
        mp = op.to_mesh_patch[np.maximum(lab, 0)]
        out: Bool = (lab >= 0) & (P.D[rp, mp] > SAME_MM)
        return out

    ns, outc = ns1.copy(), out1.copy()
    if fb.size:
        pos2, ns2, out2 = S.ascend(
            op.u, t.labels, t.dface, x0[fb], t.vox, NMAX, far=far
        )
        pos[fb] = pos2
        ns[fb] = ns2
        outc[fb] = out2
    fallback = np.zeros(len(tv), np.int8)
    fallback[fb] = 1
    k = np.arange(len(tv))
    endp = pos[k, ns - 1].astype(float)
    last = endp - pos[k, np.maximum(ns - 2, 0)]
    last = np.where((ns > 1)[:, None], last, -nrm[tv])  # one sample: inward
    end_patch = P.of_vertex[G.crossing_vertex(v, nrm, endp, last, t.vox)]
    cls = classify(outc, P.D[root_patch, end_patch])
    return pos, ns, outc, fallback, end_patch, cls


@dataclass
class Crest:
    """The fitted field of the crest methods and its crest."""

    T: F64
    notes: dict[str, Any]
    #: step_noise: T's noise over one voxel step (mm)
    noise: F64
    crest: Bool
    #: the lattice direction across the kink at each crest voxel
    across: I8


def fitted_crest(t: Tissue, base: F64, v: F64, f: I64, nrm: F64) -> Crest:
    """T as the crest methods fit it (extended one voxel into the fluid and
    WM_BAND_MM into the WM), its step noise, and its crest
    (streamlines.crest_kinks, with the white mesh's level-set curvature)."""
    T, notes = S.fitted_field(
        base, t.ctx, t.wm, t.vox, fluid=t.labels == 1, wm_band_mm=WM_BAND_MM
    )
    g = S.tissue_wm_direction(t.wm, t.ctx, t.vox)
    noise, _ = S.step_noise(base, t.ctx, g, t.vox)
    del g
    kappa = level_set_concavity(v, f, nrm, T, t.ctx, t.vox)
    free = S.fitted_free(t.ctx, t.wm, t.vox, fluid=t.labels == 1, wm_band_mm=WM_BAND_MM)
    crest, across = S.crest_kinks(T, t.ctx, noise, kappa, t.vox, free=free)
    return Crest(T, notes, noise, crest, across)


def trace_columns(
    t: Tissue,
    tracer: str = "fitted_blend",
    method: str = "crest",
    bend_deg: float | None = None,
    mesh: WhiteMesh | None = None,
) -> ColumnTraces:
    """Steps 2-6 on a box. mesh: a saved white mesh of this box and WM label
    (baillarger.white.WhiteMesh; ValueError if it is not) instead of making
    it (white_mesh)."""
    if tracer not in TRACERS:
        raise ValueError(f"tracer {tracer!r}: not one of {TRACERS}")
    if method not in METHODS:
        raise ValueError(f"method {method!r}: not one of {METHODS}")
    base = base_tensor(t)
    nh, coh = data_normal(base, t.ctx)
    if mesh is None:
        v, f, nrm = white_mesh(t.wm)
    else:
        mesh.check(t.patch.lo, t.patch.hi, t.wm)
        v, f, nrm = mesh.v, mesh.f, mesh.normals
    tv = root_vertices(v, nrm)
    P = mesh_patches(v, f, t.vox)
    root_patch = P.of_vertex[tv]
    x0 = v[tv]
    blend = BLEND_MM if tracer == "fitted_blend" else None
    op: OnePass | None = None
    T: F64 | None = None
    notes: dict[str, Any] = {}
    vol: F32 | None = None
    front: CF.Front | None = None
    if method == "front-equivolume":
        del base, nh
        front = CF.grow_equivolume(t.labels, t.dface, v, f, tv, t.vox, NMAX)
        assert front.tau is not None
        pos, ns1, out1 = front.pos, front.ns, front.out
        end1 = pos[np.arange(len(tv)), ns1 - 1].copy()
        ns, outc = ns1.copy(), out1.copy()
        crest_at = meet_at(pos, ns, front.tau, root_patch, P.D)
        vol = front.vol / front.area[:, :1]
        fallback = np.zeros(len(tv), np.int8)
        k = np.arange(len(tv))
        endp = pos[k, ns - 1].astype(float)
        last = endp - pos[k, np.maximum(ns - 2, 0)]
        last = np.where((ns > 1)[:, None], last, -nrm[tv])
        end_patch = P.of_vertex[G.crossing_vertex(v, nrm, endp, last, t.vox)]
        cls = classify(outc, P.D[root_patch, end_patch])
    elif method == "normal":
        del base, nh
        pos, ns1, out1 = S.ray(t.labels, t.dface, x0, nrm[tv], t.vox, NMAX)
        end1 = pos[np.arange(len(tv)), ns1 - 1].copy()
        ns, outc = ns1.copy(), out1.copy()
        D = S.tissue_wm_distance(t.wm, t.ctx, t.vox)
        crest_at = S.first_fall(D, pos, ns)  # NaN in obstacles: skipped
        del D
        vol = equivolume(v, f, nrm, tv, ns, t.vox)
        fallback = np.zeros(len(tv), np.int8)
        k = np.arange(len(tv))
        endp = pos[k, ns - 1].astype(float)
        last = endp - pos[k, np.maximum(ns - 2, 0)]
        last = np.where((ns > 1)[:, None], last, -nrm[tv])
        end_patch = P.of_vertex[G.crossing_vertex(v, nrm, endp, last, t.vox)]
        cls = classify(outc, P.D[root_patch, end_patch])
    elif method in ("crest", "traverse", "front"):
        if method != "crest" and tracer == "current":
            raise ValueError(f"method {method!r} follows the fitted field")
        C = fitted_crest(t, base, v, f, nrm)
        T, notes, crest = C.T, C.notes, C.crest
        field: F64 | None = None
        if method == "crest":
            field = base if tracer == "current" else S.gradient_tensor(T, t.vox)
        del base, nh, C
        if field is None:
            sigma = MESH_SIGMA if method == "front" else 0.0
            front = CF.grow(
                T, t.labels, t.dface, v, f, nrm, tv, t.vox, NMAX, sigma, blend,
                crest,
            )  # fmt: skip
            pos, ns1, out1 = front.pos, front.ns, front.out
            if method == "front":
                vol = front.vol / front.area[:, :1]
        else:
            pos, ns1, out1 = S.trace(
                field, t.labels, t.dface, x0, nrm[tv], t.vox, NMAX, blend,
                turn_deg=bend_deg, turn_mm=BEND_MM,
            )  # fmt: skip
            del field
        end1 = pos[np.arange(len(tv)), ns1 - 1].copy()
        ns, outc = ns1.copy(), out1.copy()
        if method == "crest":
            cut_at_crest(pos, ns, outc, T, nrm[tv])
        if bend_deg is not None:
            end_at_bend(pos, ns, outc, max(1, int(round(BEND_MM / S.STEP_MM))))
        crest_at = first_in(crest, pos, ns)
        fallback = np.zeros(len(tv), np.int8)
        k = np.arange(len(tv))
        endp = pos[k, ns - 1].astype(float)
        last = endp - pos[k, np.maximum(ns - 2, 0)]
        last = np.where((ns > 1)[:, None], last, -nrm[tv])
        end_patch = P.of_vertex[G.crossing_vertex(v, nrm, endp, last, t.vox)]
        cls = classify(outc, P.D[root_patch, end_patch])
    else:
        direction = S.oriented_normal(nh, coh, S.wm_direction(t.wm, t.vox), 0.1)
        del nh
        op = one_pass(t, direction, v, nrm, P)
        del direction
        if tracer == "current":
            field = base
        else:
            T, notes = S.fitted_field(base, t.ctx, t.wm, t.vox)
            field = S.gradient_tensor(T, t.vox)
        del base
        pos, ns1, out1 = S.trace(
            field, t.labels, t.dface, x0, nrm[tv], t.vox, NMAX, blend
        )
        del field
        end1 = pos[np.arange(len(tv)), ns1 - 1].copy()
        pos, ns, outc, fallback, end_patch, cls = finish(
            t, v, nrm, tv, P, op, pos, ns1, out1
        )
        crest_at = np.full(len(tv), -1, np.int64)
    q = np.clip(np.round(x0).astype(int), 0, np.array(t.wm.shape) - 1)
    near_face = t.dface[tuple(q.T)] < FACE_MM
    return ColumnTraces(
        tracer=tracer,
        method=method,
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
        patches=P,
        crest_at=crest_at,
        one_pass=op,
        T=T,
        notes=notes,
        vol=vol,
        area=front.area if method in ("front", "front-equivolume") and front else None,
        turn_at=None if front is None else front.turn_at,
        tau=None if front is None else front.tau,
    )
