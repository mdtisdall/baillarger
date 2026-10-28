"""The white surface on the image's own GM/WM edge.

Plan: reports/Crossing paths and contacts from laminar profiles.md, *The white
surface on the image's edge*. The roots must include the GM/WM boundary: a
root a little inside the WM is safe, a root in the cortex past the edge is
not. Every vertex must lie at least MARGIN_MM (0.1 mm, about the edge's blur
width, so the whole transition lies above it) inside the image's edge along
its outward normal; it may lie deeper. So only the surface with less than
that margin moves, inward into the WM, to exactly the margin, and nothing
moves outward: the label loses the WM voxels whose centres lie less than
the margin inside the edge, and the mesh moves the vertices still short of
it by the rest, under half a voxel.

1. tissue_contrast: the intensity divided by its normalized convolution over
   the WM at trend voxels, minus 1, so that a slow bias field goes and the
   WM/GM step stays where it is. A divisor with a gradient across the edge
   moves the steepest point by about 2 s² g (s the edge's blur, g the
   divisor's relative gradient): over WM and cortex together the trend holds
   the step itself (0.012 mm for a 40 % step 0.24 mm wide, at 8 voxels), over
   the WM alone it does not. Not the log either: the log of a blurred step is
   steepest nearer its dark side (0.03 mm for the same step). Zero where no
   WM is within reach of the trend.
2. EdgeField, edge_offsets (step 1, the measure): the image's gradient,
   Gaussian-smoothed at scale_mm, is sampled by cubic spline along each
   point's normal over +-window_mm, in steps of step_mm; the offset is where
   its component along the normal, signed so that WM to GM is positive, is
   largest (refined by a parabola through the three samples round the peak).
   Positive offsets mean the image's edge lies outside the point, in the
   cortex. The sign of the WM-to-GM change is the data's: the sign of the
   median derivative at the points, unless given.
3. peel, peel_blocks, edge_peel (step 2, the label): WM voxels next to the
   background (6-adjacent) are measured along the outward normal of the
   label smoothed at NORMAL_SIGMA voxels, and removed where their centre lies
   less than margin_mm inside the edge; repeated on the new boundary, at most
   ceil((window_mm + margin_mm) / voxel) + 1 rounds. A voxel whose edge is
   past the window's inner end is flagged and kept, and nothing deeper is
   tried there. The thin-blade guard keeps every voxel deeper than half the
   local thickness (its distance to the background more than half the
   largest such distance within GUARD_R voxels), so a blade is thinned to at
   most half and never cut. That is the edge's wish; the label loses only
   the wished voxels that are simple points on the Kuhn tetrahedralization
   (simple_peel: one at a time, outer rounds first), so its topology is
   kept: per-voxel decisions on noisy offsets would otherwise drill holes
   through thin WM and cut strands (on the full grid, 31 k new handles and
   4 k fragments). Wished voxels kept so are blocked: there the white
   surface cannot move in without changing the WM's topology. peel_blocks
   finds the wish in blocks with a halo (HALO voxels, more than the trend,
   the gradient, the spline, the guard and the rounds reach), so that the
   full grid fits in memory, and runs simple_peel once on the whole grid;
   edge_peel runs it on the aseg's WM label and echo 2. The label never
   grows. Handle removal follows on the peeled label
   (baillarger.white.wm_changes, removed=), with only the aseg's handles.
4. refine (step 4, the mesh): each vertex's target move is its offset minus
   margin_mm, with the edge's position smoothed along the mesh over sigma_mm
   (twicing; vertices with no edge in the window do not count): the smoothed
   offset plus the smoothed vertex position's offset from the vertex, so
   that a vertex's own jitter is not smoothed into its move. The move is the
   deeper of that and the vertex's own target, so that no vertex keeps less
   than the margin by its own measurement; then only inward moves are kept.
   A move is
   capped at just under half a voxel, so a vertex never passes a WM voxel
   centre and the mesh and the label agree about which centres are WM, and
   at half of (the local half-thickness less half a voxel), so the two sides
   of a thin blade stay apart. Vertices whose edge is past the window's inner
   end are flagged and not moved; vertices that a cap stops short of the
   margin are marked short. The faces, and so the topology, are kept.

On a curved edge the derivative scale moves the steepest point toward the
concave side by about scale_mm² H (H the mean curvature): 0.01 mm at
H = 1/mm and the default scale; it is not corrected.

Interpolation matters: linear interpolation makes the derivative piecewise
constant, and its peak falls on a voxel face (up to a voxel from the edge);
the spline leaves under 0.1 voxel on a step a voxel wide (tests/
test_wm_edge.py). Positions are voxel index coordinates; normals are unit
vectors in voxel coordinates (the mesh's, or the label's gradient),
converted to mm as covectors.

The reports: box_edges and write_box (a box's mesh against the edge, by
curvature and coherence, and step 4 with the edge measured again on the
refined mesh). The command line:

    python -m baillarger.wm_edge peel --out OUTDIR
    python -m baillarger.white changes --peel OUTDIR/wm_edge_peel.npz --out DIR
    python -m baillarger.wm_edge box --changes DIR/wm_changes.npz --out DIR
        --name NAME --lo X Y Z --hi X Y Z

peel writes wm_edge_peel.npz (EdgePeel), wm_edge_peel.json (counts) and
wm_edge_peel_overlay.nii.gz (full grid: 1 removed, 2 flagged, 3 blocked);
white
changes removes the handles of the peeled label; box writes write_box's
files, wm_edge_<NAME>.*.
"""

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from baillarger._jit import njit
from baillarger.data import ASEG_CORTEX, ASEG_WM, aseg_path, echo_path
from baillarger.handles import KUHN_OFFSETS, SIMPLE, kuhn_betti
from baillarger.mesh_smoothing import gauss_steps, smooth_normals, umbrella

type Bool = npt.NDArray[np.bool_]
type F64 = npt.NDArray[np.float64]
type I32 = npt.NDArray[np.int32]
type U8 = npt.NDArray[np.uint8]
type I64 = npt.NDArray[np.int64]

#: half-width of the search window along the normal (mm)
WINDOW_MM = 0.3
#: Gaussian scale of the image gradient (mm)
SCALE_MM = 0.1
#: sampling step along the normal (mm)
STEP_MM = 0.005
#: scale of the trend removed from the intensity (voxels)
TREND = 8.0
#: every vertex lies at least this far inside the edge (mm): about the edge's
#: blur width
MARGIN_MM = 0.1
#: smoothing of the targets along the mesh (mm)
REFINE_SIGMA_MM = 0.16
#: the label's normal: Gaussian scale (voxels)
NORMAL_SIGMA = 1.0
#: the thin-blade guard looks this far for the local half-thickness (voxels)
GUARD_R = 4
#: halo of a block in peel_blocks (voxels)
HALO = 56
#: block size of the full grid in edge_peel (voxels)
BLOCK = 128
_CHUNK = 20000
_FACE6 = ndi.generate_binary_structure(3, 1)


@dataclass(frozen=True)
class EdgeOffsets:
    """Per point: where the image's edge lies along the normal."""

    #: signed distance (mm) from the point to the edge, + toward cortex
    offset_mm: F64
    #: the derivative there (per mm), WM-to-GM positive
    slope: F64
    #: the peak is at the window's end: the edge is not inside the window
    at_end: Bool
    #: +1 if the image rises from WM to GM, -1 if it falls
    sign: float

    @property
    def inner_end(self) -> Bool:
        """No edge in the window, the search ending at its inner end."""
        out: Bool = self.at_end & (self.offset_mm < 0)
        return out


def tissue_contrast(echo: npt.ArrayLike, wm: Bool, trend: float = TREND) -> F64:
    """echo / (its normalized convolution over the WM, voxels: trend) - 1;
    zero where the WM's weight is under 1e-3."""
    x = np.asarray(echo, np.float64)
    m = np.asarray(wm, np.float64)
    w = ndi.gaussian_filter(m, trend)
    t = ndi.gaussian_filter(x * m, trend) / np.maximum(w, 1e-12)
    out: F64 = np.where(w > 1e-3, x / np.maximum(t, 1e-12) - 1.0, 0.0)
    return out


class EdgeField:
    """The image's gradient (per mm) at scale_mm, as cubic-spline
    coefficients, sampled along normals by offsets."""

    def __init__(
        self, img: npt.ArrayLike, vox: npt.ArrayLike, scale_mm: float = SCALE_MM
    ) -> None:
        a = np.asarray(img, np.float64)
        self.vox = np.asarray(vox, np.float64)
        sig = scale_mm / self.vox
        self.grads = []
        for k in range(3):
            order = [0, 0, 0]
            order[k] = 1
            gk = ndi.gaussian_filter(a, sig, order=order) / self.vox[k]
            self.grads.append(ndi.spline_filter(gk, 3))

    def offsets(
        self,
        v: F64,
        n: F64,
        window_mm: float = WINDOW_MM,
        step_mm: float = STEP_MM,
        sign: float | None = None,
    ) -> EdgeOffsets:
        """The steepest WM-to-GM point along each normal (v, n: voxel
        coordinates, unit outward normals in voxel coordinates)."""
        vx = self.vox
        nm = n / vx  # normals are covectors
        nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-12)
        m = int(round(window_mm / step_mm))
        s = np.arange(-m, m + 1) * step_mm
        der = np.empty((len(v), len(s)))
        for i in range(0, len(v), _CHUNK):
            sl = slice(i, i + _CHUNK)
            p = v[sl, None, :] + s[None, :, None] * (nm[sl] / vx)[:, None, :]
            q = p.reshape(-1, 3).T
            d = np.zeros(q.shape[1])
            for k in range(3):
                gk = ndi.map_coordinates(self.grads[k], q, order=3, prefilter=False)
                d += gk * np.repeat(nm[sl, k], len(s))
            der[sl] = d.reshape(-1, len(s))
        if sign is None:
            sign = 1.0 if not len(v) or np.median(der[:, m]) >= 0 else -1.0
        der *= sign
        if not len(v):
            z = np.zeros(0)
            return EdgeOffsets(z, z, np.zeros(0, bool), float(sign))
        j = np.argmax(der, 1)
        r = np.arange(len(v))
        at_end = (j == 0) | (j == len(s) - 1)
        jc = np.clip(j, 1, len(s) - 2)
        y0, y1, y2 = der[r, jc - 1], der[r, jc], der[r, jc + 1]
        den = y0 - 2 * y1 + y2
        frac = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
        frac = np.where(at_end, 0.0, np.clip(frac, -0.5, 0.5))
        off = s[j] + frac * step_mm
        return EdgeOffsets(off, der[r, j], at_end, float(sign))


def edge_offsets(
    img: npt.ArrayLike,
    v: F64,
    n: F64,
    vox: npt.ArrayLike,
    window_mm: float = WINDOW_MM,
    scale_mm: float = SCALE_MM,
    step_mm: float = STEP_MM,
    sign: float | None = None,
) -> EdgeOffsets:
    """EdgeField(img, vox, scale_mm).offsets(v, n, window_mm, step_mm, sign)."""
    return EdgeField(img, vox, scale_mm).offsets(v, n, window_mm, step_mm, sign)


# ---------------------------------------------------------------- step 2: the label


@dataclass(frozen=True)
class Peel:
    """The label's inward correction: voxels removed, voxels flagged (no
    edge within the window inside them; kept), voxels blocked (wanted by the
    edge, kept because removing them would change the label's topology), the
    sign used, the rounds."""

    removed: Bool
    flagged: Bool
    blocked: Bool
    sign: float
    rounds: int


@njit
def _sweep(obj: Bool, cand: I64, offs: I64, simple: Bool) -> int:
    """Remove the voxels cand (flat indices into the padded object) that are
    simple, in order, sweeping again until none goes. Returns the number."""
    n = 0
    progress = True
    while progress:
        progress = False
        for i in cand:
            if not obj[i]:
                continue
            m = 0
            for k in range(14):
                if obj[i + offs[k]]:
                    m |= 1 << k
            if simple[m]:
                obj[i] = False
                n += 1
                progress = True
    return n


def simple_peel(wm: Bool, want: npt.ArrayLike) -> tuple[Bool, Bool]:
    """Remove the wanted voxels of wm that are simple points on the Kuhn
    tetrahedralization (baillarger.handles), one at a time, so that no Betti
    number of the label or of its background changes: no handle, cavity or
    component is made or lost. want: bool, or the round in which each voxel
    was wanted (0: not wanted); lower rounds are tried first, then lower
    flat index, sweeping again until no wanted voxel goes. Outside the array
    is background. Returns (removed, blocked): blocked are the wanted voxels
    kept, none of which can go alone."""
    wm = np.asarray(wm, bool)
    key: npt.NDArray[Any] = np.asarray(want)
    w = wm & (key > 0)
    x = np.ascontiguousarray(np.pad(wm, 1))  # ravel must be a view
    sh = x.shape
    offs = (
        KUHN_OFFSETS[:, 0] * sh[1] * sh[2]
        + KUHN_OFFSETS[:, 1] * sh[2]
        + KUHN_OFFSETS[:, 2]
    ).astype(np.int64)
    idx = np.argwhere(w)
    flat: I64 = np.asarray(np.ravel_multi_index(tuple((idx + 1).T), sh), np.int64)
    cand: I64 = flat[np.lexsort((flat, key[tuple(idx.T)]))]
    _sweep(x.ravel(), cand, offs, SIMPLE)
    removed = wm & ~x[1:-1, 1:-1, 1:-1]
    return removed, w & ~removed


def _ridge(wm: Bool, vox: F64) -> tuple[F64, F64]:
    """Each WM voxel's distance (mm) to the nearest background centre, and
    the largest such distance within GUARD_R voxels (the local
    half-thickness, to the background centres)."""
    d: F64 = ndi.distance_transform_edt(wm, sampling=vox)
    ridge: F64 = ndi.maximum_filter(d, size=2 * GUARD_R + 1)
    return d, ridge


def _label_normals(wm: Bool) -> F64:
    """The outward unit normal (voxel coordinates) of the label smoothed at
    NORMAL_SIGMA, per voxel: (*shape, 3)."""
    a = wm.astype(np.float64)
    g = np.stack(
        [
            ndi.gaussian_filter(a, NORMAL_SIGMA, order=[int(i == k) for i in range(3)])
            for k in range(3)
        ],
        -1,
    )
    out: F64 = -g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
    return out


def _wish(
    wm: Bool,
    img: npt.ArrayLike,
    vox: npt.ArrayLike,
    margin_mm: float,
    window_mm: float,
    sign: float | None,
) -> tuple[U8, Bool, float, int]:
    """The edge's wish: the round (1, 2, ...) in which each voxel would go,
    0 for those that stay; the flagged voxels, the sign, the rounds."""
    vx = np.asarray(vox, np.float64)
    wm = np.asarray(wm, bool)
    field = EdgeField(img, vx)
    nrm = _label_normals(wm)
    d, ridge = _ridge(wm, vx)
    may = wm & (d <= 0.5 * ridge)
    cur = wm.copy()
    when = np.zeros(wm.shape, np.uint8)
    flagged = np.zeros_like(wm)
    rounds = math.ceil((window_mm + margin_mm) / float(vx.min())) + 1
    done = 0
    for k in range(1, rounds + 1):
        done = k
        edge = cur & ~ndi.binary_erosion(cur, _FACE6, border_value=1)
        cand = np.argwhere(edge & may & ~flagged)
        if not len(cand):
            break
        e = field.offsets(
            cand.astype(np.float64), nrm[tuple(cand.T)], window_mm, sign=sign
        )
        sign = e.sign
        go = ~e.at_end & (e.offset_mm < margin_mm)
        flagged[tuple(cand[e.inner_end].T)] = True
        if not go.any():
            break
        cur[tuple(cand[go].T)] = False
        when[tuple(cand[go].T)] = k
    return when, flagged, 1.0 if sign is None else float(sign), done


def peel(
    wm: Bool,
    img: npt.ArrayLike,
    vox: npt.ArrayLike,
    margin_mm: float = MARGIN_MM,
    window_mm: float = WINDOW_MM,
    sign: float | None = None,
) -> Peel:
    """Step 2 on one array: img is the edge image (tissue_contrast)."""
    wm = np.asarray(wm, bool)
    when, flagged, sg, rounds = _wish(wm, img, vox, margin_mm, window_mm, sign)
    removed, blocked = simple_peel(wm, when)
    return Peel(removed, flagged, blocked, sg, rounds)


def peel_blocks(
    wm: Bool,
    echo: npt.ArrayLike,
    vox: npt.ArrayLike,
    sign: float,
    margin_mm: float = MARGIN_MM,
    window_mm: float = WINDOW_MM,
    block: int = BLOCK,
    halo: int = HALO,
    log: Callable[[str], None] | None = None,
) -> Peel:
    """Step 2 on a large grid: peel(wm, tissue_contrast(echo, wm), ...).
    The edge's wish is found in blocks of block voxels with a halo (blocks
    without WM boundary are skipped), and simple_peel then runs once on the
    whole grid, so that the topology is kept across the blocks' faces too.
    sign must be given (contrast_sign), so every block uses the same. log:
    called with a line per block done."""
    wm = np.asarray(wm, bool)
    ech = np.asarray(echo)
    shape = wm.shape
    when = np.zeros(shape, np.uint8)
    flagged = np.zeros(shape, bool)
    rounds = 0
    for i in range(0, shape[0], block):
        for j in range(0, shape[1], block):
            for k in range(0, shape[2], block):
                core = tuple(
                    slice(a, min(a + block, n))
                    for a, n in zip((i, j, k), shape, strict=True)
                )
                c = wm[core]
                if c.all() or not c.any():
                    continue
                ext = tuple(
                    slice(max(s.start - halo, 0), min(s.stop + halo, n))
                    for s, n in zip(core, shape, strict=True)
                )
                w = wm[ext]
                e = ech[ext]
                wh, fl, _, r = _wish(
                    w, tissue_contrast(e, w), vox, margin_mm, window_mm, sign
                )
                inner = tuple(
                    slice(s.start - x.start, s.stop - x.start)
                    for s, x in zip(core, ext, strict=True)
                )
                when[core] = wh[inner]
                flagged[core] = fl[inner]
                rounds = max(rounds, r)
                if log is not None:
                    log(
                        f"block {i} {j} {k}: wanted {int((wh[inner] > 0).sum())}"
                        f", flagged {int(fl[inner].sum())}, rounds {r}"
                    )
    removed, blocked = simple_peel(wm, when)
    if log is not None:
        log(
            f"simple points: removed {int(removed.sum())}, blocked {int(blocked.sum())}"
        )
    return Peel(removed, flagged, blocked, float(sign), rounds)


def contrast_sign(echo: npt.ArrayLike, wm: Bool, cortex: Bool) -> float:
    """+1 if the median of echo over the cortex is above that over the WM
    (the image rises from WM to GM), else -1."""
    e = np.asarray(echo)
    return 1.0 if np.median(e[cortex]) >= np.median(e[wm]) else -1.0


@dataclass(frozen=True)
class EdgePeel:
    """Step 2 on the full grid: voxels removed from the aseg's WM label,
    voxels flagged and voxels blocked (full grid, 0-based; Peel), the sign
    and the margin used."""

    removed: I32
    flagged: I32
    blocked: I32
    sign: float
    margin_mm: float

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            removed=self.removed,
            flagged=self.flagged,
            blocked=self.blocked,
            sign=self.sign,
            margin_mm=self.margin_mm,
        )

    @classmethod
    def load(cls, path: Path) -> "EdgePeel":
        z = np.load(path)
        return cls(
            z["removed"].astype(np.int32),
            z["flagged"].astype(np.int32),
            z["blocked"].astype(np.int32),
            float(z["sign"]),
            float(z["margin_mm"]),
        )


def edge_peel(
    directory: Path | None = None,
    margin_mm: float = MARGIN_MM,
    block: int = BLOCK,
    log: Callable[[str], None] | None = None,
) -> EdgePeel:
    """Step 2 on the aseg's WM label and echo 2, on the full grid."""
    a = nib.load(aseg_path(directory))
    e = nib.load(echo_path(2, directory))
    if not isinstance(a, nib.nifti1.Nifti1Image) or not isinstance(
        e, nib.nifti1.Nifti1Image
    ):
        raise ValueError("the aseg and echo 2 must be NIfTI-1 images")
    lab = np.asarray(a.dataobj)
    wm = lab == ASEG_WM
    echo = np.asarray(e.dataobj)
    sign = contrast_sign(echo, wm, lab == ASEG_CORTEX)
    del lab
    vox = np.array(a.header.get_zooms()[:3], np.float64)
    p = peel_blocks(wm, echo, vox, sign, margin_mm, block=block, log=log)
    return EdgePeel(
        np.argwhere(p.removed).astype(np.int32),
        np.argwhere(p.flagged).astype(np.int32),
        np.argwhere(p.blocked).astype(np.int32),
        sign,
        margin_mm,
    )


# ---------------------------------------------------------------- step 4: the mesh


@dataclass(frozen=True)
class Refined:
    """The refined mesh: vertices, their normals, each vertex's move along
    its old normal (mm, <= 0), the flagged vertices (no edge within the
    window inside them; not moved), the vertices a cap stopped short of the
    margin, and the offsets measured before."""

    v: F64
    n: F64
    move_mm: F64
    flagged: Bool
    short: Bool
    offsets: EdgeOffsets


def _mesh_twice(v_mm: F64, f: I64, x: F64, w: F64, sigma_mm: float) -> F64:
    """Gaussian smoothing of x (n, k) along the mesh over sigma_mm, weighted
    by w (umbrella steps), with its shrinkage removed by twicing
    (2 G x - G G x, as baillarger.mesh_smoothing.twice)."""
    A, deg = umbrella(v_mm, f)
    steps = gauss_steps(v_mm, f, sigma_mm)

    def G(y: F64) -> F64:
        yw = np.concatenate([y * w[:, None], w[:, None]], 1)
        for _ in range(steps):
            yw = yw + 0.5 * ((A @ yw) / deg[:, None] - yw)
        out: F64 = yw[:, :-1] / np.maximum(yw[:, -1:], 1e-12)
        return out

    g = G(x)
    out: F64 = 2 * g - G(g)
    return out


def refine(
    v: F64,
    f: I64,
    n: F64,
    field: EdgeField,
    vox: npt.ArrayLike,
    wm: Bool,
    margin_mm: float = MARGIN_MM,
    window_mm: float = WINDOW_MM,
    sigma_mm: float = REFINE_SIGMA_MM,
    sign: float | None = None,
) -> Refined:
    """Step 4: move each vertex of the white mesh of wm (v, f, n: voxel
    coordinates, outward unit normals) inward toward the edge minus
    margin_mm, within the caps (module docstring)."""
    vx = np.asarray(vox, np.float64)
    wm = np.asarray(wm, bool)
    e = field.offsets(v, n, window_mm, sign=sign)
    valid = ~e.at_end
    nm = n / vx
    nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-12)
    # smooth the edge's position, not the move: the smoothed offset plus the
    # smoothed vertex position's offset from the vertex, along its normal, so
    # that a vertex's own jitter is not smoothed into its move
    vmm = v * vx
    w = valid.astype(np.float64)
    o_s = _mesh_twice(vmm, f, e.offset_mm[:, None], w, sigma_mm)[:, 0]
    v_s = _mesh_twice(vmm, f, vmm, np.ones(len(v)), sigma_mm)
    t = o_s + np.einsum("ij,ij->i", v_s - vmm, nm) - margin_mm
    t = np.minimum(t, e.offset_mm - margin_mm)
    want = np.where(valid, np.minimum(t, 0.0), 0.0)
    # the caps: just under half a voxel, and half of (half-thickness less
    # half a voxel), the half-thickness read at the WM voxels round v - n/2
    half = 0.5 * float(vx.min())
    _, ridge = _ridge(wm, vx)
    ridge = np.where(wm, ridge, 0.0)
    base = np.floor(v - 0.5 * n).astype(np.int64)
    top = np.array(ridge.shape) - 1
    h = np.zeros(len(v))
    for corner in np.ndindex(2, 2, 2):
        idx = np.clip(base + np.array(corner), 0, top)
        h = np.maximum(h, ridge[tuple(idx.T)])
    cap = np.minimum(half * (1 - 1e-3), np.maximum(0.5 * (h - half), 0.0))
    t = np.maximum(want, -cap)
    short = want < t - 1e-12
    v1 = v + t[:, None] * nm / vx
    return Refined(v1, smooth_normals(v1, f), t, e.inner_end, short, e)


# ---------------------------------------------------------------- a box's report

#: mean curvature for the report (mm); bins of it (1/mm) and of coherence
CURV_MM = 0.48
H_BINS = (-np.inf, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, np.inf)
COH_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)
#: the cortex's coherence is read this far outside the mesh (voxels)
COH_AT = 1.5


def _pct(x: npt.ArrayLike) -> dict[str, Any] | None:
    a = np.asarray(x)
    if not a.size:
        return None
    p = np.percentile(a, [5, 25, 50, 75, 95]).round(4).tolist()
    return dict(n=int(a.size), p5_25_50_75_95=p)


@dataclass
class BoxEdges:
    """A box's white mesh against the image's edge: the report (json), the
    per-vertex arrays (npz) and the tissue they were made from."""

    report: dict[str, Any]
    arrays: dict[str, npt.NDArray[Any]]
    tissue: Any


def box_edges(
    lo: tuple[int, int, int],
    hi: tuple[int, int, int],
    changes: Any,
    directory: Path | None = None,
    name: str = "box",
) -> BoxEdges:
    """The box [lo, hi) with the corrected WM label (changes: a
    baillarger.white.WMChanges): its white mesh (column_traces.white_mesh),
    the offsets of echoes 2 and 3 at every vertex (step 1), mapped by mean
    curvature and by the cortex's coherence just outside; then step 4 (refine,
    echo 2) and the offsets measured again on the refined mesh. Vertices
    within column_traces.FACE_MM of a box face are left out of the report."""
    from baillarger import column_traces as CT
    from baillarger.frame import mesh_curvatures
    from baillarger.mesh_intersect import intersecting_faces

    t = CT.load_tissue(lo, hi, name=name, directory=directory, changes=changes)
    vox = t.vox
    half = 0.5 * float(vox.min())
    v, f, n = CT.white_mesh(t.wm)
    roots = CT.root_vertices(v, n)
    k1, k2 = mesh_curvatures(v * vox, f, n, CURV_MM)
    H = 0.5 * (k1 + k2)
    _, coh = CT.data_normal(CT.base_tensor(t), t.ctx)
    coh_v = ndi.map_coordinates(coh, (v + COH_AT * n).T, order=1, mode="nearest")
    vi = np.clip(np.round(v).astype(np.int64), 0, np.array(t.wm.shape) - 1)
    keep = t.dface[tuple(vi.T)] >= CT.FACE_MM
    rep: dict[str, Any] = dict(
        lo=list(lo),
        hi=list(hi),
        vox=vox.tolist(),
        half_voxel_mm=half,
        vertices=len(v),
        roots=len(roots),
    )
    arrays: dict[str, npt.NDArray[Any]] = dict(
        lo=np.array(lo),
        vox=vox,
        v=v,
        f=f,
        n=n,
        roots=roots,
        H=H,
        coh=coh_v,
        keep=keep,
    )
    fields, E = {}, {}
    for k, nm in ((1, "echo2"), (2, "echo3")):
        fields[nm] = EdgeField(tissue_contrast(t.patch.echoes[..., k], t.wm), vox)
        e = E[nm] = fields[nm].offsets(v, n)
        ok = keep & ~e.at_end
        o = e.offset_mm
        rep[nm] = dict(
            sign=e.sign,
            at_end_frac=float(e.at_end[keep].mean()),
            inner_end_frac=float(e.inner_end[keep].mean()),
            all=_pct(o[ok]),
            roots=_pct(o[roots][ok[roots]]),
            outside_gt_half_frac=float((o[ok] > half).mean()),
            inside_gt_half_frac=float((o[ok] < -half).mean()),
            by_H={
                f"{a:g}..{b:g}": _pct(o[ok & (H >= a) & (H < b)])
                for a, b in zip(H_BINS[:-1], H_BINS[1:], strict=True)
            },
            by_coherence={
                f"{a:g}..{b:g}": _pct(o[ok & (coh_v >= a) & (coh_v < b)])
                for a, b in zip(COH_BINS[:-1], COH_BINS[1:], strict=True)
            },
        )
        for key in ("offset_mm", "slope", "at_end"):
            arrays[f"{nm}_{key}"] = getattr(e, key)
    both = keep & ~E["echo2"].at_end & ~E["echo3"].at_end
    o2, o3 = E["echo2"].offset_mm[both], E["echo3"].offset_mm[both]
    rep["echo3_minus_echo2"] = _pct(o3 - o2)
    rep["echo2_echo3_corr"] = (
        float(np.corrcoef(o2, o3)[0, 1]) if both.sum() > 1 else None
    )
    s2 = E["echo2"].sign
    R = refine(v, f, n, fields["echo2"], vox, t.wm, sign=s2)
    after = fields["echo2"].offsets(R.v, R.n, sign=s2)
    moved = R.move_mm < 0
    ok = keep & ~after.at_end
    oa = after.offset_mm
    rep["refined"] = dict(
        margin_mm=MARGIN_MM,
        moved_frac=float(moved[keep].mean()),
        move_mm=_pct(-R.move_mm[keep & moved]),
        flagged_frac=float(R.flagged[keep].mean()),
        short_frac=float(R.short[keep].mean()),
        after_short_of_margin_frac=float((oa[ok] < MARGIN_MM - 0.02).mean()),
        after=_pct(oa[ok]),
        after_inside_frac=float((oa[ok] < 0).mean()),
        after_inside_gt_0p02mm_frac=float((oa[ok] < -0.02).mean()),
        after_inside_gt_half_frac=float((oa[ok] < -half).mean()),
        intersecting_faces_before=int(intersecting_faces(v, f)[1]),
        intersecting_faces_after=int(intersecting_faces(R.v, f)[1]),
    )
    arrays.update(
        refined_v=R.v,
        refined_n=R.n,
        refined_move_mm=R.move_mm,
        refined_flagged=R.flagged,
        refined_short=R.short,
        refined_after_offset_mm=oa,
        refined_after_at_end=after.at_end,
    )
    return BoxEdges(rep, arrays, t)


def write_box(b: BoxEdges, out: Path, stem: str) -> None:
    """<stem>.json (the report), <stem>.npz (per vertex), <stem>.vtk (the
    mesh with the offsets, curvature and coherence; ITK-SNAP world
    coordinates), <stem>_refined.vtk (the refined mesh with its moves, the
    offsets after and the flags), <stem>_overlay.nii.gz (box grid; at each
    vertex's voxel, echo 2: 1 edge more than half a voxel outside, 2 more
    than half a voxel inside, 3 no edge in the window), <stem>_echo2.nii.gz
    and <stem>_wm.nii.gz (echo 2 and the corrected label, box grid)."""
    from baillarger.mesh import write_vtk

    out.mkdir(parents=True, exist_ok=True)
    a, t = b.arrays, b.tissue
    aff = t.patch.affine
    (out / f"{stem}.json").write_text(json.dumps(b.report, indent=1))
    np.savez_compressed(out / f"{stem}.npz", allow_pickle=False, **a)
    o2, e2 = a["echo2_offset_mm"], a["echo2_at_end"]
    write_vtk(
        str(out / f"{stem}.vtk"),
        aff,
        a["v"],
        a["f"],
        {
            "offset_echo2_mm": np.where(e2, np.nan, o2),
            "offset_echo3_mm": np.where(
                a["echo3_at_end"], np.nan, a["echo3_offset_mm"]
            ),
            "H_per_mm": a["H"],
            "coherence": a["coh"],
        },
    )
    write_vtk(
        str(out / f"{stem}_refined.vtk"),
        aff,
        a["refined_v"],
        a["f"],
        {
            "move_mm": a["refined_move_mm"],
            "offset_after_mm": np.where(
                a["refined_after_at_end"], np.nan, a["refined_after_offset_mm"]
            ),
            "flagged": a["refined_flagged"].astype(float),
        },
    )
    half = b.report["half_voxel_mm"]
    vi = np.clip(np.round(a["v"]).astype(np.int64), 0, np.array(t.wm.shape) - 1)
    ov = np.zeros(t.wm.shape, np.uint8)
    for code, sel in ((1, o2 > half), (2, o2 < -half), (3, e2)):
        ov[tuple(vi[sel].T)] = code
    nib.save(nib.nifti1.Nifti1Image(ov, aff), out / f"{stem}_overlay.nii.gz")
    echo2 = t.patch.echoes[..., 1].astype(np.float32)
    nib.save(nib.nifti1.Nifti1Image(echo2, aff), out / f"{stem}_echo2.nii.gz")
    wm = t.wm.astype(np.uint8)
    nib.save(nib.nifti1.Nifti1Image(wm, aff), out / f"{stem}_wm.nii.gz")


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("peel", help="step 2 on the full grid")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--margin-mm", type=float, default=MARGIN_MM)
    b = sub.add_parser("box", help="a box's mesh against the edge, and step 4")
    b.add_argument("--changes", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--name", required=True)
    b.add_argument("--lo", required=True, type=int, nargs=3)
    b.add_argument("--hi", required=True, type=int, nargs=3)
    a = ap.parse_args(argv)
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "box":
        from baillarger.white import WMChanges

        lo = (a.lo[0], a.lo[1], a.lo[2])
        hi = (a.hi[0], a.hi[1], a.hi[2])
        be = box_edges(lo, hi, WMChanges.load(a.changes), name=a.name)
        write_box(be, out, f"wm_edge_{a.name}")
        r = be.report
        print(json.dumps(dict(echo2=r["echo2"]["all"], refined=r["refined"])))
        return
    p = edge_peel(margin_mm=a.margin_mm, log=lambda m: print(m, flush=True))
    p.save(out / "wm_edge_peel.npz")
    img = nib.load(aseg_path())
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError("the aseg must be a NIfTI-1 image")
    vmm = float(np.prod(img.header.get_zooms()[:3]))
    summary: dict[str, Any] = dict(
        sign=p.sign,
        margin_mm=p.margin_mm,
        removed=len(p.removed),
        removed_mm3=round(len(p.removed) * vmm, 3),
        flagged=len(p.flagged),
        blocked=len(p.blocked),
    )
    lab = np.asarray(img.dataobj) == ASEG_WM
    nz = [
        np.flatnonzero(lab.any(axis=tuple(j for j in range(3) if j != i)))
        for i in range(3)
    ]
    box = tuple(slice(max(int(t[0]) - 2, 0), int(t[-1]) + 3) for t in nz)
    summary["betti_aseg"] = list(kuhn_betti(lab[box]))
    lab[tuple(p.removed.T)] = False
    summary["betti_peeled"] = list(kuhn_betti(lab[box]))
    del lab
    (out / "wm_edge_peel.json").write_text(json.dumps(summary, indent=1))
    ov = np.zeros(img.shape[:3], np.uint8)
    ov[tuple(p.removed.T)] = 1
    ov[tuple(p.flagged.T)] = 2
    ov[tuple(p.blocked.T)] = 3
    nib.save(
        nib.nifti1.Nifti1Image(ov, img.affine), out / "wm_edge_peel_overlay.nii.gz"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
