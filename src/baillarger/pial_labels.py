"""The pial-side label refinement: voxels that the aseg labels cortex but the
image shows are not cortex (thin fluid channels in sulci the segmentation
fused, superficial vessels), proposed as fluid, ranked by what they do to
the column traces, reviewed by the user, and saved as a change file that
baillarger.column_traces.load_tissue applies.

Plan and results: reports/Crossing paths and contacts from laminar profiles.md
(*Pial-side label refinement*). Positions are voxel indices of the box; vox
the voxel size (mm). Everything is 3D.

1. Darkness (darkness). The log of echoes 1, 2 and 3 (ECHOES) minus their
   trend over the
   cortex (normalized convolution at column_traces.TREND voxels, as in
   column_traces.tissue), with the trend estimated again without the voxels
   the first pass found dark, so that a channel does not pull its own
   reference down. Per echo, the noise of one voxel from the structure
   tensor (echo_noise): image noise adds sigma^2 g to every eigenvalue, g the
   gain of the tensor's derivative filter (gradient_gain); the layers add to
   the largest only. The median smallest eigenvalue over the cortex (away
   from its edges), divided
   by g and by the median smallest eigenvalue of a Wishart matrix with
   n_eff degrees of freedom (wishart_min_factor: the sample tensor's
   smallest eigenvalue is biased low), is sigma^2. A voxel is dark when
   - it is significantly darker than its local cortex: the noise-weighted
     mean of the echoes' log contrast, z = dbar / sigma_dbar, is below
     -k, k the one-sided normal quantile for alpha (false alarms per voxel);
   - it is more fluid than cortex: its fluid fraction, from linear
     intensities between the local cortex level and the box's fluid level
     (labelled fluid at least FLUID_DEPTH voxels from anything else), is
     above one half (noise-weighted over the echoes).
   Echo 1 has the most signal and the largest contrast of tissue against
   the fomblin; echoes 2 and 3 add the T2* contrast of blood in vessels.
   The fitted field T of step 3 is made, as in the column traces, from
   echoes 2 and 3.
2. Components (26-connected) of the dark cortex voxels, at least n_min
   voxels: the smallest size whose expected count in noise alone,
   N p (e (26 - 1) p)^(n - 1) (N cortex voxels, p the null probability of a
   dark voxel; the bound on connected sets of n vertices containing a given
   vertex in a graph of degree 26), is below one. A component is split into
   review cells (review_cells): geodesic Voronoi cells of the component's
   voxel graph, centres CT.PATCH_MM apart (the root patches' spacing). The
   aseg's cortex runs one to three voxels into the fluid along much of the
   pia, so a fused channel and that whole rim are often one component; the
   cells are what the review accepts or rejects, and steps 3-6 are made per
   cell (step 3 walks out of the whole component).
3. Position (pass_through), decided per component (all its cells share
   it): a component is a proposal ("outer") when it touches fluid or other
   labels (26-adjacent), or when the fitted field T does not pass through
   it within one bank. For each voxel, step along
   grad T until leaving the component (at most STEP_MAX voxels) on both
   sides; T passes through when both ends are cortex and T rises
   significantly into the voxel from below and on from it above (by more
   than k times step_noise per voxel step, as in streamlines.crest_kinks).
   A component that T passes through at more than half its voxels lies
   inside one bank: it is not proposed (auto "inside"). This is the stria
   of Gennari's case: a dark band in the middle of the cortex, which
   darkness alone could take for fluid. A fused channel lies where two
   banks' fields meet (T's crest) and a superficial vessel at the top of the
   cortex: T does not pass through either. A hidden channel off T's crest
   that does not reach labelled fluid looks like a dark band inside one
   bank, and is left to the review's brush.
4. Shape (shape_kind): the Westin measures of the voxel coordinates'
   covariance (mm): linear (tube: a vessel), planar (sheet: a channel) or
   spherical (blob), whichever is largest. A descriptor for review, not a
   test.
5. Impact (propose): crest voxels of T (streamlines.crest_kinks) within one
   voxel of the component, and, given a column volume, the traces that pass
   through it. Ranked by columns, then crest voxels, then size. Crest
   voxels with no dark component nearby are where T's banks meet without a
   visible channel.
6. Topology (topology_change): the Betti numbers (Kuhn adjacency,
   baillarger.handles) of the tissue (every label but fluid) in a box round
   each review cell, before and after its whole component is relabelled:
   a cavity (a channel with no mouth in the box), a tunnel (a
   vessel under a bridge of mislabelled cortex) or a split. The change file
   records the box's Betti numbers before and after the changes it applies.
7. Status, per review cell (auto_accept): a cell of a component inside a
   bank is rejected. A cell of a component that touches fluid is accepted
   at once when its relabel, on top of the cells accepted so far, keeps the
   tissue's Betti numbers in a box round it; the cells are taken in passes
   until no more can be, so the rim where the aseg's cortex runs one to
   three voxels into the fluid opens from the fluid inward. The rest (cells
   of hidden channels that reach no fluid, and cells whose relabel would
   open a cavity, a tunnel or a split, as a channel between two banks does)
   are proposed: the review queue, the cells worth a look first. The
   review can overturn any of them.

The change file (LabelChanges): full-grid voxels, their new rim label
(baillarger.data: 1 outside for fluid, 3 cortex for a brush that gives
cortex back), the component of each (-1 a brush edit), each component's
status (proposed, accepted, rejected) and its descriptors. The default is
to trust the detector: apply applies its best guess, every cell that is not
rejected (accepted and proposed) and the brush edits, to a box's rim
labels, never to a WM voxel; with reviewed_only, only the accepted cells
and the brush edits. Corrections go into the change file (the review:
with_status, with_brush), not into the labels.

Command line:

    python -m baillarger.pial_labels propose --out DIR --name NAME --lo X Y Z
        --hi X Y Z [--wm-changes FILE] [--columns column_volume_<name>.npz]

writes, in DIR: pial_labels_<NAME>.npz (the change file, every review cell
accepted, proposed or rejected by step 7); .json (summary and the ranked
cells); _labels.nii.gz (the box's rim labels with the WM changes and the
best guess applied: 0 other, 1 outside, 2 WM, 3 cortex); _overlay.nii.gz
(box grid; overlay's codes).
"""

import argparse
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix

from baillarger import column_traces as CT
from baillarger import mesh_graph as G
from baillarger import streamlines as S
from baillarger.data import RIM_CORTEX, RIM_OUTSIDE, RIM_WM, Box
from baillarger.handles import kuhn_betti

type Bool = npt.NDArray[np.bool_]
type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Betti = tuple[int, int, int]

#: The echoes the darkness test reads (0-based: echoes 1, 2 and 3).
ECHOES = (0, 1, 2)
#: False alarms per voxel of the darkness test and the pass-through test.
ALPHA = 1e-3
#: Steps (voxels) along grad T to leave a component.
STEP_MAX = 8
#: Depth (voxels) of the labelled fluid that gives the fluid level.
FLUID_DEPTH = 3.0
#: Component statuses.
PROPOSED, ACCEPTED, REJECTED = 0, 1, 2
STATUS = ("proposed", "accepted", "rejected")
KINDS = ("tube", "sheet", "blob")
#: Overlay codes (overlay): 1 + a component's status, brush edits, and the
#: dark voxels of no component.
OV_BRUSH_FLUID, OV_BRUSH_CORTEX, OV_DARK = 4, 5, 6
_C26 = np.ones((3, 3, 3), bool)


# ---------------------------------------------------------------- noise


def gradient_gain(sd: float = S.SD) -> float:
    """The variance of one component of the gradient that
    streamlines.structure_tensor takes (central differences of the image
    smoothed by sd voxels), per unit variance of white noise: the sum of the
    squared filter taps. Exact for the discrete filters."""
    m = int(np.ceil(4 * sd)) + 2
    d = np.zeros((2 * m + 1,) * 3)
    d[m, m, m] = 1.0
    r = np.gradient(ndi.gaussian_filter(d, sd), axis=0)
    return float(np.sum(r**2))


def wishart_min_factor(n: float, draws: int = 400_000, seed: int = 0) -> float:
    """The median smallest eigenvalue of W / n, W a 3 x 3 Wishart matrix with
    identity scale and n degrees of freedom (Bartlett's decomposition;
    deterministic: a fixed seed). For the structure tensor of white noise,
    n = streamlines.n_eff: the ratio of its median smallest eigenvalue to
    the noise's gradient power."""
    rng = np.random.default_rng(seed)
    A = np.zeros((draws, 3, 3))
    for i in range(3):
        A[:, i, i] = np.sqrt(rng.chisquare(n - i, draws))
        for j in range(i):
            A[:, i, j] = rng.standard_normal(draws)
    W = A @ np.swapaxes(A, 1, 2) / n
    return float(np.median(np.linalg.eigvalsh(W)[:, 0]))


def echo_noise(det: F64, ctx: Bool, sd: float = S.SD, si: float = S.SI) -> float:
    """The noise (standard deviation) of one voxel of a trend-removed log
    echo, from its structure tensor over the cortex (module docstring, step
    1). For white noise it is within a few per cent, high: the Wishart model
    of the tensor's correlated samples is approximate (1-2 %). The median is
    over the cortex voxels whose window lies within the cortex (Gaussian
    weight of the cortex over 0.95 at hypot(sd, si)), all of them if fewer
    than 1000: at the cortex's edges the jump to the masked-out voxels adds
    gradient power. For spatially correlated noise it is the white noise
    with the same gradient power."""
    lam = S.eigenvalues(S.structure_tensor(det, ctx.astype(float), sd, si))
    # where the tensor's window (sd and si) lies within the cortex: at its
    # edges the jump to the masked-out voxels adds gradient power
    inner = ctx & (ndi.gaussian_filter(ctx.astype(float), math.hypot(sd, si)) > 0.95)
    lmin = float(np.median(lam[..., 0][inner if inner.sum() >= 1000 else ctx]))
    q = wishart_min_factor(S.n_eff(sd, si))
    return math.sqrt(lmin / (gradient_gain(sd) * q))


# ---------------------------------------------------------------- darkness


@dataclass
class Darkness:
    """Step 1 on a box."""

    #: (len(ECHOES), ...) log contrast of each echo against the local cortex
    d: F64
    #: noise of one voxel, per echo
    sigma: tuple[float, ...]
    #: the labelled fluid's level (linear), per echo
    fluid: tuple[float, ...]
    #: noise-weighted contrast in noise units (negative: darker), 0 outside the cortex
    z: F64
    #: fluid fraction (noise-weighted over the echoes), 0 outside the cortex
    frac: F64
    #: dark cortex voxels: z < -k and frac > 1/2
    dark: Bool
    k: float
    #: null probability of a dark voxel, and the smallest component kept
    p_null: float
    n_min: int
    #: the cortex's MAD of each echo's contrast (noise and layers), for comparison
    mad: tuple[float, ...]


def _trend_contrast(t: CT.Tissue, mask: Bool) -> tuple[list[F64], list[F64]]:
    """Per echo of ECHOES: log intensity minus its trend over mask (zero
    outside the cortex), and the trend."""
    d, tr = [], []
    m = mask.astype(float)
    for i in ECHOES:
        L = np.log(np.maximum(t.patch.echoes[..., i].astype(float), 1.0))
        trend = S.masked_smooth(L, m, CT.TREND)
        d.append((L - trend) * t.ctx)
        tr.append(trend)
    return d, tr


def fluid_level(t: CT.Tissue, i: int) -> float:
    """Median of echo i (0-based) over the labelled fluid at least
    FLUID_DEPTH voxels from any other label (all labelled fluid if none is
    that deep; 0 without fluid)."""
    fl = t.labels == 1
    if not fl.any():
        return 0.0
    deep = fl & (ndi.distance_transform_edt(fl) >= FLUID_DEPTH)
    sel = deep if deep.any() else fl
    return float(np.median(t.patch.echoes[..., i][sel]))


def min_component(n_cortex: int, p: float, degree: int = 26) -> int:
    """The smallest n with n_cortex p (e (degree - 1) p)^(n - 1) < 1 (module
    docstring, step 2)."""
    n = 1
    c = math.e * (degree - 1) * p
    while n_cortex * p * c ** (n - 1) >= 1.0 and n < 1000:
        n += 1
    return n


def darkness(t: CT.Tissue, alpha: float = ALPHA) -> Darkness:
    """Step 1 on a box (module docstring)."""
    k = NormalDist().inv_cdf(1.0 - alpha)
    ctx = t.ctx
    fl = tuple(fluid_level(t, i) for i in ECHOES)
    ne = len(ECHOES)

    def classify(
        d: list[F64], tr: list[F64], sig: tuple[float, ...]
    ) -> tuple[F64, F64, Bool]:
        w = [1.0 / s**2 for s in sig]
        dbar = sum(w[i] * d[i] for i in range(ne)) / sum(w)
        z: F64 = np.where(ctx, dbar * math.sqrt(sum(w)), 0.0)
        fr, wf = np.zeros(ctx.shape), np.zeros(ctx.shape)
        for i in range(ne):
            r = fl[i] / np.exp(tr[i])
            fi = (1.0 - np.exp(d[i])) / np.maximum(1.0 - r, 1e-6)
            wi = ((1.0 - r) / sig[i]) ** 2
            fr += wi * fi
            wf += wi
        frac: F64 = np.where(ctx, fr / wf, 0.0)
        dark: Bool = ctx & (z < -k) & (frac > 0.5)
        return z, frac, dark

    d0, tr0 = _trend_contrast(t, ctx)
    sig = tuple(echo_noise(x, ctx) for x in d0)
    _, _, dark0 = classify(d0, tr0, sig)
    d, tr = _trend_contrast(t, ctx & ~dark0)
    z, frac, dark = classify(d, tr, sig)
    # null probability: z below -k and below the contrast of a half-fluid
    # voxel at the cortex's median level
    w = [1.0 / s**2 for s in sig]
    r = [fl[i] / float(np.exp(np.median(tr[i][ctx]))) for i in range(ne)]
    dmid = [math.log((1.0 + r[i]) / 2.0) for i in range(ne)]
    zmid = sum(w[i] * dmid[i] for i in range(ne)) / math.sqrt(sum(w))
    p = NormalDist().cdf(min(-k, zmid))
    mad = [float(1.4826 * np.median(np.abs(x[ctx] - np.median(x[ctx])))) for x in d0]
    return Darkness(
        np.stack(d),
        sig,
        fl,
        z,
        frac,
        dark,
        k,
        p,
        min_component(int(ctx.sum()), p),
        tuple(mad),
    )


# ---------------------------------------------------------------- components


def shape_kind(pts: npt.ArrayLike, vox: npt.ArrayLike) -> tuple[str, F64]:
    """The Westin measures (linear, planar, spherical) of the covariance of
    points (voxel coordinates, scaled to mm), and the largest one's kind:
    tube, sheet or blob."""
    p = np.asarray(pts, np.float64) * np.asarray(vox, np.float64)
    blob = np.array([0.0, 0.0, 1.0])
    if len(p) < 2:
        return "blob", blob
    w = np.maximum(np.sort(np.linalg.eigvalsh(np.cov(p.T)))[::-1], 0.0)
    if w[0] <= 0:
        return "blob", blob
    m: F64 = np.array([w[0] - w[1], w[1] - w[2], w[2]]) / w[0]
    return KINDS[int(np.argmax(m))], m


def pass_through(
    comp: Bool,
    pts: I64,
    T: F64,
    noise: F64,
    labels: npt.NDArray[np.integer[Any]],
    vox: npt.ArrayLike,
    k: float,
) -> Bool:
    """Per point of a component (comp, its mask; pts, its voxels): whether T
    passes through it within one bank (module docstring, step 3)."""
    v = np.asarray(vox, np.float64)
    h = float(np.mean(v))
    g = np.stack([gi[tuple(pts.T)] for gi in S.gradient(T, v)], -1)
    gn = np.linalg.norm(g, axis=1)
    step = g / (gn[:, None] + 1e-12) / v * h  # one mean voxel along grad T
    shape = np.array(T.shape)
    t0 = T[tuple(pts.T)]
    s0 = noise[tuple(pts.T)]
    res: Bool = gn > 0
    for sgn in (1.0, -1.0):
        end = np.zeros((len(pts), 3), np.int64)
        nst = np.zeros(len(pts), np.int64)
        left = np.zeros(len(pts), bool)
        inside_box = np.ones(len(pts), bool)
        for j in range(1, STEP_MAX + 1):
            q = np.round(pts + sgn * j * step).astype(np.int64)
            inb = np.all((q >= 0) & (q < shape), 1)
            qc = np.clip(q, 0, shape - 1)
            new = (~inb | ~comp[tuple(qc.T)]) & ~left
            end[new] = qc[new]
            nst[new] = j
            inside_box[new] = inb[new]
            left |= new
        good = left & inside_box
        rise = (T[tuple(end.T)] - t0) * sgn
        lab = labels[tuple(end.T)]
        res &= good & (lab == 0) & (rise > k * s0 * nst)
    return res


def review_cells(pts: I64, vox: npt.ArrayLike, spacing: float) -> I64:
    """Per point of a connected voxel set (pts), its review cell: geodesic
    Voronoi cells of the 26-neighbour graph (edge lengths in mm), centres at
    least spacing (mm) apart (mesh_graph.geodesic_patches)."""
    if len(pts) == 1:
        return np.zeros(1, np.int64)
    v = np.asarray(vox, np.float64)
    lo = pts.min(0)
    shape = tuple(int(a) for a in pts.max(0) - lo + 1)
    idx = -np.ones(shape, np.int64)
    idx[tuple((pts - lo).T)] = np.arange(len(pts))
    rows, cols, lens = [], [], []
    for off in CT._N13:
        a = tuple(slice(max(0, -o), shape[i] - max(0, o)) for i, o in enumerate(off))
        b = tuple(slice(max(0, o), shape[i] - max(0, -o)) for i, o in enumerate(off))
        ia, ib = idx[a].ravel(), idx[b].ravel()
        ok = (ia >= 0) & (ib >= 0)
        rows.append(ia[ok])
        cols.append(ib[ok])
        lens.append(np.full(ok.sum(), float(np.linalg.norm(np.multiply(off, v)))))
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.concatenate(lens), (r, c)), shape=(len(pts),) * 2).tocsr()
    _, cell = G.geodesic_patches(g.maximum(g.T).tocsr(), spacing)
    return cell


def topology_change(
    tissue: Bool, pts: I64, margin: int = 2, also: Bool | None = None
) -> tuple[Betti, Betti]:
    """The Betti numbers (Kuhn adjacency) of the tissue in the bounding box
    of pts with a margin, before and after pts (and the voxels of also in
    that box: the rest of their component) leave it."""
    lo = np.maximum(pts.min(0) - margin, 0)
    hi = np.minimum(pts.max(0) + margin + 1, tissue.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    sub = tissue[box].copy()
    before = kuhn_betti(sub)
    sub[tuple((pts - lo).T)] = False
    if also is not None:
        sub &= ~also[box]
    return before, kuhn_betti(sub)


def traces_through(lab: I64, pos: npt.ArrayLike, ns: npt.ArrayLike) -> I64:
    """Per component label (index 1..max; 0 unused), the number of traces
    with a sample in it. pos (n, nmax, 3): box voxel coordinates, nan past
    each end; ns: samples per trace."""
    P = np.asarray(pos)
    N = np.asarray(ns)
    m = int(lab.max())
    out = np.zeros(m + 1, np.int64)
    shape = np.array(lab.shape)
    B = 4096
    for a in range(0, len(P), B):
        blk = P[a : a + B].astype(np.float64)
        nb = len(blk)
        live = np.arange(blk.shape[1])[None, :] < N[a : a + B, None]
        fin = live & np.all(np.isfinite(blk), -1)
        q = np.round(np.where(fin[..., None], blk, 0.0)).astype(np.int64)
        inb = fin & np.all((q >= 0) & (q < shape), -1)
        qc = np.clip(q, 0, shape - 1)
        c = np.where(inb, lab[qc[..., 0], qc[..., 1], qc[..., 2]], 0)
        tr = np.broadcast_to(np.arange(nb)[:, None], c.shape)
        pairs = np.unique(c[c > 0] * nb + tr[c > 0])
        out += np.bincount(pairs // nb, minlength=m + 1)
    return out


# ---------------------------------------------------------------- change file


@dataclass(frozen=True)
class LabelChanges:
    """The pial-side label changes of a box (module docstring)."""

    lo: Box
    hi: Box
    #: (n, 3) int32, voxels of the full grid
    voxels: npt.NDArray[np.int32]
    #: (n,) int8, the new rim label (1 fluid, 3 cortex)
    label: npt.NDArray[np.int8]
    #: (n,) int32, the component of each voxel, -1 a brush edit
    component: npt.NDArray[np.int32]
    #: (m,) int8, PROPOSED, ACCEPTED or REJECTED per component
    status: npt.NDArray[np.int8]
    #: per component: its descriptors (propose), JSON-ready
    components: list[dict[str, Any]] = field(default_factory=list)
    #: the box's detection settings and noise levels
    info: dict[str, Any] = field(default_factory=dict)
    #: Betti numbers (Kuhn) of the box's tissue (every rim label but
    #: outside) before and after the applied changes (finalize)
    betti_before: Betti = (0, 0, 0)
    betti_after: Betti = (0, 0, 0)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            lo=np.array(self.lo),
            hi=np.array(self.hi),
            voxels=self.voxels,
            label=self.label,
            component=self.component,
            status=self.status,
            components=np.array(json.dumps(self.components)),
            info=np.array(json.dumps(self.info)),
            betti_before=np.array(self.betti_before),
            betti_after=np.array(self.betti_after),
        )

    @classmethod
    def load(cls, path: Path) -> "LabelChanges":
        z = np.load(path)
        lo, hi = z["lo"].tolist(), z["hi"].tolist()
        b0, b1 = z["betti_before"].tolist(), z["betti_after"].tolist()
        return cls(
            lo=(lo[0], lo[1], lo[2]),
            hi=(hi[0], hi[1], hi[2]),
            voxels=z["voxels"].astype(np.int32),
            label=z["label"].astype(np.int8),
            component=z["component"].astype(np.int32),
            status=z["status"].astype(np.int8),
            components=json.loads(str(z["components"])),
            info=json.loads(str(z["info"])),
            betti_before=(b0[0], b0[1], b0[2]),
            betti_after=(b1[0], b1[1], b1[2]),
        )

    def applied(self, reviewed_only: bool = False) -> Bool:
        """Which entries apply: the cells that are not rejected (accepted
        and proposed; with reviewed_only, accepted only) and brush edits."""
        c = np.maximum(self.component, 0)
        st = np.where(self.component >= 0, self.status[c], ACCEPTED)
        ok: Bool = (st == ACCEPTED) | ((not reviewed_only) & (st == PROPOSED))
        return ok

    def apply(
        self,
        rim: npt.NDArray[np.int16],
        lo: npt.ArrayLike,
        reviewed_only: bool = False,
    ) -> npt.NDArray[np.int16]:
        """rim (a box's rim labels, its corner lo on the full grid) with the
        applied entries, in order (a later brush edit of a voxel wins); WM
        voxels are never changed."""
        out = rim.copy()
        ok = self.applied(reviewed_only)
        p = self.voxels[ok].astype(np.int64) - np.asarray(lo)
        lab = self.label[ok]
        inside = np.all((p >= 0) & (p < np.array(rim.shape)), 1)
        p, lab = p[inside], lab[inside]
        flat = np.ravel_multi_index(tuple(p.T), rim.shape)
        _, last = np.unique(flat[::-1], return_index=True)
        last = len(flat) - 1 - last  # each voxel's last entry
        p, lab = p[last], lab[last]
        keep = out[tuple(p.T)] != RIM_WM
        out[tuple(p[keep].T)] = lab[keep]
        return out

    def with_status(self, comp: int, status: int) -> "LabelChanges":
        st = self.status.copy()
        st[comp] = status
        return replace(self, status=st)

    def with_brush(self, voxels: npt.ArrayLike, label: int) -> "LabelChanges":
        """The changes with brush edits (full-grid voxels, new rim label)
        appended."""
        v = np.asarray(voxels, np.int32).reshape(-1, 3)
        return replace(
            self,
            voxels=np.concatenate([self.voxels, v]),
            label=np.concatenate([self.label, np.full(len(v), label, np.int8)]),
            component=np.concatenate([self.component, np.full(len(v), -1, np.int32)]),
        )

    def finalize(self, rim: npt.NDArray[np.int16]) -> "LabelChanges":
        """With the Betti numbers of the box's tissue (rim, the box's rim
        labels before the changes; not outside) before and after the applied
        changes (the best guess)."""
        before = kuhn_betti(rim != RIM_OUTSIDE)
        after = kuhn_betti(self.apply(rim, self.lo) != RIM_OUTSIDE)
        return replace(self, betti_before=before, betti_after=after)


def overlay(changes: LabelChanges, dark: Bool | None = None) -> npt.NDArray[np.int16]:
    """The box's overlay: component voxels as 1 + status (1 proposed, 2
    accepted, 3 rejected), brush edits (OV_BRUSH_FLUID, OV_BRUSH_CORTEX), and
    the dark voxels of no component (OV_DARK)."""
    shape = tuple(int(b - a) for a, b in zip(changes.lo, changes.hi, strict=True))
    ov = np.zeros(shape, np.int16)
    if dark is not None:
        ov[dark] = OV_DARK
    p = changes.voxels.astype(np.int64) - np.asarray(changes.lo)
    c = changes.component
    st = changes.status[np.maximum(c, 0)].astype(np.int16) + 1
    brush = np.where(changes.label == RIM_CORTEX, OV_BRUSH_CORTEX, OV_BRUSH_FLUID)
    ov[tuple(p.T)] = np.where(c >= 0, st, brush)
    return ov


# ---------------------------------------------------------------- proposals


@dataclass
class Proposal:
    """propose's result: the change file and what it was made from."""

    changes: LabelChanges
    dark: Darkness
    #: connected dark component of each box voxel (1..; 0 none)
    comp: I64
    #: review cell of each box voxel (LabelChanges component + 1; 0 none)
    cell: I64
    crest: Bool


def _describe(
    uid: int,
    gid: int,
    pts: I64,
    thr: Bool,
    near: Bool,
    tissue: Bool,
    group: Bool,
    dk: Darkness,
    t: CT.Tissue,
    lo: I64,
) -> dict[str, Any]:
    """A review cell's own descriptors (propose adds its component's
    position, crest_hits and columns)."""
    touches = bool(near[tuple(pts.T)].any())
    share = float(thr.mean())
    kind, west = shape_kind(pts, t.vox)
    b0, b1 = topology_change(tissue, pts, also=group)
    at = pts[np.argmin(np.sum((pts - pts.mean(0)) ** 2, 1))]  # nearest its centre
    return dict(
        id=uid,
        group=gid,
        voxels=len(pts),
        kind=kind,
        westin=[round(float(x), 3) for x in west],
        extent_mm=np.round((np.ptp(pts, 0) + 1) * t.vox, 2).tolist(),
        touches_fluid=touches,
        pass_through=round(share, 3),
        z_median=round(float(np.median(dk.z[tuple(pts.T)])), 2),
        frac_median=round(float(np.median(dk.frac[tuple(pts.T)])), 2),
        betti_local_before=list(b0),
        betti_local_after=list(b1),
        at_box0=at.tolist(),
        at_full0=(at + lo).tolist(),
        at_itksnap=(at + lo + 1).tolist(),
    )


def auto_accept(rows: list[dict[str, Any]], cells: list[I64], tissue: Bool) -> int:
    """Step 7: accept, in passes until none is left, the proposed cells of
    components that touch fluid whose relabel, on top of the cells accepted
    so far, keeps the tissue's Betti numbers in a box round the cell
    (topology_change). Returns the number accepted."""
    state = tissue.copy()
    todo = [
        i
        for i, r in enumerate(rows)
        if r["auto_status"] == STATUS[PROPOSED] and r["group_touches_fluid"]
    ]
    n = 0
    while todo:
        left = []
        for i in todo:
            b0, b1 = topology_change(state, cells[i])
            if b0 == b1:
                state[tuple(cells[i].T)] = False
                rows[i]["auto_status"] = STATUS[ACCEPTED]
                n += 1
            else:
                left.append(i)
        if len(left) == len(todo):
            break
        todo = left
    return n


def propose(
    t: CT.Tissue,
    pos: npt.ArrayLike | None = None,
    ns: npt.ArrayLike | None = None,
    alpha: float = ALPHA,
) -> Proposal:
    """Steps 1-6 on a box (module docstring). pos, ns: a column volume's
    traces (box voxel coordinates), for the columns each component affects."""
    dk = darkness(t, alpha)
    base = CT.base_tensor(t)
    v, f, nrm = CT.white_mesh(t.wm)
    C = CT.fitted_crest(t, base, v, f, nrm)
    del base, v, f, nrm
    lab, n = ndi.label(dk.dark, _C26)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.flatnonzero(size >= dk.n_min)
    keep = keep[keep > 0]
    remap = np.zeros(n + 1, np.int64)
    remap[keep] = np.arange(1, len(keep) + 1)
    comp: I64 = remap[lab]
    near = ndi.binary_dilation((t.labels == 1) | (t.labels == 4), _C26)
    tissue_mask = t.labels != 1
    lo = np.asarray(t.patch.lo)
    rows: list[dict[str, Any]] = []
    vox_list, comp_list = [], []
    cell_of = np.zeros(comp.shape, np.int64)  # review cell id + 1
    cell_pts: list[I64] = []
    for c, s in enumerate(ndi.find_objects(comp), start=1):
        assert s is not None
        allp = np.argwhere(comp[s] == c) + np.array([a.start for a in s])
        cm = np.zeros(comp.shape, bool)
        cm[tuple(allp.T)] = True
        thr_all = pass_through(cm, allp, C.T, C.noise, t.labels, t.vox, dk.k)
        touches = bool(near[tuple(allp.T)].any())
        share = float(thr_all.mean())
        auto = "outer" if touches or share <= 0.5 else "inside"
        cells = review_cells(allp, t.vox, CT.PATCH_MM)
        for q in range(int(cells.max()) + 1):
            sel = cells == q
            pts, thr = allp[sel], thr_all[sel]
            uid = len(rows)
            cell_of[tuple(pts.T)] = uid + 1
            row = _describe(uid, c - 1, pts, thr, near, tissue_mask, cm, dk, t, lo)
            row.update(group_touches_fluid=touches, group_pass_through=round(share, 3))
            row["auto"] = auto
            row["auto_status"] = STATUS[REJECTED if auto == "inside" else PROPOSED]
            rows.append(row)
            cell_pts.append(pts)
            vox_list.append(pts + lo)
            comp_list.append(np.full(len(pts), uid, np.int32))
        del cm
    auto_accept(rows, cell_pts, tissue_mask)
    m = len(rows)
    around = ndi.maximum_filter(cell_of, footprint=_C26)
    crest_hits = np.bincount(around[C.crest], minlength=m + 1)
    cols = np.zeros(m + 1, np.int64)
    if pos is not None and ns is not None and m:
        cols = traces_through(cell_of, pos, ns)
    for r in rows:
        r["crest_hits"] = int(crest_hits[r["id"] + 1])
        r["columns"] = int(cols[r["id"] + 1])
    order = sorted(
        range(m),
        key=lambda i: (-rows[i]["columns"], -rows[i]["crest_hits"], -rows[i]["voxels"]),
    )
    for r_, i in enumerate(order):
        rows[i]["rank"] = r_
    status = np.array([STATUS.index(r["auto_status"]) for r in rows], np.int8)
    voxels = np.concatenate(vox_list) if vox_list else np.zeros((0, 3), np.int64)
    comps = np.concatenate(comp_list) if comp_list else np.zeros(0, np.int32)
    info = dict(
        alpha=alpha,
        k=round(dk.k, 3),
        echoes=[i + 1 for i in ECHOES],
        sigma=[round(s, 4) for s in dk.sigma],
        mad=[round(s, 4) for s in dk.mad],
        fluid_level=[round(s, 1) for s in dk.fluid],
        p_null=dk.p_null,
        n_min=dk.n_min,
        cortex_voxels=int(t.ctx.sum()),
        dark_voxels=int(dk.dark.sum()),
        dark_voxels_in_components=int((comp > 0).sum()),
        dark_components=len(keep),
        crest_voxels=int(C.crest.sum()),
        crest_voxels_without_component=int((C.crest & (around == 0)).sum()),
        traces=0 if pos is None else len(np.asarray(pos)),
    )
    ch = LabelChanges(
        lo=t.patch.lo,
        hi=t.patch.hi,
        voxels=voxels.astype(np.int32),
        label=np.full(len(voxels), RIM_OUTSIDE, np.int8),
        component=comps.astype(np.int32),
        status=status,
        components=rows,
        info=info,
    )
    return Proposal(ch, dk, comp, cell_of, C.crest)


def summary(changes: LabelChanges) -> dict[str, Any]:
    """Counts by status and kind, the box's Betti numbers, the settings, and
    the components in rank order."""
    rows = sorted(changes.components, key=lambda r: r["rank"])
    st = changes.status
    return dict(
        lo=list(changes.lo),
        hi=list(changes.hi),
        info=changes.info,
        betti_before=list(changes.betti_before),
        betti_after=list(changes.betti_after),
        by_status={
            STATUS[s]: dict(
                components=int((st == s).sum()),
                voxels=int(sum(r["voxels"] for r in rows if st[r["id"]] == s)),
            )
            for s in range(3)
        },
        by_kind={
            k: int(sum(r["kind"] == k and st[r["id"]] != REJECTED for r in rows))
            for k in KINDS
        },
        brush_voxels=int((changes.component < 0).sum()),
        components=[dict(r, status=STATUS[int(st[r["id"]])]) for r in rows],
    )


# ---------------------------------------------------------------- command line


def write(p: Proposal, t: CT.Tissue, out: Path, name: str) -> dict[str, Any]:
    """pial_labels_<name>.npz, .json, _labels.nii.gz and _overlay.nii.gz in
    out (module docstring); returns the summary."""
    out.mkdir(parents=True, exist_ok=True)
    ch = p.changes.finalize(t.rim)
    stem = f"pial_labels_{name}"
    ch.save(out / f"{stem}.npz")
    s = summary(ch)
    rim = ch.apply(t.rim, ch.lo)
    s["betti_accepted_only"] = list(
        kuhn_betti(ch.apply(t.rim, ch.lo, reviewed_only=True) != RIM_OUTSIDE)
    )
    nib.save(nib.nifti1.Nifti1Image(rim, t.patch.affine), out / f"{stem}_labels.nii.gz")
    (out / f"{stem}.json").write_text(json.dumps(s, indent=1))
    ov = overlay(ch, p.dark.dark & (p.comp == 0))
    nib.save(nib.nifti1.Nifti1Image(ov, t.patch.affine), out / f"{stem}_overlay.nii.gz")
    return s


def main(argv: list[str] | None = None) -> None:
    from baillarger.white import WMChanges

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("propose", help="propose relabels in a box")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--name", required=True)
    c.add_argument("--lo", required=True, type=int, nargs=3)
    c.add_argument("--hi", required=True, type=int, nargs=3)
    c.add_argument("--wm-changes", type=Path)
    c.add_argument("--columns", type=Path, help="column_volume_<name>.npz")
    c.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args(argv)
    lo, hi = (a.lo[0], a.lo[1], a.lo[2]), (a.hi[0], a.hi[1], a.hi[2])
    wm = WMChanges.load(a.wm_changes) if a.wm_changes else None
    t = CT.load_tissue(lo, hi, a.name, changes=wm)
    pos = ns = None
    if a.columns:
        Z = np.load(a.columns)
        if not np.array_equal(Z["lo"], lo):
            raise SystemExit(f"{a.columns}: box corner {Z['lo']} is not {lo}")
        pos, ns = Z["pos"], Z["ns"]
    s = write(propose(t, pos, ns, a.alpha), t, a.out, a.name)
    print(json.dumps(dict(info=s["info"], by_status=s["by_status"]), indent=1))


if __name__ == "__main__":
    main()
