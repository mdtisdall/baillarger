"""The image stop field of the flow-field phantoms stage
(docs/plans/flow-field-phantoms.md, *Inputs*, "The point-wise stop terms"; task 2.1).

`ImageStopField(phantom, tensor, reference)` gives the log-odds that a point is stopped,
at any voxel positions X (N, 3). It implements `ff_types.StopField` with source "image".
The terms are those of `top_confidence.py` (and their definitions in `top_features.py`),
evaluated at points, not per ray:

- **label**: `at_label` (+1 where the point is within one voxel of the outside medium
  label, `top_features`: distance to the outside label <= the voxel size) plus
  `label_clear` (+1 where the point's own label is the outside medium and the medium run
  through it is at least SEED_RUN samples of STEP mm, `top_features.run`). So 0, 1 or 2.
- **sheet**: S = max(lambda1 - |lambda2|, 0) of the Hessian of the detrended echo 2 (the
  plan: echo 2, Gaussian derivatives at the band scale,
  `SolverSettings.band_scale_vox`), with lambda1 >= lambda2 the two largest eigenvalues
  (`coarse_kernels.leading_t`). A dark sheet gives lambda1 > 0. The term is
  `top_confidence`'s clip(log(S / S_floor), -1, 2).
- **mirror**: the Pearson correlation of the profile below the point, mirrored, against
  the profile above it, along the tensor direction, over offsets MIRROR_MIN_MM to
  `SolverSettings.mirror_mm` at STEP mm, both echoes jointly (`top_features`). The term
  is `top_confidence`'s clip(2 (r - 0.3), -2, 1); -1 where r is not finite. Mirroring
  makes it sign-free in the direction.
- **partial_volume**: f, the outside medium's fraction in the voxel, by unmixing the two
  log echoes into the WM, cortex and medium levels (fractions sum to 1: two equations,
  two unknowns). The term is 2 f - 1 (f clipped to [0, 1]).

**Combination**: log_odds = PRIOR + label + sheet + mirror + partial_volume, with PRIOR
= 0. As in `top_confidence` (the evidence is the nansum of the terms), a term that does
not apply at a point is 0 there: sheet does not apply where the point's label is the
outside medium (its `t_sheet` is nan for label candidates), mirror does not apply within
one voxel of it (nan for `at_label` candidates), nor where S < S_floor (decision 2).

**Site statistics** (`SiteStats`) come from the phantom built with noise 0
(`reference`): S_floor is the 90th percentile of S over the cortex voxels more than
MIN_TOP_MM from the WM (`top_features`: cortex samples past the first 0.5 mm of the
ray); the class levels are the medians of the noise-free log echoes over each class's
voxels eroded by one voxel; the cortex median intensities give the mirror profiles'
cortex units (`top_features`: echoes / cortex median).

**The raw echoes.** `fold_phantoms.make` keeps only the detrended echoes. The
partial-volume and mirror terms need the log echoes before the detrend, so `log_echoes`
renders them again with the phantom's own renderer and noise draw, and checks that their
detrend equals the phantom's `det` (else it raises).

**Provisional decisions** (each one is a choice where the per-ray form has no point-wise
form):

1. PRIOR = 0. `top_confidence` has no constant: its only prior-like terms (the global
   thickness term, 0 at the modes, and the local thickness term) are functions of a
   column's top depth along its ray, not of a point. The point-wise form of the
   thickness prior (a depth from the WM) mixes the banks where they differ in thickness,
   so it is not used.
2. The terms that need a column are left out: the thickness terms, far_wm, non-WM behind
   the root and the opposite-bank term. The non-maximum suppression of S along the ray,
   and the candidate filter S >= S_floor, have no point-wise form: S below the floor
   gives down to -1. The candidate filter does gate the mirror term: it applies only
   where S >= S_floor (the coordinator's decision CD-5, after the plan's slab test was
   seen to fail without it: below the dark top band the point is an intensity extremum
   along the line, so it looks mirror-symmetric, and the ungated term was +1 there).
3. The partial-volume term (no counterpart in `top_confidence`): 2 f - 1, the simplest
   bounded term that is 0 at half a voxel of medium, with the same largest size (1) as
   each label part.
4. The sheet's Hessian is computed on `det[1]` as given (zero outside the cortex, as
   `fold_phantoms.make` masks it), with Gaussian derivatives (mode "nearest").
5. The mirror profiles are the raw echoes (exp of the log echoes) in cortex units,
   echoes 2 and 3 (the phantoms have no echo 1). A sample outside the box is not valid;
   at least MIRROR_MIN_VALID valid pairs are needed (`top_features`: 10), else r is nan.
   The tensor's coherence is not used (`top_confidence` has no coherence).
6. The label is sampled at the nearest voxel centre; `at_label` uses the trilinear
   distance to the medium label (as `top_features` samples it trilinearly). The clear
   run is counted on the line through the point along the tensor direction, both ways
   (sign-free): contiguous medium samples at STEP mm through the point, at least
   SEED_RUN of them.
7. Class levels: medians over each class eroded by one voxel (26-neighbourhood), each
   class needs at least MIN_CLASS_VOXELS voxels; else the partial-volume term is 0
   everywhere (the noise phantom: no WM and no medium).

**NumPy or Torch** (the plan's *NumPy or Torch*): NumPy float64. The voxel arrays are
computed once per phantom; a point costs a fixed number of trilinear gathers (about 120)
and one 3x3 eigen-solve, with no Python loop over points (chunks only bound the memory).
The stop field is an input: the solver reads it once per node, not per sweep.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

# the per-ray constants of top_features.py / top_confidence.py
STEP = 0.04  # mm: the ray sample step
SEED_RUN = 8  # samples: a clear label top has this many medium samples from it
MIN_TOP_MM = 0.5  # the cortex samples of the sheet floor start this far from the root
MIRROR_MIN_MM = 0.12  # the first mirror offset (top_features.MIRROR_MM[0])
MIRROR_MIN_VALID = 10  # pairs needed for a mirror correlation
SHEET_FLOOR_PCT = 90.0
SHEET_CLIP = (-1.0, 2.0)
MIRROR_CLIP = (-2.0, 1.0)
MIRROR_CENTRE = 0.3
MIRROR_NAN = -1.0
LABEL_AT = 1.0
LABEL_CLEAR = 1.0
# the provisional point-wise choices (module docstring)
PRIOR = 0.0
MIN_CLASS_VOXELS = 100  # top_features: a class mean needs 100 voxels
CHUNK = 32768  # points per chunk (memory only; the result does not depend on it)

TERM_NAMES = ("label", "sheet", "mirror", "partial_volume")
_HESS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


# ---------------------------------------------------------------- the raw echoes


def log_echoes(phantom: T.Phantom) -> tuple[T.Array, T.Array]:
    """The log echoes 2 and 3 of the phantom's image before the detrend: the phantom's
    renderer and its noise draw (the same order as `fold_phantoms.make`). Raises
    ValueError if their detrend is not the phantom's det to 1e-12."""
    fold = phantom.image["fold"]
    feats = phantom.features
    import ff_ckiss as CK
    import ff_tau as TAU

    if isinstance(fold, CK.ConnectedKiss):
        _, imgs = CK.render(fold, feats, bool(fold.p.get("flat", False)))
    elif isinstance(fold, TAU.Tau):
        _, imgs = TAU.render_tau(fold, feats)
    else:
        _, imgs = FP.render(fold, feats)  # type: ignore[no-untyped-call]
    rng = np.random.default_rng(phantom.seed)
    ctx = np.asarray(phantom.image["ctx"])
    out = []
    for k, L0 in enumerate(imgs):
        sd = float(phantom.image["noise_sd"][k])
        L = L0 + phantom.noise * sd * rng.standard_normal(L0.shape)
        det = FP.detrend(L, ctx)  # type: ignore[no-untyped-call]
        err = float(np.abs(det - phantom.det[k + 1]).max())
        if not err <= 1e-12:
            raise ValueError(
                f"echo {k + 2}: the re-rendered image differs from det by {err:g}"
            )
        out.append(np.asarray(L, np.float64))
    return out[0], out[1]


# ---------------------------------------------------------------- voxel arrays


def hessian(vol: T.Array, sigma_vox: float) -> T.Array:
    """(6, ...) the Gaussian-derivative Hessian of vol (xx, xy, xz, yy, yz, zz)."""
    v = np.asarray(vol, np.float64)
    comps = []
    for i, j in _HESS:
        order = [0, 0, 0]
        order[i] += 1
        order[j] += 1
        comps.append(ndi.gaussian_filter(v, sigma_vox, order=order, mode="nearest"))
    return np.stack(comps, 0)


def sheetness(H6: T.Array) -> T.Array:
    """S = max(lambda1 - |lambda2|, 0) per row of H6 (N, 6): lambda1 >= lambda2 the two
    largest eigenvalues of the symmetric 3x3 matrix."""
    H6 = np.asarray(H6, np.float64)
    M = np.empty((len(H6), 3, 3))
    for k, (i, j) in enumerate(_HESS):
        M[:, i, j] = H6[:, k]
        M[:, j, i] = H6[:, k]
    w = np.linalg.eigvalsh(M)  # ascending
    return np.asarray(np.maximum(w[:, 2] - np.abs(w[:, 1]), 0.0))


def _eroded(mask: T.BoolArray) -> T.BoolArray:
    return np.asarray(ndi.binary_erosion(mask, np.ones((3, 3, 3), bool)), bool)


def medium_fraction(L2: T.Array, L3: T.Array, levels: T.Array) -> T.Array:
    """f, the medium fraction per voxel: L = a wm + b ctx + f med per echo,
    a + b + f = 1, solved exactly for (a, f); f clipped to [0, 1]. levels (3, 2):
    rows WM, cortex, medium; columns echo 2, echo 3. nan levels give f = nan."""
    w, c, m = levels
    A = np.array([[w[0] - c[0], m[0] - c[0]], [w[1] - c[1], m[1] - c[1]]])
    if not np.all(np.isfinite(A)):
        return np.full(np.shape(L2), np.nan)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    r2, r3 = L2 - c[0], L3 - c[1]
    f = (A[0, 0] * r3 - A[1, 0] * r2) / det
    return np.asarray(np.clip(f, 0.0, 1.0))


@dataclass(frozen=True)
class SiteStats:
    """The site statistics of `top_confidence`, from the noise-free phantom."""

    s_floor: float  # 90th percentile of S, cortex voxels > MIN_TOP_MM from the WM
    levels: (
        T.Array
    )  # (3, 2) log-echo medians: rows WM, cortex, medium; columns echo 2, echo 3
    cortex_median: (
        T.Array
    )  # (2,) median raw intensity exp(L) over the cortex (mirror units)
    n_class: tuple[int, int, int]  # voxels behind each level (WM, cortex, medium)

    @classmethod
    def from_reference(
        cls, reference: T.Phantom, settings: T.SolverSettings = T.DEFAULTS
    ) -> SiteStats:
        if reference.noise != 0.0:
            raise ValueError("the reference phantom must have noise 0")
        lab = reference.labels
        ctx, wm, med = lab == T.CORTEX, lab == T.WM, lab == T.MEDIUM
        H6 = hessian(reference.det[1], settings.band_scale_vox)
        d_wm = (
            ndi.distance_transform_edt(~wm) * T.VOXEL_MM
            if wm.any()
            else np.full(lab.shape, np.inf)
        )
        sel = ctx & (d_wm > MIN_TOP_MM)
        S = sheetness(H6[:, sel].T)
        s_floor = float(np.percentile(S, SHEET_FLOOR_PCT)) if len(S) else float("nan")
        L2, L3 = log_echoes(reference)
        levels = np.full((3, 2), np.nan)
        ns = []
        for r, m in enumerate((_eroded(wm), _eroded(ctx), _eroded(med))):
            ns.append(int(m.sum()))
            if m.sum() >= MIN_CLASS_VOXELS:
                levels[r] = (np.median(L2[m]), np.median(L3[m]))
        cmed = np.array([np.median(np.exp(L2[ctx])), np.median(np.exp(L3[ctx]))])
        return cls(s_floor, levels, cmed, (ns[0], ns[1], ns[2]))


# ---------------------------------------------------------------- the field


def _sample(vol: T.Array, P: T.Array) -> T.Array:
    """Trilinear samples of vol at P (..., 3), edges held (`ff_types.sample_volume`)."""
    flat = np.asarray(P, np.float64).reshape(-1, 3)
    return T.sample_volume(vol, flat).reshape(P.shape[:-1])


def _nearest(vol: Any, P: T.Array) -> Any:
    """vol at the nearest voxel centre of each point P (..., 3), clipped to the box."""
    shp = np.array(vol.shape)
    idx = np.clip(np.rint(P), 0, shp - 1).astype(np.int64)
    return vol[idx[..., 0], idx[..., 1], idx[..., 2]]


class ImageStopField:
    """The image stop field (module docstring). `reference` is the same phantom built
    with noise 0, or its `SiteStats`."""

    source: T.InputSource = "image"

    def __init__(
        self,
        phantom: T.Phantom,
        tensor: T.Tensor,
        reference: T.Phantom | SiteStats,
        settings: T.SolverSettings = T.DEFAULTS,
    ) -> None:
        self.phantom = phantom
        self.tensor = tensor
        self.settings = settings
        self.stats = (
            reference
            if isinstance(reference, SiteStats)
            else SiteStats.from_reference(reference, settings)
        )
        lab = phantom.labels
        self.shape = np.array(lab.shape)
        self.medium = lab == T.MEDIUM
        self.d_out = (
            ndi.distance_transform_edt(~self.medium) * T.VOXEL_MM
            if self.medium.any()
            else np.full(lab.shape, np.inf)
        )
        self.H6 = hessian(phantom.det[1], settings.band_scale_vox)
        L2, L3 = log_echoes(phantom)
        self.f = medium_fraction(L2, L3, self.stats.levels)
        self.pv_on = bool(np.all(np.isfinite(self.f)))
        cm = self.stats.cortex_median
        self.I = (np.exp(L2) / cm[0], np.exp(L3) / cm[1])
        n_lo = int(round(MIRROR_MIN_MM / STEP))
        n_hi = int(round(settings.mirror_mm / STEP))
        self.mirror_off_vox = np.arange(n_lo, n_hi + 1) * STEP / T.VOXEL_MM
        self.run_off_vox = np.arange(-SEED_RUN, SEED_RUN + 1) * STEP / T.VOXEL_MM

    # -- the terms on one chunk
    def _chunk(self, X: T.Array) -> dict[str, T.Array]:
        n = len(X)
        d, _ = self.tensor.at(X)
        d = np.asarray(d, np.float64)
        in_med = np.asarray(_nearest(self.medium, X), bool)
        d_out = _sample(self.d_out, X)
        at_label = d_out <= T.VOXEL_MM
        # label: at_label + a clear medium run through the point along the direction
        line = X[:, None, :] + self.run_off_vox[None, :, None] * d[:, None, :]
        mline = np.asarray(_nearest(self.medium, line), bool)
        c = SEED_RUN
        right = np.cumprod(mline[:, c + 1 :], axis=1).sum(1)
        left = np.cumprod(mline[:, :c][:, ::-1], axis=1).sum(1)
        run = np.where(mline[:, c], 1 + left + right, 0)
        clear = in_med & (run >= SEED_RUN)
        label = LABEL_AT * at_label + LABEL_CLEAR * clear
        # sheet (does not apply where the label is the medium)
        H6 = np.stack([_sample(h, X) for h in self.H6], 1)
        S = sheetness(H6)
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = np.clip(np.log(np.maximum(S, 1e-12) / self.stats.s_floor), *SHEET_CLIP)
        sheet = np.where(in_med, 0.0, ts)
        # mirror (does not apply within one voxel of the medium, nor where S < S_floor:
        # top_confidence scores the mirror only at candidates, and a candidate has S >= S_floor)
        r = self._mirror(X, d)
        tm = np.where(
            np.isfinite(r), np.clip(2.0 * (r - MIRROR_CENTRE), *MIRROR_CLIP), MIRROR_NAN
        )
        mirror = np.where(at_label | (S < self.stats.s_floor), 0.0, tm)
        # partial volume
        pv = 2.0 * _sample(self.f, X) - 1.0 if self.pv_on else np.zeros(n)
        return dict(label=label, sheet=sheet, mirror=mirror, partial_volume=pv)

    def _mirror(self, X: T.Array, d: T.Array) -> T.Array:
        """The mirror correlation r at each point (nan where it is not defined)."""
        off = self.mirror_off_vox
        below = X[:, None, :] - off[None, :, None] * d[:, None, :]
        above = X[:, None, :] + off[None, :, None] * d[:, None, :]
        hi = self.shape - 1
        ok = np.all((below >= 0) & (below <= hi), -1) & np.all(
            (above >= 0) & (above <= hi), -1
        )
        a = np.stack([_sample(v, below) for v in self.I], -1)  # (n, J, 2)
        b = np.stack([_sample(v, above) for v in self.I], -1)
        okk = np.repeat(ok[:, :, None], 2, 2)
        nv = okk.sum((1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            ma = (a * okk).sum((1, 2)) / nv
            mb = (b * okk).sum((1, 2)) / nv
            da = (a - ma[:, None, None]) * okk
            db = (b - mb[:, None, None]) * okk
            r = (da * db).sum((1, 2)) / np.sqrt(
                (da**2).sum((1, 2)) * (db**2).sum((1, 2))
            )
        return np.asarray(np.where(ok.sum(1) < MIRROR_MIN_VALID, np.nan, r))

    # -- the StopField protocol
    def terms(self, X: T.Array) -> dict[str, T.Array]:
        """The four terms (additive log-odds) at voxel positions X (N, 3)."""
        X = np.asarray(X, np.float64).reshape(-1, 3)
        parts = [self._chunk(X[s : s + CHUNK]) for s in range(0, len(X), CHUNK)]
        if not parts:
            return {k: np.zeros(0) for k in TERM_NAMES}
        return {k: np.concatenate([p[k] for p in parts]) for k in TERM_NAMES}

    def log_odds(self, X: T.Array) -> T.Array:
        """PRIOR + label + sheet + mirror + partial_volume at X."""
        t = self.terms(X)
        out = np.full(len(t["label"]), PRIOR)
        for k in TERM_NAMES:
            out = out + t[k]
        return out

    def probability(self, X: T.Array) -> T.Array:
        return T.sigmoid(self.log_odds(X))

    def record(self) -> dict[str, Any]:
        """The settings and statistics of this field, for run.json."""
        s = self.stats
        return dict(
            source=self.source,
            prior=PRIOR,
            s_floor=s.s_floor,
            levels=s.levels.tolist(),
            cortex_median=s.cortex_median.tolist(),
            n_class=list(s.n_class),
            partial_volume_on=self.pv_on,
            band_scale_vox=self.settings.band_scale_vox,
            mirror_mm=self.settings.mirror_mm,
        )
