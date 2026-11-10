"""Excisions to outside: a bootstrap tool for an aseg that imputes tissue.

An excision is a block of tissue that was cut out of the specimen before the
scan. The outside medium fills it, and the FLASH is black there in each echo,
but an aseg that imputes tissue (such as purple-mri's) gives it a tissue label.
detect finds these voxels from the three echoes: each voxel is compared with
a model of the outside medium and with models of the tissue classes, so that
tissue with a short T2* (iron in the pallidum and thalamus), which is dark in
the later echoes, is not taken for medium. excise sets the excisions to 0
(outside), with the small tissue pieces they cut off, so that the white
surface closes there (the rule for cut faces, docs/formats/wm-dseg.md). Use it
only on such an aseg. Once our own labels exist, this module has nothing to
do.

The command line:

    python -m baillarger.excisions ASEG FLASH --echo1 E1 --echo3 E3 \\
        --hemi H --out DERIVATIVES

FLASH is echo 2 on the aseg's grid. E1 and E3 are echoes 1 and 3. The command
writes the excised aseg (desc-excised_dseg), the input label of
baillarger.white_dseg, the excision map (desc-excisions_mask) and the bubble
mask (desc-bubbles_mask), with their sidecars (docs/formats/wm-excisions.md).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    dataset_description,
    derivative_path,
    entities,
    provenance,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.labels import CRUMB_MM3 as CRUMB_MM3
from baillarger.timing import Laps, log_to_stderr
from baillarger.white_dseg import LUT_NAMES

type Bool = npt.NDArray[np.bool_]
type Labels = npt.NDArray[np.integer[Any]]
type F64 = npt.NDArray[np.float64]

#: the WM label and the cortex label, by hemisphere (FreeSurfer LUT)
TISSUE_LABELS = {"R": (41, 42), "L": (2, 3)}

#: labels that are not tissue (FreeSurfer LUT): the ventricles and CSF. Each
#: other label above 0 is tissue, and the detector checks it
NOT_TISSUE = frozenset({4, 5, 14, 15, 24, 43, 44, 72})

#: the detector's defaults: blocks of MIN_VOX voxels or more; the medium is
#: fitted on the outside voxels at least MEDIUM_GAP_MM from any label; each fit
#: uses at most SAMPLE voxels of its set, drawn with the seed SEED
MIN_VOX, MEDIUM_GAP_MM, SAMPLE, SEED = 250, 1.0, 1_000_000, 0

#: the old single-echo detector's defaults (dark_tissue, baillarger.frame),
#: and the dark level and smoothing of the bubble step: FRAC x the median of
#: each echo over the cortex label, after smoothing by SIGMA voxels
FRAC, SIGMA, MARGIN_MM = 0.25, 2.0, 0.3

#: added to the diagonal of each fitted covariance (log intensity squared)
RIDGE = 1e-6

#: EM stops when no share changes by more than EM_TOL, or after EM_STEPS steps
EM_TOL, EM_STEPS = 1e-7, 500

#: excision-map pieces below FRAGMENT_MM3 are dropped; tissue pieces below
#: CRUMB_MM3 (baillarger.labels) that touch the excision map are set to 0
FRAGMENT_MM3 = 0.1

#: a piece is a bubble if its dark region is closed in each echo (within the
#: piece's box, grown by BUBBLE_BOX voxels) and grows from echo 1 to echo 3 by
#: at least BUBBLE_RATIO
BUBBLE_RATIO, BUBBLE_BOX = 2.0, 20

#: docs/formats/wm-excisions.md
EXCISIONS_SPEC = SidecarSpec(
    "wm-excisions/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("TissueLabels", "array of integers", True, "labels the detector checks"),
        Key("Detector", "object", True, "the detector's parameters"),
        Key("MediumClass", "object", True, "the fitted model of the medium"),
        Key("TissueClasses", "object", True, "the fitted tissue classes"),
        Key("LabelWeights", "object", True, "each label's fitted shares"),
        Key("CortexMedian", "number", True, "median echo 2 over the cortex"),
        Key("ExcisedVoxels", "integer", True, "voxels set to 0"),
        Key("ExcisedByLabel", "object", True, "voxels set to 0, by old label"),
        Key("CortexVoxels", "integer", True, "voxels with the cortex label"),
        Key("Components", "integer", True, "components of the excision map"),
        Key("Fragments", "object", True, "excision-map pieces dropped"),
        Key("Crumbs", "object", True, "cut-off tissue pieces set to 0"),
        Key("Bubbles", "object", True, "air bubbles: not excisions"),
    ),
)


def dilation_steps(margin_mm: float, voxel_mm: tuple[float, float, float]) -> int:
    """The steps of binary dilation that grow a mask by margin_mm."""
    return int(round(margin_mm / min(voxel_mm)))


def cortex_median(aseg: npt.ArrayLike, flash: npt.ArrayLike, hemi: str) -> float:
    """The median of the FLASH over the cortex label of hemi. Raises
    ValueError if the aseg has no voxel with the WM label or the cortex label
    of hemi."""
    a = np.asarray(aseg)
    wm, ctx = TISSUE_LABELS[hemi]
    for lab, name in ((wm, "WM"), (ctx, "cortex")):
        if not np.any(a == lab):
            raise ValueError(
                f"no voxel with the {name} label of hemisphere {hemi} ({lab})"
            )
    return float(np.median(np.asarray(flash, np.float32)[a == ctx]))


def tissue_labels(aseg: npt.ArrayLike) -> list[int]:
    """The labels of the aseg that are tissue: above 0 and not in
    NOT_TISSUE."""
    return [
        int(k) for k in np.unique(np.asarray(aseg)) if k > 0 and k not in NOT_TISSUE
    ]


@dataclass(frozen=True)
class Gauss:
    """A Gaussian on the log of the three echoes."""

    mean: F64
    cov: F64

    def logpdf(self, y: F64) -> F64:
        d = y - self.mean
        q = np.einsum("ni,ij,nj->n", d, np.linalg.inv(self.cov), d)
        out: F64 = -0.5 * q - 0.5 * float(np.log(np.linalg.det(2 * np.pi * self.cov)))
        return out

    def record(self) -> dict[str, Any]:
        return dict(Mean=self.mean.tolist(), Covariance=self.cov.tolist())


def fit_gauss(y: F64, w: F64 | None = None) -> Gauss:
    """The Gaussian of the rows of y (weights w), with RIDGE on the
    diagonal."""
    w = np.ones(len(y)) if w is None else w
    mu = (w[:, None] * y).sum(0) / w.sum()
    d = y - mu
    c = np.einsum("n,ni,nj->ij", w, d, d) / w.sum() + RIDGE * np.eye(y.shape[1])
    return Gauss(mu, c)


def log_echoes(
    echoes: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike],
) -> npt.NDArray[np.float32]:
    """The log of the three echoes, voxel by voxel (values below 1 count as
    1), as one float32 array with the echo last. No smoothing: smoothing moves
    the edge of an excision into the tissue."""
    e0 = np.asarray(echoes[0])
    x = np.empty(e0.shape + (3,), np.float32)
    for i, e in enumerate(echoes):
        x[..., i] = np.log(np.maximum(np.asarray(e, np.float32), 1.0))
    return x


def _sample(
    x: npt.NDArray[np.float32], mask: Bool, n: int, rng: np.random.Generator
) -> F64:
    idx = np.flatnonzero(mask.ravel())
    if len(idx) > n:
        idx = np.sort(rng.choice(idx, n, replace=False))
    return x.reshape(-1, 3)[idx].astype(np.float64)


def _softmax_rows(ll: F64) -> F64:
    r = np.exp(ll - ll.max(1, keepdims=True))
    out: F64 = r / r.sum(1, keepdims=True)
    return out


def fit_tissue(y: F64, medium: Gauss) -> Gauss:
    """The tissue class of one label: EM on the label's voxels y for a
    mixture of the medium (held fixed) and one Gaussian (fitted)."""
    lm = medium.logpdf(y)
    share = 0.05
    g = fit_gauss(y)
    for _ in range(EM_STEPS):
        r = _softmax_rows(
            np.stack([np.log(share) + lm, np.log(1 - share) + g.logpdf(y)], 1)
        )
        g = fit_gauss(y, r[:, 1] + 1e-12)
        new = float(r[:, 0].mean())
        done = abs(new - share) <= EM_TOL
        share = min(max(new, 1e-9), 1 - 1e-9)
        if done:
            break
    return g


def fit_weights(y: F64, medium: Gauss, classes: list[Gauss]) -> F64:
    """One label's shares of the medium and of each tissue class (all held
    fixed): EM on the label's voxels y. The first share is the medium's."""
    ll = np.stack([medium.logpdf(y)] + [c.logpdf(y) for c in classes], 1)
    w = np.full(ll.shape[1], 1.0 / ll.shape[1])
    for _ in range(EM_STEPS):
        new = _softmax_rows(ll + np.log(np.maximum(w, 1e-300))).mean(0)
        done = float(np.abs(new - w).max()) <= EM_TOL
        w = new
        if done:
            break
    return w


@dataclass(frozen=True)
class Detected:
    """What detect gives: the excision candidates in blocks of min_vox voxels
    or more, the labels it checked, and the fitted models."""

    mask: Bool
    labels: list[int]
    medium: Gauss
    medium_voxels: int
    classes: dict[int, Gauss]
    weights: dict[int, F64]


def detect(
    aseg: npt.ArrayLike,
    echoes: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike],
    voxel_mm: tuple[float, float, float],
    min_vox: int = MIN_VOX,
    gap_mm: float = MEDIUM_GAP_MM,
    sample: int = SAMPLE,
    seed: int = SEED,
) -> Detected:
    """The excisions: tissue voxels that look like the outside medium in the
    three echoes (1, 2, 3; docs/formats/wm-excisions.md, steps 1-5).
    1. x = the log of the three echoes at each voxel, not smoothed.
    2. The medium: a Gaussian of x over the voxels with label 0, all echoes
       above 0, at least gap_mm (steps of binary dilation) from any label.
    3. Each tissue label (tissue_labels): EM for the medium (fixed) and one
       Gaussian, the label's tissue class.
    4. Each tissue label: EM for its shares of the medium and of every
       label's tissue class (all fixed). A tissue label does not say which
       tissue a voxel is: iron-rich tissue in the thalamus is like the
       pallidum's class.
    5. A voxel is a candidate if share x density of the medium is larger
       than the sum of share x density over the tissue classes. Keep the
       connected blocks (face adjacency) of min_vox voxels or more.
    Each fit uses at most sample voxels, drawn with seed."""
    a = np.asarray(aseg)
    labels = tissue_labels(a)
    x = log_echoes(echoes)
    rng = np.random.default_rng(seed)
    pos = np.ones(a.shape, bool)
    for e in echoes:
        pos &= np.asarray(e) > 0
    near = ndi.binary_dilation(
        a > 0, iterations=max(dilation_steps(gap_mm, voxel_mm), 1)
    )
    med_mask = (a == 0) & pos & ~near
    del near, pos
    if not med_mask.any():
        raise ValueError(
            f"no outside voxel {gap_mm} mm from the labels: no medium to fit"
        )
    medium = fit_gauss(_sample(x, med_mask, sample, rng))
    nmed = int(med_mask.sum())
    del med_mask
    ys = {lab: _sample(x, a == lab, sample, rng) for lab in labels}
    classes = {lab: fit_tissue(ys[lab], medium) for lab in labels}
    cl = [classes[lab] for lab in labels]
    weights = {lab: fit_weights(ys[lab], medium, cl) for lab in labels}
    del ys
    cand = np.zeros(a.shape, bool)
    flat = x.reshape(-1, 3)
    for lab in labels:
        w = weights[lab]
        idx = np.flatnonzero((a == lab).ravel())
        for s in range(0, len(idx), 4_000_000):
            part = idx[s : s + 4_000_000]
            z = flat[part].astype(np.float64)
            lm = np.log(max(float(w[0]), 1e-300)) + medium.logpdf(z)
            lt = np.logaddexp.reduce(
                np.stack(
                    [
                        np.log(max(float(wk), 1e-300)) + c.logpdf(z)
                        for wk, c in zip(w[1:], cl, strict=True)
                    ]
                ),
                axis=0,
            )
            cand.ravel()[part[lm > lt]] = True
    del x, flat
    lab_, n = ndi.label(cand)
    big = np.zeros(n + 1, bool)
    big[1:] = np.bincount(lab_.ravel(), minlength=n + 1)[1:] >= min_vox
    mask: Bool = big[lab_]
    return Detected(mask, labels, medium, nmed, classes, weights)


def dark_tissue(
    tis: Bool,
    cortex: Bool,
    flash: npt.ArrayLike,
    voxel_mm: tuple[float, float, float],
    frac: float = FRAC,
    sigma: float = SIGMA,
    min_vox: int = MIN_VOX,
    margin_mm: float = MARGIN_MM,
) -> Bool:
    """The old single-echo detector, kept for baillarger.frame: tissue where
    the FLASH smoothed by sigma voxels is below frac x its median over the
    cortex, in connected blocks (face adjacency) of min_vox voxels or more,
    grown by margin_mm and kept in the tissue. It takes short-T2* tissue for
    an excision; detect does not. Without cortex the result is empty."""
    if not cortex.any():
        return np.zeros(tis.shape, bool)
    f = np.asarray(flash, np.float32)
    gm_med = np.median(f[cortex])
    blk = tis & (ndi.gaussian_filter(f, sigma) < frac * gm_med)
    lab, n = ndi.label(blk)
    big = np.zeros(n + 1, bool)
    big[1:] = np.bincount(lab.ravel(), minlength=n + 1)[1:] >= min_vox
    miss: Bool = big[lab]
    it = dilation_steps(margin_mm, voxel_mm)
    if it:
        miss = ndi.binary_dilation(miss, iterations=it) & tis
    return miss


def drop_fragments(
    miss: Bool, voxel_mm: tuple[float, float, float], min_mm3: float = FRAGMENT_MM3
) -> tuple[Bool, int, int]:
    """miss without its connected components (face adjacency) of less than
    min_mm3, and the number of pieces and voxels dropped."""
    lab, n = ndi.label(miss)
    sz = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sz * float(np.prod(voxel_mm)) >= min_mm3
    keep[0] = False
    drop = ~keep[1:]
    out: Bool = keep[lab]
    return out, int(drop.sum()), int(sz[1:][drop].sum())


def cut_off(
    tis: Bool,
    miss: Bool,
    voxel_mm: tuple[float, float, float],
    max_mm3: float = CRUMB_MM3,
    structure: Bool | None = None,
) -> tuple[Bool, int]:
    """The connected components of tis (adjacency structure, default face
    adjacency) of less than max_mm3 that touch miss (a voxel of the piece is
    miss or one of its 26 neighbours is), and their number. A small piece
    that does not touch miss is not in the result."""
    lab, n = ndi.label(tis, structure)
    sz = np.bincount(lab.ravel(), minlength=n + 1)
    near = ndi.binary_dilation(miss, np.ones((3, 3, 3), bool))
    sel = np.zeros(n + 1, bool)
    sel[np.unique(lab[near & tis])] = True
    sel &= sz * float(np.prod(voxel_mm)) < max_mm3
    sel[0] = False
    out: Bool = sel[lab]
    return out, int(sel.sum())


def dark_region(
    echo: npt.ArrayLike, piece: Bool, level: float, sigma: float = SIGMA
) -> Bool:
    """The connected components (face adjacency) of (echo smoothed by sigma
    voxels) < level that overlap piece. echo and piece are one box."""
    dark = ndi.gaussian_filter(np.asarray(echo, np.float32), sigma) < level
    lab, _ = ndi.label(dark)
    ids = np.unique(lab[piece & dark])
    out: Bool = np.isin(lab, ids[ids > 0])
    return out


def _radius(voxels: int) -> float:
    return float((3 * voxels / (4 * np.pi)) ** (1 / 3))


def bubbles(
    miss: Bool,
    echoes: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike],
    levels: tuple[float, float, float],
    sigma: float = SIGMA,
    ratio: float = BUBBLE_RATIO,
    box: int = BUBBLE_BOX,
) -> tuple[Bool, Bool, int]:
    """The pieces of miss (face adjacency) that are air bubbles, the bubble
    mask, and the number of bubbles. For each piece, in its bounding box
    grown by box voxels, V_e is the dark region of echo e (dark_region at
    levels[e]). The piece is a bubble if no V_e reaches the box's side and
    V_3 >= ratio x V_1. The mask is V_3 grown by ceil(r_3 - r_1) steps of
    binary dilation (face adjacency), r_e the radius of a ball of V_e
    voxels: the growth of the artefact from echo 1 to echo 3. A piece with
    no dark region in echo 3 (V_3 = 0: a thin piece that the smoothing
    lifts above the level) is no bubble."""
    e1, e2, e3 = (np.asarray(e) for e in echoes)
    lab, n = ndi.label(miss)
    found = np.zeros(n + 1, bool)
    mask = np.zeros(miss.shape, bool)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        b = tuple(
            slice(max(s.start - box, 0), min(s.stop + box, miss.shape[k]))
            for k, s in enumerate(sl)
        )
        piece = lab[b] == i
        dark = [
            dark_region(e[b], piece, lv, sigma)
            for e, lv in zip((e1, e2, e3), levels, strict=True)
        ]
        side = np.ones(piece.shape, bool)
        side[1:-1, 1:-1, 1:-1] = False
        if any((d & side).any() for d in dark):
            continue
        v1, v3 = int(dark[0].sum()), int(dark[2].sum())
        if v3 == 0 or v3 < ratio * v1:  # no dark region in echo 3: nothing grows
            continue
        found[i] = True
        it = int(np.ceil(_radius(v3) - _radius(v1)))
        grown = ndi.binary_dilation(dark[2], iterations=it) if it > 0 else dark[2]
        mask[b] |= grown
    out: Bool = found[lab]
    return out, mask, int(found.sum())


@dataclass(frozen=True)
class Excised:
    """What excise gives: the excised aseg, the excision map (each voxel
    that excise set to 0), the bubble mask, the detector's result, and the
    counts for the sidecar."""

    dseg: Labels
    mask: Bool
    bubble_mask: Bool
    detected: Detected
    fragments: dict[str, int]
    crumbs: dict[str, int]
    bubbles: dict[str, int]


def excise(
    aseg: npt.ArrayLike,
    echoes: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike],
    voxel_mm: tuple[float, float, float],
    hemi: str,
    min_vox: int = MIN_VOX,
    fragment_mm3: float = FRAGMENT_MM3,
    crumb_mm3: float = CRUMB_MM3,
    frac: float = FRAC,
    sigma: float = SIGMA,
    bubble_ratio: float = BUBBLE_RATIO,
) -> Excised:
    """The excisions set to outside (docs/formats/wm-excisions.md). echoes
    are echoes 1, 2 and 3:
    1-5. detect;
    6. drop the pieces of the result of less than fragment_mm3;
    7. drop the pieces that are air bubbles (bubbles, at frac x the median
       of each echo over the cortex label of hemi), and make the bubble mask;
    8. relabel (set the rest to 0);
    9. set to 0 each tissue piece (the tissue labels, face adjacency) of
       less than crumb_mm3 that touches the map of step 8.
    The excision map is the voxels of steps 8 and 9. Raises ValueError if
    the aseg has no voxel with the WM label or the cortex label of hemi."""
    a = np.asarray(aseg)
    e = (np.asarray(echoes[0]), np.asarray(echoes[1]), np.asarray(echoes[2]))
    cortex_median(a, e[1], hemi)  # refuses a hemisphere without its labels
    det = detect(a, e, voxel_mm, min_vox)
    miss, nfrag, vfrag = drop_fragments(det.mask, voxel_mm, fragment_mm3)
    cortex = a == TISSUE_LABELS[hemi][1]
    lv = tuple(frac * float(np.median(x[cortex])) for x in e)
    bub, bmask, nbub = bubbles(miss, e, (lv[0], lv[1], lv[2]), sigma, bubble_ratio)
    miss &= ~bub
    x = relabel(a, miss)
    crumb, ncrumb = cut_off(np.isin(x, det.labels), miss, voxel_mm, crumb_mm3)
    x[crumb] = 0
    return Excised(
        x,
        miss | crumb,
        bmask,
        det,
        dict(Pieces=nfrag, Voxels=vfrag),
        dict(Pieces=ncrumb, Voxels=int(crumb.sum())),
        dict(Pieces=nbub, Voxels=int(bub.sum()), MaskVoxels=int(bmask.sum())),
    )


def relabel(aseg: Labels, mask: npt.ArrayLike) -> Labels:
    """A copy of aseg with each voxel of mask set to 0 (outside)."""
    out = np.array(aseg, copy=True)
    out[np.asarray(mask, bool)] = 0
    return out


def _write_tsv(path: Path, names: dict[int, str]) -> None:
    rows = "".join(f"{k}\t{v}\n" for k, v in sorted(names.items()))
    path.write_text("index\tname\n" + rows)


def write_excisions(
    aseg: Path,
    flash: Path,
    echo1: Path,
    echo3: Path,
    hemi: str,
    out: Path,
    min_vox: int = MIN_VOX,
    fragment_mm3: float = FRAGMENT_MM3,
    crumb_mm3: float = CRUMB_MM3,
    frac: float = FRAC,
    sigma: float = SIGMA,
    bubble_ratio: float = BUBBLE_RATIO,
) -> dict[str, Path]:
    """The command: excise the aseg at aseg with the FLASH's echoes 1 (echo1),
    2 (flash) and 3 (echo3), written to the BIDS derivatives root out
    (docs/formats/wm-excisions.md). Raises ValueError if a FLASH grid has a
    different shape. Returns the paths written, by role."""
    lap = Laps()
    img = nib.load(aseg)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg.name}: not a NIfTI-1 image")
    imgs = []
    for path in (echo1, flash, echo3):
        e = nib.load(path)
        if not isinstance(e, nib.nifti1.Nifti1Image):
            raise ValueError(f"{path.name}: not a NIfTI-1 image")
        if img.shape[:3] != e.shape[:3]:
            raise ValueError(
                f"{path.name}: grid {e.shape[:3]}, not the aseg's {img.shape[:3]}"
            )
        imgs.append(e)
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    a = np.asarray(img.dataobj)
    ech = tuple(np.asarray(e.dataobj, np.float32) for e in imgs)
    lap("read")
    wm, ctx = TISSUE_LABELS[hemi]
    gm_med = cortex_median(a, ech[1], hemi)
    ex = excise(
        a,
        (ech[0], ech[1], ech[2]),
        (vox[0], vox[1], vox[2]),
        hemi,
        min_vox,
        fragment_mm3,
        crumb_mm3,
        frac,
        sigma,
        bubble_ratio,
    )
    del ech
    x, miss, det = ex.dseg, ex.mask, ex.detected
    lap("excise")
    old, cnt = np.unique(a[miss], return_counts=True)
    _, ncomp = ndi.label(miss)
    ents = entities(aseg)
    dataset_description(out)
    p = dict(
        dseg=derivative_path(out, ents, "dseg", ".nii.gz", desc="excised"),
        mask=derivative_path(out, ents, "mask", ".nii.gz", desc="excisions"),
        bubbles=derivative_path(out, ents, "mask", ".nii.gz", desc="bubbles"),
    )
    p["dseg"].parent.mkdir(parents=True, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(x.dtype)
    nib.save(nib.nifti1.Nifti1Image(x, img.affine, hdr), p["dseg"])
    nib.save(nib.nifti1.Nifti1Image(miss.astype(np.uint8), img.affine), p["mask"])
    nib.save(
        nib.nifti1.Nifti1Image(ex.bubble_mask.astype(np.uint8), img.affine),
        p["bubbles"],
    )
    _write_tsv(
        p["dseg"].with_name(p["dseg"].name.replace(".nii.gz", ".tsv")),
        {k: LUT_NAMES.get(k, f"label-{k}") for k in np.unique(x).tolist()},
    )
    lap("write")
    common = dict(
        SpecVersion=EXCISIONS_SPEC.version,
        **sources(aseg, flash, echo1, echo3),
        Provenance=provenance("baillarger.excisions.write_excisions"),
        Hemisphere=hemi,
        TissueLabels=det.labels,
        Detector=dict(
            MinVoxels=min_vox,
            MediumGapMM=MEDIUM_GAP_MM,
            Sample=SAMPLE,
            Seed=SEED,
            Ridge=RIDGE,
            FragmentMM3=fragment_mm3,
            CrumbMM3=crumb_mm3,
            Frac=frac,
            Sigma=sigma,
            BubbleRatio=bubble_ratio,
            BubbleBox=BUBBLE_BOX,
        ),
        MediumClass=dict(Voxels=det.medium_voxels, **det.medium.record()),
        TissueClasses={str(k): g.record() for k, g in det.classes.items()},
        LabelWeights={
            str(k): dict(
                zip(["medium"] + [str(j) for j in det.labels], w.tolist(), strict=True)
            )
            for k, w in det.weights.items()
        },
        CortexMedian=gm_med,
        ExcisedVoxels=int(miss.sum()),
        ExcisedByLabel={str(int(k)): int(c) for k, c in zip(old, cnt, strict=True)},
        CortexVoxels=int((a == ctx).sum()),
        Components=int(ncomp),
        Fragments=ex.fragments,
        Crumbs=ex.crumbs,
        Bubbles=ex.bubbles,
    )
    write_sidecar(
        sidecar_path(p["dseg"]),
        EXCISIONS_SPEC,
        dict(
            Description="The aseg with its excisions (tissue labels where the "
            "three echoes show the outside medium) and the small tissue pieces "
            "they cut off set to 0 (outside).",
            **common,
        ),
    )
    write_sidecar(
        sidecar_path(p["mask"]),
        EXCISIONS_SPEC,
        dict(
            Description="The excision map: 1 at each voxel that the excised "
            "aseg sets to 0.",
            **common,
        ),
    )
    write_sidecar(
        sidecar_path(p["bubbles"]),
        EXCISIONS_SPEC,
        dict(
            Description="The bubble mask: 1 at each voxel of an air bubble "
            "and its artefact at echo 3. The data there are not reliable.",
            **common,
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(description="the aseg with its excisions set to 0")
    ap.add_argument("aseg", type=Path, help="a BIDS-named aseg (NIfTI)")
    ap.add_argument("flash", type=Path, help="echo 2 of the FLASH, on its grid")
    ap.add_argument("--echo1", required=True, type=Path, help="echo 1 of the FLASH")
    ap.add_argument("--echo3", required=True, type=Path, help="echo 3 of the FLASH")
    ap.add_argument("--hemi", required=True, choices=sorted(TISSUE_LABELS))
    ap.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    a = ap.parse_args(argv)
    paths = write_excisions(a.aseg, a.flash, a.echo1, a.echo3, a.hemi, a.out)
    for role, path in paths.items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
