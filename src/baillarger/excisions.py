"""Excisions to outside: a bootstrap tool for an aseg that imputes tissue.

An excision is a block of tissue that was cut out of the specimen before the
scan. The FLASH is black there, but an aseg that imputes tissue (such as
purple-mri's) labels it WM or cortex. detect finds these voxels, and excise
sets them to 0 (outside), with the small tissue pieces they cut off, so
that the white surface closes there (the rule for cut faces,
docs/formats/wm-dseg.md). Use it only on such an aseg: the
Fomblin outside the specimen is as dark as an excision. Once our own labels
exist, this module has nothing to do.

The command line:

    python -m baillarger.excisions ASEG FLASH --echo1 E1 --echo3 E3 \\
        --hemi H --out DERIVATIVES

FLASH is echo 2 on the aseg's grid. E1 and E3 are echoes 1 and 3. They tell
air bubbles from excisions. The command writes the excised aseg
(desc-excised_dseg), the input label of baillarger.white_dseg, and the
excision map (desc-excisions_mask), with their sidecars
(docs/formats/wm-excisions.md).
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
from baillarger.timing import Laps, log_to_stderr
from baillarger.white_dseg import LUT_NAMES

type Bool = npt.NDArray[np.bool_]
type Labels = npt.NDArray[np.integer[Any]]

#: the WM label and the cortex label, by hemisphere (FreeSurfer LUT)
TISSUE_LABELS = {"R": (41, 42), "L": (2, 3)}

#: the detector's defaults
FRAC, SIGMA, MIN_VOX, MARGIN_MM = 0.25, 2.0, 250, 0.3

#: excision-map pieces below FRAGMENT_MM3 are dropped; tissue pieces below
#: CRUMB_MM3 that touch the excision map are set to 0
FRAGMENT_MM3, CRUMB_MM3 = 0.1, 1.0

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
        Key("TissueLabels", "array of integers", True, "WM and cortex labels"),
        Key("Detector", "object", True, "the detector's parameters"),
        Key("CortexMedian", "number", True, "median FLASH over the cortex"),
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
    """The steps of binary dilation that grow the excisions by margin_mm."""
    return int(round(margin_mm / min(voxel_mm)))


def cortex_median(aseg: npt.ArrayLike, flash: npt.ArrayLike, hemi: str) -> float:
    """The median of the FLASH over the cortex label of hemi: the detector's
    reference level. Raises ValueError if the aseg has no voxel with the WM
    label or the cortex label of hemi."""
    a = np.asarray(aseg)
    wm, ctx = TISSUE_LABELS[hemi]
    for lab, name in ((wm, "WM"), (ctx, "cortex")):
        if not np.any(a == lab):
            raise ValueError(
                f"no voxel with the {name} label of hemisphere {hemi} ({lab})"
            )
    return float(np.median(np.asarray(flash, np.float32)[a == ctx]))


def detect(
    aseg: npt.ArrayLike,
    flash: npt.ArrayLike,
    voxel_mm: tuple[float, float, float],
    hemi: str,
    frac: float = FRAC,
    sigma: float = SIGMA,
    min_vox: int = MIN_VOX,
    margin_mm: float = MARGIN_MM,
) -> Bool:
    """The excisions: tissue (the WM or cortex label of hemi) where the
    FLASH is black. The FLASH smoothed by sigma voxels is below frac x the
    median FLASH over the cortex label, in connected blocks (face adjacency)
    of at least min_vox voxels, grown by margin_mm and kept in the tissue.
    Computed in float32. Raises ValueError if the aseg has no voxel with
    the WM label or the cortex label of hemi."""
    a = np.asarray(aseg)
    cortex_median(a, flash, hemi)  # refuses a hemisphere without its labels
    wm, ctx = TISSUE_LABELS[hemi]
    return dark_tissue(
        (a == wm) | (a == ctx),
        a == ctx,
        flash,
        voxel_mm,
        frac,
        sigma,
        min_vox,
        margin_mm,
    )


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
    """detect on masks of the tissue and the cortex, with no check of the
    labels. Without cortex the reference level is not defined and the
    result is empty."""
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
    voxels: the growth of the artefact from echo 1 to echo 3."""
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
        if v3 < ratio * v1:
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
    that excise set to 0), the bubble mask, and the counts for the
    sidecar."""

    dseg: Labels
    mask: Bool
    bubble_mask: Bool
    fragments: dict[str, int]
    crumbs: dict[str, int]
    bubbles: dict[str, int]


def excise(
    aseg: npt.ArrayLike,
    flash: npt.ArrayLike,
    voxel_mm: tuple[float, float, float],
    hemi: str,
    frac: float = FRAC,
    sigma: float = SIGMA,
    min_vox: int = MIN_VOX,
    margin_mm: float = MARGIN_MM,
    fragment_mm3: float = FRAGMENT_MM3,
    crumb_mm3: float = CRUMB_MM3,
    echo1: npt.ArrayLike | None = None,
    echo3: npt.ArrayLike | None = None,
    bubble_ratio: float = BUBBLE_RATIO,
) -> Excised:
    """The excisions set to outside (docs/formats/wm-excisions.md). flash is
    echo 2:
    1. detect;
    2. drop the pieces of the result of less than fragment_mm3;
    2b. with echo1 and echo3: drop the pieces that are air bubbles (bubbles,
       at frac x the median of each echo over the cortex label), and make
       the bubble mask;
    3. relabel (set the rest to 0);
    4. set to 0 each tissue piece (WM or cortex label of hemi, face
       adjacency) of less than crumb_mm3 that touches the map of step 3.
    The excision map is the voxels of steps 3 and 4. Raises ValueError if
    the aseg has no voxel with the WM label or the cortex label of hemi."""
    a = np.asarray(aseg)
    miss = detect(a, flash, voxel_mm, hemi, frac, sigma, min_vox, margin_mm)
    miss, nfrag, vfrag = drop_fragments(miss, voxel_mm, fragment_mm3)
    wm, ctx = TISSUE_LABELS[hemi]
    bub = np.zeros(miss.shape, bool)
    bmask = np.zeros(miss.shape, bool)
    nbub = 0
    if echo1 is not None and echo3 is not None:
        echoes = (np.asarray(echo1), np.asarray(flash), np.asarray(echo3))
        cortex = a == ctx
        lv = tuple(frac * float(np.median(e[cortex])) for e in echoes)
        bub, bmask, nbub = bubbles(
            miss, echoes, (lv[0], lv[1], lv[2]), sigma, bubble_ratio
        )
        miss &= ~bub
    x = relabel(a, miss)
    crumb, ncrumb = cut_off((x == wm) | (x == ctx), miss, voxel_mm, crumb_mm3)
    x[crumb] = 0
    return Excised(
        x,
        miss | crumb,
        bmask,
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
    frac: float = FRAC,
    sigma: float = SIGMA,
    min_vox: int = MIN_VOX,
    margin_mm: float = MARGIN_MM,
    fragment_mm3: float = FRAGMENT_MM3,
    crumb_mm3: float = CRUMB_MM3,
    bubble_ratio: float = BUBBLE_RATIO,
) -> dict[str, Path]:
    """The command: excise the aseg at aseg with the FLASH at flash (echo
    2) and its echoes 1 and 3 (for the air bubbles), written to the BIDS
    derivatives root out (docs/formats/wm-excisions.md). Raises ValueError
    if a FLASH grid has a different shape. Returns the paths written, by
    role."""
    lap = Laps()
    img = nib.load(aseg)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg.name}: not a NIfTI-1 image")
    echoes = []
    for path in (flash, echo1, echo3):
        e = nib.load(path)
        if not isinstance(e, nib.nifti1.Nifti1Image):
            raise ValueError(f"{path.name}: not a NIfTI-1 image")
        if img.shape[:3] != e.shape[:3]:
            raise ValueError(
                f"{path.name}: grid {e.shape[:3]}, not the aseg's {img.shape[:3]}"
            )
        echoes.append(e)
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    a = np.asarray(img.dataobj)
    f = np.asarray(echoes[0].dataobj, np.float32)
    f1 = np.asarray(echoes[1].dataobj)
    f3 = np.asarray(echoes[2].dataobj)
    lap("read")
    wm, ctx = TISSUE_LABELS[hemi]
    gm_med = cortex_median(a, f, hemi)
    ex = excise(
        a,
        f,
        (vox[0], vox[1], vox[2]),
        hemi,
        frac,
        sigma,
        min_vox,
        margin_mm,
        fragment_mm3,
        crumb_mm3,
        f1,
        f3,
        bubble_ratio,
    )
    del f, f1, f3
    x, miss = ex.dseg, ex.mask
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
        TissueLabels=[wm, ctx],
        Detector=dict(
            Frac=frac,
            Sigma=sigma,
            MinVoxels=min_vox,
            MarginMM=margin_mm,
            DilationSteps=dilation_steps(margin_mm, (vox[0], vox[1], vox[2])),
            FragmentMM3=fragment_mm3,
            CrumbMM3=crumb_mm3,
            BubbleRatio=bubble_ratio,
            BubbleBox=BUBBLE_BOX,
        ),
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
            Description="The aseg with its excisions (tissue where the FLASH "
            "is black) and the small tissue pieces they cut off set to 0 "
            "(outside).",
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
