"""Excisions to outside: a bootstrap tool for an aseg that imputes tissue.

An excision is a block of tissue that was cut out of the specimen before the
scan. The FLASH is black there, but an aseg that imputes tissue (such as
purple-mri's) labels it WM or cortex. detect finds these voxels and relabel
sets them to 0 (outside), so that the white surface closes there (the rule
for cut faces, docs/formats/wm-dseg.md). Use it only on such an aseg: the
Fomblin outside the specimen is as dark as an excision. Once our own labels
exist, this module has nothing to do.

The command line:

    python -m baillarger.excisions ASEG FLASH --hemi H --out DERIVATIVES

FLASH is echo 2 on the aseg's grid. The command writes the excised aseg
(desc-excised_dseg), the input label of baillarger.white_dseg, and the
excision map (desc-excisions_mask), with their sidecars
(docs/formats/wm-excisions.md).
"""

import argparse
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
    hemi: str,
    out: Path,
    frac: float = FRAC,
    sigma: float = SIGMA,
    min_vox: int = MIN_VOX,
    margin_mm: float = MARGIN_MM,
) -> dict[str, Path]:
    """The command: detect on the aseg at aseg and the FLASH at flash (echo
    2), then relabel, written to the BIDS derivatives root out
    (docs/formats/wm-excisions.md). Raises ValueError if the two grids have
    different shapes. Returns the paths written, by role."""
    lap = Laps()
    img = nib.load(aseg)
    fimg = nib.load(flash)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg.name}: not a NIfTI-1 image")
    if not isinstance(fimg, nib.nifti1.Nifti1Image):
        raise ValueError(f"{flash.name}: not a NIfTI-1 image")
    if img.shape[:3] != fimg.shape[:3]:
        raise ValueError(
            f"{flash.name}: grid {fimg.shape[:3]}, not the aseg's {img.shape[:3]}"
        )
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    a = np.asarray(img.dataobj)
    f = np.asarray(fimg.dataobj, np.float32)
    lap("read")
    wm, ctx = TISSUE_LABELS[hemi]
    gm_med = cortex_median(a, f, hemi)
    miss = detect(a, f, (vox[0], vox[1], vox[2]), hemi, frac, sigma, min_vox, margin_mm)
    del f
    lap("detect")
    old, cnt = np.unique(a[miss], return_counts=True)
    _, ncomp = ndi.label(miss)
    x = relabel(a, miss)
    lap("relabel")
    ents = entities(aseg)
    dataset_description(out)
    p = dict(
        dseg=derivative_path(out, ents, "dseg", ".nii.gz", desc="excised"),
        mask=derivative_path(out, ents, "mask", ".nii.gz", desc="excisions"),
    )
    p["dseg"].parent.mkdir(parents=True, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(x.dtype)
    nib.save(nib.nifti1.Nifti1Image(x, img.affine, hdr), p["dseg"])
    nib.save(nib.nifti1.Nifti1Image(miss.astype(np.uint8), img.affine), p["mask"])
    _write_tsv(
        p["dseg"].with_name(p["dseg"].name.replace(".nii.gz", ".tsv")),
        {k: LUT_NAMES.get(k, f"label-{k}") for k in np.unique(x).tolist()},
    )
    lap("write")
    common = dict(
        SpecVersion=EXCISIONS_SPEC.version,
        **sources(aseg, flash),
        Provenance=provenance("baillarger.excisions.write_excisions"),
        Hemisphere=hemi,
        TissueLabels=[wm, ctx],
        Detector=dict(
            Frac=frac,
            Sigma=sigma,
            MinVoxels=min_vox,
            MarginMM=margin_mm,
            DilationSteps=dilation_steps(margin_mm, (vox[0], vox[1], vox[2])),
        ),
        CortexMedian=gm_med,
        ExcisedVoxels=int(miss.sum()),
        ExcisedByLabel={str(int(k)): int(c) for k, c in zip(old, cnt, strict=True)},
        CortexVoxels=int((a == ctx).sum()),
        Components=int(ncomp),
    )
    write_sidecar(
        sidecar_path(p["dseg"]),
        EXCISIONS_SPEC,
        dict(
            Description="The aseg with its excisions (tissue where the FLASH "
            "is black) set to 0 (outside).",
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
    return p


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(description="the aseg with its excisions set to 0")
    ap.add_argument("aseg", type=Path, help="a BIDS-named aseg (NIfTI)")
    ap.add_argument("flash", type=Path, help="echo 2 of the FLASH, on its grid")
    ap.add_argument("--hemi", required=True, choices=sorted(TISSUE_LABELS))
    ap.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    a = ap.parse_args(argv)
    for role, path in write_excisions(a.aseg, a.flash, a.hemi, a.out).items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
