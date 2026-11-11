"""A crop of the WM labels for hand painting in ITK-SNAP (docs/formats/wm-paint.md).

export cuts a cube of the segmentation round a review site. It writes the
crop of the labels, an unchanged copy of it, and the crop of each echo.
The reviewer paints the labels in ITK-SNAP and saves them in place.

import compares the painted labels with the unchanged copy
(baillarger.label_edits.changed). It appends one answer, of the choice
manual-paint, to the answer log (wm_edits): the voxels that changed and
their new labels. Then it writes the voxels whose label changes into the
label edits (baillarger.label_edits.paint, with the input labels of the
dseg: the first of its sidecar's Sources).
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

from baillarger import label_edits
from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    file_sha256,
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.wm_choices import box_round
from baillarger.wm_edits import EDITS_SPEC, EditLog

type Int = npt.NDArray[np.int64]

#: docs/formats/wm-paint.md
PAINT_SPEC = SidecarSpec(
    "wm-paint/1",
    COMMON_KEYS
    + (
        Key("Site", "array of integers", True, "the site [i, j, k] on the grid"),
        Key(
            "Offset",
            "array of integers",
            True,
            "the grid voxel of the crop's voxel [0, 0, 0]",
        ),
        Key("Shape", "array of integers", True, "the grid's shape"),
        Key(
            "DsegSHA256",
            "string",
            True,
            "SHA-256 of the desc-wmtopo segmentation that was cropped",
        ),
    ),
)

#: the crop of the labels: the reviewer paints this file
LABELS = "labels.nii.gz"
#: an unchanged copy of the crop of the labels: not to edit
EXPORTED = "exported.nii.gz"
#: the sidecar of the crop
SIDECAR = "crop.json"


def _nifti(path: Path) -> nib.Nifti1Image:
    img = nib.load(path)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{path.name}: not a NIfTI-1 image")
    return img


def _crop(img: nib.Nifti1Image, box: tuple[slice, ...]) -> nib.Nifti1Image:
    out: nib.Nifti1Image = img.slicer[box]
    return out


def export(
    dseg: Path,
    site: tuple[int, int, int],
    out: Path,
    echoes: Sequence[Path] = (),
    half: int = 32,
) -> Path:
    """Write the crop of the cube of half-width `half` round `site` to the
    directory `out`: LABELS, EXPORTED, echo-<n>.nii.gz for each echo (n = 1,
    2, ... in the order given) and SIDECAR. Each echo must be on the grid of
    dseg. Raises FileExistsError if LABELS is there already. Returns out."""
    img = _nifti(dseg)
    shape = tuple(int(s) for s in img.shape[:3])
    if (out / LABELS).exists():
        raise FileExistsError(f"{LABELS}: a crop is there already")
    echo_imgs = []
    for path in echoes:
        e = _nifti(path)
        if e.shape[:3] != img.shape[:3] or not np.allclose(e.affine, img.affine):
            raise ValueError(f"{path.name}: not on the grid of {dseg.name}")
        echo_imgs.append(e)
    box = box_round(shape, site, half)
    out.mkdir(parents=True, exist_ok=True)
    nib.save(_crop(img, box), out / LABELS)
    nib.save(_crop(img, box), out / EXPORTED)
    for k, e in enumerate(echo_imgs, 1):
        nib.save(_crop(e, box), out / f"echo-{k}.nii.gz")
    write_sidecar(
        out / SIDECAR,
        PAINT_SPEC,
        dict(
            SpecVersion=PAINT_SPEC.version,
            Description="A crop of the WM labels for hand painting in ITK-SNAP.",
            **sources(dseg, *echoes),
            Provenance=provenance("baillarger.wm_paint.export"),
            Site=[int(c) for c in site],
            Offset=[int(s.start) for s in box],
            Shape=list(shape),
            DsegSHA256=file_sha256(dseg),
        ),
    )
    return out


def changes(crop: Path) -> tuple[Int, Int]:
    """The voxels where the painted labels differ from the unchanged copy, in
    grid coordinates (C order), and their painted labels. Raises ValueError
    if the two files differ in shape or affine."""
    side = read_sidecar(crop / SIDECAR, PAINT_SPEC)
    painted_img = _nifti(crop / LABELS)
    exported_img = _nifti(crop / EXPORTED)
    if painted_img.shape != exported_img.shape or not np.allclose(
        painted_img.affine, exported_img.affine
    ):
        raise ValueError(f"{LABELS} and {EXPORTED} differ in shape or affine")
    painted = np.asanyarray(painted_img.dataobj)
    exported = np.asanyarray(exported_img.dataobj)
    return label_edits.changed(exported, painted, side["Offset"])


def import_paint(
    crop: Path,
    log_path: Path,
    edits_path: Path,
    session: int,
    reviewer_seconds: float,
) -> int:
    """Append the changes of the crop as one manual-paint answer to the log at
    log_path, made in `session` at the crop's site, then write them into the
    label edits at edits_path (a .npy diff, made empty if it is not there):
    only the voxels whose label changes from the input labels with the diff
    applied. The input labels are the first of the cropped dseg's Sources.
    Raises ValueError if the log's or the diff's grid is not the crop's grid.
    Returns the answer's Id."""
    side = read_sidecar(crop / SIDECAR, PAINT_SPEC)
    grid = tuple(int(n) for n in side["Shape"])
    log = EditLog(log_path)
    if log.read().shape != grid:
        raise ValueError(f"the log's grid is not the crop's grid {side['Shape']}")
    dseg = Path(side["Sources"][0])
    dimg = _nifti(dseg)
    hemi = str(read_sidecar(log.paths["sidecar"], EDITS_SPEC)["Hemisphere"])
    if edits_path.exists():
        edits, eside, _ = label_edits.read_with_sidecar(edits_path)
        if tuple(eside["Shape"]) != grid:
            raise ValueError(f"the diff's grid is not the crop's grid {grid}")
    else:
        edits = label_edits.empty()
    inp = _nifti(Path(_dseg_sources(dseg)[0]))
    if tuple(inp.shape[:3]) != grid:
        raise ValueError(f"the input labels are not on the crop's grid {grid}")
    voxels, labels = changes(crop)
    off = np.asarray(side["Offset"], np.int64)
    size = np.asarray(_nifti(crop / LABELS).shape[:3], np.int64)
    box = tuple(slice(int(o), int(o + n)) for o, n in zip(off, size, strict=True))
    new, _ = label_edits.paint(
        edits,
        np.asanyarray(inp.dataobj[box]),
        voxels,
        labels,
        offset=off,
        shape=grid,
    )
    aid = log.answer(
        session,
        [int(c) for c in side["Site"]],
        "manual-paint",
        voxels,
        labels,
        decided_against=side["DsegSHA256"],
        reviewer_seconds=reviewer_seconds,
        params={"Crop": str(crop.resolve())},
    )
    label_edits.write(edits_path, new, hemisphere=hemi, shape=grid, affine=dimg.affine)
    return aid


def _dseg_sources(dseg: Path) -> list[str]:
    """The Sources of the dseg's sidecar (wm-dseg/1): the first is the input
    labels."""
    from baillarger.white_dseg import DSEG_SPEC

    return list(read_sidecar(sidecar_path(dseg), DSEG_SPEC)["Sources"])


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m baillarger.wm_paint",
        description="A crop of the WM labels for hand painting in ITK-SNAP.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="write a crop round a site")
    ex.add_argument("dseg", type=Path)
    ex.add_argument("--site", nargs=3, type=int, required=True, metavar=("I", "J", "K"))
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--echo", type=Path, action="append", default=[])
    ex.add_argument("--half", type=int, default=32)
    im = sub.add_parser(
        "import", help="append the painted change to the log and the label edits"
    )
    im.add_argument("crop", type=Path)
    im.add_argument("--log", type=Path, required=True)
    im.add_argument("--edits", type=Path, required=True, help="the label edits (.npy)")
    im.add_argument("--session", type=int, required=True)
    im.add_argument("--seconds", type=float, required=True)
    a = ap.parse_args(argv)
    if a.command == "export":
        site = (int(a.site[0]), int(a.site[1]), int(a.site[2]))
        export(a.dseg, site, a.out, a.echo, a.half)
    else:
        print(import_paint(a.crop, a.log, a.edits, a.session, a.seconds))


if __name__ == "__main__":
    main()
