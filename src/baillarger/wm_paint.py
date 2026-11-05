"""A crop of the WM labels for hand painting in ITK-SNAP (docs/formats/wm-paint.md).

export cuts a cube of the segmentation round a review site. It writes the
crop of the labels, an unchanged copy of it, and the crop of each echo.
The reviewer paints the labels in ITK-SNAP and saves them in place.

import compares the painted labels with the unchanged copy. It appends one
answer, of the choice manual-paint, to the answer log (wm_edits). The answer
has the voxels that changed and their new labels.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    file_sha256,
    provenance,
    read_sidecar,
    sources,
    write_sidecar,
)
from baillarger.wm_choices import box_round
from baillarger.wm_edits import EditLog

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
    mask = painted != exported
    pts = np.argwhere(mask).astype(np.int64).reshape(-1, 3)
    labels = painted[mask].astype(np.int64).reshape(-1)
    offset = np.asarray(side["Offset"], np.int64)
    return (pts + offset).astype(np.int64), labels


def import_paint(
    crop: Path, log_path: Path, session: int, reviewer_seconds: float
) -> int:
    """Append the changes of the crop as one manual-paint answer to the log at
    log_path, made in `session` at the crop's site. Raises ValueError if the
    log's grid is not the crop's grid. Returns the answer's Id."""
    side = read_sidecar(crop / SIDECAR, PAINT_SPEC)
    log = EditLog(log_path)
    if log.read().shape != tuple(side["Shape"]):
        raise ValueError(f"the log's grid is not the crop's grid {side['Shape']}")
    voxels, labels = changes(crop)
    return log.answer(
        session,
        [int(c) for c in side["Site"]],
        "manual-paint",
        voxels,
        labels,
        decided_against=side["DsegSHA256"],
        reviewer_seconds=reviewer_seconds,
        params={"Crop": str(crop.resolve())},
    )


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
    im = sub.add_parser("import", help="append the painted change to the log")
    im.add_argument("crop", type=Path)
    im.add_argument("--log", type=Path, required=True)
    im.add_argument("--session", type=int, required=True)
    im.add_argument("--seconds", type=float, required=True)
    a = ap.parse_args(argv)
    if a.command == "export":
        site = (int(a.site[0]), int(a.site[1]), int(a.site[2]))
        export(a.dseg, site, a.out, a.echo, a.half)
    else:
        print(import_paint(a.crop, a.log, a.session, a.seconds))


if __name__ == "__main__":
    main()
