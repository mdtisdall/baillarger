"""The example data: patch crops of the three FLASH echoes and the aseg, and
the rim labels.

One ex vivo hemisphere (DATA.md): three FLASH magnitude echoes and the PURPLE
segmentation (aseg), all on one grid of 700 x 384 x 1148 voxels, int16. The
files are in data/ at the top of the checkout (git-ignored), or in the
directory that the environment variable BAILLARGER_DATA names.

A patch is a box of voxels (lo inclusive, hi exclusive) round a set of areal
boundary points. load_patch reads only the box from each file, checks that
the four images share one grid, and makes the rim labels: 1 outside (aseg 0),
2 WM (41), 3 cortex (42), 0 everything else. The voxel size comes from the
header: 0.16028 x 0.16 x 0.16028 mm, not quite isotropic.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

type Box = tuple[int, int, int]

SUBJECT = "sub-INDD121850_ses-7THemix20250516"
_ECHO = SUBJECT + (
    "_acq-channelCOMBx160um_dir-positive_run-02_echo-{}_part-mag_FLASH.nii.gz"
)
_ASEG = SUBJECT + "_aseg.nii.gz"

ASEG_OUTSIDE, ASEG_WM, ASEG_CORTEX = 0, 41, 42
RIM_OTHER, RIM_OUTSIDE, RIM_WM, RIM_CORTEX = 0, 1, 2, 3

#: Patch boxes (lo, hi) in voxels of the full grid: the boundary points plus
#: a margin of 10 mm (12 mm for M1S1) on every side.
PATCHES: dict[str, tuple[Box, Box]] = {
    "V1V2": ((214, 37, 105), (354, 187, 350)),
    "PCC_SPL": ((452, 118, 360), (577, 243, 485)),
    "M1S1": ((380, 175, 421), (531, 326, 572)),
}

#: Review points in V1V2 crop coordinates, for the seams in fused cortex.
V1V2_REVIEW_POINTS: dict[str, tuple[Box, ...]] = {
    "collapsed sulcus with a faint dark line": ((69, 65, 69), (68, 60, 71)),
    "thin space between kissing cortex, bridge over a CSF tunnel": ((81, 49, 162),),
    "open sulcus wall (an earlier false slit)": ((61, 94, 78),),
}


@dataclass(frozen=True)
class Patch:
    """One patch crop."""

    name: str
    lo: Box
    hi: Box
    #: the three echoes, float32, shape (*shape, 3): the vector x(v) per voxel
    echoes: npt.NDArray[np.float32]
    #: echo times (ms) of the three echoes
    echo_times_ms: tuple[float, float, float]
    #: the labels: the aseg's, or those of load_box's labels
    aseg: npt.NDArray[np.int16]
    rim: npt.NDArray[np.int16]
    #: voxel to RAS (mm) of the crop: the full grid's affine moved to lo
    affine: npt.NDArray[np.float64]
    voxel_mm: tuple[float, float, float]

    @property
    def flash(self) -> npt.NDArray[np.float32]:
        """Echo 2, the FLASH of the single-echo methods."""
        return self.echoes[..., 1]

    @property
    def shape(self) -> Box:
        s = self.rim.shape
        return (s[0], s[1], s[2])


def data_dir() -> Path:
    """$BAILLARGER_DATA, else data/ at the top of the checkout."""
    env = os.environ.get("BAILLARGER_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def echo_path(k: int, directory: Path | None = None) -> Path:
    """Path of echo k (1, 2 or 3)."""
    return (directory or data_dir()) / _ECHO.format(k)


def aseg_path(directory: Path | None = None) -> Path:
    return (directory or data_dir()) / _ASEG


def data_available(directory: Path | None = None) -> bool:
    """Whether the three echoes and the aseg are present."""
    paths = [echo_path(k, directory) for k in (1, 2, 3)] + [aseg_path(directory)]
    return all(p.is_file() for p in paths)


def rim_labels(aseg: npt.ArrayLike) -> npt.NDArray[np.int16]:
    """Rim labels from the aseg: 1 outside (0), 2 WM (41), 3 cortex (42), 0
    everything else."""
    a = np.asarray(aseg)
    rim = np.zeros(a.shape, np.int16)
    rim[a == ASEG_OUTSIDE] = RIM_OUTSIDE
    rim[a == ASEG_WM] = RIM_WM
    rim[a == ASEG_CORTEX] = RIM_CORTEX
    return rim


def crop_affine(affine: npt.ArrayLike, lo: Box) -> npt.NDArray[np.float64]:
    """The affine of a crop whose voxel (0, 0, 0) is voxel lo of the full grid."""
    shift = np.eye(4)
    shift[:3, 3] = lo
    out: npt.NDArray[np.float64] = np.asarray(affine, np.float64) @ shift
    return out


def _echo_time_ms(path: Path) -> float:
    """EchoTime of the BIDS sidecar next to path, in ms."""
    side = path.with_name(path.name.removesuffix(".nii.gz") + ".json")
    return float(json.loads(side.read_text())["EchoTime"]) * 1000.0


def _load(path: Path) -> nib.nifti1.Nifti1Image:
    img = nib.load(path)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{path.name}: not a NIfTI-1 image")
    return img


def load_box(
    lo: Box,
    hi: Box,
    name: str = "box",
    directory: Path | None = None,
    labels: Path | None = None,
) -> Patch:
    """Read the box [lo, hi) of the three echoes and the aseg, and make the
    rim labels. labels: a segmentation to read in place of the aseg (the
    corrected segmentation, baillarger.white.write_wm_topo). Raises
    ValueError if the box is not inside the grid or the images do not share
    one grid."""
    seg = aseg_path(directory) if labels is None else labels
    paths = [echo_path(k, directory) for k in (1, 2, 3)] + [seg]
    imgs = [_load(p) for p in paths]
    ref = imgs[0]
    for p, im in zip(paths[1:], imgs[1:], strict=True):
        if im.shape != ref.shape or not np.allclose(im.affine, ref.affine):
            raise ValueError(f"{p.name}: not on the grid of {paths[0].name}")
    if not all(0 <= a < b <= n for a, b, n in zip(lo, hi, ref.shape, strict=True)):
        raise ValueError(f"box {lo}-{hi} is not inside the grid {ref.shape}")
    sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    echoes = np.stack(
        [np.asarray(im.dataobj[sl], np.float32) for im in imgs[:3]], axis=-1
    )
    raw = np.asarray(imgs[3].dataobj[sl])
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{paths[3].name}: labels are not integers ({raw.dtype})")
    aseg = raw.astype(np.int16)
    tes = tuple(_echo_time_ms(p) for p in paths[:3])
    zooms = ref.header.get_zooms()[:3]
    return Patch(
        name=name,
        lo=lo,
        hi=hi,
        echoes=echoes,
        echo_times_ms=(tes[0], tes[1], tes[2]),
        aseg=aseg,
        rim=rim_labels(aseg),
        affine=crop_affine(ref.affine, lo),
        voxel_mm=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
    )


def load_patch(name: str, directory: Path | None = None) -> Patch:
    """Read one of PATCHES (V1V2, PCC_SPL, M1S1)."""
    lo, hi = PATCHES[name]
    return load_box(lo, hi, name, directory)
