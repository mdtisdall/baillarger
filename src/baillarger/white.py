"""The handle-free WM label: the aseg's WM label (41) with every handle
removed, so that the white mesh of any box has genus-0 pieces.

The correction is made once, on the whole label (baillarger.handles,
remove_handles), because a box crop of a handle-free solid has planar surface
pieces, while a box alone cannot tell a handle from a fold that closes outside
it. wm_changes computes it (for the example hemisphere: a few minutes and
about 20 GB) and returns only the changed voxels, in full-grid coordinates.
Save them once (WMChanges.save); white_matter_box then gives the corrected
label of any box from the aseg and the saved changes.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

from baillarger.data import ASEG_WM, Box, aseg_path
from baillarger.handles import kuhn_betti, remove_handles

type Bool = npt.NDArray[np.bool_]
type Betti = tuple[int, int, int]

#: kinds of change: WM voxels removed (a cut), voxels added (a fill), WM voxels
#: removed that no front could claim (stranded: each would make a handle)
CUT, FILL, STRANDED = 1, 2, 3
KINDS = {"cut": CUT, "fill": FILL, "stranded": STRANDED}


@dataclass(frozen=True)
class WMChanges:
    """The changes that make the WM label handle-free."""

    #: (n, 3) int32, voxels of the full grid
    voxels: npt.NDArray[np.int32]
    #: (n,) int8, CUT, FILL or STRANDED
    kind: npt.NDArray[np.int8]
    #: (n,) int32, the move (one handle, or the stranded voxels) of each voxel
    move: npt.NDArray[np.int32]
    #: (b0, b1, b2) of the WM label before and after (Kuhn adjacency)
    betti_before: Betti
    betti_after: Betti

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            voxels=self.voxels,
            kind=self.kind,
            move=self.move,
            betti_before=np.array(self.betti_before),
            betti_after=np.array(self.betti_after),
        )

    @classmethod
    def load(cls, path: Path) -> "WMChanges":
        z = np.load(path)
        b0, b1 = z["betti_before"].tolist(), z["betti_after"].tolist()
        return cls(
            voxels=z["voxels"].astype(np.int32),
            kind=z["kind"].astype(np.int8),
            move=z["move"].astype(np.int32),
            betti_before=(b0[0], b0[1], b0[2]),
            betti_after=(b1[0], b1[1], b1[2]),
        )


def _aseg(directory: Path | None) -> nib.nifti1.Nifti1Image:
    img = nib.load(aseg_path(directory))
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg_path(directory).name}: not a NIfTI-1 image")
    return img


def wm_changes(directory: Path | None = None) -> WMChanges:
    """Remove the handles of the aseg's WM label, on its bounding box with a
    margin of 2 voxels, and return the changes."""
    img = _aseg(directory)
    wm = np.asarray(img.dataobj) == ASEG_WM
    nz = [
        np.flatnonzero(wm.any(axis=tuple(j for j in range(3) if j != i)))
        for i in range(3)
    ]
    lo = np.array([max(int(t[0]) - 2, 0) for t in nz])
    hi = np.array([min(int(t[-1]) + 3, s) for t, s in zip(nz, wm.shape, strict=True)])
    x = wm[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    del wm
    before = kuhn_betti(x)
    r = remove_handles(x)
    after = kuhn_betti(r.obj)
    n = [len(m.voxels) for m in r.moves]
    if r.moves:
        voxels = np.concatenate([m.voxels for m in r.moves]) + lo
    else:
        voxels = np.zeros((0, 3), np.int64)
    kind = np.repeat([KINDS[m.kind] for m in r.moves], n)
    move = np.repeat(np.arange(len(r.moves)), n)
    return WMChanges(
        voxels.astype(np.int32),
        kind.astype(np.int8),
        move.astype(np.int32),
        before,
        after,
    )


def white_matter_box(
    lo: Box, hi: Box, changes: WMChanges, directory: Path | None = None
) -> Bool:
    """The handle-free WM label of the box [lo, hi) of the full grid (bool).
    Raises ValueError if the box is not inside the grid."""
    img = _aseg(directory)
    if not all(0 <= a < b <= n for a, b, n in zip(lo, hi, img.shape, strict=True)):
        raise ValueError(f"box {lo}-{hi} is not inside the grid {img.shape}")
    sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    wm: Bool = np.asarray(img.dataobj[sl]) == ASEG_WM
    p = changes.voxels.astype(np.int64) - np.asarray(lo)
    inside = np.all((p >= 0) & (p < np.subtract(hi, lo)), axis=1)
    p, k = p[inside], changes.kind[inside]
    wm[tuple(p[k == FILL].T)] = True
    wm[tuple(p[k != FILL].T)] = False
    return wm
