"""The review of pial-side label proposals (baillarger.pial_labels) in the viewer: the
ranked review cells, accept / reject / back to proposed, a brush that sets the voxels
of a ball round a world point to fluid or back to cortex, and the overlay on the box
grid. Every change is saved at once to the change file it was loaded from (the first
change keeps the file as it was in <stem>.orig.npz), so the column traces
(column_traces.load_tissue(..., labels=...)) read the reviewed labels.
"""

import gzip
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt

from baillarger import pial_labels as PL
from baillarger.data import RIM_CORTEX, RIM_OUTSIDE

type Float = npt.NDArray[np.float64]

STATUS = {name: k for k, name in enumerate(PL.STATUS)}
BRUSH = {"fluid": RIM_OUTSIDE, "cortex": RIM_CORTEX}
BRUSH_MAX = 3  # largest brush radius (voxels)


class LabelReview:
    """A change file under review on a box: lo (full-grid corner), shape, and the box's
    world affine (the volume pane's)."""

    def __init__(
        self, path: Path, lo: npt.ArrayLike, shape: tuple[int, ...], affine: Float
    ) -> None:
        self.path = path
        self.ch = PL.LabelChanges.load(path)
        self.lo = np.asarray(lo, np.int64)
        self.shape = shape
        self.affine = affine
        if not np.array_equal(np.asarray(self.ch.lo), self.lo):
            raise ValueError(
                f"{path}: box corner {self.ch.lo} is not {self.lo.tolist()}"
            )
        self.seq = 0

    def world(self, box0: npt.ArrayLike) -> list[float]:
        p = np.asarray(box0, np.float64)
        return [round(float(x), 3) for x in (self.affine @ np.r_[p, 1.0])[:3]]

    def rows(self) -> dict[str, Any]:
        """The review cells in rank order (with status and world mm), the counts by
        status, the brush voxels and the detection settings."""
        st = self.ch.status
        rows = sorted(self.ch.components, key=lambda r: r["rank"])
        out = [
            dict(
                id=r["id"],
                rank=r["rank"],
                status=PL.STATUS[int(st[r["id"]])],
                auto=r["auto"],
                kind=r["kind"],
                voxels=r["voxels"],
                columns=r["columns"],
                crest=r["crest_hits"],
                touches=r["touches_fluid"],
                frac=r["frac_median"],
                betti=[r["betti_local_before"], r["betti_local_after"]],
                itksnap=r["at_itksnap"],
                world_mm=self.world(r["at_box0"]),
            )
            for r in rows
        ]
        counts = {s: int((st == k).sum()) for s, k in STATUS.items()}
        return dict(
            rows=out,
            counts=counts,
            brush=int((self.ch.component < 0).sum()),
            info=self.ch.info,
            seq=self.seq,
        )

    def save(self) -> None:
        orig = self.path.with_name(self.path.stem + ".orig.npz")
        if not orig.exists():
            shutil.copyfile(self.path, orig)
        self.ch.save(self.path)
        self.seq += 1

    def decide(self, cell: int, status: str) -> dict[str, Any]:
        if not 0 <= cell < len(self.ch.status):
            raise ValueError(f"no review cell {cell}")
        self.ch = self.ch.with_status(cell, STATUS[status])
        self.save()
        return dict(id=cell, status=status, seq=self.seq)

    def brush(self, mm: list[float], radius: int, label: str) -> dict[str, Any]:
        """Set the box voxels within radius voxels of the voxel nearest a world point
        to label (fluid or cortex)."""
        if not 0 <= radius <= BRUSH_MAX:
            raise ValueError(f"brush radius {radius}: 0 to {BRUSH_MAX} voxels")
        c = np.round(np.linalg.solve(self.affine, np.r_[mm, 1.0])[:3]).astype(int)
        r = np.arange(-radius, radius + 1)
        off = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        p = c + off[np.sum(off**2, 1) <= radius**2 + 1e-9]
        p = p[np.all((p >= 0) & (p < np.array(self.shape)), 1)]
        self.ch = self.ch.with_brush(p + self.lo, BRUSH[label])
        self.save()
        return dict(voxels=len(p), centre_box0=c.tolist(), seq=self.seq)

    def overlay_bytes(self) -> bytes:
        """The overlay (pial_labels.overlay) on the box grid, .nii.gz, world affine."""
        ov = PL.overlay(self.ch).astype(np.uint8)
        img = nib.Nifti1Image(ov, self.affine)
        return gzip.compress(img.to_bytes(), compresslevel=1)
