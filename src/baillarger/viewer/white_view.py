"""A white surface from the mesh command, viewed alone (no column volume): its overview
(baillarger.viewer.overview) in the 3D surface pane, and the WM selection on it, with
the true vertex that each overview vertex is; and the volume pane on a box round a
voxel (BOX voxels each way): the echoes, the true surface's faces there (from its
blocks), and an overlay of the WM mask and the topology fixes of its corrected
segmentation (OVERLAY: 1 mask, 2 handle cut, 3 handle fill, 4 cavity fill). The
page moves the box to the WM selection.

Run: python -m baillarger.viewer --white <..._white.surf.gii> --cache DIR (the
overview is written by the mesh command, or by python -m baillarger.white overview)
"""

import gzip
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiImage

from baillarger.blocks import NiftiSource, open_volume
from baillarger.data import crop_affine, echo_path
from baillarger.viewer.overview import overview_surface, surface_name
from baillarger.viewer.surface import gifti_bytes

type Float = npt.NDArray[np.float64]

BOX = 48  # voxels each way from the box's centre
#: the overlay's values
OVERLAY = {1: "WM mask", 2: "handle cut", 3: "handle fill", 4: "cavity fill"}


class WhiteView:
    """The overview's Surface (S), the overview (ov) and the true surface (W) of the
    surface at surf; see the module docstring."""

    def __init__(
        self,
        surf: Path,
        cache: Path,
        log: Callable[[str], object] = print,
        echo_dir: Path | None = None,
        box: int = BOX,
    ) -> None:
        self.surf = Path(surf)
        self.name = surface_name(self.surf)
        self.S, self.ov, self.W = overview_surface(self.surf, cache, log)
        self.no_trace = -np.ones(len(self.S.v_vox), np.int32)
        side = self.W.sidecar
        self.grid = tuple(int(x) for x in side["GridShape"])
        self.half = int(box)
        self.echoes = {k: echo_path(k, echo_dir) for k in (1, 2, 3)}
        self.dseg = Path(side["Sources"][0])
        self.fixes = self.dseg.with_name(
            self.dseg.name.replace("desc-wmtopo_", "desc-wmtopofixes_")
        )
        self.filled = tuple(int(x) for x in side["FilledLabels"])
        self.blocks = np.loadtxt(
            self.surf.with_name(side["BlocksFile"]), skiprows=1, ndmin=2
        )
        self.pts = GiftiImage.from_filename(str(self.surf), mmap=True).agg_data(
            "pointset"
        )
        self._src: dict[Path, NiftiSource] = {}
        c = self.S.world(self.S.v_vox.mean(0))[0]
        self.centre = self.voxel(int(self.ov.vertex_ids[self.S.nearest(c)]))

    def meta(self) -> dict[str, Any]:
        """The page's meta in surface-only mode: no traces, layers or flat map."""
        return dict(
            site=f"{self.name} (white surface)",
            surface_only=True,
            n=0,
            uv=[],
            rv=[],
            levels=self.S.infl_names,
            layers=[],
            classes=[],
            points=0,
            points_world=[],
            vmax=1.0,
            has_levels=False,
            tracer="none",
            wm_label="none",
            overview=dict(
                true_vertices=len(self.W.v),
                true_faces=len(self.W.f),
                vertices=len(self.ov.vertex_ids),
                faces=len(self.ov.f),
                eps_mm=self.ov.sidecar["EpsMM"],
                max_distance_mm=round(self.ov.sidecar["MaxDistanceMM"], 4),
            ),
            box=dict(centre=self.centre, half=self.half, overlay=OVERLAY),
        )

    def vertex(self, k: int) -> dict[str, Any]:
        """The WM selection at overview vertex k: its world point and the true
        vertex it is (an index into the surface's files)."""
        k = int(k)
        p = self.S.world(self.S.v_vox[k])[0]
        t = int(self.ov.vertex_ids[k])
        return dict(
            vertex=k,
            true_vertex=t,
            world_mm=[round(float(x), 3) for x in p],
            voxel=self.voxel(t),
            trace=None,
            root_mm=None,
        )

    def voxel(self, true_vertex: int) -> list[int]:
        """The voxel of the grid nearest a true vertex."""
        return [int(x) for x in np.round(np.asarray(self.W.v[true_vertex], float))]

    def snap(self, mm: npt.ArrayLike) -> dict[str, Any]:
        """The WM selection at the overview vertex nearest world point mm."""
        return self.vertex(self.S.nearest(mm))

    def mesh_payload(self, level: int, faces: bool) -> bytes:
        """The 3D pane's overview at inflated level `level`; with faces, every
        vertex without a trace."""
        return self.S.mesh_payload(level, self.no_trace if faces else None)

    def white_bytes(self) -> bytes:
        return self.S.white_bytes()

    # ------------------------------------------------------------ the volume pane's box
    def box(
        self, c: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """The box round voxel c: its first voxel and one past its last, inside
        the grid."""
        cc = np.asarray(c, np.int64)
        lo = np.clip(cc - self.half, 0, np.array(self.grid))
        hi = np.clip(cc + self.half + 1, 0, np.array(self.grid))
        return lo, hi

    def _source(self, path: Path) -> NiftiSource:
        """A volume on the surface's grid (kept open: a .nii.gz is read once)."""
        if path not in self._src:
            src = open_volume(path)
            if tuple(src.shape[:3]) != self.grid or not np.allclose(
                src.affine, self.W.affine, atol=1e-4
            ):
                raise ValueError(f"{path.name} is not on the white surface's grid")
            self._src[path] = src
        return self._src[path]

    def _nifti(self, a: npt.NDArray[Any], lo: npt.NDArray[np.int64]) -> bytes:
        corner = (int(lo[0]), int(lo[1]), int(lo[2]))
        img = nib.nifti1.Nifti1Image(a, crop_affine(self.W.affine, corner))
        return gzip.compress(img.to_bytes(), compresslevel=1)

    def echo_bytes(self, k: int, c: npt.ArrayLike) -> bytes:
        """Echo k in the box round voxel c, as .nii.gz bytes."""
        lo, hi = self.box(c)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        return self._nifti(np.asarray(self._source(self.echoes[k]).read(box)), lo)

    def overlay(self, c: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        """The overlay (OVERLAY) in the box round voxel c."""
        lo, hi = self.box(c)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        seg = self._source(self.dseg).read(box)
        fx = np.asarray(self._source(self.fixes).read(box))
        out = np.isin(seg, self.filled).astype(np.uint8)
        out[fx > 0] = fx[fx > 0] + 1
        return out

    def overlay_bytes(self, c: npt.ArrayLike) -> bytes:
        return self._nifti(self.overlay(c), self.box(c)[0])

    def surface_faces(self, c: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """The true surface's faces (true vertex ids) with a corner in the box
        round voxel c, from the blocks whose vertices' bounds meet the box."""
        lo, hi = self.box(c)
        b = self.blocks
        meet = np.all(b[:, 7:10] <= hi - 0.5, axis=1) & np.all(
            b[:, 10:13] >= lo - 0.5, axis=1
        )
        parts = [self.W.f[int(r[5]) : int(r[6])] for r in b[meet]]
        if not parts:
            return np.zeros((0, 3), np.int64)
        f = np.concatenate(parts)
        v = np.asarray(self.W.v[np.unique(f)], float)
        inside = np.all((v >= lo - 0.5) & (v <= hi - 0.5), axis=1)
        keep = np.isin(f, np.unique(f)[inside]).any(1)
        out: npt.NDArray[np.int64] = f[keep]
        return out

    def surface_bytes(self, c: npt.ArrayLike) -> bytes:
        """The true surface in the box round voxel c, as GIfTI (world mm)."""
        f = self.surface_faces(c)
        ids, inv = np.unique(f, return_inverse=True)
        return gifti_bytes(
            np.asarray(self.pts[ids], np.float64), inv.reshape(-1, 3).astype(np.int64)
        )
