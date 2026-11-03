"""Legacy: the white mesh of a box's label, for baillarger.column_traces.
Stream 2's phase 0 deletes this module. The white-surface commands are
baillarger.white_dseg, baillarger.white_surf and baillarger.white_overview.

white_mesh makes the white mesh of a box's label: marching tetrahedra at 0.5
of the binary label (so the mesh has the label's Kuhn topology), then
tangential relaxation and Gaussian smoothing without shrinkage
(baillarger.mesh_smoothing), which move vertices but keep the faces.
white_mesh_normals adds the smoothed vertex normals (out of the WM) that the
column traces start from. filled_labels is baillarger.white_dseg's.
"""

import numpy as np
import numpy.typing as npt

from baillarger import mesh_smoothing as S
from baillarger.mesh import marching_tetrahedra
from baillarger.white_dseg import filled_labels

__all__ = ["filled_labels", "white_mesh", "white_mesh_normals"]

type Bool = npt.NDArray[np.bool_]
type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def white_mesh(wm: Bool, sigma: float = 2.0) -> tuple[Float, Int]:
    """The white mesh of a WM label: vertices (voxel coordinates) and faces
    oriented out of the WM. Marching tetrahedra at 0.5 of the binary label,
    then relax and twice (sigma in voxels)."""
    v0, f = marching_tetrahedra(np.asarray(wm, np.float64), 0.5)
    f = f[:, [0, 2, 1]].copy()  # marching_tetrahedra points into the WM
    if not len(f):
        return v0, f
    return S.twice(S.relax(v0, f), f, sigma), f


def white_mesh_normals(wm: Bool, sigma: float = 2.0) -> tuple[Float, Int, Float]:
    """white_mesh and its smoothed unit vertex normals, out of the WM
    (baillarger.mesh_smoothing.smooth_normals)."""
    v, f = white_mesh(wm, sigma)
    return v, f, S.smooth_normals(v, f)
