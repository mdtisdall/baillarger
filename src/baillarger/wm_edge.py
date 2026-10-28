"""The image's own GM/WM edge against the white mesh.

Step 1 of putting the white surface on the data's GM/WM boundary (plan:
reports/Crossing paths and contacts from laminar profiles.md, *The white
surface on the image's edge*): at each mesh vertex, the signed distance along
the outward normal from the vertex to the steepest WM-to-GM change of an echo.
Steps 2-4 (correct the label where the edge is more than half a voxel away,
remove handles again, refine the vertices) are not here yet.

1. tissue_contrast: the intensity divided by its normalized convolution over
   the WM at trend voxels, minus 1, so that a slow bias field goes and the
   WM/GM step stays where it is. A divisor with a gradient across the edge
   moves the steepest point by about 2 s² g (s the edge's blur, g the
   divisor's relative gradient): over WM and cortex together the trend holds
   the step itself (0.012 mm for a 40 % step 0.24 mm wide, at 8 voxels), over
   the WM alone it does not. Not the log either: the log of a blurred step is
   steepest nearer its dark side (0.03 mm for the same step).
2. edge_offsets: the image's gradient, Gaussian-smoothed at scale_mm, is
   sampled by cubic spline along each vertex's normal over +-window_mm, in
   steps of step_mm; the offset is where its component along the normal,
   signed so that WM to GM is positive, is largest (refined by a parabola
   through the three samples round the peak). Positive offsets mean the
   image's edge lies outside the mesh, in the cortex. The sign of the WM-to-GM
   change is the data's: the sign of the median derivative at the vertices.

On a curved edge the derivative scale moves the steepest point toward the
concave side by about scale_mm² H (H the mean curvature): 0.01 mm at
H = 1/mm and the default scale; it is not corrected.

Interpolation matters: linear interpolation makes the derivative piecewise
constant, and its peak falls on a voxel face (up to a voxel from the edge);
the spline leaves under 0.1 voxel on a step a voxel wide (tests/
test_wm_edge.py). Positions are voxel index coordinates; normals are unit
vectors in voxel coordinates (the mesh's), converted to mm as covectors.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from baillarger.streamlines import masked_smooth

type Bool = npt.NDArray[np.bool_]
type F64 = npt.NDArray[np.float64]

#: half-width of the search window along the normal (mm)
WINDOW_MM = 0.3
#: Gaussian scale of the image gradient (mm)
SCALE_MM = 0.1
#: sampling step along the normal (mm)
STEP_MM = 0.005
#: scale of the trend removed from the log intensity (voxels)
TREND = 8.0
_CHUNK = 20000


@dataclass(frozen=True)
class EdgeOffsets:
    """Per vertex: where the image's edge lies along the normal."""

    #: signed distance (mm) from the vertex to the edge, + toward cortex
    offset_mm: F64
    #: the derivative there (per mm), WM-to-GM positive
    slope: F64
    #: the peak is at the window's end: the edge is not inside the window
    at_end: Bool
    #: +1 if the image rises from WM to GM, -1 if it falls
    sign: float


def tissue_contrast(echo: npt.ArrayLike, wm: Bool, trend: float = TREND) -> F64:
    """echo / (its normalized convolution over the WM, voxels: trend) - 1.
    Defined everywhere; only the WM enters the trend."""
    x = np.asarray(echo, np.float64)
    out: F64 = x / np.maximum(masked_smooth(x, wm, trend), 1e-12) - 1.0
    return out


def edge_offsets(
    img: npt.ArrayLike,
    v: F64,
    n: F64,
    vox: npt.ArrayLike,
    window_mm: float = WINDOW_MM,
    scale_mm: float = SCALE_MM,
    step_mm: float = STEP_MM,
    sign: float | None = None,
) -> EdgeOffsets:
    """The image's steepest WM-to-GM point along each vertex's normal (v,
    n: voxel coordinates, unit outward normals in voxel coordinates; vox:
    voxel size, mm). sign: +1 or -1 for the WM-to-GM change, or None to take
    the data's."""
    a = np.asarray(img, np.float64)
    vx = np.asarray(vox, np.float64)
    sig = scale_mm / vx
    grads = []
    for k in range(3):
        order = [0, 0, 0]
        order[k] = 1
        gk = ndi.gaussian_filter(a, sig, order=order) / vx[k]  # per mm
        grads.append(ndi.spline_filter(gk, 3))
    nm = n / vx  # normals are covectors
    nm /= np.linalg.norm(nm, axis=1, keepdims=True)
    m = int(round(window_mm / step_mm))
    s = np.arange(-m, m + 1) * step_mm
    der = np.empty((len(v), len(s)))
    for i in range(0, len(v), _CHUNK):
        sl = slice(i, i + _CHUNK)
        p = v[sl, None, :] + s[None, :, None] * (nm[sl] / vx)[:, None, :]
        q = p.reshape(-1, 3).T
        d = np.zeros(q.shape[1])
        for k in range(3):
            gk = ndi.map_coordinates(grads[k], q, order=3, prefilter=False)
            d += gk * np.repeat(nm[sl, k], len(s))
        der[sl] = d.reshape(-1, len(s))
    if sign is None:
        sign = 1.0 if np.median(der[:, m]) >= 0 else -1.0
    der *= sign
    j = np.argmax(der, 1)
    r = np.arange(len(v))
    at_end = (j == 0) | (j == len(s) - 1)
    jc = np.clip(j, 1, len(s) - 2)
    y0, y1, y2 = der[r, jc - 1], der[r, jc], der[r, jc + 1]
    den = y0 - 2 * y1 + y2
    frac = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
    frac = np.where(at_end, 0.0, np.clip(frac, -0.5, 0.5))
    off = s[j] + frac * step_mm
    return EdgeOffsets(off, der[r, j], at_end, float(sign))
