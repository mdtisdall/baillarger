"""The joint surfaces: the starting seams of the outer loop (plan: Outer loop,
step 0).

Where two banks of a collapsed sulcus touch, the aseg labels them as one
block of cortex, with no fluid between them: there is no pial surface there,
so the frame's streamlines meet the facing bank's inside the cortex, never
exit into fluid, and their voxels have no relative depth (baillarger.columns
leaves them out). Step 0 places geometric seams there, as the ex vivo
prototype did:

- tissue grows from the WM and fluid flows in from the outside, one queue,
  the tissue keeping its topology (growth.grow_competing; the fluid is fast
  along dark lines of the echo-2 FLASH); where two banks touch with no fluid
  between them, topology forces a cut, and the cut is the seam;
- the seams are thinned (growth.thin_seam) and each cut is moved onto the
  darkest surface between the two banks (growth.mincut_seam).

Missing tissue takes no part. The seam voxels then count as fluid
(with_seams): a streamline that crosses a seam exits there, and its column
joins the model. The frame does not change (cortex and fluid are the same
to it). On V1V2 the seams (44,728 voxels) raised the cortex in the model
from 56 % to 64 % and left the column graph in 4 pieces instead of 10, so
the merging reached every level down to 4 regions instead of 16.

Step 7 replaces these seams with the model's (fluid against tissue costs
from the templates); these come from the image's darkness only, so where a
fused stretch has no visible line the cut can lie off the true contact.

Limitation: a bank's streamlines exit at a seam only if they cross it before
they meet the facing bank's streamlines at the ridge. Where the cut lies
beyond a bank's ridge (P1: the cut at x = 28, the ridge at 31.5), that bank's
streamlines still end inside the cortex, and its columns stay out. Making
the seams barriers to the frame's distance and streamlines lets each bank
reach its seam (plan: Columns and depth frame, the finding on streamlines
regenerated in the loop); that is a frame change, left for step 7.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger import growth as gr

type Bool = npt.NDArray[np.bool_]


def initial_seams(
    rim: npt.NDArray[np.integer[Any]],
    flash: npt.ArrayLike,
    voxel_mm: tuple[float, float, float],
    missing: Bool | None = None,
) -> tuple[Bool, dict[str, int]]:
    """The step-0 seams (cortex voxels that become fluid) and a report: the
    voxels the fluid took in the growth, the voxels it left deferred, and the
    min-cut placement's counts."""
    rim_g = np.asarray(rim).astype(np.int16)
    if missing is not None:
        rim_g = np.where(missing, 0, rim_g).astype(np.int16)
    cortex = rim_g == 3
    tissue, fluid, ndef = gr.grow_competing(rim_g, flash, voxel_mm)
    t2, s2 = gr.thin_seam(tissue, fluid, flash, cortex)
    _, s3, rep = gr.mincut_seam(t2, s2, flash, cortex)
    seams: Bool = s3 & cortex
    report = dict(
        grown_fluid=int(fluid.sum()),
        deferred=int(ndef),
        seam_voxels=int(seams.sum()),
        **{f"mincut_{k}": int(v) for k, v in rep.items()},
    )
    return seams, report


def with_seams(rim: npt.NDArray[np.integer[Any]], seams: Bool) -> npt.NDArray[np.int16]:
    """The rim labels with the seam voxels as fluid (1)."""
    out = np.asarray(rim).astype(np.int16).copy()
    out[seams] = 1
    return out
