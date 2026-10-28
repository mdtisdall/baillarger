"""Remove the handles of the hemisphere's WM label (aseg 41) once, on the full grid, so that every
box's white mesh has genus-0 pieces (baillarger.white; report: Crossing paths and contacts from
laminar profiles.md, *A handle-free white surface*).

The work is the package's (baillarger.white.wm_changes and write_changes, as
`python -m baillarger.white changes`); this driver keeps the file names of the earlier runs. Every
box reads its corrected label with fixed_wm (baillarger.white.white_matter_box).

Usage: wm_handles.py <work> <data>   (writes <data>/wm_handles.npz, <data>/wm_handles.json and the
       ITK-SNAP overlay <data>/wm_handles_overlay.nii.gz on the full grid: 1 cut, 2 fill, 3 stranded)
"""

import sys
from pathlib import Path

from baillarger.white import KINDS, WMChanges, white_matter_box, wm_changes, write_changes

KIND = KINDS


def fixed_wm(lo, hi, data):
    """The handle-free WM label of the box [lo, hi) of the full grid (bool), from
    <data>/wm_handles.npz."""
    return white_matter_box(tuple(int(t) for t in lo), tuple(int(t) for t in hi),
                            WMChanges.load(Path(data) / "wm_handles.npz"))


def main(work, data):
    ch = wm_changes()
    print(f"(b0, b1, b2) {ch.betti_before} -> {ch.betti_after}, {len(ch.voxels)} voxels", flush=True)
    write_changes(ch, Path(data), "wm_handles")


if __name__ == "__main__":
    main(*sys.argv[1:3])
