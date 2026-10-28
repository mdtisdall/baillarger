"""Remove the handles of the hemisphere's WM label (aseg 41) once, on the full grid, so that every
box's white mesh has genus-0 pieces (baillarger.white; report: Crossing paths and contacts from
laminar profiles.md, *A handle-free white surface*).

This driver runs baillarger.white.wm_changes, saves the changes and writes the views. Every box
reads its corrected label with fixed_wm (baillarger.white.white_matter_box).

Usage: wm_handles.py <work> <data>   (writes <data>/wm_handles.npz, <data>/wm_handles.json and the
       ITK-SNAP overlay <data>/wm_handles_overlay.nii.gz on the full grid: 1 cut, 2 fill, 3 stranded)
"""

import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from baillarger.data import aseg_path
from baillarger.white import KINDS, WMChanges, white_matter_box, wm_changes

KIND = KINDS
NAMES = {v: k for k, v in KINDS.items()}
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


def fixed_wm(lo, hi, data):
    """The handle-free WM label of the box [lo, hi) of the full grid (bool), from
    <data>/wm_handles.npz."""
    return white_matter_box(tuple(int(t) for t in lo), tuple(int(t) for t in hi),
                            WMChanges.load(Path(data) / "wm_handles.npz"))


def main(work, data):
    ch = wm_changes()
    say(f"(b0, b1, b2) {ch.betti_before} -> {ch.betti_after}")
    ch.save(Path(data) / "wm_handles.npz")
    img = nib.load(aseg_path())
    ov = np.zeros(img.shape, np.int16)
    ov[tuple(ch.voxels.T)] = ch.kind
    nib.save(nib.Nifti1Image(ov, img.affine, img.header), f"{data}/wm_handles_overlay.nii.gz")

    vmm = float(np.prod(img.header.get_zooms()[:3]))
    moves = []
    for i in np.unique(ch.move):
        p = ch.voxels[ch.move == i]
        c = np.round(p.mean(0)).astype(int)
        moves.append(dict(move=int(i), kind=NAMES[int(ch.kind[ch.move == i][0])], voxels=len(p),
                          volume_mm3=round(len(p) * vmm, 4),
                          extent_vox=(p.max(0) - p.min(0) + 1).tolist(), centre_full0=c.tolist(),
                          centre_itksnap=(c + 1).tolist()))
    summ = dict(betti_before=ch.betti_before, betti_after=ch.betti_after, moves=len(moves),
                stranded=int((ch.kind == KINDS["stranded"]).sum()),
                voxels_removed=int((ch.kind != KINDS["fill"]).sum()),
                voxels_added=int((ch.kind == KINDS["fill"]).sum()),
                by_kind={k: dict(moves=sum(m["kind"] == k for m in moves),
                                 voxels=sum(m["voxels"] for m in moves if m["kind"] == k)) for k in KINDS},
                list=moves)
    json.dump(summ, open(f"{data}/wm_handles.json", "w"), indent=1)
    say(json.dumps({k: v for k, v in summ.items() if k != "list"}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
