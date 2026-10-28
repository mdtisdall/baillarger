"""Remove the handles of the hemisphere's WM label (aseg 41) once, on the full grid, so that every
box's white mesh has genus-0 pieces (baillarger.handles; report: Crossing paths and contacts from
laminar profiles.md, *A handle-free white surface*).

The topology is the Kuhn adjacency that marching tetrahedra contours. Each handle is cut or filled,
whichever changes fewer voxels. A box crop of a handle-free solid has planar surface pieces, so the
correction is made on the whole label and every box crops it (fixed_wm).

Usage: wm_handles.py <work> <data>   (writes <data>/wm_handles.npz, <data>/wm_handles.json and the
       ITK-SNAP overlay <data>/wm_handles_overlay.nii.gz on the full grid: 1 cut, 2 fill, 3 stranded)
"""

import json
import sys
import time

import nibabel as nib
import numpy as np

from baillarger.data import ASEG_WM, aseg_path
from baillarger.handles import kuhn_betti, remove_handles

KIND = {"cut": 1, "fill": 2, "stranded": 3}
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


def fixed_wm(lo, hi, data="prototypes/data"):
    """The WM label of the box [lo, hi) of the full grid with the handle changes applied (bool)."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    img = nib.load(aseg_path())
    wm = np.asarray(img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) == ASEG_WM
    z = np.load(f"{data}/wm_handles.npz")
    p, k = z["voxels"] - lo, z["kind"]
    inside = np.all((p >= 0) & (p < hi - lo), 1)
    p, k = p[inside], k[inside]
    wm[tuple(p[k == KIND["fill"]].T)] = True
    wm[tuple(p[k != KIND["fill"]].T)] = False
    return wm


def main(work, data):
    img = nib.load(aseg_path())
    a = np.asarray(img.dataobj)
    wm = a == ASEG_WM
    del a
    nz = np.argwhere(wm.any((1, 2)))[:, 0], np.argwhere(wm.any((0, 2)))[:, 0], np.argwhere(wm.any((0, 1)))[:, 0]
    lo = np.array([max(int(t.min()) - 2, 0) for t in nz])
    hi = np.array([min(int(t.max()) + 3, s) for t, s in zip(nz, wm.shape)])
    box = tuple(slice(int(p), int(q)) for p, q in zip(lo, hi))
    x = wm[box]
    say(f"WM label: {int(x.sum()):,} voxels, box {lo.tolist()}..{hi.tolist()}")
    before = kuhn_betti(x)
    say(f"before (b0, b1, b2) = {before}")
    r = remove_handles(x)
    say(f"{len(r.moves)} moves, {r.stranded} stranded")
    after = kuhn_betti(r.obj)
    say(f"after (b0, b1, b2) = {after}")

    vox = np.concatenate([m.voxels for m in r.moves]) + lo if r.moves else np.zeros((0, 3), int)
    kind = np.concatenate([np.full(len(m.voxels), KIND[m.kind]) for m in r.moves]) if r.moves else np.zeros(0, int)
    move = np.concatenate([np.full(len(m.voxels), i) for i, m in enumerate(r.moves)]) if r.moves else np.zeros(0, int)
    np.savez_compressed(f"{data}/wm_handles.npz", voxels=vox.astype(np.int32), kind=kind.astype(np.int8),
                        move=move.astype(np.int32))
    ov = np.zeros(wm.shape, np.int16)
    ov[tuple(vox.T)] = kind
    nib.save(nib.Nifti1Image(ov, img.affine, img.header), f"{data}/wm_handles_overlay.nii.gz")

    vmm = float(np.prod(img.header.get_zooms()[:3]))
    moves = []
    for i, m in enumerate(r.moves):
        p = m.voxels + lo
        c = np.round(p.mean(0)).astype(int)
        moves.append(dict(move=i, kind=m.kind, voxels=len(p), volume_mm3=round(len(p) * vmm, 4),
                          extent_vox=(p.max(0) - p.min(0) + 1).tolist(), centre_full0=c.tolist(),
                          centre_itksnap=(c + 1).tolist()))
    summ = dict(box_lo=lo.tolist(), box_hi=hi.tolist(), wm_voxels=int(x.sum()), betti_before=before,
                betti_after=after, moves=len(r.moves), stranded=r.stranded,
                voxels_removed=int((x & ~r.obj).sum()), voxels_added=int((r.obj & ~x).sum()),
                by_kind={k: dict(moves=sum(m["kind"] == k for m in moves),
                                 voxels=sum(m["voxels"] for m in moves if m["kind"] == k)) for k in KIND},
                list=moves)
    json.dump(summ, open(f"{data}/wm_handles.json", "w"), indent=1)
    say(json.dumps({k: v for k, v in summ.items() if k != "list"}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
