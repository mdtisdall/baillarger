"""ITK-SNAP files for reading the per-front contact against the user's points (M1S1 test crop space).

Writes, next to the other test-crop files:
  M1S1_test_echo2_detrended.nii.gz  echo 2, trend removed (cortex only), the default display
  M1S1_contact_overlay.nii.gz       labels: 1 WM boundary, 2 posterior contact (cyan in the figures),
                                    3 equal arrival (lime), 4 points 7-13, 5 the other points
  M1S1_contact_labels.txt
The contacts are one-voxel surfaces, drawn only within 2 mm of an all-WM collision sheet: voxels on the non-negative side of the zero level with a
6-neighbour on the other side. Points are small dots (the voxel and its 6 neighbours).

Usage: collide2_itksnap.py <datadir> <workdir>. Needs collide2_state.npz (collide2.py).
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import nibabel as nib
import numpy as np

import curv_lib as C
from collide import GROW, POINTS, SITE

OUT, S = sys.argv[1], sys.argv[2]
OFF_PTS = range(6, 13) if SITE == "M1S1" else range(0)  # points 7-13 at M1S1, 0-based


def surface(f, dom):
    on = np.zeros(f.shape, bool)
    pos = dom & (f >= 0)
    for a in range(3):
        for sgn in (1, -1):
            nb = np.roll(f, sgn, axis=a)
            nd = np.roll(dom, sgn, axis=a)
            edge = [slice(None)] * 3
            edge[a] = 0 if sgn == 1 else -1
            nd[tuple(edge)] = False  # no wrap-around across the box
            on |= pos & nd & (nb < 0)
    return on


def main():
    m = json.load(open(f"{S}/{POINTS}"))
    lo = np.array(m["full_lo"]) - GROW
    hi = np.array(m["full_hi"]) + GROW
    B = C.load(lo, hi, SITE)
    st = np.load(f"{S}/collide2_state.npz")
    from scipy import ndimage as ndi

    sheet = np.load(f"{S}/collide_state.npz")["ctype"] == 1
    # draw the rules only near collisions: away from them the two fronts' fields cover unrelated cortex
    dom = st["both_dom"] & (
        ndi.distance_transform_edt(~sheet, sampling=B["vox"]) <= 2.0
    )
    crop = tuple(slice(GROW, n - GROW) for n in B["rim"].shape)
    ref = (
        nib.load(f"{S}/M1S1_test_echo2.nii.gz")
        if SITE == "M1S1"
        else nib.load(f"{OUT}/{SITE}_echo2_detrended.nii.gz")
    )
    seg = np.zeros(B["rim"].shape, np.uint8)
    seg[B["shell"]] = 1
    seg[surface(np.nan_to_num(st["eq"], nan=-1.0), dom)] = 3
    seg[surface(np.nan_to_num(st["gpost"], nan=-1.0), dom)] = 2
    P = np.round(np.array(m["gap_all"], float)).astype(int) + GROW
    for k, q in enumerate(P):
        lab = 4 if k in OFF_PTS else 5
        for d in [
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]:
            seg[tuple(q + d)] = lab
    seg = seg[crop]
    assert seg.shape == ref.shape, (seg.shape, ref.shape)
    disp = np.where(B["ctx"], B["disp"], 0.0)[crop].astype(np.float32)
    if SITE == "M1S1":
        nib.save(
            nib.Nifti1Image(disp, ref.affine), f"{OUT}/M1S1_test_echo2_detrended.nii.gz"
        )
    nib.save(nib.Nifti1Image(seg, ref.affine), f"{OUT}/{SITE}_contact_overlay.nii.gz")
    open(f"{OUT}/{SITE}_contact_labels.txt", "w").write(
        '    0     0    0    0        0  0  0    "Clear Label"\n'
        '    1   255  220    0        1  1  1    "WM boundary"\n'
        '    2     0  220  255        1  1  1    "Posterior contact"\n'
        '    3    60  220   60        1  1  1    "Equal arrival"\n'
        '    4   255   40   40        1  1  1    "Points 7-13 (M1S1)"\n'
        '    5   255    0  255        1  1  1    "Your points"\n'
    )
    for lab, nm in (
        (1, "WM boundary"),
        (2, "posterior"),
        (3, "equal arrival"),
        (4, "points 7-13"),
        (5, "other points"),
    ):
        print(f"{nm}: {np.sum(seg == lab):,} voxels")


if __name__ == "__main__":
    main()
