"""Prepare a marked face for the contact pipeline (collide.py, collide2.py with SITE=<run>, POINTS=<run>_points_box.json,
GROW=0).

Reads a points file from the data folder (the user's points, ITK-SNAP numbering of the crop named in its 'crop'
entry; a file with one list per piece takes --piece), and that crop's <crop>_sites.json for its full-grid corner.
The run's box is the crop itself, or with --centre a box of 141 voxels loaded from the full volume and centred on
the points (every point then lies well inside it). Writes to the work folder <run>_points_box.json (full_lo,
full_hi, gap_all as 0-based box indices) and curv_<run>.npz (white-surface curvature at 0.32, 0.48 and 0.8 mm, as
curv_map.py writes it); with --centre also <run>_echo2_detrended.nii.gz in the data folder (the overlay's
background).

Usage: site_setup.py <datadir> <workdir> <run> <points file> [--piece N] [--centre]
"""

import argparse
import json
import sys

sys.path.insert(0, sys.argv[2])
import nibabel as nib
import numpy as np

import curv_lib as C

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("work"); ap.add_argument("run"); ap.add_argument("points")
ap.add_argument("--piece"); ap.add_argument("--centre", action="store_true")
A = ap.parse_args()
FULL = np.array([700, 384, 1148])
HALF = 70


def main():
    pf = json.load(open(f"{A.data}/{A.points}"))
    crop = pf["crop"].replace("_echo2_detrended.nii.gz", "")
    box = json.load(open(f"{A.data}/{crop}_sites.json"))
    pts_itk = pf["pieces"][A.piece]["contact_points"] if A.piece else pf["contact_points"]
    full = np.unique(np.array(pts_itk, int), axis=0) - 1 + np.array(box["full_lo"])
    if A.centre:
        c = np.round(full.mean(0)).astype(int)
        lo, hi = c - HALF, c + HALF + 1
        assert np.all(lo >= 0) and np.all(hi <= FULL), "the centred box leaves the volume"
    else:
        lo, hi = np.array(box["full_lo"]), np.array(box["full_hi"])
    P = full - lo
    margin = int(np.min(np.minimum(P, (hi - lo) - 1 - P)))
    assert margin >= 38, f"a point is {margin} voxels from a box face (under 6 mm)"
    json.dump(dict(full_lo=[int(v) for v in lo], full_hi=[int(v) for v in hi], gap_all=P.tolist(), run=A.run,
                   points_file=A.points, piece=A.piece), open(f"{A.work}/{A.run}_points_box.json", "w"), indent=1)
    B = C.load(lo, hi, A.run)
    K = {sg: C.curvature(B, sg) for sg in (0.32, 0.48, 0.8)}
    np.savez_compressed(f"{A.work}/curv_{A.run}.npz", lo=lo, pts=K[0.48]["pts"], n=K[0.48]["n"],
                        **{f"k1_{sg:g}": K[sg]["k1"] for sg in K}, **{f"k2_{sg:g}": K[sg]["k2"] for sg in K})
    if A.centre:
        nib.save(nib.Nifti1Image(np.where(B["ctx"], B["disp"], 0.0).astype(np.float32), B["p"].affine),
                 f"{A.data}/{A.run}_echo2_detrended.nii.gz")
    print(f"{A.run}: {len(P)} points (from {A.points}{', piece ' + A.piece if A.piece else ''}); box {tuple(hi - lo)} "
          f"from {tuple(lo)}; nearest point {margin} voxels from a face")


if __name__ == "__main__":
    main()
