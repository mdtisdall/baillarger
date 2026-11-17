"""Write the solver's per-step snapshots as 4D NIfTI volumes (crop grid,
crop affine) for a viewer: tau (nan outside the band, last axis = step) and
owner (-1 outside the band).

usage: tube_snapshots.py SITE_NPZ RUN_DIR
Writes RUN_DIR/tau_steps.nii.gz, RUN_DIR/owner_steps.nii.gz and
RUN_DIR/steps.json (the snapshot names in volume order).
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    site, run = Path(sys.argv[1]), Path(sys.argv[2])
    z = np.load(site)
    shape, aff = z["labels"].shape, z["affine"]
    names = sorted(p.stem for p in (run / "snapshots").glob("*.npz"))
    tau = np.full(shape + (len(names),), np.nan, np.float32)
    own = np.full(shape + (len(names),), -1, np.int32)
    for k, n in enumerate(names):
        s = np.load(run / "snapshots" / f"{n}.npz")
        i = np.unravel_index(s["index"], shape)
        tau[i + (k,)] = s["tau"]
        own[i + (k,)] = s["owner"]
    nib.save(nib.Nifti1Image(tau, aff), run / "tau_steps.nii.gz")
    nib.save(nib.Nifti1Image(own, aff), run / "owner_steps.nii.gz")
    (run / "steps.json").write_text(json.dumps(names))
    print(len(names), "snapshots")


if __name__ == "__main__":
    main()
