"""The white mesh of each box before and after the handle removal of wm_handles.py: the report and
the box-grid images are baillarger.white.write_box (as `python -m baillarger.white box`); this
driver takes each site's box from column_volume_<site>.npz, checks that its WM label is the aseg's,
and draws labelled slices through each change.

Usage: wm_handles_boxes.py <figures> <data> <site>...
       (run from prototypes/scripts; reads <data>/wm_handles.npz; writes <data>/wm_handles_<site>.json,
       <data>/wm_handles_<site>_overlay.nii.gz, <data>/wm_handles_<site>_wm_fixed.nii.gz and
       <figures>/wm_handles_<site>.png)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from baillarger.data import ASEG_WM, aseg_path
from baillarger.white import WMChanges, white_matter_box, write_box


def main(fig, data, site):
    z = np.load(f"column_volume_{site}.npz")
    lo, wm0 = tuple(int(t) for t in z["lo"]), z["wm"]
    hi = tuple(a + s for a, s in zip(lo, wm0.shape))
    img = nib.load(aseg_path())
    assert np.array_equal(
        np.asarray(img.dataobj[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]) == ASEG_WM,
        wm0,
    )
    ch = WMChanges.load(Path(data) / "wm_handles.npz")
    rep = write_box(lo, hi, ch, Path(data), f"wm_handles_{site}")
    print(json.dumps({k: v for k, v in rep.items() if k != "moves"}))
    print(f"{len(rep['moves'])} moves in the box")
    view(fig, site, wm0, white_matter_box(lo, hi, ch), rep["moves"])


def view(fig, site, wm0, wm1, moves):
    """Three orthogonal slices through the largest changes (up to 6): WM before in grey, removed
    voxels red, added voxels blue; titles give the ITK-SNAP voxel (1-based, box)."""
    big = sorted(moves, key=lambda d: -d["voxels_in_box"])[:6]
    if not big:
        return
    r = 12
    figh, axs = plt.subplots(len(big), 3, figsize=(9, 3 * len(big)), squeeze=False)
    for row, d in zip(axs, big):
        c = np.array(d["centre_box0"])
        for ax, a in zip(row, range(3)):
            sl = [slice(max(c[i] - r, 0), c[i] + r + 1) for i in range(3)]
            sl[a] = c[a]
            w0, w1 = wm0[tuple(sl)], wm1[tuple(sl)]
            rgb = np.where(w0[..., None], 0.6, 1.0) * np.ones(3)
            rgb[w0 & ~w1] = (0.85, 0.1, 0.1)
            rgb[w1 & ~w0] = (0.1, 0.3, 0.9)
            ax.imshow(
                np.transpose(rgb, (1, 0, 2)), origin="lower", interpolation="nearest"
            )
            ax.set_title(
                f"{d['kind']} {d['voxels_in_box']} vox; axis {a} = {c[a] + 1}",
                fontsize=8,
            )
            ax.set_xticks([]), ax.set_yticks([])
        row[0].set_ylabel(f"ITK-SNAP {tuple(d['centre_itksnap_box'])}", fontsize=8)
    figh.suptitle(f"{site}: WM grey, removed red, added blue", fontsize=10)
    figh.tight_layout()
    figh.savefig(f"{fig}/wm_handles_{site}.png", dpi=80)


if __name__ == "__main__":
    for s in sys.argv[3:]:
        main(sys.argv[1], sys.argv[2], s)
