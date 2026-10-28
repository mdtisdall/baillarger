"""The white mesh of each box before and after the handle removal of wm_handles.py: topology per
piece (flat_map.topology), area, the changes inside the box, an ITK-SNAP overlay on the box grid and
labelled slices through each change.

Usage: wm_handles_boxes.py <figures> <data> <site>...
       (writes <data>/wm_handles_<site>.json, <data>/wm_handles_<site>_overlay.nii.gz and
       <figures>/wm_handles_<site>.png)
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import flat_map as FM
import mesh_smoothing as MS
from baillarger.data import aseg_path
from wm_handles import KIND, fixed_wm

NAMES = {v: k for k, v in KIND.items()}


def white_mesh(wm):
    """As column_volume.white_mesh."""
    v0, f = MS.contour(wm, 0.0)
    return MS.twice(MS.relax(v0, f), f, 2.0), f


def per_piece(v, f, vox):
    lab, flab = FM.pieces(f, len(v))
    out = []
    for p in range(flab.max() + 1):
        fp = f[flab == p]
        t = FM.topology(fp)
        e1, e2 = (v[fp[:, 1]] - v[fp[:, 0]]) * vox, (v[fp[:, 2]] - v[fp[:, 0]]) * vox
        t["area_mm2"] = round(float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()), 3)
        out.append(t)
    return sorted(out, key=lambda t: -t["V"])


def main(fig, data, site):
    z = np.load(f"column_volume_{site}.npz")
    lo, wm0 = z["lo"], z["wm"]
    hi = lo + np.array(wm0.shape)
    img = nib.load(aseg_path())
    vox = np.array(img.header.get_zooms()[:3], float)
    wm1 = fixed_wm(lo, hi, data)
    assert np.array_equal(np.asarray(img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) == 41, wm0)
    before = per_piece(*white_mesh(wm0), vox)
    after = per_piece(*white_mesh(wm1), vox)

    H = np.load(f"{data}/wm_handles.npz")
    p, k, m = H["voxels"] - lo, H["kind"], H["move"]
    inside = np.all((p >= 0) & (p < hi - lo), 1)
    p, k, m = p[inside], k[inside], m[inside]
    moves = []
    for i in np.unique(m):
        q = p[m == i]
        c = np.round(q.mean(0)).astype(int)
        moves.append(dict(move=int(i), kind=NAMES[int(k[m == i][0])], voxels_in_box=len(q),
                          centre_box0=c.tolist(), centre_itksnap_box=(c + 1).tolist(),
                          centre_full0=(c + lo).tolist(), face_distance_mm=round(float(
                              np.min(np.r_[c, np.array(wm0.shape) - 1 - c] * np.r_[vox, vox])), 2)))
    ov = np.zeros(wm0.shape, np.int16)
    ov[tuple(p.T)] = k
    aff = img.affine @ np.r_[np.c_[np.eye(3), lo], [[0, 0, 0, 1]]]
    nib.save(nib.Nifti1Image(ov, aff), f"{data}/wm_handles_{site}_overlay.nii.gz")
    nib.save(nib.Nifti1Image(wm1.astype(np.int16), aff), f"{data}/wm_handles_{site}_wm_fixed.nii.gz")

    summ = dict(site=site, lo=lo.tolist(), voxels_removed=int((wm0 & ~wm1).sum()),
                voxels_added=int((wm1 & ~wm0).sum()),
                area_mm2_before=round(sum(t["area_mm2"] for t in before), 3),
                area_mm2_after=round(sum(t["area_mm2"] for t in after), 3),
                pieces_before=before, pieces_after=after, moves=moves)
    json.dump(summ, open(f"{data}/wm_handles_{site}.json", "w"), indent=1)
    print(json.dumps({k_: v_ for k_, v_ in summ.items() if k_ not in ("moves",)}, indent=None))
    print(f"{len(moves)} moves in the box")
    view(fig, site, wm0, wm1, moves)


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
            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", interpolation="nearest")
            ax.set_title(f"{d['kind']} {d['voxels_in_box']} vox; axis {a} = {c[a] + 1}", fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
        row[0].set_ylabel(f"ITK-SNAP {tuple(d['centre_itksnap_box'])}", fontsize=8)
    figh.suptitle(f"{site}: WM grey, removed red, added blue", fontsize=10)
    figh.tight_layout()
    figh.savefig(f"{fig}/wm_handles_{site}.png", dpi=80)


if __name__ == "__main__":
    for s in sys.argv[3:]:
        main(sys.argv[1], sys.argv[2], s)
