"""Anatomical context for one row: axial, coronal and sagittal world planes
through the root, wide (+-HALF_W mm) and medium (+-HALF_M mm), echo 1, with a
crosshair at the root and the ray's outward direction.
args: ROOT_X ROOT_Y ROOT_Z DIR_X DIR_Y DIR_Z TITLE OUT"""
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import top_paths as TP  # noqa: E402

ECHOES = TP.echoes()

root = np.array([float(a) for a in sys.argv[1:4]])
d = np.array([float(a) for a in sys.argv[4:7]])
d /= np.linalg.norm(d)
TITLE, OUT = sys.argv[7], Path(sys.argv[8])
img = nib.load(str(ECHOES[0]))
inv = np.linalg.inv(img.affine)
PLANES = [("axial (z fixed)", 0, 1, ("R", "L"), ("A", "P")), ("coronal (y fixed)", 0, 2, ("R", "L"), ("S", "I")),
          ("sagittal (x fixed)", 1, 2, ("A", "P"), ("S", "I"))]


def plane(ia, ib, half, res):
    t = np.arange(-half, half + 1e-9, res)
    A, B = np.meshgrid(t, t, indexing="xy")
    X = np.repeat(root[None, None], A.shape[0], 0).repeat(A.shape[1], 1)
    X[..., ia] += A
    X[..., ib] += B
    q = X.reshape(-1, 3) @ inv[:3, :3].T + inv[:3, 3]
    lo = np.maximum(np.floor(q.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(q.max(0)).astype(int) + 2, img.shape)
    box = np.asarray(img.dataobj[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))], np.float32)
    v = map_coordinates(box, (q - lo).T, order=1, mode="constant", cval=np.nan).reshape(A.shape)
    return t, v


fig, axs = plt.subplots(2, 3, figsize=(16, 11))
for row, (half, res) in enumerate(((25.0, 0.1), (8.0, 0.04))):
    for col, (name, ia, ib, la, lb) in enumerate(PLANES):
        ax = axs[row, col]
        t, v = plane(ia, ib, half, res)
        vlo, vhi = np.nanpercentile(v, [1, 99])
        ext = (root[ia] - half, root[ia] + half, root[ib] - half, root[ib] + half)
        ax.imshow(v, cmap="gray", vmin=vlo, vmax=vhi, origin="lower", extent=ext)
        ax.axhline(root[ib], color="#ff3030", lw=0.6, alpha=0.6)
        ax.axvline(root[ia], color="#ff3030", lw=0.6, alpha=0.6)
        ax.plot(root[ia], root[ib], "o", ms=8, mfc="none", mec="#ff3df0", mew=2)
        L = half * 0.15
        ax.annotate("", (root[ia] + L * d[ia], root[ib] + L * d[ib]), (root[ia], root[ib]),
                    arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=2))
        if row == 1:
            ax.add_patch(plt.Rectangle((root[ia] - 3, root[ib] - 3), 6, 6, fill=False, ec="#ffd400", lw=1.2, ls="--"))
        if row == 0:
            ax.add_patch(plt.Rectangle((root[ia] - 8, root[ib] - 8), 16, 16, fill=False, ec="#ffd400", lw=1.2, ls="--"))
        if ia == 0:
            ax.invert_xaxis()  # radiological view not wanted: patient right on the viewer's left, as in the viewer's slices
        ax.set_title(f"{name}, ±{half:g} mm")
        ax.set_xlabel(f"{'xyz'[ia]} mm ({la[0]} = +, {la[1]} = -)")
        ax.set_ylabel(f"{'xyz'[ib]} mm ({lb[0]} = +, {lb[1]} = -)")
fig.suptitle(f"{TITLE}\necho 1; magenta circle: root; cyan arrow: ray direction at the root (outward); "
             "yellow dashed: the next panel's field (bottom row: the row figure's ±3 mm)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=80)
print("wrote", OUT)
