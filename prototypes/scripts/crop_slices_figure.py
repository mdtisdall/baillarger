"""Three slices of a crop through one point: echo 2 with the edges of the WM
label (cyan) and the outside-medium label (yellow), to look at a box before
a run.

usage (from prototypes/scripts): crop_slices_figure.py CROP OUT.png TITLE [I,J,K]   (default: the box centre)
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import crop_site as CS
z = CS.site_arrays(sys.argv[1]); out = sys.argv[2]
e = z["echoes"][1]; lab = z["labels"]; c = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [s // 2 for s in lab.shape]
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), layout="constrained")
for a in range(3):
    sl = [slice(None)] * 3; sl[a] = c[a]; t = tuple(sl)
    ax[a].imshow(e[t].T, cmap="gray", origin="lower")
    ax[a].contour((lab[t] == 41).T.astype(float), [0.5], colors="c", linewidths=0.8)
    ax[a].contour((lab[t] == 0).T.astype(float), [0.5], colors="y", linewidths=0.8)
    o = [q for q in range(3) if q != a]
    ax[a].plot(c[o[0]], c[o[1]], "r+", ms=14)
    ax[a].set_title(f"{sys.argv[3]}: slice {'ijk'[a]} = {c[a]} of the crop"); ax[a].set_xticks([]); ax[a].set_yticks([])
fig.suptitle("echo 2; cyan: WM label edge; yellow: outside-medium label edge; red +: the picked point")
fig.savefig(out, dpi=90)
