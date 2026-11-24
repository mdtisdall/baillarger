import sys, json; sys.path.insert(0, sys.argv[1])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import curv_lib as C
S = sys.argv[1]
m = json.load(open(f"{S}/M1S1_test_points.json"))
lo = np.array(m["full_lo"]) - 15; hi = np.array(m["full_hi"]) + 15
B = C.load(lo, hi, "M1S1")
cz = np.load(f"{S}/curv_M1S1.npz"); pts = cz["pts"]
idx = -np.ones(B["rim"].shape, np.int64); idx[tuple(pts.T)] = np.arange(len(pts))
_, ind = ndi.distance_transform_edt(idx < 0, return_indices=True)
fig = plt.figure(figsize=(30, 20))
rows = [("display mesh as before (label smoothed 1 voxel, step 2)", 1.0, 2),
        ("smooth display mesh (label smoothed 2 voxels, step 1)", 2.0, 1)]
for r, (lab, sg, st) in enumerate(rows):
    verts, faces, _, _ = marching_cubes(ndi.gaussian_filter(B["wm"].astype(float), sg), 0.5, step_size=st)
    vi = np.clip(np.round(verts).astype(int), 0, np.array(B["rim"].shape) - 1)
    near = idx[tuple(ind[:, vi[:, 0], vi[:, 1], vi[:, 2]])]
    keep = (B["dface"][tuple(vi.T)] >= 6.0)[faces].all(1)
    # a window round the kissing stretch keeps the render readable
    P = verts * B["vox"]
    gap = (np.array(m["gap_all"]) + 15) * B["vox"]
    cen = gap.mean(0)
    keep &= (np.abs(P[faces].mean(1) - cen) < 4.5).all(1)
    f = faces[keep]; tp = P[f]
    nrm = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    for c, (sc, what) in enumerate((("0.48", "H"), ("0.32", "H"), (None, "shade"))):
        a = fig.add_subplot(2, 3, 3 * r + c + 1, projection="3d")
        el, az = 30, -60
        light = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
        shade = 0.45 + 0.55 * np.abs(nrm @ light)
        if what == "H":
            Hv = 0.5 * (cz[f"k1_{sc}"] + cz[f"k2_{sc}"])[near]
            col = plt.cm.RdBu_r(np.clip((Hv[f].mean(1) + 1) / 2, 0, 1)); col[:, :3] *= shade[:, None]
            title = f"mean curvature, scale {sc} mm"
        else:
            col = np.repeat(np.array([[0.8, 0.8, 0.8, 1.0]]), len(f), 0); col[:, :3] *= shade[:, None]
            title = "geometry only (grey, shaded)"
        a.add_collection3d(Poly3DCollection(tp, facecolors=col, linewidths=0))
        q = P[np.unique(f)]; a.set_xlim(q[:, 0].min(), q[:, 0].max()); a.set_ylim(q[:, 1].min(), q[:, 1].max()); a.set_zlim(q[:, 2].min(), q[:, 2].max())
        a.set_box_aspect(np.ptp(q, 0)); a.view_init(el, az)
        a.scatter(*gap.T, c="lime", s=14, depthshade=False)
        a.set_title(f"{lab}\n{title}", fontsize=13)
fig.suptitle("M1S1, white surface within 4.5 mm of the kissing stretch: is the staircase in the curvature or only in the display mesh?", fontsize=15)
fig.savefig(f"{S}/curv_stair_check_M1S1.png", dpi=70, bbox_inches="tight")
print("ok")
