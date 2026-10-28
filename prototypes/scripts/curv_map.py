"""White-surface curvature on the stretches used so far: is it smooth and plausible?

Two boxes loaded from the full volume: the V1V2 box covers the open V1 stretch (k = 54), the
fused k = 27 stretch and the per-bank plane j = 91 (crop numbering), with >= 7.2 mm margin; the
M1S1 box is the test crop grown by 15 voxels. kappa_1, kappa_2 at every WM-boundary voxel
(curv_lib.curvature: a weighted quadric on the exposed WM faces of the
same side) at three scales; the views show the default scale against the
other two, on the user's planes, over echo 2 (trend removed), and in 3D on the white surface.

Usage: curv_map.py <outdir> <workdir>. Writes curv_<box>.npz (used by ev_closed.py) and
curv_map_<box>.png, curv_map_<box>_3d.png.
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

import curv_lib as C

OUT, S = sys.argv[1], sys.argv[2]
SIGS = (0.32, 0.48, 0.80)  # mm
SIG0 = 0.48
V1V2 = np.array((214, 37, 105))


def boxes():
    m = json.load(open(f"{S}/M1S1_test_points.json"))
    mlo = np.array(m["full_lo"])
    gap = np.array(m["gap_all"]) + mlo
    return {
        "V1V2": dict(lo=(201, 70, 65), hi=(344, 214, 205), planes=[
            ("open V1 stretch, crop k = 54", 2, 54 + V1V2[2], [np.array(q) + V1V2 for q in ((71, 89, 54), (84, 131, 54))]),
            ("per-bank plane, crop j = 91", 1, 91 + V1V2[1], [np.array(q) + V1V2 for q in ((71, 91, 18), (67, 91, 47), (38, 91, 50), (42, 91, 5))]),
            ("fused stretch, crop k = 27", 2, 27 + V1V2[2], [np.array(q) + V1V2 for q in ((32, 78, 27), (49, 114, 27))]),
        ]),
        "M1S1": dict(lo=tuple(mlo - 15), hi=tuple(np.array(m["full_hi"]) + 15), planes=[
            ("kissing stretch, test crop i = 45", 0, 45 + mlo[0], list(gap)),
            ("test crop i = 27", 0, 27 + mlo[0], list(gap)),
            ("test crop k = 89", 2, 89 + mlo[2], list(gap)),
        ]),
    }


def main():
    for name, bx in boxes().items():
        B = C.load(bx["lo"], bx["hi"], name)
        lo = B["lo"]
        print(f"{name}: box {B['rim'].shape}, WM boundary voxels {B['shell'].sum():,}")
        K = {sg: C.curvature(B, sg) for sg in SIGS}
        pts = K[SIG0]["pts"]
        inner = B["dface"][tuple(pts.T)] >= 6.0
        np.savez_compressed(f"{S}/curv_{name}.npz", lo=lo, pts=pts,
                            **{f"k1_{sg:g}": K[sg]["k1"] for sg in SIGS}, **{f"k2_{sg:g}": K[sg]["k2"] for sg in SIGS},
                            n=K[SIG0]["n"])
        for sg in SIGS:
            k1, k2 = K[sg]["k1"][inner], K[sg]["k2"][inner]
            Hm = 0.5 * (k1 + k2)
            print(f"  sigma {sg:.2f} mm: k1 p5/50/95 {np.percentile(k1, [5, 50, 95]).round(2)}, "
                  f"k2 {np.percentile(k2, [5, 50, 95]).round(2)}, mean curv {np.percentile(Hm, [5, 50, 95]).round(2)}; "
                  f"focal within 2.5 mm (k2 < -0.4): {np.mean(k2 < -0.4):.1%}")
        # roughness: difference between neighbouring boundary voxels, per scale
        idx = -np.ones(B["rim"].shape, np.int64); idx[tuple(pts.T)] = np.arange(len(pts))
        nb = []
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            q = pts + d
            ok = np.all(q < np.array(B["rim"].shape), 1)
            j = np.full(len(pts), -1); j[ok] = idx[tuple(q[ok].T)]
            sel = (j >= 0) & inner
            nb.append((np.flatnonzero(sel), j[sel]))
        for sg in SIGS:
            Hm = 0.5 * (K[sg]["k1"] + K[sg]["k2"])
            dd = np.concatenate([Hm[a] - Hm[b] for a, b in nb])
            print(f"  sigma {sg:.2f} mm: mean curvature, neighbour difference median |d| {np.median(np.abs(dd)):.3f}, "
                  f"spread (p5-p95 of the field) {np.subtract(*np.percentile(Hm[inner], [95, 5])):.2f}")

        # ---------- slices
        planes = bx["planes"]
        fig, axs = plt.subplots(len(planes), 5, figsize=(27, 5.6 * len(planes)))
        disp, ctx = B["disp"], B["ctx"]
        v = np.nanpercentile(np.abs(disp[ctx]), 98)
        cols = [("mean curvature (k1 + k2) / 2", SIG0, "H"), ("k1 (larger)", SIG0, "k1"), ("k2 (smaller)", SIG0, "k2"),
                ("mean curvature", SIGS[0], "H"), ("mean curvature", SIGS[2], "H")]
        for r, (lab, ax_, full, marks) in enumerate(planes):
            c0 = full - lo[ax_]
            on = pts[:, ax_] == c0
            other = [a for a in range(3) if a != ax_]
            sl = [slice(None)] * 3; sl[ax_] = c0; sl = tuple(sl)
            mk = np.array([np.array(q) - lo for q in marks])
            wlo = np.maximum(mk[:, other].min(0) - 40, 0)
            whi = np.minimum(mk[:, other].max(0) + 40, np.array(B["rim"].shape)[other] - 1)
            for c, (clab, sg, what) in enumerate(cols):
                a = axs[r, c]
                a.imshow(np.where(ctx[sl], disp[sl], np.nan).T, origin="lower", cmap="gray", vmin=-v, vmax=v)
                a.contour((B["dface"][sl] >= 6.0).T.astype(float), levels=[0.5], colors="w", linewidths=0.6, linestyles=":")
                kk = K[sg]
                val = {"H": 0.5 * (kk["k1"] + kk["k2"]), "k1": kk["k1"], "k2": kk["k2"]}[what][on]
                sc = a.scatter(pts[on, other[0]], pts[on, other[1]], c=val, s=16, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                               marker="s", linewidths=0)
                for q in marks:
                    q = np.array(q) - lo
                    if abs(q[ax_] - c0) <= 1:
                        a.plot(q[other[0]], q[other[1]], "x", c="lime", ms=8, mew=2)
                a.set_xlim(wlo[0], whi[0]); a.set_ylim(wlo[1], whi[1])
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(f"{clab}, scale {sg:.2f} mm", fontsize=12)
                if c == 0:
                    a.set_ylabel(f"{lab}\n(full-grid {'ijk'[ax_]} = {full})", fontsize=12)
            if r == 0:
                fig.colorbar(sc, ax=axs[:, -1], shrink=0.4, label="curvature (1/mm): red convex (crown), blue concave (fundus)")
        fig.suptitle(f"{name}: white-surface curvature at every WM-boundary voxel (3D), on the stretches' planes, over echo 2 "
                     "(trend removed). Dotted white: 6 mm from the box faces. Green x: the user's points in or next to the plane.",
                     fontsize=14)
        fig.savefig(f"{OUT}/curv_map_{name}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)

        # ---------- 3D: the white surface coloured by mean curvature (inner region only)
        wm_s = ndi.gaussian_filter(B["wm"].astype(float), 1.0)
        verts, faces, _, _ = marching_cubes(wm_s, 0.5, step_size=2)
        tree_idx = -np.ones(B["rim"].shape, np.int64); tree_idx[tuple(pts.T)] = np.arange(len(pts))
        dmap, ind = ndi.distance_transform_edt(tree_idx < 0, return_indices=True)
        vi = np.clip(np.round(verts).astype(int), 0, np.array(B["rim"].shape) - 1)
        near = tree_idx[tuple(ind[:, vi[:, 0], vi[:, 1], vi[:, 2]])]
        Hv = 0.5 * (K[SIG0]["k1"] + K[SIG0]["k2"])[near]
        keepv = B["dface"][tuple(vi.T)] >= 6.0
        fk = keepv[faces].all(1)
        faces = faces[fk]
        Hf = Hv[faces].mean(1)
        fig = plt.figure(figsize=(26, 12))
        for c, (el, az) in enumerate(((30, -60), (30, 120))):
            a = fig.add_subplot(1, 2, c + 1, projection="3d")
            P = verts * B["vox"]
            tp = P[faces]
            fnrm = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0])
            fnrm /= np.linalg.norm(fnrm, axis=1, keepdims=True) + 1e-12
            ev, az_ = np.radians(el), np.radians(az)
            light = np.array([np.cos(ev) * np.cos(az_), np.cos(ev) * np.sin(az_), np.sin(ev)])
            shade = 0.45 + 0.55 * np.abs(fnrm @ light)
            fcol = plt.cm.RdBu_r(np.clip((Hf + 1.0) / 2.0, 0, 1))
            fcol[:, :3] *= shade[:, None]
            a.add_collection3d(Poly3DCollection(tp, facecolors=fcol, linewidths=0))
            lo3, hi3 = P[np.unique(faces)].min(0), P[np.unique(faces)].max(0)
            a.set_xlim(lo3[0], hi3[0]); a.set_ylim(lo3[1], hi3[1]); a.set_zlim(lo3[2], hi3[2])
            tri = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1.0, 1.0))
            for lab, ax_, full, marks in planes:
                for q in marks:
                    q = (np.array(q) - lo) * B["vox"]
                    a.scatter(*q, c="lime", s=12, depthshade=False)
            a.view_init(el, az)
            a.set_box_aspect(np.ptp(P[np.unique(faces)], 0))
            a.set_xlabel("i (mm)"); a.set_ylabel("j (mm)"); a.set_zlabel("k (mm)")
        fig.colorbar(tri, ax=fig.axes, shrink=0.5, label="mean curvature (1/mm), scale 0.48 mm")
        fig.suptitle(f"{name}: white surface (>= 6 mm from the box faces) coloured by mean curvature; "
                     "green dots: the user's points", fontsize=14)
        fig.savefig(f"{OUT}/curv_map_{name}_3d.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        print("wrote", f"curv_map_{name}.png", f"curv_map_{name}_3d.png")


if __name__ == "__main__":
    main()
