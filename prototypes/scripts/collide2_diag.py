"""Why the posterior misses at some of the user's points: each front's column at each point, traced
down its own T to its root, with the thickness belief at the root and the evidence around it.

Prints, per point: each front's arrival T at the point, its root (ITK-SNAP numbering of the test crop),
the root's white-surface curvature, predicted thickness mu +- SD, the local curvature coefficient, and
the evidence within 3 mm of the root (count, median, and where it came from).
Writes an ITK-SNAP overlay with the earlier labels plus: 6 front-1 columns, 7 front-2 columns,
8 roots, 9 tops used as evidence through front 1, 10 through front 2, 11 through the all-WM field.

Usage: collide2_diag.py <datadir> <workdir>. Needs collide2_state.npz, collide_state.npz, curv_M1S1.npz.
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

import flatten_fused2 as F
from collide import GROW
from collide2_itksnap import surface
from fit_T import fill

OUT, S = sys.argv[1], sys.argv[2]
VOX = np.array([0.16028, 0.16, 0.16028])
SRC = {1: "front 1", 2: "front 2", 3: "all-WM"}


def descend(Tf, g, x, wall_mask):
    pts = [x.copy()]
    shp = np.array(Tf.shape) - 1
    for _ in range(600):
        v = -np.array([F.interp(c, x) for c in g])
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            break
        x = x + 0.25 * v / nv
        pts.append(x.copy())
        if wall_mask[tuple(np.clip(np.round(x).astype(int), 0, shp))]:
            break
    return np.array(pts)


def main():
    m = json.load(open(f"{S}/M1S1_test_points.json"))
    st = np.load(f"{S}/collide2_state.npz")
    cs = np.load(f"{S}/collide_state.npz")
    cz = np.load(f"{S}/curv_M1S1.npz")
    pts = cz["pts"]
    node_of = cs["node_of"]
    root_all = cs["root"]
    tmean, tsd, a, nH, y = st["tmean"], st["tsd"], st["a"], st["nH"], st["y"]
    ev_src = st["ev_src"]
    P = np.array(m["gap_all"], float) + GROW
    shape = st["T1"].shape
    itk = lambda v: tuple(int(c) for c in np.round(np.asarray(v) - GROW + 1))
    seg_extra = np.zeros(shape, np.uint8)
    walls = {}
    fields = {}
    for X in ("1", "2"):
        T = st[f"T{X}"].astype(float)
        Tf = fill(T)
        g = np.gradient(Tf, *VOX)
        wm = np.zeros(shape, bool)
        wm[tuple(pts[st[f"wall{X}"]].T)] = True
        walls[X] = wm
        fields[X] = (Tf, g, st[f"root{X}"])
    # evidence tops and their roots (through whichever field supplied them)
    tq = np.argwhere(ev_src > 0)
    src = ev_src[tuple(tq.T)]
    troot = np.where(
        src == 1,
        st["root1"][tuple(tq.T)],
        np.where(src == 2, st["root2"][tuple(tq.T)], root_all[tuple(tq.T)]),
    )
    tval = np.where(
        src == 1,
        st["S1"][tuple(tq.T)],
        np.where(src == 2, st["S2"][tuple(tq.T)], st["S_all"][tuple(tq.T)]),
    )
    okr = troot >= 0
    tq, src, troot, tval = tq[okr], src[okr], troot[okr], tval[okr]
    rpos = pts[troot] * VOX
    print(
        "point | front: T at point, root (ITK), H, mu +- SD, a | evidence within 3 mm of the root: n, median, by source"
    )
    for k, q in enumerate(P):
        line = f"{k + 1:2d} ITK {itk(q)}"
        for X, lab in (("1", 6), ("2", 7)):
            Tf, g, root = fields[X]
            path = descend(Tf, g, q.copy(), walls[X])
            for v in np.round(path).astype(int):
                seg_extra[tuple(np.clip(v, 0, np.array(shape) - 1))] = lab
            r = root[tuple(np.round(q).astype(int))]
            end = np.round(path[-1]).astype(int)
            wl = np.flatnonzero(st[f"wall{X}"])
            r_end = wl[np.argmin(np.linalg.norm((pts[wl] - end) * VOX, axis=1))]
            plen = np.sum(np.linalg.norm(np.diff(path, axis=0) * VOX, axis=1))
            line += (
                f" | front {X} traced: path {plen:.2f} mm to {itk(pts[r_end])}"
                f" (upwind root {np.linalg.norm((pts[max(r, 0)] - pts[r_end]) * VOX):.2f} mm away; "
                f"mu there {tmean[node_of[r_end]]:.2f}+-{tsd[node_of[r_end]]:.2f})"
            )
            if r < 0:
                line += f" | front {X}: no root"
                continue
            seg_extra[tuple(pts[r])] = 8
            n_ = node_of[r]
            near = np.linalg.norm(rpos - pts[r] * VOX, axis=1) < 3.0
            bysrc = ", ".join(
                f"{SRC[s_]} {np.sum(near & (src == s_))} ({np.median(tval[near & (src == s_)]):.2f})"
                for s_ in (1, 2, 3)
                if np.any(near & (src == s_))
            )
            line += (
                f" | front {X}: S {F.interp(fill(st[f'S{X}'].astype(float)), q):.2f}, root {itk(pts[r])}, H {nH[n_]:+.2f}, "
                f"mu {tmean[n_]:.2f}+-{tsd[n_]:.2f}, a {a[n_]:+.2f} | ev n {near.sum()}, "
                f"median {np.median(tval[near]) if near.any() else np.nan:.2f}: {bysrc}"
            )
        print(line)
    # overlay: the earlier labels plus columns, roots and evidence
    crop = tuple(slice(GROW, -GROW) for _ in range(3))
    base = nib.load(f"{OUT}/M1S1_contact_overlay.nii.gz")
    seg = np.asarray(base.dataobj).astype(np.uint8)
    full = np.zeros(shape, np.uint8)
    full[crop] = seg
    for s_, lab in ((1, 9), (2, 10), (3, 11)):
        full[tuple(tq[src == s_].T)] = lab
    keep = (full == 4) | (full == 5)  # the points stay on top
    full = np.where((seg_extra > 0) & ~keep, seg_extra, full)
    nib.save(
        nib.Nifti1Image(full[crop], base.affine),
        f"{OUT}/M1S1_contact_diag_overlay.nii.gz",
    )
    open(f"{OUT}/M1S1_contact_diag_labels.txt", "w").write(
        open(f"{OUT}/M1S1_contact_labels.txt").read()
        + '    6    40  120  255        1  1  1    "Front 1 columns (from the points)"\n'
        '    7   255  140   20        1  1  1    "Front 2 columns (from the points)"\n'
        '    8   255  255  255        1  1  1    "Roots of those columns"\n'
        '    9   120  170  255      0.6  1  1    "Evidence tops via front 1"\n'
        '   10   255  190  120      0.6  1  1    "Evidence tops via front 2"\n'
        '   11   170  170  170      0.6  1  1    "Evidence tops via the all-WM field"\n'
    )
    print("wrote M1S1_contact_diag_overlay.nii.gz, M1S1_contact_diag_labels.txt")


if __name__ == "__main__":
    main()
