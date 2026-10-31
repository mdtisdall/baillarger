"""The base pial surface as a contour (plan: reports/Pial editing and edit priority.md,
step 4 follow-up). All 3D.

The pial surface starts from a surface, not from per-column tops:
- open cortex: the label's own boundary between tissue (WM, cortex, missing tissue) and
  fluid;
- fused cortex: the one-pass competition's contact (baillarger.fronts with equal weights,
  root patches of 6 voxels, cost ratio 3 across the data's layer normal, as finding 27):
  label boundaries between root patches more than 18 voxels apart along the white
  surface, at sub-voxel positions (onepass_lib.contacts). Left out:
  - pairs within 2 mm of a box face;
  - fold-like pairs: both sides reach the boundary before 0.8 of their patch's expected
    thickness (finding 27's class), the expectation being the patch's visible tops
    (onepass_lib.open_thickness: the median arc length at cortex voxels where the front
    runs into fluid; patches without tops take the nearest patch's).
  The kept contacts are carved out of the tissue as a slit, in two variants:
  - slit1: the voxel of each pair nearest the sub-voxel contact (the two sheets end
    about half a voxel either side of it);
  - slit2: both voxels of each pair (about one voxel either side of the label boundary).
The tissue with its slits is contoured and smoothed as the white surface is
(pial_height.white_surface: binary marching tetrahedra, tangential relaxation, twicing at
0.32 mm).

Reported per variant: vertices, topology, self-intersecting triangles (all, and >= 2 mm
from the box faces), the contacts by class. Written to <data dir>:
- pial_<patch>_contour_<variant>.vtk;
- pial_<patch>_contour_overlay.nii.gz with pial_contour_labels.txt (ITK-SNAP): contact
  voxels by class and whether carved;
- pial_<patch>_contour_session.npz (the editor's surface session: white and pial meshes);
- pial_<patch>_contour.json.
Views: <figures dir>/pial_contour_<patch>_<variant>.png, the same three planes as
pial_base_label.png: white surface, the label base's top, the contoured surface.

Usage: pial_contour_base.py <figures dir> <data dir> <patch>
"""

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

import curv_lib as C
import pial_height as P
from baillarger.data import PATCHES
from baillarger.mesh import topology, write_vtk
from baillarger.mesh_intersect import intersecting_faces
from onepass_lib import contact_points, contacts, open_thickness, run, setup

FIG, DATA, NAME = sys.argv[1:4]
RATIO, PATCH, SEP, FACE_MM = 3.0, 6, 18.0, 2.0
T0 = time.time()
LOG: dict = {"patch": NAME, "stages": []}


def stage(name, t):
    now = time.time()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    LOG["stages"].append(
        dict(stage=name, seconds=round(now - t, 1), peak_gb=round(peak, 2))
    )
    print(f"[{now - T0:6.0f}s] {name}: {now - t:.1f} s; peak {peak:.2f} GB", flush=True)
    return now


def main():
    lo, hi = (np.array(v) for v in PATCHES[NAME])
    t = time.time()
    B = C.load(lo, hi, f"pial_contour_{NAME}")
    vox, aff = B["vox"], B["p"].affine
    wm, ctx, miss, rim = B["wm"], B["ctx"], B["miss"], B["rim"]
    fluid = (rim == 1) & ~miss
    dface = B["dface"]
    t = stage("load", t)

    # ---- orientation and the one pass (as fronts_eval.py, equal weights)
    nh, coh = C.tensor_normal(B)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    gr = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gh = gr / (np.linalg.norm(gr, axis=-1, keepdims=True) + 1e-12)
    sgn = np.sign((nh * gh).sum(-1))
    sgn[sgn == 0] = 1
    dirv = coh[..., None] * nh * sgn[..., None] + 0.1 * gh
    dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
    del nh, coh, gr, gh, sgn, d_wm
    s = setup(wm, ctx, patch=PATCH, direction=dirv)
    f = run(s, ratio=RATIO)
    c = contacts(s, f, sep_min=SEP)
    mu, has = open_thickness(s, fluid & (dface >= FACE_MM), RATIO, along=True)
    t = stage(f"one pass and contacts: {c.p.size:,} pairs", t)

    ra, rb = c.da / mu[c.a], c.db / mu[c.b]
    cls = np.full(c.p.size, 4)  # mixed
    cls[(ra > 0.8) & (ra < 1.25) & (rb > 0.8) & (rb < 1.25)] = 1  # contact-like
    cls[(ra <= 0.8) & (rb <= 0.8)] = 2  # fold-like
    cls[(ra >= 1.25) & (rb >= 1.25)] = 3  # overlap-like
    near = (dface.ravel()[c.p] < FACE_MM) | (dface.ravel()[c.q] < FACE_MM)
    carve = (cls != 2) & ~near
    names = {1: "contact-like", 2: "fold-like", 3: "overlap-like", 4: "mixed"}
    LOG["contacts"] = dict(
        pairs=int(c.p.size),
        near_box_face=int(near.sum()),
        classes={names[k]: int(np.sum(cls == k)) for k in names},
        classes_away_from_faces={
            names[k]: int(np.sum((cls == k) & ~near)) for k in names
        },
        carved=int(carve.sum()),
        patches_with_tops=round(float(has.mean()), 3),
        expected_thickness_mm_p5_50_95=np.percentile(mu * vox.mean(), [5, 50, 95])
        .round(2)
        .tolist(),
    )
    print(json.dumps(LOG["contacts"]), flush=True)

    # ---- slits
    tissue = (rim == 2) | (rim == 3) | miss
    pc, qc = c.p[carve], c.q[carve]
    nearest = np.where(c.frac[carve] < 0.5, pc, qc)
    slits = {"slit1": nearest, "slit2": np.concatenate([pc, qc])}
    ov = np.zeros(wm.shape, np.uint8)
    ov.ravel()[c.p[cls == 2]] = 3
    ov.ravel()[c.q[cls == 2]] = 3
    ov.ravel()[c.p[near]] = 4
    ov.ravel()[c.q[near]] = 4
    ov.ravel()[np.concatenate([pc, qc])] = 2
    ov.ravel()[nearest] = 1

    # the label base for comparison (from the session file of pial_height_run.py)
    z = np.load(f"{DATA}/pial_{NAME}_session.npz")
    cols = P.Columns(
        z["pts"],
        z["cnt"].astype(np.int64),
        z["s"].astype(np.float64),
        z["ridge_s"],
        z["fluid_s"],
        z["stop_s"],
        z["stop_why"].astype(np.int64),
        z["extended"],
    )
    wv, wf = z["v"], z["f"]
    base = P.point_at(cols, z["t_label"])
    bad = ~np.isfinite(base).all(1)
    base[bad] = wv[bad]
    del cols, z

    img = np.where(ctx, B["disp"], 0.0).astype(np.float32)
    LOG["variants"] = {}
    meshes = {}
    for name, idx in slits.items():
        t = time.time()
        tis = tissue.copy()
        tis.ravel()[idx] = False
        v, fc, n = P.white_surface(tis)
        vi = np.clip(np.round(v).astype(int), 0, np.array(wm.shape) - 1)
        nf = dface[tuple(vi.T)] < FACE_MM
        xf, nx = intersecting_faces(v, fc)
        inside = ~nf[fc].any(1)
        topo = topology(fc, len(v))
        r = dict(
            vertices=int(len(v)),
            faces=int(len(fc)),
            carved_voxels=int(np.unique(idx).size),
            topology={
                k: (float(x) if isinstance(x, float) else x) for k, x in topo.items()
            },
            intersecting_faces=int(xf.sum()),
            intersecting_faces_inside=int((xf & inside).sum()),
        )
        LOG["variants"][name] = r
        meshes[name] = (v, fc, n, nf)
        print(name, json.dumps(r), flush=True)
        write_vtk(
            f"{DATA}/pial_{NAME}_contour_{name}.vtk",
            aff,
            v,
            fc,
            {
                "near_box_face": nf.astype(float),
                "intersecting": np.bincount(fc[xf].ravel(), minlength=len(v))
                .clip(0, 1)
                .astype(float),
            },
        )
        # view
        cen = np.array(wm.shape) // 2
        fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
        for a, ax in enumerate(axs):
            sl = [slice(None)] * 3
            sl[a] = cen[a]
            o = [i for i in range(3) if i != a]
            ax.imshow(
                img[tuple(sl)].T,
                cmap="gray",
                origin="lower",
                vmin=-0.25,
                vmax=0.25,
                interpolation="nearest",
            )
            for pts, ff, col, lw in (
                (wv, wf, "#1baf7a", 0.8),
                (base, wf, "#eb6834", 0.6),
                (v, fc, "#2a78d6", 1.0),
            ):
                for sg in P.plane_section(pts, ff, a, float(cen[a])):
                    ax.plot(sg[:, o[0]], sg[:, o[1]], color=col, lw=lw)
            ax.set_title(f"plane {'ijk'[a]} = {cen[a]}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{NAME}: white surface (green), label base top (orange), contoured pial, {name} (blue); "
            "echo 2, trend removed",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(f"{FIG}/pial_contour_{NAME}_{name}.png", dpi=110)
        plt.close(fig)
        stage(name, t)

    nib.save(nib.Nifti1Image(ov, aff), f"{DATA}/pial_{NAME}_contour_overlay.nii.gz")
    with open(f"{DATA}/pial_contour_labels.txt", "w") as fh:
        fh.write('# ITK-SNAP label description file\n0 0 0 0 0 0 0 "Clear Label"\n')
        fh.write('1 42 120 214 1 1 1 "contact, carved (slit1 and slit2)"\n')
        fh.write('2 27 175 122 1 1 1 "contact, carved (slit2 only)"\n')
        fh.write('3 98 80 214 1 1 1 "fold-like boundary, not carved"\n')
        fh.write(
            '4 138 137 132 1 1 1 "boundary within 2 mm of a box face, not carved"\n'
        )
    for name, (v, fc, n, nf) in meshes.items():
        np.savez(
            f"{DATA}/pial_{NAME}_contour_{name}_session.npz",
            kind="surface",
            white_v=wv,
            white_f=wf,
            v=v,
            f=fc,
            n=n,
            near_face=nf,
            affine=aff,
        )
    with open(f"{DATA}/pial_{NAME}_contour.json", "w") as fh:
        json.dump(LOG, fh, indent=1)


if __name__ == "__main__":
    main()
