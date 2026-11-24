"""Pial-as-height builder on a whole patch (plan: reports/Pial editing and edit priority.md,
§6 step 1). All 3D.

For the patch (a box of baillarger.data.PATCHES, loaded as fronts_eval.py does):
- the white surface (finding 28: twicing at 0.32 mm);
- the layer orientation and the metric direction as fronts_eval.py (data normal signed by
  the WM-distance gradient, weight coherence, plus 0.1 x that gradient), cost ratio 3;
- one unweighted march from the white mesh through cortex and fluid, carrying arc length
  and root (pial_height.march_from_mesh);
- the columns from every white vertex, with the extensions of plan §2;
- tops of two base models on the same columns (placeholders until the user picks the base
  models, plan §1): the label boundary (fluid, else the ridge of arrival) and the one pass
  with equal weights (finding 27's setup: root patches of 6 voxels, separation 18 voxels).

Written to <data dir>:
- pial_<patch>_white.vtk: the white mesh, with per-vertex arrays (box-face flag, ridge and
  fluid arc lengths in mm, column stop reason, extended flag);
- pial_<patch>_label.vtk, pial_<patch>_onepass_equal.vtk: the white mesh's topology with
  each vertex at its model's top; arrays t (mm), how the top was placed, and |difference|
  from the other model (mm);
- pial_<patch>_overlay.nii.gz with pial_labels.txt (ITK-SNAP): the voxels nearest to the
  vertices of each surface; pial_<patch>_echo2_detrended.nii.gz to view it over;
- pial_<patch>.json: counts and percentiles;
- pial_<patch>_session.npz: the white mesh, columns, carried fields and both models' tops,
  for the editor (pial_edit/editor_server.py).
Figure: <figures dir>/pial_<patch>_sections.png (three planes through the patch centre: echo 2
trend removed, with the white surface and both models' tops).

Usage: pial_height_run.py <figures dir> <data dir> <patch>
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
from baillarger.mesh import write_vtk
from onepass_lib import run, setup

FIG, DATA, NAME = sys.argv[1:4]
RATIO, PATCH, SEP, FACE_MM = 3.0, 6, 18.0, 2.0
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
T0 = time.time()
LOG: dict = {"patch": NAME, "stages": []}


def stage(name: str, t: float) -> float:
    now = time.time()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    LOG["stages"].append(dict(stage=name, seconds=round(now - t, 1), peak_gb=round(peak, 2)))
    print(f"[{now - T0:6.0f}s] {name}: {now - t:.1f} s; peak {peak:.2f} GB", flush=True)
    return now


def pct(x, q=(5, 25, 50, 75, 95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [round(float(v), 3) for v in np.percentile(x, q)] if x.size else None


def main():
    lo, hi = (np.array(v) for v in PATCHES[NAME])
    t = time.time()
    B = C.load(lo, hi, f"pial_{NAME}")
    vox = B["vox"]
    wm, ctx, miss = B["wm"], B["ctx"], B["miss"]
    fluid = (B["rim"] == 1) & ~miss
    aff = B["p"].affine
    LOG.update(shape=list(wm.shape), cortex_voxels=int(ctx.sum()))
    t = stage("load", t)

    # ---- orientation (as fronts_eval.py)
    nh, coh = C.tensor_normal(B)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    gr = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gh = gr / (np.linalg.norm(gr, axis=-1, keepdims=True) + 1e-12)
    sgn = np.sign((nh * gh).sum(-1))
    sgn[sgn == 0] = 1
    dirv = coh[..., None] * nh * sgn[..., None] + 0.1 * gh
    dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
    layer = np.where(ctx[..., None], nh, 0.0)
    del gr, gh, sgn, d_wm
    t = stage("orientation", t)

    # ---- white surface, march, columns
    v, f, n = P.white_surface(wm)
    vi = np.clip(np.round(v).astype(int), 0, np.array(wm.shape) - 1)
    near_face = B["dface"][tuple(vi.T)] < FACE_MM
    LOG.update(vertices=int(len(v)), faces=int(len(f)), vertices_near_box_face=int(near_face.sum()))
    t = stage(f"white surface: {len(v):,} vertices", t)
    g = P.march_from_mesh(v, f, wm, ctx, fluid, dirv, RATIO)
    t = stage("march", t)
    Cf = P.characteristic_field(g)
    cols = P.trace_columns(v, n, g, Cf, layer)
    del Cf
    t = stage("columns", t)

    # ---- tops
    t_lab, k_lab = P.top_label(cols, fluid)
    t = stage("label top", t)
    s = setup(wm, ctx, patch=PATCH, direction=dirv)
    fr = run(s, ratio=RATIO)
    keys = P.near_patches(s, SEP)
    t_one, k_one = P.top_onepass(cols, fr.label, keys, s.npatch, fluid)
    LOG.update(root_patches=int(s.npatch))
    del fr, s
    t = stage("one pass (equal weights) top", t)

    # ---- summary (vertices >= 2 mm from the box faces)
    keep = ~near_face
    h = P.H_MM
    diff = np.abs(t_lab - t_one) * h
    LOG["columns"] = dict(
        stop={P.STOP_NAMES[k]: int(np.sum(keep & (cols.stop_why == k))) for k in P.STOP_NAMES},
        extended_past_ridge=int(np.sum(keep & cols.extended)),
        reached_fluid=int(np.sum(keep & np.isfinite(cols.fluid_s))),
        length_mm=pct(cols.stop_s[keep] * h),
    )
    for nm, tt, kk in (("label", t_lab, k_lab), ("onepass_equal", t_one, k_one)):
        LOG[nm] = dict(
            kind={P.TOP_NAMES[k]: int(np.sum(keep & (kk == k))) for k in P.TOP_NAMES},
            thickness_mm=pct(tt[keep] * h),
            thickness_mm_by_kind={P.TOP_NAMES[k]: pct(tt[keep & (kk == k)] * h) for k in P.TOP_NAMES},
        )
    LOG["difference_mm"] = dict(
        all=pct(diff[keep], (50, 75, 90, 95, 99)),
        over_0p16=round(float(np.mean(diff[keep] > 0.16)), 3),
        over_0p5=round(float(np.mean(diff[keep] > 0.5)), 3),
        both_fluid=pct(diff[keep & (k_lab == P.TOP_FLUID) & (k_one == P.TOP_FLUID)], (50, 95, 99)),
    )

    # ---- outputs
    Pl = P.point_at(cols, t_lab)
    Po = P.point_at(cols, t_one)
    for arr in (Pl, Po):  # columns too short to reach a top: keep the white vertex
        bad = ~np.isfinite(arr).all(1)
        arr[bad] = v[bad]
    face = near_face.astype(float)
    write_vtk(f"{DATA}/pial_{NAME}_white.vtk", aff, v, f, {
        "near_box_face": face, "ridge_mm": cols.ridge_s * h, "fluid_mm": cols.fluid_s * h,
        "stop_why": cols.stop_why.astype(float), "extended": cols.extended.astype(float)})
    write_vtk(f"{DATA}/pial_{NAME}_label.vtk", aff, Pl, f, {
        "t_mm": t_lab * h, "kind": k_lab.astype(float), "diff_mm": diff, "near_box_face": face})
    write_vtk(f"{DATA}/pial_{NAME}_onepass_equal.vtk", aff, Po, f, {
        "t_mm": t_one * h, "kind": k_one.astype(float), "diff_mm": diff, "near_box_face": face})
    ov = np.zeros(wm.shape, np.uint8)
    for lab, pts in ((1, v), (2, Pl), (3, Po)):
        q = np.clip(np.round(pts).astype(int), 0, np.array(wm.shape) - 1)
        ov[tuple(q.T)] = lab
    nib.save(nib.Nifti1Image(ov, aff), f"{DATA}/pial_{NAME}_overlay.nii.gz")
    nib.save(nib.Nifti1Image(np.where(ctx, B["disp"], 0.0).astype(np.float32), aff),
             f"{DATA}/pial_{NAME}_echo2_detrended.nii.gz")
    with open(f"{DATA}/pial_labels.txt", "w") as fh:
        fh.write("# ITK-SNAP label description file\n")
        fh.write('0 0 0 0 0 0 0 "Clear Label"\n')
        fh.write(f'1 27 175 122 1 1 1 "white surface"\n')
        fh.write(f'2 42 120 214 1 1 1 "top: label boundary (fluid, else ridge)"\n')
        fh.write(f'3 235 104 52 1 1 1 "top: one pass, equal weights"\n')
    # everything the editor's session needs (pial_edit/editor_server.py), uncompressed
    np.savez(f"{DATA}/pial_{NAME}_session.npz", v=v, f=f, n=n, affine=aff, shape=np.array(wm.shape),
             near_face=near_face, pts=cols.pts, cnt=cols.n, s=cols.s.astype(np.float32),
             ridge_s=cols.ridge_s, fluid_s=cols.fluid_s, stop_s=cols.stop_s, stop_why=cols.stop_why,
             extended=cols.extended, length=g.length.astype(np.float32), root=g.root.astype(np.float32),
             t_label=t_lab, kind_label=k_lab, t_onepass_equal=t_one, kind_onepass_equal=k_one)
    t = stage("outputs", t)

    # ---- figure: three planes through the centre
    c = np.array(wm.shape) // 2
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
    img = np.where(ctx, B["disp"], 0.0)
    for a, ax in enumerate(axs):
        sl = [slice(None)] * 3
        sl[a] = c[a]
        other = [i for i in range(3) if i != a]
        ax.imshow(img[tuple(sl)].T, cmap="gray", origin="lower", vmin=-0.25, vmax=0.25,
                  interpolation="nearest")
        for pts, col, lw in ((v, AQUA, 0.8), (Pl, BLUE, 0.8), (Po, ORANGE, 0.8)):
            seg = P.plane_section(pts, f, a, float(c[a]))
            for sgm in seg:
                ax.plot(sgm[:, other[0]], sgm[:, other[1]], color=col, lw=lw)
        ax.set_title(f"plane {'ijk'[a]} = {c[a]}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{NAME}: white surface (green), top at the label boundary or the ridge (blue), "
                 "top of the one pass with equal weights (orange); echo 2, trend removed", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/pial_{NAME}_sections.png", dpi=110)
    plt.close(fig)
    stage("figure", t)

    with open(f"{DATA}/pial_{NAME}.json", "w") as fh:
        json.dump(LOG, fh, indent=1)
    print(json.dumps({k: LOG[k] for k in ("columns", "label", "onepass_equal", "difference_mm")}, indent=1))


if __name__ == "__main__":
    main()
