"""Evaluate baillarger.fronts in its intended use on a complete patch (V1V2, M1S1, PCC_SPL).

The intended use: one anisotropic pass from all WM in the whole patch (onepass_lib), equal
weights, then the thickness field's weights, and the contacts read off the label boundary. No
change to the package; this measures what it does and costs. All 3D.

Reported:
- cost: time and peak memory of each stage (setup and patches, the passes, the contacts);
- reach: the fraction of cortex the fronts reach (the rest: cortex with no WM in its piece);
- thickness: arc length at column tops (fluid along the column, >= 2 mm from the faces);
  arrival against arc length; voxels with implausible arrival (> 5 mm);
- contacts: label boundaries with roots > 3 mm apart, classed for reporting by arrival over
  belief on the two sides (d/mu): contact-like (both 0.8-1.25), fold-like (both < 0.8),
  overlap-like (both > 1.25), mixed; how many lie within 2 mm of a box face;
- the sub-voxel location: the fraction of pairs where both labels' arrivals were known.
Weights as onepass_face.py: evidence from roots >= 2 mm from a face; a patch in a piece of the field's
graph with no evidence gets the equal-arrival weight (counted).
Views: three orthogonal planes through the patch centre with the contact classes over echo 2
(trend removed); histograms. ITK-SNAP: fronts_eval_<patch>_overlay.nii.gz with
fronts_eval_labels.txt over fronts_eval_<patch>_echo2_detrended.nii.gz.

Usage: fronts_eval.py <figures dir> <work dir> <data dir> <patch>
"""

import json
import resource
import sys
import time

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

import curv_lib as C
from baillarger.data import PATCHES
from collide import H_CLIP, Field, graph_L, lognormal, shell_graph
from onepass_lib import contact_points, contacts, open_thickness, run, setup

FIG, WORK, DATA, NAME = sys.argv[1:5]
RATIO, PATCH, SEP, FACE_MM = 3.0, 6, 18.0, 2.0
BLUE, ORANGE, AQUA, RED, GREY, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984", "#6250d6"
T0 = time.time()
LOG: dict = {}


def stage(name: str, t: float) -> float:
    now = time.time()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    LOG.setdefault("stages", []).append(dict(stage=name, seconds=round(now - t, 1), peak_gb=round(rss, 2)))
    print(f"   [{name}] {now - t:.1f} s; peak so far {rss:.2f} GB", flush=True)
    return now


def main():
    lo, hi = (np.array(v) for v in PATCHES[NAME])
    t = time.time()
    B = C.load(lo, hi, f"fronts_eval_{NAME}")
    vox = B["vox"]; h = float(np.mean(vox))
    wm, ctx = B["wm"], B["ctx"]
    fluid = B["rim"] == 1
    LOG.update(patch=NAME, shape=list(wm.shape), cortex_voxels=int(ctx.sum()), wm_voxels=int(wm.sum()))
    print(f"{NAME}: {tuple(wm.shape)}, cortex {ctx.sum():,} voxels", flush=True)
    t = stage("load", t)
    nh, coh = C.tensor_normal(B)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    g = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    sgn = np.sign((nh * gh).sum(-1)); sgn[sgn == 0] = 1
    dirv = coh[..., None] * nh * sgn[..., None] + 0.1 * gh
    dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
    del nh, g, gh
    t = stage("orientation", t)
    s = setup(wm, ctx, patch=PATCH, direction=dirv)
    LOG.update(seeds=int(s.seeds.sum()), patches=int(s.npatch))
    print(f"   seeds {s.seeds.sum():,}, patches {s.npatch:,}", flush=True)
    t = stage("setup and patches", t)

    # reach: cortex pieces with no seed next to them are unreachable
    comp, ncomp = ndi.label(s.domain, structure=ndi.generate_binary_structure(3, 1))
    has_seed = np.zeros(ncomp + 1, bool); has_seed[np.unique(comp[s.seeds])] = True
    LOG["cortex_without_wm_piece"] = float(np.mean(~has_seed[comp[ctx]]))

    # ---- unweighted pass, column tops, thickness field
    f0 = run(s, ratio=RATIO)
    t = stage("unweighted pass", t)
    reach = ctx & (f0.label >= 0)
    LOG["reach"] = float(reach.sum() / ctx.sum())
    arr_mm = f0.arrival * h; len_mm = f0.length * h
    LOG["arrival_over_5mm"] = float(np.mean(arr_mm[reach] > 5.0))
    LOG["length_over_arrival_p5_50_95"] = np.percentile(len_mm[reach & (arr_mm > 0.5)] / arr_mm[reach & (arr_mm > 0.5)],
                                                        [5, 50, 95]).round(3).tolist()
    fluid_f = fluid & (B["dface"] >= FACE_MM)
    mu_tops, has = open_thickness(s, fluid_f, RATIO, along=True)
    t = stage("column tops (second unweighted pass)", t)
    LOG["patches_with_tops"] = float(has.mean())
    LOG["top_thickness_mm_p5_50_95"] = np.percentile(mu_tops[has] * h, [5, 50, 95]).round(2).tolist()

    K = C.curvature(B, 0.8)
    pts, nrm = K["pts"], K["n"]
    t = stage("curvature", t)
    Hroot = np.clip(0.5 * (K["k1"] + K["k2"]), -H_CLIP, H_CLIP)
    ptsid = -np.ones(wm.shape, np.int64); ptsid[tuple(pts.T)] = np.arange(len(pts))
    side = np.argmax(np.abs(nrm), 1) * 2 + (nrm[np.arange(len(nrm)), np.argmax(np.abs(nrm), 1)] > 0)
    _, node_of, counts = np.unique(np.c_[pts // 2, side], axis=0, return_inverse=True, return_counts=True)
    node_of = node_of.ravel(); nN = node_of.max() + 1
    ppos = pts * vox
    npos = np.stack([np.bincount(node_of, ppos[:, a], nN) for a in range(3)], 1) / counts[:, None]
    nH = np.bincount(node_of, Hroot, nN) / counts
    Gc = shell_graph(pts, nrm, vox).tocoo()
    ea, eb = node_of[Gc.row], node_of[Gc.col]
    ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
    fld = Field(graph_L(npos, ed), nH)
    seed_pts = ptsid.ravel()[s.sflat]
    miss = seed_pts < 0
    if miss.any():
        from scipy.spatial import cKDTree

        seed_pts[miss] = cKDTree(pts).query(np.stack(np.unravel_index(s.sflat[miss], wm.shape), -1))[1]
    seed_patch = s.patch.ravel()[s.sflat]
    root_ok = B["dface"].ravel()[s.sflat] >= FACE_MM
    ok = has[seed_patch] & root_ok & (mu_tops[seed_patch] * h > 0.8) & (mu_tops[seed_patch] * h < 7.0)
    y = np.full(nN, np.nan)
    nd, val = node_of[seed_pts[ok]], np.log(mu_tops[seed_patch[ok]] * h)
    o = np.argsort(nd); nd, val = nd[o], val[o]
    st = np.flatnonzero(np.r_[True, np.diff(nd) != 0])
    for i, a in enumerate(st):
        y[nd[a]] = np.median(val[a: st[i + 1] if i + 1 < st.size else nd.size])
    mu_n, sd_n, coef = fld.fit(y, trend=True, nsamp=30)
    tmean, tsd = lognormal(mu_n, sd_n)
    cnode = node_of[seed_pts[s.rep]]
    mu_f, sd_f = tmean[cnode] / h, tsd[cnode] / h
    from scipy.sparse.csgraph import connected_components

    npc, pc = connected_components(graph_L(npos, ed) != 0, directed=False)
    ev_piece = np.bincount(pc[np.isfinite(y)], minlength=npc) > 0
    nobel = ~ev_piece[pc[cnode]]
    LOG.update(no_belief_patches=int(nobel.sum()), raw_mu_mm_max=float(mu_f.max() * h),
               raw_sd_mm_max=float(sd_f.max() * h))
    mu_f[nobel] = np.median(mu_f[~nobel]); sd_f[nobel] = np.median(sd_f[~nobel])
    LOG.update(field_nodes=int(nN), evidence_nodes=int(np.isfinite(y).sum()),
               field_mu_mm_p5_50_95=np.percentile(mu_f * h, [5, 50, 95]).round(2).tolist(),
               field_sd_mm_p5_50_95=np.percentile(sd_f * h, [5, 50, 95]).round(2).tolist(),
               a0=[float(v) for v in coef["a0"]])
    t = stage("thickness field (30 samples)", t)

    out = {}
    for wname, (mu, sig) in {"equal": (np.full(s.npatch, float(np.median(mu_f))), None),
                             "field": (mu_f, sd_f)}.items():
        sig = 0.1 * mu if sig is None else sig
        f = run(s, mu, sig, ratio=RATIO)
        t = stage(f"{wname} pass", t)
        c = contacts(s, f, mu, sig, sep_min=SEP)
        t = stage(f"{wname} contacts", t)
        cp = contact_points(c, wm.shape)
        ra, rb = c.da / mu[c.a], c.db / mu[c.b]
        cls = np.full(c.p.size, 4)  # mixed
        cls[(ra > 0.8) & (ra < 1.25) & (rb > 0.8) & (rb < 1.25)] = 1  # contact-like
        cls[(ra <= 0.8) & (rb <= 0.8)] = 2  # fold-like
        cls[(ra >= 1.25) & (rb >= 1.25)] = 3  # overlap-like
        dface = B["dface"].ravel()[c.p]
        out[wname] = dict(
            pairs=int(c.p.size), area_mm2=float(c.p.size * h * h),
            classes={k: int(np.sum(cls == v)) for k, v in (("contact-like", 1), ("fold-like", 2),
                                                           ("overlap-like", 3), ("mixed", 4))},
            near_face=float(np.mean(dface < FACE_MM)) if c.p.size else None,
            both_arrivals_known=float(np.mean(c.exact)) if c.p.size else None,
            sep_mm_p5_50_95=np.percentile(np.where(np.isinf(c.sep), 999, c.sep) * h, [5, 50, 95]).round(2).tolist()
            if c.p.size else None,
            d_over_mu_p5_50_95=np.percentile(np.r_[ra, rb], [5, 50, 95]).round(2).tolist() if c.p.size else None,
        )
        out[wname]["_viz"] = (f, c, cp, cls)
        print(f"   {wname}: {c.p.size:,} pairs; {out[wname]['classes']}; near a face {out[wname]['near_face']:.1%}",
              flush=True)
    LOG["runs"] = {k: {kk: vv for kk, vv in v.items() if kk != "_viz"} for k, v in out.items()}
    LOG["total_seconds"] = round(time.time() - T0, 1)
    json.dump(LOG, open(f"{WORK}/fronts_eval_{NAME}.json", "w"), indent=1, default=float)
    views(B, s, out, h)
    overlay(B, s, out)
    stage("views and overlay", t)
    print(json.dumps(LOG, indent=1, default=float))


def views(B, s, out, h):
    shape = np.array(B["wm"].shape)
    cen = shape // 2
    disp = np.where(B["ctx"], B["disp"], np.nan)
    lim = np.nanpercentile(np.abs(disp), 98)
    cols = {1: AQUA, 2: VIOLET, 3: RED, 4: ORANGE}
    labs = {1: "contact-like (both sides near their belief)", 2: "fold-like (both short)",
            3: "overlap-like (both past)", 4: "mixed"}
    fig, axs = plt.subplots(2, 3, figsize=(16, 11))
    for row, wname in enumerate(("equal", "field")):
        f, c, cp, cls = out[wname]["_viz"]
        for ax_fix in range(3):
            a = axs[row][ax_fix]
            sl = [slice(None)] * 3; sl[ax_fix] = cen[ax_fix]
            rest = [d for d in range(3) if d != ax_fix]
            a.imshow(np.where(B["wm"][tuple(sl)], 1.0, np.nan).T, origin="lower", cmap="Greys", vmin=0, vmax=2.5)
            a.imshow(disp[tuple(sl)].T, origin="lower", cmap="gray", vmin=-lim, vmax=lim)
            m = np.abs(cp[:, ax_fix] - cen[ax_fix]) <= 0.5
            for k, col in cols.items():
                q = m & (cls == k)
                a.plot(cp[q, rest[0]], cp[q, rest[1]], ".", color=col, ms=2, label=labs[k] if ax_fix == 0 else None)
            a.set_title(f"{wname} weights; {'ijk'[ax_fix]} = {cen[ax_fix] + 1} (ITK-SNAP)", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        axs[row][0].legend(fontsize=7, loc="lower left", markerscale=4)
    fig.suptitle(f"{NAME} (complete patch): label boundaries with roots > 3 mm apart, by d/mu on the two sides; "
                 "echo 2 trend removed, WM dark grey; computed in 3D", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fronts_eval_{NAME}_planes.png", dpi=100)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for wname, col in (("equal", GREY), ("field", BLUE)):
        f, c, cp, cls = out[wname]["_viz"]
        axs[0].hist(np.clip(np.where(np.isinf(c.sep), 60, c.sep) * h, 0, 60 * h), bins=60, histtype="step", color=col,
                    label=f"{wname} weights")
        axs[1].hist(np.clip((c.da + c.db) * h, 0, 10), bins=80, histtype="step", color=col, label=f"{wname} weights")
    axs[0].set_xlabel("separation of the two roots along the white surface (mm; > 9.6 shown at 9.6)")
    axs[1].set_xlabel("L = d_a + d_b at the boundary (mm)")
    for a in axs:
        a.set_ylabel("voxel pairs"); a.legend(fontsize=8)
    fig.suptitle(f"{NAME}: the contact set", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fronts_eval_{NAME}_hist.png", dpi=100)
    plt.close(fig)


def overlay(B, s, out):
    lab = np.zeros(B["wm"].shape, np.uint8)
    lab[s.seeds] = 1
    f, c, cp, cls = out["field"]["_viz"]
    for k in (4, 2, 3, 1):
        q = cls == k
        lab.ravel()[c.p[q]] = 1 + k
        lab.ravel()[c.q[q]] = 1 + k
    aff = B["p"].affine
    nib.save(nib.Nifti1Image(lab, aff), f"{DATA}/fronts_eval_{NAME}_overlay.nii.gz")
    nib.save(nib.Nifti1Image(np.where(B["ctx"], B["disp"], 0.0).astype(np.float32), aff),
             f"{DATA}/fronts_eval_{NAME}_echo2_detrended.nii.gz")
    with open(f"{DATA}/fronts_eval_labels.txt", "w") as fh:
        fh.write("# ITK-SNAP label description file (field weights)\n")
        fh.write('    0     0    0    0        0  0  0    "Clear Label"\n')
        for i, (rgb, nm) in enumerate((((150, 150, 150), "WM boundary (seeds)"),
                                       ((27, 175, 122), "contact-like"), ((98, 80, 214), "fold-like"),
                                       ((227, 73, 72), "overlap-like"), ((235, 104, 52), "mixed")), start=1):
            fh.write(f'  {i:3d}  {rgb[0]:3d}  {rgb[1]:3d}  {rgb[2]:3d}        1  1  1    "{nm}"\n')


if __name__ == "__main__":
    main()
