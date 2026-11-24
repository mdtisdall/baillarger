"""The one-pass weighted competition at a marked face (plan, Next step 0), against findings 11-19.

Replaces the walls, the per-front fields and the sheet classes of collide.py / collide2.py with one
anisotropic fast-marching pass from all WM (onepass_lib). All 3D.

1. Box: as site_setup.py (the V1V2 banks crop; a box of 141 re-centred on the points for PCC_SPL
   pieces 1 and 2 and for M1S1); every point at least 6 mm from a face. No gap closure (finding 16).
2. Metric: unit cost along the data's layer normal, signed and blended as fit_all_wm (coherence-weighted
   normal plus 0.1 x the WM-distance gradient), cost ratio 3 across it.
3. Labels: root patches about 1 mm across (onepass_lib.setup). Contact: a label boundary whose patches
   are more than 3 mm apart along the white surface.
4. Thickness field (collide.py's Field: log t = m + (a0 + w) H + z on the WM-boundary nodes): evidence is
   the arc length of an unweighted pass at column tops (fluid along the column, >= 2 mm from the faces),
   per patch, given to the nodes of that patch's seeds that are >= 2 mm from a face. Each patch's
   (mu, sigma) is the lognormal predictive at its centre's node; a patch whose piece of the field's graph
   holds no evidence has no belief and gets the equal-arrival weight (median mu and sigma).
5. Passes: equal (equal arrival); field (start -mu/sigma, speed sigma); field mu with sigma = 0.1 mu.
6. Score at each point (onepass_lib.score_points_sides): the two sides are found locally, as the labels
   at the two ends of the column line through the point (2.5 mm either way), every patch going to the end
   nearer along the white surface; the crossings are the label boundaries (all of them, no separation
   threshold) between the two groups within 1 mm of the point and 0.4 mm of the line. Offset (mm) along
   the line, + toward the side the field predicts thinner. Per side: mu - S at the point. The score with
   the 3 mm threshold instead is reported beside it: the threshold drops contacts whose roots are close
   along the white surface (near a fundus, as at the V1V2 banks), and at 1 mm it picks up boundaries
   between neighbouring patches (biased toward zero).

Distances are in voxels inside, reported in mm (x 0.1602, the mean voxel size).

Usage: onepass_face.py <figures dir> <work dir> <data dir> <site>
  site: M1S1 | V1V2_banks | PCC_SPL_p1 | PCC_SPL_p2
"""

import argparse
import json
import sys

# collide.py reads OUT and S from argv[1] and argv[2] at import; the scripts live in the work dir
sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

import curv_lib as C
from collide import H_CLIP, Field, graph_L, lognormal, shell_graph
from onepass_lib import contact_points, contacts, open_thickness, run, score_points, score_points_sides, setup

ap = argparse.ArgumentParser()
ap.add_argument("fig"); ap.add_argument("work"); ap.add_argument("data"); ap.add_argument("site")
A = ap.parse_args()
FULL = np.array([700, 384, 1148])
HALF = 70
RATIO = float(__import__("os").environ.get("RATIO", "3.0"))  # cost across the layer normal
PATCH = 6
SEP = float(__import__("os").environ.get("SEP_MM", "2.88")) / 0.1602  # contact: roots this far apart (voxels)
TAG = __import__("os").environ.get("TAG", "")  # suffix for the outputs of a variant
FACE_MM = 2.0
BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984"


def box_and_points(site):
    if site == "M1S1":
        # the test crop grown by 15 (findings 11-16) leaves a point 4.8 mm from a face; a box of 141
        # re-centred on the points keeps every point >= 6 mm inside
        m = json.load(open(f"{A.data}/M1S1_test_points.json"))
        full = np.array(m["gap_all"]) + np.array(m["full_lo"])
        c = np.round(full.mean(0)).astype(int)
        lo, hi = c - HALF, c + HALF + 1
        assert np.all(lo >= 0) and np.all(hi <= FULL)
        return lo, hi, (full - lo).astype(float)
    name, piece, centre = {
        "V1V2_banks": ("V1V2_banks_points.json", None, False),
        "PCC_SPL_p1": ("PCC_SPL_points.json", "1", True),
        "PCC_SPL_p2": ("PCC_SPL_p2_points.json", None, True),
    }[site]
    pf = json.load(open(f"{A.data}/{name}"))
    crop = pf["crop"].replace("_echo2_detrended.nii.gz", "")
    box = json.load(open(f"{A.data}/{crop}_sites.json"))
    pts = pf["pieces"][piece]["contact_points"] if piece else pf["contact_points"]
    # sorted and unique, as site_setup.py: the point numbers of findings 18 and 19
    full = np.unique(np.array(pts, int), axis=0) - 1 + np.array(box["full_lo"])
    if centre:
        c = np.round(full.mean(0)).astype(int)
        lo, hi = c - HALF, c + HALF + 1
    else:
        lo, hi = np.array(box["full_lo"]), np.array(box["full_hi"])
    assert np.all(lo >= 0) and np.all(hi <= FULL)
    return lo, hi, (full - lo).astype(float)


def main():
    lo, hi, P = box_and_points(A.site)
    margin = np.min(np.minimum(P, (hi - lo) - 1 - P))
    assert margin >= 37, f"a point is {margin} voxels from a face"
    B = C.load(lo, hi, f"onepass_{A.site}")
    vox = B["vox"]
    h = float(np.mean(vox))
    wm, ctx = B["wm"], B["ctx"]
    fluid = B["rim"] == 1
    print(f"{A.site}: box {tuple(hi - lo)} from {tuple(lo)}; {len(P)} points; nearest {margin * h:.1f} mm from a face")

    # ---- metric direction (as fit_all_wm)
    nh, coh = C.tensor_normal(B)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    g = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    sgn = np.sign((nh * gh).sum(-1)); sgn[sgn == 0] = 1
    dirv = coh[..., None] * nh * sgn[..., None] + 0.1 * gh
    dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
    del nh, g
    s = setup(wm, ctx, patch=PATCH, direction=dirv)
    print(f"   seeds {s.seeds.sum():,}, patches {s.npatch:,}")

    # ---- thickness field from column tops
    K = C.curvature(B, 0.8)
    pts, nrm = K["pts"], K["n"]
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
    face_ok = B["dface"] >= FACE_MM
    fluid_f = fluid & face_ok  # tops only where the fluid is >= 2 mm from the faces
    mu_tops, has = open_thickness(s, fluid_f, RATIO, along=True)  # per patch, voxels
    seed_pts = ptsid.ravel()[s.sflat]
    miss = seed_pts < 0
    if miss.any():  # seeds the curvature has no point for: the nearest one
        from scipy.spatial import cKDTree

        sp = np.stack(np.unravel_index(s.sflat[miss], wm.shape), -1)
        seed_pts[miss] = cKDTree(pts).query(sp)[1]
    print(f"   {miss.sum()} seeds without a curvature point (nearest used)")
    seed_patch = s.patch.ravel()[s.sflat]
    # evidence only from roots >= 2 mm from a face (columns near a face are left out; plan rules)
    root_ok = B["dface"].ravel()[s.sflat] >= FACE_MM
    ok = has[seed_patch] & root_ok & (mu_tops[seed_patch] * h > 0.8) & (mu_tops[seed_patch] * h < 7.0)
    y = np.full(nN, np.nan)
    nd = node_of[seed_pts[ok]]
    val = np.log(mu_tops[seed_patch[ok]] * h)
    order = np.argsort(nd)
    nd, val = nd[order], val[order]
    st = np.flatnonzero(np.r_[True, np.diff(nd) != 0])
    for i, a in enumerate(st):
        b = st[i + 1] if i + 1 < st.size else nd.size
        y[nd[a]] = np.median(val[a:b])
    ev = np.isfinite(y)
    mu_n, sd_n, coef = fld.fit(y, trend=True)
    tmean, tsd = lognormal(mu_n, sd_n)
    cnode = node_of[seed_pts[s.rep]]
    mu_f = tmean[cnode] / h  # voxels
    sd_f = tsd[cnode] / h
    # a root whose piece of the field's graph has no evidence has no belief (there the lognormal
    # mean can blow up: 5.9 m at the V1V2 banks); it gets the equal-arrival weight instead
    from scipy.sparse.csgraph import connected_components

    npc, pc = connected_components(graph_L(npos, ed) != 0, directed=False)
    ev_piece = np.bincount(pc[np.isfinite(y)], minlength=npc) > 0
    nobel = ~ev_piece[pc[cnode]]
    mu_f[nobel] = np.median(mu_f[~nobel])
    sd_f[nobel] = np.median(sd_f[~nobel])
    print(f"   {nobel.sum()} patches in {np.sum(~ev_piece)} graph pieces without evidence: equal-arrival weights")
    print(f"   evidence: {has.sum():,} of {s.npatch:,} patches with column tops; {ev.sum():,} of {nN:,} nodes; "
          f"median {np.exp(np.median(y[ev])):.2f} mm; a0 {coef['a0'][0]:+.2f} +- {coef['a0'][1]:.2f}")

    settings = {
        "equal": (np.full(s.npatch, float(np.median(mu_f))), 0.1 * np.full(s.npatch, float(np.median(mu_f)))),
        "field": (mu_f, sd_f),
        "field, sigma = 0.1 mu": (mu_f, 0.1 * mu_f),
    }
    u_pts = dirv[tuple(np.round(P).astype(int).T)]
    out = dict(site=A.site, box_lo=lo.tolist(), npatch=int(s.npatch), points=len(P), no_belief=int(nobel.sum()),
               mu_mm_p1_50_99_max=np.percentile(mu_f * h, [1, 50, 99, 100]).round(2).tolist(),
               sd_mm_p1_50_99_max=np.percentile(sd_f * h, [1, 50, 99, 100]).round(2).tolist())
    keep = {}
    for name, (mu, sig) in settings.items():
        f = run(s, mu, sig, ratio=RATIO)
        c = contacts(s, f, mu, sig, sep_min=SEP)  # for the views and the overlay
        cp = contact_points(c, wm.shape)
        rows_thr = score_points(P, u_pts, c, cp, mu_f, sd_f, h)
        c_all = contacts(s, f, mu, sig, sep_min=0.0)  # every boundary, for the local-sides score
        rows = score_points_sides(s, f, P, u_pts, c_all, contact_points(c_all, wm.shape), mu_f, sd_f, h)
        off = np.array([r["offset"] for r in rows])
        off_thr = np.array([r["offset"] for r in rows_thr])
        near_mm = np.array([r.get("nearest_mm", np.nan) for r in rows])
        out[name] = dict(
            pairs=int(c.p.size),
            nearest_crossing_mm=dict(median=float(np.nanmedian(near_mm)), p90=float(np.nanpercentile(near_mm, 90)),
                                     missing=int(np.isnan(near_mm).sum())),
            threshold_score=dict(median_abs=float(np.nanmedian(np.abs(off_thr))) if np.isfinite(off_thr).any() else None,
                                 missing=int(np.isnan(off_thr).sum())),
            median=float(np.nanmedian(off)), iqr=float(np.subtract(*np.nanpercentile(off, [75, 25]))),
            median_abs=float(np.nanmedian(np.abs(off))), worst=float(np.nanmax(np.abs(off))),
            missing=int(np.isnan(off).sum()), rows=rows,
        )
        print(f"   {name:22s} pairs {c.p.size:7,}  offset median {out[name]['median']:+.2f}, IQR {out[name]['iqr']:.2f}, "
              f"|offset| {out[name]['median_abs']:.2f}, worst {out[name]['worst']:.2f} mm, missing {out[name]['missing']}"
              f"   (3 mm threshold score: |offset| {out[name]['threshold_score']['median_abs']}, "
              f"missing {out[name]['threshold_score']['missing']}); nearest crossing (3D) median "
              f"{out[name]['nearest_crossing_mm']['median']:.2f}, p90 {out[name]['nearest_crossing_mm']['p90']:.2f} mm")
        keep[name] = (f, c, cp)
    r = out["field"]["rows"]
    for k_, row in enumerate(r):
        print(f"   {k_ + 1:2d}: thin mu-S {row['thin_mu_minus_S']:+.2f} (sd {row['thin_sd']:.2f}) thick "
              f"{row['thick_mu_minus_S']:+.2f} (sd {row['thick_sd']:.2f}); S {row['S_thin']:.2f}/{row['S_thick']:.2f}; "
              f"offset {row['offset']:+.2f}" if np.isfinite(row["offset"]) else f"   {k_ + 1:2d}: no crossing")
    json.dump(out, open(f"{A.work}/onepass_{A.site}{TAG}.json", "w"), indent=1, default=float)
    figures(B, P, keep, out, h)
    overlay(B, s, P, keep)


def figures(B, P, keep, out, h):
    names = list(keep)
    k = np.arange(1, len(P) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(max(8, 0.32 * len(P) + 3), 7.5), sharex=True)
    a = axs[0]
    for j, (nm, col) in enumerate(zip(names, (GREY, BLUE, AQUA))):
        off = np.array([r["offset"] for r in out[nm]["rows"]])
        a.plot(k + (j - 1) * 0.22, off, "o", color=col, ms=5,
               label=f"{nm}: |offset| median {out[nm]['median_abs']:.2f} mm, missing {out[nm]['missing']}")
    a.axhline(0, color="#52514e", lw=0.8)
    a.set_ylabel("offset along the column (mm)\n+ toward the side predicted thinner")
    a.set_title(f"{A.site}: one-pass label boundary against your {len(P)} points (3D)")
    a.legend(fontsize=8, loc="best")
    a = axs[1]
    rows = out["field"]["rows"]
    for key, sdk, col, lab, dx in (("thin_mu_minus_S", "thin_sd", BLUE, "side predicted thinner", -0.12),
                                   ("thick_mu_minus_S", "thick_sd", ORANGE, "side predicted thicker", 0.12)):
        e = np.array([r[key] for r in rows]); sd = np.array([r[sdk] for r in rows])
        a.errorbar(k + dx, e, yerr=sd, fmt="o", color=col, ms=4, lw=1, label=f"{lab}: mu - S at the point, +-1 SD")
    a.axhline(0, color="#52514e", lw=0.8)
    a.set_ylabel("mu - S (mm), field weights")
    a.set_xlabel("your contact point")
    a.set_xticks(k)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{A.fig}/onepass_{A.site}{TAG}_offsets.png", dpi=110)
    plt.close(fig)
    # slices through the points: the plane (fixed axis) holding the most points
    Pi = np.round(P).astype(int)
    ax_fix = int(np.argmin([len(np.unique(Pi[:, d])) for d in range(3)]))
    vals, cnt = np.unique(Pi[:, ax_fix], return_counts=True)
    planes = vals[np.argsort(-cnt)][:3]
    fig, axs = plt.subplots(len(planes), len(names), figsize=(4.2 * len(names), 4.0 * len(planes)), squeeze=False)
    disp = np.where(B["ctx"], B["disp"], np.nan)
    lim = np.nanpercentile(np.abs(disp), 98)
    for i, pl in enumerate(planes):
        sl = [slice(None)] * 3; sl[ax_fix] = pl
        rest = [d for d in range(3) if d != ax_fix]
        for j, nm in enumerate(names):
            a = axs[i][j]
            a.imshow(np.where(B["wm"][tuple(sl)], 1.0, np.nan).T, origin="lower", cmap="Greys", vmin=0, vmax=2.5)
            a.imshow(disp[tuple(sl)].T, origin="lower", cmap="gray", vmin=-lim, vmax=lim)
            cp = keep[nm][2]
            m = np.abs(cp[:, ax_fix] - pl) <= 0.5
            a.plot(cp[m, rest[0]], cp[m, rest[1]], ".", color=ORANGE, ms=2.5, label="label boundary (roots > 3 mm apart)")
            q = Pi[:, ax_fix] == pl
            a.plot(P[q, rest[0]], P[q, rest[1]], "+", color=AQUA, ms=8, mew=1.6, label="your points")
            c0 = P[q][:, rest].mean(0)
            a.set_xlim(c0[0] - 30, c0[0] + 30); a.set_ylim(c0[1] - 30, c0[1] + 30)
            a.set_title(f"{nm}; {'ijk'[ax_fix]} = {pl + 1} (ITK-SNAP)", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
    axs[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{A.site}: echo 2 (trend removed), WM dark grey; computed in 3D, shown on planes through the points", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{A.fig}/onepass_{A.site}{TAG}_planes.png", dpi=110)
    plt.close(fig)


def overlay(B, s, P, keep):
    """ITK-SNAP: 1 WM boundary (seeds), 2 boundary with equal weights, 3 with field weights, 4 both, 5 points."""
    lab = np.zeros(B["wm"].shape, np.uint8)
    lab[s.seeds] = 1
    shape = B["wm"].shape
    eq = np.zeros(shape, bool); fd = np.zeros(shape, bool)
    for nm, arr in (("equal", eq), ("field", fd)):
        c = keep[nm][1]
        arr.ravel()[c.p] = True; arr.ravel()[c.q] = True
    lab[eq] = 2; lab[fd] = 3; lab[eq & fd] = 4
    for p in np.round(P).astype(int):
        lab[tuple(p)] = 5
    aff = B["p"].affine
    nib.save(nib.Nifti1Image(lab, aff), f"{A.data}/onepass_{A.site}{TAG}_overlay.nii.gz")
    nib.save(nib.Nifti1Image(np.where(B["ctx"], B["disp"], 0.0).astype(np.float32), aff),
             f"{A.data}/onepass_{A.site}_echo2_detrended.nii.gz")
    with open(f"{A.data}/onepass_labels.txt", "w") as fh:
        fh.write("# ITK-SNAP label description file\n")
        fh.write('    0     0    0    0        0  0  0    "Clear Label"\n')
        for i, (rgb, nm) in enumerate((((150, 150, 150), "WM boundary (seeds)"),
                                       ((42, 120, 214), "label boundary, equal weights"),
                                       ((235, 104, 52), "label boundary, field weights"),
                                       ((227, 73, 72), "both"),
                                       ((27, 175, 122), "your points")), start=1):
            fh.write(f'  {i:3d}  {rgb[0]:3d}  {rgb[1]:3d}  {rgb[2]:3d}        1  1  1    "{nm}"\n')


if __name__ == "__main__":
    main()
