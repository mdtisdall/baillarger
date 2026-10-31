"""Finding 28, real patches: white-surface meshes of V1V2, M1S1 and PCC_SPL (all 3D).

For each patch (boxes of baillarger.data.PATCHES, loaded as fronts_eval.py does) and each
start surface (mesh_smoothing / mesh_phantoms.surfaces):
- thin WM: WM voxels outside the opening of the label by a ball of radius 2 (parts under
  about 5 voxels thick), in pieces of >= 50 voxels at >= 2 mm from the box faces. Each
  exposed face of a thin voxel is covered when a mesh vertex lies within 1 voxel of its
  centre with a normal on the same side (dot > 0). Per piece: the covered fraction. For a
  volume-smoothed contour also the thin voxels whose smoothed label is < 0.5 (the blade is
  gone);
- self-intersections (baillarger.mesh_intersect), all and near thin WM;
- curvature: mean curvature at 0.32 and 0.48 mm (mesh_curvatures) against the voxel-face
  quadric (curv_lib.curvature) at WM-boundary voxels (nearest vertex with an agreeing
  normal): correlation, median difference, and the jitter of each field (median |H - local
  mean over the same side within 2 voxels|), the staircase proxy;
- the image's own WM edge: along each vertex's normal (-3 to +3 voxels), the position of the
  steepest WM-to-cortex step of log echo 2 with its trend removed (minus a 8-voxel normalized
  blur over tissue); offset + where the image edge lies outside the mesh (toward cortex).

Usage: mesh_real.py <figures dir> <data dir> <out json> <patch>
"""

import json
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
from scipy.spatial import cKDTree

import curv_lib as C
import mesh_smoothing as M
from baillarger.data import PATCHES
from baillarger.frame import mesh_curvatures
from baillarger.mesh import topology, write_vtk
from baillarger.mesh_intersect import intersecting_faces
from mesh_phantoms import surfaces

FIG, DATA, OUT, NAME = sys.argv[1:5]
FACE_MM = 2.0
SIGMAS = (0.5, 1.0, 1.5, 2.0)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def ball(r):
    g = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (g**2).sum(0) <= r * r + 1e-9


def jitter(P, n, H, radius=2.0):
    """median |H - mean of H over points within radius on the same side|."""
    tree = cKDTree(P)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    same = np.sum(n[pairs[:, 0]] * n[pairs[:, 1]], 1) > 0
    pairs = pairs[same]
    s = np.bincount(pairs[:, 0], H[pairs[:, 1]], len(P)) + np.bincount(
        pairs[:, 1], H[pairs[:, 0]], len(P)
    )
    c = np.bincount(pairs[:, 0], minlength=len(P)) + np.bincount(
        pairs[:, 1], minlength=len(P)
    )
    ok = c > 3
    return float(np.median(np.abs(H[ok] - s[ok] / c[ok])))


def main():
    lo, hi = (np.array(v) for v in PATCHES[NAME])
    B = C.load(lo, hi, f"mesh_real_{NAME}")
    vox = B["vox"]
    h = float(np.mean(vox))
    wm, ctx, dface = B["wm"], B["ctx"], B["dface"]
    res = dict(patch=NAME, shape=list(wm.shape))
    log(f"{NAME} {wm.shape}")

    # ---- thin WM pieces
    opened = ndi.binary_opening(wm, structure=ball(2))
    thin = wm & ~opened & (dface >= FACE_MM)
    lab, n = ndi.label(thin, structure=np.ones((3, 3, 3)))
    size = np.bincount(lab.ravel())
    big = np.flatnonzero(size >= 50)
    big = big[big > 0]
    keep = np.isin(lab, big)
    lab = np.where(keep, lab, 0)
    res["thin_voxels"] = int(keep.sum())
    res["thin_pieces"] = int(big.size)
    # local thickness of the thin WM: twice the inside distance at the medial voxels
    edt_in = ndi.distance_transform_edt(wm)
    res["thin_halfwidth_p50_p90"] = (
        np.percentile(edt_in[keep], [50, 90]).round(2).tolist() if keep.any() else None
    )
    fc, fn, fv = C.wm_faces(wm, vox)
    fc = fc / vox  # voxel coordinates
    on_thin = keep[tuple(fv.T)]
    tf_c, tf_n, tf_piece = fc[on_thin], fn[on_thin], lab[tuple(fv[on_thin].T)]
    log(f"thin WM: {keep.sum():,} voxels in {big.size} pieces; {on_thin.sum():,} faces")

    # ---- the quadric reference
    Q = {sc: C.curvature(B, sc) for sc in (0.32, 0.48)}
    qp = Q[0.32]["pts"].astype(float)
    qn = Q[0.32]["n"]
    qok = dface[tuple(Q[0.32]["pts"].T)] >= FACE_MM
    log(f"quadric at {len(qp):,} boundary voxels")
    res["quadric_jitter"] = {
        str(sc): jitter(qp[qok], qn[qok], 0.5 * (Q[sc]["k1"] + Q[sc]["k2"])[qok])
        for sc in Q
    }

    # ---- the image edge: log echo 2 with the trend removed
    tis = (wm | ctx).astype(float)
    L2 = np.log(np.maximum(B["p"].echoes[..., 1].astype(float), 1.0))
    bl = ndi.gaussian_filter(L2 * tis, 8.0) / np.maximum(
        ndi.gaussian_filter(tis, 8.0), 1e-6
    )
    E2 = np.where(tis > 0, L2 - bl, 0.0)
    del L2, bl

    S = surfaces(wm, sigmas=SIGMAS, scales=(1.0, 2.0, 3.0), taubin=())
    S = {
        k: S[k]
        for k in (
            "binary contour",
            *[f"white_mesh s{s:g}" for s in SIGMAS],
            "twice 0.16mm",
            "twice 0.32mm",
            "twice 0.48mm",
            "s0.5+twice 0.32mm",
        )
    }
    rows = {}
    lost_at = np.zeros(wm.shape, np.int16)
    edge_prof = {}
    for name, (v, f, nrm) in S.items():
        t = time.time()
        r = dict(vertices=int(len(v)), faces=int(len(f)))
        vi = np.clip(np.round(v).astype(int), 0, np.array(wm.shape) - 1)
        vok = dface[tuple(vi.T)] >= FACE_MM
        r["topology"] = {
            k: (float(x) if isinstance(x, float) else x)
            for k, x in topology(f, len(v)).items()
        }
        mask, nx = intersecting_faces(v, f)
        r["self_intersections"] = int(nx)
        fvi = vi[f].reshape(-1, 3)
        near_thin = (
            ndi.binary_dilation(keep, iterations=2)[tuple(fvi.T)].reshape(-1, 3).any(1)
        )
        r["intersecting_faces"] = int(mask.sum())
        r["intersecting_faces_near_thin"] = int((mask & near_thin).sum())
        # thin coverage
        d, j = cKDTree(v).query(tf_c, k=6, distance_upper_bound=1.0)
        okj = np.isfinite(d)
        jj = np.where(okj, j, 0)
        agree = (np.einsum("nkc,nc->nk", nrm[jj], tf_n) > 0) & okj
        cov = agree.any(1)
        r["thin_face_coverage"] = float(cov.mean()) if cov.size else float("nan")
        pc = (
            np.zeros(0)
            if not big.size
            else np.bincount(tf_piece, cov, lab.max() + 1)[big]
            / np.maximum(np.bincount(tf_piece, minlength=lab.max() + 1)[big], 1)
        )
        r["pieces_cov_ge_0.95"] = int((pc >= 0.95).sum())
        r["pieces_cov_0.5_0.95"] = int(((pc >= 0.5) & (pc < 0.95)).sum())
        r["pieces_cov_lt_0.5"] = int((pc < 0.5).sum())
        if name.startswith("white_mesh"):
            s = float(name.split("s")[-1])
            gone = keep & (ndi.gaussian_filter(wm.astype(float), s) < 0.5)
            r["thin_voxels_gone"] = float(gone.sum() / max(keep.sum(), 1))
            # per thin voxel: lost when its faces are not covered
            vcov = np.zeros(wm.shape, bool)
            vcov[tuple(fv[on_thin][cov].T)] = True
            newly = keep & ~vcov & (lost_at == 0)
            lost_at[newly] = int(round(s * 2))  # 1: s0.5, 2: s1, 3: s1.5, 4: s2
        # curvature against the quadric
        P = v[vok]
        Nn = nrm[vok]
        tree = cKDTree(P)
        dq, jq = tree.query(qp[qok], k=8, distance_upper_bound=2.0)
        jq0 = np.where(np.isfinite(dq), jq, 0)
        ag = (np.einsum("nkc,nc->nk", Nn[jq0], qn[qok]) > 0) & np.isfinite(dq)
        first = np.where(ag.any(1), ag.argmax(1), -1)
        hit = first >= 0
        jbest = jq0[np.arange(len(first)), np.maximum(first, 0)]
        r["quadric_matched"] = float(hit.mean())
        for sc in (0.32, 0.48):
            k1, k2 = mesh_curvatures(v * vox, f, nrm, sc)
            Hm = 0.5 * (k1 + k2)
            Hq = 0.5 * (Q[sc]["k1"] + Q[sc]["k2"])[qok][hit]
            Hv = Hm[vok][jbest[hit]]
            from scipy.stats import spearmanr

            r[f"H_corr_{sc}"] = float(
                spearmanr(Hq, Hv).statistic
            )  # rank: a few extreme values on
            # one-voxel WM fragments dominated Pearson's r
            r[f"H_pearson_{sc}"] = float(np.corrcoef(Hq, Hv)[0, 1])
            r[f"H_extreme_frac_{sc}"] = float(np.mean(np.abs(Hv) > 3.0))
            r[f"H_quadric_extreme_frac_{sc}"] = float(np.mean(np.abs(Hq) > 3.0))
            r[f"H_diff_med_{sc}"] = float(np.median(Hv - Hq))
            r[f"H_absdiff_med_{sc}"] = float(np.median(np.abs(Hv - Hq)))
            r[f"H_jitter_{sc}"] = jitter(P, Nn, Hm[vok])
            if sc == 0.48:
                r["H_p5_p95"] = np.percentile(Hm[vok], [5, 95]).round(3).tolist()
        # image edge along the normal
        sub = np.flatnonzero(vok)[:: max(1, int(vok.sum() // 60000))]
        s_ax = np.arange(-3.0, 3.01, 0.1)
        pts = v[sub][:, None, :] + s_ax[None, :, None] * nrm[sub][:, None, :]
        prof = ndi.map_coordinates(
            E2, pts.reshape(-1, 3).T, order=1, mode="nearest"
        ).reshape(len(sub), -1)
        prof = ndi.gaussian_filter1d(prof, 2.0, axis=1)  # 0.2 voxel
        g = np.gradient(prof, 0.1, axis=1)
        sgn = np.sign(np.median(g[:, 30]))
        gs = g * sgn
        win = np.abs(s_ax) <= 2.0
        arg = np.argmax(np.where(win[None], gs, -np.inf), 1)
        good = gs[np.arange(len(sub)), arg] > 0
        off = s_ax[arg][good]
        r["edge_sign"] = float(sgn)
        r["edge_offset_p25_50_75"] = np.percentile(off, [25, 50, 75]).round(2).tolist()
        r["edge_offset_abs_med"] = float(np.median(np.abs(off)))
        r["edge_at_window_end"] = float(np.mean(np.abs(off) >= 1.95))
        edge_prof[name] = (s_ax, prof[good].mean(0), off)
        rows[name] = r
        log(
            f"{name:20s} V {len(v):,} X {nx} (near thin {r['intersecting_faces_near_thin']}) "
            f"thin cov {r['thin_face_coverage']:.3f} pieces >=.95/.5-.95/<.5 {r['pieces_cov_ge_0.95']}/"
            f"{r['pieces_cov_0.5_0.95']}/{r['pieces_cov_lt_0.5']} "
            + (f"gone {r['thin_voxels_gone']:.3f} " if "thin_voxels_gone" in r else "")
            + f"H corr {r['H_corr_0.32']:.2f}/{r['H_corr_0.48']:.2f} diff {r['H_diff_med_0.48']:+.3f} "
            f"jit {r['H_jitter_0.32']:.3f}/{r['H_jitter_0.48']:.3f} (quadric {res['quadric_jitter']['0.32']:.3f}/"
            f"{res['quadric_jitter']['0.48']:.3f}) edge {r['edge_offset_p25_50_75']} [{time.time() - t:.0f}s]"
        )
        res["meshes"] = rows
        json.dump(res, open(OUT, "w"), indent=1)

    # ---- outputs: thin WM overlay (lost at which volume sigma), meshes as VTK
    ov = np.zeros(wm.shape, np.int16)
    ov[keep] = 5  # thin WM kept under every sigma
    ov[lost_at > 0] = lost_at[lost_at > 0]
    aff = B["p"].affine
    nib.save(nib.Nifti1Image(ov, aff), f"{DATA}/mesh_{NAME}_thin_overlay.nii.gz")
    with open(f"{DATA}/mesh_thin_labels.txt", "w") as fh:
        fh.write('0 0 0 0 0 0 0 "Clear Label"\n')
        for i, (nm, rgb) in enumerate(
            (
                ("thin WM lost at white_mesh sigma 0.5", "227 73 72"),
                ("thin WM lost at sigma 1", "235 104 52"),
                ("thin WM lost at sigma 1.5", "250 190 50"),
                ("thin WM lost at sigma 2", "27 175 122"),
                ("thin WM kept at every sigma", "42 120 214"),
            ),
            start=1,
        ):
            fh.write(f'{i} {rgb} 1 1 1 "{nm}"\n')
    for nm in ("white_mesh s1", "twice 0.32mm"):
        v, f, nrm = S[nm]
        k1, k2 = mesh_curvatures(v * vox, f, nrm, 0.48)
        tag = nm.replace(" ", "_").replace(".", "p")
        write_vtk(
            f"{DATA}/mesh_{NAME}_{tag}.vtk", aff, v, f, {"H_048": 0.5 * (k1 + k2)}
        )
    # ---- figures: a plane through the thin WM with every mesh's section; edge profiles
    zc = int(np.argmax(keep.sum((0, 1)))) if keep.any() else wm.shape[2] // 2
    names = list(S)
    fig, axs = plt.subplots(
        2, (len(names) + 1) // 2, figsize=(4.4 * ((len(names) + 1) // 2), 9)
    )
    disp = np.where(ctx | wm, E2, np.nan)[:, :, zc]
    from mesh_phantoms import section
    from matplotlib.collections import LineCollection

    for a, nm in zip(axs.ravel(), names, strict=False):
        a.imshow(
            disp.T,
            origin="lower",
            cmap="gray",
            vmin=np.nanpercentile(disp, 2),
            vmax=np.nanpercentile(disp, 98),
        )
        a.contour(keep[:, :, zc].T, [0.5], colors="#1baf7a", linewidths=0.6)
        sg = section(S[nm][0], S[nm][1], 2, zc + 1e-3)
        if len(sg):
            a.add_collection(LineCollection(sg, colors="#e34948", lw=0.6))
        a.set_title(f"{nm}", fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
    for a in axs.ravel()[len(names) :]:
        a.axis("off")
    fig.suptitle(
        f"{NAME}: plane k = {zc} (most thin WM); log echo 2, trend removed; green: thin WM; red: mesh",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/mesh_{NAME}_sections.png", dpi=100)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for nm, (s_ax, mp, off) in edge_prof.items():
        axs[0].plot(s_ax * h, mp, lw=1, label=nm)
        axs[1].hist(
            off * h, bins=np.arange(-2, 2.05, 0.1) * h, histtype="step", label=nm
        )
    axs[0].set_xlabel("along the mesh normal (mm; + toward cortex)")
    axs[0].set_ylabel("log echo 2, trend removed")
    axs[1].set_xlabel("steepest WM-to-cortex step (mm from the mesh)")
    axs[1].set_ylabel("vertices")
    axs[0].legend(fontsize=7)
    fig.suptitle(
        f"{NAME}: where the image's WM edge lies against each mesh", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/mesh_{NAME}_edge.png", dpi=100)
    plt.close(fig)
    json.dump(res, open(OUT, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
