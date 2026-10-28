"""Where the image's own GM/WM edge lies against the white mesh of a box (baillarger.wm_edge,
step 1 of putting the white surface on the data's edge): per vertex, the offset along the
outward normal to the steepest WM-to-GM change of echo 2 and of echo 3, mapped by mean
curvature, by the cortex's coherence just outside, and by the column trace's outcome (open to
fluid, or crossing to WM: fused or kissing cortex). Diagnostic views, not pass/fail.

Usage: wm_edge_maps.py <figures> <data> --name NAME --box X Y Z [--size N]
           --wm-changes WM_CHANGES.npz [--turns TURN_DIAG.npz]
       Writes <data>/wm_edge_<NAME>.json (summary), <data>/wm_edge_<NAME>.npz (per vertex),
       <data>/wm_edge_<NAME>.vtk (the mesh with the offsets, ITK-SNAP world coordinates),
       <data>/wm_edge_<NAME>_overlay.nii.gz (box grid; at each vertex's voxel: 1 edge more
       than half a voxel outside, 2 more than half a voxel inside, 3 no edge in the window),
       <figures>/wm_edge_<NAME>_{maps,planes}.png.
       --turns: a turn_diag npz of the same box and label; its per-root outcome is used when
       its mesh is this one.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from baillarger import column_traces as CT
from baillarger.frame import mesh_curvatures
from baillarger.mesh import write_vtk
from baillarger.white import WMChanges
from baillarger.wm_edge import edge_offsets, tissue_contrast

CURV_MM = 0.48
COH_AT = 1.5  # voxels outside the mesh, where the cortex's coherence is read
H_BINS = [-np.inf, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, np.inf]
COH_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def pct(x):
    x = np.asarray(x)
    if not x.size:
        return None
    return dict(
        n=int(x.size),
        p5_25_50_75_95=np.percentile(x, [5, 25, 50, 75, 95]).round(4).tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figures")
    ap.add_argument("data")
    ap.add_argument("--name", required=True)
    ap.add_argument("--box", type=int, nargs=3, required=True)
    ap.add_argument("--size", type=int, default=141)
    ap.add_argument("--wm-changes", required=True)
    ap.add_argument("--turns")
    a = ap.parse_args()
    fig_dir, out = Path(a.figures), Path(a.data)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    lo = tuple(a.box)
    hi = tuple(x + a.size for x in lo)
    t0 = time.time()
    t = CT.load_tissue(lo, hi, name=a.name, changes=WMChanges.load(Path(a.wm_changes)))
    vox = t.vox
    half = 0.5 * float(vox.min())
    v, f, n = CT.white_mesh(t.wm)
    roots = CT.root_vertices(v, n)
    print(
        f"mesh V {len(v):,} F {len(f):,}, roots {len(roots):,} [{time.time() - t0:.0f}s]"
    )

    res = dict(
        name=a.name, lo=list(lo), hi=list(hi), vox=vox.tolist(), half_voxel_mm=half
    )
    E = {}
    for k, nm in ((1, "echo2"), (2, "echo3")):
        img = tissue_contrast(t.patch.echoes[..., k], t.wm)
        E[nm] = edge_offsets(img, v, n, vox)
        print(f"{nm}: sign {E[nm].sign:+.0f} [{time.time() - t0:.0f}s]")
    k1, k2 = mesh_curvatures(v * vox, f, n, CURV_MM)
    H = 0.5 * (k1 + k2)
    _, coh = CT.data_normal(CT.base_tensor(t), t.ctx)
    coh_v = ndi.map_coordinates(coh, (v + COH_AT * n).T, order=1, mode="nearest")
    vi = np.clip(np.round(v).astype(int), 0, np.array(t.wm.shape) - 1)
    keep = t.dface[tuple(vi.T)] >= CT.FACE_MM

    outcome = None
    if a.turns:
        z = np.load(a.turns)
        if (
            tuple(int(x) for x in z["lo"]) == lo
            and z["v"].shape == v.shape
            and np.allclose(z["v"], v, atol=1e-4)
            and np.array_equal(z["tv"], roots)
        ):
            names = list(z["outcomes"])
            outcome = np.full(len(v), -1)
            outcome[roots] = z["out"]
            res["outcome_names"] = names
        else:
            print("turns npz: not this mesh, outcome not used")

    # ---- summary
    for nm, e in E.items():
        ok = keep & ~e.at_end
        o = e.offset_mm
        r = dict(
            sign=e.sign,
            at_end_frac=float(e.at_end[keep].mean()),
            all=pct(o[ok]),
            roots=pct(o[roots][ok[roots]]),
            abs_gt_half_voxel_frac=float((np.abs(o[ok]) > half).mean()),
            outside_gt_half_frac=float((o[ok] > half).mean()),
            inside_gt_half_frac=float((o[ok] < -half).mean()),
        )
        r["by_H"] = {
            f"{lo_:g}..{hi_:g}": pct(o[ok & (H >= lo_) & (H < hi_)])
            for lo_, hi_ in zip(H_BINS[:-1], H_BINS[1:])
        }
        r["by_coherence"] = {
            f"{lo_:g}..{hi_:g}": pct(o[ok & (coh_v >= lo_) & (coh_v < hi_)])
            for lo_, hi_ in zip(COH_BINS[:-1], COH_BINS[1:])
        }
        r["at_end_by_coherence"] = {
            f"{lo_:g}..{hi_:g}": float(
                e.at_end[keep & (coh_v >= lo_) & (coh_v < hi_)].mean()
            )
            for lo_, hi_ in zip(COH_BINS[:-1], COH_BINS[1:])
        }
        if outcome is not None:
            r["by_outcome"] = {
                names[c]: pct(o[ok & (outcome == c)])
                for c in np.unique(outcome[outcome >= 0])
            }
        res[nm] = r
    both = keep & ~E["echo2"].at_end & ~E["echo3"].at_end
    d23 = E["echo3"].offset_mm[both] - E["echo2"].offset_mm[both]
    res["echo3_minus_echo2"] = pct(d23)
    res["echo2_echo3_corr"] = float(
        np.corrcoef(E["echo2"].offset_mm[both], E["echo3"].offset_mm[both])[0, 1]
    )
    (out / f"wm_edge_{a.name}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(
        out / f"wm_edge_{a.name}.npz",
        lo=np.array(lo),
        vox=vox,
        v=v,
        f=f,
        n=n,
        roots=roots,
        H=H,
        coh=coh_v,
        keep=keep,
        **{
            f"{nm}_{k}": getattr(e, k)
            for nm, e in E.items()
            for k in ("offset_mm", "slope", "at_end")
        },
        **({} if outcome is None else dict(outcome=outcome)),
    )
    write_vtk(
        str(out / f"wm_edge_{a.name}.vtk"),
        t.patch.affine,
        v,
        f,
        {
            "offset_echo2_mm": np.where(
                E["echo2"].at_end, np.nan, E["echo2"].offset_mm
            ),
            "offset_echo3_mm": np.where(
                E["echo3"].at_end, np.nan, E["echo3"].offset_mm
            ),
            "H_per_mm": H,
            "coherence": coh_v,
        },
    )
    ov = np.zeros(t.wm.shape, np.uint8)
    o2 = E["echo2"].offset_mm
    for code, sel in ((1, o2 > half), (2, o2 < -half), (3, E["echo2"].at_end)):
        ov[tuple(vi[sel].T)] = code
    nib.save(
        nib.Nifti1Image(ov, t.patch.affine), out / f"wm_edge_{a.name}_overlay.nii.gz"
    )
    print(
        json.dumps(
            {
                nm: {
                    k: res[nm][k]
                    for k in ("at_end_frac", "all", "abs_gt_half_voxel_frac")
                }
                for nm in E
            },
            indent=1,
        )
    )
    print(
        "echo3 - echo2",
        res["echo3_minus_echo2"],
        "corr",
        round(res["echo2_echo3_corr"], 3),
    )

    maps(
        fig_dir / f"wm_edge_{a.name}_maps.png",
        a.name,
        E,
        H,
        coh_v,
        keep,
        half,
        outcome,
        res.get("outcome_names"),
    )
    planes(fig_dir / f"wm_edge_{a.name}_planes.png", a.name, t, v, f, E["echo2"], half)
    print(f"done [{time.time() - t0:.0f}s]")


def binned(ax, x, y, bins, label, color):
    c, m, q1, q3 = [], [], [], []
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        s = (x >= lo_) & (x < hi_)
        if s.sum() < 50:
            continue
        c.append(np.median(x[s]))
        p = np.percentile(y[s], [25, 50, 75])
        q1.append(p[0]), m.append(p[1]), q3.append(p[2])
    ax.plot(c, m, "o-", color=color, label=label)
    ax.fill_between(c, q1, q3, color=color, alpha=0.2)


def maps(path, name, E, H, coh, keep, half, outcome, onames):
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    cols = {"echo2": "C0", "echo3": "C1"}
    hb = np.linspace(-3, 3, 25)
    cb = np.linspace(0, 1, 11)
    for nm, e in E.items():
        ok = keep & ~e.at_end
        o = e.offset_mm
        axs[0, 0].hist(
            o[ok],
            bins=np.linspace(-0.3, 0.3, 61),
            histtype="step",
            color=cols[nm],
            label=f"{nm}: median {np.median(o[ok]):+.3f}, IQR {np.subtract(*np.percentile(o[ok], [75, 25])):.3f} mm",
        )
        binned(axs[0, 1], H[ok], o[ok], hb, nm, cols[nm])
        binned(axs[0, 2], coh[ok], o[ok], cb, nm, cols[nm])
        sl = np.abs(e.slope[ok])
        binned(axs[1, 2], coh[ok], sl / np.median(sl), cb, nm, cols[nm])
        axs[1, 0].plot(
            cb[:-1] + 0.05,
            [
                e.at_end[keep & (coh >= a) & (coh < b)].mean()
                if (keep & (coh >= a) & (coh < b)).sum()
                else np.nan
                for a, b in zip(cb[:-1], cb[1:])
            ],
            "o-",
            color=cols[nm],
            label=nm,
        )
    for ax in (axs[0, 0],):
        for s in (-half, half):
            ax.axvline(s, color="k", ls=":", lw=1)
        ax.set_xlabel(
            "offset of the image's edge from the mesh (mm; + outside, in cortex)"
        )
        ax.legend(fontsize=8)
    for ax in (axs[0, 1], axs[0, 2]):
        ax.axhline(0, color="k", lw=0.5)
        for s in (-half, half):
            ax.axhline(s, color="k", ls=":", lw=1)
        ax.set_ylabel("offset (mm), median and IQR")
        ax.legend(fontsize=8)
    axs[0, 1].set_xlabel(f"mean curvature H at {CURV_MM} mm (1/mm, > 0 convex)")
    axs[0, 2].set_xlabel(f"cortex coherence {COH_AT} voxels outside")
    axs[1, 0].set_xlabel("cortex coherence")
    axs[1, 0].set_ylabel("fraction with no edge in ±0.3 mm")
    axs[1, 0].legend(fontsize=8)
    axs[1, 2].set_xlabel("cortex coherence")
    axs[1, 2].set_ylabel("edge slope / its median")
    axs[1, 2].legend(fontsize=8)
    ax = axs[1, 1]
    if outcome is not None:
        e = E["echo2"]
        ok = keep & ~e.at_end
        cs = [
            c
            for c in np.unique(outcome[outcome >= 0])
            if (ok & (outcome == c)).sum() >= 50
        ]
        ax.boxplot([e.offset_mm[ok & (outcome == c)] for c in cs], showfliers=False)
        ax.set_xticks(
            range(1, len(cs) + 1),
            [f"{onames[c]}\n{(ok & (outcome == c)).sum():,}" for c in cs],
            fontsize=8,
        )
        ax.axhline(0, color="k", lw=0.5)
        for s in (-half, half):
            ax.axhline(s, color="k", ls=":", lw=1)
        ax.set_ylabel("echo 2 offset (mm), roots")
        ax.set_title(
            "by the root's trace outcome (WM: crosses to far WM, fused/kissing)",
            fontsize=9,
        )
    else:
        ax.axis("off")
    fig.suptitle(
        f"{name}: the image's GM/WM edge against the white mesh (vertices ≥ {CT.FACE_MM} mm from the box faces; "
        f"dotted: half a voxel)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def planes(path, name, t, v, f, e, half):
    """Three orthogonal planes through the box centre: echo 2, the mesh's section, and the
    vertices within half a voxel of the plane, coloured by offset."""
    sh = np.array(t.wm.shape)
    c = sh // 2
    fig, axs = plt.subplots(1, 3, figsize=(19, 7))
    for ax_i, ax in enumerate(axs):
        idx = [slice(None)] * 3
        idx[ax_i] = c[ax_i]
        img = t.patch.echoes[..., 1][tuple(idx)]
        other = [i for i in range(3) if i != ax_i]
        ax.imshow(
            img.T,
            origin="lower",
            cmap="gray",
            vmin=np.percentile(img, 1),
            vmax=np.percentile(img, 99.5),
        )
        ax.contour(t.wm[tuple(idx)].T.astype(float), [0.5], colors="y", linewidths=0.4)
        s = (np.abs(v[:, ax_i] - c[ax_i]) < 0.5) & ~e.at_end
        sc = ax.scatter(
            v[s, other[0]],
            v[s, other[1]],
            c=e.offset_mm[s],
            s=3,
            cmap="RdBu_r",
            vmin=-0.2,
            vmax=0.2,
        )
        se = (np.abs(v[:, ax_i] - c[ax_i]) < 0.5) & e.at_end
        ax.scatter(v[se, other[0]], v[se, other[1]], c="lime", s=4, marker="x")
        ax.set_title(f"axis {ax_i} = {c[ax_i]} (box)", fontsize=9)
    fig.colorbar(
        sc,
        ax=axs,
        shrink=0.7,
        label="echo 2 edge offset (mm; red: edge outside the mesh)",
    )
    fig.suptitle(
        f"{name}: echo 2; yellow: WM label; dots: mesh vertices by edge offset; green x: no edge in ±0.3 mm",
        fontsize=11,
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
