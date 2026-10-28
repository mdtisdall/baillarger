"""Figures for a box's white mesh against the image's own GM/WM edge. The numbers and files are
the package's (baillarger.wm_edge.box_edges and write_box, as `python -m baillarger.wm_edge
box`): the offsets of echoes 2 and 3 by curvature and coherence, and step 4 (refine) with the
edge measured again. This driver adds the root's column-trace outcome (from a turn diagnostic
of the same box and label) and draws. Diagnostic views, not pass/fail.

Usage: wm_edge_maps.py <figures> <data> --name NAME --box X Y Z [--size N]
           --wm-changes WM_CHANGES.npz [--turns TURN_DIAG.npz]
       Writes write_box's files <data>/wm_edge_<NAME>.*, <data>/wm_edge_<NAME>_outcome.json
       (with --turns, when its mesh is this one) and <figures>/wm_edge_<NAME>_{maps,planes}.png.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baillarger.column_traces import FACE_MM
from baillarger.white import WMChanges
from baillarger.wm_edge import COH_AT, CURV_MM, EdgeOffsets, box_edges, write_box


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
    lo = tuple(a.box)
    hi = tuple(x + a.size for x in lo)
    t0 = time.time()
    be = box_edges(lo, hi, WMChanges.load(Path(a.wm_changes)), name=a.name)
    write_box(be, out, f"wm_edge_{a.name}")
    A, rep = be.arrays, be.report
    print(
        json.dumps({k: rep[k] for k in ("vertices", "roots")}),
        f"[{time.time() - t0:.0f}s]",
    )
    for nm in ("echo2", "echo3"):
        print(
            nm,
            json.dumps(
                {
                    k: rep[nm][k]
                    for k in (
                        "at_end_frac",
                        "all",
                        "outside_gt_half_frac",
                        "inside_gt_half_frac",
                    )
                }
            ),
        )
    print("refined", json.dumps(rep["refined"]))
    v, roots, keep = A["v"], A["roots"], A["keep"]
    E = {
        nm: EdgeOffsets(
            A[f"{nm}_offset_mm"], A[f"{nm}_slope"], A[f"{nm}_at_end"], rep[nm]["sign"]
        )
        for nm in ("echo2", "echo3")
    }
    outcome, names = None, None
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
            e = E["echo2"]
            ok = keep & ~e.at_end
            by = {
                names[c]: pct(e.offset_mm[ok & (outcome == c)])
                for c in np.unique(outcome[outcome >= 0])
            }
            (out / f"wm_edge_{a.name}_outcome.json").write_text(
                json.dumps(by, indent=1)
            )
        else:
            print("turns npz: not this mesh, outcome not used")
    half = rep["half_voxel_mm"]
    maps(
        fig_dir / f"wm_edge_{a.name}_maps.png",
        a.name,
        E,
        A["H"],
        A["coh"],
        keep,
        half,
        outcome,
        names,
    )
    planes(
        fig_dir / f"wm_edge_{a.name}_planes.png",
        a.name,
        be.tissue,
        v,
        A["refined_v"],
        E["echo2"],
        A["refined_move_mm"],
    )
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
        f"{name}: the image's GM/WM edge against the white mesh (vertices ≥ {FACE_MM} mm from the box faces; "
        f"dotted: half a voxel)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def planes(path, name, t, v, v1, e, move):
    """Three orthogonal planes through the box centre: echo 2, the label's outline, the mesh
    vertices within half a voxel of the plane coloured by offset (before refinement), and the
    vertices that step 4 moved, at their new places (black)."""
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
        sm = (np.abs(v1[:, ax_i] - c[ax_i]) < 0.5) & (move < 0)
        ax.scatter(v1[sm, other[0]], v1[sm, other[1]], c="k", s=1)
        ax.set_title(f"axis {ax_i} = {c[ax_i]} (box)", fontsize=9)
    fig.colorbar(
        sc,
        ax=axs,
        shrink=0.7,
        label="echo 2 edge offset (mm; red: edge outside the mesh)",
    )
    fig.suptitle(
        f"{name}: echo 2; yellow: WM label; dots: vertices by edge offset; green x: no edge in "
        "±0.3 mm; black: moved vertices after step 4",
        fontsize=11,
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
