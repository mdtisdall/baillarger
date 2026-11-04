"""Phase 1.2, Part E: the views of the whole-hemisphere orient pass
(orient_hemi.py): the agreement map on the overview mesh, the echoes'
trace shares by depth, the departures of each echo's normal, and the axis
dependence. Prints the summary numbers as JSON.

usage:
    orient_hemi_figures.py OUT --surf SURF --fig FIG
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from baillarger.white_overview import WhiteOverview, overview_paths  # noqa: E402

NAMES = ("echo 1", "echo 2", "echo 3")
REGIONS = ("total", "V1V2", "M1S1", "PCC_SPL")
COLORS = ("#1f5f99", "#c25e1f", "#2e8540")


def region(h: dict, name: str) -> dict:
    return h["total"] if name == "total" else h["sites"][name]


def within(hist: np.ndarray, bins: np.ndarray, deg: float) -> float:
    k = int(np.searchsorted(bins, deg))
    return float(hist[..., :k].sum() / max(hist.sum(), 1))


def summary(h: dict) -> dict:
    ab = np.array(h["bins"]["angle"])
    gb = np.array(h["bins"]["agree"])
    out = {}
    for name in REGIONS:
        r = region(h, name)
        ang = np.array(r["angle"])  # (4, depth, angle)
        ag = np.array(r["agree"])
        cum = np.cumsum(ag) / max(ag.sum(), 1)
        out[name] = {
            "interior_voxels": float(ang[3].sum()),
            "within10_e2_e3": within(ang[3], ab, 10),
            "within10_e1_combined": within(ang[0], ab, 10),
            "within10_e2_combined": within(ang[1], ab, 10),
            "within10_e3_combined": within(ang[2], ab, 10),
            "roots": float(ag.sum()),
            "agreement_median_deg": float(gb[np.searchsorted(cum, 0.5) + 1])
            if ag.sum()
            else None,
            "agreement_within10": within(ag, gb, 10),
            "agreement_within20": within(ag, gb, 20),
            "agreement_over45": 1 - within(ag, gb, 45),
        }
    return out


def face_values(
    ov: WhiteOverview, vertex: np.ndarray, angle: np.ndarray, n_true: int
) -> np.ndarray:
    """The mean angle per overview face over its corners and the true
    vertices it carries; NaN where none has an angle."""
    a = np.full(n_true, np.nan, np.float32)
    a[vertex] = angle
    nf = len(ov.f)
    s = np.zeros(nf)
    c = np.zeros(nf)
    for k in range(3):
        v = a[ov.vertex_ids[ov.f[:, k]]]
        ok = np.isfinite(v)
        s[ok] += v[ok]
        c[ok] += 1
    step = 2_000_000
    for f0 in range(0, nf, step):
        ids, face = ov.carried(np.arange(f0, min(f0 + step, nf)))
        v = a[ids]
        ok = np.isfinite(v)
        np.add.at(s, face[ok], v[ok])
        np.add.at(c, face[ok], 1)
    with np.errstate(invalid="ignore"):
        return s / c


def render(ax, v, f, val, view: np.ndarray, up: np.ndarray, vmax: float) -> None:  # type: ignore[no-untyped-def]
    """Faces of the mesh seen along -view (the camera at +view), painter's
    order, flat Lambert shade times the colour of val."""
    right = np.cross(up, view)
    right /= np.linalg.norm(right)
    up2 = np.cross(view, right)
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    front = nrm @ view > 0
    depth = (tri.mean(1) @ view)[front]
    o = np.argsort(depth)
    t = tri[front][o]
    xy = np.stack([t @ right, t @ up2], -1)
    cmap = plt.get_cmap("viridis_r")
    col = cmap(np.clip(val[front][o] / vmax, 0, 1))
    col[~np.isfinite(val[front][o])] = (0.75, 0.75, 0.75, 1)
    shade = 0.35 + 0.65 * np.abs(nrm[front][o] @ view)
    col[:, :3] *= shade[:, None]
    ax.add_collection(PolyCollection(xy, facecolors=col, edgecolors="none"))
    ax.set_xlim(xy[..., 0].min(), xy[..., 0].max())
    ax.set_ylim(xy[..., 1].min(), xy[..., 1].max())
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--surf", required=True, type=Path)
    ap.add_argument("--fig", required=True, type=Path)
    a = ap.parse_args()
    a.fig.mkdir(parents=True, exist_ok=True)
    h = json.loads((a.out / "hemi.json").read_text())
    summ = summary(h)
    log = h["superblocks"]
    summ["run"] = {
        "superblocks": len(log),
        "seconds_sum": float(sum(x["seconds"] for x in log)),
        "peak_gb_max": float(max(x.get("peak_gb", 0.0) for x in log)),
    }
    print(json.dumps(summ, indent=1))
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    db = np.array(h["bins"]["depth"])
    dc = 0.5 * (db[:-1] + db[1:])
    ab = np.array(h["bins"]["angle"])
    xb = np.array(h["bins"]["axis"])
    xc = 0.5 * (xb[:-1] + xb[1:])
    gb = np.array(h["bins"]["agree"])
    gc = 0.5 * (gb[:-1] + gb[1:])

    # 1. trace share by depth
    f, ax = plt.subplots(1, 4, figsize=(15, 3.4), dpi=90, sharey=True)
    for i, name in enumerate(REGIONS):
        r = region(h, name)
        tr = np.array(r["trace"])
        n = np.array(r["trace_n"])
        m = n > 100
        for e in range(3):
            ax[i].plot(dc[m], tr[m, e] / n[m], color=COLORS[e], lw=2, label=NAMES[e])
        ax[i].set_title(f"{name} ({int(n.sum()):,} voxels)", fontsize=9)
        ax[i].set_xlabel("WM distance, mm")
        ax[i].grid(alpha=0.3)
    ax[0].set_ylabel("share of J's trace")
    ax[0].legend(fontsize=7, frameon=False)
    f.suptitle(
        "Each echo's share of the whitened tensor's trace, by depth", fontsize=10
    )
    f.tight_layout()
    f.savefig(a.fig / "trace_share.png", bbox_inches="tight")
    plt.close(f)

    # 2. departures by depth: the share within 10 degrees
    f, ax = plt.subplots(1, 4, figsize=(15, 3.4), dpi=90, sharey=True)
    lbl = (
        "echo 1 vs combined",
        "echo 2 vs combined",
        "echo 3 vs combined",
        "echo 2 vs echo 3",
    )
    sty = ("-", "-", "-", "--")
    col = COLORS + ("#555555",)
    k10 = int(np.searchsorted(ab, 10))
    for i, name in enumerate(REGIONS):
        ang = np.array(region(h, name)["angle"])
        n = ang.sum(-1)
        for k in range(4):
            m = n[k] > 100
            ax[i].plot(
                dc[m],
                ang[k][m, :k10].sum(-1) / n[k][m],
                sty[k],
                color=col[k],
                lw=2,
                label=lbl[k],
            )
        ax[i].set_title(name, fontsize=9)
        ax[i].set_xlabel("WM distance, mm")
        ax[i].grid(alpha=0.3)
    ax[0].set_ylabel("share within 10 degrees (interior)")
    ax[0].legend(fontsize=7, frameon=False)
    f.suptitle(
        "Agreement of each echo's normal with the combined normal, by depth",
        fontsize=10,
    )
    f.tight_layout()
    f.savefig(a.fig / "departure_depth.png", bbox_inches="tight")
    plt.close(f)

    # 3. axis dependence (total)
    r = h["total"]
    f, ax = plt.subplots(1, 3, figsize=(13, 3.4), dpi=90)
    coh = np.array(r["coh_axis"]) / np.maximum(np.array(r["coh_axis_n"]), 1)
    e23 = np.array(r["e23_axis"]) / np.maximum(np.array(r["e23_axis_n"]), 1)
    agx = np.array(r["agree_axis"])  # (3, axis, agree)
    for k in range(3):
        ax[0].plot(
            xc, coh[k], marker="o", ms=4, color=COLORS[k], label=f"array axis {k}"
        )
        ax[1].plot(xc, e23[k], marker="o", ms=4, color=COLORS[k])
        cum = np.cumsum(agx[k], -1) / np.maximum(agx[k].sum(-1, keepdims=True), 1)
        med = np.array([gc[min(np.searchsorted(c, 0.5), len(gc) - 1)] for c in cum])
        ax[2].plot(xc, med, marker="o", ms=4, color=COLORS[k])
    ax[0].set_ylabel("mean coherence (interior)")
    ax[1].set_ylabel("echo 2 vs echo 3 within 10 degrees")
    ax[2].set_ylabel("median surface agreement, degrees")
    for x in ax:
        x.set_xlabel("angle of the normal to the array axis, degrees")
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=7, frameon=False)
    f.suptitle(
        "Dependence on the normal's angle to each array axis (whole hemisphere)",
        fontsize=10,
    )
    f.tight_layout()
    f.savefig(a.fig / "axis.png", bbox_inches="tight")
    plt.close(f)

    # 4. site slices of the departure
    sites = [n for n in REGIONS[1:] if (a.out / f"site_{n}.npz").exists()]
    if sites:
        f, ax = plt.subplots(
            len(sites), 4, figsize=(13, 3.3 * len(sites)), dpi=80, squeeze=False
        )
        for i, n in enumerate(sites):
            z = np.load(a.out / f"site_{n}.npz")
            e3 = z["echo3"].T
            ax[i, 0].imshow(np.log(np.maximum(e3, 1)), cmap="gray", origin="lower")
            ax[i, 0].set_title(f"{n}: log echo 3", fontsize=9)
            for e in range(3):
                d = np.where(np.isfinite(z["coherence"]), z["departure"][e], np.nan).T
                im = ax[i, e + 1].imshow(
                    d, cmap="magma", vmin=0, vmax=45, origin="lower"
                )
                ax[i, e + 1].set_title(f"echo {e + 1} vs combined (deg)", fontsize=9)
            for x in ax[i]:
                x.axis("off")
        f.colorbar(im, ax=ax[:, 1:], shrink=0.6)
        f.savefig(a.fig / "departure_sites.png", bbox_inches="tight")
        plt.close(f)

    # 5. the agreement map on the overview mesh, and its histogram
    ov = WhiteOverview.load(overview_paths(a.surf)["overview"])
    z = np.load(a.out / "agreement.npz")
    n_true = int(ov.sidecar["TrueVertices"])
    val = face_values(ov, z["vertex"], z["angle"], n_true)
    v = ov.v.astype(np.float64) @ ov.affine[:3, :3].T + ov.affine[:3, 3]
    f = plt.figure(figsize=(14, 9), dpi=80)
    # the views: along the principal axes of the vertices (the shortest,
    # about medial-lateral, first), the longest axis horizontal
    c = v - v.mean(0)
    _, pc = np.linalg.eigh(c.T @ c)
    p_mid, p_short = pc[:, 1], pc[:, 0]
    views = [
        ("along +PC3 (shortest axis)", p_short, p_mid),
        ("along -PC3", -p_short, p_mid),
        ("along +PC2", p_mid, p_short),
        ("along -PC2", -p_mid, p_short),
    ]
    for i, (nm, view, up) in enumerate(views):
        ax = f.add_subplot(2, 3, (1, 2, 4, 5)[i])
        render(ax, v, ov.f, val, view, up, 45.0)
        ax.set_title(nm, fontsize=9)
    ax = f.add_subplot(2, 3, 3)
    ag = np.array(h["total"]["agree"])
    ax.bar(gc, ag / ag.sum(), width=2.0, color="#1f5f99")
    ax.set_xlim(0, 90)
    ax.set_xlabel("angle, degrees")
    ax.set_ylabel("share of roots")
    ax.set_title("surface agreement, whole hemisphere", fontsize=9)
    ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(0, 45))
    cax = f.add_axes([0.70, 0.12, 0.2, 0.02])
    f.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        label="mean angle per overview face, degrees (grey: no value)",
    )
    f.suptitle(
        "Angle between the white surface's normal and orient's data normal", fontsize=11
    )
    f.savefig(a.fig / "agreement_map.png", bbox_inches="tight")
    plt.close(f)


if __name__ == "__main__":
    main()
