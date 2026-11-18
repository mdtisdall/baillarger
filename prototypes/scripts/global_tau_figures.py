"""Figures of a global-τ run.

usage: global_tau_figures.py SITE_NPZ RUN_DIR FIG_DIR
Writes to FIG_DIR:
- tau_levels.png: three orthogonal slices through the site centre: echo 2,
  the τ level sets every 0.5 mm below the top (one hue, light = deep), the
  top τ = 0 (orange) and the contact faces (red), for τ0 and the final τ;
- ownership.png: the same slices; the owners coloured by the white-surface
  piece of their patch (3 categorical slots), the owner-patch borders in
  grey, the contact faces in red; first march and final;
- tau_outside.png: the cortex voxels with τ > 0 (beyond the top) after the
  first solve, over echo 2;
- contacts.png: the contact measures (contacts.json): τ of the face voxels,
  the thickness of the contact and open patches, the profile correlation;
- timing.png: per outer iteration the times (solve, upwind) and the owner
  changes; per CG solve the iterations and the time.
"""

import json
import sys
from pathlib import Path

import global_tau_prep as TP
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import csgraph  # noqa: E402

PIECE = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
TOP = "#eb6834"
FACE = "#e34948"
INK, MUTED = "#0b0b0b", "#52514e"
LEVELS = np.arange(-3.0, -0.01, 0.5)
AX = np.eye(3, dtype=np.int64)


def pieces(r: np.lib.npyio.NpzFile) -> np.ndarray:
    n = len(r["centre"])
    adj = r["adj"]
    A = sparse.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    _, lab = csgraph.connected_components(A, directed=False)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[lab]


def face_mask(faces: np.ndarray, shape: tuple[int, ...]) -> list[np.ndarray]:
    """Per axis a boolean grid: a contact face between v and v + e_axis."""
    out = [np.zeros(shape, bool) for _ in range(3)]
    for a in range(3):
        f = faces[faces[:, 3] == a, :3]
        out[a][tuple(f.T)] = True
    return out


def draw_faces(ax: plt.Axes, fm: list[np.ndarray], axis: int, k: int) -> None:
    """Contact faces cut by the slice (axis = k) as red segments."""
    rest = [a for a in range(3) if a != axis]
    segs = []
    for a in rest:
        m = np.take(fm[a], k, axis=axis)
        ii, jj = np.nonzero(m)
        for i, j in zip(ii, jj):
            if a == rest[0]:  # the face is between rows i and i+1
                segs.append(((j - 0.5, i + 0.5), (j + 0.5, i + 0.5)))
            else:
                segs.append(((j + 0.5, i - 0.5), (j + 0.5, i + 0.5)))
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(segs, colors=FACE, linewidths=1.2))


def tau_panel(
    ax: plt.Axes,
    e2s: np.ndarray,
    ts: np.ndarray,
    dm: np.ndarray,
    fm: list[np.ndarray],
    axis: int,
    k: int,
    title: str,
) -> None:
    ax.imshow(e2s, cmap="gray", origin="lower", interpolation="nearest")
    t = np.where(dm, ts, np.nan)
    cmap = plt.get_cmap("Blues")
    cols = [
        cmap(0.35 + 0.6 * (i / max(len(LEVELS) - 1, 1))) for i in range(len(LEVELS))
    ][::-1]
    ax.contour(np.nan_to_num(t, nan=5.0), levels=LEVELS, colors=cols, linewidths=0.8)
    ax.contour(np.nan_to_num(t, nan=5.0), levels=[0.0], colors=[TOP], linewidths=1.6)
    draw_faces(ax, fm, axis, k)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])


def slices(vol: np.ndarray, c: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    return [(a, int(c[a]), np.take(vol, int(c[a]), axis=a)) for a in range(3)]


def main() -> None:
    site_p, run, figd = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    figd.mkdir(parents=True, exist_ok=True)
    z = np.load(site_p)
    r = np.load(run / "result.npz")
    log = json.loads((run / "log.json").read_text())
    e2 = z["echoes"][1].astype(float)
    lab = z["labels"]
    dom = r["domain"]
    c = np.round(r["centre_vox"]).astype(int)
    shape = lab.shape
    fm = face_mask(r["faces"], shape)
    names = ["x", "y", "z"]

    # tau levels: tau0 and final
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.6))
    fm0 = face_mask(np.zeros((0, 4), int), shape)
    for row, (key, title, fmr) in enumerate(
        [("tau0", "τ0 (distance init)", fm0), ("tau", "final τ", fm)]
    ):
        for a, k, e2s in slices(e2, c):
            ts = np.take(r[key], k, axis=a)
            dms = np.take(dom, k, axis=a)
            tau_panel(
                axs[row, a], e2s, ts, dms, fmr, a, k, f"{title}, {names[a]} = {k}"
            )
    fig.suptitle(
        "τ level sets over echo 2 (every 0.5 mm; light = deep), top τ = 0 orange, contact faces red",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(figd / "tau_levels.png", dpi=110)
    plt.close(fig)

    # ownership
    pc = pieces(r)
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.6))
    for row, (key, title) in enumerate(
        [("owner_march", "first march (tensor metric)"), ("owner", "final (upwind)")]
    ):
        own = r[key]
        for a, k, e2s in slices(e2, c):
            o = np.take(own, k, axis=a)
            img = np.ones((*o.shape, 3))
            for p in range(min(3, pc.max() + 1)):
                m = o >= 0
                mm = np.zeros_like(m)
                mm[m] = pc[o[m]] == p
                img[mm] = matplotlib.colors.to_rgb(PIECE[p])
            g = e2s / np.percentile(e2s, 99)
            img = 0.55 * img + 0.45 * np.clip(g, 0, 1)[..., None]
            img[o < 0] = np.clip(g, 0, 1)[o < 0, None] * 0.6
            ax = axs[row, a]
            ax.imshow(img, origin="lower", interpolation="nearest")
            b = (np.diff(o, axis=0, prepend=o[:1]) != 0) | (
                np.diff(o, axis=1, prepend=o[:, :1]) != 0
            )
            ax.imshow(
                np.ma.masked_where(~b | (o < 0), b),
                cmap="Greys",
                vmin=0,
                vmax=3,
                origin="lower",
                interpolation="nearest",
                alpha=0.6,
            )
            draw_faces(
                ax, fm if row == 1 else face_mask(np.zeros((0, 4), int), shape), a, k
            )
            ax.set_title(f"{title}, {names[a]} = {k}", fontsize=9, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
    hd = [
        matplotlib.patches.Patch(color=PIECE[p], label=f"white-surface piece {p}")
        for p in range(3)
    ]
    hd.append(matplotlib.lines.Line2D([], [], color=FACE, label="contact face (final)"))
    fig.legend(handles=hd, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        "Owner patches coloured by their white-surface piece; grey = owner-patch borders",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figd / "ownership.png", dpi=110)
    plt.close(fig)

    # tau > 0 in cortex after the first solve
    ctx = (lab == TP.CORTEX) & dom & ~r["inner"]
    t1 = r["snap_tau"][0]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.2))
    for a, k, e2s in slices(e2, c):
        ax = axs[a]
        ax.imshow(e2s, cmap="gray", origin="lower", interpolation="nearest")
        m = np.take(ctx & (t1 > 0), k, axis=a)
        ov = np.zeros((*m.shape, 4))
        ov[m] = (*matplotlib.colors.to_rgb(FACE), 0.6)
        ax.imshow(ov, origin="lower", interpolation="nearest")
        draw_faces(ax, face_mask(r["faces"], shape), a, k)
        ax.set_title(f"{names[a]} = {k}", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    n1 = int((ctx & (t1 > 0)).sum())
    fig.suptitle(
        f"Cortex-label voxels with τ > 0 after the first solve (red fill, {n1} of {int(ctx.sum())}); "
        "contact faces red lines",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(figd / "tau_outside.png", dpi=110)
    plt.close(fig)

    # contact measures
    cj = run / "contacts.json"
    if cj.exists():
        cd = json.loads(cj.read_text())
        cs = cd["contacts"]
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
        tf = np.array([t for o in cs for t in o["tau_face_mm"]])
        axs[0].hist(tf, bins=40, color=PIECE[0])
        axs[0].axvline(0, color=MUTED, lw=1)
        axs[0].set_xlabel("median τ of the face voxels per side (mm)")
        axs[0].set_ylabel("contact pair sides")
        thc = np.array([t for o in cs for t in o["thickness_mm"]])
        th = r["thickness"]
        cont = np.zeros(len(th), bool)
        for o in cs:
            cont[o["a"]] = cont[o["b"]] = True
        bins = np.linspace(0, 6, 31)
        axs[1].hist(
            th[~cont & np.isfinite(th)],
            bins=bins,
            color=PIECE[0],
            alpha=0.9,
            label="open patches",
        )
        axs[1].hist(
            thc[np.isfinite(thc)],
            bins=bins,
            color=PIECE[1],
            alpha=0.9,
            label="contact patch sides",
        )
        axs[1].set_xlabel("thickness -τ(root) (mm)")
        axs[1].legend(fontsize=8, frameon=False)
        cc = np.array([o["signature_corr"] for o in cs])
        axs[2].hist(cc[np.isfinite(cc)], bins=30, color=PIECE[0])
        axs[2].set_xlabel("echo-2 profile correlation, a vs b")
        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            f"Contact measures: {cd['summary']['contact_pairs']} patch pairs, "
            f"{cd['summary']['contact_faces']} faces",
            fontsize=10,
            color=INK,
        )
        fig.tight_layout()
        fig.savefig(figd / "contacts.png", dpi=110)
        plt.close(fig)

    # timing
    o = log["outer"]
    k = np.arange(len(o))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    axs[0].bar(
        k, [x["solve_s"] for x in o], color=PIECE[0], width=0.7, label="LM solve"
    )
    axs[0].bar(
        k,
        [x["upwind_s"] + x["face_s"] for x in o],
        bottom=[x["solve_s"] for x in o],
        color=PIECE[1],
        width=0.7,
        label="upwind + face test",
    )
    axs[0].set_xlabel("outer iteration")
    axs[0].set_ylabel("time (s)")
    axs[0].legend(fontsize=8, frameon=False)
    axs[1].plot(
        k,
        [max(x["owner_changes_total"], 0.5) for x in o],
        color=PIECE[0],
        lw=2,
        marker="o",
        ms=4,
    )
    axs[1].set_yscale("log")
    axs[1].set_xlabel("outer iteration")
    axs[1].set_ylabel("owner changes")
    axs[2].plot(
        k, [x["faces_after"] for x in o], color=PIECE[0], lw=2, marker="o", ms=4
    )
    axs[2].set_xlabel("outer iteration")
    axs[2].set_ylabel("contact faces")
    sv = [s for s in log["solves"] if "/" not in s["tag"]]
    axs[3].scatter(
        [s["cg_iters"] for s in sv], [s["cg_s"] for s in sv], s=10, color=PIECE[0]
    )
    axs[3].set_xlabel("CG iterations (per solve)")
    axs[3].set_ylabel("CG time (s)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Timing and the fixed point ({log['domain_voxels']} voxels)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(figd / "timing.png", dpi=110)
    plt.close(fig)
    print("figures in", figd)


if __name__ == "__main__":
    main()
