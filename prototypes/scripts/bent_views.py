"""The diagnostic figures of the bent-columns trial at the site
(docs/plans/bent-columns-trial.md, phase 6, "Diagnostics"): views for the
user, not pass/fail numbers. Each view shows the straight rays and each fit
side by side.

    python prototypes/scripts/bent_views.py --box BOX.npz \\
        --texture TEXTURE.npz --out DIR [--figures FIGDIR]

reads the outputs of bent_columns.py in DIR (straight.npz, <arm>_paths.npz,
<arm>_costs.json, wrong_labels.npz, sparse.json, sparse_columns.npz,
scan.md) and writes the PNGs to FIGDIR (default DIR/figures) and the
numbers of the views to DIR/views.json.

Geometry
--------
- The vertex column: the column of the box's vertex; n_v its normal, r_v
  its root.
- The section (as the viewer's section pane): the inner columns whose roots
  lie within SECTION_HALF_MM of the plane through r_v that holds n_v and the
  section axis u_x (world x made perpendicular to n_v), in order of
  (r - r_v) . u_x.
- The drops: the column pairs of the plan (DROPS). The cuts go through the
  first column of each pair: along the section (the plane of n_c and u_x
  made perpendicular to n_c) and across it (n_c and n_c x that axis).
- The map coordinates: (r - r_v) . t, t = n_v x u_s, with u_s
  bent_texture's split axis (the texture file's "u", across the sulcus);
  and across the sulcus the signed distance along the mesh from the fundus
  line (unrolled). The map is a view, not an isometric flattening.
- The parts (bank -u, fundus, bank +u): bent_texture.split_banks, read from
  the texture file ("part").

The quantities per column are bent_columns.column_stats (saved by the fit).
Where the tops land (fundus columns): the gradient of the soft outside map
O at the top point (scanner frame). "tip" if its angle to the column's
normal n_c is below TIP_DEG (the outside lies above the column: the end of
the slot); else the wall whose outward side (the direction of the gradient)
is +u_s (the pia of bank +u, whose normals point +u_s) or -u_s. Crowding:
for each inner face whose three columns all have a top, the signed area of
the triangle of the three top points (oriented by the face's root normal)
over the root triangle's area; per part (faces whose three columns are in
one part), the sum of the top areas over the sum of the root areas.
Alignment: the angle (0-90 deg) between the path's tangent and the data
normal v (the texture's leading eigenvector, nearest voxel) at the samples
with mask >= 0.5, in the cortex label, where the coherence c (nearest
voxel) is at least C_HIGH.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent))
import bent_box as BB  # noqa: E402
import bent_columns as BC  # noqa: E402
import bent_fit as BF  # noqa: E402
import bent_terms as BT  # noqa: E402

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

#: the half width of the section, mm (the viewer's default)
SECTION_HALF_MM = 0.25
#: the drops of the plan's "Why": pairs of true vertices
DROPS = ((4924217, 4924847), (12183700, 12184198))
#: the cuts: half width along the plane, the range along n_c, the step, and
#: the half thickness of the slab of columns drawn
CUT_HALF_MM = 3.0
CUT_V = (-1.0, 7.0)
CUT_STEP = 0.04
CUT_SLAB_MM = 0.04
CUT_MAX_COLUMNS = 40
#: the depth bins of the profiles by depth, mm
BIN_MM = 0.16
#: the coherence above which the alignment angle is read
C_HIGH = 0.7
#: a top is at the tip if the gradient of O is within this angle of n_c
TIP_DEG = 45.0
#: a wrong-label voxel this far from label 0 is "deep" (not the rim), mm
DEEP_MM = 0.32
#: the fundus line of the maps: |n . u_s| below this
FLOOR_NU = 0.05
#: a jump of the soft top between neighbours along the section, mm
JUMP_MM = 1.0
PART_NAMES = {-1: "bank -u", 0: "fundus", 1: "bank +u"}
RUN_ORDER = ("straight", "A7", "A", "B7", "B")


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


class Site:
    """The box, the inputs and every run's saved quantities."""

    def __init__(self, box_path: Path, texture: Path, out: Path) -> None:
        self.out = out
        self.box, self.prob = BC.load_problem(box_path, texture)
        self.inp = self.prob.inputs
        b = self.box
        with np.load(texture) as z:
            self.c = np.asarray(z["c"], np.float64)
            self.lead = np.asarray(z["leading"], np.float64)
            self.part = np.asarray(z["part"], np.int64)
            self.u_s = np.asarray(z["u"], np.float64)
        self.wrong = BC.load_wrong(out)
        vids = b.rays.roots.vertex
        self.vids = np.asarray(vids, np.int64)
        self.vi = int(np.flatnonzero(self.vids == b.vertex)[0])
        self.n_v = b.normals[self.vi].astype(np.float64)
        self.r_v = b.roots[self.vi].astype(np.float64)
        ux = np.array([1.0, 0.0, 0.0]) - self.n_v[0] * self.n_v
        self.u_x = ux / np.linalg.norm(ux)
        w = np.cross(self.n_v, self.u_x)
        d = b.roots.astype(np.float64) - self.r_v
        sel = np.flatnonzero(b.inner & (np.abs(d @ w) <= SECTION_HALF_MM))
        o = np.argsort(d[sel] @ self.u_x)
        self.section = sel[o]
        self.section_u = (d @ self.u_x)[self.section]
        t = np.cross(self.n_v, self.u_s)
        self.t = t / np.linalg.norm(t)
        xt = d @ self.t
        nu_ = b.normals.astype(np.float64) @ self.u_s
        self.map_xy = np.stack([xt, unrolled(b, nu_)], 1)
        self.runs: dict[str, dict[str, Any]] = {}
        for name in RUN_ORDER:
            q = out / ("straight.npz" if name == "straight" else f"{name}_paths.npz")
            if not q.exists():
                continue
            with np.load(q) as z:
                r: dict[str, Any] = {
                    k[4:]: np.asarray(z[k]) for k in z.files if k.startswith("col_")
                }
                r["params"] = torch.as_tensor(np.asarray(z["params"], np.float32))
            self.runs[name] = r
        self._paths: dict[str, BT.Paths] = {}
        self.deep: Bool | None = None

    def paths(self, name: str) -> BT.Paths:
        if name not in self._paths:
            with torch.no_grad():
                self._paths[name] = BT.paths(self.runs[name]["params"], self.inp)
        return self._paths[name]

    def col(self, vid: int) -> int:
        return int(np.flatnonzero(self.vids == vid)[0])

    def arms(self) -> list[str]:
        return [k for k in self.runs if k != "straight"]


def unrolled(b: BB.Box, nu: F64) -> F64:
    """The across coordinate of the maps: the distance along the mesh edges
    (scipy dijkstra over box.edges, root to root, mm) from the fundus line,
    signed by n . u_s. The fundus line: the columns with |n . u_s| below
    FLOOR_NU (the floor of the fundus, where the normals turn from one bank
    to the other)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    r = b.roots.astype(np.float64)
    e = np.asarray(b.edges, np.int64)
    ln = np.linalg.norm(r[e[:, 0]] - r[e[:, 1]], axis=1)
    n = len(r)
    G = coo_matrix(
        (np.r_[ln, ln], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
        shape=(n, n),
    ).tocsr()
    seeds = np.flatnonzero(np.abs(nu) < FLOOR_NU)
    dist = np.asarray(dijkstra(G, indices=seeds, min_only=True))
    out: F64 = np.sign(nu) * dist
    return out


def _np(t: torch.Tensor) -> F64:
    out: F64 = t.double().cpu().numpy()
    return out


# ---------------------------------------------------------------- the cuts


def _plane_img(
    site: Site, origin: F64, eh: F64, ev: F64
) -> tuple[F64, npt.NDArray[np.int64], F64, F64]:
    h = np.arange(-CUT_HALF_MM, CUT_HALF_MM + 1e-9, CUT_STEP)
    v = np.arange(CUT_V[0], CUT_V[1] + 1e-9, CUT_STEP)
    H, V = np.meshgrid(h, v, indexing="xy")
    x = origin + H[..., None] * eh + V[..., None] * ev
    p = site.box.to_box(x).reshape(-1, 3).T
    img = map_coordinates(
        site.box.echoes[1].astype(np.float64), p, order=1, mode="nearest"
    ).reshape(H.shape)
    lab = map_coordinates(site.box.labels, p, order=0, mode="nearest").reshape(H.shape)
    return img, lab.astype(np.int64), h, v


def view_cuts(site: Site, path: Path, drop: tuple[int, int]) -> dict[str, Any]:
    """Two rows (along, across the section) by one panel per run."""
    plt = _plt()
    c0 = site.col(drop[0])
    b = site.box
    nc = b.normals[c0].astype(np.float64)
    rc = b.roots[c0].astype(np.float64)
    al = site.u_x - (site.u_x @ nc) * nc
    al /= np.linalg.norm(al)
    ac = np.cross(nc, al)
    runs = list(site.runs)
    fig, axs = plt.subplots(2, len(runs), figsize=(3.0 * len(runs), 7.6), squeeze=False)
    p0 = _np(site.paths("straight").x)
    info: dict[str, Any] = {}
    roots = b.roots.astype(np.float64) - rc
    for row, (eh, en, nm) in enumerate(((al, ac, "along"), (ac, al, "across"))):
        img, lab, h, v = _plane_img(site, rc, eh, nc)
        lo, hi = np.percentile(img, [1, 99])
        near = np.flatnonzero(
            (np.abs(roots @ en) <= CUT_SLAB_MM) & (np.abs(roots @ eh) <= CUT_HALF_MM)
        )
        near = near[np.argsort(roots[near] @ eh)]
        pick = near[:: max(1, int(np.ceil(len(near) / CUT_MAX_COLUMNS)))]
        if c0 not in pick:
            pick = np.r_[pick, c0]
        info[nm] = dict(columns_in_slab=len(near), drawn=len(pick))
        for k, run in enumerate(runs):
            a = axs[row, k]
            a.imshow(
                img,
                origin="lower",
                cmap="gray",
                vmin=lo,
                vmax=hi,
                extent=(h[0], h[-1], v[0], v[-1]),
            )
            a.contour(
                h, v, (lab == b.cortex).astype(float), [0.5], colors="y", linewidths=0.6
            )
            a.contour(h, v, (lab == 0).astype(float), [0.5], colors="c", linewidths=0.6)
            st = site.runs[run]
            P = _np(site.paths(run).x)
            for j in pick:
                for X, colr, lw, rs in (
                    (p0, "#30d5ff", 0.5, site.runs["straight"]),
                    (P, "#ff4040", 0.8, st),
                ):
                    if run == "straight" and colr == "#ff4040":
                        continue
                    q = X[j] - rc
                    ns = int(rs["ns"][j])
                    hh, vv = q @ eh, q @ nc
                    if rs["censored"][j]:
                        a.plot(hh, vv, "-", c="orange", lw=lw)
                    else:
                        a.plot(hh[:ns], vv[:ns], "-", c=colr, lw=lw)
                        a.plot(hh[ns - 1 :], vv[ns - 1 :], ":", c=colr, lw=0.4)
            a.plot(0, 0, "m*", ms=7)
            a.set_xlim(h[0], h[-1])
            a.set_ylim(v[0], v[-1])
            a.set_title(f"{run}, {nm} the section", fontsize=8)
            a.tick_params(labelsize=6)
            if k == 0:
                a.set_ylabel(f"along n_c (mm), column of v {drop[0]}", fontsize=7)
    fig.suptitle(
        f"drop {drop[0]} / {drop[1]}: echo 2; yellow cortex, cyan outside label "
        "edge; cyan straight rays, red fitted paths (solid to the soft top, dotted "
        "beyond), orange censored; star: the root",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return info


# ---------------------------------------------------------------- section


def _first_outside_dseg(b: BB.Box) -> F64:
    BTX: Any = __import__("bent_texture")
    r = b.rays
    out: F64 = BTX.first_outside(
        r.interval_start, r.interval_s0, r.interval_s1, r.interval_class
    )
    return out


def view_sawtooth(site: Site, path: Path) -> dict[str, Any]:
    plt = _plt()
    sec, u = site.section, site.section_u
    fo = _first_outside_dseg(site.box)
    runs = list(site.runs)
    fig, axs = plt.subplots(len(runs), 1, figsize=(12, 2.3 * len(runs)), sharex=True)
    out: dict[str, Any] = dict(section_columns=len(sec))
    drops = [(site.col(a), site.col(b)) for a, b in DROPS]
    for a, run in zip(axs, runs, strict=True):
        st = site.runs[run]
        top = st["top_arc"][sec]
        hard = st["hard_top"][sec]
        cen = st["censored"][sec]
        if run != "straight":
            a.plot(
                u,
                site.runs["straight"]["top_arc"][sec],
                "-",
                c="0.75",
                lw=0.8,
                label="straight soft top",
            )
        else:
            a.plot(u, fo[sec], ".", c="k", ms=1.5, label="dseg first outside")
        a.plot(u, hard, ".", c="tab:blue", ms=2, label="first label-0 sample (arc)")
        a.plot(u, top, ".", c="tab:orange", ms=2, label="soft top (arc)")
        a.plot(u[cen], np.full(cen.sum(), 8.2), "v", c="r", ms=2, label="censored")
        for c1, c2 in drops:
            for cc in (c1, c2):
                k = np.flatnonzero(sec == cc)
                if len(k):
                    a.axvline(u[k[0]], c="m", lw=0.5, ls="--")
        dt = np.diff(top)
        ok = np.isfinite(dt)
        out[run] = dict(
            censored=int(cen.sum()),
            soft_top_median=float(np.nanmedian(top)),
            jumps_gt_1mm=int((np.abs(dt[ok]) > JUMP_MM).sum()),
            neighbour_abs_diff_median=float(np.median(np.abs(dt[ok])))
            if ok.any()
            else None,
            neighbour_abs_diff_p99=float(np.percentile(np.abs(dt[ok]), 99))
            if ok.any()
            else None,
            drops={
                f"{v1}/{v2}": dict(
                    soft_top=[float(st["top_arc"][c1]), float(st["top_arc"][c2])],
                    first_label0=[float(st["hard_top"][c1]), float(st["hard_top"][c2])],
                    length=[float(st["length"][c1]), float(st["length"][c2])],
                    censored=[bool(st["censored"][c1]), bool(st["censored"][c2])],
                )
                for (v1, v2), (c1, c2) in zip(DROPS, drops, strict=True)
            },
        )
        a.set_ylim(0, 8.5)
        a.set_ylabel(f"{run}\ndepth (mm)", fontsize=8)
        a.legend(fontsize=6, loc="upper left", ncol=5)
        a.tick_params(labelsize=7)
    axs[-1].set_xlabel("u along the section (world x, perpendicular to n_v), mm")
    fig.suptitle(
        f"the first outside along the section ({len(sec)} inner columns, half "
        f"width {SECTION_HALF_MM} mm); magenta: the drop columns",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return out


def view_profiles(site: Site, path: Path) -> None:
    """The section's profiles, as the viewer's section pane: echo 2 along
    s (outward down), the inward part straight, one column per root."""
    plt = _plt()
    sec = site.section
    inp = site.inp
    s_in = -np.arange(int(round(2.0 / BT.STEP)), 0, -1) * BT.STEP
    runs = list(site.runs)
    fig, axs = plt.subplots(len(runs), 1, figsize=(12, 2.6 * len(runs)), sharex=True)
    rv = _np(inp.root_vox)[sec]
    nv = _np(inp.normal)[sec] @ _np(inp.lin).T
    vin = rv[:, None] + s_in[None, :, None] * nv[:, None]
    e2 = site.box.echoes[1].astype(np.float64)
    lo = hi = None
    for a, run in zip(axs, runs, strict=True):
        vox = np.concatenate([vin, _np(site.paths(run).vox)[sec]], 1)
        img = map_coordinates(e2, vox.reshape(-1, 3).T, order=1, mode="nearest")
        img = img.reshape(len(sec), -1)
        if lo is None:
            lo, hi = np.percentile(img, [1, 99])
        ss = np.r_[s_in, np.arange(BT.NS) * BT.STEP]
        a.imshow(
            img.T,
            aspect="auto",
            cmap="gray",
            vmin=lo,
            vmax=hi,
            extent=(-0.5, len(sec) - 0.5, ss[-1], ss[0]),
        )
        st = site.runs[run]
        x = np.arange(len(sec))
        a.plot(x, st["top_arc"][sec], ".", c="tab:orange", ms=1.2)
        a.plot(x, st["hard_top"][sec], ".", c="tab:cyan", ms=1.2)
        a.axhline(0, c="#ff2bd6", ls="--", lw=0.6)
        a.set_ylabel(f"{run}\ns (mm)", fontsize=8)
        a.tick_params(labelsize=7)
    axs[-1].set_xlabel("section columns in order of u", fontsize=8)
    fig.suptitle(
        "section profiles, echo 2 along each path (outward down; s < 0 the "
        "straight inward part); orange soft top, cyan first label-0 sample "
        "(arc lengths)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ---------------------------------------------------------------- maps


def _map(
    site: Site,
    a: Any,
    val: F64,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    mask: Bool | None = None,
    wrong: Bool | None = None,
) -> Any:
    xy = site.map_xy
    inn = site.box.inner
    a.scatter(xy[~inn, 0], xy[~inn, 1], s=0.3, c="0.85", rasterized=True)
    m = inn if mask is None else inn & mask
    rest = inn & ~m
    a.scatter(xy[rest, 0], xy[rest, 1], s=0.3, c="0.55", rasterized=True)
    sc = a.scatter(
        xy[m, 0],
        xy[m, 1],
        s=0.4,
        c=val[m],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    if wrong is not None and wrong.any():
        w = wrong & inn
        a.scatter(xy[w, 0], xy[w, 1], s=0.6, c="k", marker="x", lw=0.3, rasterized=True)
    a.set_aspect("equal")
    a.set_title(title, fontsize=7)
    a.tick_params(labelsize=6)
    return sc


def _wrong(site: Site, run: str) -> Bool | None:
    st = site.runs[run]
    if "wrong" not in st:
        return None
    w: Bool = np.asarray(st["wrong"], bool) | np.asarray(
        site.runs["straight"]["wrong"], bool
    )
    return w


def view_bend_map(site: Site, path: Path) -> dict[str, Any]:
    plt = _plt()
    arms = site.arms()
    fig, axs = plt.subplots(2, len(arms), figsize=(3.6 * len(arms), 7), squeeze=False)
    inn = site.box.inner
    vals = [site.runs[r][k][inn] for r in arms for k in ("root_angle", "top_angle")]
    vmax = float(np.percentile(np.concatenate(vals), 99)) if vals else 1.0
    out: dict[str, Any] = {}
    for k, run in enumerate(arms):
        st = site.runs[run]
        for row, key in enumerate(("root_angle", "top_angle")):
            sc = _map(
                site,
                axs[row, k],
                st[key],
                f"{run}: {key} (deg)",
                0,
                vmax,
                "magma",
                wrong=_wrong(site, run),
            )
        out[run] = {
            key: dict(
                median=float(np.median(st[key][inn])),
                p90=float(np.percentile(st[key][inn], 90)),
                max=float(st[key][inn].max()),
            )
            for key in ("root_angle", "top_angle", "turning")
        }
    fig.colorbar(sc, ax=axs, shrink=0.6, label="degrees")
    fig.suptitle(
        "bend: angle of the tangent to the normal at the root (top row) and at "
        "the top (bottom row); map: t along the fundus x mesh distance from "
        "the fundus line, signed by n . u (mm); "
        "x: passes a wrong-label piece; grey: halo",
        fontsize=8,
    )
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return out


def view_bend_depth(site: Site, path: Path) -> dict[str, Any]:
    """Median shear |d(s)| and the median angle of t(s) to n by depth, per
    part, over the inner columns with mask >= 0.5 at s."""
    plt = _plt()
    inn = site.box.inner
    nrm = _np(site.inp.normal)
    edges = np.arange(0.0, BT.S_MAX + BIN_MM, BIN_MM)
    sb = np.arange(BT.NS) * BT.STEP
    kb = np.digitize(sb, edges) - 1
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    out: dict[str, Any] = {}
    for run in site.arms():
        p = site.paths(run)
        with torch.no_grad():
            m = _np(BT.soft_top(p, site.inp).mask)
        d = np.linalg.norm(_np(p.d), axis=-1)
        ang = np.degrees(
            np.arccos(np.clip(np.abs((_np(p.tangent) * nrm[:, None]).sum(-1)), 0, 1))
        )
        out[run] = {}
        for j, q in enumerate((-1, 0, 1)):
            sel = inn & (site.part == q)
            md: list[float] = []
            ma: list[float] = []
            pd: list[float] = []
            pa: list[float] = []
            for i in range(len(edges) - 1):
                use = m[sel][:, kb == i] >= 0.5
                if int(use.sum()) < 50:
                    for x in (md, ma, pd, pa):
                        x.append(np.nan)
                    continue
                dd = d[sel][:, kb == i][use]
                aa = ang[sel][:, kb == i][use]
                md.append(float(np.median(dd)))
                ma.append(float(np.median(aa)))
                pd.append(float(np.percentile(dd, 90)))
                pa.append(float(np.percentile(aa, 90)))
            mid = (edges[:-1] + edges[1:]) / 2
            ln = axs[0, j].plot(mid, md, "-o", ms=2, label=f"{run} median")
            axs[0, j].plot(
                mid, pd, "--", c=ln[0].get_color(), lw=0.8, label=f"{run} p90"
            )
            ln = axs[1, j].plot(mid, ma, "-o", ms=2, label=f"{run} median")
            axs[1, j].plot(
                mid, pa, "--", c=ln[0].get_color(), lw=0.8, label=f"{run} p90"
            )

            def rd(v: list[float], k: int) -> list[float | None]:
                return [None if np.isnan(x) else round(x, k) for x in v]

            out[run][PART_NAMES[q]] = dict(
                depth_mm=[round(float(x), 2) for x in mid],
                shear_median_mm=rd(md, 4),
                shear_p90_mm=rd(pd, 4),
                angle_median_deg=rd(ma, 2),
                angle_p90_deg=rd(pa, 2),
            )
            axs[0, j].set_title(PART_NAMES[q], fontsize=9)
    axs[0, 0].set_ylabel("|d(s)| (mm): median, p90 dashed")
    axs[1, 0].set_ylabel("angle of t(s) to n (deg): median, p90 dashed")
    for a in axs[1]:
        a.set_xlabel("s (mm), samples below the soft top")
    axs[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(
        "bend against depth: constant angle from the root = bent at the root; "
        "growing with s = curving with depth",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return out


def view_lengths(site: Site, path: Path) -> dict[str, Any]:
    plt = _plt()
    runs = list(site.runs)
    st0 = site.runs["straight"]
    ref = st0["hard_top"]
    fig, axs = plt.subplots(2, len(runs), figsize=(3.4 * len(runs), 7), squeeze=False)
    out: dict[str, Any] = {}
    inn = site.box.inner
    for k, run in enumerate(runs):
        st = site.runs[run]
        ok = ~st["censored"]
        sc = _map(
            site,
            axs[0, k],
            st["length"],
            f"{run}: L (mm), grey censored",
            0,
            6,
            mask=ok,
            wrong=_wrong(site, run),
        )
        dl = st["length"] - ref
        ok2 = ok & np.isfinite(ref)
        sc2 = _map(
            site,
            axs[1, k],
            dl,
            f"{run}: L - straight first label 0 (mm)",
            -1,
            1,
            "coolwarm",
            mask=ok2,
            wrong=_wrong(site, run),
        )
        use = inn & ok2
        dL0 = st["length"] - st0["length"]
        use0 = inn & ok & ~st0["censored"]
        out[run] = dict(
            L_median=float(np.median(st["length"][inn & ok])),
            dL_vs_straight_first_label0=dict(
                median=float(np.median(dl[use])),
                p10=float(np.percentile(dl[use], 10)),
                p90=float(np.percentile(dl[use], 90)),
            ),
            dL_vs_straight_L=dict(
                median=float(np.median(dL0[use0])),
                p10=float(np.percentile(dL0[use0], 10)),
                p90=float(np.percentile(dL0[use0], 90)),
            ),
        )
    fig.colorbar(sc, ax=axs[0], shrink=0.7)
    fig.colorbar(sc2, ax=axs[1], shrink=0.7)
    fig.suptitle(
        "lengths L_c on the mesh (top) and their difference from the straight "
        "ray's first label-0 sample (bottom); x: passes a wrong-label piece",
        fontsize=8,
    )
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return out


def _grad_outside(site: Site) -> F64:
    """(X, Y, Z, 3) the gradient of the soft outside map, scanner frame."""
    O_ = _np(site.inp.volumes[0])
    gr: list[F64] = list(np.gradient(O_))
    g = np.stack(gr, -1)  # per voxel step
    lin = _np(site.inp.lin)  # mm -> voxel
    out: F64 = g @ lin  # d/dmm = d/dvox . dvox/dmm
    return out


def view_tops(site: Site, path: Path) -> dict[str, Any]:
    plt = _plt()
    b = site.box
    inn = b.inner
    G = _grad_outside(site)
    runs = list(site.runs)
    nrm = b.normals.astype(np.float64)
    fund = inn & (site.part == 0)
    shares: dict[str, dict[str, float]] = {}
    crowd: dict[str, Any] = {}
    f = b.faces
    a0 = _np(site.inp.a0)
    nu = _np(site.inp.nu)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for run in runs:
        st = site.runs[run]
        cen = st["censored"]
        tp = st["top_mm"]
        g = np.full((len(cen), 3), np.nan)
        ok = ~cen
        vox = b.to_box(tp[ok])
        gg = np.stack(
            [
                map_coordinates(G[..., k], vox.T, order=1, mode="nearest")
                for k in range(3)
            ],
            1,
        )
        g[ok] = gg / np.maximum(np.linalg.norm(gg, axis=1, keepdims=True), 1e-12)
        cosn = (g * nrm).sum(1)
        tip = ok & (cosn >= np.cos(np.radians(TIP_DEG)))
        gu = g @ site.u_s
        bp = ok & ~tip & (gu > 0)
        bn = ok & ~tip & (gu <= 0)
        nf = int(fund.sum())
        shares[run] = dict(
            tip=float((tip & fund).sum() / nf),
            wall_bank_minus_u=float((bn & fund).sum() / nf),
            wall_bank_plus_u=float((bp & fund).sum() / nf),
            censored=float((cen & fund).sum() / nf),
        )
        allok = inn[f].all(1) & ok[f].all(1)
        P = tp[f]
        ar = (np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) * nu).sum(1) / 2
        rel = ar / a0
        crowd[run] = {}
        for q in (-1, 0, 1):
            fm = allok & (site.part[f] == q).all(1)
            crowd[run][PART_NAMES[q]] = dict(
                faces=int(fm.sum()),
                area_ratio_sum=float(ar[fm].sum() / a0[fm].sum()),
                rel_median=float(np.median(rel[fm])),
                rel_p10=float(np.percentile(rel[fm], 10)),
                rel_p90=float(np.percentile(rel[fm], 90)),
                negative=int((rel[fm] < 0).sum()),
            )
            if q != 0:
                axs[1].hist(
                    np.clip(rel[fm], -2, 6),
                    bins=80,
                    histtype="step",
                    label=f"{run} {PART_NAMES[q]}",
                    lw=0.8,
                )
    keys = ("tip", "wall_bank_minus_u", "wall_bank_plus_u", "censored")
    x = np.arange(len(runs))
    bot = np.zeros(len(runs))
    for k in keys:
        v = np.array([shares[r][k] for r in runs])
        axs[0].bar(x, v, bottom=bot, label=k)
        bot += v
    axs[0].set_xticks(x, runs)
    axs[0].set_ylabel("share of the inner fundus columns")
    axs[0].legend(fontsize=7)
    axs[0].set_title(
        f"where the fundus columns' tops land ({int(fund.sum())} columns)", fontsize=9
    )
    axs[1].set_xlabel("top triangle area / root triangle area (clipped to -2..6)")
    axs[1].set_title("crowding on the banks", fontsize=9)
    axs[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return dict(shares=shares, crowding=crowd)


def view_alignment(site: Site, path: Path) -> dict[str, Any]:
    plt = _plt()
    b = site.box
    inn = b.inner
    v = BT._to_scanner(
        np.where(np.isfinite(site.lead), site.lead, 0.0),
        b.affine,
        b.voxel_mm,
    )
    edges = np.arange(0.0, BT.S_MAX + BIN_MM, BIN_MM)
    kb = np.digitize(np.arange(BT.NS) * BT.STEP, edges) - 1
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    out: dict[str, Any] = {}
    for run in site.runs:
        p = site.paths(run)
        with torch.no_grad():
            m = _np(BT.soft_top(p, site.inp).mask)
        vox = np.rint(_np(p.vox)).astype(np.int64)
        sh = np.asarray(b.labels.shape)
        okv = np.all((vox >= 0) & (vox < sh), -1)
        vc = np.clip(vox, 0, sh - 1)
        idx = (vc[..., 0], vc[..., 1], vc[..., 2])
        cc = np.where(okv, site.c[idx], np.nan)
        lab = np.where(okv, b.labels[idx], -1)
        vv = v[idx]
        t = _np(p.tangent)
        ang = np.degrees(np.arccos(np.clip(np.abs((t * vv).sum(-1)), 0, 1)))
        use = (m >= 0.5) & (lab == b.cortex) & np.isfinite(cc) & (cc >= C_HIGH)
        out[run] = {}
        for j, q in enumerate((-1, 0, 1)):
            sel = (inn & (site.part == q))[:, None] & use
            a_all = ang[sel]
            med = []
            for i in range(len(edges) - 1):
                s_ = sel[:, kb == i]
                aa = ang[:, kb == i][s_]
                med.append(float(np.median(aa)) if aa.size >= 50 else np.nan)
            mid = (edges[:-1] + edges[1:]) / 2
            axs[j].plot(mid, med, "-o", ms=2, label=run)
            out[run][PART_NAMES[q]] = dict(
                samples=int(a_all.size),
                median=float(np.median(a_all)) if a_all.size else None,
                p90=float(np.percentile(a_all, 90)) if a_all.size else None,
            )
            axs[j].set_title(PART_NAMES[q], fontsize=9)
            axs[j].set_xlabel("s (mm)")
    axs[0].set_ylabel("median angle path tangent to data normal (deg)")
    axs[0].legend(fontsize=7)
    fig.suptitle(
        f"angle to the data normal where c >= {C_HIGH} (cortex, below the soft "
        "top); with cost 7 in the fit it is not independent: A and B are the "
        "independent check",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return out


def view_censored(site: Site, path: Path) -> dict[str, Any]:
    plt = _plt()
    runs = list(site.runs)
    inn = site.box.inner
    nr = len(runs)
    fig = plt.figure(figsize=(3.3 * nr, 7.4))
    gs = fig.add_gridspec(2, nr)
    maps = [fig.add_subplot(gs[0, k]) for k in range(nr)]
    hax = fig.add_subplot(gs[1, : max(2, nr // 2)])
    out: dict[str, Any] = {}
    for k, run in enumerate(runs):
        st = site.runs[run]
        cen = st["censored"]
        sc = _map(
            site,
            maps[k],
            st["closest"],
            f"{run}: closest outside of censored (mm)",
            0,
            1.0,
            "viridis_r",
            mask=cen,
        )
        cl = st["closest"][cen & inn]
        hax.hist(cl, bins=50, range=(0, 2), histtype="step", label=run)
        h = st["hard_top"]
        out[run] = dict(
            censored_inner=int((cen & inn).sum()),
            closest_median=float(np.median(cl)) if cl.size else None,
            closest_lt_0p16=int((cl < 0.16).sum()),
            closest_lt_0p32=int((cl < 0.32).sum()),
            censored_through_label0=int((cen & inn & np.isfinite(h)).sum()),
            by_part={
                PART_NAMES[q]: int((cen & inn & (site.part == q)).sum())
                for q in (-1, 0, 1)
            },
        )
    fig.colorbar(sc, ax=maps, shrink=0.7)
    hax.set_xlabel("closest outside (mm), censored inner columns")
    hax.legend(fontsize=7)
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return out


def wrong_deep(site: Site, run: str) -> Bool:
    """(n,) the path passes a wrong-label voxel (nearest voxel) more than
    DEEP_MM from the nearest label-0 voxel centre, below its soft top (all
    samples if censored): the pieces less their rim along the outside."""
    from scipy.ndimage import distance_transform_edt

    b = site.box
    if site.wrong is None:
        return np.zeros(len(b.inner), bool)
    if site.deep is None:
        d = distance_transform_edt(b.labels != 0, sampling=b.voxel_mm)
        site.deep = site.wrong & (d > DEEP_MM)
    deep = site.deep
    p = site.paths(run)
    with torch.no_grad():
        m = _np(BT.soft_top(p, site.inp).mask)
    lab = BC.nearest_labels(deep.astype(np.int32), _np(p.vox)) == 1
    cen = np.asarray(site.runs[run]["censored"], bool)
    out: Bool = (lab & ((m >= 0.5) | cen[:, None])).any(1)
    return out


def view_returns(site: Site, path: Path) -> dict[str, Any]:
    plt = _plt()
    runs = list(site.runs)
    inn = site.box.inner
    xy = site.map_xy
    fig, axs = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 4.2), squeeze=False)
    out: dict[str, Any] = {}
    for k, run in enumerate(runs):
        st = site.runs[run]
        a = axs[0, k]
        a.scatter(xy[inn, 0], xy[inn, 1], s=0.3, c="0.85", rasterized=True)
        cats = (
            ("censored", st["censored"], "0.5"),
            ("early outside", st["early_outside"], "tab:blue"),
            ("returns to WM", st["returns"], "tab:red"),
        )
        for nm, m, cc in cats:
            mm = np.asarray(m, bool) & inn
            a.scatter(
                xy[mm, 0],
                xy[mm, 1],
                s=0.5,
                c=cc,
                label=f"{nm} {int(mm.sum())}",
                rasterized=True,
            )
        w = _wrong(site, run)
        if w is not None:
            ww = w & inn
            a.scatter(
                xy[ww, 0],
                xy[ww, 1],
                s=0.6,
                c="k",
                marker="x",
                lw=0.3,
                label=f"wrong label {int(ww.sum())}",
                rasterized=True,
            )
        a.set_aspect("equal")
        a.set_title(run, fontsize=8)
        lg = a.legend(fontsize=5, loc="lower left", markerscale=6)
        del lg
        a.tick_params(labelsize=6)
        ret = np.asarray(st["returns"], bool)
        out[run] = dict(
            returns_inner=int((ret & inn).sum()),
            returns_by_part={
                PART_NAMES[q]: int((ret & inn & (site.part == q)).sum())
                for q in (-1, 0, 1)
            },
            return_arc_median=float(np.nanmedian(st["return_arc"][ret & inn]))
            if (ret & inn).any()
            else None,
            early_outside_inner=int(
                (np.asarray(st["early_outside"], bool) & inn).sum()
            ),
            early_outside_not_censored_inner=int(
                (np.asarray(st["early_outside"], bool) & inn & ~st["censored"]).sum()
            ),
            wrong_label_inner=int((w & inn).sum()) if w is not None else None,
            wrong_label_own_path_inner=int((np.asarray(st["wrong"], bool) & inn).sum())
            if "wrong" in st
            else None,
            wrong_label_deep_own_path_inner=int((wrong_deep(site, run) & inn).sum())
            if site.wrong is not None
            else None,
        )
    fig.suptitle(
        "returns to WM (nearest label filled after a non-filled one, below the "
        "top), early outside (a label-0 sample at least 0.16 mm before the soft "
        "top, or on a censored path), wrong-label pieces (straight ray or path)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return out


def view_costs(site: Site, path: Path) -> dict[str, Any]:
    recs: dict[str, list[dict[str, Any]]] = {}
    out: dict[str, Any] = {}
    for run in site.arms():
        cj = json.loads((site.out / f"{run}_costs.json").read_text())
        recs[run] = cj["history"]
        out[run] = dict(
            stop=cj["stop"],
            seconds=cj["seconds"],
            evaluations=cj["evaluations"],
            seconds_per_evaluation=cj["seconds_per_evaluation"],
            rounds=[
                dict(
                    round=h["round"],
                    before=h["before"]["total"],
                    after=h["after"]["total"],
                    move_mm=h["move_mm"],
                    accepted=h["accepted"],
                    undone=h["descent_undone"],
                    censored=h["censored"],
                    evaluations=h["evaluations"],
                    seconds=h["seconds"],
                )
                for h in cj["history"][1:]
            ],
        )
    if recs:
        BF.view_costs(path, "site", recs)
    return out


def view_sparse(site: Site, path: Path, path_map: Path) -> dict[str, Any] | None:
    q = site.out / "sparse.json"
    if not q.exists():
        return None
    plt = _plt()
    sp = json.loads(q.read_text())
    runs = list(sp["runs"])
    terms = [k for k in (*BT.TERMS, "total") if k in sp["runs"][runs[0]]["dense"]]
    fig, axs = plt.subplots(
        1, len(runs), figsize=(3.6 * len(runs), 4.2), sharey=True, squeeze=False
    )
    for a, run in zip(axs[0], runs, strict=True):
        r = sp["runs"][run]
        y = np.arange(len(terms))
        for off, key, cc in (
            (-0.2, "sparse1", "tab:blue"),
            (0.0, "sparse1_full", "tab:green"),
            (0.2, "sparse2", "tab:orange"),
        ):
            mu, sd = [], []
            for k in terms:
                d = r["dense"][k]
                s = r[key]["terms"][k]
                den = abs(d) if abs(d) > 1e-12 else 1.0
                mu.append((s["mean"] - d) / den)
                sd.append(s["sd"] / den)
            a.errorbar(
                np.clip(mu, -1, 1), y + off, xerr=sd, fmt="o", ms=3, c=cc, label=key
            )
        a.axvline(0, c="k", lw=0.5)
        a.set_yticks(np.arange(len(terms)), terms, fontsize=7)
        a.set_title(run, fontsize=8)
        a.set_xlabel(
            "(sparse mean - dense) / |dense|, +- SD (clipped to +-1)", fontsize=6
        )
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)
    col = site.out / "sparse_columns.npz"
    if col.exists():
        z = np.load(col)
        fig, axs = plt.subplots(
            2, len(runs), figsize=(3.4 * len(runs), 7), squeeze=False
        )
        for ki, run in enumerate(runs):
            for row, key in enumerate(("sparse1", "sparse2")):
                dm = np.nan_to_num(z[f"{run}_{key}_top_diff_max"], nan=0.0)
                sc = _map(
                    site,
                    axs[row, ki],
                    dm,
                    f"{run} {key}: largest |top diff| (mm)",
                    0,
                    0.32,
                    "magma",
                )
                xy = site.map_xy
                for nm, cc, mk in (
                    ("censored_differs", "c", "s"),
                    ("return_differs", "lime", "x"),
                ):
                    m = z[f"{run}_{key}_{nm}"] & site.box.inner
                    axs[row, ki].scatter(
                        xy[m, 0],
                        xy[m, 1],
                        s=1.5,
                        c=cc,
                        marker=mk,
                        lw=0.4,
                        rasterized=True,
                    )
        fig.colorbar(sc, ax=axs, shrink=0.6)
        fig.suptitle(
            "dense against sparse: the largest top difference over the seeds; "
            "cyan squares: censored differs; green x: return flag differs",
            fontsize=8,
        )
        fig.savefig(path_map, dpi=75)
        plt.close(fig)
    out: dict[str, Any] = sp
    return out


def view_scan(site: Site, path: Path) -> None:
    q = site.out / "scan.md"
    if not q.exists():
        return
    rows = [r for r in q.read_text().splitlines() if r and not r.startswith("|---")]
    plt = _plt()
    fig = plt.figure(figsize=(17, 0.2 * len(rows) + 0.6))
    fig.text(0.005, 0.99, "\n".join(rows), family="monospace", fontsize=6, va="top")
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--texture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    figs = a.figures or (a.out / "figures")
    figs.mkdir(parents=True, exist_ok=True)
    site = Site(a.box, a.texture, a.out)
    q = a.out / "views.json"
    res: dict[str, Any] = json.loads(q.read_text()) if q.exists() else {}
    views: Mapping[str, Any] = {
        "cuts": lambda: {
            f"{d[0]}/{d[1]}": view_cuts(site, figs / f"cuts_drop{i + 1}.png", d)
            for i, d in enumerate(DROPS)
        },
        "sawtooth": lambda: view_sawtooth(site, figs / "sawtooth.png"),
        "profiles": lambda: view_profiles(site, figs / "section_profiles.png"),
        "bend_map": lambda: view_bend_map(site, figs / "bend_map.png"),
        "bend_depth": lambda: view_bend_depth(site, figs / "bend_depth.png"),
        "lengths": lambda: view_lengths(site, figs / "lengths_map.png"),
        "tops": lambda: view_tops(site, figs / "tops.png"),
        "alignment": lambda: view_alignment(site, figs / "align_angle.png"),
        "censored": lambda: view_censored(site, figs / "censored_map.png"),
        "returns": lambda: view_returns(site, figs / "returns_map.png"),
        "costs": lambda: view_costs(site, figs / "costs.png"),
        "sparse": lambda: view_sparse(
            site, figs / "sparse.png", figs / "sparse_map.png"
        ),
        "scan": lambda: view_scan(site, figs / "scan.png"),
    }
    for k, fn in views.items():
        if a.only and k not in a.only:
            continue
        BF.say(f"view {k}")
        r = fn()
        if r is not None and k != "sparse":
            res[k] = r
    q.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
