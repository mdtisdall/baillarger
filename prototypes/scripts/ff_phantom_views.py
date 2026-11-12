"""Views of the flow-field phantoms and their truth (docs/plans/flow-field-phantoms.md, PR 1, task 1.4).

One figure for each phantom: a section through the detrended echo 2 image and a section through
the labels. Both show the true stop surface and up to 30 true columns. One more figure,
`truth_summary.png`, counts the true roots by stop kind and shows the true jumps of the abraded
phantom.

The sections are axis-aligned planes of the phantom's local frame (the planes are in `PLANES`).
They are sampled in mm, with trilinear interpolation for the image and nearest voxel for the
labels. The true stop surface is the zero level of `geometry.stop_sdf`, evaluated in closed form.
Near the surface it is evaluated on a finer grid, so that a slot thinner than a voxel still shows
as a line. The columns drawn are those whose whole path stays within half a voxel of the plane,
spread evenly over the root order. Their stops are marked with a dot.

Usage:
    python ff_phantom_views.py OUT_DIR --features FEATURES_JSON [--site V1V2_banks]
        [--truth TRUTH_DIR] [--noise 1.0] [--names NAME ...]
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch as LegendPatch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import ff_truth as FT  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

H = T.VOXEL_MM
STEP_MM = 0.05  # the sampling step of the sections
FINE = 8  # the sub-samples across one cell, for the stop surface
BAND_MM = 0.08  # a cell is near the stop surface if its sdf is above minus this
MAX_COLUMNS = 30
NEAR_PLANE_MM = 0.5 * H
DPI = 110
MAX_BYTES = 1_000_000
PANEL_H_IN = 4.0
MAX_FIG_W_IN = 15.0

LABEL_COLOURS = {T.CORTEX: "#bdbdbd", T.MEDIUM: "#2b3a55", T.WM: "#f4f4f4"}
LABEL_NAMES = {T.CORTEX: "cortex", T.MEDIUM: "outside medium", T.WM: "WM"}
KIND_COLOURS = {T.StopKind.PIA: "#009E73", T.StopKind.CONTACT: "#E69F00",
                T.StopKind.HAIRPIN: "#CC79A7", T.StopKind.ABRADED: "#D55E00"}
SURFACE_COLOUR = "#00c8ff"
AXIS_NAMES = "xyz"


@dataclass(frozen=True)
class Plane:
    """An axis-aligned plane of the local frame: axes h (horizontal) and v (vertical) are indices
    0, 1, 2 of (x, y, z); `fixed` is the third axis, held at `value` (mm; None: the box centre)."""

    h: int
    v: int
    fixed: int
    value: float | None
    note: str
    alpha: float | None = None  # radians: the plane through the z axis at this azimuth (crowns)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(horizontal axis, vertical axis, normal): unit vectors of the local frame."""
        if self.alpha is not None:
            a = self.alpha
            return (np.array([np.cos(a), np.sin(a), 0.0]), np.array([0.0, 0.0, 1.0]),
                    np.array([-np.sin(a), np.cos(a), 0.0]))
        e = np.eye(3)
        return e[self.h], e[self.v], e[self.fixed]

    def label(self, which: str) -> str:
        if self.alpha is not None:
            return {"h": f"radial axis at azimuth {np.degrees(self.alpha):.1f} deg (mm)", "v": "local z (mm)"}[which]
        return f"local {AXIS_NAMES[self.h if which == 'h' else self.v]} (mm)"


def planes_of(name: str) -> list[Plane]:
    if name == "slab":
        return [Plane(1, 0, 2, None, "plane x-u (u = local y), through the box centre")]
    if name == "tau":
        return [Plane(1, 0, 2, None, "plane x-u (u = local y), through the box centre")]
    if name.startswith("crown"):
        return [Plane(0, 2, 1, 0.0, "plane y = 0 (through the axis; local x-z)")]
    if name.startswith(("fundus", "curved", "abraded")):
        return [Plane(0, 1, 2, 0.0, "plane z = 0 (local x-y)")]
    if name.startswith("ckiss"):
        return [Plane(0, 1, 2, -4.0, "z = -4 mm (open slot)"),
                Plane(0, 1, 2, 4.0, "z = +4 mm (fused contact)"),
                Plane(2, 0, 1, 1.0, "y = 1.0 mm (along the slot; g(z))")]
    return [Plane(0, 1, 2, None, "plane x-y, through the box centre")]  # noise


# ---------------------------------------------------------------- sampling the sections


@dataclass
class Section:
    plane: Plane
    value: float
    hs: np.ndarray  # (nh,) mm
    vs: np.ndarray  # (nv,) mm
    inbox: np.ndarray  # (nv, nh) bool
    image: np.ndarray  # (nv, nh), nan outside the box
    labels: np.ndarray  # (nv, nh) int, -1 outside the box
    surface: np.ndarray  # (K, 2): (h, v) of the cells that hold the stop surface

    def local(self, hh: np.ndarray, vv: np.ndarray) -> np.ndarray:
        eh, ev, n = self.plane.basis()
        return hh.reshape(-1, 1) * eh + vv.reshape(-1, 1) * ev + self.value * n


def box_centre_local(geo: Any) -> np.ndarray:
    c = (np.array(geo.shape, float) - 1.0) / 2.0
    return np.asarray(geo.to_local(c[None, :])[0])


def in_box(geo: Any, X: np.ndarray, pad: float = 0.0) -> np.ndarray:
    hi = np.array(geo.shape, float) - 1.0 + pad
    return np.asarray(np.all((X >= -pad) & (X <= hi), axis=1))


def make_section(geo: Any, ph: T.Phantom, plane: Plane) -> Section:
    centre = box_centre_local(geo)
    eh, ev, n = plane.basis()
    value = float(centre @ n if plane.value is None else plane.value)
    shp = np.array(geo.shape, float) - 1.0
    corners = np.array([[i, j, k] for i in (0, shp[0]) for j in (0, shp[1]) for k in (0, shp[2])])
    C = geo.to_local(corners)
    h0, h1 = (C @ eh).min(), (C @ eh).max()
    v0, v1 = (C @ ev).min(), (C @ ev).max()
    hs = np.arange(h0, h1 + STEP_MM, STEP_MM)
    vs = np.arange(v0, v1 + STEP_MM, STEP_MM)
    HH, VV = np.meshgrid(hs, vs)
    sec = Section(plane, value, hs, vs, np.zeros(HH.shape, bool), np.full(HH.shape, np.nan),
                  np.full(HH.shape, -1), np.zeros((0, 2)))
    X = geo.to_vox(sec.local(HH, VV))
    inb = in_box(geo, X)
    if not inb.any():
        raise RuntimeError(f"{ph.name}: the plane {plane.note} does not meet the box")
    # crop to the part of the plane that is in the box
    ib = inb.reshape(HH.shape)
    rows, cols = np.flatnonzero(ib.any(1)), np.flatnonzero(ib.any(0))
    hs, vs = hs[cols[0]:cols[-1] + 1], vs[rows[0]:rows[-1] + 1]
    HH, VV = np.meshgrid(hs, vs)
    sec = Section(plane, value, hs, vs, np.zeros(HH.shape, bool), np.full(HH.shape, np.nan),
                  np.full(HH.shape, -1), np.zeros((0, 2)))
    Y = sec.local(HH, VV)
    X = geo.to_vox(Y)
    inb = in_box(geo, X)
    sec.inbox = inb.reshape(HH.shape)
    img = np.full(inb.shape, np.nan)
    img[inb] = T.sample_volume(np.asarray(ph.det[1], float), X[inb])
    sec.image = img.reshape(HH.shape)
    lab = np.full(inb.shape, -1)
    lab[inb] = np.rint(ndi.map_coordinates(ph.labels.astype(float), X[inb].T, order=0,
                                            mode="nearest")).astype(int)
    sec.labels = lab.reshape(HH.shape)
    if hasattr(geo, "stop_sdf"):
        sec.surface = stop_surface(geo, sec, Y, inb)
    return sec


def stop_surface(geo: Any, sec: Section, Y: np.ndarray, inb: np.ndarray) -> np.ndarray:
    """The (h, v) of every cell of the section that holds a point of the stop surface: the cell
    has points with stop_sdf >= 0 and points with stop_sdf < 0 (on a FINE x FINE sub-grid)."""
    shape = sec.inbox.shape
    sdf = np.full(inb.shape, np.nan)
    sdf[inb] = geo.stop_sdf(Y[inb])
    S = sdf.reshape(shape)
    neg = np.nan_to_num(S, nan=1.0) < 0
    pos = np.nan_to_num(S, nan=-1.0) >= 0
    near_neg = np.zeros(shape, bool)
    near_neg[1:] |= neg[:-1]
    near_neg[:-1] |= neg[1:]
    near_neg[:, 1:] |= neg[:, :-1]
    near_neg[:, :-1] |= neg[:, 1:]
    cand = (neg & (S > -BAND_MM)) | (pos & near_neg)
    iv, ih = np.nonzero(cand & sec.inbox)
    if iv.size == 0:
        return np.zeros((0, 2))
    off = ((np.arange(FINE) + 0.5) / FINE - 0.5) * STEP_MM
    oh, ov = np.meshgrid(off, off)
    oh, ov = oh.ravel(), ov.ravel()
    hh = sec.hs[ih][:, None] + oh[None, :]
    vv = sec.vs[iv][:, None] + ov[None, :]
    sd = geo.stop_sdf(sec.local(hh, vv)).reshape(hh.shape)
    mixed = (sd >= 0).any(1) & (sd < 0).any(1)
    return np.stack([sec.hs[ih][mixed], sec.vs[iv][mixed]], 1)


# ---------------------------------------------------------------- the columns


def snap_value(geo: Any, tr: T.Truth, plane: Plane, nominal: float) -> float:
    """The root-lattice value of the plane's fixed coordinate that is nearest to `nominal`, if it
    is within half a root spacing. The roots lie on a lattice that no plane of the frame is
    chosen to meet, so a plane at the nominal value can miss every root."""
    vals = np.unique(np.round(geo.to_local(tr.roots) @ plane.basis()[2], 6))
    j = int(np.argmin(np.abs(vals - nominal)))
    return float(vals[j]) if abs(vals[j] - nominal) <= 0.5 * tr.root_spacing_mm else nominal


def crown_plane(geo: Any, tr: T.Truth) -> Plane:
    """The plane through the crown's axis (local z) at the azimuth that holds most columns. A
    crown's columns are straight and radial, so each lies in a plane through the axis. The azimuth
    is chosen over steps of 0.25 degree, to maximise the count of columns within NEAR_PLANE_MM of
    the plane, with each column's largest horizontal radius."""
    rho = np.array([np.linalg.norm(geo.to_local(tr.path(i))[:, :2], axis=1).max() for i in range(tr.n)])
    th = np.array([np.arctan2(*geo.to_local(tr.roots[i][None, :])[0, 1::-1]) for i in range(tr.n)])
    best, best_a = -1, 0.0
    for a in np.radians(np.arange(0.0, 180.0, 0.25)):
        cnt = int((rho * np.abs(np.sin(th - a)) <= NEAR_PLANE_MM).sum())
        if cnt > best:
            best, best_a = cnt, float(a)
    return Plane(-1, -1, -1, 0.0, "plane through the axis (azimuth chosen for the most columns)", alpha=best_a)


def near_columns(geo: Any, tr: T.Truth, sec: Section) -> tuple[np.ndarray, int, float]:
    """The indices of the columns to draw (up to MAX_COLUMNS, spread over the root order), the
    count of columns whose whole path is within the tolerance of the plane, and the tolerance
    (mm), always half a voxel. If fewer than MAX_COLUMNS columns qualify, all of them are shown."""
    n = sec.plane.basis()[2]
    dev = np.array([np.abs(geo.to_local(tr.path(i)) @ n - sec.value).max() for i in range(tr.n)])
    tol = NEAR_PLANE_MM
    ok_a = np.flatnonzero(dev <= tol)
    if ok_a.size > MAX_COLUMNS:
        pick = np.unique(np.rint(np.linspace(0, ok_a.size - 1, MAX_COLUMNS)).astype(int))
        return ok_a[pick], int(ok_a.size), tol
    return ok_a, int(ok_a.size), tol


# ---------------------------------------------------------------- drawing


def extent_of(sec: Section) -> tuple[float, float, float, float]:
    s = 0.5 * STEP_MM
    return (sec.hs[0] - s, sec.hs[-1] + s, sec.vs[0] - s, sec.vs[-1] + s)


def label_rgb(lab: np.ndarray) -> np.ndarray:
    out = np.ones(lab.shape + (3,))
    for k, c in LABEL_COLOURS.items():
        out[lab == k] = to_rgb(c)
    return out


def draw_overlays(ax: Any, geo: Any, tr: T.Truth | None, sec: Section, cols: np.ndarray, name: str) -> None:
    p = sec.plane
    eh, ev, _ = p.basis()
    if sec.surface.size:
        ax.scatter(sec.surface[:, 0], sec.surface[:, 1], s=1.6, marker="s", c=SURFACE_COLOUR,
                   linewidths=0, zorder=3)
    if tr is not None:
        for i in cols:
            P = geo.to_local(tr.path(i))
            kind = T.StopKind(int(tr.stop_kind[i]))
            ax.plot(P @ eh, P @ ev, "-", c=KIND_COLOURS[kind], lw=0.8, alpha=0.95, zorder=4)
            ax.plot(P[-1:] @ eh, P[-1:] @ ev, "o", ms=2.2, mfc="k", mec="w", mew=0.4, zorder=5)
    if name == "tau" and p.alpha is None and p.h == 1:  # the border at u = 0 (local y)
        ax.axvline(0.0, c="#e03030", ls="--", lw=1.0, zorder=3)
        ax.text(0.0, sec.vs[-1], " u = 0", color="#e03030", fontsize=7, va="top", ha="left", zorder=6)


def style_axes(ax: Any, sec: Section, title: str) -> None:
    ext = extent_of(sec)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xlabel(sec.plane.label("h"), fontsize=8)
    ax.set_ylabel(sec.plane.label("v"), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8)


def legend_handles(has_truth: bool, kinds: set[int]) -> list[Any]:
    hs: list[Any] = [LegendPatch(fc=LABEL_COLOURS[k], ec="0.5", label=LABEL_NAMES[k])
                     for k in (T.CORTEX, T.MEDIUM, T.WM)]
    hs.append(Line2D([], [], c=SURFACE_COLOUR, marker="s", ls="", ms=4, label="true stop surface"))
    if has_truth:
        for k in T.StopKind:
            if int(k) in kinds:
                hs.append(Line2D([], [], c=KIND_COLOURS[k], lw=1.5, label=f"column, stop {k.name}"))
        hs.append(Line2D([], [], marker="o", ls="", mfc="k", mec="w", ms=4, label="stop"))
    return hs


def figure_for(ph: T.Phantom, tr: T.Truth | None, out: Path) -> dict[str, Any]:
    name = ph.name
    geo = ph.geometry if ph.geometry is not None else ph.image["fold"]
    planes = planes_of(name)
    if tr is not None:
        centre = box_centre_local(geo)
        if name.startswith("crown"):
            planes = [crown_plane(geo, tr)]
        else:
            planes = [Plane(p.h, p.v, p.fixed,
                            snap_value(geo, tr, p, float(centre @ p.basis()[2] if p.value is None else p.value)),
                            p.note) for p in planes]
    secs = [make_section(geo, ph, pl) for pl in planes]
    picks = [near_columns(geo, tr, s) if tr is not None else (np.zeros(0, int), 0, 0.0) for s in secs]
    nrow = 2
    ncol = len(secs)
    ratios = [(extent_of(s)[1] - extent_of(s)[0]) / (extent_of(s)[3] - extent_of(s)[2]) for s in secs]
    widths = [PANEL_H_IN * r for r in ratios]
    scale = min(1.0, (MAX_FIG_W_IN - 1.5) / sum(widths))
    fig_w = max(sum(widths) * scale + 1.5, 8.0)
    fig_h = 2 * PANEL_H_IN * scale + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrow, ncol, width_ratios=ratios, left=0.07, right=0.97, top=0.88, bottom=0.14,
                          hspace=0.35, wspace=0.28)
    info: dict[str, Any] = {}
    for c, (sec, cols_nn) in enumerate(zip(secs, picks)):
        cols, nnear, tol = cols_nn
        for r in range(nrow):
            ax = fig.add_subplot(gs[r, c])
            ext = extent_of(sec)
            if r == 0:
                v = sec.image[sec.inbox]
                lo, hi = np.percentile(v, [1, 99])
                cm = plt.get_cmap("gray").copy()
                cm.set_bad("white")
                ax.imshow(np.ma.masked_invalid(sec.image), origin="lower", extent=ext, cmap=cm,
                          vmin=lo, vmax=hi, interpolation="nearest", zorder=1)
                what = "detrended echo 2"
            else:
                ax.imshow(label_rgb(sec.labels), origin="lower", extent=ext, interpolation="nearest", zorder=1)
                what = "labels"
            draw_overlays(ax, geo, tr, sec, cols, name)
            if tr is None:
                cnote = ""
            else:
                cnote = f"\n{len(cols)} true columns shown, within {tol / H:g} voxel of the plane"
                if nnear < MAX_COLUMNS:
                    cnote += f" (only {nnear} lie near it)"
            fx = f"{AXIS_NAMES[sec.plane.fixed]} = {sec.value:.2f} mm" if sec.plane.alpha is None else ""
            style_axes(ax, sec, f"{what}: {sec.plane.note} {('[' + fx + ']') if fx else ''}{cnote}")
        info[sec.plane.note] = dict(value=round(sec.value, 3), near=picks[c][1], shown=int(len(picks[c][0])), tol_vox=round(picks[c][2] / H, 2))
    kinds = {int(tr.stop_kind[i]) for cols, _, _ in picks for i in cols} if tr is not None else set()
    fig.legend(handles=legend_handles(tr is not None, kinds), loc="lower center", ncol=3, fontsize=7,
               frameon=False)
    tnote = "no truth (no structure)" if tr is None else f"{tr.n} true columns"
    fig.suptitle(f"{name}   noise {ph.noise:g}   {tnote}   box {tuple(geo.shape)}", fontsize=10)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    size = path.stat().st_size
    assert size < MAX_BYTES, f"{path} is {size} bytes"
    info["bytes"] = size
    return info


# ---------------------------------------------------------------- the summary


def summary_figure(truths: dict[str, T.Truth], out: Path) -> int:
    names = list(truths)
    has_abr = "abraded" in truths
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 2 if has_abr else 1, width_ratios=[3, 1.2] if has_abr else [1], wspace=0.25,
                          left=0.06, right=0.98, bottom=0.30, top=0.90)
    ax = fig.add_subplot(gs[0, 0])
    bottom = np.zeros(len(names))
    for k in T.StopKind:
        cnt = np.array([int((truths[n].stop_kind == k).sum()) for n in names], float)
        ax.bar(range(len(names)), cnt, bottom=bottom, color=KIND_COLOURS[k], label=k.name, width=0.7)
        bottom += cnt
    for i, tot in enumerate(bottom):
        ax.text(i, tot, f"{int(tot)}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("true roots (count)")
    ax.set_title("true roots by stop kind", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    if has_abr:
        ax2 = fig.add_subplot(gs[0, 1])
        tr = truths["abraded"]
        geo, ts = FT.geometry_of("abraded", tuple(int(s) for s in _shape_of(tr, "abraded")))
        sel = np.flatnonzero(tr.stop_kind == T.StopKind.ABRADED)
        jump = tr.jump[sel]
        Ys = np.array([geo.to_local(tr.stops[i][None, :])[0] for i in sel])
        Yb = Ys + FT.BEYOND_MM * FT._unit_grad(geo, Ys)
        pid = geo.patch_of(Yb)
        bins = np.linspace(0.0, max(float(jump.max()) * 1.05, 0.8), 60)
        data = [jump[pid == k] for k in range(len(geo.patches))]
        ax2.hist(data, bins=bins, stacked=True, color=["#D55E00", "#7a2e00"][: len(data)],
                 label=[f"patch {k}: {len(d)} roots" for k, d in enumerate(data)])
        for k, pt in enumerate(geo.patches):
            ex = pt.depth / geo.p["tA"] * ts
            ax2.axvline(ex, c="k", ls=":", lw=1.0)
            ax2.text(ex, ax2.get_ylim()[1], f" depth {pt.depth:g} / tA x tau_scale", rotation=90,
                     fontsize=7, va="top", ha="left")
        ax2.set_xlabel("true jump (tau units)")
        ax2.set_ylabel("roots")
        ax2.set_title("abraded: true jump", fontsize=10)
        ax2.legend(fontsize=7, frameon=False, loc="center")
    fig.savefig(out / "truth_summary.png", dpi=DPI)
    plt.close(fig)
    size = (out / "truth_summary.png").stat().st_size
    assert size < MAX_BYTES, f"truth_summary.png is {size} bytes"
    return size


def _shape_of(tr: T.Truth, name: str) -> tuple[int, ...]:
    geo, _ = FT.geometry_of(name)
    return tuple(geo.shape)


# ---------------------------------------------------------------- the CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--names", nargs="+", default=list(FT.PHANTOMS))
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    feats = FP.features(a.site, str(a.features))
    truths: dict[str, T.Truth] = {}
    t00 = time.perf_counter()
    for name in a.names:
        t0 = time.perf_counter()
        ph = FT.build_phantom(name, feats, a.site, a.noise, seed=0)
        tr: T.Truth | None = None
        if name != "noise":
            f = a.truth / name / "truth.npz" if a.truth else None
            if f is not None and f.exists():
                tr = T.Truth.from_npz(f)
            else:
                tr = FT.truth(ph.geometry, ph.tau_scale, name)
            truths[name] = tr
        info = figure_for(ph, tr, a.out_dir)
        print(f"{name}: {time.perf_counter() - t0:.1f} s  {info}", flush=True)
    size = summary_figure(truths, a.out_dir) if truths else 0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    print(f"truth_summary.png: {size} bytes; total {time.perf_counter() - t00:.1f} s; "
          f"peak {rss / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
