"""Views of the inputs of the flow-field phantoms stage (docs/plans/flow-field-phantoms.md, PR 2,
task 2.3), and the per-term numbers that the checkpoint report reads.

For each phantom of PHANTOMS and each noise level, the image stop field
(`ff_stop.ImageStopField`, with `ff_oracle.ImageTensor.of` and the noise-0 phantom as the site
reference) and the oracle stop field (`ff_oracle.OracleStopField`) are evaluated at points only:
on the sections of `ff_phantom_views` (its `planes_of`, `snap_value`, `crown_plane` and
`make_section`), on a zoomed section at the fused contact, and at sets of points listed below.
Nothing is evaluated on the whole box except what `ImageTensor.of` and `ImageStopField` compute
once per phantom.

Outputs:
- FIG_DIR/inputs_<name>.png: for each section of the phantom and each noise level (noise 1.0
  first, then the others), the image stop probability, the oracle stop probability (the clipped
  input: sigmoid of `OracleStopField.log_odds`) and their difference (image - oracle), with the
  true stop surface (stop_sdf = 0) on each panel. The noise phantom has no geometry: only the image
  field is shown.
- FIG_DIR/inputs_terms_ckiss_smooth.png and ..._flat.png: the four terms and their sum on a
  section z = FUSED_Z_MM across the fused contact, one diverging scale centred on 0.
- FIG_DIR/inputs_terms_summary.png: the median of each term and of the log-odds at the true stops
  of each kind, at the main noise level.
- OUT_DIR/inputs_terms.json and inputs_terms.md: for each phantom, noise level and region, the
  10th percentile, median and 90th percentile of each term and of the log-odds, the fraction with
  p > 0.5 and the median oracle p.
- OUT_DIR/run.json: time, peak memory, arguments and the image fields' records.

Regions (points in voxel space):
- stop, stop -0.02 mm, stop -0.08 mm for each StopKind: the true stop of a column and the points
  0.02 mm and 0.08 mm before it along the column (at most MAX_COLUMNS columns of each kind, evenly
  spread in the root order);
- medium_deep: the outside medium at stop_sdf > MEDIUM_DEEP_MM;
- cortex_deep: the cortex (label CORTEX) at stop_sdf < -CORTEX_DEEP_MM;
- random: the noise phantom only (no geometry): random points of the box.
Random points are drawn from the voxels of the label, with a jitter of up to half a voxel. Every
point must be at least EDGE_VOX voxels from the faces of the box (where the image tensor is
defined); the counts before and after are in the json (n_total, n). The tensor is built on the
cortex only, so it can be missing at a point next to the medium: `frac_no_tensor` is the fraction
of the points where `ImageTensor.at` gives coherence 0.

Usage:
    python ff_input_views.py OUT_DIR FIG_DIR --features FEATURES_JSON --truth TRUTH_DIR
        [--site V1V2_banks] [--noise 0 1.0] [--seed 0] [--names NAME ...]
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import ff_oracle as FO  # noqa: E402
import ff_phantom_views as PV  # noqa: E402
import ff_stop as ST  # noqa: E402
import ff_truth as FT  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

H = T.VOXEL_MM
TERMS = ST.TERM_NAMES  # label, sheet, mirror, partial_volume
COLUMNS = (*TERMS, "log_odds")
BEFORE_MM = (0.0, 0.02, 0.08)  # the points at and before the stop, along the column
MAX_COLUMNS = 3000  # columns of each kind
MAX_RANDOM = 3000  # random points of each region
MEDIUM_DEEP_MM = 0.16
CORTEX_DEEP_MM = 0.5
# a point is at least this far from the faces of the box: the reach of orient (the tensor is nan nearer)
EDGE_VOX = max(FO.O.orient_reach(np.full(3, H), FO.O.SIGMA))
MAIN_NOISE = 1.0
FUSED_Z_MM = 4.0  # the fused part of the ckiss phantoms (PR 1's plane z = +4 mm)
TERMS_PHANTOMS = ("ckiss_smooth", "ckiss_smooth_flat")
ZOOM_HALF_MM = 2.0  # half-width of the zoomed section
ZOOM_STEP_MM = 0.02
ZOOM_OFFSET_MM = 0.0031  # moves the grid off the slot's symmetry plane
TERM_SCALE = 3.0  # the colour scale of the terms is +-TERM_SCALE
DPI = 80
MAX_BYTES = 1_000_000
PCTS = (10.0, 50.0, 90.0)


# ---------------------------------------------------------------- points


def _pcts(v: np.ndarray) -> dict[str, float | None]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"p10": None, "median": None, "p90": None}
    a = np.percentile(v, PCTS)
    return {"p10": float(a[0]), "median": float(a[1]), "p90": float(a[2])}


def point_ok(X: np.ndarray, tensor: FO.ImageTensor) -> np.ndarray:
    """True at the points that are at least EDGE_VOX voxels from the faces of the box."""
    shp = np.array(tensor.valid.shape)
    return np.asarray(np.all((X >= EDGE_VOX) & (X <= shp - 1 - EDGE_VOX), axis=1))


def point_before_stop(tr: T.Truth, i: int, back_mm: float) -> np.ndarray:
    """The point of column i that is back_mm (arc length) before its stop; the root if the column
    is shorter."""
    P = tr.path(i)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1) * H
    s = np.concatenate([[0.0], np.cumsum(seg)])  # arc length from the root
    target = s[-1] - back_mm
    if target <= 0.0:
        return np.asarray(P[0])
    return np.array([np.interp(target, s, P[:, a]) for a in range(3)])


def region_points(
    name: str,
    ph: T.Phantom,
    tr: T.Truth | None,
    tensor: FO.ImageTensor,
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    """{region: {"X": points, "n_total": count before the edge filter, "kind": ...}}."""
    out: dict[str, dict[str, Any]] = {}
    geo = ph.geometry
    if geo is None or tr is None:
        idx = np.argwhere(np.ones(ph.labels.shape, bool)[::2, ::2, ::2]) * 2
        take = rng.choice(len(idx), size=min(len(idx), 6 * MAX_RANDOM), replace=False)
        X = idx[take] + rng.uniform(-0.5, 0.5, (len(take), 3))
        keep = point_ok(X, tensor)
        out["random"] = {"X": X[keep][:MAX_RANDOM], "n_total": len(X), "kind": None}
        return out
    for kind in T.StopKind:
        sel = np.flatnonzero(tr.stop_kind == kind)
        n_all = len(sel)
        if n_all > MAX_COLUMNS:
            sel = sel[
                np.unique(np.rint(np.linspace(0, n_all - 1, MAX_COLUMNS)).astype(int))
            ]
        for back in BEFORE_MM:
            key = f"stop_{kind.name}" + (f"_m{back:g}" if back else "")
            X = (
                np.array([point_before_stop(tr, int(i), back) for i in sel])
                if len(sel)
                else np.zeros((0, 3))
            )
            keep = point_ok(X, tensor) if len(X) else np.zeros(0, bool)
            out[key] = {
                "X": X[keep],
                "n_total": n_all,
                "n_drawn": len(sel),
                "kind": int(kind),
            }
    lab = ph.labels
    for key, want, test in (
        ("medium_deep", T.MEDIUM, lambda s: s > MEDIUM_DEEP_MM),
        ("cortex_deep", T.CORTEX, lambda s: s < -CORTEX_DEEP_MM),
    ):
        idx = np.argwhere(lab == want)
        if len(idx) == 0:
            out[key] = {"X": np.zeros((0, 3)), "n_total": 0, "kind": None}
            continue
        take = rng.choice(len(idx), size=min(len(idx), 8 * MAX_RANDOM), replace=False)
        X = idx[take] + rng.uniform(-0.5, 0.5, (len(take), 3))
        X = X[point_ok(X, tensor)]
        sdf = geo.stop_sdf(geo.to_local(X))
        X = X[test(sdf)]
        out[key] = {"X": X[:MAX_RANDOM], "n_total": len(idx), "kind": None}
    return out


# ---------------------------------------------------------------- numbers


def evaluate(
    field: ST.ImageStopField, oracle: FO.OracleStopField | None, X: np.ndarray
) -> dict[str, Any]:
    """The statistics of the terms and of the log-odds at the points X."""
    if len(X) == 0:
        return {"n": 0}
    t = field.terms(X)
    lo = np.full(len(X), ST.PRIOR, float)
    for k in TERMS:
        lo = lo + t[k]
    m = min(len(X), 50)
    if not np.allclose(lo[:m], field.log_odds(X[:m]), equal_nan=True):
        raise RuntimeError("the sum of the terms differs from ImageStopField.log_odds")
    p = T.sigmoid(lo)
    res: dict[str, Any] = {"n": int(len(X))}
    for k in TERMS:
        res[k] = _pcts(t[k])
    res["log_odds"] = _pcts(lo)
    fin = np.isfinite(p)
    res["frac_p_gt_half"] = float(np.mean(p[fin] > 0.5)) if fin.any() else None
    res["frac_no_tensor"] = float(np.mean(field.tensor.at(X)[1] == 0.0))
    res["n_nan_log_odds"] = int((~fin).sum())
    if oracle is not None:
        po = T.sigmoid(oracle.log_odds(X))
        res["oracle_p_median"] = float(np.median(po))
    return res


# ---------------------------------------------------------------- sections


def sections_for(ph: T.Phantom, tr: T.Truth | None) -> tuple[Any, list[PV.Section]]:
    """PR 1's sections of the phantom (the planes of figure_for in ff_phantom_views)."""
    geo = ph.geometry if ph.geometry is not None else ph.image["fold"]
    planes = PV.planes_of(ph.name)
    if tr is not None:
        centre = PV.box_centre_local(geo)
        if ph.name.startswith("crown"):
            planes = [PV.crown_plane(geo, tr)]
        else:
            planes = [
                PV.Plane(
                    p.h,
                    p.v,
                    p.fixed,
                    PV.snap_value(
                        geo,
                        tr,
                        p,
                        float(centre @ p.basis()[2] if p.value is None else p.value),
                    ),
                    p.note,
                )
                for p in planes
            ]
    return geo, [PV.make_section(geo, ph, pl) for pl in planes]


def section_fields(
    geo: Any,
    sec: PV.Section,
    field: ST.ImageStopField,
    oracle: FO.OracleStopField | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """(image p, oracle p) on the section, nan outside the box."""
    HH, VV = np.meshgrid(sec.hs, sec.vs)
    X = geo.to_vox(sec.local(HH.ravel(), VV.ravel()))
    inb = sec.inbox.ravel()
    pi = np.full(inb.shape, np.nan)
    pi[inb] = field.probability(X[inb])
    po = None
    if oracle is not None:
        po = np.full(inb.shape, np.nan)
        po[inb] = T.sigmoid(oracle.log_odds(X[inb]))
        po = po.reshape(sec.inbox.shape)
    return pi.reshape(sec.inbox.shape), po


def save(fig: Any, path: Path) -> int:
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    size = path.stat().st_size
    assert size < MAX_BYTES, f"{path} is {size} bytes"
    return size


def phantom_figure(
    name: str,
    geo: Any,
    secs: list[PV.Section],
    data: dict[float, list[tuple[np.ndarray, np.ndarray | None]]],
    has_oracle: bool,
    out: Path,
) -> int:
    """data[noise][plane index] = (image p, oracle p)."""
    noises = sorted(data, key=lambda n: (n != MAIN_NOISE, -n))  # the main level first
    ncol = 3 if has_oracle else 1
    col_w = 3.4 if has_oracle else 4.6
    ratios = []
    for s in secs:
        e = PV.extent_of(s)
        ratios.append((e[3] - e[2]) / (e[1] - e[0]))
    row_h = [col_w * r for r in ratios for _ in noises]
    fig = plt.figure(
        figsize=(ncol * col_w + 1.4, sum(row_h) + 0.6 * len(row_h) + 1.0),
        layout="constrained",
    )
    gs = fig.add_gridspec(len(row_h), ncol, height_ratios=[h + 0.6 for h in row_h])
    ims: list[list[Any]] = [[] for _ in range(ncol)]
    axs: list[list[Any]] = [[] for _ in range(ncol)]
    r = 0
    for si, sec in enumerate(secs):
        ext = PV.extent_of(sec)
        for noise in noises:
            pi, po = data[noise][si]
            panels: list[tuple[str, np.ndarray, Any]] = [
                ("image p", pi, (0.0, 1.0, "magma"))
            ]
            if po is not None:
                panels += [
                    ("oracle p", po, (0.0, 1.0, "magma")),
                    ("image - oracle", pi - po, (-1.0, 1.0, "RdBu_r")),
                ]
            for c, (what, arr, (lo, hi, cmap)) in enumerate(panels):
                ax = fig.add_subplot(gs[r, c])
                cm = plt.get_cmap(cmap).copy()
                cm.set_bad("white")
                im = ax.imshow(
                    np.ma.masked_invalid(arr),
                    origin="lower",
                    extent=ext,
                    cmap=cm,
                    vmin=lo,
                    vmax=hi,
                    interpolation="nearest",
                )
                PV.draw_overlays(ax, geo, None, sec, np.zeros(0, int), name)
                PV.style_axes(ax, sec, f"{what}, noise {noise:g}\n{sec.plane.note}")
                ims[c].append(im)
                axs[c].append(ax)
            r += 1
    for c in range(ncol):
        fig.colorbar(ims[c][0], ax=axs[c], shrink=0.5, pad=0.02, aspect=30)
    if has_oracle:
        title = f"{name}: stop probability of the image terms and of the oracle\ntrue stop surface in cyan"
    else:
        title = f"{name}: image stop probability only\nno geometry: no oracle field, no true stop surface"
    fig.suptitle(title, fontsize=9)
    return save(fig, out / f"inputs_{name}.png")


# ---------------------------------------------------------------- the terms at the fused contact


def zoom_grid(geo: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A section z = FUSED_Z_MM, ZOOM_HALF_MM round local x = 0 and round the box centre's local y.
    Returns (hs, vs, X voxel positions (nv * nh, 3), local points)."""
    cy = float(PV.box_centre_local(geo)[1])
    hs = np.arange(-ZOOM_HALF_MM, ZOOM_HALF_MM + 1e-9, ZOOM_STEP_MM) + ZOOM_OFFSET_MM
    vs = np.arange(cy - ZOOM_HALF_MM, cy + ZOOM_HALF_MM + 1e-9, ZOOM_STEP_MM)
    HH, VV = np.meshgrid(hs, vs)
    Y = np.stack([HH.ravel(), VV.ravel(), np.full(HH.size, FUSED_Z_MM)], 1)
    return hs, vs, geo.to_vox(Y), Y


def terms_figure(
    name: str,
    hs: np.ndarray,
    vs: np.ndarray,
    sdf: np.ndarray,
    gz: float,
    panels: dict[float, dict[str, np.ndarray]],
    out: Path,
) -> int:
    noises = sorted(panels, key=lambda n: (n != MAIN_NOISE, -n))
    cols = (*TERMS, "sum")
    fig, axes = plt.subplots(
        len(noises),
        len(cols),
        figsize=(2.6 * len(cols) + 1.0, 2.6 * len(noises) + 0.9),
        squeeze=False,
        layout="constrained",
    )
    ext = (
        hs[0] - ZOOM_STEP_MM / 2,
        hs[-1] + ZOOM_STEP_MM / 2,
        vs[0] - ZOOM_STEP_MM / 2,
        vs[-1] + ZOOM_STEP_MM / 2,
    )
    im = None
    for r, noise in enumerate(noises):
        for c, k in enumerate(cols):
            ax = axes[r, c]
            arr = panels[noise][k].reshape(len(vs), len(hs))
            im = ax.imshow(
                arr,
                origin="lower",
                extent=ext,
                cmap="RdBu_r",
                vmin=-TERM_SCALE,
                vmax=TERM_SCALE,
                interpolation="nearest",
            )
            ax.contour(
                hs, vs, sdf, levels=[0.0], colors=PV.SURFACE_COLOUR, linewidths=0.8
            )
            ax.set_aspect("equal")
            ax.set_title(f"{k.replace('_', ' ')}, noise {noise:g}", fontsize=8)
            ax.tick_params(labelsize=6)
            if r == len(noises) - 1:
                ax.set_xlabel("local x (mm)", fontsize=7)
            if c == 0:
                ax.set_ylabel("local y (mm)", fontsize=7)
    fig.colorbar(
        im, ax=axes, shrink=0.6, pad=0.02, label="log-odds (clipped at the scale)"
    )
    fig.suptitle(
        f"{name}: stop terms at the fused contact, section z = {FUSED_Z_MM:g} mm (g = {gz:.3f} mm); "
        f"true stop surface (stop_sdf = 0) in cyan; one scale +-{TERM_SCALE:g}",
        fontsize=8,
    )
    return save(fig, out / f"inputs_terms_{name}.png")


# ---------------------------------------------------------------- the summary figure


def summary_figure(rows: list[tuple[str, dict[str, Any]]], out: Path) -> int:
    n = len(rows)
    M = np.full((n, len(COLUMNS)), np.nan)
    for i, (_, rec) in enumerate(rows):
        for j, k in enumerate(COLUMNS):
            v = rec[k]["median"]
            M[i, j] = np.nan if v is None else v
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * n + 1.8), layout="constrained")
    im = ax.imshow(M, cmap="RdBu_r", vmin=-TERM_SCALE, vmax=TERM_SCALE, aspect="auto")
    for i in range(n):
        for j in range(len(COLUMNS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels([c.replace("_", " ") for c in COLUMNS], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, label="log-odds")
    fig.suptitle(
        f"median of each term at the true stops, noise {MAIN_NOISE:g} (phantom, stop kind, n)",
        fontsize=9,
    )
    return save(fig, out / "inputs_terms_summary.png")


# ---------------------------------------------------------------- the table


def _cell(d: dict[str, Any], k: str) -> str:
    v = d.get(k)
    if not v or v["median"] is None:
        return "-"
    return f"{v['median']:.2f} ({v['p10']:.2f}..{v['p90']:.2f})"


def table_md(records: list[dict[str, Any]], args: argparse.Namespace) -> str:
    L = [
        "# Inputs: the stop terms by region",
        "",
        f"Seed {args.seed}, site {args.site}. Cells: median (10th..90th percentile) of each term and of",
        "the log-odds, at the points of the region. `n` is the count of points used; `n_total` the",
        "count before the edge filter (for the stop regions, the count of columns of",
        "that kind). `p>0.5` is the fraction of the points with the image stop probability above 0.5;",
        "`oracle p` is the median oracle stop probability (clipped input) at the same points.",
        "`no tensor` is the fraction of the points where the image tensor gives coherence 0.",
        "Region `stop_K` is the true stop of the columns of kind K; `stop_K_m0.02` and `stop_K_m0.08`",
        "are the points 0.02 mm and 0.08 mm before it along the column.",
        "",
    ]
    heads = [
        "noise",
        "region",
        "n",
        "n_total",
        *[c.replace("_", " ") for c in COLUMNS],
        "p>0.5",
        "oracle p",
        "no tensor",
    ]
    names = list(dict.fromkeys(r["phantom"] for r in records))
    for nm in names:
        L += [
            f"## {nm}",
            "",
            "| " + " | ".join(heads) + " |",
            "|" + "---|" * len(heads),
        ]
        for r in records:
            if r["phantom"] != nm:
                continue
            s = r["stats"]
            if s["n"] == 0:
                L.append(
                    f"| {r['noise']:g} | {r['region']} | 0 | {r['n_total']} |"
                    + " - |" * (len(COLUMNS) + 3)
                )
                continue
            fr = s["frac_p_gt_half"]
            op = s.get("oracle_p_median")
            cells = [_cell(s, k) for k in COLUMNS]
            L.append(
                f"| {r['noise']:g} | {r['region']} | {s['n']} | {r['n_total']} | "
                + " | ".join(cells)
                + f" | {'-' if fr is None else f'{fr:.2f}'} | {'-' if op is None else f'{op:.2f}'} | {s['frac_no_tensor']:.2f} |"
            )
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- the run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("fig_dir", type=Path)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", nargs="+", default=list(FT.PHANTOMS))
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.fig_dir.mkdir(parents=True, exist_ok=True)
    feats = FP.features(a.site, str(a.features))
    t00 = time.perf_counter()
    records: list[dict[str, Any]] = []
    summary_rows: list[tuple[str, dict[str, Any]]] = []
    run_info: dict[str, Any] = {"phantoms": {}, "fields": {}, "files": {}}
    for name in a.names:
        t0 = time.perf_counter()
        ph0 = FT.build_phantom(name, feats, a.site, 0.0, seed=a.seed)
        stats = ST.SiteStats.from_reference(ph0)
        tr: T.Truth | None = None
        if name != "noise":
            tr = T.Truth.from_npz(a.truth / name / "truth.npz")
        geo, secs = sections_for(ph0, tr)
        oracle = FO.OracleStopField(ph0.geometry) if ph0.geometry is not None else None
        rng = np.random.default_rng(a.seed)
        regions: dict[str, dict[str, Any]] | None = None
        sec_data: dict[float, list[tuple[np.ndarray, np.ndarray | None]]] = {}
        term_panels: dict[float, dict[str, np.ndarray]] = {}
        zoom: tuple[Any, ...] | None = None
        for noise in a.noise:
            ph = (
                ph0
                if noise == 0.0
                else FT.build_phantom(name, feats, a.site, noise, seed=a.seed)
            )
            tensor = FO.ImageTensor.of(ph)
            field = ST.ImageStopField(ph, tensor, stats)
            run_info["fields"][f"{name}@{noise:g}"] = field.record()
            if regions is None:
                regions = region_points(name, ph0, tr, tensor, rng)
            for region, rp in regions.items():
                s = evaluate(field, oracle, rp["X"])
                records.append(
                    dict(
                        phantom=name,
                        noise=noise,
                        region=region,
                        kind=rp["kind"],
                        n_total=rp["n_total"],
                        stats=s,
                    )
                )
                if (
                    noise == MAIN_NOISE
                    and rp["kind"] is not None
                    and region.startswith("stop_")
                    and "_m" not in region
                    and s["n"]
                ):
                    summary_rows.append((f"{name} {region[5:]} (n={s['n']})", s))
            sec_data[noise] = [section_fields(geo, sec, field, oracle) for sec in secs]
            if name in TERMS_PHANTOMS:
                if zoom is None:
                    hs, vs, X, Y = zoom_grid(geo)
                    sdf = geo.stop_sdf(Y).reshape(len(vs), len(hs))
                    zoom = (hs, vs, X, sdf, float(geo.g(np.array([FUSED_Z_MM]))[0]))
                t = field.terms(zoom[2])
                t["sum"] = sum(t[k] for k in TERMS)
                term_panels[noise] = t
            del field, tensor
        run_info["files"][f"inputs_{name}.png"] = phantom_figure(
            name, geo, secs, sec_data, oracle is not None, a.fig_dir
        )
        if zoom is not None:
            run_info["files"][f"inputs_terms_{name}.png"] = terms_figure(
                name, zoom[0], zoom[1], zoom[3], zoom[4], term_panels, a.fig_dir
            )
        run_info["phantoms"][name] = round(time.perf_counter() - t0, 1)
        print(f"{name}: {run_info['phantoms'][name]} s", flush=True)
    (a.out_dir / "inputs_terms.json").write_text(json.dumps(records, indent=1))
    (a.out_dir / "inputs_terms.md").write_text(table_md(records, a))
    if summary_rows:
        run_info["files"]["inputs_terms_summary.png"] = summary_figure(
            summary_rows, a.fig_dir
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    run_info.update(
        total_s=round(time.perf_counter() - t00, 1),
        peak_rss_gib=round(rss / 2**30, 3),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
    )
    (a.out_dir / "run.json").write_text(json.dumps(run_info, indent=1))
    print(f"total {run_info['total_s']} s; peak {run_info['peak_rss_gib']} GiB")


if __name__ == "__main__":
    main()
