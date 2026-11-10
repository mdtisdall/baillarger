"""A prior score for each probe at the site of the bent-columns trial.

Each probe gets P, the product of soft terms in [0, 1], one for each thing we
expect of a correct column. P is a score under stated assumptions: the terms
are treated as independent, and no term has been calibrated against known
columns. It is not yet a measured frequency of correct columns.

Usage:
    python prototypes/scripts/bent_confidence.py run --box BOX.npz \\
        --texture TEXTURE.npz --fit-dir DIR --probe-dir DIR --out OUT \\
        [--probes l1,normal,smooth97,bend_ac]
    python prototypes/scripts/bent_confidence.py views --box BOX.npz \\
        --texture TEXTURE.npz --fit-dir DIR --probe-dir DIR --out OUT

A term is 1 inside its allowed range. Past the range x0, it falls as
exp(-((x - x0) / tau)^2 / 2). The constants are set before the first run.

Geometry:
- thickness: the top is at 1 to 4.5 mm (the usual range for human cortex).
- directness: top / distance from the root to the nearest label 0.
- arrival: the angle between the probe and the normal of the outside surface
  at the top (a column meets that surface about square on).
- bank: the largest fall of the distance from WM below the top (a column goes
  away from its WM, and it does not cross to another bank).

Laminae:
- crossing: the column's median angle to l1 where the coherence is high (a
  column crosses the laminae about square on). For the l1 probe this term is
  close to 1 by construction, so it is not evidence there.
- neighbourhood: the mean squared difference between the column's profile and
  the mean profile of its mesh neighbourhood (6 hops, about 0.5 mm), at equal
  arc length, in units of the cortex's robust variance.

Smoothness:
- top step: the largest step of the top to a mesh neighbour. A neighbour with
  no top counts as an infinite step.

Two rules set P to 0:
- no top within the 8 mm cap;
- the probes fold (local Jacobian below 0) at a depth not past the top.

Support is a separate value, not a term: the share of samples below the top,
in cortex, with coherence of at least 0.7. Under 25 % the column is "silent":
the data do little to test it, so a high P means only that nothing disagrees.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import bent_box as BB
import bent_columns as BC
import bent_probes as BP
import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.sparse import coo_matrix, identity
from scipy.sparse.csgraph import connected_components

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

STEP = BP.STEP
#: (x0, tau) of each term; see the module docstring
THICK_MIN = (1.0, 0.25)
THICK_MAX = (4.5, 0.5)
DIRECT = (1.3, 0.35)
ARRIVAL_DEG = (30.0, 15.0)
FALL_MM = (0.16, 0.16)
CROSSING_DEG = (15.0, 15.0)
NEIGHBOUR_MSD = (0.25, 0.35)
TOP_STEP_MM = (0.2, 0.2)
#: the mesh hops of the neighbourhood (median edge 0.086 mm: about 0.5 mm)
HOPS = 6
#: the folding depths, mm
FOLD_DEPTHS = (0.5, *BP.FOLD_DEPTHS)
#: support: the coherence and the share under which a column is silent
SUPPORT_C = 0.7
SUPPORT_SHARE = 0.25
#: the band called "high"
HIGH = 0.8
#: the terms of each group
GROUPS: Mapping[str, tuple[str, ...]] = {
    "geometry": ("thickness", "directness", "arrival", "bank"),
    "laminae": ("crossing", "neighbourhood"),
    "smoothness": ("top_step",),
}
TERM_NAMES = tuple(t for g in GROUPS.values() for t in g)
#: the failing stretch of the trial's section, mm along world x
STRETCH = (-2.0, 0.7)
#: the places of the cuts along the section, mm
CUTS_U = (-3.5, -0.6, -0.1, 2.0)


def soft(x: npt.ArrayLike, x0: float, tau: float) -> F64:
    """1 up to x0, then a half Gaussian of width tau. NaN gives 1: no data
    is not evidence against a column."""
    a = np.asarray(x, np.float64)
    e = np.maximum(a - x0, 0.0)
    out: F64 = np.exp(-0.5 * (e / tau) ** 2)
    out[np.isnan(a)] = 1.0
    return out


def soft_low(x: npt.ArrayLike, x0: float, tau: float) -> F64:
    """1 down to x0, then a half Gaussian of width tau. NaN gives 1."""
    return soft(-np.asarray(x, np.float64), -x0, tau)


def below_top(top: F64, s: F64) -> Bool:
    """(n, NS) True at the samples not past the top; none for no top."""
    out: Bool = s[None, :] <= np.where(np.isfinite(top), top, -1.0)[:, None]
    return out


def largest_fall(d: F64, top: F64, s: F64) -> F64:
    """Per row: the largest fall of d from an earlier peak, below the top."""
    m = below_top(top, s)
    peak = np.maximum.accumulate(np.where(m, d, -np.inf), axis=1)
    fall = np.where(m, peak - d, 0.0)
    out: F64 = fall.max(axis=1)
    out[~np.isfinite(top)] = np.nan
    return out


def grad_mm(vol: npt.ArrayLike, affine: npt.ArrayLike) -> F64:
    """(..., 3) the gradient of a volume in the affine's mm frame."""
    g = np.stack(np.gradient(np.asarray(vol, np.float64)), -1)
    inv = np.linalg.inv(np.asarray(affine, np.float64)[:3, :3])
    out: F64 = g @ inv
    return out


def arrival_deg(P: F64, top: F64, grad: F64, q_of: Any) -> F64:
    """Per probe: the angle between its heading at the top and the outward
    normal of the surface (grad of the outside field). NaN for no top."""
    out = np.full(len(P), np.nan)
    ok = np.flatnonzero(np.isfinite(top))
    if not ok.size:
        return out
    k = np.clip(np.round(top[ok] / STEP).astype(np.int64), 1, P.shape[1] - 1)
    t = P[ok, k] - P[ok, k - 1]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    x = P[ok, k - 1] + t * (top[ok] - (k - 1) * STEP)[:, None]
    q = q_of(x).T
    g = np.stack(
        [map_coordinates(grad[..., i], q, order=1, mode="nearest") for i in range(3)],
        1,
    )
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    out[ok] = np.degrees(np.arccos(np.clip((t * g).sum(1), -1, 1)))
    return out


def adjacency(edges: I64, n: int) -> Any:
    """The sparse graph of the mesh edges, with self loops."""
    a, b = edges[:, 0], edges[:, 1]
    A = coo_matrix(
        (np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)
    ).tocsr()
    return (A + identity(n, format="csr")).tocsr()


def neighbour_msd(f: F64, top: F64, s: F64, edges: I64, hops: int) -> F64:
    """Per column: the mean squared difference between its profile and the
    mean profile of its neighbourhood, at equal arc length, to its own top."""
    m = below_top(top, s).astype(np.float64)
    num = np.nan_to_num(f) * m
    den = m.copy()
    A = adjacency(edges, len(f))
    for _ in range(hops):
        num = A @ num
        den = A @ den
        scale = np.maximum(den.max(axis=1, keepdims=True), 1.0)
        num = num / scale
        den = den / scale
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan)
    d = (f - mean) ** 2
    use = (m > 0) & np.isfinite(d)
    cnt = use.sum(axis=1)
    out: F64 = np.full(len(f), np.nan)
    has = cnt > 0
    out[has] = np.where(use[has], d[has], 0.0).sum(axis=1) / cnt[has]
    return out


def top_step(top: F64, edges: I64) -> F64:
    """Per column: the largest |top difference| to a mesh neighbour. A
    neighbour with no top is an infinite step. NaN for a column with no top."""
    a, b = top[edges[:, 0]], top[edges[:, 1]]
    d = np.abs(a - b)
    d[np.isfinite(a) != np.isfinite(b)] = np.inf
    d[~np.isfinite(a) & ~np.isfinite(b)] = np.nan
    out = np.zeros(len(top))
    okd = ~np.isnan(d)
    np.maximum.at(out, edges[okd, 0], d[okd])
    np.maximum.at(out, edges[okd, 1], d[okd])
    out[~np.isfinite(top)] = np.nan
    return out


def combine(terms: Mapping[str, F64], names: Sequence[str], gate: Bool) -> F64:
    """The product of the named terms, 0 where the gate is False."""
    out = np.ones_like(gate, dtype=np.float64)
    for k in names:
        out = out * terms[k]
    out[~gate] = 0.0
    return out


def components(mask: Bool, edges: I64) -> I64:
    """The component label of each masked column over the mesh edges, -1
    elsewhere. Labels run from 0, largest component first."""
    n = len(mask)
    e = edges[mask[edges].all(1)]
    G = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    out: I64 = np.full(n, -1, np.int64)
    if not mask.any():
        return out
    u, inv, cnt = np.unique(lab[mask], return_inverse=True, return_counts=True)
    rank = np.empty(len(u), np.int64)
    rank[np.argsort(-cnt, kind="stable")] = np.arange(len(u))
    out[mask] = rank[inv]
    return out


def fold_gate(box: BB.Box, P: F64, top: F64) -> Bool:
    """True unless the probes fold at a depth not past the column's top. The
    columns of the box's outer ring have too little support and pass."""
    R = box.roots.astype(np.float64)
    N = np.asarray(box.normals, np.float64)
    ii = np.flatnonzero(box.inner)
    ok = np.ones(len(P), bool)
    for d in FOLD_DEPTHS:
        det = BP.fold_dets(R, N, P[:, int(round(d / STEP))], ii, BP.FOLD_SUPPORT_MM)
        ok[ii[np.isfinite(top[ii]) & (top[ii] >= d) & (det < 0)]] = False
    return ok


def score_probe(
    box: BB.Box,
    P: F64,
    top: F64,
    align: F64,
    grad: F64,
    dist0: F64,
    dist_wm: F64,
    field: Any,
    sd: float,
) -> dict[str, F64]:
    """Every per-column array of one probe."""
    s = BP.s_grid()
    edges = np.asarray(box.edges, np.int64)
    d0 = BP.sample(dist0, box, P[:, :1])[:, 0]
    ratio = top / np.maximum(d0, float(np.min(box.voxel_mm)))
    arr = arrival_deg(P, top, grad, box.to_box)
    fall = largest_fall(BP.sample(dist_wm, box, P), top, s)
    msd = neighbour_msd(BP.profiles(box, P, sd), top, s, edges, HOPS)
    step = top_step(top, edges)
    terms = dict(
        thickness=soft_low(top, *THICK_MIN) * soft(top, *THICK_MAX),
        directness=soft(ratio, *DIRECT),
        arrival=soft(arr, *ARRIVAL_DEG),
        bank=soft(fall, *FALL_MM),
        crossing=soft(align, *CROSSING_DEG),
        neighbourhood=soft(msd, *NEIGHBOUR_MSD),
        top_step=soft(step, *TOP_STEP_MM),
    )
    reach = np.isfinite(top)
    fold = fold_gate(box, P, top)
    gate = reach & fold
    # support
    _, c, ok = field(P.reshape(-1, 3))
    c = c.reshape(P.shape[:2])
    ok = ok.reshape(P.shape[:2])
    lab = BP.sample(box.labels, box, P, order=0).astype(np.int64)
    m = below_top(top, s) & (lab == box.cortex)
    n_m = m.sum(1)
    hi = (m & ok & (c >= SUPPORT_C)).sum(1)
    support = np.where(n_m > 0, hi / np.maximum(n_m, 1), 0.0)
    out: dict[str, F64] = {f"t_{k}": v for k, v in terms.items()}
    out.update(
        P=combine(terms, TERM_NAMES, gate),
        top=top,
        ratio=ratio,
        arrival=arr,
        fall=fall,
        align=align,
        msd=msd,
        step=step,
        support=support,
        reach=reach.astype(np.float64),
        fold=fold.astype(np.float64),
    )
    for g, names in GROUPS.items():
        out[f"P_{g}"] = combine(terms, names, gate)
    return out


def summary(
    cols: Mapping[str, F64], inner: Bool, edges: I64, stretch: I64
) -> dict[str, Any]:
    """The counts for one probe, on the inner columns."""
    P = cols["P"]
    high = (P >= HIGH) & inner
    sup = cols["support"] >= SUPPORT_SHARE
    A = high & sup
    lab = components(A, edges)
    sizes = np.bincount(lab[lab >= 0]) if A.any() else np.zeros(0, np.int64)
    n = int(inner.sum())
    terms = np.stack([cols[f"t_{k}"] for k in TERM_NAMES], 1)
    limit = np.argmin(terms, 1)
    gate = (cols["reach"] > 0) & (cols["fold"] > 0)
    low = inner & gate & (P < HIGH)
    return dict(
        inner=n,
        no_top=int((inner & (cols["reach"] == 0)).sum()),
        folded=int((inner & (cols["reach"] > 0) & (cols["fold"] == 0)).sum()),
        P_ge={str(t): int((inner & (P >= t)).sum()) for t in (0.5, 0.8, 0.9)},
        high_supported=int(A.sum()),
        high_silent=int((high & ~sup).sum()),
        patches=dict(
            count=len(sizes),
            largest=[int(x) for x in sizes[:5]],
            ge_100=int((sizes >= 100).sum()),
            columns_in_ge_100=int(sizes[sizes >= 100].sum()),
        ),
        groups_high={g: int((inner & (cols[f"P_{g}"] >= HIGH)).sum()) for g in GROUPS},
        limiting_term={
            k: int((low & (limit == i)).sum()) for i, k in enumerate(TERM_NAMES)
        },
        without_term={
            k: int(
                (
                    inner
                    & (
                        combine(
                            {t: cols[f"t_{t}"] for t in TERM_NAMES},
                            [t for t in TERM_NAMES if t != k],
                            gate,
                        )
                        >= HIGH
                    )
                ).sum()
            )
            for k in TERM_NAMES
        },
        stretch=dict(
            columns=len(stretch),
            high=int(high[stretch].sum()),
            high_supported=int(A[stretch].sum()),
            laminae_high=int((cols["P_laminae"][stretch] >= HIGH).sum()),
            geometry_high=int((cols["P_geometry"][stretch] >= HIGH).sum()),
        ),
    )


def settings() -> dict[str, Any]:
    return dict(
        thick_min=THICK_MIN,
        thick_max=THICK_MAX,
        direct=DIRECT,
        arrival_deg=ARRIVAL_DEG,
        fall_mm=FALL_MM,
        crossing_deg=CROSSING_DEG,
        neighbour_msd=NEIGHBOUR_MSD,
        top_step_mm=TOP_STEP_MM,
        hops=HOPS,
        fold_depths=FOLD_DEPTHS,
        support_c=SUPPORT_C,
        support_share=SUPPORT_SHARE,
        high=HIGH,
    )


def stretch_columns(box: BB.Box) -> I64:
    """The columns of the trial's section inside the failing stretch."""
    sec = BP.section_of(box)
    vids = np.asarray(box.rays.roots.vertex, np.int64)
    vi = int(np.flatnonzero(vids == box.vertex)[0])
    n_v = box.normals[vi].astype(np.float64)
    ux = np.array([1.0, 0.0, 0.0]) - n_v[0] * n_v
    ux /= np.linalg.norm(ux)
    u = (box.roots[sec].astype(np.float64) - box.roots[vi].astype(np.float64)) @ ux
    out: I64 = sec[(u >= STRETCH[0]) & (u <= STRETCH[1])]
    return out


def run(
    box_path: Path,
    texture: Path,
    fit_dir: Path,
    probe_dir: Path,
    out: Path,
    names: Sequence[str],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    box, prob = BC.load_problem(box_path, texture)
    with np.load(texture) as z:
        tensor = np.asarray(z["tensor"], np.float64)
    phi = np.load(probe_dir / "phi_outside.npy")
    grad = grad_mm(phi, box.affine)
    vm = np.asarray(box.voxel_mm, np.float64)
    dist0 = distance_transform_edt(box.labels != 0, sampling=vm)
    dist_wm = distance_transform_edt(
        ~np.isin(box.labels, np.asarray(box.filled)), sampling=vm
    )
    field = BP.l1_field(tensor, box)
    sd = BP.cortex_sd(box)
    inner = np.asarray(box.inner, bool)
    edges = np.asarray(box.edges, np.int64)
    stretch = stretch_columns(box)
    rec: dict[str, Any] = {}
    for nm in names:
        P, _ = BP.make_probes([nm], box, prob, tensor, fit_dir, probe_dir)[nm]
        with np.load(probe_dir / f"probe_{nm}.npz") as z:
            top, align = z["top"], z["align"]
        cols = score_probe(box, P, top, align, grad, dist0, dist_wm, field, sd)
        np.savez_compressed(out / f"confidence_{nm}.npz", **cols)
        rec[nm] = summary(cols, inner, edges, stretch)
        print(f"{nm}: {json.dumps(rec[nm])}", flush=True)
    meta = dict(settings=settings(), probes=rec)
    (out / "confidence.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------- the views


def bands(cols: Mapping[str, F64]) -> I64:
    """Per column: 0 no top or folded, 1 P < 0.5, 2 0.5 <= P < HIGH,
    3 high but silent, 4 high with support."""
    P = cols["P"]
    sup = cols["support"] >= SUPPORT_SHARE
    out: I64 = np.ones(len(P), np.int64)
    out[(P >= 0.5) & (P < HIGH)] = 2
    out[(P >= HIGH) & ~sup] = 3
    out[(P >= HIGH) & sup] = 4
    out[(cols["reach"] == 0) | (cols["fold"] == 0)] = 0
    return out


BAND_COLOURS = ("#8c4bd6", "#ff3030", "#ffa500", "#30b0ff", "#30e040")
BAND_NAMES = (
    "no top or folded",
    "P < 0.5",
    "0.5 to 0.8",
    "high, silent",
    "high, supported",
)


def views(
    box_path: Path,
    texture: Path,
    fit_dir: Path,
    probe_dir: Path,
    out: Path,
) -> None:
    import bent_views as BV
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    site = BV.Site(box_path, texture, fit_dir)
    b = site.box
    meta = json.loads((out / "confidence.json").read_text())
    names = list(meta["probes"])
    data = {nm: dict(np.load(out / f"confidence_{nm}.npz")) for nm in names}
    cm = ListedColormap(BAND_COLOURS)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=5, c=c, label=n)
        for c, n in zip(BAND_COLOURS, BAND_NAMES, strict=True)
    ]

    # the bands and P, per probe
    fig, axs = plt.subplots(
        2, len(names), figsize=(3.8 * len(names), 7.0), squeeze=False
    )
    for j, nm in enumerate(names):
        BV._map(
            site,
            axs[0, j],
            bands(data[nm]).astype(np.float64),
            f"{nm}: bands",
            -0.5,
            4.5,
            cmap=cm,
        )
        sc = BV._map(site, axs[1, j], data[nm]["P"], f"{nm}: P", 0, 1, cmap="viridis")
    fig.colorbar(sc, ax=axs[1, -1], shrink=0.8)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(
        "Probe score over the white surface. x: along the fundus, y: mesh distance "
        "from the fundus line (mm). Light grey: outer ring, not scored",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(figs / "confidence_maps.png", dpi=75)
    plt.close(fig)

    # the groups of terms, per probe
    keys = [f"P_{g}" for g in GROUPS]
    fig, axs = plt.subplots(
        len(keys),
        len(names),
        figsize=(3.8 * len(names), 3.2 * len(keys)),
        squeeze=False,
    )
    for i, k in enumerate(keys):
        for j, nm in enumerate(names):
            sc = BV._map(site, axs[i, j], data[nm][k], f"{nm}: {k[2:]}", 0, 1)
    fig.colorbar(sc, ax=axs[:, -1], shrink=0.5)
    fig.suptitle("The score of each group of terms alone", fontsize=9)
    fig.savefig(figs / "confidence_groups.png", dpi=70, bbox_inches="tight")
    plt.close(fig)

    # cuts across the sulcus, image only above and rays below
    R = b.roots.astype(np.float64)
    N = b.normals.astype(np.float64)
    phi = np.load(probe_dir / "phi_outside.npy")
    s = BP.s_grid()
    secs = site.section
    usec = site.section_u
    with np.load(texture) as z:
        tensor = np.asarray(z["tensor"], np.float64)
    for nm in names:
        Pts, _ = BP.make_probes([nm], b, site.prob, tensor, fit_dir, probe_dir)[nm]
        band = bands(data[nm])
        top = data[nm]["top"]
        fig, axs = plt.subplots(2, len(CUTS_U), figsize=(4.0 * len(CUTS_U), 8.2))
        for j, ut in enumerate(CUTS_U):
            c = secs[np.argmin(np.abs(usec - ut))]
            n, r = N[c], R[c]
            us = site.u_s - (site.u_s @ n) * n
            us /= np.linalg.norm(us)
            t = np.cross(n, us)
            gg = np.arange(-2.5, 2.51, 0.02)
            h = np.arange(-0.5, 6.51, 0.02)
            A_, B_ = np.meshgrid(gg, h)
            X = r + A_[..., None] * us + B_[..., None] * n
            q = b.to_box(X.reshape(-1, 3)).T
            img = map_coordinates(
                b.echoes[1].astype(np.float64), q, order=1, mode="nearest"
            ).reshape(A_.shape)
            ph = map_coordinates(phi, q, order=1, mode="nearest").reshape(A_.shape)
            ext = (gg[0], gg[-1], h[0], h[-1])
            hi = np.percentile(img, 99)
            axs[0, j].imshow(
                img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=hi
            )
            axs[0, j].set_title(f"image only: u = {ut} mm", fontsize=9)
            a = axs[1, j]
            a.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=hi)
            a.contour(A_, B_, ph, [0], colors="w", linewidths=0.7)
            near = np.flatnonzero(b.inner & (np.abs((R - r) @ t) < 0.05))
            near = near[:: max(1, len(near) // 60)]
            for k in near:
                L = top[k] if np.isfinite(top[k]) else BP.S_MAX
                Y = Pts[k][s <= L] - r
                a.plot(Y @ us, Y @ n, "-", c=BAND_COLOURS[band[k]], lw=0.8)
            a.set_title(f"{nm}: rays coloured by band", fontsize=9)
            for q_ in axs[:, j]:
                q_.set_xlim(-2.5, 2.5)
                q_.set_ylim(-0.5, 6.5)
                q_.tick_params(labelsize=7)
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
        fig.suptitle(
            f"Cuts across the sulcus, {nm}. Echo 2. White line: outside edge. "
            "Failing stretch: u from -2 to 0.7 mm",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fig.savefig(figs / f"confidence_cuts_{nm}.png", dpi=70)
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("run", "views"):
        p = sub.add_parser(nm)
        p.add_argument("--box", type=Path, required=True)
        p.add_argument("--texture", type=Path, required=True)
        p.add_argument("--fit-dir", type=Path, required=True)
        p.add_argument("--probe-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if nm == "run":
            p.add_argument("--probes", default="l1,normal,smooth97,bend_ac")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.box, a.texture, a.fit_dir, a.probe_dir, a.out, a.probes.split(","))
    else:
        views(a.box, a.texture, a.fit_dir, a.probe_dir, a.out)


if __name__ == "__main__":
    main()
