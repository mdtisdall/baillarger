"""Cheap probes at the site of the bent-columns trial, scored side by side.

A probe is a path from a white-mesh root that collects laminar evidence: the
profile along it, where it crosses the laminae, and where the top is. The
columns are a separate model (a diffeomorphism refined later); probes may
cross. This script asks which cheap probe collects the best evidence.

Usage:
    python prototypes/scripts/bent_probes.py run --box BOX.npz \\
        --texture TEXTURE.npz --fit-dir DIR --out OUT [--probes a,b,...]
    python prototypes/scripts/bent_probes.py views --box BOX.npz \\
        --texture TEXTURE.npz --fit-dir DIR --out OUT FIGS

The probes (each sampled every STEP mm from the root to S_MAX):
- normal: straight along the box's normals (the published smoothed
  normals).
- smooth24, smooth97: straight along the normals of
  mesh_smoothing.smooth_normals on the box mesh with 24 and 97 steps. One
  step averages a vertex with the mean of its 1-ring (weight 1/2), so k
  steps spread over about h sqrt(k / 2), with h the median edge (0.086 mm
  here): about 0.3 and 0.6 mm.
- l1: a streamline of orient's leading eigenvector, from the root along
  the normal. Each step (midpoint rule) takes l1, with the sign closest to
  the current heading, where the coherence c >= L1_MIN_C and the angle
  between l1 and the heading is at most L1_MAX_TURN_DEG. Elsewhere (low
  coherence, a turn too large to trust, no tensor) it keeps its heading.
  The constants were set before the run.
- bend_ac: the alternating fit of bent_fit with only the agreement,
  alignment, length and bending terms (the other weights 0): a short local
  bend toward laminar agreement, with no barrier and no crowding.
- B7, and any other run of the fit directory: the fit's paths as saved.

The top of every probe: the first crossing of 0 by the antialiased field of
label 0 (baillarger.antialias.thin_plate, the white surface's method),
linear between samples. A probe with no crossing up to S_MAX is censored.

The scores, on the inner columns:
- censored: the count with no top.
- far_top: the count whose top is more than FAR_RATIO times the distance
  from its root to the nearest label 0 voxel centre (it runs up a bank).
- jumps: the share of inner mesh edges whose two tops differ by more than
  1 mm; and along the trial's section, the count of neighbour pairs that do.
- drop2: the tops of columns 12183700 and 12184198 (label 0 at 5.92 and
  2.24 mm on the straight rays).
- align: the angle between the probe and l1 at samples below the top, in
  the cortex label, with c >= ALIGN_MIN_C (median and p90 of all such
  samples).
- agreement: for each inner edge, the mean squared difference of the two
  probes' profiles at equal arc length, from 0 to the shorter top. The
  profile is log echo 2 minus a Gaussian trend of TREND_MM along the probe,
  divided by the robust SD over the box cortex. Equal arc length, not a
  warp, so that every probe is read the same way (median and p90 over the
  edges).
- returns: the count that enter the WM labels again after they left them,
  below the top.
- inverted: folding of the probes (a property of columns, not a probe
  score; given for reference). Per inner root, the determinant of the local
  linear map from the roots within FOLD_SUPPORT_MM to the probe points at a
  depth, in the root's tangent plane (fold_dets). The share with det < 0 at
  each of FOLD_DEPTHS, and the share folded at some depth not past their
  own top. A per-triangle test is not used: the mesh's sliver triangles
  flip with any small difference of direction.
- seconds: the time to make the probes (not the top or the scores).
"""

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import bent_box as BB
import bent_columns as BC
import bent_fit as BF
import bent_terms as BT
import bent_texture as BX
import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates

from baillarger import antialias as AA
from baillarger import backend
from baillarger import mesh_smoothing as MS
from baillarger import orient as O

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

STEP = BT.STEP
S_MAX = BT.S_MAX
#: smoothing steps of the smoothed-normal probes
SMOOTH_STEPS = (24, 97)
#: the l1 probe: coherence and turn limits (set before the run)
L1_MIN_C = 0.3
L1_MAX_TURN_DEG = 45.0
#: far_top: the top beyond this many times the distance to label 0
FAR_RATIO = 2.0
#: align: samples with at least this coherence
ALIGN_MIN_C = 0.7
#: the profile trend (mm, Gaussian sigma along the probe)
TREND_MM = 0.5
#: the two drop-2 columns of the trial's section
DROP2 = (12183700, 12184198)
#: folding: the depths (mm) and the support radius (mm) of the local Jacobian
FOLD_DEPTHS = (1.0, 1.5, 2.0, 2.5, 3.0)
FOLD_SUPPORT_MM = 0.4
#: the terms kept in the bend_ac fit
BEND_AC_TERMS = ("agreement", "alignment", "length", "bending")


def say(msg: str) -> None:
    print(msg, flush=True)


def s_grid() -> F64:
    """The arc lengths of the samples, 0 to S_MAX."""
    return np.arange(int(round(S_MAX / STEP)) + 1, dtype=np.float64) * STEP


# ---------------------------------------------------------------- the probes


def straight(roots: F64, dirs: F64) -> F64:
    """(n, NS, 3) points r + s d, mm."""
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    out: F64 = roots[:, None, :] + s_grid()[None, :, None] * d[:, None, :]
    return out


def smoothed_normals(box: BB.Box, steps: int) -> F64:
    """The box mesh's normals after `steps` smoothing steps, signed to
    agree with the box's normals."""
    n: F64 = MS.smooth_normals(
        box.roots.astype(np.float64), np.asarray(box.faces, np.int64), steps
    )
    sgn = np.sign((n * box.normals).sum(1))
    sgn[sgn == 0] = 1
    out: F64 = n * sgn[:, None]
    return out


def l1_field(tensor: F64, box: BB.Box) -> Callable[[F64], tuple[F64, F64, Bool]]:
    """A function of points (m, 3) mm giving l1 (unit, mm frame), its
    coherence c and whether the tensor is defined there."""
    D = np.asarray(box.affine)[:3, :3] / np.asarray(box.voxel_mm)

    def f(x: F64) -> tuple[F64, F64, Bool]:
        t = BX.sample_tensor(tensor, box.to_box(x), 1)
        ok = np.all(np.isfinite(t), 0)
        M = np.zeros((x.shape[0], 3, 3))
        for q, (i, j) in enumerate(O.PAIRS):
            M[:, i, j] = M[:, j, i] = np.where(ok, t[q], 0)
        w, V = np.linalg.eigh(M)
        v = V[:, :, 2] @ D.T
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.clip((w[:, 2] - w[:, 1]) / (w[:, 2] + w[:, 1]), 0, 1)
        return v, np.where(ok, c, 0.0), ok

    return f


def l1_heading(
    v: F64, c: F64, ok: Bool, h: F64, min_c: float, max_turn_deg: float
) -> F64:
    """The next heading: l1 with the sign closest to h, where ok, c >= min_c
    and the angle to h is at most max_turn_deg; else h."""
    dot = np.einsum("ij,ij->i", v, h)
    s = np.where(dot < 0, -1.0, 1.0)
    vv = v * s[:, None]
    use = ok & (c >= min_c) & (np.abs(dot) >= np.cos(np.radians(max_turn_deg)))
    out: F64 = np.where(use[:, None], vv, h)
    return out


def l1_probes(
    roots: F64,
    normals: F64,
    field: Callable[[F64], tuple[F64, F64, Bool]],
    min_c: float = L1_MIN_C,
    max_turn_deg: float = L1_MAX_TURN_DEG,
) -> F64:
    """(n, NS, 3) streamlines of l1 (module docstring), to S_MAX."""
    ns = len(s_grid())
    P = np.empty((len(roots), ns, 3))
    x = roots.astype(np.float64).copy()
    h = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    P[:, 0] = x
    for k in range(1, ns):
        h1 = l1_heading(*field(x), h, min_c, max_turn_deg)
        h2 = l1_heading(*field(x + 0.5 * STEP * h1), h1, min_c, max_turn_deg)
        x = x + STEP * h2
        h = h2
        P[:, k] = x
    return P


@contextmanager
def only_terms(keep: Sequence[str]) -> Any:
    """bent_fit.start_weights with every weight not in keep set to 0."""
    orig = BF.start_weights

    def sw(
        prob: BF.Problem, levels: BT.Levels, form: str, seed: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        w, rec = orig(prob, levels, form, seed)
        for k in w:
            if k not in keep:
                rec[k] = dict(rec[k], rule_weight=w[k], scan_factor=0.0)
                w[k] = 0.0
        return w, rec

    BF.start_weights = sw
    try:
        yield
    finally:
        BF.start_weights = orig


def fit_points(params: npt.ArrayLike, inputs: BT.Inputs, box: BB.Box) -> F64:
    """(n, NS, 3) the points of a fit's paths, mm, on this script's grid."""
    with torch.no_grad():
        p = BT.paths(torch.as_tensor(np.asarray(params)), inputs)
    x = p.x.numpy().astype(np.float64)
    s = p.s.numpy().astype(np.float64)
    g = s_grid()
    if len(s) != len(g) or not np.allclose(s, g, atol=1e-5):
        raise ValueError("the fit's samples are not this script's grid")
    return x


# ---------------------------------------------------------------- the top


def first_crossing(f: F64, s: F64) -> F64:
    """Per row of f (n, NS): the arc length s where f first becomes > 0,
    linear between samples; s[0] if f[:, 0] > 0; NaN if never."""
    pos = f > 0
    any_ = pos.any(1)
    k = np.argmax(pos, 1)
    k0 = np.maximum(k - 1, 0)
    r = np.arange(f.shape[0])
    fa, fb = f[r, k0], f[r, k]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip(np.where(fb != fa, -fa / (fb - fa), 0.0), 0, 1)
    out: F64 = s[k0] + t * (s[k] - s[k0])
    out[k == 0] = s[0]
    out[~any_] = np.nan
    return out


def sample(vol: npt.ArrayLike, box: BB.Box, P: F64, order: Literal[0, 1] = 1) -> F64:
    """vol (box grid) at the points P (..., 3) mm."""
    q = box.to_box(P.reshape(-1, 3)).T
    out: F64 = map_coordinates(
        np.asarray(vol, np.float64), q, order=order, mode="nearest"
    ).reshape(P.shape[:-1])
    return out


def fold_dets(R: F64, N: F64, X: F64, idx: I64, support: float) -> F64:
    """Per root in idx: the determinant of the local linear map from the
    roots within `support` mm (in the tangent plane of N) to the points X
    (in the same plane), by least squares. Below 0: the probes fold there;
    1: no change of area. NaN with fewer than 6 roots in the support."""
    from scipy.spatial import cKDTree

    tree = cKDTree(R)
    out = np.full(len(idx), np.nan)
    nb = tree.query_ball_point(R[idx], support)
    for k, (i, js) in enumerate(zip(idx, nb, strict=True)):
        j = np.asarray(js, np.int64)
        if len(j) < 6:
            continue
        n = N[i] / np.linalg.norm(N[i])
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        t1 = np.cross(n, a)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        dr, dx = R[j] - R[i], X[j] - X[i]
        U = np.c_[dr @ t1, dr @ t2, np.ones(len(j))]
        V = np.c_[dx @ t1, dx @ t2]
        J = np.linalg.lstsq(U, V, rcond=None)[0]
        out[k] = float(np.linalg.det(J[:2]))
    return out


# ---------------------------------------------------------------- the scores


def _q(a: F64) -> dict[str, float | None]:
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(median=None, p90=None)
    return dict(
        median=round(float(np.median(a)), 4), p90=round(float(np.percentile(a, 90)), 4)
    )


def profiles(box: BB.Box, P: F64, cortex_sd: float) -> F64:
    """log echo 2 along P minus a Gaussian trend of TREND_MM along the
    probe, divided by cortex_sd."""
    e = np.log(np.maximum(sample(box.echoes[1], box, P), 1.0))
    tr = gaussian_filter1d(e, TREND_MM / STEP, axis=1, mode="nearest")
    out: F64 = (e - tr) / cortex_sd
    return out


def cortex_sd(box: BB.Box) -> float:
    """The robust SD (1.4826 MAD) of the detrended log echo 2 over the box
    cortex (trend: Gaussian of TREND_MM in 3D)."""
    from scipy.ndimage import gaussian_filter

    e = np.log(np.maximum(box.echoes[1].astype(np.float64), 1.0))
    d = e - gaussian_filter(e, TREND_MM / box.voxel_mm)
    v = d[box.labels == 42] if (box.labels == 42).any() else d[box.cortex]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def scores(
    box: BB.Box,
    P: F64,
    phi: F64,
    dist0: F64,
    field: Callable[[F64], tuple[F64, F64, Bool]],
    sd: float,
    section: I64,
) -> tuple[dict[str, Any], dict[str, F64]]:
    """The scores of the probes P (module docstring) and the per-column
    arrays (top, far, agreement per column, align median per column)."""
    s = s_grid()
    inner = np.asarray(box.inner, bool)
    n = len(P)
    top = first_crossing(sample(phi, box, P), s)
    cen = ~np.isfinite(top)
    d0 = sample(dist0, box, P[:, :1])[:, 0]
    far = np.isfinite(top) & (
        top > FAR_RATIO * np.maximum(d0, float(np.min(box.voxel_mm)))
    )
    E = np.asarray(box.edges, np.int64)
    ei = inner[E].all(1)
    both = ei & np.isfinite(top[E]).all(1)
    dL = np.abs(top[E[:, 0]] - top[E[:, 1]])
    sec_t = top[section]
    sec_j = np.abs(np.diff(sec_t))
    vids = np.asarray(box.rays.roots.vertex, np.int64)
    drop2 = []
    for v in DROP2:
        i = np.flatnonzero(vids == v)
        drop2.append(round(float(top[i[0]]), 3) if i.size else None)
    below = s[None, :] <= np.where(np.isfinite(top), top, -1)[:, None]
    lab = sample(box.labels, box, P, order=0).astype(np.int64)
    # alignment
    tang = np.gradient(P, STEP, axis=1)
    tang /= np.maximum(np.linalg.norm(tang, axis=2, keepdims=True), 1e-12)
    flat = P.reshape(-1, 3)
    v1, c1, ok1 = field(flat)
    vv = v1.reshape(P.shape)
    cc = c1.reshape(P.shape[:2])
    okk = ok1.reshape(P.shape[:2])
    ang = np.degrees(np.arccos(np.clip(np.abs((vv * tang).sum(-1)), 0, 1)))
    am = below & okk & (cc >= ALIGN_MIN_C) & (lab == 42) & inner[:, None]
    col_align = np.full(n, np.nan)
    has = am.any(1)
    col_align[has] = np.array([np.median(ang[i][am[i]]) for i in np.flatnonzero(has)])
    # agreement at equal arc length, to the shorter top
    f = profiles(box, P, sd)
    lo = np.fmin(top[E[:, 0]], top[E[:, 1]])
    ok_e = ei & np.isfinite(lo)
    msd = np.full(len(E), np.nan)
    m = (s[None, :] <= lo[ok_e][:, None]).astype(np.float64)
    dd = (f[E[ok_e, 0]] - f[E[ok_e, 1]]) ** 2
    msd[ok_e] = (dd * m).sum(1) / np.maximum(m.sum(1), 1)
    col_agree = np.full(n, np.nan)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    okm = np.isfinite(msd)
    for k in range(2):
        np.add.at(acc, E[okm, k], msd[okm])
        np.add.at(cnt, E[okm, k], 1)
    col_agree[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    # returns into WM below the top
    wm = np.isin(lab, np.asarray(box.filled))
    left = np.maximum.accumulate(~wm, axis=1)
    ret = (left & wm & below).any(1)
    # folding: the local Jacobian of the map from the roots to the probe
    # points at a depth (fold_dets)
    R = box.roots.astype(np.float64)
    N = np.asarray(box.normals, np.float64)
    ii = np.flatnonzero(inner)
    inv: dict[str, float] = {}
    dets = {}
    for d in FOLD_DEPTHS:
        k = int(round(d / STEP))
        dets[d] = fold_dets(R, N, P[:, k], ii, FOLD_SUPPORT_MM)
        inv[f"{d:g}mm"] = round(float(np.nanmean(dets[d] < 0)), 4)
    # folded below the top: det < 0 at the largest fold depth not past the top
    t_i = top[ii]
    fb = np.zeros(len(ii), bool)
    for d in FOLD_DEPTHS:
        fb |= np.isfinite(t_i) & (t_i >= d) & (dets[d] < 0)
    inv["below_top"] = round(float(np.mean(fb)), 4)
    rec = dict(
        inner=int(inner.sum()),
        censored=int((cen & inner).sum()),
        far_top=int((far & inner).sum()),
        jumps_edges=round(float(np.mean(dL[both] > 1.0)), 4),
        jumps_section=int(np.sum(sec_j[np.isfinite(sec_j)] > 1.0)),
        drop2=drop2,
        align_deg=_q(ang[am]),
        agreement=_q(msd[ok_e]),
        returns=int((ret & inner).sum()),
        inverted=inv,
        top=_q(top[inner]),
    )
    cols = dict(top=top, far=far.astype(np.float64), agree=col_agree, align=col_align)
    return rec, cols


# ---------------------------------------------------------------- the run


def make_probes(
    names: Sequence[str],
    box: BB.Box,
    prob: BF.Problem,
    tensor: F64,
    fit_dir: Path,
    out: Path,
) -> dict[str, tuple[F64, float]]:
    """Each named probe's points and the seconds to make them. bend_ac is
    fitted once and saved to out/bend_ac_paths.npz; a saved fit is read."""
    roots = box.roots.astype(np.float64)
    res: dict[str, tuple[F64, float]] = {}
    for nm in names:
        t0 = time.time()
        if nm == "normal":
            P = straight(roots, box.normals.astype(np.float64))
        elif nm.startswith("smooth"):
            P = straight(roots, smoothed_normals(box, int(nm[len("smooth") :])))
        elif nm == "l1":
            P = l1_probes(roots, box.normals.astype(np.float64), l1_field(tensor, box))
        elif nm == "bend_ac":
            q = out / "bend_ac_paths.npz"
            if q.exists():
                with np.load(q) as z:
                    params = z["params"]
                sec = float(
                    json.loads((out / "bend_ac_costs.json").read_text())["seconds"]
                )
            else:
                with only_terms(BEND_AC_TERMS):
                    f = BF.fit(
                        prob, "B", True, 4, 15, 0, log=lambda m: say(f"  bend_ac: {m}")
                    )
                params = f.params.numpy()
                sec = time.time() - t0
                np.savez_compressed(q, params=params)
                (out / "bend_ac_costs.json").write_text(
                    json.dumps(
                        dict(
                            stop=f.stop,
                            seconds=sec,
                            weights=f.weights,
                            history=f.history,
                        ),
                        indent=1,
                        default=str,
                    )
                )
            P = fit_points(params, prob.inputs, box)
            res[nm] = (P, sec)
            continue
        else:
            with np.load(fit_dir / f"{nm}_paths.npz") as z:
                params = z["params"]
            P = fit_points(params, prob.inputs, box)
            sec = float(
                json.loads((fit_dir / f"{nm}_costs.json").read_text())["seconds"]
            )
            res[nm] = (P, sec)
            continue
        res[nm] = (P, time.time() - t0)
        say(f"{nm}: probes in {res[nm][1]:.1f} s")
    return res


def section_of(box: BB.Box) -> I64:
    """The trial's section (bent_views.Site): inner columns within
    SECTION_HALF_MM of the plane of world x and the vertex normal, in order
    along world x."""
    import bent_views as BV

    vids = np.asarray(box.rays.roots.vertex, np.int64)
    vi = int(np.flatnonzero(vids == box.vertex)[0])
    n_v = box.normals[vi].astype(np.float64)
    r_v = box.roots[vi].astype(np.float64)
    ux = np.array([1.0, 0.0, 0.0]) - n_v[0] * n_v
    ux /= np.linalg.norm(ux)
    w = np.cross(n_v, ux)
    d = box.roots.astype(np.float64) - r_v
    sel = np.flatnonzero(box.inner & (np.abs(d @ w) <= BV.SECTION_HALF_MM))
    out: I64 = sel[np.argsort(d[sel] @ ux)]
    return out


def run(
    box_path: Path,
    texture: Path,
    fit_dir: Path,
    out: Path,
    names: Sequence[str],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    box, prob = BC.load_problem(box_path, texture)
    with np.load(texture) as z:
        tensor = np.asarray(z["tensor"], np.float64)
    q = out / "phi_outside.npy"
    if q.exists():
        phi = np.load(q)
    else:
        phi, sol = AA.thin_plate(box.labels == 0, backend.device("cpu"))
        np.save(q, phi)
        say(f"antialiased label 0: {sol.summary()}")
    dist0 = distance_transform_edt(
        box.labels != 0, sampling=np.asarray(box.voxel_mm, np.float64)
    )
    field = l1_field(tensor, box)
    sd = cortex_sd(box)
    sec = section_of(box)
    probes = make_probes(names, box, prob, tensor, fit_dir, out)
    allrec: dict[str, Any] = {}
    for nm, (P, seconds) in probes.items():
        rec, cols = scores(box, P, phi, dist0, field, sd, sec)
        rec["seconds"] = round(seconds, 1)
        allrec[nm] = rec
        np.savez_compressed(
            out / f"probe_{nm}.npz",
            top=cols["top"],
            far=cols["far"],
            agree=cols["agree"],
            align=cols["align"],
        )
        say(f"{nm}: {json.dumps(rec)}")
    meta = dict(
        settings=dict(
            step=STEP,
            s_max=S_MAX,
            smooth_steps=SMOOTH_STEPS,
            l1_min_c=L1_MIN_C,
            l1_max_turn_deg=L1_MAX_TURN_DEG,
            far_ratio=FAR_RATIO,
            align_min_c=ALIGN_MIN_C,
            trend_mm=TREND_MM,
            cortex_sd=sd,
            bend_ac_terms=BEND_AC_TERMS,
        ),
        probes=allrec,
    )
    prev = out / "probes.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        old.get("probes", {}).update(allrec)
        meta["probes"] = old["probes"]
    prev.write_text(json.dumps(meta, indent=1))
    return meta


def table(meta: Mapping[str, Any]) -> list[str]:
    rows = [
        "| probe | censored | far top | jumps (edges / section) | drop 2 (mm) "
        "| align med / p90 (deg) | agreement med / p90 | returns "
        "| folded 1 / 2.5 mm / below top | s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for nm, r in meta["probes"].items():
        d2 = " / ".join(
            "—" if x is None or not np.isfinite(x) else f"{x:.2f}" for x in r["drop2"]
        )
        rows.append(
            f"| {nm} | {r['censored']} | {r['far_top']} | "
            f"{100 * r['jumps_edges']:.1f} % / {r['jumps_section']} | {d2} | "
            f"{r['align_deg']['median']} / {r['align_deg']['p90']} | "
            f"{r['agreement']['median']} / {r['agreement']['p90']} | {r['returns']} | "
            f"{100 * r['inverted']['1mm']:.0f} % / "
            f"{100 * r['inverted']['2.5mm']:.0f} % / "
            f"{100 * r['inverted']['below_top']:.1f} % | {r['seconds']} |"
        )
    return rows


def views(box_path: Path, texture: Path, fit_dir: Path, out: Path, figs: Path) -> None:
    """Maps of each probe's top, far tops, agreement and alignment, and the
    section's tops."""
    import matplotlib

    matplotlib.use("Agg")
    import bent_views as BV
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    site = BV.Site(box_path, texture, fit_dir)
    meta = json.loads((out / "probes.json").read_text())
    names = list(meta["probes"])
    cols = {nm: dict(np.load(out / f"probe_{nm}.npz")) for nm in names}
    rows = [
        ("top", "top (mm)", 0, 6, "viridis"),
        ("agree", "profile difference with neighbours", 0, 2, "magma"),
        ("align", "angle to l1, c >= 0.7 (deg)", 0, 60, "magma"),
    ]
    fig, axs = plt.subplots(
        len(rows),
        len(names),
        figsize=(3.6 * len(names), 3.4 * len(rows)),
        squeeze=False,
    )
    for i, (key, lab, lo, hi, cm) in enumerate(rows):
        for j, nm in enumerate(names):
            val = cols[nm][key]
            sc = BV._map(
                site,
                axs[i, j],
                val,
                f"{nm}: {lab}",
                lo,
                hi,
                cmap=cm,
                mask=np.isfinite(val),
            )
            if key == "top":
                far = cols[nm]["far"] > 0
                xy = site.map_xy
                m = far & site.box.inner
                axs[i, j].scatter(xy[m, 0], xy[m, 1], s=0.5, c="r", marker="x", lw=0.3)
        fig.colorbar(sc, ax=axs[i, -1], shrink=0.8)
    fig.suptitle(
        "Probes at the site: top (red x: far top, grey: censored), "
        "profile difference with neighbours, angle to l1",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figs / "probe_maps.png", dpi=50)
    plt.close(fig)
    sec = section_of(site.box)
    u = site.section_u
    fig, axs = plt.subplots(
        len(names), 1, figsize=(10, 1.9 * len(names)), sharex=True, squeeze=False
    )
    for a, nm in zip(axs[:, 0], names, strict=True):
        t = cols[nm]["top"][sec]
        far = cols[nm]["far"][sec] > 0
        a.plot(u, t, ".", ms=2, c="C0")
        a.plot(u[far], t[far], "x", ms=3, c="r")
        cen = ~np.isfinite(t)
        a.plot(u[cen], np.full(cen.sum(), 7.8), "v", ms=2, c="k")
        a.set_ylim(0, 8.2)
        a.set_ylabel(nm, fontsize=8)
    axs[-1, 0].set_xlabel("u along the section (mm)")
    fig.suptitle(
        "Antialiased top along the section (red x: far top, black v: censored)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figs / "probe_section.png", dpi=80)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("run", "views"):
        p = sub.add_parser(nm)
        p.add_argument("--box", type=Path, required=True)
        p.add_argument("--texture", type=Path, required=True)
        p.add_argument("--fit-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if nm == "run":
            p.add_argument("--probes", default="normal,smooth24,smooth97,l1,B7")
        else:
            p.add_argument("figs", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        meta = run(a.box, a.texture, a.fit_dir, a.out, a.probes.split(","))
        (a.out / "probes.md").write_text("\n".join(table(meta)) + "\n")
        print("\n".join(table(meta)))
    else:
        views(a.box, a.texture, a.fit_dir, a.out, a.figs)


if __name__ == "__main__":
    main()
