"""The command line of the bent-columns trial at the site
(docs/plans/bent-columns-trial.md, phase 6): box -> fit -> outputs.

Commands (run from the repository root, each heavy run under
scripts/memguard, one at a time):

    python prototypes/scripts/bent_columns.py fit --box BOX.npz \\
        --texture TEXTURE.npz --out DIR [--arms A7 A B7 B] \\
        [--scale TERM:FACTOR ...]
    python prototypes/scripts/bent_columns.py wrong --box BOX.npz \\
        --texture-json TEXTURE.json --out DIR
    python prototypes/scripts/bent_columns.py sparse --box BOX.npz \\
        --texture TEXTURE.npz --out DIR [--runs straight A7 A B7 B]
    python prototypes/scripts/bent_columns.py scan-table --out DIR

The problem
-----------
bent_terms.Inputs.from_box on the box (the box's labels: the painted ones
when bent_box was run with --paint), with the laminar field of cost 7
bent_terms.align_field(leading, c, box.affine, box.voxel_mm) of the texture
file (bent_texture's texture.npz, box voxel frame). The inner columns move;
the halo columns are fixed (straight). bent_fit.make_problem sets the
feature scales and the energy reference of the straight profiles.

fit
---
For each arm of bent_fit.ARMS (form A/B of cost 3, cost 7 on/off),
bent_fit.fit with the settings of phase 5: ROUNDS rounds of at most STEPS
L-BFGS iterations, extend bent_fit.EXTEND, the weight rule of
bent_fit.start_weights, seed SEED, float32 on cpu. The warp of the straight
rays is fitted once per process and given to every arm (warp0).

Writes, in DIR:
- straight.npz, straight.json: the column quantities of the straight rays
  (column_stats) and their terms at the straight warp with the weights of
  each form (with alignment).
- <run>_costs.json: the form, alignment, the stop reason, the run time, the
  weights, the record of each weight (bent_fit.start_weights: the term at
  the straight rays and at p_ref, and which set the weight), and the record
  of each round (bent_fit.fit's history).
- <run>_paths.npz: the coefficients (params), the free lengths, the last
  accepted warp (s_lo, tau, r) and the column quantities (column_stats).

<run> is the arm's name, or for the weight scan (--scale TERM:FACTOR ...)
<arm>_<TERM>_x<FACTOR>: the start weights of the rule with the weight
of TERM multiplied by FACTOR (scaled_weights). bent_fit.fit sets the weights
inside; scaled_weights replaces bent_fit.start_weights for the duration of
one fit with a function that calls it and multiplies one weight, so the fit
itself is bent_fit's, unchanged.

wrong
-----
Cortex voxels that are dark in all three echoes: z < -WRONG_Z in every
echo, with z the detrended log echo over its robust SD, computed as
bent_texture.outliers computes it (orient.local_level over the cortex label
with orient.SIGMA, on the box grown by orient's reach as bent_texture --grow
reads it; robust SD = 1.4826 MAD over the grown cortex). Their connected
pieces (26 neighbours) of at least WRONG_MIN voxels (counted on the grown
box) are kept: likely outside medium that the labels call cortex. Writes
DIR/wrong_labels.npz (mask and piece index on the box grid) and
DIR/wrong_labels.json (the pieces: size, a voxel in the box, the robust SDs
against those of the texture json).

sparse
------
Phase 6 step 5: evaluation only, no refit. At the straight rays and at each
run's final paths (with the run's last warp; the straight rays with the
straight warp), every term of bent_terms.TERMS is evaluated

- dense: bent_terms.total (step 0.04 mm to the cap), and the same terms by
  this module's grid evaluator (eval_grid) on the dense grid, as a check of
  the evaluator;
- sparse 1: one sample per voxel cell [k H, (k + 1) H), H = SPARSE_MM =
  0.16 mm, at k H + o with one random offset o in [0, H) per evaluation,
  the same depths for every column (so the columns of an edge or a face
  share them); each sample weighs H (the jittered rectangle rule: its
  expectation over o is the integral);
- sparse 2: as sparse 1, but in the cells within WINDOW_MM of the column's
  sparse-1 top, and in the two cells round each change of the nearest-voxel
  label class (outside / WM / cortex / other) between the samples of two
  neighbouring cells, the cell is sampled at its four dense midpoints
  (k H + (j + 1/2) STEP, weight STEP) in place of the one jittered sample.

  The depths then differ between columns, so the shear of cost 4 (which
  compares neighbours at equal s) is taken from sparse 1 of the same seed.

Each sparse grid ends, column by column, MARGIN_MM above its own top (the
first sample with mask < 0.5): the mask is set to 0 beyond. sparse1_full is
sparse 1 without that end (to the cap), to separate the effect of the
sampling from that of the end. A column with no
such sample is censored. The censored set and the free lengths of cost 3
are each evaluation's own censored set and the run's free lengths (the
straight rays: their soft lengths). SEEDS seeds (0 .. SEEDS - 1) for the
offsets.

Per column: the top (the arc length at which the mask crosses 0.5, linear
between samples) and the return flag (RETURN_FLAG: the largest
g relu(W - W_min) of cost 6 along the samples with mask >= 0.5 is at least
0.25), dense against sparse.

Cost 4 also in closed form: the integral of |d''|^2 over [0, S_MAX] and
over [0, top] (dense top) for each column, by Simpson's rule on each knot
interval (exact: d'' is linear between knots), against the sampled sums.

Writes DIR/sparse.json and DIR/sparse_columns.npz.

scan-table
----------
The table of the weight-scan runs in DIR (and of the arms): per run the
total, each weighted term (after the last accepted round), the median and
p90 of the largest shear of the inner columns, the censored count of the
inner columns, and the median and p90 of L - L_straight over the inner
columns not censored in either. Writes DIR/scan.md and DIR/scan.json.
"""

import argparse
import json
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.interpolate import BSpline
from scipy.ndimage import label as ndi_label

sys.path.insert(0, str(Path(__file__).parent))
import bent_box as BB  # noqa: E402
import bent_fit as BF  # noqa: E402
import bent_terms as BT  # noqa: E402
from bent_sample import trilinear  # noqa: E402

from baillarger import flattening as FLT  # noqa: E402
from baillarger import orient as O  # noqa: E402

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

#: the settings of phase 5
ROUNDS = 4
STEPS = 15
SEED = 0
ARMS = BF.ARMS
#: the wrong-label rule: z below -WRONG_Z in every echo, pieces of at least
#: WRONG_MIN voxels
WRONG_Z = 5.0
WRONG_MIN = 200
#: sparse depths: the cell, the window round the top, the margin above the
#: top, the seeds
SPARSE_MM = 0.16
WINDOW_MM = 0.32
MARGIN_MM = 0.48
SEEDS = 10
#: the return flag: the largest gated rise of W over its running minimum
RETURN_FLAG = 0.25
#: the label classes of sparse 2 (nearest voxel)
CLS_OUTSIDE, CLS_WM, CLS_CORTEX, CLS_OTHER = 0, 1, 2, 3
#: the columns per chunk of the per-column trend matrices
CHUNK = 1024


def say(msg: str) -> None:
    BF.say(msg)


# ---------------------------------------------------------------- the problem


def load_problem(
    box_path: Path, texture_path: Path, device: str = "cpu"
) -> tuple[BB.Box, BF.Problem]:
    """The box and the fit's Problem (module docstring)."""
    b = BB.load(box_path)
    with np.load(texture_path) as z:
        lead = np.asarray(z["leading"], np.float64)
        c = np.asarray(z["c"], np.float64)
    if lead.shape[:3] != b.labels.shape:
        raise ValueError(f"texture {lead.shape[:3]} is not on the box {b.labels.shape}")
    al = BT.align_field(lead, c, b.affine, b.voxel_mm)
    inp = BT.Inputs.from_box(
        b, device, scales=np.ones(len(BT.FEATURE_ECHOES)), dtype=torch.float32, align=al
    )
    return b, BF.make_problem(inp)


@contextmanager
def scaled_weights(term: str | None, factor: float) -> Iterator[None]:
    """For the duration of the block, bent_fit.start_weights gives the
    weights of its rule with the weight of term multiplied by factor (and
    the record says so). term None: no change."""
    orig = BF.start_weights
    if term is None:
        yield
        return
    if term not in BT.TERMS:
        raise ValueError(f"unknown term {term!r}")

    def sw(
        prob: BF.Problem, levels: BT.Levels, form: str, seed: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        w, rec = orig(prob, levels, form, seed)
        if term not in w:
            raise ValueError(f"term {term!r} has no weight")
        rec[term] = dict(rec[term], rule_weight=w[term], scan_factor=factor)
        w[term] *= factor
        rec[term]["weight"] = w[term]
        return w, rec

    BF.start_weights = sw
    try:
        yield
    finally:
        BF.start_weights = orig


# ---------------------------------------------------------------- columns


def nearest_labels(labels: npt.NDArray[np.int32], vox: F64) -> I64:
    """The label at the nearest voxel of each point (..., 3); -1 outside the
    box."""
    i = np.rint(vox).astype(np.int64)
    sh = np.asarray(labels.shape)
    ok = np.all((i >= 0) & (i < sh), -1)
    j = np.clip(i, 0, sh - 1)
    out = np.where(ok, labels[j[..., 0], j[..., 1], j[..., 2]], -1)
    return out.astype(np.int64)


def label_class(lab: I64, filled: I64, cortex: int) -> I64:
    """outside (0), WM (filled), cortex, other (incl. off the box)."""
    out = np.full(lab.shape, CLS_OTHER, np.int64)
    out[lab == 0] = CLS_OUTSIDE
    out[np.isin(lab, filled)] = CLS_WM
    out[lab == cortex] = CLS_CORTEX
    return out


def _angle(a: F64, b: F64) -> F64:
    d = np.clip(np.sum(a * b, -1), -1.0, 1.0)
    out: F64 = np.degrees(np.arccos(d))
    return out


def column_stats(
    params: torch.Tensor, inputs: BT.Inputs, box: BB.Box, wrong: Bool | None = None
) -> dict[str, Any]:
    """The column quantities of the paths of params (each (n,)):

    - length: the soft length L; censored: mask >= 0.5 at the cap.
    - top_arc: the arc length at which the mask crosses 0.5 (linear between
      samples), NaN if censored; top_mm (n, 3): the path's point there.
    - hard_top: the arc length to the first sample whose nearest-voxel
      label is 0 (outside), NaN if none.
    - early_outside: the path meets a label-0 sample at least one voxel
      (0.16 mm) of arc before its soft top, or it is censored and meets one.
    - returns: among the samples with mask >= 0.5 (all samples if
      censored), a sample with a filled (WM) label after one that is not;
      return_arc: the arc length of the first such sample.
    - shear: bent_fit.shear_mm; closest: bent_terms.closest_outside.
    - root_angle: the angle between the tangent at s = 0 and the normal;
      top_angle: between the tangent at the top sample and the normal;
      turning: the sum of the angles between consecutive tangents up to the
      top sample (degrees).
    - wrong: (with the wrong-label mask) a sample with mask >= 0.5 (all
      samples if censored) in the mask.
    - ns: the count of samples with mask >= 0.5."""
    with torch.no_grad():
        p = BT.paths(params, inputs)
        top = BT.soft_top(p, inputs)
        clo = BT.closest_outside(p, inputs)
    m = top.mask.double().cpu().numpy()
    cen = top.censored.cpu().numpy()
    sp = p.speed.double().cpu().numpy()
    arc = np.concatenate(
        [np.zeros((len(sp), 1)), np.cumsum((sp[:, 1:] + sp[:, :-1]) / 2 * BT.STEP, 1)],
        1,
    )
    vox = p.vox.double().cpu().numpy()
    x = p.x.double().cpu().numpy()
    tg = p.tangent.double().cpu().numpy()
    n = len(m)
    rows = np.arange(n)
    below = m < 0.5
    k = np.where(below.any(1), below.argmax(1), m.shape[1] - 1)
    k0 = np.maximum(k - 1, 0)
    m0, m1 = m[rows, k0], m[rows, k]
    f = np.where(m0 > m1, (m0 - 0.5) / np.where(m0 > m1, m0 - m1, 1.0), 0.0)
    f = np.clip(f, 0.0, 1.0)
    top_arc = np.where(cen, np.nan, arc[rows, k0] + f * (arc[rows, k] - arc[rows, k0]))
    top_mm = np.where(
        cen[:, None], np.nan, x[rows, k0] + f[:, None] * (x[rows, k] - x[rows, k0])
    )
    lab = nearest_labels(box.labels, vox)
    out0 = lab == 0
    hard = np.where(out0.any(1), arc[rows, out0.argmax(1)], np.nan)
    inside_top = (m >= 0.5) | cen[:, None]
    wm = np.isin(lab, box.filled)
    left = np.maximum.accumulate(~wm & inside_top, 1)
    ret = wm & inside_top & np.concatenate([np.zeros((n, 1), bool), left[:, :-1]], 1)
    returns = ret.any(1)
    return_arc = np.where(returns, arc[rows, ret.argmax(1)], np.nan)
    early = np.where(
        cen, np.isfinite(hard), np.isfinite(hard) & (hard < top_arc - SPARSE_MM)
    )
    nrm = inputs.normal.double().cpu().numpy()
    kt = np.where(cen, m.shape[1] - 1, k0)
    dots = np.clip(np.sum(tg[:, 1:] * tg[:, :-1], -1), -1.0, 1.0)
    step_ang = np.degrees(np.arccos(dots))
    upto = np.arange(m.shape[1] - 1)[None] < kt[:, None]
    out: dict[str, Any] = dict(
        length=top.length.double().cpu().numpy(),
        censored=cen,
        top_arc=top_arc,
        top_mm=top_mm,
        hard_top=hard,
        early_outside=early,
        returns=returns,
        return_arc=return_arc,
        shear=BF.shear_mm(params, inputs),
        closest=clo.double().cpu().numpy(),
        root_angle=_angle(tg[:, 0], nrm),
        top_angle=_angle(tg[rows, kt], nrm),
        turning=(step_ang * upto).sum(1),
        ns=(m >= 0.5).sum(1),
    )
    if wrong is not None:
        w = nearest_labels(wrong.astype(np.int32), vox) == 1
        out["wrong"] = (w & inside_top).any(1)
    return out


def load_wrong(out: Path) -> Bool | None:
    q = out / "wrong_labels.npz"
    if not q.exists():
        return None
    with np.load(q) as z:
        r: Bool = np.asarray(z["mask"], bool)
    return r


# ---------------------------------------------------------------- fit


def _warp_arrays(w: FLT.Warp) -> dict[str, Any]:
    return dict(warp_s_lo=np.float64(w.s_lo), warp_tau=w.tau, warp_r=w.r)


def load_warp(z: Mapping[str, Any]) -> FLT.Warp:
    return FLT.Warp(
        float(z["warp_s_lo"]),
        np.asarray(z["warp_tau"], np.float64),
        np.asarray(z["warp_r"], np.float64),
        0,
        [],
    )


def _stats_arrays(st: Mapping[str, Any]) -> dict[str, Any]:
    return {f"col_{k}": np.asarray(v) for k, v in st.items()}


def run_fit(
    box: BB.Box,
    prob: BF.Problem,
    arms: Sequence[str],
    out: Path,
    rounds: int,
    steps: int,
    seed: int,
    scales: Sequence[tuple[str, float] | None],
) -> None:
    """Fit each arm with each scale (None: the rule's weights). A run whose
    costs file exists is skipped. The straight warp is read from
    straight.npz when it exists (the fit is deterministic: it is the same
    warp), else fitted and saved there."""
    inp = prob.inputs
    wrong = load_wrong(out)
    t0 = time.time()
    if (out / "straight.npz").exists():
        with np.load(out / "straight.npz") as z:
            w0 = load_warp(z)
        say("straight warp: read from straight.npz")
    else:
        w0 = BF.fit_warp(prob, BF.zeros(inp))
        say(f"straight warp: {time.time() - t0:.1f} s")
    if not (out / "straight.npz").exists():
        lv = BT.Levels.from_warp(inp, w0)
        p0 = BF.zeros(inp)
        with torch.no_grad():
            cen = BT.soft_top(BT.paths(p0, inp), inp).censored
        rec: dict[str, Any] = dict(columns=inp.n, inner=int(box.inner.sum()))
        for form in ("A", "B"):
            w, wrec = BF.start_weights(prob, lv, form, seed)
            with torch.no_grad():
                t = BF.terms(p0, prob, lv, form, cen, None, w)
            rec[form] = dict(
                weights=w,
                weights_record=wrec,
                terms={k: float(v) for k, v in t.items()},
            )
        st = column_stats(p0, inp, box, wrong)
        rec["censored"] = int(st["censored"].sum())
        rec["censored_inner"] = int((st["censored"] & box.inner).sum())
        np.savez_compressed(
            out / "straight.npz",
            params=p0.numpy(),
            free_length=st["length"],
            **_warp_arrays(w0),
            **_stats_arrays(st),
        )
        (out / "straight.json").write_text(json.dumps(rec, indent=1))
        say(f"straight: {rec['censored_inner']} inner columns censored")
    for arm, scale in [(a, sc) for a in arms for sc in scales]:
        form, al = ARMS[arm]
        name = arm if scale is None else f"{arm}_{scale[0]}_x{scale[1]:g}"
        if (out / f"{name}_costs.json").exists():
            say(f"{name}: exists, skipped")
            continue
        t1 = time.time()
        with scaled_weights(*(scale if scale else (None, 1.0))):
            f = BF.fit(
                prob,
                form,
                al,
                rounds,
                steps,
                seed,
                log=lambda m, a=name: say(f"  {a}: {m}"),
                warp0=w0,
            )
        sec = time.time() - t1
        ev = sum(int(h.get("evaluations", 0)) for h in f.history[1:])
        hist = [dict(f.history[0], weights=f.weights), *f.history[1:]]
        (out / f"{name}_costs.json").write_text(
            json.dumps(
                dict(
                    run=name,
                    arm=arm,
                    form=form,
                    alignment=al,
                    scale=None
                    if scale is None
                    else dict(term=scale[0], factor=scale[1]),
                    rounds=rounds,
                    steps=steps,
                    seed=seed,
                    stop=f.stop,
                    seconds=sec,
                    evaluations=ev,
                    seconds_per_evaluation=sec / max(ev, 1),
                    weights=f.weights,
                    weights_record=f.history[0]["weights_record"],
                    history=hist,
                ),
                indent=1,
            )
        )
        st = column_stats(f.params, inp, box, wrong)
        np.savez_compressed(
            out / f"{name}_paths.npz",
            params=f.params.numpy(),
            free_length=f.free_length.numpy(),
            **_warp_arrays(f.warp),
            **_stats_arrays(st),
        )
        say(
            f"{name}: stop {f.stop}, {sec:.0f} s, {ev} evaluations, censored inner "
            f"{int((st['censored'] & box.inner).sum())}, shear max "
            f"{float(st['shear'].max()):.3f} mm"
        )


# ---------------------------------------------------------------- wrong labels


def run_wrong(box: BB.Box, box_path: Path, texture_json: Path, out: Path) -> None:
    BTX: Any = __import__("bent_texture")
    side = json.loads((box_path.parent / BB.SIDECAR).read_text())
    reach = O.orient_reach(box.voxel_mm, O.SIGMA)
    g = BTX.read_grown(box, side, reach)
    ctx = g.labels == box.cortex
    lv = np.asarray(O.local_level(g.echoes, ctx, O.SIGMA, box.voxel_mm))
    det = np.asarray(O.log_echo(g.echoes)) - lv
    use = ctx & np.all(np.isfinite(det), 0)
    sd = np.array(
        [BTX.MAD_SD * float(np.median(np.abs(d[use] - np.median(d[use])))) for d in det]
    )
    z = np.nan_to_num(det / sd[:, None, None, None], nan=0.0)
    dark = use & np.all(z < -WRONG_Z, 0)
    pieces, npc = ndi_label(dark, structure=np.ones((3, 3, 3), bool))
    size = np.bincount(pieces.ravel(), minlength=npc + 1)
    big = np.flatnonzero(size >= WRONG_MIN)
    big = big[big > 0]
    keep = np.isin(pieces, big)
    crop = g.crop
    mask = keep[crop]
    pc = np.where(mask, pieces[crop], 0)
    tj = json.loads(texture_json.read_text())
    rec: dict[str, Any] = dict(
        rule=f"cortex, z < -{WRONG_Z} in all echoes, 26-connected pieces of at "
        f">= {WRONG_MIN} voxels (counted on the grown box)",
        robust_sd=[round(float(x), 4) for x in sd],
        texture_outlier_sd=tj.get("outlier_sd"),
        dark_voxels_grown=int(dark.sum()),
        pieces_total=int(npc),
        pieces=[],
        mask_voxels_in_box=int(mask.sum()),
    )
    for q in big:
        inb = pc == q
        vox = np.argwhere(inb)
        rec["pieces"].append(
            dict(
                piece=int(q),
                voxels=int(size[q]),
                voxels_in_box=int(inb.sum()),
                box_voxel_example=vox[len(vox) // 2].tolist() if len(vox) else None,
                box_voxel_min=vox.min(0).tolist() if len(vox) else None,
                box_voxel_max=vox.max(0).tolist() if len(vox) else None,
            )
        )
    ref = np.array([59, 84, 10])
    rec["piece_at_59_84_10"] = int(pc[tuple(ref)])
    near = np.argwhere(mask)
    rec["nearest_mask_voxel_to_59_84_10"] = (
        near[np.argmin(np.linalg.norm(near - ref, axis=1))].tolist()
        if len(near)
        else None
    )
    np.savez_compressed(out / "wrong_labels.npz", mask=mask, piece=pc.astype(np.int32))
    (out / "wrong_labels.json").write_text(json.dumps(rec, indent=1))
    say(
        f"wrong labels: {len(big)} pieces, {int(mask.sum())} voxels in the box; "
        f"robust SD {sd.round(4).tolist()} (texture {tj.get('outlier_sd')})"
    )


# ---------------------------------------------------------------- sparse depths


def basis_at(s: F64) -> tuple[F64, F64, F64]:
    """bent_terms.basis at the depths s (any shape): (..., K) each, the same
    splines and the same constraint d(0) = 0 (value exactly 0 at s = 0)."""
    K = BT.K
    m = K + 1
    t = BT.KNOT_MM * (np.arange(m + 4) - 3.0)
    spl = BSpline(t, np.eye(m), 3, extrapolate=True)
    z = np.zeros((m, K))
    z[0, 0] = 1.0
    z[1, 0] = z[1, 1] = -0.25
    z[2:, 1:] = np.eye(K - 1)
    out = [
        np.asarray(spl.derivative(d)(s) if d else spl(s), np.float64) @ z
        for d in range(3)
    ]
    out[0][s == 0] = 0.0
    return out[0], out[1], out[2]


def paths_at(params: torch.Tensor, inputs: BT.Inputs, S: F64) -> BT.Paths:
    """bent_terms.paths at the depths S: (M,) shared or (n, M) per column.
    Paths.s is (n, M)."""
    dt, dev = inputs.dtype, inputs.device
    b0, b1, b2 = (torch.as_tensor(x).to(dt).to(dev) for x in basis_at(S))
    c = params * inputs.moving[:, None, None]
    if S.ndim == 1:
        a, b = c[:, 0] @ b0.T, c[:, 1] @ b0.T
        a1, b1_ = c[:, 0] @ b1.T, c[:, 1] @ b1.T
        a2, b2_ = c[:, 0] @ b2.T, c[:, 1] @ b2.T
        s = torch.as_tensor(S).to(dt).to(dev)[None].expand(inputs.n, -1)
    else:

        def ev(ci: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return torch.einsum("nk,nmk->nm", ci, B)

        a, b = ev(c[:, 0], b0), ev(c[:, 1], b0)
        a1, b1_ = ev(c[:, 0], b1), ev(c[:, 1], b1)
        a2, b2_ = ev(c[:, 0], b2), ev(c[:, 1], b2)
        s = torch.as_tensor(S).to(dt).to(dev)
    e1, e2, nrm = inputs.e1[:, None], inputs.e2[:, None], inputs.normal[:, None]
    d = a[..., None] * e1 + b[..., None] * e2
    y = s[..., None] * nrm + d
    vox = inputs.root_vox[:, None] + y @ inputs.lin.T
    f = trilinear(inputs.volumes, vox)
    speed = torch.sqrt(1 + a1 * a1 + b1_ * b1_)
    dx = nrm + a1[..., None] * e1 + b1_[..., None] * e2
    return BT.Paths(
        s=s,
        y=y,
        d=d,
        speed=speed,
        curv2=a2 * a2 + b2_ * b2_,
        tangent=dx / speed[..., None],
        vox=vox,
        outside=f[0],
        wm=f[1],
        cortex=f[2],
        echo=f[3:],
        root=inputs.root,
    )


def levels_at(inputs: BT.Inputs, warp: FLT.Warp, S: F64) -> BT.Levels:
    """bent_terms.Levels of warp read on the depths S ((M,) or (n, M), each
    row non-decreasing): linear between the two samples round the level,
    the first (last) sample's value below (above) them."""
    n = inputs.n
    S2 = np.broadcast_to(S, (n, S.shape[-1])) if S.ndim == 1 else S
    M = S2.shape[1]
    q = float(warp.s_lo) + np.asarray(warp.tau)[None] + np.asarray(warp.r)
    valid = (q >= 0) & (q <= BT.S_MAX)
    big = 4 * (BT.S_MAX + 10.0)
    row = np.arange(n)[:, None]
    flat = (S2 + row * big).ravel()
    qq = np.clip(q, 0, BT.S_MAX) + row * big
    lo = np.searchsorted(flat, qq.ravel(), side="right").reshape(q.shape) - 1 - row * M
    lo = np.clip(lo, 0, M - 2)
    sl = np.take_along_axis(S2, lo, 1)
    sh = np.take_along_axis(S2, lo + 1, 1)
    wd = sh - sl
    frac = np.clip(np.where(wd > 0, (q - sl) / np.where(wd > 0, wd, 1.0), 0.0), 0, 1)
    dev, dt = inputs.device, inputs.dtype
    rows = n * M
    return BT.Levels(
        s=torch.as_tensor(q).to(dt).to(dev),
        valid=torch.as_tensor(valid.astype(np.float64)).to(dt).to(dev),
        frac=torch.as_tensor(frac).to(dt).to(dev),
        lo=BT.Gather(row * M + lo, rows, dev),
        hi=BT.Gather(row * M + lo + 1, rows, dev),
    )


def _trend_kernel(ds: F64) -> F64:
    """g(ds) normalized as bent_terms' trend matrix (the discrete kernel of
    sigma TREND_MM / STEP samples, truncated at 4 sigma, sum 1 on the
    dense grid)."""
    sig = BT.TREND_MM / BT.STEP
    r = int(4.0 * sig + 0.5)
    xs = np.arange(-r, r + 1, dtype=np.float64)
    norm = np.exp(-0.5 * (xs / sig) ** 2).sum()
    k = ds / BT.STEP
    out: F64 = np.where(
        np.abs(k) <= r + 1e-9, np.exp(-0.5 * (k / sig) ** 2) / norm, 0.0
    )
    return out


def features_at(
    p: BT.Paths, mask: torch.Tensor, inputs: BT.Inputs, S: F64, wt: F64
) -> torch.Tensor:
    """bent_terms.features on a grid: the trend is the sum over the samples
    j of g(s_i - s_j) wt_j w_j L_j over the same sum without L (wt_j: the
    sample's weight in units of STEP; 1 on the dense grid, where it is
    bent_terms' matrix up to its end samples, which there weigh 1/2)."""
    L = torch.log(p.echo.clamp(min=1.0))
    w = mask * p.cortex
    dt, dev = inputs.dtype, inputs.device
    if S.ndim == 1:
        G = (
            torch.as_tensor(_trend_kernel(S[:, None] - S[None]) * wt[None])
            .to(dt)
            .to(dev)
        )
        num = (w[None] * L) @ G.T
        den = (w @ G.T).clamp(min=BT.TREND_DEN)
    else:
        num = torch.empty_like(L)
        den = torch.empty_like(w)
        for i in range(0, inputs.n, CHUNK):
            sl = slice(i, i + CHUNK)
            Gn = _trend_kernel(S[sl, :, None] - S[sl, None, :]) * wt[sl, None, :]
            G = torch.as_tensor(Gn).to(dt).to(dev)
            num[:, sl] = torch.einsum("nij,enj->eni", G, w[None, sl] * L[:, sl])
            den[sl] = torch.einsum("nij,nj->ni", G, w[sl]).clamp(min=BT.TREND_DEN)
    out: torch.Tensor = (
        p.cortex[None] * (L - num / den[None]) / inputs.scales[:, None, None]
    )
    return out


@dataclass
class GridEval:
    """The terms of one evaluation on a grid, and per column: the top (arc
    length at mask 0.5, NaN if censored), the censored flag, the return
    flag, the return cost and the mask-weighted bending."""

    terms: dict[str, float]
    top: F64
    censored: Bool
    returns: Bool
    return_cost: F64
    bending: F64


def eval_grid(
    params: torch.Tensor,
    inputs: BT.Inputs,
    warp: FLT.Warp,
    form: str,
    free_length: torch.Tensor,
    S: F64,
    ds: F64,
    truncate: bool,
    weights: Mapping[str, float],
    shear_from: float | None = None,
) -> GridEval:
    """Every term of bent_terms.total on the grid S ((M,) or (n, M)) with
    the integration weights ds (same shape). truncate: the mask is 0 beyond
    MARGIN_MM above the first sample with mask < 0.5. The censored set is
    the evaluation's own. shear_from: the shear term to report in place of
    this grid's (for a per-column grid, see the module docstring)."""
    n = inputs.n
    dt, dev = inputs.dtype, inputs.device
    with torch.no_grad():
        p = paths_at(params, inputs, S)
        m = 1 - BT.running_max(p.outside.clamp(0, 1))
        S2 = np.broadcast_to(S, (n, S.shape[-1])) if S.ndim == 1 else S
        # the padding of a per-column grid (weight 0, beyond S_MAX) is not
        # a sample: no top, no return there, mask 0
        real = np.broadcast_to(np.asarray(ds) > 0, S2.shape)
        m = m * torch.as_tensor(real.astype(np.float64)).to(dt).to(dev)
        mn = m.double().cpu().numpy()
        below = (mn < 0.5) & real
        has = below.any(1)
        k = np.where(has, below.argmax(1), S2.shape[1] - 1)
        rows = np.arange(n)
        s_top = np.where(has, S2[rows, k], np.inf)
        if truncate:
            keep = S2 <= s_top[:, None] + MARGIN_MM
            m = m * torch.as_tensor(keep.astype(np.float64)).to(dt).to(dev)
        cen_np = ~has
        cen = torch.as_tensor(cen_np, device=dev)
        dsw = torch.as_tensor(np.broadcast_to(ds, S2.shape).copy()).to(dt).to(dev)
        length = (m * p.speed * dsw).sum(-1)
        lv = levels_at(inputs, warp, S)
        wt = np.asarray(ds, np.float64) / BT.STEP
        feat = features_at(p, m, inputs, S, wt)
        barrier, crowd = BT.cost_crossing(p, m, lv, inputs, BF.EXTEND, True)
        wmin = BT.running_min(p.wm)
        g = (1 - 2 * wmin).clamp(0, 1)
        r = torch.relu(p.wm - wmin)
        ret_c = (m * g * r * r * dsw).sum(-1)
        rise = torch.where(m >= 0.5, g * r, torch.zeros_like(r)).amax(-1)
        bend_c = (m * p.curv2 * dsw).sum(-1)
        if S.ndim == 1:
            d0, d1 = inputs.edge0(p.d), inputs.edge1(p.d)
            w = inputs.edge0(m) * inputs.edge1(m)
            dse = torch.as_tensor(np.asarray(ds, np.float64)).to(dt).to(dev)
            shear = float((w * dse * ((d0 - d1) ** 2).sum(-1)).sum())
        else:
            shear = float("nan")
        if shear_from is not None:
            shear = shear_from
        out: dict[str, float] = dict(
            agreement=float(BT.cost_agreement(feat, m, lv, inputs)),
            length=float(BT.cost_length(length, cen)),
            length_agreement=float(
                BT.cost_length_agreement(length, cen, free_length, inputs, form)
            ),
            bending=float(bend_c.sum()),
            shear=shear,
            barrier=float(barrier),
            crowding=float(crowd),
            **{"return": float(ret_c.sum())},
        )
        if inputs.align is not None:
            A = trilinear(inputs.align, p.vox)
            t = p.tangent
            q = torch.zeros_like(p.speed)
            for kk, (i, j) in enumerate(BT.PAIRS):
                q = q + (1.0 if i == j else 2.0) * A[kk] * t[..., i] * t[..., j]
            out["alignment"] = float((m * q * dsw).sum())
        out["total"] = float(
            sum(
                float(weights.get(kk, 0.0)) * v
                for kk, v in out.items()
                if kk != "total"
            )
        )
        # the top as an arc length at mask 0.5 (linear between samples)
        sp = p.speed.double().cpu().numpy()
        darc = (sp[:, 1:] + sp[:, :-1]) / 2 * np.diff(S2, axis=1)
        arc = np.concatenate([np.zeros((n, 1)), np.cumsum(darc, 1)], 1)
        mm = (1 - BT.running_max(p.outside.clamp(0, 1))).double().cpu().numpy()
        k0 = np.maximum(k - 1, 0)
        m0, m1 = mm[rows, k0], mm[rows, k]
        f = np.clip(
            np.where(m0 > m1, (m0 - 0.5) / np.where(m0 > m1, m0 - m1, 1), 0), 0, 1
        )
        top = arc[rows, k0] + f * (arc[rows, k] - arc[rows, k0])
        # the arc from s = 0 to the first sample (s > 0 on a sparse grid):
        # its depth times its speed
        top = np.where(cen_np, np.nan, top + S2[:, 0] * sp[:, 0])
    return GridEval(
        out,
        top,
        cen_np,
        rise.double().cpu().numpy() >= RETURN_FLAG,
        ret_c.double().cpu().numpy(),
        bend_c.double().cpu().numpy(),
    )


def dense_grid() -> tuple[F64, F64]:
    s: F64 = np.arange(BT.NS, dtype=np.float64) * BT.STEP
    ds: F64 = np.full(BT.NS, BT.STEP, np.float64)
    ds[0] = ds[-1] = BT.STEP / 2
    return s, ds


def sparse_grid(rng: np.random.Generator) -> tuple[F64, F64]:
    """Sparse 1: one jittered sample per cell, weight SPARSE_MM."""
    o = float(rng.uniform(0.0, SPARSE_MM))
    nc = int(round(BT.S_MAX / SPARSE_MM))
    s: F64 = np.arange(nc, dtype=np.float64) * SPARSE_MM + o
    ds: F64 = np.full(nc, SPARSE_MM, np.float64)
    return s, ds


def windowed_grid(s1: F64, top1: F64, labels_cls: I64) -> tuple[F64, F64]:
    """Sparse 2 (module docstring): per column, the cells of sparse 1 within
    WINDOW_MM of its sparse-1 top (arc length, NaN if censored: no window)
    and round its label changes get four dense midpoints. Returns S, ds
    (n, M), padded beyond S_MAX with weight 0."""
    n, nc = labels_cls.shape
    h = SPARSE_MM
    cell0 = np.arange(nc) * h
    near = np.abs(cell0[None] + h / 2 - np.nan_to_num(top1, nan=-1e9)[:, None]) <= (
        WINDOW_MM + h / 2
    )
    ch = labels_cls[:, 1:] != labels_cls[:, :-1]
    dense = near.copy()
    dense[:, :-1] |= ch
    dense[:, 1:] |= ch
    sub = int(round(h / BT.STEP))
    mids = (np.arange(sub) + 0.5) * BT.STEP
    dn = dense[..., None]
    cand = np.where(
        dn, cell0[None, :, None] + mids[None, None], s1[None, :, None] + 0 * mids
    )
    first = (np.arange(sub) == 0)[None, None]
    valid = (dn | first).reshape(n, -1)
    wts = np.where(dn, BT.STEP, h) * np.ones(sub)
    cand = cand.reshape(n, -1)
    wts = wts.reshape(n, -1)
    M = int(valid.sum(1).max())
    S = np.empty((n, M))
    D = np.zeros((n, M))
    cnt = valid.sum(1)
    S[:] = BT.S_MAX + BT.STEP * (1 + np.arange(M))[None]
    idx = np.cumsum(valid, 1) - 1
    r, cidx = np.nonzero(valid)
    S[r, idx[r, cidx]] = cand[r, cidx]
    D[r, idx[r, cidx]] = wts[r, cidx]
    del cnt
    return S, D


def bending_closed(
    params: torch.Tensor, inputs: BT.Inputs, top: F64
) -> tuple[F64, F64]:
    """The integral of |d''|^2 per column over [0, S_MAX] and over
    [0, top] (top NaN: S_MAX), exact by Simpson on each knot interval."""
    c = (params * inputs.moving[:, None, None]).double().cpu().numpy()
    n = len(c)
    knots = np.arange(0.0, BT.S_MAX + 1e-9, BT.KNOT_MM)

    def f2(S: F64) -> F64:
        _, _, B2 = basis_at(S)
        if S.ndim == 1:
            a = c[:, 0] @ B2.T
            b = c[:, 1] @ B2.T
        else:
            a = np.einsum("nk,nmk->nm", c[:, 0], B2)
            b = np.einsum("nk,nmk->nm", c[:, 1], B2)
        out: F64 = a * a + b * b
        return out

    lo, hi = knots[:-1], knots[1:]
    S = np.stack([lo, (lo + hi) / 2, hi], 1).ravel()
    v = f2(S).reshape(n, len(lo), 3)
    full = ((hi - lo) / 6 * (v[..., 0] + 4 * v[..., 1] + v[..., 2])).sum(1)
    tt = np.where(np.isfinite(top), np.clip(top, 0, BT.S_MAX), BT.S_MAX)
    lo2 = np.minimum(lo[None], tt[:, None])
    hi2 = np.minimum(hi[None], tt[:, None])
    S2 = np.stack([lo2, (lo2 + hi2) / 2, hi2], 2).reshape(n, -1)
    v2 = f2(S2).reshape(n, len(lo), 3)
    upto = ((hi2 - lo2) / 6 * (v2[..., 0] + 4 * v2[..., 1] + v2[..., 2])).sum(1)
    return full, upto


def _stat(x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(x, np.float64)
    return dict(
        mean=float(a.mean()),
        sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        min=float(a.min()),
        max=float(a.max()),
    )


def run_sparse(
    box: BB.Box, prob: BF.Problem, runs: Sequence[str], out: Path, seeds: int
) -> None:
    inp = prob.inputs
    res: dict[str, Any] = dict(
        settings=dict(
            cell_mm=SPARSE_MM,
            window_mm=WINDOW_MM,
            margin_mm=MARGIN_MM,
            seeds=seeds,
            return_flag=RETURN_FLAG,
        ),
        runs={},
    )
    cols: dict[str, Any] = {}
    inner = box.inner
    for run in runs:
        t0 = time.time()
        q = out / f"{run}_paths.npz" if run != "straight" else out / "straight.npz"
        with np.load(q) as z:
            params = torch.as_tensor(np.asarray(z["params"], np.float32))
            free = torch.as_tensor(np.asarray(z["free_length"], np.float32))
            warp = load_warp(z)
        if run == "straight":
            form, w = (
                "B",
                json.loads((out / "straight.json").read_text())["B"]["weights"],
            )
        else:
            cj = json.loads((out / f"{run}_costs.json").read_text())
            form, w = str(cj["form"]), dict(cj["weights"])
        lv = BT.Levels.from_warp(inp, warp)
        with torch.no_grad():
            cen_d = BT.soft_top(BT.paths(params, inp), inp).censored
            ref = BF.terms(params, prob, lv, form, cen_d, free, w)
        dense_bt = {k: float(v) for k, v in ref.items()}
        sd, dsd = dense_grid()
        dg = eval_grid(params, inp, warp, form, free, sd, dsd, False, w)
        full, upto = bending_closed(params, inp, dg.top)
        p = paths_at(params, inp, sd)
        with torch.no_grad():
            c2 = p.curv2.double().cpu().numpy()
        r: dict[str, Any] = dict(
            form=form,
            weights=w,
            dense=dense_bt,
            dense_grid_evaluator=dg.terms,
            cost4_closed=dict(
                full_range=float(full.sum()),
                to_dense_top=float(upto.sum()),
                sampled_dense_unmasked=float((c2 * dsd[None]).sum()),
                sampled_dense_masked=dense_bt["bending"],
            ),
            censored_dense_inner=int((dg.censored & inner).sum()),
            returns_dense_inner=int((dg.returns & inner).sum()),
        )
        keys = ("sparse1", "sparse1_full", "sparse2")
        acc: dict[str, dict[str, list[Any]]] = {
            k: dict(terms=[], dtop=[], cen=[], ret=[]) for k in keys
        }
        unm1 = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            s1, d1 = sparse_grid(rng)
            e1 = eval_grid(params, inp, warp, form, free, s1, d1, True, w)
            e1f = eval_grid(params, inp, warp, form, free, s1, d1, False, w)
            with torch.no_grad():
                p1 = paths_at(params, inp, s1)
                unm1.append(float((p1.curv2.double().cpu().numpy() * d1[None]).sum()))
                vox1 = p1.vox.double().cpu().numpy()
            cls = label_class(nearest_labels(box.labels, vox1), box.filled, box.cortex)
            S2, D2 = windowed_grid(s1, e1.top, cls)
            e2 = eval_grid(
                params, inp, warp, form, free, S2, D2, True, w, e1.terms["shear"]
            )
            for k, e in zip(keys, (e1, e1f, e2), strict=True):
                acc[k]["terms"].append(e.terms)
                acc[k]["dtop"].append(e.top - dg.top)
                acc[k]["cen"].append(e.censored != dg.censored)
                acc[k]["ret"].append(e.returns != dg.returns)
            say(f"  {run} seed {seed}: sparse 2 width {S2.shape[1]}")
        for key in keys:
            A_ = acc[key]
            tk = A_["terms"][0].keys()
            r[key] = dict(terms={k: _stat([t[k] for t in A_["terms"]]) for k in tk})
            D = np.abs(np.stack(A_["dtop"]))
            Cn = np.stack(A_["cen"])
            Rt = np.stack(A_["ret"])
            fin = np.isfinite(D)
            dmax = np.where(fin.any(0), np.where(fin, D, -np.inf).max(0), np.nan)
            Di = D[:, inner]
            r[key]["top_abs_diff_mm_inner"] = dict(
                median=float(np.nanmedian(Di)),
                p90=float(np.nanpercentile(Di, 90)),
                max=float(np.nanmax(Di)),
            )
            r[key]["columns_top_diff_gt_0.04_any_seed_inner"] = int(
                ((dmax > 0.04) & inner).sum()
            )
            r[key]["columns_top_diff_gt_0.16_any_seed_inner"] = int(
                ((dmax > 0.16) & inner).sum()
            )
            r[key]["columns_censored_differs_any_seed_inner"] = int(
                (Cn.any(0) & inner).sum()
            )
            r[key]["columns_return_differs_any_seed_inner"] = int(
                (Rt.any(0) & inner).sum()
            )
            cols[f"{run}_{key}_top_diff_max"] = dmax
            cols[f"{run}_{key}_censored_differs"] = Cn.any(0)
            cols[f"{run}_{key}_return_differs"] = Rt.any(0)
        r["sparse1"]["cost4_unmasked"] = _stat(unm1)
        r["seconds"] = time.time() - t0
        res["runs"][run] = r
        say(f"sparse {run}: {r['seconds']:.0f} s")
        (out / "sparse.json").write_text(json.dumps(res, indent=1))
        np.savez_compressed(out / "sparse_columns.npz", **cols)


# ---------------------------------------------------------------- scan table


def _q(a: F64) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    if not a.size:
        return float("nan"), float("nan")
    return float(np.median(a)), float(np.percentile(a, 90))


def run_scan_table(box: BB.Box, out: Path) -> None:
    with np.load(out / "straight.npz") as z:
        L0 = np.asarray(z["col_length"])
        c0 = np.asarray(z["col_censored"], bool)
    inner = box.inner
    rows: list[dict[str, Any]] = []
    for q in sorted(out.glob("*_costs.json")):
        cj = json.loads(q.read_text())
        name = cj["run"]
        with np.load(out / f"{name}_paths.npz") as z:
            sh = np.asarray(z["col_shear"])
            cen = np.asarray(z["col_censored"], bool)
            L = np.asarray(z["col_length"])
        acc = [h for h in cj["history"][1:] if h.get("accepted")]
        last = acc[-1] if acc else None
        w = cj["weights"]
        terms = (
            {
                k: float(w.get(k, 0.0)) * float(last["after"][k])
                for k in BT.TERMS
                if k in last["after"]
            }
            if last
            else {}
        )
        use = inner & ~cen & ~c0
        dl = _q(L[use] - L0[use])
        s = _q(sh[inner])
        rows.append(
            dict(
                run=name,
                scale=cj.get("scale"),
                stop=cj["stop"],
                rounds_accepted=len(acc),
                seconds=round(float(cj["seconds"]), 1),
                total=float(last["after"]["total"]) if last else None,
                weighted=terms,
                shear_median=s[0],
                shear_p90=s[1],
                censored_inner=int((cen & inner).sum()),
                dL_median=dl[0],
                dL_p90=dl[1],
            )
        )
    keys = [k for k in BT.TERMS]
    head = (
        "| run | stop | total | "
        + " | ".join(keys)
        + " | shear med / p90 | censored | dL med / p90 | s |"
    )
    md = [head, "|" + "---|" * (len(keys) + 7)]
    for r in rows:
        t = r["weighted"]
        md.append(
            f"| {r['run']} | {r['stop']} | "
            + (f"{r['total']:.3f}" if r["total"] is not None else "-")
            + " | "
            + " | ".join(f"{t[k]:.3g}" if k in t else "-" for k in keys)
            + f" | {r['shear_median']:.3f} / {r['shear_p90']:.3f} | "
            f"{r['censored_inner']} | {r['dL_median']:.3f} / {r['dL_p90']:.3f} | "
            f"{r['seconds']:.0f} |"
        )
    (out / "scan.md").write_text("\n".join(md) + "\n")
    (out / "scan.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md))


# ---------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the arms at the site")
    s = sub.add_parser("sparse", help="dense against sparse depths")
    w = sub.add_parser("wrong", help="the wrong-label pieces")
    t = sub.add_parser("scan-table", help="the table of the runs")
    for p in (f, s, w, t):
        p.add_argument("--box", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    for p in (f, s):
        p.add_argument("--texture", type=Path, required=True)
    f.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    f.add_argument("--rounds", type=int, default=ROUNDS)
    f.add_argument("--steps", type=int, default=STEPS)
    f.add_argument("--seed", type=int, default=SEED)
    f.add_argument(
        "--scale",
        nargs="*",
        metavar="TERM:FACTOR",
        help="weight-scan runs, each TERM:FACTOR (one after the other)",
    )
    s.add_argument("--runs", nargs="*", default=["straight", *ARMS])
    s.add_argument("--seeds", type=int, default=SEEDS)
    w.add_argument("--texture-json", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(True)
    if a.cmd == "wrong":
        run_wrong(BB.load(a.box), a.box, a.texture_json, a.out)
        return
    if a.cmd == "scan-table":
        run_scan_table(BB.load(a.box), a.out)
        return
    t0 = time.time()
    box, prob = load_problem(a.box, a.texture)
    say(
        f"problem: {prob.inputs.n} columns, {int(box.inner.sum())} inner, "
        f"{time.time() - t0:.1f} s"
    )
    if a.cmd == "fit":
        scales: list[tuple[str, float] | None] = (
            [None]
            if not a.scale
            else [(x.split(":")[0], float(x.split(":")[1])) for x in a.scale]
        )
        run_fit(box, prob, a.arms, a.out, a.rounds, a.steps, a.seed, scales)
    else:
        run_sparse(box, prob, a.runs, a.out, a.seeds)
    say(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
