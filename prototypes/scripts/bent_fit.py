"""The alternating fit of the bent-columns trial (docs/plans/bent-columns-
trial.md, phase 5), and its runs on the phantoms.

The fit
-------
The paths are those of bent_terms: the spline coefficients (n, 2, K) of the
outward shear (s >= 0). The roots and the straight inward part stay fixed.
A column with moving = False stays straight (bent_terms.paths).

1. Start from the straight rays (all coefficients 0).
2. Each round:
   a. Sample the profiles along the current paths (sample_profiles): the
      straight inward part from S_LO to 0 and the current path from 0 to
      S_MAX, at bent_terms.STEP. Fit the warp on them with flattening.fit
      (numpy). The warp is a constant of the round.
   b. Fix the censored set (bent_terms.soft_top at the round's start
      paths). With the warp and that set fixed, take at most `steps`
      iterations of torch.optim.LBFGS (strong Wolfe line search) on the
      coefficients and on the free lengths of the censored columns (cost 3:
      a censored column's length is a parameter of its own, started at its
      soft length).
   c. If the round's total after its descent is above the round's total
      before it, the descent is undone (the paths stay those of the start
      of the round).
3. The fit stops after `rounds` rounds, or when a round moves the paths by
   less than MOVE_TOL mm (move_mm), or when a round's total (at its warp,
   after its descent) is above the previous round's total. Such a round is
   recorded with accepted = False and its paths are not kept.

So the record of accepted rounds has totals that do not increase. The
totals of different rounds are at different warps and censored sets: the
fit does not prove that the paths and the warp agree. Each round's record
holds the totals before and after its descent, at its own warp, so a warp
that pulls against the paths is seen there.

The settings fixed before any fit
---------------------------------
- Cost 5 uses the extended barrier, extend = EXTEND = 0.1.
- Cost 1 is not normalized (bent_terms.cost_agreement, normalize False);
  cost 5's crowding is relative (relative True). These are bent_terms'
  defaults.
- The feature scales (bent_terms.Inputs.scales and the scales of
  flattening.features) are flattening.scales_of over the cortex samples of
  the straight profiles, fixed for every round.
- The energy reference of the pair weights (flattening.pair_shifts) is the
  one of the straight profiles, fixed for every round.
- flattening.fit with its defaults (DTAU, EPS), no warm start: the warp of
  a round depends only on the round's paths.

The weights
-----------
start_weights: one rule for every term of bent_terms.TERMS (costs 1-7;
cost 4 is two terms, bending and shear, and cost 5 is two terms, barrier
and crowding):

    w_k = 1 / |T_k(p0)|      if |T_k(p0)| >= ZERO,
    w_k = 1 / |T_k(p_ref)|   if not, and |T_k(p_ref)| >= ZERO,
    w_k = 1                  if not,

with T_k the term at the warp of the straight rays and their censored set,
p0 the straight paths, p_ref the reference perturbation below and
ZERO = 1e-9 (in the term's own units: a term below it is 0 up to float32
rounding; the return term of the paths of crown_w0.32, which do not
re-enter WM, was 6e-29 at p_ref). So every term that is not 0 at the straight rays has
weighted value +-1 there (the barrier can be negative: -log of a face that
grows). A term that is 0 at the start (bending and shear always; return,
and possibly others, on some phantoms) has weighted value +-1 at p_ref.
The weights are fixed before the fit and not changed.
The arm without cost 7 sets the weight of alignment to 0; the other weights
are the same.

The reference perturbation p_ref bends every moving column with a constant
curvature KAPPA_REF = 0.05 / mm (radius 20 mm) in a direction phi_c of its
tangent plane, drawn uniformly at random for each column from
numpy.random.default_rng(seed):

    d_c(s) = KAPPA_REF s^2 / 2 (cos phi_c e1_c + sin phi_c e2_c).

s^2 / 2 is in the space of the cubic splines with d(0) = 0, so its
coefficients are exact (least squares on bent_terms.basis). At s = 2.4 mm
the offset is 0.144 mm, about one voxel of 0.16 mm. The directions of
neighbours are independent, so the shear term at p_ref is that of
uncorrelated neighbours. The seed enters only here: two runs with the same
seed give the same paths.

The phantoms
------------
For each phantom of fold_phantoms.standard() and the curved phantom of
bent_phantom.curved() (phantom_case):

- Image. fold_phantoms.render gives the noise-free log intensity of the two
  feature echoes (echo 2 and 3 of the site's profile). The noise is added as
  fold_phantoms.make adds it (the same generator calls, so the noisy log
  image L is the one whose detrended form make returns; a test checks it).
  The fit's echoes are exp(L + LOG_OFFSET). The offset keeps the echoes
  above 1, where log(max(echo, 1)) of the features is log(echo); a constant
  in the log is removed by the features' trend and by orient's local level.
- Labels: 0 cortex, 1 outside medium (fluid), 2 WM (fold_phantoms). Outside
  = label 1, WM = label 2, cortex = label 0.
- Mesh and roots: column_tracers.white_mesh of the WM label (all its
  vertices are columns, in a slab; see select_columns), voxel coordinates;
  mm = voxel * 0.16. The errors are reported on the roots of
  column_tracers.pick_vertices in the slab's moving part that are at least
  EVAL_DFACE_MM from a box face (fold_eval's rule).
- Cost 7. orient.orient_arrays on the echoes within the cortex label, less
  the dark outliers of bent_texture.outliers (k = OUTLIER_K = 5, as at the
  site), with the phantom's noise model: the phantom's noise is Gaussian on
  the log intensity with the SD sd_e of each echo, independent across
  echoes and constant over the image. In orient's model C = g diag(sigma /
  I) R diag(sigma / I) + C_rest, that is the rest alone: sigma = 0, R = I,
  g = orient.gradient_gain(SD), C_rest = g diag(sd_e^2) (PhantomNoise).
  The tensor is set to NaN where the box does not hold orient's reach
  (bent_texture.valid_mask). orient.leading gives l1 and the coherence;
  bent_terms.align_field gives the field.
- True columns: fold.column (straight) and fold_eval.errors for the fold
  phantoms; bent_phantom.true_columns and bent_phantom.path_errors for the
  curved one.
- A trace for the errors: the samples of a path (voxel coordinates) from
  s = 0 to its last sample with soft mask >= 0.5 (all samples for a
  censored column). The straight rays are traced by the same rule.

The command
-----------
    python prototypes/scripts/bent_fit.py phantoms --out DIR \\
        --features fold_features.json [--site V1V2_banks] [--rounds R] \\
        [--steps N] [--phantoms NAME ...] [--arms A7 A B7 B]
    python prototypes/scripts/bent_fit.py views --out DIR \\
        --features fold_features.json [--phantoms NAME ...]

writes, for each phantom and arm: DIR/<phantom>_<arm>_costs.json (the
weights, the record of each round), DIR/<phantom>_<arm>_paths.npz, and
for each phantom DIR/<phantom>_errors.json and the views
DIR/<phantom>_cut.png and DIR/<phantom>_costs.png; and DIR/errors.md and
DIR/errors.png over every phantom in DIR. "views" redraws the cuts from
the saved paths (the case is rebuilt with the same noise and seed).
"""

import argparse
import importlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bent_terms as BT  # noqa: E402
from bent_sample import inside, trilinear  # noqa: E402

from baillarger import flattening as FLT  # noqa: E402
from baillarger import orient as O  # noqa: E402
from baillarger import rays as R  # noqa: E402

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

#: the start of the profiles of the warp: the inward cap, on the warp's
#: level grid (flattening.grid_start), mm
S_LO = FLT.grid_start(R.CAP_IN)
#: the extended barrier of cost 5
EXTEND = 0.1
#: a term below this (absolute) is 0 for the weights
ZERO = 1e-9
#: the curvature of the reference perturbation, 1 / mm
KAPPA_REF = 0.05
#: a round that moves the paths by less than this ends the fit, mm
MOVE_TOL = 0.01
#: the default number of rounds and of LBFGS iterations per round
ROUNDS = 6
STEPS = 20
#: the LBFGS history
HISTORY = 10
#: the constant added to the log image of a phantom before exp
LOG_OFFSET = 10.0
#: the outlier threshold before orient (bent_texture.outliers), robust SDs
OUTLIER_K = 5.0
#: the four arms: (cost 3 form, cost 7 on)
ARMS: dict[str, tuple[str, bool]] = {
    "A7": ("A", True),
    "A": ("A", False),
    "B7": ("B", True),
    "B": ("B", False),
}


# ---------------------------------------------------------------- the problem


@dataclass
class Problem:
    """The fixed inputs of a fit: the torch inputs of bent_terms, the mesh
    edges as flattening pairs, the fixed feature scales and energy
    reference."""

    inputs: BT.Inputs
    pairs: I64
    scales: F64
    energy_ref: float
    s_lo: float = S_LO


def zeros(inputs: BT.Inputs) -> torch.Tensor:
    """The straight paths: coefficients (n, 2, K) of 0."""
    return torch.zeros(
        (inputs.n, 2, inputs.K), dtype=inputs.dtype, device=inputs.device
    )


@dataclass
class Profiles:
    """flattening.Padded of the paths, and the positions (n, m, 3) in voxels
    of its samples s = s_lo + i STEP."""

    padded: FLT.Padded
    vox: F64


def sample_profiles(params: torch.Tensor, inputs: BT.Inputs, s_lo: float) -> Profiles:
    """The profiles along the paths of params: the straight inward part
    r + s n for s_lo <= s < 0, then the path for 0 <= s <= S_MAX, at STEP.
    The echoes are the feature echoes of inputs (trilinear). valid: the
    sample is in the volume. cortex: valid and the soft cortex map C
    (bent_terms) >= 0.5 at the sample, in place of the label intervals of
    the ray sampler."""
    k_in = int(round(-s_lo / BT.STEP))
    if abs(k_in * BT.STEP + s_lo) > 1e-6:
        raise ValueError(f"s_lo {s_lo}: not a multiple of the step")
    with torch.no_grad():
        p = BT.paths(params, inputs)
        s_in = -BT.STEP * torch.arange(
            k_in, 0, -1, dtype=inputs.dtype, device=inputs.device
        )
        y_in = s_in[None, :, None] * inputs.normal[:, None]
        y = torch.cat([y_in, p.y], 1)
        vox = inputs.root_vox[:, None] + y @ inputs.lin.T
        f = trilinear(inputs.volumes, vox)
        ok = inside(inputs.shape, vox)
    okn = ok.cpu().numpy()
    echo = f[3:].double().cpu().numpy()
    echo[:, ~okn] = np.nan
    cortex = okn & (f[2].double().cpu().numpy() >= 0.5)
    pad = FLT.Padded(float(s_lo), BT.STEP, echo, okn, cortex)
    return Profiles(pad, vox.double().cpu().numpy())


def make_problem(inputs: BT.Inputs, s_lo: float = S_LO) -> Problem:
    """The Problem of inputs: the scales and the energy reference of the
    straight profiles. inputs.scales is set to those scales."""
    pr = sample_profiles(zeros(inputs), inputs, s_lo)
    raw = FLT.raw_features(pr.padded)
    sc = FLT.scales_of(raw, pr.padded.cortex)
    if not np.all(np.isfinite(sc) & (sc > 0)):
        raise ValueError(f"feature scales {sc}: not positive")
    inputs.scales = torch.as_tensor(sc).to(inputs.dtype).to(inputs.device)
    pairs = FLT.orient(inputs.edges, np.arange(inputs.n, dtype=np.int64))
    F, M, _ = FLT.features(pr.padded, sc)
    ps = FLT.pair_shifts(F, M, pairs, BT.STEP)
    return Problem(inputs, pairs, sc, ps.energy_ref, s_lo)


def fit_warp(prob: Problem, params: torch.Tensor) -> FLT.Warp:
    """flattening.fit on the profiles of params, with the problem's fixed
    scales and energy reference."""
    pr = sample_profiles(params, prob.inputs, prob.s_lo)
    F, M, _ = FLT.features(pr.padded, prob.scales)
    ps = FLT.pair_shifts(F, M, prob.pairs, BT.STEP, energy_ref=prob.energy_ref)
    return FLT.fit(ps, prob.inputs.n, prob.s_lo, BT.STEP)


# ---------------------------------------------------------------- weights


def reference_params(inputs: BT.Inputs, seed: int) -> torch.Tensor:
    """p_ref (see the module docstring)."""
    B0, _, _ = BT.basis()
    s = np.arange(BT.NS) * BT.STEP
    q, *_ = np.linalg.lstsq(B0, s * s / 2, rcond=None)
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2 * np.pi, inputs.n)
    c = KAPPA_REF * np.stack([np.cos(phi), np.sin(phi)], 1)[:, :, None] * q[None, None]
    return torch.as_tensor(c).to(inputs.dtype).to(inputs.device)


def terms(
    params: torch.Tensor,
    prob: Problem,
    levels: BT.Levels,
    form: str,
    censored: torch.Tensor,
    free_length: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """bent_terms.total with the fit's fixed settings."""
    return BT.total(
        params,
        prob.inputs,
        weights or {},
        levels,
        form,
        censored=censored,
        free_length=free_length,
        extend=EXTEND,
    )


def start_weights(
    prob: Problem, levels: BT.Levels, form: str, seed: int
) -> tuple[dict[str, float], dict[str, Any]]:
    """The weights of the rule of the module docstring, and a record: the
    terms at p0 and at p_ref, and which of the two set each weight."""
    inp = prob.inputs
    p0 = zeros(inp)
    with torch.no_grad():
        cen = BT.soft_top(BT.paths(p0, inp), inp).censored
        t0 = terms(p0, prob, levels, form, cen)
        tr = terms(reference_params(inp, seed), prob, levels, form, cen)
    w: dict[str, float] = {}
    rec: dict[str, Any] = {}
    for k in BT.TERMS:
        if k not in t0:
            continue
        a, b = float(t0[k]), float(tr[k])
        by = "start" if abs(a) >= ZERO else ("reference" if abs(b) >= ZERO else "none")
        w[k] = (
            1.0 / abs(a)
            if by == "start"
            else (1.0 / abs(b) if by == "reference" else 1.0)
        )
        rec[k] = dict(start=a, reference=b, weight=w[k], by=by)
    return w, rec


# ---------------------------------------------------------------- the fit


def move_mm(a: torch.Tensor, b: torch.Tensor, inputs: BT.Inputs) -> float:
    """The largest distance (mm) between the paths of a and b at equal s,
    over the samples below the soft top (mask >= 0.5) of either."""
    with torch.no_grad():
        pa, pb = BT.paths(a, inputs), BT.paths(b, inputs)
        ma = BT.soft_top(pa, inputs).mask
        mb = BT.soft_top(pb, inputs).mask
        d = torch.linalg.norm(pa.y - pb.y, dim=-1)
        use = (ma >= 0.5) | (mb >= 0.5)
        return (
            float(torch.where(use, d, torch.zeros_like(d)).max()) if d.numel() else 0.0
        )


def _floats(t: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v) for k, v in t.items()}


@dataclass
class Fit:
    """The result of fit: the coefficients, the free lengths of the last
    accepted round, the last accepted warp, the record of each round and why
    the fit stopped."""

    params: torch.Tensor
    free_length: torch.Tensor
    warp: FLT.Warp
    weights: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)
    stop: str = ""


def descend(
    prob: Problem,
    levels: BT.Levels,
    weights: Mapping[str, float],
    form: str,
    params: torch.Tensor,
    censored: torch.Tensor,
    free: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """At most `steps` LBFGS iterations on (params, free) at a fixed warp
    and censored set. Returns the new values and the number of evaluations
    of the total."""
    x = params.detach().clone().requires_grad_(True)
    fl = free.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [x, fl],
        lr=1.0,
        max_iter=steps,
        history_size=HISTORY,
        line_search_fn="strong_wolfe",
    )
    count = [0]

    def closure() -> torch.Tensor:
        opt.zero_grad()
        t = terms(x, prob, levels, form, censored, fl, weights)["total"]
        t.backward()  # type: ignore[no-untyped-call]
        count[0] += 1
        return t

    if steps > 0:
        opt.step(closure)  # type: ignore[no-untyped-call]
    return x.detach(), fl.detach(), count[0]


def fit(
    prob: Problem,
    form: str,
    alignment: bool,
    rounds: int = ROUNDS,
    steps: int = STEPS,
    seed: int = 0,
    tol: float = MOVE_TOL,
    log: Any = None,
    warp0: FLT.Warp | None = None,
) -> Fit:
    """The alternating fit (module docstring). alignment: cost 7 in the
    total (its weight from start_weights), or weight 0. The weights are set
    once, at the straight rays' warp. warp0: the warp of the straight rays
    (fit_warp of zeros), if it is already known (it is the same for every
    arm)."""
    torch.manual_seed(seed)
    inp = prob.inputs
    p = zeros(inp)
    warp = fit_warp(prob, p) if warp0 is None else warp0
    lv = BT.Levels.from_warp(inp, warp)
    w, wrec = start_weights(prob, lv, form, seed)
    if not alignment and "alignment" in w:
        w["alignment"] = 0.0
    out = Fit(p, torch.zeros(inp.n, dtype=inp.dtype), warp, w)
    out.history.append(dict(round=0, weights_record=wrec))
    prev_total: float | None = None
    for k in range(1, rounds + 1):
        t0 = time.time()
        if k > 1:
            warp = fit_warp(prob, p)
            lv = BT.Levels.from_warp(inp, warp)
        with torch.no_grad():
            top = BT.soft_top(BT.paths(p, inp), inp)
        cen, free = top.censored, top.length.detach()
        with torch.no_grad():
            before = _floats(terms(p, prob, lv, form, cen, free, w))
        pn, fn, nev = descend(prob, lv, w, form, p, cen, free, steps)
        with torch.no_grad():
            after = _floats(terms(pn, prob, lv, form, cen, fn, w))
        undone = not after["total"] <= before["total"]
        if undone:
            pn, fn, after = p, free, before
        mv = move_mm(p, pn, inp)
        rec: dict[str, Any] = dict(
            round=k,
            before=before,
            after=after,
            weighted={kk: w.get(kk, 0.0) * after[kk] for kk in BT.TERMS if kk in after},
            move_mm=mv,
            censored=int(cen.sum()),
            evaluations=nev,
            descent_undone=undone,
            seconds=time.time() - t0,
        )
        accepted = prev_total is None or after["total"] <= prev_total
        rec["accepted"] = accepted
        out.history.append(rec)
        if log:
            log(
                f"round {k}: total {before['total']:.4f} -> {after['total']:.4f}, "
                f"move {mv:.4f} mm, {nev} evals, {rec['seconds']:.1f} s"
                + ("" if accepted else " (rejected: total above the last round's)")
            )
        if not accepted:
            out.stop = "total increased"
            break
        p, prev_total = pn, after["total"]
        out.params, out.free_length, out.warp = pn, fn, warp
        if mv < tol:
            out.stop = "converged"
            break
    else:
        out.stop = "rounds"
    return out


# ---------------------------------------------------------------- traces


def traces(params: torch.Tensor, inputs: BT.Inputs) -> tuple[F64, I64, Bool]:
    """The traces of the paths for the errors: positions (n, NS, 3) in
    voxels, the count of samples from s = 0 with soft mask >= 0.5 (at least
    1), and the censored flags."""
    with torch.no_grad():
        p = BT.paths(params, inputs)
        top = BT.soft_top(p, inputs)
        ns = (top.mask >= 0.5).sum(-1).clamp(min=1)
    return (
        p.vox.double().cpu().numpy(),
        ns.cpu().numpy().astype(np.int64),
        top.censored.cpu().numpy(),
    )


def shear_mm(params: torch.Tensor, inputs: BT.Inputs) -> F64:
    """(n,) the largest |d| (the distance from the straight ray at equal s)
    of each path over its samples with soft mask >= 0.5, mm."""
    with torch.no_grad():
        p = BT.paths(params, inputs)
        m = BT.soft_top(p, inputs).mask
        d = torch.linalg.norm(p.d, dim=-1)
        out = torch.where(m >= 0.5, d, torch.zeros_like(d)).amax(-1)
    r: F64 = out.double().cpu().numpy()
    return r


# ---------------------------------------------------------------- phantoms

#: the prototype modules without type annotations, as Any
FP: Any = importlib.import_module("fold_phantoms")
H: float = FP.H
#: the slab of columns: |local coordinate on the slab axis| <= SLAB_MM + the
#: halo, roots at least SEL_DFACE_MM from a box face; the halo (mesh
#: distance from the slab mesh's boundary below HALO_MM) is fixed
SLAB_MM = 0.4
HALO_MM = 0.3
SEL_DFACE_MM = 1.0
#: the errors: roots at least this far from a box face (fold_eval's rule)
EVAL_DFACE_MM = 2.0
#: the most columns drawn in a cut
VIEW_COLUMNS = 60
#: the slab axis of each kind, in the local frame (the axis normal to the
#: view plane of fold_eval.view)
SLAB_AXIS = {"crown": 1, "fundus": 2, "kiss": 2, "bent": 2}


@dataclass(frozen=True)
class PhantomNoise:
    """orient's noise model (orient.NoiseModel) of a phantom: Gaussian noise
    on the log intensity, SD sd_e per echo, independent, constant: the rest
    term alone (module docstring)."""

    sigma: F64
    R: F64
    gain: float
    rest: F64

    @classmethod
    def of(cls, sds: Sequence[float]) -> "PhantomNoise":
        sd = np.asarray(sds, np.float64)
        if not np.all(sd > 0):
            raise ValueError(f"noise SDs {sd}: orient's C needs SD > 0")
        g = O.gradient_gain(O.SD)
        return cls(np.zeros(len(sd)), np.eye(len(sd)), g, g * np.diag(sd * sd))


def phantom_image(
    fold: Any, feats: Mapping[str, Any], noise: float = 1.0, seed: int = 0
) -> tuple[npt.NDArray[np.int16], F64, list[float]]:
    """The labels, the noisy log image L (2, X, Y, Z) of echo 2 and 3, and
    the noise SDs, by the steps and generator calls of fold_phantoms.make."""
    labels, imgs = FP.render(fold, feats)
    ctx = labels == 0
    rng = np.random.default_rng(seed)
    out, sds = [], []
    for e, L0 in zip(("echo2", "echo3"), imgs, strict=True):
        s0 = FP.hp_stat(FP.detrend(L0, ctx), ctx)
        z = np.random.default_rng(seed + 99).standard_normal(L0.shape)
        k1 = FP.hp_stat(FP.detrend(z, ctx), ctx)
        sd = np.sqrt(max(feats[e]["hp_stat"] ** 2 - s0**2, 0.0)) / k1
        sds.append(float(sd))
        out.append(L0 + noise * sd * rng.standard_normal(L0.shape))
    return np.asarray(labels, np.int16), np.stack(out), sds


def phantom_align(
    echoes: F64, cortex: Bool, sds: Sequence[float]
) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
    """The laminar field of cost 7 on a phantom (module docstring), and a
    record: the outlier count and the coherence quartiles in the cortex."""
    BTX: Any = importlib.import_module("bent_texture")
    vs = np.full(3, H)
    odd, _ = BTX.outliers(echoes, cortex, vs, OUTLIER_K)
    ctx = cortex & ~odd
    a = O.orient_arrays(echoes, ctx, PhantomNoise.of(sds), vs)
    lead, c = O.leading(np.asarray(a["tensor"], np.float64))
    aff = np.diag([H, H, H, 1.0])
    al = BT.align_field(lead, c, aff, vs)
    cc = c[ctx & np.isfinite(c)]
    rec = dict(
        outliers=int(odd.sum()),
        cortex_voxels=int(cortex.sum()),
        coherence_p25_50_75=np.percentile(cc, [25, 50, 75]).round(3).tolist()
        if cc.size
        else [],
    )
    return al, rec


def select_columns(fold: Any, v: F64, f: I64, axis: int) -> tuple[I64, I64, Bool]:
    """The slab of columns (see SLAB_MM): the indices of the chosen
    vertices, the faces between them (reindexed, zero-area faces dropped),
    and moving (mesh distance from the slab mesh's boundary >= HALO_MM).
    The distance to a box face is that of the vertex's continuous voxel
    position."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    from baillarger.mesh_smoothing import boundary_vertices, edges

    shape = np.asarray(fold.shape, np.float64)
    dface = np.minimum(v, shape - 1 - v).min(1) * H
    loc = fold.to_local(v)
    sel = (dface >= SEL_DFACE_MM) & (np.abs(loc[:, axis]) <= SLAB_MM + HALO_MM)
    fk = f[sel[f].all(1)]
    cr = np.cross(v[fk[:, 1]] - v[fk[:, 0]], v[fk[:, 2]] - v[fk[:, 0]])
    fk = fk[np.linalg.norm(cr, axis=1) > 1e-9]
    idx = np.unique(fk)
    remap = np.full(len(v), -1, np.int64)
    remap[idx] = np.arange(len(idx))
    fs = remap[fk]
    n = len(idx)
    e = edges(fs).astype(np.int64)
    ln = np.linalg.norm(v[idx[e[:, 0]]] - v[idx[e[:, 1]]], axis=1) * H
    G = coo_matrix(
        (np.r_[ln, ln], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
        shape=(n, n),
    ).tocsr()
    bnd = np.flatnonzero(boundary_vertices(fs, n))
    if len(bnd):
        dist = np.asarray(dijkstra(G, indices=bnd, min_only=True, limit=2 * HALO_MM))
    else:
        dist = np.full(n, np.inf)
    return idx, fs, dist >= HALO_MM


@dataclass
class Case:
    """A phantom ready to fit."""

    name: str
    fold: Any
    labels: npt.NDArray[np.int16]
    det: F64
    problem: Problem
    #: the chosen mesh vertices (indices into the full mesh), voxel roots
    vertex: I64
    roots_vox: F64
    #: the columns (indices into the problem's columns) of the errors
    eval_cols: I64
    truth: Any
    align_record: dict[str, Any]
    noise_sd: list[float]


def phantom_case(
    name: str,
    fold: Any,
    feats: Mapping[str, Any],
    noise: float = 1.0,
    seed: int = 0,
    align: bool = True,
    device: str = "cpu",
) -> Case:
    """The fit's inputs on one phantom (module docstring)."""
    CT: Any = importlib.import_module("column_tracers")
    labels, L, sds = phantom_image(fold, feats, noise, seed)
    echoes = np.exp(L + LOG_OFFSET)
    cortex, wm, outside = labels == 0, labels == 2, labels == 1
    v, f, nrm = CT.white_mesh(wm)
    v = np.asarray(v, np.float64)
    nrm = np.asarray(nrm, np.float64)
    picked = np.asarray(CT.pick_vertices(v, nrm), np.int64)
    axis = SLAB_AXIS[fold.kind]
    idx, fs, moving = select_columns(fold, v, np.asarray(f, np.int64), axis)
    al, arec = phantom_align(echoes, cortex, sds) if align else (None, {})
    inputs = BT.Inputs.from_arrays(
        roots=v[idx] * H,
        normals=nrm[idx],
        faces=fs,
        mm_to_vox=np.diag([1 / H, 1 / H, 1 / H, 1.0]),
        outside=outside,
        wm=wm,
        cortex=cortex,
        echoes=echoes,
        scales=np.ones(len(echoes)),
        moving=moving,
        align=al,
        device=device,
        dtype=torch.float32,
    )
    prob = make_problem(inputs)
    shape = np.asarray(fold.shape, np.float64)
    vv = v[idx]
    dface = np.minimum(vv, shape - 1 - vv).min(1) * H
    loc = fold.to_local(vv)
    ev = (
        np.isin(idx, picked)
        & moving
        & (dface >= EVAL_DFACE_MM)
        & (np.abs(loc[:, axis]) <= SLAB_MM)
    )
    cols = np.flatnonzero(ev)
    if fold.kind == "bent":
        BP: Any = importlib.import_module("bent_phantom")
        truth = BP.true_columns(fold, vv[cols])
    else:
        truth = fold.column(vv[cols])
    det = FP.detrend(L[0], cortex)
    return Case(name, fold, labels, det, prob, idx, vv, cols, truth, arec, sds)


def case_errors(case: Case, params: torch.Tensor) -> dict[str, F64]:
    """The errors of the paths of params on the case's error roots:
    fold_eval.errors (fold phantoms) or bent_phantom.path_errors (curved),
    and the largest shear (mm)."""
    pos, ns, cen = traces(params, case.problem.inputs)
    c = case.eval_cols
    if case.fold.kind == "bent":
        BP: Any = importlib.import_module("bent_phantom")
        e = BP.path_errors(case.fold, case.truth, pos[c], ns[c])
    else:
        FE: Any = importlib.import_module("fold_eval")
        e = FE.errors(case.fold, case.truth, pos[c], ns[c], np.full(3, H))
    return dict(
        emax=e[0],
        emean=e[1],
        eend=e[2],
        econ=e[3],
        shear=shear_mm(params, case.problem.inputs)[c],
        censored=cen[c].astype(np.float64),
    )


# ---------------------------------------------------------------- the runs


def _q(a: F64) -> dict[str, float | None]:
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(median=None, p90=None, max=None)
    return dict(
        median=round(float(np.median(a)), 4),
        p90=round(float(np.percentile(a, 90)), 4),
        max=round(float(a.max()), 4),
    )


def summary(e: Mapping[str, F64]) -> dict[str, Any]:
    out: dict[str, Any] = {
        k: _q(e[k]) for k in ("emax", "emean", "eend", "econ", "shear")
    }
    out["censored"] = int(np.sum(e["censored"]))
    out["n"] = len(e["emax"])
    return out


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def view_cut(path: Path, case: Case, fits: Mapping[str, torch.Tensor]) -> None:
    """A cut through the phantom (the plane of fold_eval.view: local
    coordinate 0 on the slab axis): echo 2 (trend removed), and for the
    error roots within 0.1 mm of the plane (at most VIEW_COLUMNS, evenly
    in their order) the true columns (gold), the straight rays (cyan) and
    the fitted paths (red), one panel per arm. Below: every error root,
    projected on the plane, coloured by its largest shear (shear_mm)."""
    from scipy import ndimage as ndi

    plt = _plt()
    fold = case.fold
    ax_n = SLAB_AXIS[fold.kind]
    ax_u, ax_v = [a for a in range(3) if a != ax_n]
    ext = np.asarray(fold.shape) * H / 2
    g = np.arange(-ext.min(), ext.min(), H / 2)
    U, V = np.meshgrid(g, g, indexing="ij")
    Y = np.zeros((*U.shape, 3))
    Y[..., ax_u], Y[..., ax_v] = U, V
    img = ndi.map_coordinates(
        case.det, fold.to_vox(Y.reshape(-1, 3)).T, order=1, mode="constant", cval=np.nan
    ).reshape(U.shape)
    c = case.eval_cols
    loc = fold.to_local(case.roots_vox[c])
    near = np.flatnonzero(np.abs(loc[:, ax_n]) < 0.1)
    sel = near[:: max(1, int(np.ceil(len(near) / VIEW_COLUMNS)))]
    inp = case.problem.inputs
    p0, n0, _ = traces(zeros(inp), inp)
    names = list(fits)
    fig, axs = plt.subplots(
        2, len(names), figsize=(4.6 * len(names), 9.2), squeeze=False
    )
    lr = fold.to_local(case.roots_vox[c])
    shears = {nm: shear_mm(fits[nm], inp)[c] for nm in names}
    smax = max([float(x.max()) for x in shears.values()] + [1e-3])
    for a, nm in zip(axs[1], names, strict=True):
        sc = a.scatter(
            lr[:, ax_u],
            lr[:, ax_v],
            c=shears[nm],
            s=3,
            cmap="viridis",
            vmin=0,
            vmax=smax,
        )
        a.set_aspect("equal")
        a.set_title(
            f"{case.name} {nm}: largest shear of each error root, mm", fontsize=8
        )
        fig.colorbar(sc, ax=a, shrink=0.8)
    vmax = float(np.nanpercentile(np.abs(img), 98))
    for a, nm in zip(axs[0], names, strict=True):
        a.imshow(
            img.T,
            origin="lower",
            cmap="gray",
            vmin=-vmax,
            vmax=vmax,
            extent=(g[0], g[-1], g[0], g[-1]),
        )
        pf, nf, _ = traces(fits[nm], inp)
        for i in sel:
            if fold.kind == "bent":
                t = fold.to_local(case.truth[i])
            else:
                tr = case.truth
                cc, u = tr["c"][i], tr["u"][i]
                t = np.array([cc + tr["r0"][i] * u, cc + tr["r1"][i] * u])
            a.plot(t[:, ax_u], t[:, ax_v], "-", c="#ffd23f", lw=1.2, alpha=0.9)
            j = c[i]
            q = fold.to_local(p0[j, : n0[j]])
            a.plot(q[:, ax_u], q[:, ax_v], "-", c="#30d5ff", lw=0.6)
            q = fold.to_local(pf[j, : nf[j]])
            a.plot(q[:, ax_u], q[:, ax_v], "-", c="#e34948", lw=0.7)
        a.set_title(f"{case.name} {nm}: gold true, cyan straight, red fit", fontsize=8)
        a.set_xlim(g[0], g[-1])
        a.set_ylim(g[0], g[-1])
        a.set_xlabel(f"local axis {ax_u}, mm", fontsize=7)
        a.set_ylabel(f"local axis {ax_v}, mm", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def view_costs(
    path: Path, name: str, records: Mapping[str, list[dict[str, Any]]]
) -> None:
    """The weighted terms and the total after each round's descent, per arm
    (round 0: at the straight rays, the "before" of round 1)."""
    plt = _plt()
    fig, axs = plt.subplots(
        1, len(records), figsize=(4.2 * len(records), 3.4), squeeze=False
    )
    for a, (arm, hist) in zip(axs[0], records.items(), strict=True):
        rr = [h for h in hist if h.get("round", 0) >= 1]
        if not rr:
            continue
        w = hist[0]["weights"]
        x = [0] + [h["round"] for h in rr]
        for k in [*BT.TERMS, "total"]:
            if k not in rr[0]["after"]:
                continue
            wk = 1.0 if k == "total" else w.get(k, 0.0)
            if wk == 0.0:
                continue
            y = [wk * rr[0]["before"][k]] + [wk * h["after"][k] for h in rr]
            a.plot(x, y, "-o", ms=2, lw=1.6 if k == "total" else 0.8, label=k)
        a.set_title(f"{name} {arm}", fontsize=8)
        a.set_xlabel("round", fontsize=7)
        a.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def run_case(
    case: Case, arms: Sequence[str], out: Path, rounds: int, steps: int, seed: int
) -> dict[str, Any]:
    """Fit every arm on a case; write the costs, paths, errors and views."""
    inp = case.problem.inputs
    e0 = case_errors(case, zeros(inp))
    p0, n0, _ = traces(zeros(inp), inp)
    res: dict[str, Any] = dict(
        straight=summary(e0),
        columns=inp.n,
        moving=int(inp.moving.sum()),
        error_roots=len(case.eval_cols),
        noise_sd=case.noise_sd,
        align=case.align_record,
        arms={},
    )
    w0 = fit_warp(case.problem, zeros(inp))
    fits: dict[str, torch.Tensor] = {}
    recs: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        form, al = ARMS[arm]
        if al and inp.align is None:
            continue
        t0 = time.time()
        f = fit(
            case.problem,
            form,
            al,
            rounds,
            steps,
            seed,
            log=lambda m, a=arm: say(f"  {case.name} {a}: {m}"),
            warp0=w0,
        )
        sec = time.time() - t0
        fits[arm] = f.params
        hist = [dict(f.history[0], weights=f.weights), *f.history[1:]]
        recs[arm] = hist
        (out / f"{case.name}_{arm}_costs.json").write_text(
            json.dumps(
                dict(
                    form=form,
                    alignment=al,
                    stop=f.stop,
                    seconds=sec,
                    weights=f.weights,
                    history=hist,
                ),
                indent=1,
            )
        )
        e = case_errors(case, f.params)
        pf, nf, cen = traces(f.params, inp)
        arrays: dict[str, Any] = dict(
            roots_vox=case.roots_vox,
            vertex=case.vertex,
            moving=inp.moving.numpy(),
            params=f.params.numpy(),
            free_length=f.free_length.numpy(),
            pos=pf.astype(np.float32),
            ns=nf,
            censored=cen,
            pos_straight=p0.astype(np.float32),
            ns_straight=n0,
            eval_cols=case.eval_cols,
            shear_mm=shear_mm(f.params, inp),
        )
        arrays.update({f"fit_{k}": v for k, v in e.items()})
        arrays.update({f"straight_{k}": v for k, v in e0.items()})
        np.savez_compressed(out / f"{case.name}_{arm}_paths.npz", **arrays)
        res["arms"][arm] = dict(
            fit=summary(e),
            stop=f.stop,
            seconds=round(sec, 1),
            rounds=len(f.history) - 1,
            largest_shear_all_moving_mm=round(float(shear_mm(f.params, inp).max()), 4),
        )
        say(
            f"  {case.name} {arm}: stop {f.stop}, {sec:.0f} s, emax median "
            f"{res['straight']['emax']['median']} -> "
            f"{res['arms'][arm]['fit']['emax']['median']}"
        )
    (out / f"{case.name}_errors.json").write_text(json.dumps(res, indent=1))
    view_cut(out / f"{case.name}_cut.png", case, fits)
    view_costs(out / f"{case.name}_costs.png", case.name, recs)
    return res


def table(results: Mapping[str, Any]) -> list[str]:
    """The error table, markdown: median / p90 of each error, mm."""

    def m(s: Mapping[str, Any], k: str) -> str:
        q = s[k]
        if q["median"] is None:
            return "-"
        return f"{q['median']:.3f} / {q['p90']:.3f}"

    rows = [
        "| phantom | arm | n | emax | emean | eend | econ | shear max | censored |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nm, r in results.items():
        items = [("straight", r["straight"])] + [
            (a, v["fit"]) for a, v in r["arms"].items()
        ]
        for arm, s in items:
            rows.append(
                f"| {nm} | {arm} | {s['n']} | {m(s, 'emax')} | {m(s, 'emean')} | "
                f"{m(s, 'eend')} | {m(s, 'econ')} | {s['shear']['max']} | "
                f"{s['censored']} |"
            )
    return rows


def view_table(path: Path, rows: Sequence[str]) -> None:
    plt = _plt()
    fig = plt.figure(figsize=(11, 0.18 * len(rows) + 0.6))
    fig.text(
        0.01,
        0.99,
        "\n".join(r for r in rows if not r.startswith("|---")),
        family="monospace",
        fontsize=6.5,
        va="top",
    )
    fig.savefig(path, dpi=90)
    plt.close(fig)


T0 = time.time()


def say(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def phantoms(names: Sequence[str] | None = None) -> dict[str, Any]:
    """fold_phantoms.standard() and the curved phantom ("curved")."""
    BP: Any = importlib.import_module("bent_phantom")
    out: dict[str, Any] = dict(FP.standard())
    out["curved"] = BP.curved()
    if names:
        out = {k: out[k] for k in names}
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("phantoms", help="fit every arm on the phantoms")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--features",
        type=Path,
        required=True,
        help="fold_features.json (fold_measure.py)",
    )
    p.add_argument("--site", default="V1V2_banks")
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--phantoms", nargs="*")
    p.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    v = sub.add_parser(
        "views", help="redraw the cut views from the paths of a phantoms run"
    )
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--features", type=Path, required=True)
    v.add_argument("--site", default="V1V2_banks")
    v.add_argument("--noise", type=float, default=1.0)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--phantoms", nargs="*")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    feats = json.loads(a.features.read_text())[a.site]
    if a.cmd == "views":
        for nm, fold in phantoms(a.phantoms).items():
            case = phantom_case(nm, fold, feats, a.noise, a.seed, align=False)
            fits = {}
            for arm in ARMS:
                q = a.out / f"{nm}_{arm}_paths.npz"
                if q.exists():
                    d = np.load(q)
                    if not np.array_equal(d["vertex"], case.vertex):
                        raise ValueError(f"{q.name}: other columns than the case")
                    fits[arm] = torch.as_tensor(d["params"])
            view_cut(a.out / f"{nm}_cut.png", case, fits)
            say(f"{nm}: views")
        return
    results: dict[str, Any] = {}
    for nm, fold in phantoms(a.phantoms).items():
        say(f"{nm}: shape {fold.shape}")
        case = phantom_case(nm, fold, feats, a.noise, a.seed, align=True)
        say(
            f"  {nm}: {case.problem.inputs.n} columns, "
            f"{int(case.problem.inputs.moving.sum())} moving, "
            f"{len(case.eval_cols)} error roots"
        )
        results[nm] = run_case(case, a.arms, a.out, a.rounds, a.steps, a.seed)
        # the table over every phantom in the directory (earlier runs too)
        allr = {
            q.name[: -len("_errors.json")]: json.loads(q.read_text())
            for q in sorted(a.out.glob("*_errors.json"))
        }
        rows = table(allr)
        (a.out / "errors.md").write_text("\n".join(rows) + "\n")
        view_table(a.out / "errors.png", rows)
    (a.out / "run.json").write_text(
        json.dumps(
            dict(
                argv=list(sys.argv),
                site=a.site,
                noise=a.noise,
                seed=a.seed,
                rounds=a.rounds,
                steps=a.steps,
                seconds=time.time() - T0,
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
