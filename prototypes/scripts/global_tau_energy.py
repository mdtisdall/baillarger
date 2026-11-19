"""The energy of the global sparse τ solve (JAX, float64): a copy of
tube_energy.py (branch feature/tube-solver) with the same terms and constants,
on one global slot set instead of tube windows.

Slots: the S domain voxels and one dummy slot (index S, value 0, never a
valid neighbour). Each slot has six neighbour slots (nb (S1, 3, 2): axis,
minus/plus), a flag nbv (a real neighbour on the same side of every contact
face) and a flag ghost (a contact face between the slot and that neighbour).

Contact faces (user decision G7, 2026-11-17): a contact face is a shared
stop set with τ = 0 on the face itself, half a voxel from each side. The
first derivative of τ across it uses this ghost value: (0 - τ)/(h/2). The
second-level derivatives (of ĝ and of v) use real neighbours only, so they
are one-sided at a contact.

Terms per slot, in nats (the user's choices B-J and R1, 2026-11-17; the
same as tube_energy.py):
  f      = clip(0.5 - τ / HTAU, 0, 1), the tissue fraction (C)
  stop   = f c, c = +-log((1 - ε) / ε) for label 0 / cortex (C)
  align  = f κ coh (1 - (ĝ·n̂)²) (G)
  smooth = f |J P|² / (2 σ_t²), J = ∂ĝ, P = I - ĝĝᵀ (H)
  equi   = f wq (div v)², v = ∇τ / |∇τ|², wq = W_EQ / (2 A_owner) (I)
  pull   = [f (ĝ₀·∇τ - 1)² + (1 - f) max(|∇τ| - 1, 0)²] / (2 σ_p²) (J)
The gradient terms count only where the slot has a neighbour or a ghost on
all three axes (valid).

Ridge form (P["ridge"] = 1; settings contacts="ridge", user decisions
2026-11-18): no contact faces and no ghosts.
  - The first derivative on each axis is the continuous upwind split
    max(D-, 0) + min(D+, 0) (Engquist-Osher), D- and D+ the one-sided
    differences: the backward difference where τ rises along the axis, the
    forward one where it falls, 0 at a symmetric ridge or minimum. It is
    continuous in τ, and a voxel-to-voxel alternation changes it. (A first
    version switched between D- and D+ by the lower neighbour. That energy
    jumps where the choice switches, because the direction terms use ∇τ's
    sign; it diverged on site 1.)
  - div v uses each neighbour's v weighted by the neighbour's tissue
    fraction f: w v_nb + (1 - w) v_self. A neighbour beyond the top (f = 0)
    then adds nothing (equivolume in tissue only, as in the 1D convex form),
    and the term is continuous in τ.
In the faces form (P["ridge"] = 0) the energy is unchanged. Inner slots (inside the white surface) carry the mm
pull only (R1). wq comes from the area of the slot's owner patch.

Per patch (F, E): t = -τ(root) (trilinear over the root's corners in the
domain), -log p = log t + r² / (2 SD²) with r = log t - log median; and one
Student-t (ν = 3) term on r_i - r_j for each pair of mesh-adjacent patches
(once per pair; the tube solver's local problems are the restriction of this
sum to one tube's unknowns).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

LOG_ODDS = float(np.log(0.95 / 0.05))  # C: label error rate 0.05
KAPPA = 4.0  # G
SIGMA_T = 1.0  # H (per mm)
W_EQ = 0.569  # I: PROVISIONAL (the 1D E5 placeholder), flagged
SIGMA_P = float(np.log(2.0))  # J: SD log 2 (E6)
THICK_SD = 0.35  # F
NU = 3.0  # F
EPS = 1e-9
VOXEL_TERMS = ("stop", "align", "smooth", "equi", "pull")
TERMS = (*VOXEL_TERMS, "thick", "tsmooth")
NRES = 15  # voxel residuals per slot: align 3, smooth 9, equi 1, pull 2

Array = Any


def _fade(P: dict[str, Array]) -> tuple[Array, Array] | None:
    """The fade-in of the voxels next to a fixed top surface Σ (T5,
    top-surface plan; user, 2026-11-19: "Do the fade-in first"), from the
    signed distance sdist (S1,) of each slot to Σ (mm, negative inside):
      ω = clip(-2 s / h, 0, 1): 0 at Σ, 1 from h/2 inside. A slot's own
          voxel residuals have the weight ω, and a neighbour of weight ω
          counts with the weight ω in its neighbours' stencils (below).
      θ' = 1 + clip(-s, 0, h/2) / h: where a neighbour fades, the other
          stencil on that side is the ghost τ = 0 at θ' h, which is the
          ghost the slot had (θ = 1) when the neighbour entered the domain
          at s = 0. (θ' is the distance for a Σ normal to the axis; for an
          oblique Σ the crossing is farther. It only needs to be continuous
          and equal to 1 at s = 0.)
    So nothing switches on at once when Σ passes over a voxel centre, and E
    is continuous in Σ. None when P has no sdist (no fade)."""
    sd = P.get("sdist")
    if sd is None:
        return None
    ht = P["htau"]
    return jnp.clip(-2.0 * sd / ht, 0.0, 1.0), 1.0 + jnp.clip(-sd, 0.0, 0.5 * ht) / ht


def _three_point(Fm: Array, dm: Array, Fp: Array, dp: Array, F: Array) -> Array:
    """The first derivative at the slot from Fm at -dm, F at 0 and Fp at +dp
    (Shortley-Weller): exact for a quadratic. With dm = dp it is the central
    difference (F has the weight 0). With a ghost at θh on one side, F has
    the weight ±(1 - θ)/(θh) (- for a ghost on the plus side), so the slot's own τ is in its gradient."""
    return (-dp / (dm * (dm + dp))) * Fm + ((dp - dm) / (dm * dp)) * F + (dm / (dp * (dm + dp))) * Fp


def _one_sided(
    vm: Array, Fm: Array, dm: Array, vp: Array, Fp: Array, dp: Array, F: Array, sw: bool = False
) -> Array:
    both = _three_point(Fm, dm, Fp, dp, F) if sw else (Fp - Fm) / (dm + dp)
    plus = (Fp - F) / dp
    minus = (F - Fm) / dm
    return jnp.where(vp & vm, both, jnp.where(vp, plus, jnp.where(vm, minus, 0.0)))


def _deriv(F: Array, nb: Array, nbv: Array, h: Array, a: int, om: Array | None = None) -> Array:
    """Derivative along axis a of F (S,) or (S, k), real neighbours only.
    With om (the fade weights ω): a neighbour of weight ω counts with the
    weight ω, the stencil without it with 1 - ω (bilinear over the two
    sides)."""
    m, p = nb[:, a, 0], nb[:, a, 1]
    vm, vp = nbv[:, a, 0], nbv[:, a, 1]
    wm = wp = None
    if om is not None:
        wm, wp = jnp.where(vm, om[m], 1.0), jnp.where(vp, om[p], 1.0)
    if F.ndim == 2:
        vm, vp = vm[:, None], vp[:, None]
        if om is not None:
            wm, wp = wm[:, None], wp[:, None]
    Fm, Fp = F[m], F[p]
    hh = h[a]
    d = _one_sided(vm, Fm, hh, vp, Fp, hh, F)
    if om is None:
        return d
    no = jnp.zeros_like(vm)
    return (
        wm * wp * d
        + wm * (1 - wp) * _one_sided(vm, Fm, hh, no, Fp, hh, F)
        + (1 - wm) * wp * _one_sided(no, Fm, hh, vp, Fp, hh, F)
        + (1 - wm) * (1 - wp) * _one_sided(no, Fm, hh, no, Fp, hh, F)
    )


def _grad_axis(x: Array, P: dict[str, Array], a: int) -> Array:
    """dτ/dx_a with the ghost value τ = 0 at gdist × h across a contact
    face (h/2) or a crossing of a fixed top surface (θh), or in the ridge
    form the continuous upwind split."""
    nb, nbv, gh, h = P["nb"], P["nbv"], P["ghost"], P["h"]
    m, p = nb[:, a, 0], nb[:, a, 1]
    gm, gp = gh[:, a, 0], gh[:, a, 1]
    vm, vp = nbv[:, a, 0] | gm, nbv[:, a, 1] | gp
    Fm = jnp.where(gm, 0.0, x[m])
    Fp = jnp.where(gp, 0.0, x[p])
    # the ghost τ = 0 sits gdist × h from the slot: 0.5 at a contact face,
    # θ (0 < θ ≤ 1) where the axis crosses a fixed top surface Σ (T3);
    # without P["gdist"], 0.5 everywhere
    gd = P.get("gdist")
    gdm = 0.5 if gd is None else gd[:, a, 0]
    gdp = 0.5 if gd is None else gd[:, a, 1]
    dm = jnp.where(gm, gdm * h[a], h[a])
    dp = jnp.where(gp, gdp * h[a], h[a])
    # with a fixed top surface Σ (gdist given): the three-point derivative,
    # so that a slot with a ghost on one side and a real neighbour on the
    # other keeps its own τ in its gradient (user, 2026-11-19: "Option 1").
    # A contact face (no gdist): the wide difference, as before.
    sw = gd is not None
    both = _three_point(Fm, dm, Fp, dp, x) if sw else (Fp - Fm) / (dm + dp)
    plus = (Fp - x) / dp
    minus = (x - Fm) / dm
    one = jnp.where(vp, plus, jnp.where(vm, minus, 0.0))
    base = jnp.where(vp & vm, both, one)
    fd = _fade(P)
    if fd is not None:
        # a real neighbour of weight ω: the stencil with it (ω) and the
        # stencil with the ghost τ = 0 at θ' h in its place (1 - ω)
        om, th = fd
        rm, rp = nbv[:, a, 0] & ~gm, nbv[:, a, 1] & ~gp
        wm, wp = jnp.where(rm, om[m], 1.0), jnp.where(rp, om[p], 1.0)
        dm2 = jnp.where(rm, th[m] * h[a], dm)
        dp2 = jnp.where(rp, th[p] * h[a], dp)
        Fm2 = jnp.where(rm, 0.0, Fm)
        Fp2 = jnp.where(rp, 0.0, Fp)
        base = (
            wm * wp * base
            + wm * (1 - wp) * _one_sided(vm, Fm, dm, vp, Fp2, dp2, x, sw)
            + (1 - wm) * wp * _one_sided(vm, Fm2, dm2, vp, Fp, dp, x, sw)
            + (1 - wm) * (1 - wp) * _one_sided(vm, Fm2, dm2, vp, Fp2, dp2, x, sw)
        )
    split = jnp.where(
        vp & vm, jnp.maximum(minus, 0.0) + jnp.minimum(plus, 0.0), one
    )
    return jnp.where(P["ridge"] > 0, split, base)


def gradient(x: Array, P: dict[str, Array]) -> Array:
    return jnp.stack([_grad_axis(x, P, a) for a in range(3)], -1)


def _deriv_w(F: Array, w: Array, nb: Array, nbv: Array, h: Array, a: int) -> Array:
    """_deriv of F (S,) with each neighbour's value blended toward the
    slot's own: w_nb F_nb + (1 - w_nb) F."""
    m, p = nb[:, a, 0], nb[:, a, 1]
    Fe = jnp.stack([w[m] * F[m] + (1 - w[m]) * F, w[p] * F[p] + (1 - w[p]) * F], 1)
    vm, vp = nbv[:, a, 0], nbv[:, a, 1]
    both = (Fe[:, 1] - Fe[:, 0]) / (2 * h[a])
    plus = (Fe[:, 1] - F) / h[a]
    minus = (F - Fe[:, 0]) / h[a]
    return jnp.where(vp & vm, both, jnp.where(vp, plus, jnp.where(vm, minus, 0.0)))


def _fields(x: Array, P: dict[str, Array]) -> tuple[Array, ...]:
    nb, nbv, h = P["nb"], P["nbv"], P["h"]
    g = gradient(x, P)
    s2 = jnp.sum(g * g, -1) + EPS
    s = jnp.sqrt(s2)
    gh = g / s[:, None]
    v = g / s2[:, None]
    ftis = jnp.clip(0.5 - x / P["htau"], 0.0, 1.0)
    w = jnp.where(P["ridge"] > 0, ftis, 1.0)
    divv = sum(_deriv_w(v[:, a], w, nb, nbv, h, a) for a in range(3))
    J = jnp.stack([_deriv(gh, nb, nbv, h, a) for a in range(3)], -1)  # (S, comp, axis)
    Pm = jnp.eye(3)[None] - gh[:, :, None] * gh[:, None, :]
    JP = jnp.einsum("sia,sab->sib", J, Pm)
    return g, s, gh, divv, JP


def voxel_terms(x: Array, P: dict[str, Array], g0hat: Array) -> dict[str, Array]:
    """Per-slot term values (not yet weighted by evalw)."""
    g, s, gh, divv, JP = _fields(x, P)
    f = jnp.clip(0.5 - x / P["htau"], 0.0, 1.0)
    valid = P["valid"]
    pull_in = (jnp.sum(g0hat * g, -1) - 1.0) ** 2
    pull_out = jnp.maximum(s - 1.0, 0.0) ** 2
    cosn = jnp.sum(gh * P["nhat"], -1)
    tis = 1.0 - P["inner"]
    return {
        "stop": tis * f * P["c"],
        "align": tis * valid * f * KAPPA * P["coh"] * (1.0 - cosn**2),
        "smooth": tis * valid * f * jnp.sum(JP * JP, (-1, -2)) / (2 * SIGMA_T**2),
        "equi": tis * valid * f * P["wq"] * divv**2,
        "pull": valid * (f * pull_in + (1 - f) * pull_out) / (2 * SIGMA_P**2),
    }


def thickness(x: Array, P: dict[str, Array]) -> Array:
    """Thickness -τ(root) of each patch (T,)."""
    return -jnp.sum(P["rw"] * x[P["rslot"]], -1)


def patch_terms(x: Array, P: dict[str, Array]) -> tuple[Array, Array]:
    """Per patch: the log-normal thickness term (T,); per pair: the Student-t
    term (Q,)."""
    t = jnp.maximum(thickness(x, P), 0.02)
    lt = jnp.log(t)
    r = lt - P["logmed"]
    thick = P["hasroot"] * (lt + r**2 / (2 * THICK_SD**2))
    d = r[P["pi"]] - r[P["pj"]]
    ts = P["pm"] * (NU + 1) / 2 * jnp.log1p(d**2 / (NU * P["psd"] ** 2))
    return thick, ts


def g0_of(x: Array, P: dict[str, Array]) -> Array:
    g = gradient(x, P)
    return jax.lax.stop_gradient(g / jnp.sqrt(jnp.sum(g * g, -1) + EPS)[:, None])


def energy_parts(x: Array, P: dict[str, Array], g0hat: Array) -> dict[str, Array]:
    vt = voxel_terms(x, P, g0hat)
    out = {k: jnp.sum(P["evalw"] * val) for k, val in vt.items()}
    thick, ts = patch_terms(x, P)
    out["thick"] = jnp.sum(thick)
    out["tsmooth"] = jnp.sum(ts)
    return out


def energy(x: Array, P: dict[str, Array], g0hat: Array) -> Array:
    return sum(energy_parts(x, P, g0hat).values())


def voxel_residuals(x: Array, P: dict[str, Array], g0hat: Array, fw: Array) -> Array:
    """Least-squares residuals per slot (S1, NRES) with the tissue fraction fw
    frozen (from tube_energy.ls_residuals): align 3, smooth 9, equi 1, pull
    in 1, pull out 1. Σ r² equals the voxel energy without the stop term when
    f = fw. Each residual of a slot depends on τ within L1 distance 2."""
    g, s, gh, divv, JP = _fields(x, P)
    ev = P["evalw"] * P["valid"]
    tis = ev * (1.0 - P["inner"])
    r_al = jnp.sqrt(tis * fw * KAPPA * P["coh"])[:, None] * jnp.cross(gh, P["nhat"])
    r_sm = jnp.sqrt(tis * fw / (2 * SIGMA_T**2))[:, None, None] * JP
    r_eq = jnp.sqrt(tis * fw * P["wq"]) * divv
    r_pi = jnp.sqrt(ev * fw / (2 * SIGMA_P**2)) * (jnp.sum(g0hat * g, -1) - 1.0)
    r_po = jnp.sqrt(ev * (1 - fw) / (2 * SIGMA_P**2)) * jnp.maximum(s - 1.0, 0.0)
    return jnp.concatenate(
        [r_al, r_sm.reshape(-1, 9), r_eq[:, None], r_pi[:, None], r_po[:, None]], axis=1
    )


def patch_residuals(x: Array, P: dict[str, Array], tw: Array) -> tuple[Array, Array]:
    """Least-squares residuals of the patch terms: the log-normal's quadratic
    part (T,) and the IRLS quadratic of each Student-t pair term (Q,), with the
    weights tw (Q,) frozen. The log t term is not least squares; it enters
    through the full gradient only."""
    t = jnp.maximum(thickness(x, P), 0.02)
    r = jnp.log(t) - P["logmed"]
    r_th = P["hasroot"] * r / (jnp.sqrt(2.0) * THICK_SD)
    r_ts = jnp.sqrt(P["pm"] * tw / 2) * (r[P["pi"]] - r[P["pj"]])
    return r_th, r_ts


def irls_weights(x: Array, P: dict[str, Array]) -> tuple[Array, Array]:
    """fw = f(τ) per slot and the Student-t IRLS weights per pair at x."""
    fw = jnp.clip(0.5 - x / P["htau"], 0.0, 1.0)
    t = jnp.maximum(thickness(x, P), 0.02)
    r = jnp.log(t) - P["logmed"]
    d = r[P["pi"]] - r[P["pj"]]
    tw = (NU + 1) / (NU * P["psd"] ** 2 + d**2)
    return fw, tw


# ------------------------------------------------------------ jitted entry points


@jax.jit
def j_energy(x: Array, P: dict[str, Array], g0: Array) -> Array:
    return energy(x, P, g0)


@jax.jit
def j_energy_grad(x: Array, P: dict[str, Array], g0: Array) -> tuple[Array, Array]:
    return jax.value_and_grad(energy)(x, P, g0)


@jax.jit
def j_g0(x: Array, P: dict[str, Array]) -> Array:
    return g0_of(x, P)


@jax.jit
def j_weights(x: Array, P: dict[str, Array]) -> tuple[Array, Array]:
    return irls_weights(x, P)


@jax.jit
def j_residual_jvp(
    x: Array, P: dict[str, Array], g0: Array, fw: Array, t: Array
) -> Array:
    """J t of the voxel residuals (S1, NRES) for one tangent t."""
    _, jt = jax.jvp(lambda y: voxel_residuals(y, P, g0, fw), (x,), (t,))
    return jt


@jax.jit
def j_patch_residuals(x: Array, P: dict[str, Array], tw: Array) -> tuple[Array, Array]:
    return patch_residuals(x, P, tw)


@jax.jit
def j_slot_parts(
    x: Array, P: dict[str, Array], g0: Array
) -> tuple[Array, Array, Array]:
    """Per-slot voxel terms (5, S1) weighted by evalw; per patch thick (T,);
    per pair tsmooth (Q,)."""
    vt = voxel_terms(x, P, g0)
    vox = jnp.stack([P["evalw"] * vt[k] for k in VOXEL_TERMS])
    thick, ts = patch_terms(x, P)
    return vox, thick, ts


# ------------------------------------------------------------ scaled terms, lagged form
#
# Scaled terms (user, 2026-11-19, option A): align, smooth and equi are
# multiplied by a power of s = |∇τ| so that nothing is divided by s. Where
# s = 1 they equal the terms above; where s → 0 (a ridge of τ) they go to 0
# instead of without bound (the equivolume term of the first site-1 T5 run
# went from 4,071 to 55,940 when one voxel reached s = 0.04).
#   align  κ coh |∇τ × n̂|²                  (= s² κ coh (1 - (ĝ·n̂)²))
#   smooth |P H P|² / (2σ_t²)                (= s² |∂ĝ P|²: ∂ĝ = P H / s)
#   equi   wq (Δτ - 2 ĝᵀHĝ)²                (= s⁴ wq (div(∇τ/s²))², exact)
#   pull   |∇τ - ĝ|² / (2σ_p²)              (= (s - 1)² at ĝ = ∇τ/s)
# H is the derivative of the discrete ∇τ (the derivative of ∇τ on each
# axis, real neighbours only, as _deriv), so the discrete identities hold
# only approximately. The thickness terms are those above (log-normal and
# Student-t). scaled_energy is this energy with ĝ = ∇τ/|∇τ| of τ itself:
# the energy that T5 (top_solve) evaluates and differentiates.
#
# The lagged form (a fixed top surface Σ; user, 2026-11-19: "start from the
# cost function we expect to be able to solve") freezes ĝ0 and P0 = I -
# ĝ0ĝ0ᵀ, the thickness t0 and the Student-t weights tw0 at the previous
# iterate x0 (lag_state): every residual is affine in τ, and one outer step
# is one convex least-squares solve. With Σ fixed the domain is inside Σ:
# f = 1, no stop term.
#   pull   |∇τ - ĝ0|²: a majoriser of (|∇τ| - 1)² (||g| - 1| <= |g - ĝ0|
#          for a unit ĝ0), equal at x0. (The first form, (ĝ0·∇τ - 1)², left
#          ∇τ free across ĝ0; where the pull is the only term, the inner
#          voxels at the start of T5, τ diverged: site 1, 2026-11-19.)
#   thick  r_lin² / (2 SD²) + t / t0, r_lin = log t0 + (t - t0)/t0 - log
#          median; t / t0 is the tangent of log t (a majoriser)
#   tsmooth tw0 (r_lin,i - r_lin,j)² / 2: the IRLS majoriser of the
#          Student-t
# r_lin is a linearisation (not a majoriser), so the energy need not fall at
# each outer step: the solver reports it.

# Lagged scale weights (user, 2026-11-19). The scaled align and smooth
# terms are s² × the original and the equivolume term s⁴ × the original, so
# at a fixed Σ they favour a flatter τ (smaller s): on site 1 they made the
# solved columns thinner as Σ moved out of the WM. With P["wlag"] = 1/s0,
# s0 = |∇τ| at a lag point x0 (lag_weights), the align and smooth residuals
# are multiplied by 1/s0 and the equivolume residual by 1/s0²: each residual
# stays affine in the scaled form's sense (the weight is a constant), and at
# x = x0 the three terms equal the original ones. The solver repeats the
# solve with the weights renewed (Settings.scale_lag_iters).

NRES_L = 16  # align 3, smooth 9, equi 1, pull 3


def lag_weights(x: Array, P: dict[str, Array]) -> Array:
    """1/s0 per slot, s0 = |∇τ| at x (the EPS of ĝ as the floor of s0²)."""
    g = gradient(x, P)
    return 1.0 / jnp.sqrt(jnp.sum(g * g, -1) + EPS)


def lag_state(x: Array, P: dict[str, Array]) -> dict[str, Array]:
    g = gradient(x, P)
    s = jnp.sqrt(jnp.sum(g * g, -1) + EPS)
    gh = g / s[:, None]
    t = jnp.maximum(thickness(x, P), 0.02)
    d = jnp.log(t[P["pi"]]) - jnp.log(t[P["pj"]])
    return {
        "s": s,
        "gh": gh,
        "Pm": jnp.eye(3)[None] - gh[:, :, None] * gh[:, None, :],
        "t": t,
        "tw": (NU + 1) / (NU * P["psd"] ** 2 + d**2),
    }


def _ssqrt(w: Array) -> Array:
    """sqrt(w) for weights w ≥ 0 with a finite derivative at w = 0 (the
    residual weights fade to 0 with ω, and E is differentiated in ω)."""
    pos = w > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, w, 1.0)), 0.0)


def scaled_voxel_residuals(x: Array, P: dict[str, Array], gh: Array, Pm: Array) -> Array:
    """(S1, NRES_L) residuals of the scaled terms with the column direction
    gh (S1, 3) and its projector Pm (S1, 3, 3) given: affine in x when gh
    and Pm are fixed (the lagged form)."""
    nb, nbv, h = P["nb"], P["nbv"], P["h"]
    g = gradient(x, P)
    ev = P["evalw"] * P["valid"]
    fd = _fade(P)
    om = None
    if fd is not None:
        om = fd[0]
        ev = ev * om  # the slot's own residuals fade in with ω
    H = jnp.stack([_deriv(g, nb, nbv, h, a, om) for a in range(3)], -1)  # (S, comp, axis)
    return point_residuals(g, H, gh, Pm, ev, P["inner"], P["coh"], P["nhat"], P["wq"], P.get("wlag"))


def point_residuals(
    g: Array, H: Array, gh: Array, Pm: Array, ev: Array, inner: Array, coh: Array, nhat: Array,
    wq: Array, wlag: Array | None = None,
) -> Array:
    """(n, NRES_L) residuals of the scaled terms at n points, from ∇τ g
    (n, 3), the Hessian H (n, 3, 3), the column direction gh and its
    projector Pm, the point weights ev (n,), the inner flag, the data (coh,
    nhat, wq) and the optional lagged weights. The voxel solve gets g and H
    from its stencils (scaled_voxel_residuals); another representation of τ
    (prototypes/scripts/tau_field.py, a meshfree basis) passes its own
    analytic g and H, so both use the same terms."""
    tis = ev * (1.0 - inner)
    r_al = _ssqrt(tis * KAPPA * coh)[:, None] * jnp.cross(g, nhat)
    PHP = jnp.einsum("sij,sja,sab->sib", Pm, H, Pm)
    r_sm = _ssqrt(tis / (2 * SIGMA_T**2))[:, None, None] * PHP
    lap = H[:, 0, 0] + H[:, 1, 1] + H[:, 2, 2]
    r_eq = _ssqrt(tis * wq) * (lap - 2.0 * jnp.einsum("si,sij,sj->s", gh, H, gh))
    r_pi = _ssqrt(ev / (2 * SIGMA_P**2))[:, None] * (g - gh)
    if wlag is not None:  # the lagged 1/s0 weights (see "Lagged scale weights" above)
        r_al = r_al * wlag[:, None]
        r_sm = r_sm * wlag[:, None, None]
        r_eq = r_eq * wlag**2
    return jnp.concatenate([r_al, r_sm.reshape(-1, 9), r_eq[:, None], r_pi], axis=1)


def lagged_voxel_residuals(x: Array, P: dict[str, Array], L: dict[str, Array]) -> Array:
    """(S1, NRES_L), affine in x."""
    return scaled_voxel_residuals(x, P, L["gh"], L["Pm"])


def gh_Pm(g: Array) -> tuple[Array, Array]:
    """ĝ = g/|g| (EPS in |g|²) and its projector I - ĝĝᵀ."""
    gh = g / jnp.sqrt(jnp.sum(g * g, -1) + EPS)[:, None]
    return gh, jnp.eye(3)[None] - gh[:, :, None] * gh[:, None, :]


def _gh_Pm(x: Array, P: dict[str, Array]) -> tuple[Array, Array]:
    return gh_Pm(gradient(x, P))


def scaled_parts(x: Array, P: dict[str, Array]) -> dict[str, Array]:
    """The scaled energy by term, ĝ = ∇τ/|∇τ| of x (not frozen)."""
    gh, Pm = _gh_Pm(x, P)
    rv = scaled_voxel_residuals(x, P, gh, Pm)
    thick, ts = patch_terms(x, P)
    return {
        "align": jnp.sum(rv[:, :3] ** 2),
        "smooth": jnp.sum(rv[:, 3:12] ** 2),
        "equi": jnp.sum(rv[:, 12] ** 2),
        "pull": jnp.sum(rv[:, 13:16] ** 2),
        "thick": jnp.sum(thick),
        "tsmooth": jnp.sum(ts),
    }


def scaled_energy(x: Array, P: dict[str, Array]) -> Array:
    return sum(scaled_parts(x, P).values())


def scaled_residuals(x: Array, P: dict[str, Array]) -> Array:
    """(S1, NRES_L) the voxel residuals of the scaled energy with ĝ = ĝ(x)
    (not frozen): Σ r² is the voxel part of scaled_energy. The estimator
    "lm_scaled" (Gauss-Newton / LM on the scaled energy itself) uses their
    Jacobian. ĝ at a slot depends on τ within L1 distance 1, H within 2, so
    each residual depends on τ within L1 distance 2, as the original."""
    gh, Pm = _gh_Pm(x, P)
    return scaled_voxel_residuals(x, P, gh, Pm)


def lagged_patch_residuals(
    x: Array, P: dict[str, Array], L: dict[str, Array]
) -> tuple[Array, Array, Array]:
    """r_th (T,), r_ts (Q,) and the linear term Σ t / t0 (scalar)."""
    t = thickness(x, P)
    t0 = L["t"]
    rl = jnp.log(t0) + (t - t0) / t0 - P["logmed"]
    r_th = P["hasroot"] * rl / (jnp.sqrt(2.0) * THICK_SD)
    r_ts = jnp.sqrt(P["pm"] * L["tw"] / 2) * (rl[P["pi"]] - rl[P["pj"]])
    lin = jnp.sum(P["hasroot"] * t / t0)
    return r_th, r_ts, lin


def lagged_energy(x: Array, P: dict[str, Array], L: dict[str, Array]) -> Array:
    """The quadratic model Q(x) of one outer step."""
    rv = lagged_voxel_residuals(x, P, L)
    r_th, r_ts, lin = lagged_patch_residuals(x, P, L)
    return jnp.sum(rv * rv) + jnp.sum(r_th * r_th) + jnp.sum(r_ts * r_ts) + lin


def lagged_parts(x: Array, P: dict[str, Array], L: dict[str, Array]) -> dict[str, Array]:
    rv = lagged_voxel_residuals(x, P, L)
    r_th, r_ts, lin = lagged_patch_residuals(x, P, L)
    return {
        "align": jnp.sum(rv[:, :3] ** 2),
        "smooth": jnp.sum(rv[:, 3:12] ** 2),
        "equi": jnp.sum(rv[:, 12] ** 2),
        "pull": jnp.sum(rv[:, 13:16] ** 2),
        "thick": jnp.sum(r_th**2) + lin,
        "tsmooth": jnp.sum(r_ts**2),
    }


@jax.jit
def j_lag_state(x: Array, P: dict[str, Array]) -> dict[str, Array]:
    return lag_state(x, P)


@jax.jit
def j_lagged_energy_grad(
    x: Array, P: dict[str, Array], L: dict[str, Array]
) -> tuple[Array, Array]:
    return jax.value_and_grad(lagged_energy)(x, P, L)


@jax.jit
def j_lagged_residuals(x: Array, P: dict[str, Array], L: dict[str, Array]) -> Array:
    return lagged_voxel_residuals(x, P, L)


@jax.jit
def j_lagged_jvp(x: Array, P: dict[str, Array], L: dict[str, Array], t: Array) -> Array:
    """J t of the lagged voxel residuals (S1, NRES_L); J does not depend on x."""
    _, jt = jax.jvp(lambda y: lagged_voxel_residuals(y, P, L), (x,), (t,))
    return jt


@jax.jit
def j_lagged_parts(x: Array, P: dict[str, Array], L: dict[str, Array]) -> dict[str, Array]:
    return lagged_parts(x, P, L)


@jax.jit
def j_scaled_energy(x: Array, P: dict[str, Array]) -> Array:
    return scaled_energy(x, P)


@jax.jit
def j_lag_weights(x: Array, P: dict[str, Array]) -> Array:
    return lag_weights(x, P)


@jax.jit
def j_scaled_parts(x: Array, P: dict[str, Array]) -> dict[str, Array]:
    return scaled_parts(x, P)


@jax.jit
def j_scaled_energy_grad(x: Array, P: dict[str, Array]) -> tuple[Array, Array]:
    return jax.value_and_grad(scaled_energy)(x, P)


@jax.jit
def j_scaled_residuals(x: Array, P: dict[str, Array]) -> Array:
    return scaled_residuals(x, P)


@jax.jit
def j_scaled_jvp(x: Array, P: dict[str, Array], t: Array) -> Array:
    """J t of the scaled voxel residuals (S1, NRES_L) at x."""
    _, jt = jax.jvp(lambda y: scaled_residuals(y, P), (x,), (t,))
    return jt


@jax.jit
def j_energy_parts(x: Array, P: dict[str, Array], g0: Array) -> dict[str, Array]:
    return energy_parts(x, P, g0)
