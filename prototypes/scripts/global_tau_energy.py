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


def _deriv(F: Array, nb: Array, nbv: Array, h: Array, a: int) -> Array:
    """Derivative along axis a of F (S,) or (S, k), real neighbours only."""
    m, p = nb[:, a, 0], nb[:, a, 1]
    vm, vp = nbv[:, a, 0], nbv[:, a, 1]
    if F.ndim == 2:
        vm, vp = vm[:, None], vp[:, None]
    Fm, Fp = F[m], F[p]
    both = (Fp - Fm) / (2 * h[a])
    plus = (Fp - F) / h[a]
    minus = (F - Fm) / h[a]
    return jnp.where(vp & vm, both, jnp.where(vp, plus, jnp.where(vm, minus, 0.0)))


def _grad_axis(x: Array, P: dict[str, Array], a: int) -> Array:
    """dτ/dx_a with the ghost value τ = 0 at h/2 across a contact face, or
    in the ridge form the continuous upwind split."""
    nb, nbv, gh, h = P["nb"], P["nbv"], P["ghost"], P["h"]
    m, p = nb[:, a, 0], nb[:, a, 1]
    gm, gp = gh[:, a, 0], gh[:, a, 1]
    vm, vp = nbv[:, a, 0] | gm, nbv[:, a, 1] | gp
    Fm = jnp.where(gm, 0.0, x[m])
    Fp = jnp.where(gp, 0.0, x[p])
    dm = jnp.where(gm, 0.5 * h[a], h[a])
    dp = jnp.where(gp, 0.5 * h[a], h[a])
    both = (Fp - Fm) / (dm + dp)
    plus = (Fp - x) / dp
    minus = (x - Fm) / dm
    one = jnp.where(vp, plus, jnp.where(vm, minus, 0.0))
    base = jnp.where(vp & vm, both, one)
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
