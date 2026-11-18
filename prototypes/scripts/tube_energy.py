"""The tube energy and the batched Newton solve (JAX, float64, CPU or CUDA).

A local problem is a set of S voxels (local slots) with τ values x (S,); a
subset are the unknowns (unk, padded with the dummy slot S - 1); the energy
is summed over the evaluation slots (evalw > 0). Each slot has six neighbour
slots (nb (S, 3, 2): axis, minus/plus) with a validity flag; a missing
neighbour (not stored, outside the crop, or across a contact) gives a
one-sided difference, two missing give a zero derivative on that axis.

Terms per evaluation voxel, in nats (the user's choices B-J, 2026-11-17):
  f      = clip(0.5 - τ / HTAU, 0, 1), the tissue fraction (C)
  stop   = f c, c = +-log((1 - ε) / ε) for the labels' outside / cortex (C)
  align  = f κ coh (1 - (ĝ·n̂)²) (G)
  smooth = f |J P|² / (2 σ_t²), J = ∂ĝ, P = I - ĝĝᵀ (H)
  equi   = f wq (div v)², v = ∇τ / |∇τ|², wq = W_EQ / (2 A_patch) (I)
  pull   = [f (ĝ₀·∇τ - 1)² + (1 - f) max(|∇τ| - 1, 0)²] / (2 σ_p²) (J; E1 c
           and the convex envelope beyond the top, ledger 2026-11-13)
The gradient terms (align, smooth, equi, pull) count only where the slot has
a neighbour on all three axes (valid). Inner slots (inside the white surface,
the roots' support) carry the mm pull only (user, 2026-11-17: without it the
thickness prior lowered τ there and the cortex did not change).
Per tube (F, E): the thickness t = -τ(root) (trilinear over the root's
corners), -log p = log t + r² / (2 SD²) with r = log t - log median, plus
Student-t (ν = 3) terms on r - r_j for the same-bank neighbour patches j.
"""

from __future__ import annotations

from functools import partial
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
TERMS = ("stop", "align", "smooth", "equi", "pull", "thick", "tsmooth")

Array = Any


def _deriv(F: Array, nb: Array, nbv: Array, h: Array, a: int) -> Array:
    """Derivative along axis a of F (S,) or (S, k) at every slot."""
    m, p = nb[:, a, 0], nb[:, a, 1]
    vm, vp = nbv[:, a, 0], nbv[:, a, 1]
    if F.ndim == 2:
        vm, vp = vm[:, None], vp[:, None]
    Fm, Fp = F[m], F[p]
    both = (Fp - Fm) / (2 * h[a])
    plus = (Fp - F) / h[a]
    minus = (F - Fm) / h[a]
    return jnp.where(vp & vm, both, jnp.where(vp, plus, jnp.where(vm, minus, 0.0)))


def gradient(x: Array, P: dict[str, Array]) -> Array:
    return jnp.stack([_deriv(x, P["nb"], P["nbv"], P["h"], a) for a in range(3)], -1)


def voxel_terms(x: Array, P: dict[str, Array], g0hat: Array) -> dict[str, Array]:
    """Per-slot term values (not yet weighted by evalw)."""
    nb, nbv, h = P["nb"], P["nbv"], P["h"]
    g = gradient(x, P)
    s2 = jnp.sum(g * g, -1) + EPS
    s = jnp.sqrt(s2)
    gh = g / s[:, None]
    v = g / s2[:, None]
    divv = sum(_deriv(v[:, a], nb, nbv, h, a) for a in range(3))
    J = jnp.stack([_deriv(gh, nb, nbv, h, a) for a in range(3)], -1)  # (S, 3 comp, 3 axis)
    Pm = jnp.eye(3)[None] - gh[:, :, None] * gh[:, None, :]
    JP = jnp.einsum("sia,sab->sib", J, Pm)
    f = jnp.clip(0.5 - x / P["htau"], 0.0, 1.0)
    valid = P["valid"]
    pull_in = (jnp.sum(g0hat * g, -1) - 1.0) ** 2
    pull_out = jnp.maximum(s - 1.0, 0.0) ** 2
    cosn = jnp.sum(gh * P["nhat"], -1)
    tis = 1.0 - P["inner"]  # inner voxels (inside the white surface): mm pull only
    return {
        "stop": tis * f * P["c"],
        "align": tis * valid * f * KAPPA * P["coh"] * (1.0 - cosn**2),
        "smooth": tis * valid * f * jnp.sum(JP * JP, (-1, -2)) / (2 * SIGMA_T**2),
        "equi": tis * valid * f * P["wq"] * divv**2,
        "pull": valid * (f * pull_in + (1 - f) * pull_out) / (2 * SIGMA_P**2),
    }


def thickness(x: Array, P: dict[str, Array]) -> Array:
    """Thickness -τ(root) of each root row (T,)."""
    return -jnp.sum(P["rw"] * x[P["rslot"]], -1)


def tube_terms(x: Array, P: dict[str, Array]) -> dict[str, Array]:
    """Thickness prior of the root rows (T): rslot, rw (T, 8); logmed,
    hasroot (T,); rj, sj, mj (T, K) the frozen neighbour residuals."""
    t = jnp.maximum(thickness(x, P), 0.02)
    lt = jnp.log(t)
    r = lt - P["logmed"]
    thick = P["hasroot"] * (lt + r**2 / (2 * THICK_SD**2))
    d = r[:, None] - P["rj"]
    ts = (NU + 1) / 2 * jnp.log1p(d**2 / (NU * P["sj"] ** 2))
    return {"thick": jnp.sum(thick), "tsmooth": jnp.sum(P["hasroot"][:, None] * P["mj"] * ts)}


def g0_of(x: Array, P: dict[str, Array]) -> Array:
    g = gradient(x, P)
    return jax.lax.stop_gradient(g / jnp.sqrt(jnp.sum(g * g, -1) + EPS)[:, None])


def energy_parts(x: Array, P: dict[str, Array], g0hat: Array) -> dict[str, Array]:
    vt = voxel_terms(x, P, g0hat)
    out = {k: jnp.sum(P["evalw"] * val) for k, val in vt.items()}
    out.update(tube_terms(x, P))
    return out


def energy(x: Array, P: dict[str, Array], g0hat: Array) -> Array:
    parts = energy_parts(x, P, g0hat)
    return sum(parts.values())


def _embed(u: Array, P: dict[str, Array]) -> Array:
    return P["x0"].at[P["unk"]].set(u)


@partial(jax.jit, static_argnames=("iters",))
def newton_batch(P: dict[str, Array], iters: int = 20, tol: Array = 1e-6) -> tuple[Array, ...]:
    """Damped Newton on each problem of a batch (leading axis). Returns
    u (B, U), E0, E1 (B,), iterations used (B,), and E parts before/after."""

    def one(P: dict[str, Array]) -> tuple[Array, ...]:
        g0 = g0_of(P["x0"], P)
        E = lambda u: energy(_embed(u, P), P, g0)  # noqa: E731
        u0 = P["x0"][P["unk"]]
        e0 = E(u0)
        pad = 1.0 - P["umask"]
        nev = jnp.maximum(jnp.sum(P["evalw"]), 1.0)
        alphas = 0.5 ** jnp.arange(20.0)

        def body(i: Array, st: tuple[Array, ...]) -> tuple[Array, ...]:
            u, e, done, it = st
            gr = jax.grad(E)(u) * P["umask"]
            H = jax.hessian(E)(u) * (P["umask"][:, None] * P["umask"][None, :]) + jnp.diag(pad)
            w, V = jnp.linalg.eigh(H)
            aw = jnp.abs(w)
            aw = jnp.maximum(aw, 1e-6 * jnp.max(aw) + 1e-12)
            d = -(V @ ((V.T @ gr) / aw)) * P["umask"]
            es = jax.vmap(lambda a: E(u + a * d))(alphas)
            ok = es <= e + 1e-4 * alphas * jnp.dot(gr, d)
            k = jnp.argmax(ok)
            anyok = jnp.any(ok) & ~done
            un = jnp.where(anyok, u + alphas[k] * d, u)
            en = jnp.where(anyok, es[k], e)
            done = done | ~anyok | (e - en < tol * nev)
            return un, en, done, it + 1

        u, e1, _, it = jax.lax.while_loop(
            lambda st: (st[3] < iters) & ~st[2], lambda st: body(0, st), (u0, e0, False, 0)
        )
        p0 = energy_parts(P["x0"], P, g0)
        p1 = energy_parts(_embed(u, P), P, g0)
        return u, e0, e1, it, jnp.stack([p0[k] for k in TERMS]), jnp.stack([p1[k] for k in TERMS])

    return jax.vmap(one)(P)


@jax.jit
def energy_along(P: dict[str, Array], u1: Array, alphas: Array) -> Array:
    """E at u0 + α (u1 - u0) for each α (B, A), with g0 from x0."""

    def one(P: dict[str, Array], u1: Array) -> Array:
        g0 = g0_of(P["x0"], P)
        u0 = P["x0"][P["unk"]]
        return jax.vmap(lambda a: energy(_embed(u0 + a * (u1 - u0), P), P, g0))(alphas)

    return jax.vmap(one)(P, u1)


@jax.jit
def global_parts(P: dict[str, Array]) -> Array:
    """Per-slot voxel terms (5, S) of one large problem (g0 from x0)."""
    g0 = g0_of(P["x0"], P)
    vt = voxel_terms(P["x0"], P, g0)
    return jnp.stack([P["evalw"] * vt[k] for k in TERMS[:5]])


@partial(jax.jit, static_argnames=("iters",))
def shift_batch(P: dict[str, Array], iters: int = 20, tol: Array = 1e-6) -> tuple[Array, ...]:
    """Coarse correction: one shift s per problem added to all its unknowns
    (umask); damped 1D Newton. Returns s (B,), E0, E1, E parts before/after."""

    def one(P: dict[str, Array]) -> tuple[Array, ...]:
        g0 = g0_of(P["x0"], P)
        u0 = P["x0"][P["unk"]]
        E = lambda s: energy(_embed(u0 + s * P["umask"], P), P, g0)  # noqa: E731
        e0 = E(0.0)
        nev = jnp.maximum(jnp.sum(P["evalw"]), 1.0)
        alphas = 0.5 ** jnp.arange(20.0)

        def body(i: Array, st: tuple[Array, ...]) -> tuple[Array, ...]:
            s, e, done = st
            gr = jax.grad(E)(s)
            H = jax.grad(jax.grad(E))(s)
            d = -gr / jnp.maximum(jnp.abs(H), 1e-12)
            es = jax.vmap(lambda a: E(s + a * d))(alphas)
            ok = es <= e + 1e-4 * alphas * gr * d
            k = jnp.argmax(ok)
            anyok = jnp.any(ok) & ~done
            sn = jnp.where(anyok, s + alphas[k] * d, s)
            en = jnp.where(anyok, es[k], e)
            done = done | ~anyok | (e - en < tol * nev)
            return sn, en, done

        s, e1, _ = jax.lax.fori_loop(0, iters, body, (0.0, e0, False))
        p0 = energy_parts(P["x0"], P, g0)
        p1 = energy_parts(_embed(u0 + s * P["umask"], P), P, g0)
        return s, e0, e1, jnp.stack([p0[k] for k in TERMS]), jnp.stack([p1[k] for k in TERMS])

    return jax.vmap(one)(P)


def ls_residuals(x: Array, P: dict[str, Array], g0hat: Array, fw: Array, tw: Array) -> Array:
    """Least-squares residuals of the energy with frozen weights, for
    Gauss-Newton diagnostics: fw (S,) the tissue fraction f and tw (T, K) the
    Student-t IRLS weights w = (ν + 1) / (ν s² + d²), both from the current
    iterate. Σ r² equals the energy without the stop term, without the
    log-normal's log t, and with each Student-t term replaced by its IRLS
    quadratic w d² / 2 (same gradient at the iterate when f is constant)."""
    nb, nbv, h = P["nb"], P["nbv"], P["h"]
    g = gradient(x, P)
    s2 = jnp.sum(g * g, -1) + EPS
    s = jnp.sqrt(s2)
    gh = g / s[:, None]
    v = g / s2[:, None]
    divv = sum(_deriv(v[:, a], nb, nbv, h, a) for a in range(3))
    J = jnp.stack([_deriv(gh, nb, nbv, h, a) for a in range(3)], -1)
    Pm = jnp.eye(3)[None] - gh[:, :, None] * gh[:, None, :]
    JP = jnp.einsum("sia,sab->sib", J, Pm)
    ev = P["evalw"] * P["valid"]
    tis = ev * (1.0 - P["inner"])
    r_al = jnp.sqrt(tis * fw * KAPPA * P["coh"])[:, None] * jnp.cross(gh, P["nhat"])
    r_sm = jnp.sqrt(tis * fw / (2 * SIGMA_T**2))[:, None, None] * JP
    r_eq = jnp.sqrt(tis * fw * P["wq"]) * divv
    r_pi = jnp.sqrt(ev * fw / (2 * SIGMA_P**2)) * (jnp.sum(g0hat * g, -1) - 1.0)
    r_po = jnp.sqrt(ev * (1 - fw) / (2 * SIGMA_P**2)) * jnp.maximum(s - 1.0, 0.0)
    t = jnp.maximum(thickness(x, P), 0.02)
    r = jnp.log(t) - P["logmed"]
    r_th = P["hasroot"] * r / (jnp.sqrt(2.0) * THICK_SD)
    r_ts = jnp.sqrt(P["hasroot"][:, None] * P["mj"] * tw / 2) * (r[:, None] - P["rj"])
    return jnp.concatenate([r_al.ravel(), r_sm.ravel(), r_eq, r_pi, r_po, r_th, r_ts.ravel()])


def irls_weights(x: Array, P: dict[str, Array]) -> tuple[Array, Array]:
    """fw = f(τ) and the Student-t IRLS weights at x."""
    fw = jnp.clip(0.5 - x / P["htau"], 0.0, 1.0)
    t = jnp.maximum(thickness(x, P), 0.02)
    d = (jnp.log(t) - P["logmed"])[:, None] - P["rj"]
    tw = (NU + 1) / (NU * P["sj"] ** 2 + d**2)
    return fw, tw
