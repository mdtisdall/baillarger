"""T5 of docs/plans/top-surface.md: Σ and τ with all the costs, from the
white surface (user, 2026-11-19: no separate start phase, "use all the
costs").

The energy is E(τ, Σ) = E_τ(τ; Σ) + E_stop(Σ):
  - E_τ: the scaled τ energy (global_tau_energy.scaled_energy; user,
    2026-11-19, option A: no term divides by |∇τ|), τ = 0 at the crossings
    of Σ (the ghost at θh), at the τ that the estimator tau_estimator
    finds behind Σ (the rim at τ0): "lm_scaled" (LM on the scaled energy
    itself, a descent method; user, 2026-11-19) or "lagged" (the lagged
    quadratic form, which can cycle).
  - E_stop (T4): Σ c f over the voxels, c = -log(19) in the cortex label,
    +log(19) in label 0, 0 elsewhere; f = clip(0.5 - s/h, 0, 1) with s the
    signed distance from the voxel centre to Σ (positive outside). s comes
    from an axis crossing within one voxel (user, 2026-11-19: "Axis
    crossings"): on the axis whose crossing face has the largest |n_a|,
    s = (k - t) n_a (n the unit outward face normal, t the crossing, k the
    voxel centre on that axis). A voxel with no crossing within one voxel
    has f = 1 inside Σ and 0 outside.

One step (T5, user, 2026-11-19):
  1. ∂E/∂V at the current (τ, Σ), with τ fixed: through θ of each ghost
     side (∂E_τ/∂θ by JAX) and through t and n_a of each stop crossing.
     Both are functions of the crossing face's three vertices.
  2. A trial Σ' = Σ - α ∂E/∂V, each vertex at most h/2, halved up to
     HALVINGS times per vertex (then refused) until it keeps the gap and
     makes no intersection and no new fold (T2).
  3. τ' behind Σ' (lagged, warm start from τ). Accept when E(τ', Σ') < E:
     α doubles. Else Σ stays, α halves, and step 2 is tried again (at most
     max_tries times; then the run ends "no decrease").
  4. After an accepted step: the tangential relaxation and the remeshing
     (T7), then τ and E again (the relaxation changes E a little; logged).
The run ends when the accepted step is below stop_vox × h for every
vertex ("step below stop_vox": not a test of convergence, α may only have
become small), with "no decrease", or at max_steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import global_tau_energy as GE
import global_tau_prep as TP
import global_tau_solver as GS
import jax
import jax.numpy as jnp
import numpy as np
import top_surface as TS
import top_tau as TT

from baillarger import mesh_checks as MC

HALVINGS = TS.HALVINGS


@dataclass
class SolveSettings:
    gap_vox: float = 0.25  # T2
    split_ratio: float = 2.0  # T7
    collapse_ratio: float = 0.5  # T7
    fold_cos: float = MC.FOLD_COS
    step_max_vox: float = 0.5  # T5: at most h/2 per vertex
    step0_vox: float = 0.25  # the first trial: the largest vertex step h/4
    max_steps: int = 200
    max_tries: int = 8
    stop_vox: float = 1e-3  # converged: every accepted vertex step below this × h
    tau_estimator: str = "lm_scaled"  # or "lagged"
    tau_iters: int = 10  # LM iterations or lagged outer steps per τ solve (warm start)
    tau_tol_mm: float = 1e-3  # lagged only: stops when max |Δτ| is below this
    rim_fixed: bool = True  # the crop-edge voxels at τ0 (global_tau_solver)
    #: the tangential relaxation after an accepted step (T7): "all" vertices,
    #: "none", or only the "flagged" ones (a face of theirs has its smallest
    #: altitude below the step cap step_max_vox × h: one step could flip it)
    relax: str = "all"
    measure_relax: bool = False  # also E right after the relaxation (one more τ solve)
    #: the mesh after an accepted step: "remesh" (relax, then uniform splits
    #: and collapses, T7) or "refine" (no relaxation, no collapse: longest-edge
    #: splits of the faces that an indicator marks, see refine_marks)
    mesh: str = "remesh"
    refine_var: float = 1.0  # force variation: max |g_i - g_j| / max |g_i| over a face
    refine_held: int = 2  # a vertex refused in this many steps in a row
    refine_min_vox: float = 1.0  # no split makes an edge shorter than this × h


def stop_weights(labels: np.ndarray) -> np.ndarray:
    c = np.zeros(labels.shape)
    c[labels == TP.CORTEX] = -GE.LOG_ODDS
    c[labels == TP.OUTSIDE] = GE.LOG_ODDS
    return c


# ------------------------------------------------------------ crossings as functions of Σ


def _cross(Vf: Any, uw: Any, a: int) -> Any:
    """t (the crossing of the line (u, w) along axis a, voxel units) and n_a
    (the a component of the unit face normal) for a face Vf (3, 3)."""
    b, c = (a + 1) % 3, (a + 2) % 3
    p0, p1, p2 = Vf[0], Vf[1], Vf[2]
    ub, wc = uw[0] + TT.EPS_B, uw[1] + TT.EPS_C
    x0, y0, x1, y1, x2, y2 = p0[b], p0[c], p1[b], p1[c], p2[b], p2[c]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    l0 = ((y1 - y2) * (ub - x2) + (x2 - x1) * (wc - y2)) / den
    l1 = ((y2 - y0) * (ub - x2) + (x0 - x2) * (wc - y2)) / den
    l2 = 1.0 - l0 - l1
    t = l0 * p0[a] + l1 * p1[a] + l2 * p2[a]
    n = jnp.cross(p1 - p0, p2 - p0)
    return jnp.stack([t, n[a] / jnp.linalg.norm(n)])


@partial(jax.jit, static_argnums=2)
def _cross_and_jac(Vf: Any, uw: Any, a: int) -> tuple[Any, Any]:
    """Per crossing: (t, n_a) (k, 2) and their derivatives (k, 2, 3, 3)."""
    f = jax.vmap(lambda v, q: _cross(v, q, a))
    j = jax.vmap(jax.jacfwd(lambda v, q: _cross(v, q, a)))
    return f(Vf, uw), j(Vf, uw)


def crossing_derivs(
    P: np.ndarray, F: np.ndarray, vox: np.ndarray, a: np.ndarray, face: np.ndarray, uw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(t, n_a) per crossing (k, 2) and ∂(t, n_a)/∂V (k, 2, 3, 3) in mm
    (P in voxel units; the crossing's line along axis a at uw)."""
    k = len(face)
    val = np.zeros((k, 2))
    jac = np.zeros((k, 2, 3, 3))
    for ax in range(3):
        sel = np.flatnonzero(a == ax)
        if not len(sel):
            continue
        v, j = _cross_and_jac(jnp.asarray(P[F[face[sel]]]), jnp.asarray(uw[sel].astype(float)), ax)
        val[sel] = np.asarray(v)
        jac[sel] = np.asarray(j)
    jac /= vox[None, None, None, :]  # ∂/∂(mm) = ∂/∂(voxel) / vox
    return val, jac


def _line_uw(ijk: np.ndarray, a: np.ndarray) -> np.ndarray:
    b, c = (a + 1) % 3, (a + 2) % 3
    r = np.arange(len(a))
    return np.c_[ijk[r, b], ijk[r, c]]


def stop_energy(
    c: np.ndarray, geo: dict[str, np.ndarray], P: np.ndarray, F: np.ndarray, vox: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """E_stop and, for the voxels on the ramp (0 < f < 1), the crossing used
    and ∂E_stop/∂(t, n_a). Only voxels with c ≠ 0 count."""
    inside = geo["inside"]
    cross, theta, face = geo["cross"], geo["theta"], geo["face"]
    live = c != 0
    has = cross.any((3, 4)) & live
    f = np.where(inside & live, 1.0, 0.0)
    ijk = np.argwhere(has)
    cr = cross[has]  # (k, 3, 2)
    th = theta[has]
    fc = face[has]
    # n_a of every candidate crossing (numpy, unit normals of the faces)
    Fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    Fn /= np.maximum(np.linalg.norm(Fn, axis=1, keepdims=True), 1e-300)
    nabs = np.zeros(cr.shape)
    for ax in range(3):
        for sd in range(2):
            nabs[:, ax, sd] = np.where(cr[:, ax, sd], np.abs(Fn[np.maximum(fc[:, ax, sd], 0), ax]), -1.0)
    best = nabs.reshape(len(ijk), 6).argmax(1)
    ba, bs = best // 2, best % 2
    r = np.arange(len(ijk))
    side = np.where(bs == 1, 1.0, -1.0)
    tk = side * th[r, ba, bs]  # t - k
    bf = fc[r, ba, bs]
    nb = Fn[bf, ba]
    fr = 0.5 + tk * nb
    fv = np.clip(fr, 0.0, 1.0)
    f[tuple(ijk.T)] = fv
    cv = c[tuple(ijk.T)]
    ramp = (fr > 0.0) & (fr < 1.0)
    e = float((c * f).sum())
    return e, {
        "ijk": ijk[ramp],
        "a": ba[ramp],
        "face": bf[ramp],
        "dE_dt": (cv * nb)[ramp],  # f = 0.5 + (t - k) n_a
        "dE_dn": (cv * tk)[ramp],
        "ramp_voxels": int(ramp.sum()),
    }


# ------------------------------------------------------------ τ behind Σ and the energy


@jax.jit
def _energy_and_dgdist(x: Any, P: dict[str, Any]) -> tuple[Any, Any]:
    """The scaled E_τ and ∂E_τ/∂gdist (S1, 3, 2)."""

    def e(gd: Any) -> Any:
        return GE.scaled_energy(x, {**P, "gdist": gd})

    return jax.value_and_grad(e)(P["gdist"])


@jax.jit
def _energy_parts(x: Any, P: dict[str, Any]) -> dict[str, Any]:
    return GE.scaled_parts(x, P)


@dataclass
class State:
    V: np.ndarray
    E: float
    E_tau: float
    E_stop: float
    grad: np.ndarray  # (n, 3) ∂E/∂V, mm
    tau: np.ndarray  # grid, nan off the domain
    info: dict[str, Any]


class Evaluator:
    """τ behind Σ (tau_estimator, warm start) and E with its derivative in V."""

    def __init__(self, site: TP.Site, s: SolveSettings):
        self.site, self.s = site, s
        self.gs = GS.Settings(
            face_test=False, estimator=s.tau_estimator, rim_fixed=s.rim_fixed,
            lag_iters=s.tau_iters, lag_tol=s.tau_tol_mm, lm_iters=s.tau_iters,
        )
        self.pt = TP.make_patches(site, self.gs.patch_mm)
        self.centre_vox = np.array(site.labels.shape) / 2.0
        self.c = stop_weights(site.labels)

    def __call__(self, V: np.ndarray, F: np.ndarray, tau_prev: np.ndarray | None) -> State:
        t0 = time.time()
        site = self.site
        vox = site.vox.astype(float)
        P = V / vox
        sig, ginfo = TT.sigma_top(site, V, F, tau_prev, self.gs)
        geo = sig.geo
        assert geo is not None
        e_stop, sd = stop_energy(self.c, geo, P, F, vox)
        shape = site.labels.shape
        grad = np.zeros_like(V)
        tau = np.full(shape, np.nan)
        info: dict[str, Any] = {"sigma": ginfo, "ramp_voxels": sd["ramp_voxels"]}
        e_tau = 0.0
        rec_a, rec_f, rec_uw, rec_dt, rec_dn = [sd["a"]], [sd["face"]], [_line_uw(sd["ijk"], sd["a"])], [sd["dE_dt"]], [sd["dE_dn"]]
        if sig.domain.any():
            gt = GS.GlobalTau(site, self.pt, self.gs, self.centre_vox, sigma=sig)
            sinfo, _ = gt.solve_global("sigma")
            pb = gt._pb
            x = np.zeros(pb.S1)
            x[: pb.S] = gt.tau
            xj = jnp.asarray(x)
            e, dgd = _energy_and_dgdist(xj, pb.P)
            e_tau = float(e)
            parts = {k: float(v) for k, v in _energy_parts(xj, pb.P).items()}
            dgd = np.asarray(dgd)[: pb.S]
            ijk = gt.G.coords[: pb.S] - GS.PAD
            X = tuple(ijk.T)
            sg = sig.ghost[X]  # (S, 3, 2) the ghost sides of Σ
            th = geo["theta"][X]
            use = sg & (th > TT.THETA_MIN) & (th <= 1.0)
            s_i, a_i, d_i = np.nonzero(use)
            side = np.where(d_i == 1, 1.0, -1.0)  # θ = side (t - k): ∂θ/∂t = side
            rec_a.append(a_i)
            rec_f.append(geo["face"][X][s_i, a_i, d_i])
            rec_uw.append(_line_uw(ijk[s_i], a_i))
            rec_dt.append(dgd[s_i, a_i, d_i] * side)
            rec_dn.append(np.zeros(len(s_i)))
            tau = gt.G.grid(gt.tau)
            its = sinfo["iters"]
            info.update({
                "tau_s": sinfo["s"],
                "lag_iters": len(its),
                "cg_iters": [r["cg_iters"] for r in its],
                "last_step_mm": its[-1]["step_mm"] if its else 0.0,
                "s0_below_0.1": its[-1].get("s0_below_0.1", -1) if its else 0,
                "e_start": sinfo["e0"],
                "unknowns": sinfo["n_unk"],
                "at_bound": int((gt.tau == 0.0).sum()),
                "tau_min": float(gt.tau.min()),
                "ghost_derivs": int(len(s_i)),
                "parts": parts,
            })
        a = np.concatenate(rec_a).astype(np.int64)
        fcs = np.concatenate(rec_f).astype(np.int64)
        uw = np.concatenate(rec_uw).astype(np.int64)
        dt = np.concatenate(rec_dt)
        dn = np.concatenate(rec_dn)
        if len(fcs):
            _, jac = crossing_derivs(P, F, vox, a, fcs, uw)
            gv = dt[:, None, None] * jac[:, 0] + dn[:, None, None] * jac[:, 1]  # (k, 3, 3)
            bad = ~np.isfinite(gv).all((1, 2))
            if bad.any():  # diagnose, do not hide: the run stops here
                raise FloatingPointError(
                    f"{int(bad.sum())} crossings with a non-finite ∂E/∂V; E_tau {e_tau}, info {info}"
                )
            np.add.at(grad, F[fcs].ravel(), gv.reshape(-1, 3))
        info["s"] = time.time() - t0
        return State(V.copy(), e_tau + e_stop, e_tau, e_stop, grad, tau, info)


# ------------------------------------------------------------ mesh quality


def min_altitude(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """The smallest altitude of each face (2 × area / its longest edge)."""
    e = np.stack([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 1]], V[F[:, 0]] - V[F[:, 2]]], 1)
    area2 = np.linalg.norm(np.cross(e[:, 0], -e[:, 2]), axis=1)
    return area2 / np.maximum(np.linalg.norm(e, axis=2).max(1), 1e-300)


def flagged_vertices(V: np.ndarray, F: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertices with a face whose smallest altitude is below limit; and the
    altitudes."""
    alt = min_altitude(V, F)
    fl = np.zeros(len(V), bool)
    fl[F[alt < limit].ravel()] = True
    return fl, alt


def quality(V: np.ndarray, F: np.ndarray, h: float, limit: float) -> dict[str, Any]:
    fl, alt = flagged_vertices(V, F, limit)
    return {
        "alt_over_h_p1_5_50": [float(v) for v in np.percentile(alt / h, [1, 5, 50])],
        "faces_below_limit": int((alt < limit).sum()),
        "flagged_vertices": int(fl.sum()),
    }


# ------------------------------------------------------------ the constrained Σ step


def constrained_move(
    m: TS.Mesh, delta: np.ndarray, chk: MC.Checker, fold_cos: float
) -> tuple[np.ndarray, dict[str, int]]:
    """Each vertex's move delta (n, 3) is halved up to HALVINGS times, then
    refused, until it keeps the gap and makes no intersection and no new
    fold (as top_surface.expand_step). Returns the moves taken."""
    d = delta.copy()
    V0 = m.V.copy()
    mov = np.abs(d).sum(1) > 0
    changed = mov.copy()
    halved = np.zeros(len(d), bool)
    refused = np.zeros(len(d), bool)
    k = 0
    while changed.any():
        Vn = V0 + d
        bad, _, _, _ = chk.check(Vn, changed, np.zeros(len(d), bool))
        fb, _, _ = TS.new_folds(m, Vn, V0, changed, fold_cos)
        badm = (bad | fb) & (np.abs(d).sum(1) > 0)
        if not badm.any():
            break
        if k < HALVINGS:
            d[badm] *= 0.5
            halved |= badm
        else:
            d[badm] = 0.0
            refused |= badm
        changed = badm
        k += 1
    m.held = np.where(refused, m.held + 1, np.where(mov & ~refused, 0.0, m.held))
    return d, {
        "moved": int((np.abs(d).sum(1) > 0).sum()),
        "halved": int(halved.sum()),
        "refused": int(refused.sum()),
        "_halved": halved,
        "_refused": refused,
    }


def refine_marks(
    m: TS.Mesh, grad: np.ndarray, halved: np.ndarray, vox: np.ndarray, s: SolveSettings
) -> tuple[np.ndarray, dict[str, int]]:
    """The faces to split, by three indicators (plan, coarse to fine):
      - force: the force -∂E/∂V varies strongly within the face, max over
        its vertex pairs of |g_i - g_j| >= refine_var × max |g_i|;
      - facing: a vertex of the face has a facing part of Σ
        (top_surface.contacts) nearer than the face's longest edge;
      - held: a vertex of the face was refused in refine_held steps in a
        row, or halved in this step.
    Returns the marked faces and the count for each indicator."""
    F = m.F
    G = grad[F]  # (m, 3, 3)
    gmax = np.linalg.norm(G, axis=2).max(1)
    dif = np.stack([np.linalg.norm(G[:, i] - G[:, (i + 1) % 3], axis=1) for i in range(3)], 1).max(1)
    force = (gmax > 0) & (dif >= s.refine_var * gmax)
    longest = np.linalg.norm(m.V[F] - m.V[F][:, [1, 2, 0]], axis=2).max(1)
    fac = TS.contacts(m, vox)["facing_mm"]
    facing = (fac[F] < longest[:, None]).any(1)
    hv = (m.held >= s.refine_held) | halved
    held = hv[F].any(1)
    return force | facing | held, {
        "force": int(force.sum()), "facing": int(facing.sum()), "held": int(held.sum()),
    }


def refine(m: TS.Mesh, marked: np.ndarray, min_len: float) -> int:
    """Splits, at the midpoint (Σ's shape does not change), the longest edge
    of each marked face if it is at least 2 × min_len (top_surface._split
    with a per-vertex target: the shortest longest-edge of the marked faces
    at the vertex; unmarked vertices: no split). Returns the splits."""
    if not marked.any():
        return 0
    F = m.F
    longest = np.linalg.norm(m.V[F] - m.V[F][:, [1, 2, 0]], axis=2).max(1)
    size = np.full(len(m.V), np.inf)
    fm = np.flatnonzero(marked & (longest >= 2.0 * min_len))
    if not len(fm):
        return 0
    np.minimum.at(size, F[fm].ravel(), np.repeat(longest[fm] * (1.0 - 1e-9), 3))
    return TS._split(m, 1.0, size)


@dataclass
class SolveRun:
    site: TP.Site
    s: SolveSettings
    progress: Any = None
    snapshot: Any = None  # callable(name, mesh, disp, meta, tau)
    start: TS.Mesh | None = None  # the initial Σ (default: the white mesh, remeshed)
    log: list[dict[str, Any]] = field(default_factory=list)
    ended: str = ""
    contacts: dict[str, np.ndarray] = field(default_factory=dict)

    def run(self) -> TS.Mesh:
        site, s = self.site, self.s
        h = float(site.vox.mean())
        gap = s.gap_vox * h
        m = TS.start_mesh(site) if self.start is None else self.start
        u = TS._edges(m.F)
        med = float(np.median(np.linalg.norm(m.V[u[:, 0]] - m.V[u[:, 1]], axis=1)))
        self.median_edge = med
        lmax, lmin = s.split_ratio * med, s.collapse_ratio * med
        lo = np.zeros(3)
        hi = (np.array(site.labels.shape) - 1) * site.vox
        ev = Evaluator(site, s)
        t0 = time.time()
        white = TS.counts(m, gap, s.fold_cos)
        sp, co = (0, 0) if s.mesh == "refine" else TS.remesh(m, gap, lmax, lmin, s.fold_cos)
        cur = ev(m.V, m.F, None)
        self.initial = {"quality": quality(m.V, m.F, h, s.step_max_vox * h), "vertices": len(m.V), "faces": len(m.F), "white": white, "splits": sp,
                        "collapses": co, "E": cur.E, "E_tau": cur.E_tau, "E_stop": cur.E_stop,
                        "eval": cur.info, "s": time.time() - t0}
        if self.progress:
            self.progress(f"start: E {cur.E:.1f} (τ {cur.E_tau:.1f}, stop {cur.E_stop:.1f}), {cur.info['s']:.1f} s")
        if self.snapshot:
            self.snapshot("solve-000", m, np.zeros(len(m.V)), self._meta(0, cur, {}), cur.tau)
        alpha = None
        self.ended = "max_steps"
        for k in range(1, s.max_steps + 1):
            t1 = time.time()
            g = cur.grad
            gn = np.linalg.norm(g, axis=1)
            if alpha is None:
                alpha = s.step0_vox * h / max(float(gn.max()), 1e-300)
            chk = MC.Checker(m.top, m.V, gap, slack=s.step_max_vox * h)
            tries = []
            accepted = False
            for _ in range(s.max_tries):
                d = -alpha * g
                ln = np.linalg.norm(d, axis=1)
                cap = s.step_max_vox * h
                d *= np.minimum(1.0, cap / np.maximum(ln, 1e-300))[:, None]
                Vn = m.V + d
                d[np.any((Vn < lo) | (Vn > hi), axis=1)] = 0.0  # never out of the crop
                saved = m.save()
                fl, _ = flagged_vertices(m.V, m.F, s.step_max_vox * h)
                dd, cinfo = constrained_move(m, d, chk, s.fold_cos)
                cinfo["halved_flagged"] = int((cinfo["_halved"] & fl).sum())
                cinfo["refused_flagged"] = int((cinfo["_refused"] & fl).sum())
                halved = cinfo.pop("_halved")
                del cinfo["_refused"]
                trial = ev(m.V + dd, m.F, cur.tau)
                tries.append({"alpha": alpha, "E": trial.E, "E_tau": trial.E_tau, "E_stop": trial.E_stop,
                              "max_step_vox": float(np.linalg.norm(dd, axis=1).max() / h), **cinfo,
                              "eval_s": trial.info["s"], "lag_iters": trial.info.get("lag_iters", 0)})
                if trial.E < cur.E:
                    accepted = True
                    m.V = m.V + dd
                    m.travel = m.travel + np.linalg.norm(dd, axis=1)
                    disp = np.linalg.norm(dd, axis=1)
                    alpha *= 2.0
                    break
                m.restore(saved)
                alpha *= 0.5
            if not accepted:
                self.ended = "no decrease"
                self.log.append({"step": k, "tries": tries, "accepted": False, "s": time.time() - t1})
                if self.progress:
                    self.progress(f"step {k}: no decrease in {len(tries)} tries; E {cur.E:.1f}")
                break
            e_trial = trial.E
            q_before = quality(m.V, m.F, h, s.step_max_vox * h)
            marks: dict[str, int] = {}
            if s.mesh == "refine" or s.relax == "none":
                r, nrel = np.zeros(len(m.V)), 0
            else:
                only = flagged_vertices(m.V, m.F, s.step_max_vox * h)[0] if s.relax == "flagged" else None
                r, nrel = TS.relax(m, chk, s.fold_cos, only=only)
            m.last = np.sqrt(disp**2 + r**2)
            e_relax = None
            if nrel and s.measure_relax:
                e_relax = ev(m.V, m.F, trial.tau).E  # E after the relaxation alone (diagnostic)
            if s.mesh == "refine":
                mk, marks = refine_marks(m, trial.grad, halved, site.vox, s)
                sp, co = refine(m, mk, s.refine_min_vox * h), 0
            else:
                sp, co = TS.remesh(m, gap, lmax, lmin, s.fold_cos)
            cur = trial if sp == 0 and co == 0 and nrel == 0 else ev(m.V, m.F, trial.tau)
            c = TS.counts(m, gap, s.fold_cos)
            rec = {"step": k, "tries": tries, "accepted": True, "E_trial": e_trial, "E_relaxed": e_relax,
                   "E": cur.E, "quality_before_relax": q_before,
                   "quality": quality(m.V, m.F, h, s.step_max_vox * h),
                   "E_tau": cur.E_tau, "E_stop": cur.E_stop, "relaxed": nrel, "splits": sp, "collapses": co,
                   "marks": marks,
                   **c, "vertices": len(m.V), "faces": len(m.F), "eval": cur.info,
                   "max_step_vox": float(disp.max() / h), "s": time.time() - t1}
            self.log.append(rec)
            if self.progress:
                ti = cur.info
                self.progress(
                    f"step {k}: E {cur.E:.1f} (τ {cur.E_tau:.1f}, stop {cur.E_stop:.1f}; trial {e_trial:.1f}), "
                    f"tries {len(tries)}, α {tries[-1]['alpha']:.3g}, max step {disp.max() / h:.3f} h, "
                    f"moved {tries[-1]['moved']} halved {tries[-1]['halved']} refused {tries[-1]['refused']}, "
                    f"τ: {ti.get('unknowns', 0)} unknowns, lag {ti.get('lag_iters', 0)} (last Δτ {ti.get('last_step_mm', 0):.1e} mm), "
                    f"bound {ti.get('at_bound', 0)}, ramp {ti['ramp_voxels']}, "
                    f"halved/refused at flagged {tries[-1]['halved_flagged']}/{tries[-1]['refused_flagged']}, "
                    f"relaxed {nrel} (E after relax {e_relax}), marks {marks}, splits {sp} collapses {co}, "
                    f"quality before relax {q_before}, after remesh {rec['quality']}, "
                    f"intersections {c['intersecting_faces']} gap {c['gap_vertices']}, {rec['s']:.1f} s"
                )
            if self.snapshot:
                self.snapshot(f"solve-{k:03d}", m, m.last, self._meta(k, cur, rec), cur.tau)
            if disp.max() < s.stop_vox * h:
                # not a test of convergence: α may only have become small
                self.ended = "step below stop_vox"
                break
        self.contacts = TS.contacts(m, site.vox)
        self.final = cur
        return m

    @staticmethod
    def _meta(k: int, st: State, rec: dict[str, Any]) -> dict[str, Any]:
        out = {"step": k, "energy": st.E, "E_tau": st.E_tau, "E_stop": st.E_stop}
        out.update({kk: v for kk, v in rec.items() if kk in ("max_step_vox", "s")})
        if "parts" in st.info:
            out["parts"] = st.info["parts"]
        return out
