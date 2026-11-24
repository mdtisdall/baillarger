"""Σ moved by the stop energy alone (user, 2026-11-23: a first stage "more
like FreeSurfer", whose point is the topology of Σ): the solve loop of
top_solve.py (prototype branch claude/tau-shell-cc08e1) with only what a
run with no τ uses, step="pressure". The results are those of top_solve.py
with tau_field_run.py basis=stop.

The energy is E_stop (T4 of the plan docs/plans/top-surface.md on the
prototype branch): Σ c f over the voxels, c = -log(19) in the cortex label,
+log(19) in label 0, 0 elsewhere;
f = clip(0.5 - s/h, 0, 1) with s the signed distance from the voxel centre
to Σ (positive outside), 1 inside Σ and 0 outside off the band. The
evaluator (stop_field.StopField) gives E and ∂E/∂V.

One step:
  1. The pressure on each vertex: the stop energy's force along its normal
     per area, as a fraction of the largest there can be (Σ wholly in cortex
     or wholly in outside medium: LOG_ODDS / h³), clipped to [-1, 1]; the
     vertex moves along its normal by that fraction × step_max_vox × h.
  2. The trial Σ' = Σ + α × that move (α = 1 at first), capped at
     step_max_vox × h per vertex and kept in the crop; each vertex's move is
     halved up to HALVINGS times per vertex (then refused) until it keeps
     the gap and makes no intersection and no new fold (T2,
     constrained_move).
  3. Accept when E(Σ') < E(Σ). Else Σ stays, α halves, and step 2 is tried
     again (at most max_tries times; then the run ends "no decrease").
  4. After an accepted step: the tangential relaxation and the remeshing
     (T7, mesh="remesh"), then E again.
The run ends when the accepted step is below stop_vox × h for every
vertex ("step below stop_vox": not a test of convergence), with "no
decrease", or at max_steps.

Before the first step the start is remeshed in start_passes rounds of
splits, collapses and flips (more than 1 for a start with slivers).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import crop_site as CS
import numpy as np
import sigma_surface as SU

from baillarger import mesh_checks as MC

HALVINGS = SU.HALVINGS
LOG_ODDS = float(np.log(0.95 / 0.05))  # the stop weight |c|: label error rate 0.05


@dataclass
class SolveSettings:
    gap_vox: float = 0.25  # T2
    #: T7 (mesh="remesh"): the target edge length × each start vertex's mean
    #: edge (sigma_surface.StartSizes; surface_remesh splits above 4/3,
    #: collapses below 4/5 of it), or × the start's median edge (edge_target)
    edge_ratio: float = 1.0
    #: the remesh target: "start" (StartSizes: per start vertex, its mean
    #: edge) or "median" (one length: the start's median edge; 2026-11-22:
    #: the per-vertex sizes of an uneven start, edges 0.4 to 6 h, make the
    #: remesh split and collapse hundreds of edges per step without end)
    edge_target: str = "start"
    #: the reference of the fold test (sigma_surface.FOLD_REFERENCE): "origin",
    #: or "none" (only the dihedral part of the test)
    fold_reference: str = "origin"
    fold_cos: float = MC.FOLD_COS
    #: the remesh of the start, before the first step: this many rounds of
    #: splits, collapses and flips (sigma_surface.remesh), ended when one
    #: changes nothing. More than 1 for a start with slivers (the overview
    #: mesh, 2026-11-23).
    start_passes: int = 1
    step_max_vox: float = 0.5  # at most h/2 per vertex
    max_steps: int = 200
    max_tries: int = 8
    #: the step rule: only "pressure" (user, 2026-11-23, for a Σ moved by
    #: the stop energy alone; the other rules of top_solve.py need τ)
    step: str = "pressure"
    stop_vox: float = 1e-3  # every accepted vertex step below this × h: the run ends
    #: the tangential relaxation after an accepted step (T7): "all" vertices,
    #: "none", or only the "flagged" ones (a face of theirs has its smallest
    #: altitude below the step cap step_max_vox × h: one step could flip it)
    relax: str = "all"
    measure_relax: bool = False  # also E right after the relaxation (one more evaluation)
    #: the mesh after an accepted step: "remesh" (relax, then uniform splits,
    #: collapses and flips, T7) or "none" (Σ keeps its faces and vertices)
    mesh: str = "remesh"


@dataclass
class State:
    V: np.ndarray
    E: float
    grad: np.ndarray  # (n, 3) ∂E/∂V, mm
    info: dict[str, Any]


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
    m: SU.Mesh, delta: np.ndarray, chk: MC.Checker, fold_cos: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Each vertex's move delta (n, 3) is halved up to HALVINGS times, then
    refused, until it keeps the gap and makes no intersection and no new
    fold. Returns the moves taken and counts (with the masks _halved and
    _refused)."""
    d = delta.copy()
    V0 = m.V.copy()
    mov = np.abs(d).sum(1) > 0
    changed = mov.copy()
    halved = np.zeros(len(d), bool)
    refused = np.zeros(len(d), bool)
    # the checks that flagged a vertex when it was refused (a vertex may have both)
    why = {"gap_or_cross": np.zeros(len(d), bool), "new_fold": np.zeros(len(d), bool)}
    k = 0
    while changed.any():
        Vn = V0 + d
        bad, _, _, _ = chk.check(Vn, changed, np.zeros(len(d), bool))
        fb, _, _ = SU.new_folds(m, Vn, V0, changed, fold_cos)
        badm = (bad | fb) & (np.abs(d).sum(1) > 0)
        if not badm.any():
            break
        if k < HALVINGS:
            d[badm] *= 0.5
            halved |= badm
        else:
            d[badm] = 0.0
            refused |= badm
            for name, mask in (("gap_or_cross", bad), ("new_fold", fb)):
                why[name] |= mask & badm
        changed = badm
        k += 1
    m.held = np.where(refused, m.held + 1, np.where(mov & ~refused, 0.0, m.held))
    return d, {
        "moved": int((np.abs(d).sum(1) > 0).sum()),
        "halved": int(halved.sum()),
        "refused": int(refused.sum()),
        "refused_why": {name: int(mask.sum()) for name, mask in why.items()},
        "_halved": halved,
        "_refused": refused,
    }


@dataclass
class SolveRun:
    site: Any  # labels (the grid) and vox (mm per axis); v and f (the white mesh) when start is None
    s: SolveSettings
    progress: Any = None
    snapshot: Any = None  # callable(name, mesh, disp, meta)
    start: SU.Mesh | None = None  # the initial Σ (default: the white mesh)
    #: E and ∂E/∂V: callable(V, F) -> State (default: stop_field.StopField)
    evaluator: Any = None
    #: callable(run), after each step's log record (e.g. to write the log
    #: as the run goes, so a stopped run keeps it)
    on_step: Any = None
    log: list[dict[str, Any]] = field(default_factory=list)
    ended: str = ""
    contacts: dict[str, np.ndarray] = field(default_factory=dict)

    def run(self) -> SU.Mesh:
        site, s = self.site, self.s
        if s.step != "pressure":
            raise ValueError(f"step: {s.step} (only pressure without τ)")
        if s.mesh not in ("remesh", "none"):
            raise ValueError(f"mesh: {s.mesh}")
        h = float(site.vox.mean())
        gap = s.gap_vox * h
        SU.FOLD_REFERENCE = s.fold_reference
        m = SU.start_mesh(site) if self.start is None else self.start
        m.update_topology()  # the reference, by fold_reference
        u = SU._edges(m.F)
        med = float(np.median(np.linalg.norm(m.V[u[:, 0]] - m.V[u[:, 1]], axis=1)))
        self.median_edge = med
        if s.edge_target == "median":
            def sizes(_m: SU.Mesh) -> Any:
                return s.edge_ratio * med
        elif s.edge_target == "start":
            sizes = SU.StartSizes(m, s.edge_ratio)
        else:
            raise ValueError(f"edge_target: {s.edge_target}")
        lo = np.zeros(3)
        hi = (np.array(site.labels.shape) - 1) * site.vox
        ev = self.evaluator
        if ev is None:
            import stop_field

            ev = stop_field.StopField(site)
        t0 = time.time()
        white = SU.counts(m, gap, s.fold_cos)
        sp, co = 0, 0
        if s.mesh == "remesh":
            m, sp, co = SU.remesh(m, gap, sizes(m), s.fold_cos, h, passes=s.start_passes)
        cur = ev(m.V, m.F)
        self.initial = {"quality": quality(m.V, m.F, h, s.step_max_vox * h), "vertices": len(m.V), "faces": len(m.F),
                        "white": white, "splits": sp, "collapses": co, "E": cur.E, "E_stop": cur.E,
                        "eval": cur.info, "s": time.time() - t0}
        if self.progress:
            self.progress(f"start: E {cur.E:.1f}, {cur.info['s']:.1f} s")
        if self.snapshot:
            self.snapshot("solve-000", m, np.zeros(len(m.V)), self._meta(0, cur, {}))
        self.ended = "max_steps"
        for k in range(1, s.max_steps + 1):
            t1 = time.time()
            chk = MC.Checker(m.top, m.V, gap, slack=s.step_max_vox * h)
            tries: list[dict[str, Any]] = []
            accepted = False
            # the stop energy's pressure on Σ: per vertex, the force along its normal per area, as a
            # fraction of the largest there can be (Σ wholly in cortex or wholly in outside medium:
            # LOG_ODDS / h³), times the cap. No α to keep between steps.
            nrm, area = CS.vertex_normals_areas(m.V, m.F)
            p = np.clip((cur.grad * nrm).sum(1) * h**3 / (np.maximum(area, 1e-300) * LOG_ODDS), -1.0, 1.0)
            g = (s.step_max_vox * h * p)[:, None] * nrm
            alpha = 1.0
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
                del cinfo["_halved"], cinfo["_refused"]
                trial = ev(m.V + dd, m.F)
                tries.append({"alpha": alpha, "E": trial.E, "max_step_vox": float(np.linalg.norm(dd, axis=1).max() / h),
                              **cinfo, "eval_s": trial.info["s"]})
                if self.progress:  # each trial: why the line search refuses it (user, 2026-11-20)
                    self.progress(
                        f"  try {len(tries)}: α {alpha:.3g}, max step {tries[-1]['max_step_vox']:.3f} h, "
                        f"E {trial.E:.1f} vs {cur.E:.1f}, moved {cinfo['moved']} halved {cinfo['halved']} "
                        f"refused {cinfo['refused']}")
                if trial.E < cur.E:
                    accepted = True
                    m.V = m.V + dd
                    m.travel = m.travel + np.linalg.norm(dd, axis=1)
                    disp = np.linalg.norm(dd, axis=1)
                    break
                m.restore(saved)
                alpha *= 0.5
            if not accepted:
                self.ended = "no decrease"
                self.log.append({"step": k, "tries": tries, "accepted": False, "s": time.time() - t1})
                if self.on_step:
                    self.on_step(self)
                if self.progress:
                    self.progress(f"step {k}: no decrease in {len(tries)} tries; E {cur.E:.1f}")
                break
            e_trial = trial.E
            q_before = quality(m.V, m.F, h, s.step_max_vox * h)
            if s.mesh == "none" or s.relax == "none":
                r, nrel = np.zeros(len(m.V)), 0
            else:
                only = flagged_vertices(m.V, m.F, s.step_max_vox * h)[0] if s.relax == "flagged" else None
                r, nrel = SU.relax(m, chk, s.fold_cos, only=only)
            m.last = np.sqrt(disp**2 + r**2)
            e_relax = None
            if nrel and s.measure_relax:
                e_relax = ev(m.V, m.F).E  # E after the relaxation alone (diagnostic)
            m_pre = m  # a remesh returns a new Σ (flips alone included)
            if s.mesh == "none":
                sp, co = 0, 0
            else:
                m, sp, co = SU.remesh(m, gap, sizes(m), s.fold_cos, h)
            cur = trial if m is m_pre and nrel == 0 else ev(m.V, m.F)
            c = SU.counts(m, gap, s.fold_cos)
            # E_stop: the same as E (the run logs of top_solve.py had both; readers use either)
            rec = {"step": k, "tries": tries, "accepted": True, "E_trial": e_trial, "E_relaxed": e_relax,
                   "E": cur.E, "E_stop": cur.E, "quality_before_relax": q_before,
                   "quality": quality(m.V, m.F, h, s.step_max_vox * h),
                   "relaxed": nrel, "splits": sp, "collapses": co,
                   **c, "vertices": len(m.V), "faces": len(m.F), "eval": cur.info,
                   "max_step_vox": float(disp.max() / h), "s": time.time() - t1}
            self.log.append(rec)
            if self.on_step:
                self.on_step(self)
            if self.progress:
                self.progress(
                    f"step {k}: E {cur.E:.1f} (trial {e_trial:.1f}), "
                    f"tries {len(tries)}, α {tries[-1]['alpha']}, max step {disp.max() / h:.3f} h, "
                    f"moved {tries[-1]['moved']} halved {tries[-1]['halved']} refused {tries[-1]['refused']} "
                    f"(held ≥ 3 steps {int((m.held >= 3).sum())}), ramp {cur.info['ramp_voxels']}, "
                    f"halved/refused at flagged {tries[-1]['halved_flagged']}/{tries[-1]['refused_flagged']}, "
                    f"relaxed {nrel} (E after relax {e_relax}), splits {sp} collapses {co}, "
                    f"quality before relax {q_before}, after remesh {rec['quality']}, "
                    f"intersections {c['intersecting_faces']} gap {c['gap_vertices']}, {rec['s']:.1f} s"
                )
            if self.snapshot:
                self.snapshot(f"solve-{k:03d}", m, m.last, self._meta(k, cur, rec))
            if disp.max() < s.stop_vox * h:
                # not a test of convergence: α may only have become small
                self.ended = "step below stop_vox"
                break
        self.contacts = SU.contacts(m, site.vox)
        self.final = cur
        return m

    @staticmethod
    def _meta(k: int, st: State, rec: dict[str, Any]) -> dict[str, Any]:
        out = {"step": k, "energy": st.E, "E_stop": st.E}
        out.update({kk: v for kk, v in rec.items() if kk in ("max_step_vox", "s")})
        return out
