"""refine1d: iterative refinement with ordinary continuous elements on contact1d (S1, form A),
with no kink encoding (K3 off). Oracle inputs, noise 0.

The loop (the user's decisions L1–L7, N2), from h = 0.16 mm:
SOLVE (ffc_tau1d.solve, plain P1) -> ESTIMATE -> MARK -> REFINE, until a stop rule.

- ESTIMATE (L1 a): per cell, η_e = (1 / VOXEL_MM) ∫ |ℓo| dx over the part of the cell where the
  side of τ (beyond the top: τ > 0) and the evidence (beyond: ℓo > 0) disagree. The integral uses
  the basis's quadrature split at τ = 0 (`integrate_cut`, per cell) (|ℓo| is 0 where the evidence changes side, so the
  integrand is continuous). It reads τ only through the basis.
- MARK (L2 a, N2 a): Dörfler with θ = 0.5 over all the cells: the fewest cells with the largest η
  whose sum is at least θ · Σ η. Only the marked cells above the deepest level are refined.
- REFINE: bisect them, with 2:1 grading (neighbours at most one level apart); no coarsening (L5).
  The solution is moved to the new mesh exactly (nested P1).
- Stop (L3 a + c): no cell marked (Σ η = 0), or no marked cell can be refined ("deepest level
  reached": 5 levels below 0.16 mm, h_min = 0.005 mm, L4 b), or the wall time (120 s for each
  case and start). A solve that ends without convergence does not stop the loop: its reason is
  recorded.
- Starts (L7 a): the six tents of ffc_contact1d. Where a start's peak is inside a cell, that cell is
  bisected (with grading) before the first solve, until the peak is on a node (L6 b).

Measures per level: each side's top error, the ridge state (the maxima of τ: position, value,
open or collapsed onto τ = 0), the width of {τ > 0}, the number of cells, the Newton iterations,
the solve's stop reason, Σ η and the number of marked cells.

Run from the repository root (each invocation stays below 10 minutes: at most 4 starts):
    nix develop --command uv run python prototypes/scripts/ffc_refine1d.py OUT_DIR \\
        --thickness ckiss --g 0.02 --centre mid --starts good B-1mm A+1mm both
    nix develop --command uv run python prototypes/scripts/ffc_refine1d.py OUT_DIR --figure FIG.png
(--figure draws the one figure from all the run files in OUT_DIR; no solve.)
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import ffc_contact as CT  # noqa: E402
import ffc_contact1d as R  # noqa: E402
import ffc_tau1d as P  # noqa: E402

H0 = 0.16
THETA = 0.5  # Dörfler (L2 a)
MAX_LEVEL = 5  # h_min = 0.16 / 2⁵ = 0.005 mm (L4 b)
WALL_S = 120.0
F64 = torch.float64
START_NAMES = ("good", "B-1mm", "A+1mm", "both", "shift-0.8mm", "shift+0.8mm")


def indicator(p: P.Problem, z: T.Array) -> T.Array:
    """η per cell (L1 a), in nats."""
    def density(x: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        lo = p.log_odds_t(x)
        return lo.abs() * (side * lo < 0)

    with torch.no_grad():
        eta = p.basis.integrate_cut(torch.as_tensor(z, dtype=F64), density, per_cell=True)
    return np.asarray(eta.numpy() / T.VOXEL_MM)


def dorfler(eta: T.Array, theta: float) -> T.IntArray:
    """The fewest cells with the largest η whose sum is at least θ · Σ η (none if Σ η = 0)."""
    tot = float(eta.sum())
    if tot <= 0.0:
        return np.zeros(0, np.int64)
    order = np.argsort(-eta, kind="stable")
    k = int(np.searchsorted(np.cumsum(eta[order]), theta * tot)) + 1
    return np.sort(order[: min(k, len(eta))]).astype(np.int64)


def levels(p: P.Problem, z: T.Array) -> T.IntArray:
    return np.rint(np.log2(H0 / p.basis.cell_sizes(z))).astype(np.int64)


@dataclass
class Config:
    """The energy and the domain of a run (the old refine1d: "speed" on the whole line)."""

    energy: str = "speed"
    domain: str = "line"  # "line": from start point to start point; "roots": the line ends at the roots (E8 b)
    ends: str = "keep"  # the mesh ends at the roots: "keep" or "merge" (ffc_contact.Contact.nodes_roots)
    mm_scale: float = 1.0
    slope_bounds: bool = False  # Q1 and Q3d as linear bounds (S1 c)
    progress: Path | None = None  # a log file: one line per level

    def nodes(self, c: CT.Contact, h: float, centre: str) -> T.Array:
        return c.nodes(h, centre) if self.domain == "line" else c.nodes_roots(h, centre, self.ends)

    def problem(self, c: CT.Contact, x: T.Array) -> P.Problem:
        return P.Problem(x, c.log_odds_t, c.settings, roots=R.roots(c), energy_form=self.energy,
                         mm_scale=self.mm_scale, slope_bounds=self.slope_bounds)

    def grid(self, c: CT.Contact) -> tuple[float, float] | None:
        return None if self.domain == "line" else (c.root_b, c.root_a)


CFG = Config()


def start_state(c: CT.Contact, x: T.Array, fB: float, fA: float) -> tuple[P.Problem, T.Array, int]:
    """The tent start on plain P1, its peak put on a node by bisection (L6 b); and the number
    of bisections."""
    peak = 0.5 * (fA + fB)
    p = CFG.problem(c, x)
    n = 0
    while not np.any(np.abs(p.x - peak) < 1e-12):
        if n >= MAX_LEVEL:
            raise ValueError("the start's peak is not on a node at the deepest level")
        p = CFG.problem(c, p.basis.refine(p.basis.cells_at(peak)).x)  # type: ignore[attr-defined]
        n += 1
    return p, R.tent(p.x, fB, fA, peak), n


def ridge_state(p: P.Problem, z: T.Array) -> dict[str, Any]:
    mx = [ex for ex in p.basis.extrema(z) if ex.kind == "max"]
    if not mx:
        return dict(state="none", ridges=[])
    st = "collapsed" if all(abs(ex.value) < P.ACTIVE_TOL for ex in mx) else "open"
    return dict(state=st, ridges=[(ex.x, ex.value) for ex in mx])


def laplace_state(p: P.Problem, z: T.Array) -> dict[str, Any]:
    """The Laplace approximation at z: the smallest eigenvalue of H; if H > 0, σ of τ at each top
    (a crossing of τ = 0) with σ of the top's position (σ_τ / |τ'|), and σ of τ at the nodes
    beyond the top (median and max). With H not positive definite, σ is not defined (None)."""
    _, H = P.gradient_hessian(p, z)
    w = np.linalg.eigvalsh(H)
    out: dict[str, Any] = dict(min_eig=float(w[0]), sigma_tops=None, sigma_beyond_median=None,
                               sigma_beyond_max=None)
    if not w[0] > 0:
        return out
    cr = P.crossings(p, z)
    tau, xs = p.basis.nodal(z)
    beyond = xs[tau > 0]
    xq = np.r_[[x for x, _ in cr], beyond]
    if len(xq) == 0:
        return out
    try:
        la = P.laplace(p, z, xq)
    except np.linalg.LinAlgError:
        return out  # H numerically singular: σ not defined
    tops = []
    for k, (x, _) in enumerate(cr):
        g = abs(float(p.basis.gradients(torch.as_tensor(z, dtype=torch.float64), np.array([x])).detach()[0]))
        tops.append(dict(x=x, sigma_tau=float(la.sd[k]), sigma_x_mm=float(la.sd[k] / g) if g > 0 else None))
    sb = la.sd[len(cr):]
    out.update(sigma_tops=tops, sigma_beyond_median=float(np.median(sb)) if len(sb) else None,
               sigma_beyond_max=float(sb.max()) if len(sb) else None)
    return out


def run_start(c: CT.Contact, x0: T.Array, fB: float, fA: float, label: str = "") -> dict[str, Any]:
    p, z, n_peak = start_state(c, x0, fB, fA)
    out = run_from(c, p, z, label)
    out.update(start_faces=[fB, fA], peak_bisections=n_peak, start_ridge=0.5 * (fA + fB))
    return out


def run_from(c: CT.Contact, p: P.Problem, z: T.Array, label: str = "") -> dict[str, Any]:
    """The loop SOLVE → ESTIMATE → MARK → REFINE from the start (p, z) (module docstring)."""
    t0 = time.time()
    deadline = t0 + WALL_S
    recs: list[dict[str, Any]] = []
    etas: list[T.Array] = []
    reason = ""
    while True:
        z, log = P.solve(p, z, deadline=deadline)
        sm = R.side_measures(c, p, z, CFG.grid(c))
        lv = levels(p, z)
        eta = indicator(p, z)
        marked = dorfler(eta, THETA)
        refinable = marked[lv[marked] < MAX_LEVEL]
        recs.append(dict(level=int(lv.max()), n_cells=p.basis.n_cells, newton=log.iterations,
                         solve_reason=log.reason, converged=log.converged,
                         active=log.active[-1] if log.active else 0,
                         top_error_B=sm["B"]["top_error_mm"], top_error_A=sm["A"]["top_error_mm"],
                         width=sm["width"], **ridge_state(p, z), eta_total=float(eta.sum()),
                         n_marked=len(marked), n_refined=len(refinable), energy=log.energy[-1],
                         damping_last=log.damping[-1] if log.damping else None,
                         damping_max=max(log.damping) if log.damping else None,
                         laplace=laplace_state(p, z) if CFG.energy != "speed" else None,
                         seconds=time.time() - t0))
        etas.append(np.c_[p.x[:-1], p.x[1:], eta])
        if CFG.progress is not None:
            q = recs[-1]
            with open(CFG.progress, "a") as f:
                f.write(f"{label} pass {len(recs) - 1} level {q['level']} cells {q['n_cells']} newton {q['newton']} "
                        f"{q['solve_reason']} top_err_B {q['top_error_B']} top_err_A {q['top_error_A']} "
                        f"{q['state']} ridges {q['ridges']} eta {q['eta_total']:.3e} {q['seconds']:.1f}s\n")
        if log.reason == "wall time" or time.time() > deadline:
            reason = "wall time"
            break
        if len(marked) == 0:
            reason = "no cell marked"
            break
        if len(refinable) == 0:
            reason = "deepest level reached"
            break
        fine = p.basis.refine(refinable)
        z = p.basis.prolong(z, fine)
        p = CFG.problem(c, fine.x)  # type: ignore[attr-defined]
    return dict(levels=recs, stop=reason, levels_used=int(max(r["level"] for r in recs)),
                seconds=time.time() - t0, x=p.x, z=z, eta=etas)


def figure(path: Path, runs: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), dpi=90)
    col = {s: f"C{i}" for i, s in enumerate(START_NAMES)}

    def field(key: str, start: str, a: Any, lim: float, title: str) -> None:
        r = runs[key]["starts"][start]
        c = runs[key]["contact"]
        x, z = r["x"], r["z"]
        xf = np.linspace(-lim, lim, 2000)
        a.plot(xf, c.tau_true(xf), "k-", lw=3, alpha=0.3, label="true τ (one tau_scale)")
        sel = np.abs(x) <= lim
        a.plot(x[sel], z[sel], "o-", ms=2.5, label=f"{start}: solved τ, last level")
        a.axhline(0, color="k", lw=0.5)
        a.set(xlabel="x (mm)", ylabel="τ", title=title)
        a.legend(fontsize=7)

    k0 = "ckiss_g0.02_mid"
    if k0 in runs:
        field(k0, "good", ax[0, 0], 0.6, f"{k0}, good start: the sheet")
        field(k0, "shift-0.8mm", ax[0, 1], 1.6, f"{k0}, shift−0.8mm start")
        a = ax[0, 2]
        r = runs[k0]["starts"]["shift-0.8mm"]
        for i, e in enumerate(r["eta"]):
            m = 0.5 * (e[:, 0] + e[:, 1])
            sel = np.abs(m) < 1.6
            a.semilogy(m[sel], np.maximum(e[sel, 2], 1e-20), ".-", ms=3, label=f"pass {i}")
        a.set(xlabel="x (mm)", ylabel="η per cell (nats), floor 1e-20", title=f"{k0}, shift−0.8mm: indicator")
        a.legend(fontsize=6)
    a = ax[1, 0]
    a2 = ax[1, 1]
    for key, rr in runs.items():
        for s, r in rr["starts"].items():
            lv = r["levels"]
            err = [max(abs(q["top_error_B"]) if q["top_error_B"] is not None else 1.0,
                       abs(q["top_error_A"]) if q["top_error_A"] is not None else 1.0) for q in lv]
            a.semilogy(range(len(lv)), np.maximum(err, 1e-16), "-o", ms=2, color=col[s], alpha=0.5,
                       label=s if key == next(iter(runs)) else None)
            a2.plot(range(len(lv)), [q["n_cells"] for q in lv], "-o", ms=2, color=col[s], alpha=0.5)
    a.set(xlabel="pass (SOLVE → ESTIMATE → MARK → REFINE)", ylabel="max |top error| (mm); 1 if no top",
          title="all cases: top error per pass")
    a.legend(fontsize=6)
    a2.set(xlabel="pass", ylabel="cells", title="all cases: cells per pass")
    a = ax[1, 2]
    labels = []
    for i, (key, rr) in enumerate(runs.items()):
        for j, s in enumerate(START_NAMES):
            if s not in rr["starts"]:
                continue
            q = rr["starts"][s]["levels"][-1]
            for side, mk in (("B", "<"), ("A", ">")):
                e = q[f"top_error_{side}"]
                a.plot(i + 0.12 * j, abs(e) + 1e-16 if e is not None else 1.0, mk, color=col[s],
                       mfc="none" if e is None else col[s], ms=5)
        labels.append(key.replace("_", "\n"))
    a.set_yscale("log")
    a.set_xticks(range(len(runs)))
    a.set_xticklabels(labels, fontsize=6)
    a.set(ylabel="|top − true face| (mm); open marker at 1: no top", title="last pass: top errors")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_runs(out: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for f in sorted(out.glob("run_*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        rr = runs.setdefault(d["key"], dict(contact=d["contact"], starts={}))
        rr["starts"].update(d["starts"])
    order = [f"{th}_g{g}_{ce}" for th in ("ckiss", "equal") for g in R.GS for ce in R.CENTRES]
    return {k: runs[k] for k in order if k in runs}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--thickness", choices=sorted(CT.THICKNESS))
    ap.add_argument("--g", type=float, choices=R.GS)
    ap.add_argument("--centre", choices=R.CENTRES)
    ap.add_argument("--starts", nargs="+", choices=START_NAMES, default=list(START_NAMES))
    ap.add_argument("--figure", type=Path, default=None, help="draw the figure from the runs in OUT; no solve")
    ap.add_argument("--energy", choices=P.ENERGY_FORMS, default="speed")
    ap.add_argument("--domain", choices=("line", "roots"), default="line")
    ap.add_argument("--ends", choices=("keep", "merge"), default="keep")
    ap.add_argument("--mm-scale", type=float, default=1.0, help="a factor on the mm-pull weight (a check only)")
    ap.add_argument("--slope-bounds", action="store_true", help="Q1 and Q3d as linear bounds (S1 c)")
    a = ap.parse_args(argv)
    CFG.energy, CFG.domain, CFG.ends, CFG.mm_scale = a.energy, a.domain, a.ends, a.mm_scale
    CFG.slope_bounds = a.slope_bounds
    CFG.progress = a.out / "progress.log"
    a.out.mkdir(parents=True, exist_ok=True)
    if a.figure:
        runs = load_runs(a.out)
        a.figure.parent.mkdir(parents=True, exist_ok=True)
        figure(a.figure, runs)
        return
    if a.thickness is None or a.g is None or a.centre is None:
        ap.error("--thickness, --g and --centre are needed for a run")
    tA, tB = CT.THICKNESS[a.thickness]
    c = CT.Contact(tA, tB, a.g)
    x0 = CFG.nodes(c, H0, a.centre)
    key = f"{a.thickness}_g{a.g}_{a.centre}"
    st = R.starts(c, x0)
    t0 = time.time()
    res: dict[str, Any] = {}
    for s in a.starts:
        fB, fA = st[s]
        res[s] = run_start(c, x0, fB, fA, f"{key} {s}")
        r = res[s]
        print(key, s, r["stop"], r["levels_used"], [(q["top_error_B"], q["top_error_A"], q["state"], q["n_cells"],
              q["newton"], q["solve_reason"]) for q in r["levels"]], f"{r['seconds']:.1f} s", flush=True)
    tag = "_".join(a.starts)
    with open(a.out / f"run_{key}_{tag}.pkl", "wb") as f:
        pickle.dump(dict(key=key, contact=c, starts=res), f)
    rec = dict(key=key, h0_mm=H0, theta=THETA, max_level=MAX_LEVEL, inputs="oracle", noise=0.0,
               settings=T.DEFAULTS.record(), energy=a.energy, domain=a.domain, ends=a.ends,
               mm_scale=a.mm_scale, slope_bounds=a.slope_bounds, mm_pull_coefficient=a.mm_scale * P.mm_pull_coefficient(T.DEFAULTS),
               equivolume_weight=P.EQUIVOLUME_LAMBDA_A_PROVISIONAL,
               equivolume_weight_note="PROVISIONAL placeholder (the old speed prior's λA; user's E5)",
               starts={s: {k: v for k, v in r.items() if k not in ("x", "z", "eta")} for s, r in res.items()},
               seconds=time.time() - t0,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    (a.out / f"run_{key}_{tag}.json").write_text(json.dumps(T._jsonable(rec), indent=1))


if __name__ == "__main__":
    main()
