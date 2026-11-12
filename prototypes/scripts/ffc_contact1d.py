"""contact1d (and slab1d once) in the S1 form: one τ field, the top is {τ = 0}, oracle inputs,
noise 0, h = 0.16 mm. Writes the measures (JSON), the solutions (npz) and one figure.

Cases: thickness "ckiss" (tA 2.5, tB 1.2) and "equal" (2.4, 2.4); g = 0.02 and 0.32 mm (the
half-width); the sheet centre in the middle of a cell ("mid") and on a node ("node").

Two forms per case:
- "kink": the XFEM kink enrichment (K3a′, ffc_tau1d.solve_kink): one extra node with a free
  position and value in the cell that holds a reversal of ∇τ, decided from the field;
- "refine": plain P1, the cells round the reversal refined from the field (K5a): solve, bisect
  the two cells at the node where the slope changes sign (with 2:1 grading), interpolate, solve
  again, until the smallest cell is at most g / 2. From the good start only.

Starts (kink form): a tent τ0 with slope 1 up to a peak, zero at faces fB < fA:
- good: fB, fA at the nodes nearest to the oracle's p = 0.5 crossings;
- B−1mm: fB moved 1 mm into side B's cortex; A+1mm: fA 1 mm into side A's cortex; both: the two.
- shift−0.8mm, shift+0.8mm: both faces moved 0.8 mm together, so the start's ridge is 0.8 mm
  inside one side's cortex, not near the true sheet (the contact must come from the roots).
Starts that move a face toward the other side by 1 mm would put fB > fA (an overlap); one
continuous τ cannot hold them, so they are not run (the user dropped the overlapping start).

The peak of the tent is at (fA + fB) / 2. In the kink form the free node is put there. In the
refine form the peak is moved to the nearest node (P1 cannot have a peak inside a cell).

Measures (sides are used only here, from the truth): each side's top (the crossing of τ = 0)
against its true face ±g; the width of {τ > 0} against 2g; per side, τ and the speed against
the truth after an affine map of τ fitted on that side's nodes (the map's scale α, the largest
τ residual, the largest speed error); the top probability P(τ > 0) of the Laplace
approximation against the oracle p; Newton iterations; the energy against the true state's.

Run from the repository root:
    nix develop --command uv run python prototypes/scripts/ffc_contact1d.py OUT_DIR FIGURE.png \
        [--thickness ckiss|equal ...]
    nix develop --command uv run python prototypes/scripts/ffc_contact1d.py - FIGURE.png --combine DIR1 DIR2
(Each thickness set is one run, to keep each run below 10 minutes; --combine draws the one
figure from both.)
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import ffc_contact as CT  # noqa: E402
import ffc_line as LN  # noqa: E402
import ffc_tau1d as P  # noqa: E402

H_MM = 0.16
WRONG_MM = 1.0
RIDGE_SHIFT_MM = 0.8
WALL_S = 120.0  # the wall-time limit of each start (kink form) and of each refine sequence
GS = (0.02, 0.32)
CENTRES = ("mid", "node")
MAX_REFINE = 8
F64 = torch.float64


# ---------------------------------------------------------------- starts


def tent(x: T.Array, fB: float, fA: float, peak_x: float) -> T.Array:
    """Zero at fB and fA, linear up to the peak at peak_x, height (fA − fB) / 2."""
    Pk = 0.5 * (fA - fB)
    return np.where(x <= peak_x, (x - fB) / (peak_x - fB) * Pk, (fA - x) / (fA - peak_x) * Pk)


def oracle_faces(c: CT.Contact, x: T.Array) -> tuple[float, float]:
    """The nodes nearest to the first and the last crossing of p = 0.5 by the oracle stop field
    (sampled at 0.001 mm; the field is evaluable at any point). If both are the same node, its
    two neighbours."""
    xf = np.arange(x[0], x[-1], 0.001)
    above = c.stop_probability(xf) >= 0.5
    i = np.nonzero(above)[0]
    xB, xA = xf[i[0]], xf[i[-1]]
    fB = float(x[np.argmin(np.abs(x - xB))])
    fA = float(x[np.argmin(np.abs(x - xA))])
    if fA <= fB:
        k = int(np.argmin(np.abs(x - fB)))
        fB, fA = float(x[k - 1]), float(x[k + 1])
    return fB, fA


def starts(c: CT.Contact, x: T.Array) -> dict[str, tuple[float, float]]:
    fB, fA = oracle_faces(c, x)
    return {"good": (fB, fA), "B-1mm": (fB - WRONG_MM, fA), "A+1mm": (fB, fA + WRONG_MM),
            "both": (fB - WRONG_MM, fA + WRONG_MM),
            # no ridge near the true sheet: the whole start medium moved into one side's cortex
            "shift-0.8mm": (fB - RIDGE_SHIFT_MM, fA - RIDGE_SHIFT_MM),
            "shift+0.8mm": (fB + RIDGE_SHIFT_MM, fA + RIDGE_SHIFT_MM)}


def roots(c: CT.Contact) -> tuple[tuple[float, float], ...]:
    """The two roots on the white surface and their normals (toward the cortex)."""
    return ((-c.g - c.tB, 1.0), (c.g + c.tA, -1.0))


def base_problem(c: CT.Contact, x: T.Array) -> P.Problem:
    return P.Problem(x, c.log_odds_t, c.settings, roots=roots(c))


def kink_start(x: T.Array, fB: float, fA: float) -> tuple[T.Array, int | None, float | None, float | None]:
    """(τ0 at the fixed nodes, the enriched cell or None, τ_k, x_k): the tent with its peak at
    (fA + fB) / 2; the enrichment holds the peak when it is inside a cell."""
    peak = 0.5 * (fA + fB)
    tau0 = tent(x, fB, fA, peak)
    on = np.abs(x - peak) < 1e-12
    if on.any():
        return tau0, None, None, None
    e = int(np.searchsorted(x, peak) - 1)
    return tau0, e, 0.5 * (fA - fB), peak


# ---------------------------------------------------------------- measures


def side_measures(c: CT.Contact, p: P.Problem, z: T.Array) -> dict[str, Any]:
    tau, xs = (a.numpy() for a in p.split(torch.as_tensor(z, dtype=F64)))
    cr = P.crossings(p, z)
    out: dict[str, Any] = dict(crossings=[x for x, _ in cr])
    up = [x for x, s in cr if s > 0]  # τ rises through 0: a top reached from the left (side B)
    dn = [x for x, s in cr if s < 0]
    tops = {"B": up[0] if up else None, "A": dn[-1] if dn else None}
    for side, true_face in (("B", -c.g), ("A", c.g)):
        t = tops[side]
        m: dict[str, Any] = dict(top=t, top_error_mm=None if t is None else t - true_face)
        if t is not None:
            sel = (xs <= t) if side == "B" else (xs >= t)
            sel &= tau < 0
            tt = c.tau_true(xs[sel])
            Amat = np.c_[tau[sel], np.ones(sel.sum())]
            (alpha, beta), *_ = np.linalg.lstsq(Amat, tt, rcond=None)
            m["map_alpha"] = float(alpha)
            m["map_beta"] = float(beta)
            m["tau_residual_max"] = float(np.max(np.abs(alpha * tau[sel] - tt + beta)))
            idx = np.nonzero(sel)[0]
            cells = idx[:-1] if side == "B" else idx[:-1]
            cells = np.array([e for e in cells if sel[e] and sel[e + 1]], int)
            v = np.abs((xs[cells + 1] - xs[cells]) / (tau[cells + 1] - tau[cells]))
            mid = 0.5 * (xs[cells] + xs[cells + 1])
            m["speed_raw_minmax"] = [float(v.min()), float(v.max())]
            m["speed_true"] = float(c.speed_true(np.array([mid[0]]))[0])
            m["speed_mapped_error_max"] = float(np.max(np.abs(v / abs(alpha) - c.speed_true(mid))))
        out[side] = m
    out["width"] = None if tops["A"] is None or tops["B"] is None else tops["A"] - tops["B"]
    out["width_true"] = 2 * c.g
    return out


def true_state(c: CT.Contact, p: P.Problem) -> tuple[P.Problem, T.Array]:
    """The truth in the discrete space: τ_true at the nodes, and the kink at the true ridge
    x_ridge (a node, or the extra node of the cell that holds it)."""
    x = p.x
    xr = c.x_ridge
    if np.any(np.abs(x - xr) < 1e-12):
        q = P.Problem(x, c.log_odds_t, c.settings, roots=p.roots)
        return q, q.pack(c.tau_true(x))
    e = int(np.searchsorted(x, xr) - 1)
    q = P.Problem(x, c.log_odds_t, c.settings, enrich_cell=e, roots=p.roots)
    return q, q.pack(c.tau_true(x), float(c.tau_true(np.array([xr]))[0]), xr)


def top_probability(c: CT.Contact, p: P.Problem, z: T.Array) -> dict[str, Any]:
    xq = np.linspace(-c.g - 0.5, c.g + 0.5, 801)
    la = P.laplace(p, z, xq)
    po = c.stop_probability(xq)
    return dict(x=xq, p=la.p_beyond, p_oracle=po, sd=la.sd, min_eig=la.min_eig,
                max_abs_diff=float(np.max(np.abs(la.p_beyond - po))),
                sd_at_tops=[float(np.interp(-c.g, xq, la.sd)), float(np.interp(c.g, xq, la.sd))],
                p_at_centre=float(np.interp(0.0, xq, la.p_beyond)),
                p_oracle_at_centre=float(np.interp(0.0, xq, po)))


# ---------------------------------------------------------------- refinement (K5a)


def refine_nodes(x: T.Array, r: int) -> T.Array:
    """Bisect the cells r − 1 and r (at node r), then bisect cells until neighbouring cells
    differ by at most a factor 2."""
    mids = [0.5 * (x[r - 1] + x[r]), 0.5 * (x[r] + x[r + 1])]
    x = np.sort(np.r_[x, mids])
    while True:
        L = np.diff(x)
        bad = [e for e in range(len(L) - 1) if L[e] > 2 * L[e + 1] + 1e-15 or L[e + 1] > 2 * L[e] + 1e-15]
        if not bad:
            return x
        add = []
        for e in bad:
            k = e if L[e] > L[e + 1] else e + 1
            add.append(0.5 * (x[k] + x[k + 1]))
        x = np.unique(np.r_[x, add])


def run_refine(c: CT.Contact, x0: T.Array) -> dict[str, Any]:
    fB, fA = oracle_faces(c, x0)
    peak = x0[int(np.argmin(np.abs(x0 - 0.5 * (fA + fB))))]
    h = float(np.diff(x0).min())
    if peak <= fB:  # P1 needs the peak on a node strictly between the faces
        fB = float(peak - h)
    if peak >= fA:
        fA = float(peak + h)
    x = x0.copy()
    z = tent(x, fB, fA, peak)
    rounds = []
    deadline = time.time() + WALL_S
    for _ in range(MAX_REFINE + 1):
        p = base_problem(c, x)
        z, log = P.solve(p, z, deadline=deadline)
        sm = side_measures(c, p, z)
        r = P.reversal_node(p, z)
        rounds.append(dict(min_cell=float(np.diff(x).min()), n_nodes=len(x), newton=log.iterations,
                           converged=log.converged, reason=log.reason, active=log.active[-1] if log.active else 0, reversal_x=None if r is None else float(x[r]),
                           top_error_B=sm["B"]["top_error_mm"], top_error_A=sm["A"]["top_error_mm"],
                           width=sm["width"], energy=log.energy[-1]))
        if np.diff(x).min() <= c.g / 2 + 1e-12 or r is None:
            break
        xn = refine_nodes(x, r)
        z = np.interp(xn, x, z)
        x = xn
    return dict(rounds=rounds, x=x, tau=z)


# ---------------------------------------------------------------- slab1d in the S1 form


def slab_s1() -> dict[str, Any]:
    line = LN.slab1d()
    x = line.nodes(H_MM)
    st = line.settings
    sig = st.oracle_blur_vox * T.VOXEL_MM

    def lo_t(xx: torch.Tensor) -> torch.Tensor:
        p_ = torch.clamp(torch.special.ndtr((xx - line.x_stop_true) / sig), *st.oracle_clip)
        return torch.log(p_) - torch.log1p(-p_)

    p = P.Problem(x, lo_t, st, roots=((0.0, 1.0),))
    out = {}
    for name, s0 in (("good", 2.32), ("wrong+1mm", 3.4), ("wrong-1mm", 1.4)):
        z, log = P.solve(p, x - s0)
        cr = P.crossings(p, z)
        mov = x <= cr[0][0]
        out[name] = dict(stop_error_mm=cr[0][0] - line.x_stop_true,
                         tau_abs_error_max=float(np.max(np.abs(z[mov] - line.tau_true(x[mov])))),
                         newton=log.iterations, reason=log.reason, energy=log.energy[-1])
    out["energy_true"] = p.total(line.tau_true(x))
    return out


# ---------------------------------------------------------------- the run


def run_case(c: CT.Contact, centre: str) -> dict[str, Any]:
    x = c.nodes(H_MM, centre)
    res: dict[str, Any] = dict(n_nodes=len(x), x_first=float(x[0]), x_last=float(x[-1]), kink={})
    for name, (fB, fA) in starts(c, x).items():
        tau0, cell, tk, xk = kink_start(x, fB, fA)
        t0 = time.time()
        p, z, klog = P.solve_kink(base_problem(c, x), tau0, cell, tk, xk, deadline=time.time() + WALL_S)
        m = side_measures(c, p, z)
        ts = true_state(c, base_problem(c, x))
        m.update(start_faces=[fB, fA], start_enriched_cell=cell, newton=klog.newton, passes=klog.passes,
                 converged=all(bool(q["converged"]) for q in klog.passes if not q.get("trial")),
                 reason=klog.passes[-1]["reason"], energy=p.total(z),
                 wall_time_hit=any(q["reason"] == "wall time" for q in klog.passes),
                 energy_true=None if ts is None else ts[0].total(ts[1]),
                 enriched_cell=p.enrich_cell, kink_x=float(z[-1]) if p.enrich_cell is not None else None,
                 n_reversals=p.n_reversals(z), seconds=time.time() - t0)
        if name == "good":
            tp = top_probability(c, p, z)
            m["top_probability"] = {k: v for k, v in tp.items() if not isinstance(v, np.ndarray)}
            res["_tp"] = tp
            res["_good"] = (p, z)
        res["kink"][name] = m
    res["refine"] = run_refine(c, x)
    return res


def figure(path: Path, cases: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), dpi=90)
    key = "ckiss_g0.02_mid"
    c, r = cases[key]["contact"], cases[key]["res"]
    p, z = r["_good"]
    tau, xs = (a.numpy() for a in p.split(torch.as_tensor(z, dtype=F64)))
    xf = np.linspace(xs[0], xs[-1], 3000)
    a = ax[0, 0]
    a.plot(xf, c.tau_true(xf), "k-", lw=3, alpha=0.3, label="true τ (one tau_scale)")
    a.plot(xs, tau, "o-", ms=3, label="solved τ (good start)")
    a.axhline(0, color="k", lw=0.5)
    a.set(xlabel="x (mm)", ylabel="τ", title=f"{key}: τ on the whole line (unmapped)")
    a.legend(fontsize=7)
    a = ax[0, 1]
    for key2, mk in (("ckiss_g0.02_mid", "o"), ("ckiss_g0.02_node", "s")):
        c2, r2 = cases[key2]["contact"], cases[key2]["res"]
        p2, z2 = r2["_good"]
        tau2, xs2 = (q.numpy() for q in p2.split(torch.as_tensor(z2, dtype=F64)))
        sel = np.abs(xs2) < 0.6
        a.plot(xs2[sel], tau2[sel], mk + "-", ms=4, mfc="none", label=f"{key2} (kink at " + (f"{z2[-1]:+.4f}, enriched)" if p2.enrich_cell is not None else "a node)"))
    xz = np.linspace(-0.6, 0.6, 600)
    a.plot(xz, cases["ckiss_g0.02_mid"]["contact"].tau_true(xz), "k-", lw=3, alpha=0.3, label="true τ")
    a.axhline(0, color="k", lw=0.5)
    a.set(xlabel="x (mm)", ylabel="τ", title="the sheet (g = 0.02): kink form, nodes", ylim=(-0.8, 0.15))
    a.legend(fontsize=7)
    a = ax[0, 2]
    for key2 in ("ckiss_g0.02_mid", "ckiss_g0.32_mid"):
        tp = cases[key2]["res"]["_tp"]
        a.plot(tp["x"], tp["p"], "-", label=f"{key2}: Laplace P(τ > 0)")
        a.plot(tp["x"], tp["p_oracle"], ":", label=f"{key2}: oracle p")
    a.set(xlabel="x (mm)", ylabel="probability beyond the top", title="top probability (good start)")
    a.legend(fontsize=6)
    # top errors per case and start
    a = ax[1, 0]
    labels = []
    for i, (k2, cr) in enumerate(cases.items()):
        for j, (sn, m) in enumerate(cr["res"]["kink"].items()):
            for side, mk in (("B", "<"), ("A", ">")):
                e = m[side]["top_error_mm"]
                if e is not None:
                    a.plot(i + 0.15 * j, abs(e) + 1e-16, mk, color=f"C{j}", ms=5,
                           label=f"{sn} ({side})" if i == 0 else None)
        labels.append(k2.replace("_", "\n"))
    a.set_yscale("log")
    a.set_xticks(range(len(cases)))
    a.set_xticklabels(labels, fontsize=6)
    a.set(ylabel="|top − true face| (mm), floor 1e-16", title="kink form: top errors")
    a.legend(fontsize=6, ncol=2)
    a = ax[1, 1]
    for i, (k2, cr) in enumerate(cases.items()):
        rr = cr["res"]["refine"]["rounds"]
        mc = [q["min_cell"] for q in rr]
        err = [max(abs(q["top_error_B"] or 0), abs(q["top_error_A"] or 0)) for q in rr]
        a.loglog(mc, np.maximum(err, 1e-16), "o-", ms=3, label=k2)
    a.set(xlabel="smallest cell (mm)", ylabel="max |top error| (mm)", title="refine form (plain P1, from the field)")
    a.invert_xaxis()
    a.legend(fontsize=6)
    a = ax[1, 2]
    for i, (k2, cr) in enumerate(cases.items()):
        its = [m["newton"] for m in cr["res"]["kink"].values()]
        a.plot([i] * len(its), its, "o", mfc="none")
    a.set_xticks(range(len(cases)))
    a.set_xticklabels(labels, fontsize=6)
    a.set(ylabel="Newton iterations (4 starts)", title="kink form: iterations")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("fig", type=Path)
    ap.add_argument("--thickness", nargs="+", default=["ckiss", "equal"], choices=sorted(CT.THICKNESS))
    ap.add_argument("--combine", nargs="+", type=Path, default=None,
                    help="draw the figure from the cases.pkl of earlier runs (these directories); no solve")
    a = ap.parse_args(argv)
    if a.combine:
        merged: dict[str, Any] = {}
        for d in a.combine:
            with open(d / "cases.pkl", "rb") as f:
                merged.update(pickle.load(f))
        a.fig.parent.mkdir(parents=True, exist_ok=True)
        figure(a.fig, merged)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    a.fig.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cases: dict[str, Any] = {}
    for th in a.thickness:
        tA, tB = CT.THICKNESS[th]
        for g in GS:
            c = CT.Contact(tA, tB, g)
            for centre in CENTRES:
                key = f"{th}_g{g}_{centre}"
                cases[key] = dict(contact=c, res=run_case(c, centre))
                print(key, {s: (m["B"]["top_error_mm"], m["A"]["top_error_mm"], m["newton"], m["reason"])
                            for s, m in cases[key]["res"]["kink"].items()}, flush=True)
    slab = slab_s1()
    rec = dict(form="S1, one τ field, kink form K3a′, refine form K5a", h_mm=H_MM, inputs="oracle", noise=0.0,
               settings=T.DEFAULTS.record(), slab1d_s1=slab,
               cases={k: dict(tA=v["contact"].tA, tB=v["contact"].tB, g=v["contact"].g,
                              **{kk: vv for kk, vv in v["res"].items() if not kk.startswith("_") and kk != "refine"},
                              refine=v["res"]["refine"]["rounds"])
                      for k, v in cases.items()},
               seconds=time.time() - t0,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    (a.out / "contact1d.json").write_text(json.dumps(T._jsonable(rec), indent=1))
    np.savez_compressed(a.out / "contact1d.npz", **{
        f"{k}_{q}": v["res"]["_good"][1] if q == "z" else v["res"]["_good"][0].x
        for k, v in cases.items() for q in ("z", "x")})
    with open(a.out / "cases.pkl", "wb") as f:
        pickle.dump(cases, f)
    if "ckiss" in a.thickness:
        figure(a.fig, cases)
    print("slab1d S1:", slab)
    print(f"{rec['seconds']:.1f} s; peak {rec['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
