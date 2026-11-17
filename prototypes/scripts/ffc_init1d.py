"""init1d: the initialisation test on contact1d (and slab1d as a check), with the equivolume
model (--energy, default "equivolume_convex"; --slope-bounds; the line ends at the roots), oracle inputs, noise 0.

The start (the user's decisions of 2026-11-13; plan-draft notes, "Initialisation"):
- d_bank(x): the distance from a bank's root (its white-matter end), outward.
- t̂_bank: the estimated distance from that root to the top. Variants:
  (i)   "label": the label's outer face (the true face): t̂_B = tB, t̂_A = tA;
  (ii)  "label±s": the label face shifted by s = 0.16 or 0.5 mm, inward (toward the root,
        t̂ − s) or outward (t̂ + s), face B alone, face A alone, or both faces (E10);
  (iii) "fused": no label at the contact: t̂ = d at the ridge of the distance field, the
        midpoint between the two roots, for both banks;
  (iv)  "probe": the first crossing of p = 0.5 by the stop evidence, from each root (E11 a).
- τ₀ = the lower envelope of the banks' fields, min over banks of (d_bank − t̂_bank) (E9 b),
  at the nodes (the peak handling: --peak, the open choice F3).
- The rules are checked on τ₀ before the solve: admissible (no flat cell in tissue), Q1 (n·∇τ > 0
  at the roots), Q2c (every maximum has τ ≥ 0), Q3d (no interior minimum). A start that fails
  is not solved; its failures are recorded.
- Then the refine loop of ffc_refine1d (SOLVE → ESTIMATE → MARK → REFINE, the same settings).

slab1d: one root (x = 0), the line from the root to the end in the outside medium; variants
(i), (ii) (face A only) and (iv); (iii) is not defined (one bank).

Run from the repository root (at most 4 variants per invocation, to stay below 10 minutes):
    nix develop --command uv run python prototypes/scripts/ffc_init1d.py OUT_DIR \\
        --thickness ckiss --g 0.02 --centre mid --variants label fused probe
    nix develop --command uv run python prototypes/scripts/ffc_init1d.py OUT_DIR --slab
    nix develop --command uv run python prototypes/scripts/ffc_init1d.py OUT_DIR --figure FIG.png
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
import ffc_basis as B  # noqa: E402
import ffc_contact as CT  # noqa: E402
import ffc_contact1d as R  # noqa: E402
import ffc_line as LN  # noqa: E402
import ffc_refine1d as RF  # noqa: E402
import ffc_tau1d as P  # noqa: E402

H0 = RF.H0
SHIFTS = (0.16, 0.5)
F64 = torch.float64


def variant_names() -> list[str]:
    out = ["label"]
    for s in SHIFTS:
        for d in ("in", "out"):
            for f in ("B", "A", "BA"):
                out.append(f"label_{f}_{d}{s}")
    return out + ["fused", "probe"]


VARIANTS = variant_names()


def probe_distance(c: CT.Contact, root: float, direction: float) -> float | None:
    """The distance from the root to the first crossing of p = 0.5, walking outward (sampled at
    0.001 mm, as ffc_contact1d.oracle_faces)."""
    d = np.arange(0.0, c.root_a - c.root_b, 0.001)
    p = c.stop_probability(root + direction * d)
    k = np.nonzero(p >= 0.5)[0]
    return None if len(k) == 0 else float(d[k[0]])


def t_hat(c: CT.Contact, v: str) -> tuple[float, float] | None:
    """(t̂_B, t̂_A) for a variant, or None (no crossing for the probe)."""
    if v == "label":
        return c.tB, c.tA
    if v == "fused":
        m = 0.5 * (c.root_a - c.root_b)
        return m, m
    if v == "probe":
        b = probe_distance(c, c.root_b, 1.0)
        a = probe_distance(c, c.root_a, -1.0)
        return None if a is None or b is None else (b, a)
    _, faces, ds = v.split("_")
    s = float(ds[2:] if ds.startswith("in") else ds[3:])
    sign = -1.0 if ds.startswith("in") else 1.0
    tb = c.tB + (sign * s if "B" in faces else 0.0)
    ta = c.tA + (sign * s if "A" in faces else 0.0)
    return tb, ta


def envelope(c: CT.Contact, x: T.Array, tb: float, ta: float) -> T.Array:
    return np.asarray(np.minimum((x - c.root_b) - tb, (c.root_a - x) - ta))


def envelope_peak(c: CT.Contact, tb: float, ta: float) -> float:
    """Where the two banks' fields meet: x − root_b − t̂_B = root_a − x − t̂_A."""
    return 0.5 * (c.root_a + c.root_b + tb - ta)


def start_nodes(x: T.Array, peak: float, mode: str) -> T.Array:
    """The nodes for τ₀ (the open choice F3): "interp" as they are; "bisect" the cell that holds
    the peak down to the deepest level; "insert" a node at the peak (then 2:1 grading)."""
    if mode == "interp" or np.any(np.abs(x - peak) < 1e-12) or not (x[0] < peak < x[-1]):
        return x
    if mode == "bisect":
        b = B.P1Basis(x)
        for _ in range(RF.MAX_LEVEL):
            b = b.refine(b.cells_at(peak))
        return b.x
    if mode == "insert":
        # the new node, then 2:1 grading (bisect the larger neighbours until it holds)
        return B.graded(np.sort(np.r_[x, peak]), np.zeros(0, np.int64))
    raise ValueError(f"peak mode: {mode!r}")


def rule_check(p: P.Problem, z: T.Array) -> dict[str, Any]:
    ex = p.basis.extrema(z)
    out = dict(admissible=bool(p.basis.admissible(z, flat_beyond_top=p.energy_form != "speed")),
               Q1=bool(np.all(p.root_slopes(z) > 0)),
               Q2c=bool(all(e.value >= 0 for e in ex if e.kind == "max")),
               Q3d=bool(not any(e.kind == "min" for e in ex)),
               maxima=[(e.x, e.value) for e in ex if e.kind == "max"])
    out["valid"] = bool(p.valid(z))
    return out


def run_variant(c: CT.Contact, x0: T.Array, v: str, peak_mode: str, label: str) -> dict[str, Any]:
    th = t_hat(c, v)
    if th is None:
        return dict(variant=v, t_hat=None, solved=False, note="the probe found no crossing")
    tb, ta = th
    pk = envelope_peak(c, tb, ta)
    x = start_nodes(x0, pk, peak_mode)
    p = RF.CFG.problem(c, x)
    z0 = envelope(c, p.x, tb, ta)
    chk = rule_check(p, z0)
    rec: dict[str, Any] = dict(variant=v, t_hat=[tb, ta], envelope_peak=pk, n_cells0=len(x) - 1, rules=chk,
                               tau0_x=p.x, tau0=z0)
    if not chk["valid"]:
        rec.update(solved=False, note="the start breaks a rule: not solved")
        return rec
    rec.update(solved=True, **RF.run_from(c, p, z0, label))
    return rec


# ---------------------------------------------------------------- slab1d (one root)


def slab_contact_like() -> tuple[LN.Line, Any]:
    line = LN.slab1d()
    st = line.settings
    sig = st.oracle_blur_vox * T.VOXEL_MM

    def lo_t(xx: torch.Tensor) -> torch.Tensor:
        p_ = torch.clamp(torch.special.ndtr((xx - line.x_stop_true) / sig), *st.oracle_clip)
        return torch.log(p_) - torch.log1p(-p_)

    return line, lo_t


def run_slab(peak_mode: str) -> dict[str, Any]:
    """slab1d: the line from the root (x = 0) to its end in the outside medium; τ₀ = d − t̂;
    the refine loop with the slab's own stop measure."""
    line, lo_t = slab_contact_like()
    x_end = float(line.nodes(H0)[-1])
    out: dict[str, Any] = {}
    names = ["label"] + [f"label_A_{d}{s}" for s in SHIFTS for d in ("in", "out")] + ["probe"]
    for v in names:
        if v == "label":
            t = line.x_stop_true
        elif v == "probe":
            d = np.arange(0.0, x_end, 0.001)
            p_ = line.stop_probability(d)
            k = np.nonzero(p_ >= 0.5)[0]
            t = float(d[k[0]])
        else:
            ds = v.split("_")[2]
            s = float(ds[2:] if ds.startswith("in") else ds[3:])
            t = line.x_stop_true + (-s if ds.startswith("in") else s)
        n = int(round(x_end / H0))
        x = np.linspace(0.0, n * H0, n + 1)
        p = P.Problem(x, lo_t, line.settings, roots=((0.0, 1.0),), energy_form=RF.CFG.energy,
                      mm_scale=RF.CFG.mm_scale, slope_bounds=RF.CFG.slope_bounds)
        z0 = x - t
        chk = rule_check(p, z0)
        rec: dict[str, Any] = dict(t_hat=t, rules=chk)
        if chk["valid"]:
            t0 = time.time()
            deadline = t0 + RF.WALL_S
            levels = []
            while True:
                z0, log = P.solve(p, z0, deadline=deadline)
                cr = p.basis.crossings(z0)
                eta = RF.indicator(p, z0)
                lv = RF.levels(p, z0)
                marked = RF.dorfler(eta, RF.THETA)
                refinable = marked[lv[marked] < RF.MAX_LEVEL]
                levels.append(dict(level=int(lv.max()), n_cells=p.basis.n_cells, newton=log.iterations,
                                   solve_reason=log.reason, stop_error_mm=cr[0][0] - line.x_stop_true if cr else None,
                                   eta_total=float(eta.sum()),
                                   damping_last=log.damping[-1] if log.damping else None,
                                   damping_max=max(log.damping) if log.damping else None,
                                   laplace=RF.laplace_state(p, z0)))
                if RF.CFG.progress is not None:
                    with open(RF.CFG.progress, "a") as f:
                        q = levels[-1]
                        f.write(f"slab1d {v} pass {len(levels) - 1} level {q['level']} cells {q['n_cells']} "
                                f"newton {q['newton']} {q['solve_reason']} stop_err {q['stop_error_mm']} "
                                f"eta {q['eta_total']:.3e}\n")
                if log.reason == "wall time" or time.time() > deadline:
                    rec["stop"] = "wall time"
                    break
                if len(marked) == 0:
                    rec["stop"] = "no cell marked"
                    break
                if len(refinable) == 0:
                    rec["stop"] = "deepest level reached"
                    break
                fine = p.basis.refine(refinable)
                z0 = p.basis.prolong(z0, fine)
                p = P.Problem(fine.x, lo_t, line.settings, roots=((0.0, 1.0),), energy_form=RF.CFG.energy,
                              mm_scale=RF.CFG.mm_scale, slope_bounds=RF.CFG.slope_bounds)
            tau_f, xs_f = p.basis.nodal(z0)
            sl = np.diff(tau_f) / np.diff(xs_f)
            # the shift as a whole: the slope stays 1 in tissue and the offset changes
            rec.update(levels=levels, seconds=time.time() - t0, start_stop_error_mm=t - line.x_stop_true,
                       slope_tissue_min=float(sl[tau_f[1:] <= 0].min()) if np.any(tau_f[1:] <= 0) else None,
                       slope_tissue_max=float(sl[tau_f[1:] <= 0].max()) if np.any(tau_f[1:] <= 0) else None,
                       x=xs_f, z=tau_f)
        out[v] = rec
    return out


# ---------------------------------------------------------------- the figure


def figure(path: Path, runs: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), dpi=90)
    k0 = "ckiss_g0.02_mid"
    if k0 in runs:
        c = runs[k0]["contact"]
        xf = np.linspace(c.root_b, c.root_a, 2000)
        for a, vs, title in ((ax[0, 0], ("label", "fused", "probe"), "τ₀ (variants i, iii, iv)"),
                             (ax[0, 1], ("label", "fused", "probe"), "solved τ, last pass")):
            a.plot(xf, c.tau_true(xf) / c.tau_scale, "k-", lw=3, alpha=0.3, label="true τ / tau_scale")
            for v in vs:
                r = runs[k0]["variants"].get(v)
                if r is None:
                    continue
                if a is ax[0, 0]:
                    a.plot(r["tau0_x"], r["tau0"], ".-", ms=3, label=f"{v} (rules ok: {r['rules']['valid']})")
                elif r.get("solved"):
                    a.plot(r["x"], r["z"], ".-", ms=3, label=v)
            a.axhline(0, color="k", lw=0.5)
            a.set(xlabel="x (mm)", ylabel="τ", title=f"{k0}: {title}", xlim=(-1.5, 1.5), ylim=(-1.5, 1.0))
            a.legend(fontsize=6)
    names = VARIANTS
    a = ax[0, 2]
    keys = list(runs)
    M = np.zeros((len(keys), len(names)))
    for i, k in enumerate(keys):
        for j, v in enumerate(names):
            r = runs[k]["variants"].get(v)
            M[i, j] = np.nan if r is None else (1.0 if r["rules"]["valid"] else 0.0) if "rules" in r else np.nan
    a.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=90, fontsize=5)
    a.set_yticks(range(len(keys)))
    a.set_yticklabels(keys, fontsize=6)
    a.set_title("τ₀ passes all rules (green) or not (red)")
    a = ax[1, 0]
    for i, k in enumerate(keys):
        for j, v in enumerate(names):
            r = runs[k]["variants"].get(v)
            if not r or not r.get("solved"):
                continue
            q = r["levels"][-1]
            for side, mk in (("B", "<"), ("A", ">")):
                e = q[f"top_error_{side}"]
                a.plot(j + 0.08 * i, abs(e) + 1e-16 if e is not None else 1.0, mk, color=f"C{i}", ms=4,
                       label=k if (j == 0 and side == "B") else None)
    a.set_yscale("log")
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=90, fontsize=5)
    a.set(ylabel="|top error| (mm), last pass; 1 = no top", title="top errors after the refine loop")
    a.legend(fontsize=5)
    a = ax[1, 1]
    for i, k in enumerate(keys):
        for j, v in enumerate(names):
            r = runs[k]["variants"].get(v)
            if r and r.get("solved"):
                q = r["levels"][-1]
                a.plot(j + 0.08 * i, q["n_cells"], "o", color=f"C{i}", ms=3, mfc="none")
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=90, fontsize=5)
    a.set(ylabel="cells, last pass", title="cells")
    a = ax[1, 2]
    for i, k in enumerate(keys):
        for j, v in enumerate(names):
            r = runs[k]["variants"].get(v)
            if r and r.get("solved"):
                st = r["levels"][-1]["state"]
                a.plot(j + 0.08 * i, {"open": 2, "collapsed": 1, "none": 0}[st], "s", color=f"C{i}", ms=3)
    a.set_yticks([0, 1, 2])
    a.set_yticklabels(["no ridge", "collapsed", "open"])
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=90, fontsize=5)
    a.set_title("ridge state, last pass")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_runs(out: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for f in sorted(out.glob("init_*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        rr = runs.setdefault(d["key"], dict(contact=d["contact"], variants={}))
        rr["variants"].update(d["variants"])
    order = [f"{th}_g{g}_{ce}" for th in ("ckiss", "equal") for g in R.GS for ce in R.CENTRES]
    return {k: runs[k] for k in order if k in runs}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--thickness", choices=sorted(CT.THICKNESS))
    ap.add_argument("--g", type=float, choices=R.GS)
    ap.add_argument("--centre", choices=R.CENTRES)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    ap.add_argument("--ends", choices=("keep", "merge"), required=True)
    ap.add_argument("--peak", choices=("interp", "bisect", "insert"), required=True)
    ap.add_argument("--slab", action="store_true")
    ap.add_argument("--energy", choices=[e for e in P.ENERGY_FORMS if e != "speed"], default="equivolume_convex")
    ap.add_argument("--slope-bounds", action="store_true", help="Q1 and Q3d as linear bounds (S1 c)")
    ap.add_argument("--figure", type=Path, default=None)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.figure:
        figure(a.figure, load_runs(a.out))
        return
    RF.CFG.energy, RF.CFG.domain, RF.CFG.ends = a.energy, "roots", a.ends
    RF.CFG.slope_bounds = a.slope_bounds
    RF.CFG.progress = a.out / "progress.log"
    t0 = time.time()
    common = dict(energy=a.energy, slope_bounds=a.slope_bounds, domain="roots", ends=a.ends, peak=a.peak, inputs="oracle", noise=0.0,
                  mm_pull_coefficient=P.mm_pull_coefficient(T.DEFAULTS),
                  equivolume_weight=P.EQUIVOLUME_LAMBDA_A_PROVISIONAL,
                  equivolume_weight_note="PROVISIONAL placeholder (the old speed prior's λA; user's E5)",
                  settings=T.DEFAULTS.record())
    if a.slab:
        res = run_slab(a.peak)
        (a.out / "init_slab1d.json").write_text(json.dumps(T._jsonable(dict(
            key="slab1d", **common, variants=res, seconds=time.time() - t0)), indent=1))
        return
    if a.thickness is None or a.g is None or a.centre is None:
        ap.error("--thickness, --g and --centre are needed")
    tA, tB = CT.THICKNESS[a.thickness]
    c = CT.Contact(tA, tB, a.g)
    x0 = RF.CFG.nodes(c, H0, a.centre)
    key = f"{a.thickness}_g{a.g}_{a.centre}"
    res = {}
    for v in a.variants:
        res[v] = run_variant(c, x0, v, a.peak, f"{key} {v}")
        r = res[v]
        print(key, v, r["rules"] if "rules" in r else r.get("note"), r.get("stop"),
              [(q["top_error_B"], q["top_error_A"], q["state"]) for q in r.get("levels", [])][-1:], flush=True)
    tag = "_".join(a.variants)
    with open(a.out / f"init_{key}_{tag}.pkl", "wb") as f:
        pickle.dump(dict(key=key, contact=c, variants=res), f)
    rec = dict(key=key, **common,
               variants={v: {k: val for k, val in r.items() if k not in ("x", "z", "eta", "tau0", "tau0_x")}
                         for v, r in res.items()},
               seconds=time.time() - t0, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    (a.out / f"init_{key}_{tag}.json").write_text(json.dumps(T._jsonable(rec), indent=1))


if __name__ == "__main__":
    main()
