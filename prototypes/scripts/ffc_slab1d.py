"""slab1d with the continuous column energy (route R1, stop form S2): oracle inputs, noise 0,
h = 0.16 mm, from three starts. Writes the measures (JSON), the solutions (npz) and one
figure.

Starts: τ = x − s0 (slope 1, the speed prior's mean), with
- "good": s0 = the node nearest to the oracle's p = 0.5 (the lower one on a tie);
- "wrong+1mm": s0 = true stop + 1 mm (in the outside medium);
- "wrong-1mm": s0 = true stop − 1 mm (in the cortex).

Measures (ff_types / the stage-1 measures on one column):
- the stop: s − x_stop_true (mm);
- τ against the truth at the nodes of the moving region, and at the start point (the offset);
- the speed v_e = 1 / τ'_e of each cell that holds a part of the moving region, against the
  true speed 1;
- the readout at the stop: τ(s) (the jump J; 0 by the J0 branch);
- the Newton iterations, the energy per iteration, the energy of the true state;
- the J+ branch: its energy along shifts J of the solution;
- time and peak memory.

Run from the repository root:
    nix develop --command uv run python prototypes/scripts/ffc_slab1d.py OUT_DIR FIGURE.png
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import ffc_line as LN  # noqa: E402
import ffc_solve1d as F  # noqa: E402

H_MM = 0.16
WRONG_MM = 1.0
J_PROBE = np.array([1e-6, 1e-3, 1e-2, 0.1])


def starts(line: LN.Line, d: F.Discretisation) -> dict[str, F.State]:
    lo = line.stop_log_odds(d.x)
    i = int(np.argmin(np.abs(lo)))
    out = {}
    for name, s0 in (("good", float(d.x[i])), ("wrong+1mm", line.x_stop_true + WRONG_MM),
                     ("wrong-1mm", line.x_stop_true - WRONG_MM)):
        out[name] = F.State(d.x - s0, s0)
    return out


def measures(line: LN.Line, d: F.Discretisation, k: F.Constants, st: F.State, log: F.SolveLog) -> dict[str, Any]:
    tt = line.tau_true(d.x)
    mov = d.x <= st.s
    na = F.n_active(d.x, st.s)
    g = (d.G[: na - 1, :na] @ st.tau[:na])
    v = 1.0 / g
    c = F.stop_cell(d.x, st.s)
    tau_s = st.tau[c] + (st.s - d.x[c]) / d.h * (st.tau[c + 1] - st.tau[c])
    e = F.terms(d, k, st, "J0")
    return dict(
        stop_error_mm=st.s - line.x_stop_true,
        tau_abs_error_max=float(np.max(np.abs(st.tau[mov] - tt[mov]))),
        tau_error_at_start=float(st.tau[0] - tt[0]),
        speed_abs_error_max=float(np.max(np.abs(v - line.speed_true(0.5 * (d.x[: na - 1] + d.x[1:na]))))),
        speed_cells=int(na - 1),
        tau_at_stop=float(tau_s),
        newton_iterations=log.iterations,
        converged=log.converged,
        reason=log.reason,
        energy=e,
        energy_per_iteration=log.energy,
        decrement_per_iteration=log.decrement,
        step_per_iteration=log.step,
        damping_per_iteration=log.damping,
        s_per_iteration=log.s,
        jump_branch=dict(J=J_PROBE.tolist(), energy=F.jump_branch(d, k, st, J_PROBE).tolist(),
                         infimum=e["total"] - k.e0 + k.e1),
    )


def figure(path: Path, line: LN.Line, d: F.Discretisation, runs: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(10, 7), dpi=100)
    xf = np.linspace(line.x_start, line.x_end, 2000)
    a = ax[0, 0]
    a.plot(xf, line.tau_true(xf), "k-", lw=3, alpha=0.3, label="true τ")
    for (name, r), mk in zip(runs.items(), ("o", "s", "^"), strict=True):
        st = r["state"]
        m = d.x <= st.s + d.h
        a.plot(d.x[m], st.tau[m], mk, ms=4, mfc="none", label=f"{name}: τ at nodes")
        a.plot(r["start"].s, 0.0, mk, ms=8, color="r")
    a.axvline(line.x_stop_true, color="k", ls=":", lw=1)
    a.set(xlabel="x (mm)", ylabel="τ", title="τ (red: the start's stop)")
    a.legend(fontsize=7)
    a = ax[0, 1]
    for name, r in runs.items():
        st = r["state"]
        na = F.n_active(d.x, st.s)
        v = 1.0 / (d.G[: na - 1, :na] @ st.tau[:na])
        a.step(d.x[:na], np.r_[v, v[-1]], where="post", label=name)
    a.axhline(1.0, color="k", ls=":", lw=1)
    a.set(xlabel="x (mm)", ylabel="speed v = 1/τ' (mm per unit τ)", title="speed per cell (true: 1)")
    a.legend(fontsize=7)
    a = ax[1, 0]
    a.plot(xf, line.stop_probability(xf), "k-", label="oracle p (before the clip)")
    for name, r in runs.items():
        a.axvline(r["state"].s, ls="--", lw=1, label=f"{name}: s − true = {r['m']['stop_error_mm']:.1e} mm")
    for xn in d.x:
        a.axvline(xn, color="0.85", lw=0.5, zorder=0)
    a.set(xlim=(line.x_stop_true - 0.5, line.x_stop_true + 0.5), xlabel="x (mm), grey: nodes",
          ylabel="p", title="stop (true stop in the middle of a cell)")
    a.legend(fontsize=7)
    a = ax[1, 1]
    for name, r in runs.items():
        E = np.asarray(r["m"]["energy_per_iteration"]) - r["E_true"]
        a.semilogy(np.arange(len(E)), np.maximum(E, 1e-16), "o-", ms=3, label=name)
    a.set(xlabel="Newton iteration", ylabel="E − E(true state)", title="energy (floor 1e-16)")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="output directory (JSON and npz)")
    ap.add_argument("fig", type=Path, help="the figure (PNG)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.fig.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    line = LN.slab1d()
    d = F.discretise(line, H_MM)
    k = F.constants(line.settings)
    E_true = F.terms(d, k, F.true_state(d), "J0")["total"]
    runs: dict[str, Any] = {}
    for name, st0 in starts(line, d).items():
        t1 = time.time()
        st, log = F.solve(d, k, st0, "J0")
        m = measures(line, d, k, st, log)
        m["seconds"] = time.time() - t1
        runs[name] = dict(state=st, start=st0, m=m, E_true=E_true)
    rec = dict(
        phantom="slab1d", inputs="oracle", noise=0.0, h_mm=H_MM, route="R1", stop_form="S2",
        n_nodes=len(d.x), x_start=line.x_start, x_end=float(d.x[-1]), x_stop_true=line.x_stop_true,
        pia_in_cells=(line.x_stop_true - line.x_start) / H_MM, energy_true_state=E_true,
        constants=k.__dict__, settings=line.settings.record(),
        runs={n: dict(start_s=r["start"].s, **r["m"]) for n, r in runs.items()},
        seconds=time.time() - t0,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,  # macOS: bytes
    )
    (a.out / "slab1d.json").write_text(json.dumps(T._jsonable(rec), indent=1))
    np.savez_compressed(a.out / "slab1d.npz", x=d.x, **{f"tau_{n}": r["state"].tau for n, r in runs.items()},
                        **{f"s_{n}": r["state"].s for n, r in runs.items()})
    figure(a.fig, line, d, runs)
    for n, r in rec["runs"].items():
        print(f"{n}: stop {r['stop_error_mm']:+.2e} mm, |τ err| {r['tau_abs_error_max']:.2e}, "
              f"|v err| {r['speed_abs_error_max']:.2e}, τ(s) {r['tau_at_stop']:+.1e}, "
              f"{r['newton_iterations']} Newton, {r['reason']}")
    print(f"E(true) {E_true:.12f}; {rec['seconds']:.1f} s; peak {rec['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
