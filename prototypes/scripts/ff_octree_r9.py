"""The provisional design change for R9 (docs/plans/flow-field-phantoms.md, *An
overnight run*, item 3): the acceptance of a refinement or a coarsening by the free
energy, against a control solve.

The failure of PR 5: `ff_octree.adapt` compares F of a trial (a solve of the new grid
with init = the current field) with F of the current solve (a solve that started from
the initial field). A second solve of the SAME grid from its own field lowers F by
about as much as the trial does (noise phantom: −9418.77 against −9417.63). So the
gain is the restart of the solve, not the change of the grid.

The changed rule: a nominated change is accepted only if F of the trial is lower than
F of a control, a solve of the current grid with init = the current field, with the
same settings and the same sweep limits as the trial. Only the grid differs. No
threshold and no margin. The rule is `ff_octree.adapt(..., acceptance="control")`;
the default of `adapt` stays the original rule.

The run: `adapt` with the changed rule on the slab (oracle inputs, noise 0) and on the
noise phantom (null inputs of `ff_octree.null_inputs`, noise 1.0, seed 0), as the
`adapt` runs of `ff_octree_runs.py`. It writes ``<out>/<phantom>/adapt_control.json``
and prints one line per trial: the nominated count, ΔF of the trial against the
control and against the current solve, the verdict, and the solve times.

Usage (repository root):
    nix develop --command uv run python prototypes/scripts/ff_octree_r9.py \\
        --phantom noise --features FEATURES_JSON --shape 24,24,24 --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import ff_octree as O  # noqa: E402
import ff_octree_runs as R  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402


def adapt_control(
    phantom: T.Phantom, inputs: T.Inputs, settings: T.SolverSettings = T.DEFAULTS
) -> tuple[O.OctreeGrid, dict[str, Any]]:
    """`ff_octree.adapt` with the changed rule. Returns (final octree, record)."""
    g, _, _, _, rec = O.adapt(
        phantom, inputs, phantom.image["noise_sd"], settings, acceptance="control"
    )
    return g, rec


def trial_lines(rec: dict[str, Any]) -> list[str]:
    """One line per trial of an `adapt` record with the changed rule."""
    out = [f"base F {rec['base']['terms']['free_energy']:.10g}"]
    for r in rec["rounds"]:
        out.append(
            f"round {r['round']}: nominated refine {r['n_refine_nominated']}, "
            f"coarsen {r['n_coarsen_nominated']}"
        )
        for tr in r["trials"]:
            if tr["free_energy_after"] is None:
                out.append(f"  {tr['kind']}: the octree does not change; not solved")
                continue
            c = tr.get("control", {})
            out.append(
                f"  {tr['kind']}: dF vs control {tr.get('delta_free_energy_control')}"
                f", dF vs current {tr['delta_free_energy']:.6g}, accepted "
                f"{tr['accepted']}, trial {tr['solve']['time_total_s']:.1f} s, "
                f"control {c.get('time_total_s', float('nan')):.1f} s, leaves "
                f"{tr['level_counts_after']['leaves_per_level']}"
            )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", choices=["slab", "noise"], required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--shape", default=None, help="e.g. 24,24,24 (default: the plan's box)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    shape = tuple(int(s) for s in a.shape.split(",")) if a.shape else None
    feats = FP.features(a.site, str(a.features))  # type: ignore[no-untyped-call]
    ph, inputs = R.phantom_and_inputs(a.phantom, feats, a.site, shape)  # type: ignore[arg-type]
    t0 = time.perf_counter()
    g, rec = adapt_control(ph, inputs)
    rec["time_total_s"] = time.perf_counter() - t0
    rec["final_n_colours"] = g.n_colours
    rec["phantom"] = ph.record()
    rec["inputs"] = "oracle" if a.phantom == "slab" else "null (ff_octree.null_inputs)"
    rec["peak_rss_bytes"] = R._peak_rss_bytes()
    d = a.out / a.phantom
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapt_control.json").write_text(json.dumps(T._jsonable(rec), indent=1) + "\n")
    print("\n".join(trial_lines(rec)))
    print(f"wrote {d / 'adapt_control.json'}: {rec['time_total_s']:.0f} s, "
          f"peak {rec['peak_rss_bytes'] / 2**30:.2f} GiB")  # fmt: skip


if __name__ == "__main__":
    main()
