"""The PR 5 runs of the octree (docs/plans/flow-field-phantoms.md, PR 5; coordinator
decisions CD-6 and CD-10). The output directory is an argument; nothing here names a
fixed path.

Two kinds of run, on the slab (oracle inputs, noise 0) and on the noise phantom (null
inputs of `ff_octree.null_inputs`, noise 1.0, seed 0):

- ``priors``: the finding of CD-6 in numbers. The free-energy test of a refinement
  from 0.64 mm to 0.32 mm compares two different priors (at 0.64 mm the Q1 prior
  λ(K + M/ℓ²) with ℓ = 0.5 mm changes sign along the axes). The run solves the octree
  of level-0 leaves (0.64 mm) and the octree of level-1 leaves everywhere (0.32 mm;
  it has no hanging nodes), each from the initial field, and gives F and each term.
  Controls: each grid solved again from its own field (the restart effect of the R9
  finding), and the 0.32-mm grid solved from the 0.64-mm field (as an adapt trial
  does).
- ``adapt``: `ff_octree.adapt` with the defaults, its record of every round, and the
  step at the hanging nodes of the final octree (`ff_octree.hanging_step`).

Each run writes ``<out>/<phantom>/<kind>.json``, with the time and the peak RSS.

Usage (repository root):
    nix develop --command uv run python prototypes/scripts/ff_octree_runs.py \\
        --kind priors --phantom slab --features FEATURES_JSON --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_meanfield as MF  # noqa: E402
import ff_octree as O  # noqa: E402
import ff_oracle as FO  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def phantom_and_inputs(
    name: str, feats: Any, site: str, shape: tuple[int, int, int] | None
) -> tuple[T.Phantom, T.Inputs]:
    """slab: oracle inputs, noise 0. noise: null inputs, noise 1.0, seed 0."""
    if name == "slab":
        ph = FTR.build_phantom("slab", feats, site=site, noise=0.0, shape=shape)
        return ph, FO.oracle_inputs(ph)
    if name == "noise":
        ph = FTR.build_phantom(
            "noise", feats, site=site, noise=1.0, seed=0, shape=shape
        )
        return ph, O.null_inputs(ph)
    raise ValueError(f"unknown phantom {name!r}")


def level_tree(shape: tuple[int, int, int], level: int) -> O.OctreeGrid:
    """The octree with every leaf at `level` (no hanging nodes)."""
    g = O.OctreeGrid.base(shape)
    for _ in range(level):
        g = g.refine(np.arange(g.n_cells))
    return g


def _solve(
    g: O.OctreeGrid, ph: T.Phantom, inputs: T.Inputs, init: T.Field | None
) -> tuple[T.Field, dict[str, Any]]:
    t0 = time.perf_counter()
    field, _, log = MF.solve(g, inputs, ph.image["noise_sd"], init=init)
    terms = {k: float(v) for k, v in log["terms"][-1].items() if k != "width"}
    q = field.q_stop
    return field, dict(
        n_nodes=g.n_nodes,
        n_colours=g.n_colours,
        cell_mm=float(g.cell_size_mm.min()),
        terms=terms,
        free_energy=terms["free_energy"],
        sweeps=log["sweeps"],
        end_reason=log["end_reason"],
        defects=log["defects"],
        n_stopped=int((q > 0.5).sum()),
        time_s=time.perf_counter() - t0,
    )


def run_priors(ph: T.Phantom, inputs: T.Inputs) -> dict[str, Any]:
    g64 = level_tree(ph.shape, 0)
    g32 = level_tree(ph.shape, 1)
    f64, s64 = _solve(g64, ph, inputs, None)
    f32, s32 = _solve(g32, ph, inputs, None)
    _, s64r = _solve(g64, ph, inputs, f64)
    _, s32r = _solve(g32, ph, inputs, f32)
    _, s32i = _solve(g32, ph, inputs, f64)

    def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
        return {k: b["terms"][k] - a["terms"][k] for k in a["terms"]}

    return dict(
        solves={
            "0.64": s64,
            "0.32": s32,
            "0.64_restart": s64r,
            "0.32_restart": s32r,
            "0.32_from_0.64": s32i,
        },  # fmt: skip
        delta={
            "0.32 - 0.64 (both from the initial field)": delta(s64, s32),
            "0.32 from the 0.64 field - 0.64 (the adapt trial)": delta(s64, s32i),
            "0.64 restart - 0.64": delta(s64, s64r),
            "0.32 restart - 0.32": delta(s32, s32r),
        },
    )


def run_adapt(ph: T.Phantom, inputs: T.Inputs) -> dict[str, Any]:
    t0 = time.perf_counter()
    g, _, _, _, rec = O.adapt(ph, inputs, ph.image["noise_sd"], log_fn=print)
    rec["time_s"] = time.perf_counter() - t0
    rec["final_n_colours"] = g.n_colours
    return rec


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["priors", "adapt"], required=True)
    ap.add_argument("--phantom", choices=["slab", "noise"], required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument(
        "--shape", default=None, help="e.g. 24,24,24 (default: the plan's box)"
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    shape = tuple(int(s) for s in a.shape.split(",")) if a.shape else None
    feats = FP.features(a.site, str(a.features))  # type: ignore[no-untyped-call]
    ph, inputs = phantom_and_inputs(a.phantom, feats, a.site, shape)  # type: ignore[arg-type]
    t0 = time.perf_counter()
    rec = run_priors(ph, inputs) if a.kind == "priors" else run_adapt(ph, inputs)
    rec["phantom"] = ph.record()
    rec["inputs"] = "oracle" if a.phantom == "slab" else "null (ff_octree.null_inputs)"
    rec["time_total_s"] = time.perf_counter() - t0
    rec["peak_rss_bytes"] = _peak_rss_bytes()
    d = a.out / a.phantom
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.kind}.json").write_text(json.dumps(T._jsonable(rec), indent=1) + "\n")
    print(f"wrote {d / (a.kind + '.json')}: {rec['time_total_s']:.0f} s, "
          f"peak {rec['peak_rss_bytes'] / 2**30:.2f} GiB")  # fmt: skip


if __name__ == "__main__":
    main()
