"""The PR 6 runs of the flow-field phantoms stage (docs/plans/flow-field-phantoms.md,
PR 6: hairpins, jumps, the grid and calibration). The output directory is an argument;
nothing here names a fixed path.

Modes
-----

- ``probe``: the cost of one phantom on one grid. It builds the phantom, the inputs and
  the grid, and solves with ``max_sweeps`` = ``--sweeps`` per width. It writes
  ``<out>/<phantom>/probe_<source>_<noise>_<grid>.json``: the node count, the time to
  build the inputs, the time per sweep of each width, and the peak RSS. The run plan
  (the time and peak of each full run) is estimated from these numbers.
- ``octree``: one run on the adaptive octree (`ff_octree.adapt` with the defaults). It
  writes the run directory of `ff_run` (``ff_types.run_dir``, grid label "octree"):
  field.npz (with the octree's arrays), columns.npz, column_errors.npz, run.json and
  the views of `ff_run`, and ``adapt.json`` (the record of every round, the
  prior-shaped leaves and `ff_octree.hanging_step` of the final field). The columns
  start at the roots (the PR 3 start fix). The phantom ``noise`` has no truth: its
  run writes adapt.json only, with the null inputs of `ff_octree.null_inputs`.

The uniform-grid runs use `ff_run.py` itself (``--grid uniform:S``).

Usage (repository root):
    nix develop --command uv run python prototypes/scripts/ff_pr6_runs.py probe \\
        fundus_g0.04 --source oracle --noise 0 --grid uniform:0.16 --sweeps 2 \\
        --features FEATURES_JSON --out OUT_DIR
    nix develop --command uv run python prototypes/scripts/ff_pr6_runs.py octree \\
        fundus_g0.04 --source oracle --noise 0 --truth TRUTH_DIR \\
        --features FEATURES_JSON --out OUT_DIR

NumPy or Torch: this file has no kernel. It calls `ff_meanfield.solve`,
`ff_octree.adapt` and the measures of `ff_run`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as FG  # noqa: E402
import ff_meanfield as MF  # noqa: E402
import ff_octree as O  # noqa: E402
import ff_run as FR  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

OCTREE_LABEL = "octree"


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def phantom_inputs(
    name: str,
    source: str,
    noise: float,
    feats: Any,
    site: str,
    seed: int,
    shape: tuple[int, int, int] | None,
) -> tuple[T.Phantom, T.Inputs]:
    """The phantom and its inputs. ``noise`` takes the null inputs of
    `ff_octree.null_inputs` (PR 2: its image stop field is not a null), whatever the
    source. Else `ff_run.make_inputs`, with the noise-0 phantom as the reference of the
    image stop field."""
    ph = FTR.build_phantom(name, feats, site=site, noise=noise, seed=seed, shape=shape)
    if name == "noise":
        return ph, O.null_inputs(ph)
    ref = None
    if source == "image" and noise != 0.0:
        ref = FTR.build_phantom(
            name, feats, site=site, noise=0.0, seed=seed, shape=shape
        )
    return ph, FR.make_inputs(ph, source, ref)  # type: ignore[arg-type]


def make_grid(spec: str, shape: Sequence[int]) -> T.Grid:
    """ "uniform:S" (ff_grid.UniformGrid) or "octree" (the level-0 octree)."""
    if spec == OCTREE_LABEL:
        return O.OctreeGrid.base(tuple(int(s) for s in shape))  # type: ignore[arg-type]
    _, spacing = FR.parse_grid(spec)
    return FG.UniformGrid((int(shape[0]), int(shape[1]), int(shape[2])), spacing)


def _versions(truth_path: Path | None, features: Path) -> dict[str, str]:
    commit = FR._git("rev-parse", "HEAD")
    dirty = FR._git("status", "--porcelain", "--untracked-files=no")
    v = {
        "git_commit": commit or "unknown",
        "git_dirty": "unknown" if dirty is None else str(bool(dirty)).lower(),
        "features_sha256": FR._sha256(features),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if truth_path is not None:
        v["truth_sha256"] = FR._sha256(truth_path)
    return v


# ---------------------------------------------------------------- probe


def probe(a: argparse.Namespace, feats: Any, shape: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    ph, inputs = phantom_inputs(
        a.phantom, a.source, a.noise, feats, a.site, a.seed, shape
    )
    t_inputs = time.perf_counter() - t0
    t0 = time.perf_counter()
    grid = make_grid(a.grid, ph.shape)
    n_nodes = int(grid.n_nodes)
    n_colours = int(grid.n_colours)
    t_grid = time.perf_counter() - t0
    settings = dataclasses.replace(T.DEFAULTS, max_sweeps=int(a.sweeps))
    t0 = time.perf_counter()
    _, _, log = MF.solve(grid, inputs, ph.image["noise_sd"], settings)
    t_solve = time.perf_counter() - t0
    per_sweep = [
        float(t) / max(int(s), 1)
        for t, s in zip(log["time_s"], log["sweeps"], strict=True)
    ]
    return dict(
        phantom=a.phantom,
        source=a.source,
        noise=a.noise,
        grid=a.grid,
        shape=list(ph.shape),
        n_nodes=n_nodes,
        n_colours=n_colours,
        sweeps=[int(s) for s in log["sweeps"]],
        time_inputs_s=t_inputs,
        time_grid_s=t_grid,
        time_solve_s=t_solve,
        time_per_width_s=[float(t) for t in log["time_s"]],
        time_per_sweep_s=per_sweep,
        time_setup_s=t_solve - float(sum(log["time_s"])),
        peak_rss_bytes=peak_rss_bytes(),
    )


# ---------------------------------------------------------------- octree


def octree_run(a: argparse.Namespace, feats: Any, shape: Any) -> dict[str, Any]:
    t_start = time.perf_counter()
    ph, inputs = phantom_inputs(
        a.phantom, a.source, a.noise, feats, a.site, a.seed, shape
    )
    say = None if a.quiet else (lambda s: print(s, flush=True))
    t0 = time.perf_counter()
    grid, field, extras, log, arec = O.adapt(
        ph, inputs, ph.image["noise_sd"], log_fn=say, max_sweeps=a.max_sweeps
    )
    t_adapt = time.perf_counter() - t0
    label = OCTREE_LABEL + (f"_seed{a.seed}" if a.seed else "")
    if a.max_sweeps is not None:
        label += f"_test{a.max_sweeps}"

    out = T.run_dir(a.out, ph.name, a.source, a.noise, label)  # type: ignore[arg-type]
    arec["time_s"] = t_adapt
    arec["phantom"] = ph.record()
    arec["inputs"] = (
        "null (ff_octree.null_inputs)" if a.phantom == "noise" else a.source
    )
    if a.phantom == "noise":
        arec["peak_rss_bytes"] = peak_rss_bytes()
        (out / "adapt.json").write_text(json.dumps(T._jsonable(arec), indent=1) + "\n")
        return dict(out=str(out), time_s=time.perf_counter() - t_start,
                    peak=arec["peak_rss_bytes"])  # fmt: skip
    tpath = FR._truth_path(a.truth, a.phantom)
    truth = T.Truth.from_npz(tpath)
    field.to_npz(out / "field.npz", extras)
    rec = T.RunRecord(
        phantom=ph.record(),
        source=a.source,
        grid=FR.grid_record(grid, OCTREE_LABEL),
        settings=T.DEFAULTS.record(),
        changed_settings={},
        origin="top",
        versions=_versions(tpath, a.features),
        free_energy=log["free_energy"],
        sweeps=[int(s) for s in log["sweeps"]],
        free_energy_rises=log["free_energy_rises"],
        pruning=log["pruning"],
        defects=list(log["defects"]),
        measures={},
    )
    rec.measures["inference"] = dict(
        sweeps_per_width=rec.sweeps,
        end_reason=log["end_reason"],
        free_energy_last=[fl[-1] if fl else None for fl in log["free_energy"]],
        n_rises=len(log["free_energy_rises"]),
        terms=log["terms"],
        run=log["run"],
        note="the log of the final octree's solve; adapt.json has every round",
    )
    rec.measures["inputs"] = {
        **inputs.record(),
        **(inputs.stop.record() if hasattr(inputs.stop, "record") else {}),
    }
    rec.measures["octree"] = dict(
        final=arec["final"]["level_counts"],
        n_rounds=len(arec["rounds"]),
        hanging_step=arec["final"]["hanging_step"],
        n_prior_shaped=arec["final"]["prior_shaped"]["count"],
    )
    t_int, t_meas = FR._measure_write(
        ph, truth, field, extras, out, rec,
        source=a.source, settings=T.DEFAULTS, columns_from="root",
    )  # fmt: skip
    rec.time_s = time.perf_counter() - t_start
    rec.peak_mem_bytes = peak_rss_bytes()
    rec.measures["cost"] = dict(
        time_s=rec.time_s,
        adapt_s=t_adapt,
        integrate_s=t_int,
        measure_s=t_meas,
        peak_mem_bytes=rec.peak_mem_bytes,
        n_nodes=int(grid.n_nodes),
        n_columns=int(truth.n),
    )
    rec.to_json(out / "run.json")
    arec["peak_rss_bytes"] = rec.peak_mem_bytes
    (out / "adapt.json").write_text(json.dumps(T._jsonable(arec), indent=1) + "\n")
    return dict(out=str(out), time_s=rec.time_s, peak=rec.peak_mem_bytes)


# ---------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=("probe", "octree"))
    ap.add_argument("phantom", choices=FTR.PHANTOMS)
    ap.add_argument("--source", choices=("oracle", "image"), required=True)
    ap.add_argument("--noise", type=float, required=True)
    ap.add_argument("--grid", default="uniform:0.16", help="probe: uniform:S or octree")
    ap.add_argument("--sweeps", type=int, default=2, help="probe: sweeps per width")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", default=None, help="X,Y,Z voxels (not the plan's box)")
    ap.add_argument(
        "--max-sweeps",
        type=int,
        default=None,
        help="octree: a test of the script only (the run directory says so)",
    )
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    shape = tuple(int(s) for s in a.shape.split(",")) if a.shape else None
    feats = FP.features(a.site, str(a.features))  # type: ignore[no-untyped-call]
    if a.mode == "probe":
        rec = probe(a, feats, shape)
        d = Path(a.out) / a.phantom
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"probe_{a.source}_{a.noise:g}_{a.grid.replace(':', '')}.json"
        p.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"wrote {p}: {rec['n_nodes']} nodes, per sweep "
              f"{[round(t, 1) for t in rec['time_per_sweep_s']]} s, peak "
              f"{rec['peak_rss_bytes'] / 2**30:.2f} GiB")  # fmt: skip
        return
    if a.phantom != "noise" and a.truth is None:
        ap.error("octree needs --truth")
    res = octree_run(a, feats, shape)
    print(f"wrote {res['out']} ({res['time_s']:.0f} s, peak "
          f"{res['peak'] / 2**30:.2f} GiB)")  # fmt: skip


if __name__ == "__main__":
    main()
