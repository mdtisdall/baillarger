"""Diagnostics of the R11 failure (the provisional change for R11; not a change of the
solver). Each runs the ORIGINAL solver (recentre_every_sweep off) through
`ff_run.run`, with a start field (`init`) that only replaces the initial field:

- start "truth": init = the true τ (min(τ_true, TAU_TOP)), the true column direction
  (the oracle tensor) and the oracle stop probability. It tests whether the
  energy's minimum is near the truth at full size.
- start "coarse": the plan's coarse-grid option of question 6. The field is solved on
  the uniform grid of settings.coarse_first_mm first; that field is the init of the
  run. init is a start field only, not a prior (CD-9). Q1 does not carry ℓ at
  0.64 mm (CD-6).

Usage (from the repository root; every path is an argument):
    uv run python prototypes/scripts/ff_r11_diag.py tau truth --out <dir> \\
        --truth <truth dir> --features <fold_features.json> --profile-dtype float32

The run directory is <out>/<start>/<phantom>/<source>_<noise>_<grid>/ (as ff_run).
The coarse start also writes coarse_log.json there.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_meanfield as MF  # noqa: E402
import ff_run as FR  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402


def truth_field(phantom: T.Phantom, inputs: T.Inputs, grid: T.Grid) -> T.Field:
    """A one-slot Field of the truth: τ = min((φ − 1) · tau_scale, TAU_TOP), the oracle
    tensor's direction (+x where it has none), q_stop = the oracle stop probability."""
    geo = phantom.geometry
    assert geo is not None
    nodes = grid.nodes
    phi = geo.phi(geo.to_local(nodes))
    tau = np.minimum((phi - 1.0) * phantom.tau_scale, T.TAU_TOP)
    d, _ = inputs.tensor.at(nodes)
    d = np.nan_to_num(np.asarray(d, np.float64))
    nr = np.linalg.norm(d, axis=1)
    d = np.where(nr[:, None] > 0, d / np.maximum(nr, 1e-300)[:, None], [1.0, 0.0, 0.0])
    qs = T.sigmoid(np.asarray(inputs.stop.log_odds(nodes), np.float64))
    n = len(nodes)
    return T.Field(
        grid=grid,
        tau=tau[:, None],
        direction=d[:, None, :],
        q=(1.0 - qs)[:, None],
        q_stop=qs,
        n_kept=np.ones(n, np.int64),
    )


def main(argv: Sequence[str] | None = None) -> T.RunRecord:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("phantom")
    ap.add_argument("start", choices=("truth", "coarse"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--grid", default="uniform:0.16")
    ap.add_argument(
        "--profile-dtype", choices=("float32", "float64"), default="float64"
    )
    a = ap.parse_args(argv)
    settings = T.DEFAULTS
    dtype = np.dtype(a.profile_dtype).type
    feats = FP.features(a.site, str(a.features))  # type: ignore[no-untyped-call]
    phantom = FTR.build_phantom(a.phantom, feats, site=a.site, noise=0.0, seed=0)
    inputs = FR.make_inputs(phantom, "oracle", None)
    grid = FR.make_grid(a.grid, phantom.shape)
    truth = T.Truth.from_npz(a.truth / a.phantom / "truth.npz")
    out = T.run_dir(a.out / a.start, phantom.name, "oracle", 0.0, FR.grid_label(a.grid))
    out.mkdir(parents=True, exist_ok=True)

    def say(s: str) -> None:
        print(s, flush=True)

    if a.start == "truth":
        init = truth_field(phantom, inputs, grid)
    else:
        cspec = f"uniform:{settings.coarse_first_mm:g}"
        cgrid = FR.make_grid(cspec, phantom.shape)
        say(f"coarse solve on {cspec}: {cgrid.n_nodes} nodes")
        noise_sd = [float(s) for s in phantom.image["noise_sd"]]
        init, _, clog = MF.solve(
            cgrid, inputs, noise_sd, settings, profile_dtype=dtype, log_fn=say
        )
        (out / "coarse_log.json").write_text(json.dumps(clog, allow_nan=True))
    versions = {
        "git_commit": FR._git("rev-parse", "HEAD") or "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "diagnostic_start": a.start,
    }
    rec = FR.run(
        phantom, inputs, truth, grid, out,
        source="oracle", grid_spec=a.grid, settings=settings, profile_dtype=dtype,
        init=init, versions=versions, log_fn=say,
    )  # fmt: skip
    say(f"wrote {out} ({rec.time_s:.1f} s, peak {rec.peak_mem_bytes / 1e9:.2f} GB)")
    return rec


if __name__ == "__main__":
    main()
