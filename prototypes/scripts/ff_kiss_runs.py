"""The runs of the kissing-surfaces PR (docs/plans/flow-field-phantoms.md, *PR 4*, task 4.1).

The plain runs of the matrix (every connected-kiss variant, both input sources, both noise
levels, uniform grid at 0.16 mm) use `ff_run.py` as it is. This script adds the three runs that
`ff_run.py` cannot make alone:

- ``coarse``: question 6. Solve at 0.64 mm first (`SolverSettings.coarse_first_mm`), then solve
  at the fine grid with the coarse field as the initial field (`ff_run.run(init=...)`). The
  coarse field is an initial field only, not a prior (coordinator decision CD-9). The coarse
  solve's log goes to ``coarse.json`` in the run directory.
- ``anchor``: question 6. The origin of τ is one anchor node, not the top
  (`ff_run.run(origin="anchor", anchor_node=...)`, coordinator decision CD-8). The anchor is the
  grid node nearest to the point at relative depth 0.5 on one true bank-A column far from the
  fundus (`anchor_point`).
- ``shiftB``: the independence test of *Measures* (R6). Bank B's image is changed: in bank B
  the cortical profile of both echoes is moved by SHIFT_MM toward the top (the value at
  relative depth φ is the original value at φ − SHIFT_MM / tB). The labels, the geometry and so
  the truth and the oracle inputs do not change. Bank A's image does not change.

The run directory is ``ff_types.run_dir(out, phantom, source, noise, label)`` with the label
"uniform0.16_coarse", "uniform0.16_anchor" or "uniform0.16_shiftB", so these runs sit next to
the plain run "uniform0.16" of the same phantom.

Usage:
    python ff_kiss_runs.py {coarse,anchor,shiftB} PHANTOM --source oracle --noise 0
        --out OUT_DIR --truth TRUTH_DIR --features FEATURES_JSON [--grid uniform:0.16]
        [--profile-dtype float32] [--quiet]

NumPy or Torch: no kernel here. The script builds inputs and calls the solver.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_ckiss as CK  # noqa: E402
import ff_meanfield as MF  # noqa: E402
import ff_run as FR  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

SHIFT_MM = 0.2  # the plan's *Measures*, item *Independence across the contact*
ANCHOR_PHI = 0.5
ANCHOR_FAR_FRACTION = 0.1  # the anchor column is among the 10 % of bank-A roots farthest from the fundus
MODES = ("coarse", "anchor", "shiftB")


# ---------------------------------------------------------------- the shifted bank B


class ShiftedB(CK.ConnectedKiss):
    """A connected kiss whose relative depth in bank B is moved by `shift_mm`: `geom` gives
    φ − shift_mm / tB as the relative depth of bank-B cortex. Only the image uses this value
    (`ff_ckiss._values` reads the profile at it). The labels, φ, the stop surface and the bank
    are those of the parent."""

    shift_mm: float = SHIFT_MM

    def geom(self, Y: Any) -> Any:
        lab, rel, bank, r, c, u = super().geom(Y)
        rel = np.where(bank == 1, rel - self.shift_mm / self.p["tB"], rel)
        return lab, rel, bank, r, c, u


def shifted_phantom(
    variant: str, feats: Any, site: str, noise: float, seed: int, shift_mm: float = SHIFT_MM,
    shape: Any = None,
) -> T.Phantom:
    """The phantom `variant` with bank B's profile moved by `shift_mm` toward the top. The
    Phantom's geometry is the parent's (so the truth and the oracle inputs are the same); only
    the image differs."""
    base = CK.ckiss(variant, feats, site=site, noise=noise, seed=seed, shape=shape)
    geo = base.geometry
    assert isinstance(geo, CK.ConnectedKiss)
    sgeo = ShiftedB(geo.kind, geo.shape, dict(geo.p), geo.R.copy(), geo.origin_mm)
    sgeo.shift_mm = shift_mm
    image = CK.make(sgeo, feats, noise, seed, bool(geo.p["flat"]))
    image["fold"] = geo
    params = dict(base.params, shift_bank_B_mm=shift_mm)
    return T.Phantom(name=base.name, geometry=geo, image=image, features=feats, site=site,
                     noise=base.noise, seed=base.seed, tau_scale=base.tau_scale, params=params)


# ---------------------------------------------------------------- the anchor


def anchor_point(truth: T.Truth, geo: T.Geometry, phi: float = ANCHOR_PHI) -> tuple[int, T.Array]:
    """(column index, voxel point): among the bank-A true columns, the ANCHOR_FAR_FRACTION whose
    roots have the largest local y (farthest along the flat bank from the fundus), the one with
    the smallest |local z|; the point of that column where φ first reaches `phi`."""
    Yr = geo.to_local(truth.roots)
    a = np.flatnonzero(truth.bank == 0)
    if len(a) == 0:
        raise ValueError("the truth has no bank-A column")
    y = Yr[a, 1]
    far = a[y >= np.quantile(y, 1.0 - ANCHOR_FAR_FRACTION)]
    i = int(far[np.argmin(np.abs(Yr[far, 2]))])
    P = truth.path(i)
    ph = geo.phi(geo.to_local(P))
    k = int(np.argmax(ph >= phi))
    return i, P[k]


def nearest_node(grid: T.Grid, X: T.Array) -> int:
    return int(np.argmin(np.sum((np.asarray(grid.nodes) - X[None]) ** 2, axis=1)))


# ---------------------------------------------------------------- the runs


def run_mode(
    mode: str, name: str, *, source: T.InputSource, noise: float, grid_spec: str, out: Path,
    truth_arg: Path, features: Path, site: str = "V1V2_banks", seed: int = 0,
    profile_dtype: Any = np.float64, quiet: bool = False,
) -> T.RunRecord:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of {MODES}")
    if name not in CK.NAMES:
        raise ValueError(f"{name!r} is not a connected kiss")
    feats = FP.features(site, str(features))  # type: ignore[no-untyped-call]
    build = (lambda nz: shifted_phantom(name, feats, site, nz, seed)) if mode == "shiftB" else (
        lambda nz: CK.ckiss(name, feats, site=site, noise=nz, seed=seed))
    phantom = build(noise)
    reference = build(0.0) if (source == "image" and noise != 0.0) else None
    inputs = FR.make_inputs(phantom, source, reference)
    grid = FR.make_grid(grid_spec, phantom.shape)
    tpath = truth_arg if truth_arg.suffix == ".npz" else truth_arg / name / "truth.npz"
    truth = T.Truth.from_npz(tpath)
    settings = T.DEFAULTS
    log_fn = None if quiet else (lambda s: print(s, flush=True))
    out_dir = T.run_dir(out, name, source, noise, f"{FR.grid_label(grid_spec)}_{mode}")
    extra: dict[str, Any] = {}
    kw: dict[str, Any] = {}
    if mode == "coarse":
        cspec = f"uniform:{settings.coarse_first_mm:g}"
        cgrid = FR.make_grid(cspec, phantom.shape)
        t0 = time.perf_counter()
        cfield, _, clog = MF.solve(cgrid, inputs, [float(s) for s in phantom.image["noise_sd"]],
                                   settings, profile_dtype=profile_dtype, log_fn=log_fn)
        extra = dict(grid=FR.grid_record(cgrid, cspec), sweeps=[int(s) for s in clog["sweeps"]],
                     free_energy_last=[fl[-1] if fl else None for fl in clog["free_energy"]],
                     n_rises=len(clog["free_energy_rises"]), time_s=time.perf_counter() - t0,
                     note="CD-9: the coarse field is the initial field of the fine solve, not a prior")
        kw["init"] = cfield
    elif mode == "anchor":
        geo = phantom.geometry
        assert geo is not None
        col, X = anchor_point(truth, geo)
        node = nearest_node(grid, X)
        kw.update(origin="anchor", anchor_node=node)
        extra = dict(column=col, point_vox=X.tolist(), node=node,
                     node_vox=np.asarray(grid.nodes)[node].tolist(), phi=ANCHOR_PHI)
    else:
        extra = dict(shift_bank_B_mm=SHIFT_MM, direction="toward the top")
    versions = {
        "git_commit": FR._git("rev-parse", "HEAD") or "unknown",
        "truth_sha256": FR._sha256(tpath),
        "features_sha256": FR._sha256(features),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    rec = FR.run(phantom, inputs, truth, grid, out_dir, source=source, grid_spec=grid_spec,
                 settings=settings, profile_dtype=profile_dtype, versions=versions,
                 log_fn=log_fn, **kw)
    (out_dir / f"{mode}.json").write_text(json.dumps(T._jsonable(extra), indent=1) + "\n")
    if not quiet:
        print(f"wrote {out_dir} ({rec.time_s:.1f} s, peak {rec.peak_mem_bytes / 1e9:.2f} GB)")
    return rec


def main(argv: Sequence[str] | None = None) -> T.RunRecord:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("phantom", choices=CK.NAMES)
    ap.add_argument("--source", choices=("oracle", "image"), default="oracle")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--grid", default="uniform:0.16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile-dtype", choices=("float32", "float64"), default="float64")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    return run_mode(a.mode, a.phantom, source=a.source, noise=a.noise, grid_spec=a.grid,
                    out=a.out, truth_arg=a.truth, features=a.features, site=a.site, seed=a.seed,
                    profile_dtype=np.dtype(a.profile_dtype).type, quiet=a.quiet)


if __name__ == "__main__":
    main()
