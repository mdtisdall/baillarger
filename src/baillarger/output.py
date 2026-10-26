"""The outputs of a run (plan: Outputs), and the command line.

    python -m baillarger.output --patch V1V2 --out OUTDIR [--levels 64 8 2]

In OUTDIR:

- merge_tree.csv: one row per merge (step, the two region ids, the new id,
  cost, columns of the new region; ids 0..n-1 are the model's columns);
- run.json: parameters, the frame and model-column counts, the number of
  connected pieces of the column graph and the levels not reached (merging
  joins only touching regions, so it stops at that number), and per level the
  energy terms after each inner round, K of each region and the seconds;
- level_<N>/ for each re-fit level (N regions):
  - regions.nii.gz: region (1..N) of each voxel of the table (cortex and the
    WM extension), 0 elsewhere;
  - bands.nii.gz: band (1..K_R) of each cortex voxel after the nudge; −1 for
    the WM extension down to the region's bottom; 0 elsewhere;
  - filtered.nii.gz: per echo (4D), the normalized intensity (x − a) / s of
    each cortex voxel shifted by the TV change of its cell (u − x̂);
  - regions.csv: per region: columns, K_R, Z_R, template (per band and
    echo), bottom b_R and why its search stopped, and the cross-validated
    residual of each K;
  - columns.csv: per column: the frame's column id, region, thickness T_c,
    gain and offset per echo, noise per echo, boundaries z.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from baillarger import bands as bd
from baillarger import data as dt
from baillarger import frame as fr
from baillarger.columns import ColumnData, column_data
from baillarger.refine import DEFAULT_LEVELS, Level, Params, Refiner

type F64 = npt.NDArray[np.float64]


def _save(path: Path, img: npt.NDArray[Any], affine: F64) -> None:
    nib.save(nib.nifti1.Nifti1Image(img, affine), path)


def write_level(outdir: Path, D: ColumnData, lv: Level, affine: F64) -> None:
    """The files of one level (see the module notes)."""
    outdir.mkdir(parents=True, exist_ok=True)
    reg, K = lv.reg, lv.K
    nR = len(K)
    _save(
        outdir / "regions.nii.gz",
        D.image(reg[D.col] + 1, fill=0, dtype=np.int16),
        affine,
    )
    cx = D.ctx
    c = D.col[cx]
    bandv = bd.band_of(D.m[cx], lv.z[c])
    band = np.zeros(len(D.m), np.int16)
    band[cx] = bandv + 1
    wm = ~cx
    bottom = np.array([lv.bottoms[R][0] for R in range(nR)])
    band[wm & (D.m >= bottom[reg[D.col]])] = -1
    _save(outdir / "bands.nii.gz", D.image(band, fill=0, dtype=np.int16), affine)
    if lv.u.size:
        a, s = lv.a, lv.s
        xh = (D.f[cx] - a[c]) / np.where(s[c] != 0, s[c], 1.0)
        shift = (lv.u - lv.xhat)[c, bandv]
        filt = np.zeros((len(D.m), D.n_echoes), np.float32)
        filt[cx] = xh + shift
        img = np.stack(
            [D.image(filt[:, e], fill=0, dtype=np.float32) for e in range(D.n_echoes)],
            -1,
        )
        _save(outdir / "filtered.nii.gz", img, affine)
    with open(outdir / "regions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "columns", "K", "Z", "template", "bottom", "why", "cv"])
        for R in range(nR):
            w.writerow(
                [
                    R + 1,
                    int((reg == R).sum()),
                    K[R],
                    " ".join(f"{z:.4f}" for z in lv.Z[R]),
                    ";".join(" ".join(f"{v:.4f}" for v in row) for row in lv.mus[R]),
                    f"{lv.bottoms[R][0]:.4f}",
                    lv.bottoms[R][1],
                    " ".join(f"{v:.6g}" for v in lv.cv[R]),
                ]
            )
    with open(outdir / "columns.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        E = D.n_echoes
        w.writerow(
            ["column", "frame_column", "region", "thickness_mm"]
            + [f"a{e + 1}" for e in range(E)]
            + [f"s{e + 1}" for e in range(E)]
            + [f"sigma{e + 1}" for e in range(E)]
            + ["z"]
        )
        for i in range(D.n):
            zz = lv.z[i][np.isfinite(lv.z[i])]
            w.writerow(
                [i, int(D.labels[i]), int(reg[i]) + 1, f"{D.thickness[i]:.4f}"]
                + [f"{v:.6g}" for v in lv.a[i]]
                + [f"{v:.6g}" for v in lv.s[i]]
                + [f"{np.sqrt(v):.6g}" for v in D.sigma2[i]]
                + [" ".join(f"{v:.4f}" for v in zz)]
            )


def run_patch(
    name: str,
    outdir: Path,
    levels: list[int] | None = None,
    params: Params | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Load one patch, build the frame and the voxel table, run the model and
    write the outputs. Returns the run summary (also in run.json)."""
    t0 = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    p = dt.load_patch(name, directory)
    miss = fr.missing_tissue(p.rim, p.flash, p.voxel_mm)
    F = fr.build_frame(p.rim, p.voxel_mm, miss)
    D, why = column_data(F, p.rim, p.echoes, miss)
    t_inputs = time.perf_counter() - t0
    R = Refiner(D, params)
    seconds: dict[int, float] = {}
    inner = R.refit

    def refit(regions: dict[int, list[int]]) -> Level:
        t = time.perf_counter()
        lv = inner(regions)
        write_level(outdir / f"level_{len(regions):04d}", D, lv, p.affine)
        seconds[len(regions)] = round(time.perf_counter() - t, 1)
        return lv

    R.refit = refit  # type: ignore[method-assign]
    tree, out = R.run(levels)
    with open(outdir / "merge_tree.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "region_1", "region_2", "new_region", "cost", "columns"])
        w.writerows(tree)
    A = sparse.coo_matrix((D.w, (D.edges[:, 0], D.edges[:, 1])), shape=(D.n, D.n))
    n_pieces = int(connected_components(A, directed=False)[0])
    asked = DEFAULT_LEVELS if levels is None else levels
    summary = dict(
        patch=name,
        params=asdict(R.p),
        columns=why,
        graph_pieces=n_pieces,
        levels_not_reached=[L for L in asked if L < D.n and L not in out],
        missing_voxels=int(miss.sum()),
        inputs_seconds=round(t_inputs, 1),
        levels={
            str(L): dict(
                energies=lv.energies,
                K=[lv.K[r] for r in range(len(lv.K))],
                seconds=seconds.get(L),
            )
            for L, lv in out.items()
        },
        seconds=round(time.perf_counter() - t0, 1),
    )
    (outdir / "run.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--patch", required=True, choices=sorted(dt.PATCHES))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    a = ap.parse_args(argv)
    s = run_patch(a.patch, a.out, a.levels)
    print(f"{a.patch}: {s['seconds']} s, levels {list(s['levels'])}")


if __name__ == "__main__":
    main()
