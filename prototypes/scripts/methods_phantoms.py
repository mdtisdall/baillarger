"""The column methods of baillarger.column_traces (--columns crest, traverse, one-pass) on the fold
phantoms (fold_phantoms.py), at a site's measured features and noise, through the package itself.

Per phantom and method, over the roots at least 2 mm from a box face: the share of the right class
(open or crossing, fold_eval.true_class), the classes, the share of columns that meet T's crest,
and the path, end and contact errors (fold_eval.errors, mm, p50 and p90).

Writes <work>/methods_phantoms_<site>.json. Usage: methods_phantoms.py <work dir> --site SITE
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fold_eval as FE
import fold_phantoms as FP

from baillarger import column_traces as CT


def tissue_of(B):
    """A column_traces.Tissue from a phantom (its patch is not used by the tracing)."""
    return CT.Tissue(patch=None, vox=np.asarray(B["vox"], float), rim=B["rim"], miss=B["miss"],
                     ctx=B["ctx"], wm=B["wm"], det=B["det"], dface=B["dface"], labels=B["labels"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--site", required=True)
    A = ap.parse_args()
    out = {}
    for pname, fold in FP.standard().items():
        B = FP.make(fold, A.site, noise=1.0)
        t = tissue_of(B)
        row = {}
        for m in CT.METHODS:
            R = CT.trace_columns(t, "fitted_blend", m)
            keep = ~R.near_face
            truth = fold.column(R.v[R.tv])
            tc = FE.true_class(truth)
            emax, _, eend, econ = FE.errors(fold, truth, R.pos, R.ns, t.vox)
            k = keep
            row[m] = dict(
                right_class=round(float(np.mean(R.cls[k] == tc[k])), 3),
                classes={CT.CLASS[c]: int(np.sum(k & (R.cls == c))) for c in range(5)},
                meet_crest=round(float(np.mean(R.crest_at[k] >= 0)), 3),
                path_err_p50_p90=np.nanpercentile(emax[k], [50, 90]).round(3).tolist(),
                end_err_p50_p90=np.nanpercentile(eend[k], [50, 90]).round(3).tolist(),
                contact_err_p50_p90=(np.nanpercentile(econ[k], [50, 90]).round(3).tolist()
                                     if np.isfinite(econ[k]).any() else None))
            print(pname, m, json.dumps(row[m]), flush=True)
        out[pname] = row
    json.dump(out, open(f"{A.work}/methods_phantoms_{A.site}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
