"""An inflated white surface: the command line and the figure. The algorithms are in
baillarger.viewer.inflation (inflation with the boundary pinned, sulcal depth, tangent transfer).

Usage: inflate.py <figures> <work> <site>    (writes <work>/inflate_<site>.npz, .json and
       <figures>/inflate_<site>.png)
"""

import json
import sys
import time

import flat_map as FM
import numpy as np

from baillarger.viewer.inflation import (  # noqa: F401  (re-exported for the prototypes)
    LEVELS,
    inflate,
    sulcal_depth,
    transfer,
    umbrella,
    vertex_faces,
    vertex_normals,
)


def main(fig, work, site):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    Z = np.load(f"{work}/column_volume_{site}.npz")
    v = Z["v"].astype(float) * Z["vox"]
    f = Z["f"].astype(np.int64)
    X = inflate(v, f, log=say)
    a0 = FM.face_areas(v, f)
    summ = dict(site=site, levels=[])
    fig_, ax = plt.subplots(1, len(X), figsize=(6 * len(X), 4.5))
    for lv, x, a in zip(LEVELS, X, ax):
        la = np.log2(np.maximum(FM.face_areas(x, f), 1e-30) / np.maximum(a0, 1e-30))
        sd = sulcal_depth(v, x, f)
        q = FM.wpct(la, a0, [1, 5, 50, 95, 99])
        summ["levels"].append(
            dict(
                lam=lv,
                log2_area_p1_p5_p50_p95_p99=[round(t, 2) for t in q],
                sulcal_depth_mm_p1_p50_p99=[
                    round(float(t), 2) for t in np.percentile(sd, [1, 50, 99])
                ],
            )
        )
        a.hist(la, bins=120, range=(-6, 6), weights=a0)
        a.set_title(
            f"lam {lv:g}: log2 area factor (inflated / true), area-weighted", fontsize=9
        )
    fig_.tight_layout()
    fig_.savefig(f"{fig}/inflate_{site}.png", dpi=70)
    plt.close(fig_)
    np.savez_compressed(
        f"{work}/inflate_{site}.npz",
        levels=np.array(LEVELS),
        x=np.stack(X).astype(np.float32),
    )
    summ["seconds"] = round(time.time() - t0, 1)
    json.dump(summ, open(f"{work}/inflate_{site}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
