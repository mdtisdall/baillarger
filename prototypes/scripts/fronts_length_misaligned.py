import sys, numpy as np
from scipy.spatial import cKDTree
from baillarger.fronts import march, riemann_from_direction
L = 56.0  # box side in native voxels (160 um each)
for R in (3.0, 6.0, 12.0):
    for ratio in (1.0, 3.0, 10.0):
        m = riemann_from_direction(np.array([1.0, 2.0, 0.5]), ratio)
        chol = np.linalg.cholesky(m)
        for h in (1.0, 0.5):
            n = int(L / h) + 1
            ax = np.arange(n) * h
            g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
            r = np.linalg.norm(g - L / 2, axis=-1)
            ball = r <= R
            f = march(m, np.where(ball, 0, -1).astype(np.int64), spacing=(h, h, h))
            d, idx = cKDTree(g[ball] @ chol).query(g.reshape(-1, 3) @ chol)
            d = d.reshape(r.shape)
            ex = np.linalg.norm(g.reshape(-1, 3) - g[ball][idx], axis=-1).reshape(r.shape)
            keep = (ex > 12.0) & (ex < 20.0)
            ea = np.abs(f.arrival[keep] - d[keep]) / d[keep]
            el = np.abs(f.length[keep] - ex[keep])
            print(f"R={R:4.0f} ratio={ratio:4.0f} h={h}: arrival rel med {np.median(ea):.3f} p95 {np.percentile(ea,95):.3f} | "
                  f"length abs err (voxels@160um) med {np.median(el):.2f} p95 {np.percentile(el,95):.2f}", flush=True)
