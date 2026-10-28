import numpy as np
from baillarger.fronts import march, riemann_from_direction
L = 56.0  # box side in native voxels (160 um each)
for R in (3.0, 6.0, 12.0):
    for ratio in (1.0, 3.0, 10.0):
        for h in (1.0, 0.5):
            n = int(L / h) + 1
            ax = np.arange(n) * h
            g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
            v = g - L / 2
            r = np.linalg.norm(v, axis=-1)
            m = riemann_from_direction(v, ratio)
            ball = r <= R
            f = march(m, np.where(ball, 0, -1).astype(np.int64), spacing=(h, h, h))
            ex = r - R  # radial paths: arrival and length both equal r - R
            keep = (ex > 12.0) & (ex < 20.0)
            ea = np.abs(f.arrival[keep] - ex[keep])
            el = np.abs(f.length[keep] - ex[keep])
            sa = np.median(f.arrival[keep] - ex[keep]); sl = np.median(f.length[keep] - ex[keep])
            print(f"R={R:4.0f} ratio={ratio:4.0f} h={h}: abs err in native voxels | arrival med {np.median(ea):.2f} p95 {np.percentile(ea,95):.2f} bias {sa:+.2f} "
                  f"| length med {np.median(el):.2f} p95 {np.percentile(el,95):.2f} bias {sl:+.2f}", flush=True)
