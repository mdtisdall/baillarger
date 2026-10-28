"""Is there radial (tangential-gradient) texture in thick cortex at 160 um?

Model-free check, independent of the pipeline. For each patch:
- interior cortex: rim 3, not missing tissue, >= 3 voxels from WM and fluid,
  >= 6 voxels from the crop faces, open cortex only (thickness proxy < 5 mm);
- thickness proxy t = d_wm + d_fluid (EDT, mm), depth fraction r = d_wm / t;
- cross-echo structure tensor (sym. mean of grad(log e_a) grad(log e_b)^T over
  the 3 echo pairs; thermal noise is independent between echoes, so it mostly
  cancels), derivative scale sd, integration scale 2 voxels;
- eigenvalues l1 >= l2 >= l3: planar (layers) -> l2/l1 small; linear (resolved
  radial bundles) -> l2/l1 near 1. Also the angle between e1 and the
  geometric normal grad(d_wm - d_fluid).
"""

import sys

import numpy as np
from scipy import ndimage as ndi

from baillarger.data import load_patch
from baillarger.frame import missing_tissue

SI = 2.0


def run(name: str) -> None:
    p = load_patch(name)
    rim = p.rim
    vox = p.voxel_mm
    miss = missing_tissue(rim, p.flash, vox)
    ctx = (rim == 3) & ~miss
    d_wm = ndi.distance_transform_edt(rim != 2, sampling=vox)
    d_fl = ndi.distance_transform_edt(rim != 1, sampling=vox)
    t = d_wm + d_fl
    r = d_wm / np.maximum(t, 1e-6)
    v3 = 3 * min(vox)
    face = np.zeros(rim.shape, bool)
    face[6:-6, 6:-6, 6:-6] = True
    interior = ctx & (d_wm >= v3) & (d_fl >= v3) & face & (t < 5.0)
    phi = ndi.gaussian_filter(d_wm - d_fl, 2.0)
    gphi = np.stack(np.gradient(phi), -1)
    nrm = gphi / (np.linalg.norm(gphi, axis=-1, keepdims=True) + 1e-12)

    logs = [np.log(np.maximum(p.echoes[..., e].astype(np.float64), 1.0)) for e in range(3)]
    idx = np.nonzero(interior)
    print(f"\n=== {name}: interior voxels {idx[0].size:,} "
          f"(cortex {ctx.sum():,}); t median {np.median(t[interior]):.2f} mm")

    tb = [(0, 2.5), (2.5, 3.2), (3.2, 5.0)]
    rb = [(0.15, 0.4), (0.4, 0.65), (0.65, 0.9)]
    for sd in (0.7, 1.0, 1.5):
        g = [np.stack([ndi.gaussian_filter(L, sd, order=o) for o in
                       ((1, 0, 0), (0, 1, 0), (0, 0, 1))], -1) for L in logs]
        comps = {}
        for i in range(3):
            for j in range(i, 3):
                acc = np.zeros(rim.shape)
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    acc += 0.5 * (g[a][..., i] * g[b][..., j] + g[b][..., i] * g[a][..., j])
                comps[i, j] = ndi.gaussian_filter(acc / 3.0, SI)[idx]
        n_pts = idx[0].size
        S = np.empty((n_pts, 3, 3))
        for (i, j), c in comps.items():
            S[:, i, j] = c
            S[:, j, i] = c
        w, V = np.linalg.eigh(S)
        l1, l2 = w[:, 2], w[:, 1]
        ok = l1 > 0
        lin = np.where(ok, np.clip(l2, 0, None) / np.where(ok, l1, 1), np.nan)
        e1 = V[:, :, 2]
        cosang = np.abs(np.einsum("ij,ij->i", e1, nrm[idx]))
        ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
        tt, rr = t[idx], r[idx]
        print(f"-- derivative scale {sd} vox: l2/l1 median [angle e1 vs normal, median deg] (n)")
        print("   t (mm) \\ r      " + "  ".join(f"{a:.2f}-{b:.2f}".ljust(20) for a, b in rb))
        for ta, tb_ in tb:
            row = []
            for ra, rb_ in rb:
                m = (tt >= ta) & (tt < tb_) & (rr >= ra) & (rr < rb_) & ok
                if m.sum() < 200:
                    row.append("-".ljust(20))
                    continue
                row.append(f"{np.nanmedian(lin[m]):.2f} [{np.median(ang[m]):4.1f}] ({m.sum()//1000}k)".ljust(20))
            print(f"   {ta:.1f}-{tb_:.1f}       " + "  ".join(row))
        del g, comps, S, V


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["M1S1", "V1V2"]:
        run(nm)
