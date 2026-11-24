"""The mesh checks of each crop's white mesh, before any start is made from it:
check counts, dihedral turns, the share of the white area facing outside
medium, cortex and WM, the label counts, and where the gap and the faces
toward outside medium are.

usage (from prototypes/scripts): crop_white_checks.py CROP ...
"""
import sys, numpy as np
import crop_site as CS
from baillarger import mesh_checks as MC
for p in sys.argv[1:]:
    site = CS.load_site(p); h = float(site.vox.mean()); V = site.v * site.vox; F = np.asarray(site.f, np.int64)
    r = MC.check_mesh(V, F, gap=0.25 * h); di, _ = MC.edge_dihedrals(V, F)
    q = np.rint(site.v + site.normal / np.linalg.norm(site.normal, axis=1)[:, None] / site.vox * h).astype(int)
    q = np.clip(q, 0, np.array(site.labels.shape) - 1); out = site.labels[q[:, 0], q[:, 1], q[:, 2]]
    a = site.varea
    print(p.split("/")[-1], len(V), r.counts(), "dihedral>90:", int((di > 90).sum()), "max", round(float(np.nanmax(di)), 1),
          "area share facing outside medium: %.3f, cortex: %.3f, WM: %.3f" % tuple(a[out == u].sum() / a.sum() for u in (0, 42, 41)))
    print("  labels", {int(u): int((site.labels == u).sum()) for u in np.unique(site.labels)})
    for nm, m in (("gap", r.gap), ("facing outside", out == 0)):
        g = site.v[m]
        if len(g): print("  %s vertices: crop-voxel range" % nm, g.min(0).round(0), g.max(0).round(0))
