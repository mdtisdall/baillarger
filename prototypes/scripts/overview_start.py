"""The hemisphere's overview mesh inside a crop's box, as a start mesh (V in the
crop's mm), with its mesh checks. (As a start it did not work, 2026-11-23:
the no-τ Σ from it was worse than from the coarse start, and ten start
remesh rounds, start_passes=10, did not cure its slivers.)

usage (from prototypes/scripts): overview_start.py CROP OVERVIEW.surf.gii OUT.npz [margin_vox]
"""
import sys, numpy as np, nibabel as nib
import crop_site as CS
from baillarger import mesh_checks as MC
z = CS.site_arrays(sys.argv[1]); A = np.asarray(z["affine"], float); shape = np.array(z["labels"].shape)
vox = np.abs(np.diag(A)[:3]); h = float(vox.mean()); margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
g = nib.load(sys.argv[2]); W = np.asarray(g.darrays[0].data, float); F = np.asarray(g.darrays[1].data, np.int64)
P = (np.linalg.inv(A) @ np.c_[W, np.ones(len(W))].T).T[:, :3]           # the crop's voxel coordinates
ins = np.all((P >= margin) & (P <= shape - 1 - margin), 1)
F = F[ins[F].all(1)]
used = np.unique(F); idx = -np.ones(len(W), np.int64); idx[used] = np.arange(len(used))
V, F = P[used] * vox, idx[F]
e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), 1), axis=0, return_counts=True)
L = np.linalg.norm(V[e[0][:, 0]] - V[e[0][:, 1]], axis=1)
r = MC.check_mesh(V, F, gap=0.25 * h)
print("vertices", len(V), "faces", len(F), "open edges", int((e[1] == 1).sum()), "edge p5/50/95/max (h)", (np.percentile(L, [5, 50, 95, 100]) / h).round(2), "median mm %.4f" % np.median(L), r.counts())
np.savez(sys.argv[3], V=V, F=F)
