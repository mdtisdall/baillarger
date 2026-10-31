"""V1V2_banks: the long tops (s > 4 mm) behind the high-budget patches. For each used top: s, the
straight-line distance from its root R to the nearest fluid voxel, and to the nearest box face / missing
tissue; whether its root vertex is in a patch with posterior b > 4.5 mm."""

import sys

sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], "V1V2_banks"]
sys.path.insert(0, sys.argv[2])
import numpy as np
from scipy import ndimage as ndi
import levelset_face as LF
import levelset_fill as L
import levelset_mesh as LM

F = LF.Face("V1V2_banks")
h = F.h
prior = np.exp(LM.M_MU + LM.A0_MU * F.H)
f_ev = F.run(prior / h, L.LENGTH)
E = LF.evidence(F, f_ev)
tp, ok = E["tops"], E["ok"]
D = np.load(f"{sys.argv[2]}/levelset_diag_v1v2_patches.npz")
b = np.exp(D["mu"])
R = tp.R[ok]
s = tp.s[ok] * h
vtx = F.root_vertex(R)
dfl = ndi.distance_transform_edt(~F.fluid) * h
Rr = tuple(np.clip(np.round(R).astype(int), 0, np.array(F.wm.shape) - 1).T)
dR = dfl[Rr]
top = tp.x[ok] if hasattr(tp, "x") else None
inp = b[vtx] > 4.5
for lab, m in (
    ("all used tops", np.ones_like(inp)),
    ("roots in patches", inp),
    ("s > 4 mm", s > 4),
    ("s > 4 mm, roots in patches", (s > 4) & inp),
):
    print(
        f"{lab:28s} n {m.sum():6d}  s p50 {np.median(s[m]):.2f}  straight to fluid from R p50 {np.median(dR[m]):.2f}  ratio p50 {np.median(s[m] / dR[m]):.2f}"
    )
print("fraction of used tops with s > 4 mm:", f"{(s > 4).mean():.1%}")
print("fields of tp:", [k for k in dir(tp) if not k.startswith("_")])
# where the long tops sit: the root's coordinates (box voxels) of the patches' long tops
q = np.flatnonzero((s > 4) & inp)
print(
    "long-top roots in patches, voxel coords p5/50/95:",
    np.percentile(R[q], [5, 50, 95], axis=0).round(0).tolist(),
)
