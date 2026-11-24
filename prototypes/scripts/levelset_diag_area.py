"""Diagnostic: the cross-section ratio a on real data (M1S1), from the divergence of the data normal
(carried in the march) against a from the gradient of the root map, and the divergence's own noise.
Usage: levelset_diag_area.py <figures> <work> <data> <site>"""
import sys

sys.path.insert(0, sys.argv[2])
import numpy as np

import levelset_face as LF
import levelset_fill as L
import levelset_mesh as LM

F = LF.Face(LF.A.site)
h = F.h
b = np.full(len(F.mesh.verts), 2.8 / h)
f = F.run(b, L.VOLUME)
m = L.claimed(f) & F.ctx
aR = L.a_from_R(f)
inner = m & (F.dface > 2.0)
ok = inner & np.isfinite(aR)
lr = np.log(f.a[ok]) - np.log(aR[ok])
print(f"claimed {m.sum():,}; capped {f.capped[m].sum():,} ({f.capped[m].mean():.1%})")
print("a (divergence) p5/25/50/75/95", np.percentile(f.a[inner], [5, 25, 50, 75, 95]).round(3))
print("a (grad R)     p5/25/50/75/95", np.nanpercentile(aR[inner], [5, 25, 50, 75, 95]).round(3))
print("log(a_div / a_R) p5/25/50/75/95", np.percentile(lr, [5, 25, 50, 75, 95]).round(3))
D = f.D[inner] / h
print("divergence D (per mm) p5/25/50/75/95", np.percentile(D, [5, 25, 50, 75, 95]).round(2), "IQR", np.subtract(*np.percentile(D, [75, 25])).round(2))
# by depth s
for lo_, hi_ in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)):
    sel = ok & (f.s * h >= lo_) & (f.s * h < hi_)
    print(f"  s {lo_}-{hi_} mm: n {sel.sum():,}; a_div median {np.median(f.a[sel]):.2f} IQR {np.subtract(*np.percentile(f.a[sel],[75,25])):.2f}; "
          f"a_R median {np.nanmedian(aR[sel]):.2f} IQR {np.subtract(*np.nanpercentile(aR[sel],[75,25])):.2f}; capped {f.capped[sel].mean():.1%}; "
          f"D IQR {np.subtract(*np.percentile(f.D[sel] / h, [75, 25])):.2f}/mm; coherence median {np.median(F.coh[sel]):.2f}")
