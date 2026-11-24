"""V1V2_banks: posterior length budgets of 5-6 mm at vertices whose evidence within 2 mm is 2.4 mm.
Is the spike local (a few vertices) or a broad patch; what are the parts (z, a*H); what is the mesh
like there (vertex area, angles, valence); what is the evidence at the vertex itself."""
import sys
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], "V1V2_banks"]
sys.path.insert(0, sys.argv[2])
import numpy as np
from scipy import sparse
import levelset_face as LF
import levelset_fill as L
import levelset_mesh as LM

F = LF.Face("V1V2_banks")
h = F.h
prior = np.exp(LM.M_MU + LM.A0_MU * F.H)
f_ev = F.run(prior / h, L.LENGTH)
E = LF.evidence(F, f_ev)
y = E["len"]
fld = LM.MeshField(F.mesh, F.H, alpha=2)
r = fld.fit(y, trend=True)
mu, a = r["mu"], r["a"]
z = mu - r["m"] - F.H * a
G = F.G
# 1-ring median of mu
A = (G > 0).astype(float).tocsr()
nb_mean = (A @ mu) / np.maximum(A.sum(1).A1, 1)
dev = mu - nb_mean
print("mu - 1-ring mean: p50 %.4f p99 %.4f p99.9 %.4f max %.3f" % tuple(np.percentile(np.abs(dev), [50, 99, 99.9]).tolist() + [np.abs(dev).max()]))
b = np.exp(mu)
print("b p50 %.2f p99 %.2f p99.9 %.2f max %.2f; vertices with b > 4.5 mm: %d" % (*np.percentile(b, [50, 99, 99.9]), b.max(), (b > 4.5).sum()))
print("parts at b > 4.5: z p50 %.2f, a*H p50 %.2f, a p50 %.2f, H p50 %.2f" % (np.median(z[b > 4.5]), np.median((a * F.H)[b > 4.5]), np.median(a[b > 4.5]), np.median(F.H[b > 4.5])))
print("parts overall: z p5/50/95", np.percentile(z, [5, 50, 95]).round(3), " a*H", np.percentile(a * F.H, [5, 50, 95]).round(3))
# patch size: connected components of b > 4.5
from scipy.sparse.csgraph import connected_components
m = b > 4.5
idx = np.flatnonzero(m)
sub = G[idx][:, idx]
nc, lab = connected_components(sub, directed=False)
sz = np.bincount(lab)
print(f"patches with b > 4.5: {nc}, sizes p50 {np.median(sz):.0f} max {sz.max()}; largest five {np.sort(sz)[-5:]}")
ar = fld.area
print("vertex area mm^2: all p1/50 %.2e %.2e; in patches p1/50 %.2e %.2e" % (*np.percentile(ar, [1, 50]), *np.percentile(ar[m], [1, 50])))
# evidence in patches
ym = y[m]
print(f"evidence in patches: {np.isfinite(ym).mean():.0%} of vertices, exp median {np.exp(np.nanmedian(ym)):.2f}, p99 {np.exp(np.nanpercentile(ym, 99)):.2f}")
print(f"evidence overall: {np.isfinite(y).mean():.0%}, exp median {np.exp(np.nanmedian(y)):.2f}")
# the largest patch: H, a there, and the extent
big = idx[lab == np.argmax(sz)]
pts = F.mesh.mm[big]
print(f"largest patch: {big.size} vertices, extent {np.ptp(pts, 0).round(2)} mm, H p5/50/95 {np.percentile(F.H[big], [5, 50, 95]).round(2)}, a p50 {np.median(a[big]):.2f}, z p50 {np.median(z[big]):.2f}, y there {np.exp(np.nanmedian(y[big])):.2f} ({np.isfinite(y[big]).mean():.0%} with evidence)")
# a field and w: global
print("a p1/50/99", np.percentile(a, [1, 50, 99]).round(2), " H p1/50/99", np.percentile(F.H, [1, 50, 99]).round(2))
np.savez(f"{sys.argv[2]}/levelset_diag_v1v2_patches.npz", mu=mu, a=a, z=z, y=y, H=F.H)
