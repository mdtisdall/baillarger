"""V1V2_banks: why the posterior length budgets are worse than the prior. Per point and end: the root
vertex, its H, the prior budget, the posterior budget, whether it is in a piece without evidence, the
evidence within 2 mm along the mesh (count, median s), and the geodesic distance to the nearest vertex
with evidence."""
import sys
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], "V1V2_banks"]
sys.path.insert(0, sys.argv[2])
import numpy as np
import levelset_face as LF
import levelset_fill as L
import levelset_mesh as LM

F = LF.Face("V1V2_banks")
h = F.h
P = F.P
Pi = np.round(P).astype(int)
u_pts = F.nh[tuple(Pi.T)]
prior = np.exp(LM.M_MU + LM.A0_MU * F.H)
f_ev = F.run(prior / h, L.LENGTH)
E = LF.evidence(F, f_ev)
fld = LM.MeshField(F.mesh, F.H, alpha=2)
r = LF.budgets(F, fld, E["len"])
b = r["b"]
nobel = LM.pieces_without_evidence(F.mesh, np.isfinite(E["len"]))
has = np.isfinite(E["len"])
rt = F.root_vertex(f_ev.R)
claimed = L.claimed(f_ev) & F.ctx
steps = np.arange(-2.5, 2.5 + 1e-9, 0.05) / h
print("pt end  vtx      H    prior  post  nobel  n2mm  med2mm  d_ev_mm")
for k, Pk in enumerate(P):
    u = u_pts[k] / (np.linalg.norm(u_pts[k]) + 1e-12)
    line = np.round(Pk + steps[:, None] * u).astype(int)
    ok = claimed[tuple(np.clip(line, 0, np.array(F.wm.shape) - 1).T)]
    neg, pos = np.flatnonzero(ok & (steps < 0)), np.flatnonzero(ok & (steps > 0))
    if not (neg.size and pos.size):
        continue
    for end, j in (("-", neg[0]), ("+", pos[-1])):
        v = rt[tuple(line[j])]
        D = LM.geodesic(F.G, np.array([v]), limit=20.0)[0]
        near = (D <= 2.0) & has
        dev = D[has].min() if np.isfinite(D[has]).any() else np.inf
        med = np.exp(np.median(E["len"][near])) if near.any() else np.nan
        print(f"{k+1:2d}  {end}  {v:7d} {F.H[v]:+.2f}  {prior[v]:.2f}  {b[v]:.2f}  {int(nobel[v])}  "
              f"{near.sum():5d}  {med:.2f}  {dev:.2f}")
