"""Where two sheets of a mesh are within the contact gap (0.25 h): how many
places, how large, and what labels lie between. Prints the counts and writes
OUT.json (the largest places) and OUT.npz (the gap vertices in voxels, their
place, facing or not).

usage (from prototypes/scripts): mesh_gap_sites.py DSEG.nii.gz MESH.npz OUT.json   (mesh V in the grid's mm)
"""
import sys, json, numpy as np, nibabel as nib
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
import crop_site as CS
from baillarger import mesh_checks as MC
img = nib.load(sys.argv[1]); lab = np.asarray(img.dataobj).astype(np.int16); vox = np.diag(np.asarray(img.affine)[:3, :3]).copy(); h = float(vox.mean())
z = np.load(sys.argv[2]); V = np.asarray(z["V"], float); F = np.asarray(z["F"], np.int64)
r = MC.check_mesh(V, F, gap=0.25 * h)
print(r.counts())
n, area = CS.vertex_normals_areas(V, F)
g = np.flatnonzero(r.gap); fac = r.gap_facing[g]
def lab_at(P):
    q = np.clip(np.rint(P / vox).astype(int), 0, np.array(lab.shape) - 1); return lab[q[:, 0], q[:, 1], q[:, 2]]
out_l, in_l = lab_at(V[g] + 0.75 * h * n[g]), lab_at(V[g] - 0.75 * h * n[g])
name = lambda a: np.where(a == 41, "WM", np.where(a == 42, "cortex", np.where(a == 0, "outside", "other")))
from collections import Counter
print("gap vertices", len(g), "| between sheets that face each other:", int(fac.sum()), "| label 0.75 h outward:", dict(Counter(name(out_l).tolist())), "| inward:", dict(Counter(name(in_l).tolist())))
print("  facing vertices, label outward:", dict(Counter(name(out_l[fac]).tolist())), "| not facing, label inward:", dict(Counter(name(in_l[~fac]).tolist())))
# places: gap vertices within 3 voxels of each other
P = V[g] / vox
tree = cKDTree(P); pairs = tree.query_pairs(3.0, output_type="ndarray")
parent = np.arange(len(P))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for a, b in pairs:
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
root = np.array([find(i) for i in range(len(P))]); u, inv, cnt = np.unique(root, return_inverse=True, return_counts=True)
ext = np.array([np.ptp(P[inv == k], 0).max() for k in range(len(u))])
print("places:", len(u), "| vertices per place p50/90/max", np.percentile(cnt, [50, 90, 100]).astype(int).tolist(), "| extent (voxels) p50/90/max", np.percentile(ext, [50, 90, 100]).round(1).tolist(),
      "| places with 1-3 vertices:", int((cnt <= 3).sum()), "| with over 50:", int((cnt > 50).sum()), "| share of gap vertices in the 20 largest places: %.0f %%" % (100 * np.sort(cnt)[-20:].sum() / len(g)))
big = np.argsort(-cnt)[:12]
rows = [{"vertices": int(cnt[k]), "centre_vox": P[inv == k].mean(0).round(0).tolist(), "extent_vox": np.ptp(P[inv == k], 0).round(0).tolist(), "facing": int(fac[inv == k].sum())} for k in big]
for x in rows: print("  ", x)
json.dump({"counts": r.counts(), "places": int(len(u)), "largest": rows}, open(sys.argv[3], "w"), indent=1)
np.savez(sys.argv[3].replace(".json", ".npz"), P=P, place=inv, facing=fac)
