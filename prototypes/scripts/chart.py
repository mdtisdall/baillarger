"""A local 3D chart of the column volume around a point (plan: reports/Crossing paths and contacts
from laminar profiles.md, *The column volume*). Reads column_volume_<site>.npz; writes NIfTI cubes
for ITK-SNAP. All 3D: the chart only re-indexes samples that keep their 3D positions.

1. The point: ITK-SNAP coordinates (1-based) of the picking image column_volume_<site>_echo2_detrended.nii.gz
   (written on first use; for V1V2_banks it is the same grid as V1V2_banks_echo2_detrended.nii.gz).
   Its tissue (cortex, fluid, WM, missing tissue, other) and the distance to the nearest trace are
   printed and saved; a warning when that trace, or a bank's anchor trace, is over WARN_MM away.
2. Traces (roots >= 2 mm from a face) with any sample within the radius of the point.
3. Banks: the pieces of the roots' footprint on the white mesh (mesh vertices within 1 mm of a root);
   groups of fewer than MIN_TRACES are dropped. One chart per bank, largest first.
4. Chart of a bank: each root at its geodesic distance (along the mesh) from the root of the trace
   nearest the point, in the direction of its projection on that root's tangent plane
   (crossing_lib.exp_chart: radial distances exact, angles distorted where the surface bends round).
5. Cube: u, v on a 0.16 mm grid, depth from the root on a 0.04 mm grid; each (u, v) column is the
   nearest trace within 0.24 mm in the chart (nearest, not interpolated), empty otherwise.
   Outputs per bank: echo 2 and echo 3 (trend removed), an overlay of horizons (one voxel thick at
   each trace's horizon depth) and your points, and a second overlay of each column's trace class.
   The cube's spacing is (0.16, 0.16, 0.04) mm; its origin is the chart's corner, not scanner space.

Usage: chart.py <work dir> <data dir> <site> <i> <j> <k> <radius mm> [--zero-based]
  (from the repository root, scripts dir = prototypes/scripts)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import crossing_lib as X
import mesh_smoothing as MS
from baillarger.data import crop_affine, echo_path, load_box
from baillarger.frame import missing_tissue

ap = argparse.ArgumentParser()
ap.add_argument("work")
ap.add_argument("data")
ap.add_argument("site")
ap.add_argument("i", type=float)
ap.add_argument("j", type=float)
ap.add_argument("k", type=float)
ap.add_argument("radius", type=float)
ap.add_argument(
    "--zero-based",
    action="store_true",
    help="the point is 0-based box voxel coordinates",
)
A = ap.parse_args()

UV_MM, NEAR_MM, JOIN_MM, MIN_TRACES = 0.16, 0.24, 1.0, 20
WARN_MM = 0.3  # the anchor trace further than this from the point: the chart is not centred on it
TISSUE = {0: "cortex", 1: "fluid", 2: "WM", 3: "missing tissue", 4: "other"}
HOR = ["top", "far WM", "fallback contact", "profile, top"]
# overlay labels: horizons 1-4, your points 5
LABELS = [
    (1, (255, 255, 255), "top (open pia)"),
    (2, (48, 213, 255), "far WM (crossing traces)"),
    (3, (255, 43, 214), "contact (where a to-the-contact column ends)"),
    (4, (42, 120, 214), "profile split, top-anchored (subsample)"),
    (5, (27, 175, 122), "your points"),
]
CLASSES = [
    (1, (27, 175, 122), "open"),
    (2, (42, 120, 214), "crossing"),
    (3, (235, 104, 52), "own bank"),
    (4, (199, 154, 0), "to the contact"),
    (5, (227, 73, 72), "failed"),
]


def labels_file(path, rows):
    with open(path, "w") as fh:
        fh.write("# ITK-SNAP label description file\n")
        fh.write('    0     0    0    0        0  0  0    "Clear Label"\n')
        for k, (r, g, b), name in rows:
            fh.write(f'{k:5d} {r:5d} {g:4d} {b:4d}        1  1  1    "{name}"\n')


def tissue_labels(Z):
    """0 cortex, 1 fluid, 2 WM, 3 missing tissue, 4 other: saved by column_volume.py, or rebuilt
    from the box as it does (volumes built before it saved them)."""
    if "labels" in Z.files:
        return Z["labels"]
    lo = tuple(int(a) for a in Z["lo"])
    hi = tuple(int(a) + n for a, n in zip(lo, Z["disp"].shape))
    bp = load_box(lo, hi, name="chart")
    lab = np.full(bp.rim.shape, 4, np.int8)
    lab[bp.rim == 3] = 0
    lab[bp.rim == 1] = 1
    lab[bp.rim == 2] = 2
    lab[missing_tissue(bp.rim, bp.flash, bp.voxel_mm)] = 3
    return lab


def main():
    Z = np.load(f"{A.work}/column_volume_{A.site}.npz")
    v, f, nrm, tv, pos, ns, fv = (
        Z["v"],
        Z["f"],
        Z["nrm"],
        Z["tv"],
        Z["pos"],
        Z["ns"],
        Z["fv"],
    )
    cls, tracer, near_face, vox, lo, P = (
        Z["cls"],
        Z["tracer"],
        Z["near_face"],
        Z["vox"],
        Z["lo"],
        Z["P"],
    )
    H = {k: Z["H_" + k] for k in HOR}
    shape = Z["disp"].shape
    aff = crop_affine(nib.load(echo_path(2)).affine, tuple(int(a) for a in lo))
    pick = Path(A.data) / f"column_volume_{A.site}_echo2_detrended.nii.gz"
    if not pick.exists():
        nib.save(
            nib.Nifti1Image(np.where(Z["ctx"], Z["disp"], 0.0).astype(np.float32), aff),
            pick,
        )
        print(f"wrote the picking image {pick}")
    p = np.array([A.i, A.j, A.k], float) - (0 if A.zero_based else 1)
    assert np.all(p >= 0) and np.all(p < shape), f"point {p} outside the box {shape}"
    tissue = TISSUE[int(tissue_labels(Z)[tuple(np.round(p).astype(int))])]
    print(
        f"{A.site}: point {p} (0-based box voxels), in {tissue}; radius {A.radius} mm"
    )

    # ---- traces within the radius
    sel = []
    pm = p * vox
    for a in range(0, len(tv), 20000):
        q = pos[a : a + 20000].astype(float) * vox
        d = np.linalg.norm(q - pm, axis=2)
        dmin = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1)
        sel.append(a + np.flatnonzero(dmin <= A.radius))
    sel = np.concatenate(sel)
    sel = sel[~near_face[sel]]
    print(f"traces with a sample within {A.radius} mm: {sel.size:,}")
    stem = f"chart_{A.site}_{int(round(A.i))}_{int(round(A.j))}_{int(round(A.k))}_r{A.radius:g}"
    info = dict(
        site=A.site,
        point_box0=p.tolist(),
        point_tissue=tissue,
        radius_mm=A.radius,
        banks=[],
    )
    if sel.size == 0:
        print("no chart: no trace comes within the radius of the point")
        return
    q = pos[sel].astype(float) * vox
    d_all = np.nanmin(
        np.where(np.isfinite(q[..., 0]), np.linalg.norm(q - pm, axis=2), np.inf), 1
    )
    info["nearest_trace_mm"] = round(float(d_all.min()), 3)
    print(f"nearest trace: {d_all.min():.2f} mm from the point")
    if d_all.min() > WARN_MM:
        print(
            f"WARNING: no trace passes within {WARN_MM} mm of the point (point in {tissue}); "
            "the charts are centred on the nearest traces, not on the point"
        )

    # ---- banks along the white mesh
    E = MS.edges(f)
    w = np.linalg.norm((v[E[:, 0]] - v[E[:, 1]]) * vox, axis=1)
    G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v),) * 2).tocsr()
    G = G.maximum(G.T).tocsr()
    roots = tv[sel]
    # the roots' footprint on the mesh: vertices within JOIN_MM of a root; its pieces are the banks
    foot = np.unique(
        np.concatenate(cKDTree(v * vox).query_ball_point(v[roots] * vox, JOIN_MM))
    )
    _, comp_f = connected_components(G[foot][:, foot], directed=False)
    comp = comp_f[np.searchsorted(foot, roots)]
    groups = sorted(
        [np.flatnonzero(comp == c) for c in np.unique(comp)], key=len, reverse=True
    )
    groups = [g for g in groups if g.size >= MIN_TRACES]
    print(f"banks: {[g.size for g in groups]} traces")

    labels_file(Path(A.data) / "chart_labels.txt", LABELS)
    labels_file(Path(A.data) / "chart_class_labels.txt", CLASSES)
    for bi, g in enumerate(groups):
        t = sel[g]
        r = tv[t]
        # anchor: the trace whose samples come nearest the point
        q = pos[t].astype(float) * vox
        dd = np.nanmin(
            np.where(np.isfinite(q[..., 0]), np.linalg.norm(q - pm, axis=2), np.inf), 1
        )
        anc = int(np.argmin(dd))
        anc_mm = float(dd[anc])
        geo = dijkstra(G, indices=r[anc], limit=4 * A.radius + 10)[r]
        ok = np.isfinite(geo)
        t, r, geo = t[ok], r[ok], geo[ok]
        anc = int(np.flatnonzero(tv[t] == tv[sel[g][anc]])[0])
        uv = X.exp_chart(v[r] * vox, v[r[anc]] * vox, nrm[r[anc]], geo)
        # grid
        lo_uv = np.floor(uv.min(0) / UV_MM) * UV_MM
        nu, nv_ = np.ceil((uv.max(0) - lo_uv) / UV_MM).astype(int) + 1
        gu, gv = np.meshgrid(
            lo_uv[0] + UV_MM * np.arange(nu),
            lo_uv[1] + UV_MM * np.arange(nv_),
            indexing="ij",
        )
        dist, idx = cKDTree(uv).query(np.c_[gu.ravel(), gv.ravel()])
        col = np.where(dist <= NEAR_MM, t[idx], -1).reshape(nu, nv_)
        nd = int(ns[t].max())
        e2 = np.zeros((nu, nv_, nd), np.float32)
        e3 = np.zeros((nu, nv_, nd), np.float32)
        ov = np.zeros((nu, nv_, nd), np.uint8)
        cl = np.zeros((nu, nv_, nd), np.uint8)
        for a in range(nu):
            for b in range(nv_):
                tt = col[a, b]
                if tt < 0:
                    continue
                m = ns[tt]
                e2[a, b, :m] = np.nan_to_num(fv[0, tt, :m])
                e3[a, b, :m] = np.nan_to_num(fv[1, tt, :m])
                cl[a, b, :m] = int(cls[tt]) + 1
                for li, hk in enumerate(HOR, start=1):
                    y = H[hk][tt]
                    if np.isfinite(y):
                        ov[a, b, min(int(round(y / X.STEP_MM)), m - 1)] = li
                # your points: their depth on traces passing within 0.3 mm
                qq = pos[tt, :m].astype(float) * vox
                dp = np.linalg.norm(qq[:, None] - (P * vox)[None], axis=2)
                if dp.size and dp.min() <= 0.3:
                    ov[a, b, np.unravel_index(np.argmin(dp), dp.shape)[0]] = (
                        len(HOR) + 1
                    )
        caff = np.diag([UV_MM, UV_MM, X.STEP_MM, 1.0])
        out = Path(A.data)
        nib.save(nib.Nifti1Image(e2, caff), out / f"{stem}_bank{bi + 1}_echo2.nii.gz")
        nib.save(nib.Nifti1Image(e3, caff), out / f"{stem}_bank{bi + 1}_echo3.nii.gz")
        nib.save(nib.Nifti1Image(ov, caff), out / f"{stem}_bank{bi + 1}_overlay.nii.gz")
        nib.save(nib.Nifti1Image(cl, caff), out / f"{stem}_bank{bi + 1}_class.nii.gz")
        filled = col >= 0
        cnt = {name: int(np.sum(cls[t] == c - 1)) for c, _, name in CLASSES}
        ai, aj = np.unravel_index(
            np.argmin((gu - uv[anc, 0]) ** 2 + (gv - uv[anc, 1]) ** 2), gu.shape
        )
        b = dict(
            bank=bi + 1,
            traces=int(t.size),
            classes=cnt,
            primary=int(np.sum(tracer[t] == 0)),
            cube_voxels=[int(nu), int(nv_), nd],
            filled_columns=round(float(filled.mean()), 3),
            anchor_column_itk=[int(ai) + 1, int(aj) + 1],
            anchor_root_box0=v[r[anc]].round(1).tolist(),
            anchor_trace_to_point_mm=round(anc_mm, 3),
            max_geodesic_mm=round(float(geo.max()), 2),
        )
        info["banks"].append(b)
        print(
            f"bank {bi + 1}: {t.size:,} traces ({cnt}); cube {nu} x {nv_} x {nd} "
            f"({filled.mean():.0%} of columns filled); its anchor trace, {anc_mm:.2f} mm from the point, is column "
            f"{ai + 1}, {aj + 1} in ITK-SNAP"
            + (f"  [WARNING: over {WARN_MM} mm]" if anc_mm > WARN_MM else "")
        )
    json.dump(info, open(Path(A.data) / f"{stem}.json", "w"), indent=1)
    print(
        f"wrote {stem}_bank*_{{echo2,echo3,overlay,class}}.nii.gz and {stem}.json in {A.data}; "
        "labels: chart_labels.txt (overlay), chart_class_labels.txt (class)"
    )


if __name__ == "__main__":
    main()
