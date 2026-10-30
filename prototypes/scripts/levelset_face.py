"""Level-set filling at a marked face (test plan §5 item 8), against findings 13, 18, 19 and 25.

All 3D, at 1x grid. For one site (M1S1 | V1V2_banks | PCC_SPL_p1 | PCC_SPL_p2):
1. Box and points as onepass_face.py (every point >= 6 mm from a face).
2. Data normal and coherence: curv_lib.tensor_normal (the plan's data term).
3. White surface: finding 28's (mesh_smoothing.py): the binary contour of the WM label (marching
   tetrahedra, faces outward), relaxed tangentially, smoothed along the mesh by twicing at 0.32 mm
   (2 voxels); vertex normals smoothed as white_mesh's. SURFACE=white_mesh uses frame.white_mesh at its
   default (sigma = 1) instead. H per vertex from frame.mesh_curvatures at 0.8 mm, clipped to +-0.8 /mm
   as collide.py. Start band: WM voxels within BAND_WM = 2 voxels of the surface and next to cortex, and
   cortex voxels less than a voxel from it, at their signed distance (the sign from the pseudonormal of
   the closest feature, finding 28); R the foot point, d the smoothed vertex normal there.
4. Evidence run (prior budgets), tops (levelset_fill.tops; not within 2 mm of a face or next to missing
   tissue; roots >= 2 mm from a face; 0.8-7 mm as onepass_face), mapped to the root's nearest vertex: the
   median log s (length) and log w (volume) per vertex, in mm.
5. Field on the mesh (levelset_mesh.MeshField: cotangent FEM, second order, level + curvature trend as
   collide.py); budgets exp(mean) (the median); vertices in pieces without evidence get the median.
6. Passes: budgets equal (the evidence's median), prior only (exp(m0 + a0 H) of the prior), posterior,
   each in the length and the volume form. Variants (environment): ALPHA=1 (first-order field),
   TOPO=1 (topology test), STOP=1 (stop at tau = 1), ITER=1 (one convergence iteration).
7. Score at each point: the line along the data normal through it, 2.5 mm either way; the two ends'
   roots (deepest claimed cortex on each side) are the two fronts; each seam voxel goes to the end whose
   root is nearer along the white mesh (Dijkstra on its edges); a crossing is a seam pair split between
   the ends, within 1 mm of the point and 0.4 mm of the line. Offset (mm) along the line, + toward the
   side whose posterior length budget at its end root is smaller (fixed per point for every run). Per
   side: mu - S (length) and v - w (volume) at the point. Also: the nearest crossing in any direction,
   the two roots' distance along the mesh, coherence and the blend weight at the point.
8. Calibration (CALIB=1, at one face): 5 spatial folds of evidence vertices (slabs along their principal
   axis, as collide.py), held-out z SD and residual IQR in log, for the mesh field and for collide's
   voxel-graph field on the same evidence.

Usage: levelset_face.py <figures dir> <work dir> <data dir> <site>
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import curv_lib as C
import levelset_fill as L
import levelset_mesh as LM
from baillarger.frame import mesh_curvatures, white_mesh

ap = argparse.ArgumentParser()
ap.add_argument("fig")
ap.add_argument("work")
ap.add_argument("data")
ap.add_argument("site")
A = ap.parse_args()
FULL = np.array([700, 384, 1148])
HALF = 70
FACE_MM = 2.0
ALPHA = int(os.environ.get("ALPHA", "2"))
TOPO = os.environ.get("TOPO", "0") == "1"
STOP = os.environ.get("STOP", "0") == "1"
ITER = os.environ.get("ITER", "0") == "1"
CALIB = os.environ.get("CALIB", "0") == "1"
NSAMP_CAL = int(
    os.environ.get("NSAMP_CAL", "20")
)  # perturb-and-MAP samples per fold (each a CG solve)
FORMS = os.environ.get("FORMS", "length,volume").split(
    ","
)  # which forms' passes to run
SURFACE = os.environ.get("SURFACE", "twice")
BAND_WM = 2.0
TAG = os.environ.get("TAG", "") + ("" if SURFACE == "twice" else "_white_mesh")
H_CLIP = 0.8
BLUE, ORANGE, AQUA, RED, GREY, VIOLET = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#e34948",
    "#8a8984",
    "#8a5cd1",
)


def box_and_points(site):
    """As onepass_face.box_and_points."""
    if site == "M1S1":
        m = json.load(open(f"{A.data}/M1S1_test_points.json"))
        full = np.array(m["gap_all"]) + np.array(m["full_lo"])
        c = np.round(full.mean(0)).astype(int)
        lo, hi = c - HALF, c + HALF + 1
        return lo, hi, (full - lo).astype(float)
    name, piece, centre = {
        "V1V2_banks": ("V1V2_banks_points.json", None, False),
        "PCC_SPL_p1": ("PCC_SPL_points.json", "1", True),
        "PCC_SPL_p2": ("PCC_SPL_p2_points.json", None, True),
    }[site]
    pf = json.load(open(f"{A.data}/{name}"))
    crop = pf["crop"].replace("_echo2_detrended.nii.gz", "")
    box = json.load(open(f"{A.data}/{crop}_sites.json"))
    pts = pf["pieces"][piece]["contact_points"] if piece else pf["contact_points"]
    full = np.unique(np.array(pts, int), axis=0) - 1 + np.array(box["full_lo"])
    if centre:
        c = np.round(full.mean(0)).astype(int)
        lo, hi = c - HALF, c + HALF + 1
    else:
        lo, hi = np.array(box["full_lo"]), np.array(box["full_hi"])
    assert np.all(lo >= 0) and np.all(hi <= FULL)
    return lo, hi, (full - lo).astype(float)


def pseudo_sign(v, f, X, P, F, fn, vn, en):
    """+1 outside the WM, -1 inside: (X - P) . the pseudonormal of the feature holding P (the face, an
    edge or a vertex; Baerentzen and Aanaes), as mesh_smoothing.closest_on_mesh."""
    a, b, c = v[f[F, 0]], v[f[F, 1]], v[f[F, 2]]
    # barycentric of P in its face
    v0, v1, v2 = b - a, c - a, P - a
    d00, d01, d11 = np.sum(v0 * v0, 1), np.sum(v0 * v1, 1), np.sum(v1 * v1, 1)
    d20, d21 = np.sum(v2 * v0, 1), np.sum(v2 * v1, 1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    bary = np.stack([1 - bv - bw, bv, bw], 1)
    nz = bary > 1e-9
    cnt = nz.sum(1)
    pn = fn[F].copy()
    e2 = cnt == 2
    if e2.any():
        zero = np.argmin(nz[e2], 1)
        pn[e2] = en[F[e2], (zero + 1) % 3]
    e1 = cnt == 1
    if e1.any():
        corner = np.argmax(nz[e1], 1)
        pn[e1] = vn[f[F[e1], corner]]
    return np.where(np.sum((X - P) * pn, 1) >= 0, 1.0, -1.0)


def log(msg):
    print(msg, flush=True)


class Face:
    """Everything fixed for a site: box, normals, mesh, band."""

    def __init__(self, site):
        t0 = time.time()
        self.lo, self.hi, self.P = box_and_points(site)
        B = C.load(self.lo, self.hi, f"levelset_{site}")
        self.B = B
        self.vox = B["vox"]
        self.h = float(np.mean(self.vox))
        self.wm, self.ctx = B["wm"], B["ctx"]
        self.miss = B["miss"]
        self.fluid = (B["rim"] == 1) & ~self.miss
        self.dface = B["dface"]
        margin = np.min(np.minimum(self.P, (self.hi - self.lo) - 1 - self.P))
        assert margin * self.h >= 6.0, (
            f"a point is {margin * self.h:.1f} mm from a face"
        )
        self.nh, self.coh = C.tensor_normal(B)
        log(
            f"{site}: box {tuple(self.hi - self.lo)}; {len(self.P)} points (nearest {margin * self.h:.1f} mm "
            f"from a face); cortex {self.ctx.sum():,}; loaded and tensor in {time.time() - t0:.0f} s"
        )
        t0 = time.time()
        import mesh_smoothing as MS

        if SURFACE == "twice":
            v0, f = MS.contour(self.wm, 0.0)
            v = MS.twice(MS.relax(v0, f), f, 2.0)
            n = MS.smooth_normals(v, f)
        else:
            v, f, n = white_mesh(self.wm, tuple(self.vox))
            f = f[:, [0, 2, 1]].copy()  # white_mesh's faces point into the WM
        self.mesh = LM.Mesh(v, f, n, self.vox)
        k1, k2 = mesh_curvatures(self.mesh.mm, f, n, 0.8)
        self.H = np.clip(0.5 * (k1 + k2), -H_CLIP, H_CLIP)
        self.G = LM.edge_graph(self.mesh)
        self.vtree = cKDTree(v)
        el = self.G.data
        log(
            f"   white mesh: {len(v):,} vertices, {len(f):,} faces, edge median {np.median(el):.3f} mm; "
            f"H p5/50/95 {np.percentile(self.H, [5, 50, 95]).round(2)} ({time.time() - t0:.0f} s)"
        )
        t0 = time.time()
        st = np.ones((3, 3, 3), bool)
        near_ctx = ndi.binary_dilation(self.ctx, st, iterations=int(np.ceil(BAND_WM)))
        near_wm = ndi.binary_dilation(self.wm, st, iterations=2)
        cand = (self.wm & near_ctx) | (self.ctx & near_wm)
        ci = np.argwhere(cand)
        phi, foot, nrm, tri, bary = LM.closest(self.mesh, ci.astype(float))
        # the sign from the pseudonormal of the closest feature (finding 28); the foot and triangle agree
        Pm, Fm, _ = MS.closest_on_mesh(v, f, ci.astype(float))
        fn, vn, en = MS.pseudonormals(v, f)
        same = np.linalg.norm(Pm - foot, axis=1) < 1e-6
        self.sign_disagree = float(np.mean(~same))
        sgn = pseudo_sign(v, f, ci.astype(float), Pm, Fm, fn, vn, en)
        phi = np.abs(phi) * sgn
        self.phi = np.zeros(self.wm.shape)
        self.foot = np.zeros(self.wm.shape + (3,))
        self.normal = np.zeros(self.wm.shape + (3,))
        self.phi[tuple(ci.T)] = phi
        self.foot[tuple(ci.T)] = foot
        self.normal[tuple(ci.T)] = nrm
        self.band = (self.wm & near_ctx & (self.phi >= -BAND_WM)) | (
            self.ctx & near_wm & (self.phi < 1.0)
        )
        self.band_tri = np.full(self.wm.shape, -1, np.int64)
        self.band_tri[tuple(ci.T)] = tri
        self.band_bary = np.zeros(self.wm.shape + (3,))
        self.band_bary[tuple(ci.T)] = bary
        bp = self.phi[self.band]
        wmb = self.phi[self.band & self.wm]
        log(
            f"   band {self.band.sum():,} voxels; phi p1/50/99 {np.percentile(bp, [1, 50, 99]).round(2)} voxels; "
            f"WM band voxels outside the mesh (phi > 0) {np.mean(wmb > 0):.1%}; cortex band voxels inside "
            f"{np.mean(self.phi[self.band & self.ctx] < 0):.1%}; closest points differing between the two "
            f"searches {self.sign_disagree:.2%} ({time.time() - t0:.0f} s)"
        )
        self.face_near = self.dface < FACE_MM
        self.miss_near = ndi.binary_dilation(self.miss, st)

    def band_budget(self, vb):
        """Budget per band voxel: barycentric interpolation of log budget at the foot's triangle."""
        out = np.full(self.wm.shape, np.nan)
        m = self.band
        lb = np.log(vb)[self.mesh.faces[self.band_tri[m]]]
        out[m] = np.exp(np.sum(lb * self.band_bary[m], 1))
        return out

    def run(self, vb, mode, tau_stop=np.inf, topology=False):
        t0 = time.time()
        f = L.march(
            self.ctx,
            self.wm,
            self.band,
            self.phi,
            self.foot,
            self.normal,
            self.band_budget(vb),
            self.nh,
            self.coh,
            mode=mode,
            tau_stop=tau_stop,
            topology=topology,
        )
        f.elapsed = time.time() - t0
        return f

    def root_vertex(self, R):
        ok = np.all(np.isfinite(R), -1)
        out = np.full(R.shape[:-1], -1, np.int64)
        out[ok] = self.vtree.query(R[ok])[1]
        return out


def evidence(F, f):
    """Per vertex, median log s and log w (mm) at the visible tops whose roots map to it."""
    excl = F.face_near | F.miss_near
    tp = L.tops(f, F.fluid, exclude=excl)
    Rr = np.clip(np.round(tp.R).astype(int), 0, np.array(F.wm.shape) - 1)
    ok = F.dface[tuple(Rr.T)] >= FACE_MM
    s_mm, w_mm = tp.s * F.h, tp.w * F.h
    ok &= (s_mm > 0.8) & (s_mm < 7.0) & np.isfinite(w_mm) & (w_mm > 0)
    vtx = F.root_vertex(tp.R[ok])
    nv = len(F.mesh.verts)
    out = {}
    for key, val in (("len", np.log(s_mm[ok])), ("vol", np.log(w_mm[ok]))):
        y = np.full(nv, np.nan)
        order = np.argsort(vtx)
        vs, vv = vtx[order], val[order]
        st = np.flatnonzero(np.r_[True, np.diff(vs) != 0])
        en = np.r_[st[1:], vs.size]
        for a, b in zip(st, en):
            y[vs[a]] = np.median(vv[a:b])
        out[key] = y
    out["ntops"] = int(tp.idx.size)
    out["ntops_used"] = int(ok.sum())
    out["tops"] = tp
    out["ok"] = ok
    return out


def budgets(F, fld, y, trend=True):
    """Posterior median budget per vertex (mm), with pieces without evidence at the median."""
    has = np.isfinite(y)
    r = fld.fit(y, trend=trend)
    b = np.exp(r["mu"])
    nobel = LM.pieces_without_evidence(F.mesh, has)
    b[nobel] = np.median(b[~nobel])
    r["nobel"] = int(nobel.sum())
    r["b"] = b
    return r


# ------------------------------------------------------------------ scoring


def score(F, f, sm, P, u_pts, thin_sign, h):
    """Per point: offset (mm) of the crossing between the two ends' fronts, + toward thin_sign's side."""
    rows = []
    steps = np.arange(-2.5, 2.5 + 1e-9, 0.05) / h
    shape = np.array(F.wm.shape)
    claimed = L.claimed(f) & F.ctx
    vp_ = F.root_vertex(sm.R_a)
    vq_ = F.root_vertex(sm.R_b)
    rt = F.root_vertex(f.R)
    for k, Pk in enumerate(P):
        u = u_pts[k] / (np.linalg.norm(u_pts[k]) + 1e-12)
        line = np.round(Pk + steps[:, None] * u).astype(int)
        inside = np.all((line >= 0) & (line < shape), 1)
        ok = np.zeros(steps.size, bool)
        ok[inside] = claimed[tuple(line[inside].T)]
        neg = np.flatnonzero(ok & (steps < 0))
        pos = np.flatnonzero(ok & (steps > 0))
        row = dict(k=k + 1, offset=np.nan, n=0, nearest_mm=np.nan, root_sep_mm=np.nan)
        if neg.size == 0 or pos.size == 0:
            rows.append(row)
            continue
        vm, vp = rt[tuple(line[neg[0]])], rt[tuple(line[pos[-1]])]
        D = LM.geodesic(F.G, np.array([vm, vp]), limit=80.0)
        row["root_sep_mm"] = float(D[0, vp])
        side_a = np.where(D[0][vp_] < D[1][vp_], -1, 1)
        side_b = np.where(D[0][vq_] < D[1][vq_], -1, 1)
        cross = side_a != side_b
        dd = sm.pos - Pk
        along = dd @ u
        perp = np.linalg.norm(dd - along[:, None] * u, axis=1)
        if cross.any():
            row["nearest_mm"] = float(np.min(np.linalg.norm(dd[cross], axis=1)) * h)
        near = cross & (np.linalg.norm(dd, axis=1) * h < 1.0) & (perp * h < 0.4)
        if not near.any():
            rows.append(row)
            continue
        i = np.flatnonzero(near)
        sg = thin_sign[k]  # -1: the thin side is the end at -u
        o = along[i] * sg * h
        # per side at the point: the thin side is the one whose end is at sg * u
        a_thin = side_a[i] == sg
        s_thin = np.where(a_thin, sm.s_a[i], sm.s_b[i]) * h
        s_thick = np.where(a_thin, sm.s_b[i], sm.s_a[i]) * h
        w_thin = np.where(a_thin, sm.w_a[i], sm.w_b[i]) * h
        w_thick = np.where(a_thin, sm.w_b[i], sm.w_a[i]) * h
        t_thin = np.where(a_thin, sm.tau_a[i], sm.tau_b[i])
        t_thick = np.where(a_thin, sm.tau_b[i], sm.tau_a[i])
        b_thin = np.where(a_thin, sm.b_a[i], sm.b_b[i]) * h
        b_thick = np.where(a_thin, sm.b_b[i], sm.b_a[i]) * h
        row.update(
            offset=float(np.median(o)),
            n=int(i.size),
            S_thin=float(np.median(s_thin + o)),
            S_thick=float(np.median(s_thick - o)),
            W_thin=float(np.median(w_thin + o)),
            W_thick=float(np.median(w_thick - o)),
            b_thin=float(np.median(b_thin)),
            b_thick=float(np.median(b_thick)),
            tau_thin=float(np.median(t_thin)),
            tau_thick=float(np.median(t_thick)),
            angle=float(np.median(sm.angle[i])),
        )
        rows.append(row)
    return rows


def summary(rows):
    off = np.array([r["offset"] for r in rows])
    if not np.isfinite(off).any():
        return dict(
            median=np.nan,
            iqr=np.nan,
            median_abs=np.nan,
            within_016=0.0,
            within_016_of_all=0.0,
            worst=np.nan,
            worst_point=0,
            missing=len(rows),
            nearest_median=np.nan,
            nearest_p90=np.nan,
        )
    near = np.array([r["nearest_mm"] for r in rows])
    a = np.abs(off)
    return dict(
        median=float(np.nanmedian(off)),
        iqr=float(np.subtract(*np.nanpercentile(off, [75, 25]))),
        median_abs=float(np.nanmedian(a)),
        within_016=float(np.nanmean(a <= 0.16)) if np.isfinite(a).any() else 0.0,
        within_016_of_all=float(np.sum(a <= 0.16) / len(rows)),
        worst=float(np.nanmax(a)),
        worst_point=int(np.nanargmax(a)) + 1,
        missing=int(np.isnan(off).sum()),
        nearest_median=float(np.nanmedian(near)),
        nearest_p90=float(np.nanpercentile(near, 90)),
    )


def seam_stats(f, sm, F):
    """Coverage, seam classes by tau, and the volume claimed beyond tau = 1."""
    cov = L.coverage(f, F.wm, F.fluid, face=F.face_near, missing=F.miss)
    cl = L.seam_class(sm)
    m = L.claimed(f) & F.ctx
    inner = ~F.face_near.ravel()[sm.p]
    return dict(
        coverage=cov,
        seams=int(sm.p.size),
        seams_inner=int(inner.sum()),
        classes_inner=np.bincount(cl[inner], minlength=4).tolist(),
        beyond_tau1=float(np.mean(f.tau[m] > 1.0)),
        capped=int(f.capped[m].sum()),
        capped_beyond_3mm=float(np.mean(f.s[m & f.capped] * F.h > 3.0))
        if f.capped[m].any()
        else 0.0,
        moved=int(f.moved[m].sum()),
        blend_median=float(np.median(f.blend[m])),
        elapsed=float(f.elapsed),
    )


# ------------------------------------------------------------------ calibration


def calibration(F, fld, y, label):
    ev = np.flatnonzero(np.isfinite(y))
    c = F.mesh.mm[ev]
    ax_ = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][0]
    folds = np.array_split(ev[np.argsort((c - c.mean(0)) @ ax_)], 5)
    zs, err = [], []
    for fo in folds:
        yy = y.copy()
        yy[fo] = np.nan
        r = fld.fit(yy, trend=True, nsamp=NSAMP_CAL, at=fo, rtol=1e-6)
        zs.append((y[fo] - r["mu"][fo]) / r["sd_at"])
        err.append(y[fo] - r["mu"][fo])
    z, e = np.concatenate(zs), np.concatenate(err)
    full = fld.fit(y, trend=True)
    res = y[ev] - full["mu"][ev]
    out = dict(
        label=label,
        n=int(ev.size),
        z_sd=float(z.std()),
        in196=float(np.mean(np.abs(z) < 1.96)),
        rmse=float(np.sqrt(np.mean(e**2))),
        resid_iqr=float(np.subtract(*np.percentile(res, [75, 25]))),
        resid_median=float(np.median(res)),
    )
    log(f"   calibration {label}: {out}")
    return out


def calibration_old(F, tp, ok):
    """collide.py's field on the voxel graph (WM-boundary voxels pooled in 2x2x2 blocks by side), with
    the same tops (each top's root to its nearest WM-boundary voxel), and the same folds."""
    sys.argv = [sys.argv[0], A.fig, A.work]
    from collide import Field, graph_L, shell_graph
    import collide

    K = C.curvature(F.B, 0.8)
    pts, nrm = K["pts"], K["n"]
    Hroot = np.clip(0.5 * (K["k1"] + K["k2"]), -H_CLIP, H_CLIP)
    side = np.argmax(np.abs(nrm), 1) * 2 + (
        nrm[np.arange(len(nrm)), np.argmax(np.abs(nrm), 1)] > 0
    )
    _, node_of, counts = np.unique(
        np.c_[pts // 2, side], axis=0, return_inverse=True, return_counts=True
    )
    node_of = node_of.ravel()
    nN = node_of.max() + 1
    ppos = pts * F.vox
    npos = (
        np.stack([np.bincount(node_of, ppos[:, a], nN) for a in range(3)], 1)
        / counts[:, None]
    )
    nH = np.bincount(node_of, Hroot, nN) / counts
    Gc = shell_graph(pts, nrm, F.vox).tocoo()
    ea, eb = node_of[Gc.row], node_of[Gc.col]
    ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
    fld = Field(graph_L(npos, ed), nH)
    node = node_of[cKDTree(pts).query(tp.R[ok])[1]]
    val = np.log(tp.s[ok] * F.h)
    y = np.full(nN, np.nan)
    order = np.argsort(node)
    ns, vs = node[order], val[order]
    st = np.flatnonzero(np.r_[True, np.diff(ns) != 0])
    en = np.r_[st[1:], ns.size]
    for a, b in zip(st, en):
        y[ns[a]] = np.median(vs[a:b])
    ev = np.flatnonzero(np.isfinite(y))
    c = npos[ev]
    ax_ = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][0]
    folds = np.array_split(ev[np.argsort((c - c.mean(0)) @ ax_)], 5)
    zs, err = [], []
    for fo in folds:
        yy = y.copy()
        yy[fo] = np.nan
        m_, s_, _ = fld.fit(yy, trend=True, nsamp=NSAMP_CAL)
        zs.append((y[fo] - m_[fo]) / np.sqrt(s_[fo] ** 2 + collide.SIG_EPS**2))
        err.append(y[fo] - m_[fo])
    z, e = np.concatenate(zs), np.concatenate(err)
    m_, _, coef = fld.fit(y, trend=True, nsamp=2)
    res = y[ev] - m_[ev]
    out = dict(
        label="old voxel graph, length",
        n=int(ev.size),
        z_sd=float(z.std()),
        in196=float(np.mean(np.abs(z) < 1.96)),
        rmse=float(np.sqrt(np.mean(e**2))),
        resid_iqr=float(np.subtract(*np.percentile(res, [75, 25]))),
        resid_median=float(np.median(res)),
        a0=float(coef["a0"][0]),
    )
    log(f"   calibration old: {out}")
    return out


# ------------------------------------------------------------------ main


def main():
    F = Face(A.site)
    h = F.h
    P = F.P
    Pi = np.round(P).astype(int)
    u_pts = F.nh[tuple(Pi.T)]
    out = dict(
        site=A.site,
        box_lo=F.lo.tolist(),
        points=len(P),
        h=h,
        alpha=ALPHA,
        topology=TOPO,
        mesh=dict(vertices=len(F.mesh.verts), faces=len(F.mesh.faces)),
    )

    # ---- evidence run: the prior budgets
    prior_len = np.exp(LM.M_MU + LM.A0_MU * F.H) / h  # voxels
    prior_vol = prior_len.copy()
    f_ev = F.run(prior_len, L.LENGTH)
    E = evidence(F, f_ev)
    log(
        f"   evidence run {f_ev.elapsed:.0f} s: {E['ntops']:,} tops, {E['ntops_used']:,} used, "
        f"{np.isfinite(E['len']).sum():,} vertices; median s {np.exp(np.nanmedian(E['len'])):.2f} mm, "
        f"w {np.exp(np.nanmedian(E['vol'])):.2f} mm"
    )
    out["evidence"] = dict(
        tops=E["ntops"],
        used=E["ntops_used"],
        vertices=int(np.isfinite(E["len"]).sum()),
        median_s=float(np.exp(np.nanmedian(E["len"]))),
        median_w=float(np.exp(np.nanmedian(E["vol"]))),
    )

    # ---- fields
    t0 = time.time()
    fld = LM.MeshField(F.mesh, F.H, alpha=ALPHA)
    post, fits = {}, {}
    for key in ("len", "vol"):
        r = budgets(F, fld, E[key], trend=True)
        r0 = fld.fit(E[key], trend=False)
        res = E[key][np.isfinite(E[key])] - r["mu"][np.isfinite(E[key])]
        fits[key] = dict(
            a0=r["a0"],
            a_p5_50_95=np.percentile(r["a"], [5, 50, 95]).round(3).tolist(),
            m=float(np.exp(r["m"])),
            no_belief=r["nobel"],
            b_p1_50_99=np.percentile(r["b"], [1, 50, 99]).round(2).tolist(),
            resid_iqr=float(np.subtract(*np.percentile(res, [75, 25]))),
            notrend_m=float(np.exp(r0["m"])),
        )
        post[key] = r["b"] / h
    log(f"   fields ({time.time() - t0:.0f} s): {json.dumps(fits)}")
    out["fields"] = fits
    eq = {
        k: np.full(len(F.mesh.verts), float(np.exp(np.nanmedian(E[k]))) / h)
        for k in ("len", "vol")
    }
    settings = {
        "equal": eq,
        "prior": {"len": prior_len, "vol": prior_vol},
        "posterior": post,
    }

    # the + side at each point: the end whose posterior length budget is smaller
    f_tmp = f_ev
    rt = F.root_vertex(f_tmp.R)
    thin_sign = np.zeros(len(P))
    steps = np.arange(-2.5, 2.5 + 1e-9, 0.05) / h
    for k, Pk in enumerate(P):
        u = u_pts[k] / (np.linalg.norm(u_pts[k]) + 1e-12)
        line = np.round(Pk + steps[:, None] * u).astype(int)
        inside = np.all((line >= 0) & (line < np.array(F.wm.shape)), 1)
        ok = np.zeros(steps.size, bool)
        ok[inside] = (L.claimed(f_tmp) & F.ctx)[tuple(line[inside].T)]
        neg, pos = np.flatnonzero(ok & (steps < 0)), np.flatnonzero(ok & (steps > 0))
        if neg.size and pos.size:
            bm, bp = (
                post["len"][rt[tuple(line[neg[0]])]],
                post["len"][rt[tuple(line[pos[-1]])]],
            )
            thin_sign[k] = -1.0 if bm < bp else 1.0
        else:
            thin_sign[k] = 1.0

    if os.environ.get("CALIB_ONLY", "0") == "1":
        out["calibration"] = [
            calibration(F, fld, E["len"], "mesh field, length"),
            calibration_old(F, E["tops"], E["ok"]),
            calibration(F, fld, E["vol"], "mesh field, volume"),
        ]
        json.dump(
            out,
            open(f"{A.work}/levelset_{A.site}_calibration.json", "w"),
            indent=1,
            default=float,
        )
        return
    runs = {}
    for sname, bud in settings.items():
        for mode, key in ((L.LENGTH, "len"), (L.VOLUME, "vol")):
            if ("length" if mode == L.LENGTH else "volume") not in FORMS:
                continue
            name = f"{sname}, {'length' if mode == L.LENGTH else 'volume'}"
            f = F.run(bud[key], mode, topology=TOPO)
            sm = L.seams(f)
            rows = score(F, f, sm, P, u_pts, thin_sign, h)
            # per side error: mu - S (length), v - w (volume) with this run's budgets at the seam
            for r in rows:
                if np.isfinite(r["offset"]):
                    if mode == L.LENGTH:
                        r["err_thin"], r["err_thick"] = (
                            r["b_thin"] - r["S_thin"],
                            r["b_thick"] - r["S_thick"],
                        )
                    else:
                        r["err_thin"], r["err_thick"] = (
                            r["b_thin"] - r["W_thin"],
                            r["b_thick"] - r["W_thick"],
                        )
                r["coh"] = float(F.coh[tuple(Pi[r["k"] - 1])])
                r["blend"] = float(f.blend[tuple(Pi[r["k"] - 1])])
            s = summary(rows)
            st = seam_stats(f, sm, F)
            out[name] = dict(summary=s, stats=st, rows=rows)
            log(
                f"   {name:22s} |offset| {s['median_abs']:.2f} mm (median {s['median']:+.2f}, IQR {s['iqr']:.2f}), "
                f"within 0.16 mm {s['within_016_of_all']:.0%} of points, worst {s['worst']:.2f} (point {s['worst_point']}), "
                f"missing {s['missing']}; nearest {s['nearest_median']:.2f}/{s['nearest_p90']:.2f}; "
                f"coverage {st['coverage']['fraction']:.4f}; classes c/f/o/m {st['classes_inner']}; "
                f"beyond tau 1 {st['beyond_tau1']:.1%}; capped {st['capped']} ({st['capped_beyond_3mm']:.0%} beyond 3 mm); "
                f"moved by the budgets {st['moved']:,}; {st['elapsed']:.0f} s"
            )
            runs[name] = (f, sm)
    # tops against budgets: the evidence of the posterior length run against the evidence run
    Ep = evidence(F, runs["posterior, length"][0])
    both = np.isfinite(E["len"]) & np.isfinite(Ep["len"])
    dlen = np.abs(E["len"][both] - Ep["len"][both])
    out["tops_vs_budget"] = dict(
        vertices=int(both.sum()),
        frac_identical=float(np.mean(dlen < 1e-9)),
        p99_abs_log=float(np.percentile(dlen, 99)),
        max_abs_log=float(dlen.max()),
    )
    log(f"   tops vs budgets: {out['tops_vs_budget']}")
    # carried budget against the budget looked up at the carried R (posterior, length)
    f = runs["posterior, length"][0]
    m = L.claimed(f) & F.ctx
    rv = F.root_vertex(f.R[m])
    diff = np.abs(np.log(f.b[m]) - np.log(post["len"][rv]))
    out["carried_vs_lookup"] = dict(
        median=float(np.median(diff)),
        p95=float(np.percentile(diff, 95)),
        p99=float(np.percentile(diff, 99)),
    )
    log(f"   carried budget vs lookup at R (|log|): {out['carried_vs_lookup']}")
    # A9-like: correlation of the two sides' errors at the points (posterior, length)
    rows = out["posterior, length"]["rows"]
    et = np.array([r.get("err_thin", np.nan) for r in rows])
    ek = np.array([r.get("err_thick", np.nan) for r in rows])
    okk = np.isfinite(et) & np.isfinite(ek)
    if okk.sum() > 5:
        out["side_error_corr"] = float(spearmanr(et[okk], ek[okk]).statistic)
    if CALIB:
        out["calibration"] = [
            calibration(F, fld, E["len"], "mesh field, length"),
            calibration(F, fld, E["vol"], "mesh field, volume"),
            calibration_old(F, E["tops"], E["ok"]),
        ]
    json.dump(
        out, open(f"{A.work}/levelset_{A.site}{TAG}.json", "w"), indent=1, default=float
    )
    figures(F, out, runs)
    overlay(F, runs)


def figures(F, out, runs):
    P = F.P
    k = np.arange(1, len(P) + 1)
    names = [n for n in out if isinstance(out[n], dict) and "summary" in out[n]]
    cols = {"equal": GREY, "prior": VIOLET, "posterior": BLUE}
    fig, axs = plt.subplots(3, 1, figsize=(max(9, 0.34 * len(P) + 3), 10), sharex=True)
    a = axs[0]
    for j, nm in enumerate(names):
        off = np.array([r["offset"] for r in out[nm]["rows"]])
        sname = nm.split(",")[0]
        mk = "o" if "length" in nm else "s"
        s = out[nm]["summary"]
        a.plot(
            k + (j - 2.5) * 0.12,
            off,
            mk,
            color=cols[sname],
            ms=4,
            mfc="none" if "length" in nm else cols[sname],
            label=f"{nm}: |offset| {s['median_abs']:.2f} mm, missing {s['missing']}",
        )
    a.axhspan(-0.16, 0.16, color="#dddddd", zorder=0, label="one voxel")
    a.axhline(0, color="#52514e", lw=0.8)
    a.set_ylabel("offset along the normal (mm)\n+ toward the side predicted thinner")
    a.set_title(f"{A.site}: level-set seam against your {len(P)} points (3D, 1x grid)")
    a.legend(fontsize=7, ncol=2)
    for ax_i, (nm, lab) in enumerate(
        (("posterior, length", "mu - S (mm)"), ("posterior, volume", "v - w (mm)")),
        start=1,
    ):
        a = axs[ax_i]
        if nm not in out:
            a.axis("off")
            continue
        rows = out[nm]["rows"]
        for key, col, l2, dx in (
            ("err_thin", BLUE, "side predicted thinner", -0.12),
            ("err_thick", ORANGE, "side predicted thicker", 0.12),
        ):
            e = np.array([r.get(key, np.nan) for r in rows])
            a.plot(k + dx, e, "o", color=col, ms=4, label=f"{l2}")
        a.axhline(0, color="#52514e", lw=0.8)
        a.set_ylabel(f"{nm}:\n{lab} at the point")
        a.legend(fontsize=7)
    axs[-1].set_xlabel("your contact point")
    axs[-1].set_xticks(k)
    fig.tight_layout()
    fig.savefig(f"{A.fig}/levelset_{A.site}{TAG}_offsets.png", dpi=110)
    plt.close(fig)
    # planes through the points: the seams of the posterior runs over echo 2 (trend removed), tau as contours
    Pi = np.round(P).astype(int)
    ax_fix = int(np.argmin([len(np.unique(Pi[:, d])) for d in range(3)]))
    vals, cnt = np.unique(Pi[:, ax_fix], return_counts=True)
    planes = vals[np.argsort(-cnt)][:3]
    show = [
        nm
        for nm in (
            "equal, length",
            "prior, length",
            "posterior, length",
            "posterior, volume",
        )
        if nm in runs
    ][:3]
    fig, axs = plt.subplots(
        len(planes),
        len(show),
        figsize=(4.4 * len(show), 4.2 * len(planes)),
        squeeze=False,
    )
    disp = np.where(F.ctx, F.B["disp"], np.nan)
    lim = np.nanpercentile(np.abs(disp), 98)
    for i, pl in enumerate(planes):
        sl = [slice(None)] * 3
        sl[ax_fix] = pl
        rest = [d for d in range(3) if d != ax_fix]
        for j, nm in enumerate(show):
            a = axs[i][j]
            f, sm = runs[nm]
            a.imshow(
                np.where(F.wm[tuple(sl)], 1.0, np.nan).T,
                origin="lower",
                cmap="Greys",
                vmin=0,
                vmax=2.5,
            )
            a.imshow(
                disp[tuple(sl)].T, origin="lower", cmap="gray", vmin=-lim, vmax=lim
            )
            tau = np.where(L.claimed(f) & F.ctx, f.tau, np.nan)[tuple(sl)]
            a.contour(
                tau.T,
                levels=[0.25, 0.5, 0.75, 1.0],
                colors=["#7fb8ff"] * 3 + ["#ffd23f"],
                linewidths=0.6,
                origin="lower",
            )
            mm_ = np.abs(sm.pos[:, ax_fix] - pl) <= 0.5
            a.plot(
                sm.pos[mm_, rest[0]],
                sm.pos[mm_, rest[1]],
                ".",
                color=ORANGE,
                ms=2.5,
                label="seam",
            )
            q = Pi[:, ax_fix] == pl
            a.plot(
                P[q, rest[0]],
                P[q, rest[1]],
                "+",
                color=AQUA,
                ms=8,
                mew=1.6,
                label="your points",
            )
            c0 = P[q][:, rest].mean(0)
            a.set_xlim(c0[0] - 30, c0[0] + 30)
            a.set_ylim(c0[1] - 30, c0[1] + 30)
            a.set_title(f"{nm}; {'ijk'[ax_fix]} = {pl + 1} (ITK-SNAP)", fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
    axs[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"{A.site}: echo 2 (trend removed), WM dark grey; tau 0.25/0.5/0.75 blue, 1 yellow; "
        "computed in 3D, shown on planes through the points",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{A.fig}/levelset_{A.site}{TAG}_planes.png", dpi=110)
    plt.close(fig)


def overlay(F, runs):
    """ITK-SNAP: labels (1 start band, 2 seam posterior length, 3 seam posterior volume, 4 both, 5 your
    points, 6-11 side ownership by root: the root's 1 mm block, coloured cyclically), and tau (posterior,
    volume) as a float image."""
    shape = F.wm.shape
    lab = np.zeros(shape, np.uint8)
    f, _ = runs[
        "posterior, volume" if "posterior, volume" in runs else "posterior, length"
    ]
    m = L.claimed(f) & F.ctx
    blk = np.floor(f.R[m] * F.h / 1.0).astype(np.int64)
    code = (blk[:, 0] * 7 + blk[:, 1] * 13 + blk[:, 2] * 29) % 6
    lab[m] = 6 + code
    lab[F.band] = 1
    sl, sv = np.zeros(shape, bool), np.zeros(shape, bool)
    for nm, arr in (("posterior, length", sl), ("posterior, volume", sv)):
        if nm not in runs:
            continue
        sm = runs[nm][1]
        arr.ravel()[sm.p] = True
        arr.ravel()[sm.q] = True
    lab[sl] = 2
    lab[sv] = 3
    lab[sl & sv] = 4
    for p in np.round(F.P).astype(int):
        lab[tuple(p)] = 5
    aff = F.B["p"].affine
    nib.save(
        nib.Nifti1Image(lab, aff), f"{A.data}/levelset_{A.site}{TAG}_overlay.nii.gz"
    )
    nib.save(
        nib.Nifti1Image(np.where(m, f.tau, 0).astype(np.float32), aff),
        f"{A.data}/levelset_{A.site}{TAG}_tau.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(np.where(F.ctx, F.B["disp"], 0.0).astype(np.float32), aff),
        f"{A.data}/levelset_{A.site}_echo2_detrended.nii.gz",
    )
    with open(f"{A.data}/levelset_labels.txt", "w") as fh:
        fh.write("# ITK-SNAP label description file\n")
        fh.write('    0     0    0    0        0  0  0    "Clear Label"\n')
        names = [
            ((150, 150, 150), "start band (WM boundary)"),
            ((42, 120, 214), "seam, posterior, length"),
            ((235, 104, 52), "seam, posterior, volume"),
            ((227, 73, 72), "seam, both forms"),
            ((27, 175, 122), "your points"),
        ]
        pal = [
            (102, 194, 165),
            (252, 141, 98),
            (141, 160, 203),
            (231, 138, 195),
            (166, 216, 84),
            (255, 217, 47),
        ]
        names += [(c, f"side ownership: root block {i + 1}") for i, c in enumerate(pal)]
        for i, (rgb, nm) in enumerate(names, start=1):
            vis = 1 if i <= 5 else 0
            fh.write(
                f'  {i:3d}  {rgb[0]:3d}  {rgb[1]:3d}  {rgb[2]:3d}        1  {vis}  1    "{nm}"\n'
            )


if __name__ == "__main__":
    main()
