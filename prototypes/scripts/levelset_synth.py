"""Phantom results for level-set filling (test plan §5 items 1-7), both forms. Voxels; all 3D.

The exact-answer tests are in test_levelset_fill.py; this script reports the numbers and draws
levelset_synth_*.png. Budgets from the evidence on phantoms: for each root, the median of the visible
tops' s (length) or w (volume) whose roots lie within 12.5 voxels (2 mm) on the same side (outward
normals agreeing). The mesh field is run once end to end on P4 (finding 28's surface of P4's WM, the cotangent
field, budgets at the band voxels' feet), against the 12 / 18 answer.

Usage: levelset_synth.py <figures dir> <work dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import levelset_fill as L
import levelset_mesh as LM
import levelset_phantoms as PH
from test_levelset_fill import crown_exact, p_budgets, region_seams, seam_vs_reference, tau_to_radius, wedge_reference

FIG = Path(sys.argv[1])
WORK = Path(sys.argv[2])
MODES = ((L.LENGTH, "length"), (L.VOLUME, "volume"))


def run(ph, budget, mode, n=None, coh=None, **kw):
    b = budget if isinstance(budget, np.ndarray) else np.full(ph.wm.shape, float(budget))
    return L.march(ph.cortex, ph.wm, ph.band, ph.phi, ph.foot, ph.normal, b, ph.n if n is None else n, coh,
                   mode=mode, **kw)


def evidence_budget(ph, f, mode, radius=12.5):
    """Local median of the tops' s or w over roots within radius on the same side, per band voxel."""
    tp = L.tops(f, ph.fluid)
    val = tp.s if mode == L.LENGTH else tp.w
    nr = ph.normal.reshape(-1, 3)[tp.idx]  # top voxel's normal ~ its root's normal (flat banks)
    tree = cKDTree(tp.R)
    out = np.full(ph.wm.shape, np.nan)
    bi = np.argwhere(ph.band)
    feet = ph.foot[ph.band]
    nb = ph.normal[ph.band]
    nn = tree.query_ball_point(feet, radius)
    allv = np.median(val)
    vals = np.empty(len(bi))
    for i, idx in enumerate(nn):
        idx = np.array(idx, int)
        if idx.size:
            idx = idx[np.sum(nr[idx] * nb[i], 1) > 0]
        if idx.size == 0:
            _, j = tree.query(feet[i])
            vals[i] = val[j] if np.dot(nr[j], nb[i]) > 0 else allv
        else:
            vals[i] = np.median(val[idx])
    out[tuple(bi.T)] = vals
    return out


def main():
    res: dict = {}
    panels = []

    # 1. slab
    ph = PH.slab()
    z = np.broadcast_to(np.arange(ph.wm.shape[2])[None, None, :], ph.wm.shape) - 9.5
    r = {}
    for mode, mn in MODES:
        f = run(ph, 12.0, mode)
        c = ph.cortex
        tp = L.tops(f, ph.fluid)
        r[mn] = dict(s_maxerr=float(np.abs(f.s[c] - z[c]).max()), tau_maxerr=float(np.abs(f.tau[c] - z[c] / 12).max()),
                     tops=float(np.median(tp.s)), tops_w=float(np.median(tp.w)))
    res["slab"] = r

    # 2. crown
    for kind in ("cylinder", "sphere"):
        ph = getattr(PH, kind)()
        rr, k, r0, r1, vol, v1 = crown_exact(ph, kind)
        r = {}
        for mode, mn in MODES:
            f = run(ph, (r1 - r0) if mode == L.LENGTH else v1, mode)
            m = ph.cortex & L.claimed(f)
            err = tau_to_radius(f.tau[m], mode, k, r0, r1, v1) - rr[m]
            inner = m & (rr < r1 - 1)
            ex = (rr / r0) ** k
            aR = L.a_from_R(f)
            r[mn] = dict(radius_err_median=float(np.median(err)), radius_err_p95=float(np.percentile(np.abs(err), 95)),
                         radius_err_max=float(np.abs(err).max()),
                         a_div_p95=float(np.percentile(np.abs(f.a[inner] / ex[inner] - 1), 95)),
                         a_R_p95=float(np.nanpercentile(np.abs(aR[inner] / ex[inner] - 1), 95)),
                         capped=int(f.capped[m].sum()))
            if mode == L.VOLUME:
                panels.append((kind, ph, f, None))
        # the other form's level sets against the equal-volume radii: how far apart the forms are
        f_len = run(ph, r1 - r0, L.LENGTH)
        m = ph.cortex & L.claimed(f_len)
        diff = tau_to_radius(f_len.tau[m], L.VOLUME, k, r0, r1, v1) - rr[m]
        r["length_form_vs_equal_volume_radii_max"] = float(np.abs(diff).max())
        res[kind] = r

    # 3. P4, P8
    for name in ("p4", "p8"):
        ph = getattr(PH, name)()
        r = {}
        for mode, mn in MODES:
            for lab in ("oracle", "open 12/18", "evidence"):
                if lab == "evidence":
                    f0 = run(ph, p_budgets(ph, True)[0], mode)
                    bud = evidence_budget(ph, f0, mode)
                    pred = None
                else:
                    bud, pred = p_budgets(ph, lab == "oracle")
                f = run(ph, bud, mode)
                sm, inr, nl = region_seams(ph, f)
                e = sm.pos[inr, 0] - ph.extra["known"]
                r[f"{mn}, {lab}"] = dict(lines=nl, found=int(inr.sum()), offset_median=float(np.median(e)),
                                         offset_max=float(np.abs(e).max()), pred=None if pred is None else float(pred - ph.extra["known"]),
                                         tau=[float(np.median(sm.tau_a[inr])), float(np.median(sm.tau_b[inr]))],
                                         coverage=L.coverage(f, ph.wm, ph.fluid)["fraction"])
                if lab == "evidence":
                    sideA = ph.foot[..., 0] < ph.extra["split"]
                    plate = ph.band & (ph.foot[..., 2] > 20)
                    r[f"{mn}, {lab}"]["budgets_A_B"] = [float(np.median(bud[plate & sideA])), float(np.median(bud[plate & ~sideA]))]
                if mode == L.LENGTH and lab == "evidence":
                    panels.append((name.upper(), ph, f, sm))
        res[name] = r
    res["p4"]["old one pass (finding 24)"] = dict(column_tops=2.45, oracle=0.0)
    res["p8"]["old one pass (finding 24)"] = dict(column_tops=-0.19, oracle=0.0)

    # 4. wedge
    for th in (20, 45, 90):
        ph = PH.wedge(th)
        e = ph.extra
        bud = np.where(ph.foot[..., 0] < e["cx"], e["mu_a"], e["mu_b"])
        tA, tB = wedge_reference(ph)
        r = {}
        for ver in "ABC":
            n, coh = PH.wedge_normals(ph, ver)
            for mode, mn in MODES:
                f = run(ph, bud, mode, n=n, coh=coh)
                err = seam_vs_reference(ph, f, tA, tB)
                sm = L.seams(f)
                k = sm.axis == 0
                bis = sm.pos[k, 0] - e["cx"]
                r[f"{ver}, {mn}"] = dict(median=float(np.median(err)) if err.size else None,
                                         max=float(np.abs(err).max()) if err.size else None, n=int(err.size),
                                         seams=int(sm.p.size), moved=int(f.moved.sum()),
                                         seam_minus_bisector_median=float(np.median(bis)) if bis.size else None,
                                         blend_voxels=int(np.sum(f.blend > 0)),
                                         classes=np.bincount(L.seam_class(sm), minlength=4).tolist())
                if mode == L.LENGTH and th == 90:
                    panels.append((f"wedge 90 {ver}", ph, f, sm))
        # the exact seam's distance from the bisector (where B's normal flips) at the height where the
        # seam is L / 2 from the apex: how far the flip is from the answer
        nA, nB, ma, mb = e["nA"], e["nB"], e["mu_a"], e["mu_b"]
        hz = 0.5 * e["L"] * np.cos(np.radians(th) / 2)
        r["exact_seam_minus_bisector_at_half_L"] = float(hz * (nB[1] / mb - nA[1] / ma) / (nA[0] / ma - nB[0] / mb))
        res[f"wedge{th}"] = r

    # 5. U-slab
    for R in (6.0, 22.0):
        ph = PH.u_slab(R)
        e = ph.extra
        r = {}
        for mode, mn in MODES:
            f = run(ph, 16.0, mode)
            sm = L.seams(f)
            m = L.claimed(f) & ph.cortex
            d = dict(seams=int(sm.p.size), capped=int(f.capped[m].sum()), capped_frac=float(f.capped[m].mean()),
                     a_min=float(np.nanmin(f.a[m])))
            if sm.p.size:
                d.update(x_minus_cx_max=float(np.abs(sm.pos[:, 0] - e["cx"]).max()),
                         z_range=[float(sm.pos[:, 2].min()), float(sm.pos[:, 2].max())],
                         z_expected=[e["zo"], e["zmax"]],
                         tau_range=[float(min(sm.tau_a.min(), sm.tau_b.min())), float(max(sm.tau_a.max(), sm.tau_b.max()))],
                         angle_median=float(np.median(sm.angle)),
                         classes=np.bincount(L.seam_class(sm), minlength=4).tolist())
            r[mn] = d
            if mode == L.VOLUME and R == 6.0:
                panels.append(("U-slab R 6 (volume)", ph, f, sm))
        res[f"uslab_R{int(R)}"] = r

    # 6. omega
    ph = PH.omega()
    e = ph.extra
    fx = ph.foot[..., 0]
    r = {}
    for mode, mn in MODES:
        fs = run(ph, np.where(fx < e["cx"], e["tl"], e["tr"]), mode)
        fe = run(ph, 15.0, mode)
        floor = ph.cortex & L.claimed(fs) & (np.abs(fx - e["cx"]) < 12) & (ph.foot[..., 2] < 20)
        dR = np.linalg.norm(fs.R - ph.foot, axis=-1)
        m = L.claimed(fs) & ph.cortex
        sm = L.seams(fs)
        yz = np.round(sm.pos[:, 1:]).astype(int)
        inr = (sm.axis == 0) & e["region"][yz[:, 0], yz[:, 1]]
        err = sm.pos[inr, 0] - e["known"]
        r[mn] = dict(floor_R_err_p99=float(np.percentile(dR[floor], 99)),
                     floor_R_step_vs_equal_max=float(np.abs(fs.R[floor] - fe.R[floor]).max()),
                     all_R_err_median=float(np.median(dR[m])), all_R_err_p99=float(np.percentile(dR[m], 99)),
                     neck_offset_median=float(np.median(err)), neck_offset_max=float(np.abs(err).max()),
                     neck_lines=[int(inr.sum()), int(e["region"].sum())])
        if mode == L.LENGTH:
            panels.append(("omega", ph, fs, sm))
    res["omega"] = r

    # 7. blade
    ph = PH.blade()
    e = ph.extra
    f = run(ph, 12.0, L.LENGTH)
    m = L.claimed(f) & ph.cortex
    sides = m & (ph.foot[..., 2] < e["ztip"] - 0.6)
    wrong = np.sign(f.R[sides][:, 0] - e["mid"]) != np.sign(ph.foot[sides][:, 0] - e["mid"])
    res["blade"] = dict(wrong_side=int(wrong.sum()), R_err_max=float(np.linalg.norm(f.R - ph.foot, axis=-1)[m].max()))

    # mesh field end to end on P4
    res["p4_mesh_field"] = mesh_end_to_end()

    (WORK / "levelset_synth.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    figure(panels)


def mesh_end_to_end():
    import mesh_smoothing as MS

    ph = PH.p4()
    vox = np.array([0.16, 0.16, 0.16])
    v0, fa = MS.contour(ph.wm, 0.0)  # finding 28's surface, as levelset_face
    v = MS.twice(MS.relax(v0, fa), fa, 2.0)
    nr = MS.smooth_normals(v, fa)
    mesh = LM.Mesh(v, fa, nr, vox)
    # band and feet on the mesh, as levelset_face
    from scipy import ndimage as ndi

    st = np.ones((3, 3, 3), bool)
    near_ctx = ndi.binary_dilation(ph.cortex, st)
    near_wm = ndi.binary_dilation(ph.wm, st, iterations=2)
    cand = (ph.wm & near_ctx) | (ph.cortex & near_wm)
    ci = np.argwhere(cand)
    phi, foot, nrm, tri, bary = LM.closest(mesh, ci.astype(float))
    PHI = np.zeros(ph.wm.shape)
    FOOT = np.zeros(ph.wm.shape + (3,))
    NRM = np.zeros(ph.wm.shape + (3,))
    PHI[tuple(ci.T)], FOOT[tuple(ci.T)], NRM[tuple(ci.T)] = phi, foot, nrm
    band = (ph.wm & near_ctx) | (ph.cortex & near_wm & (PHI < 1.0))
    TRI = np.full(ph.wm.shape, -1)
    TRI[tuple(ci.T)] = tri
    BARY = np.zeros(ph.wm.shape + (3,))
    BARY[tuple(ci.T)] = bary

    def bb(vb):
        out = np.full(ph.wm.shape, np.nan)
        out[band] = np.exp(np.sum(np.log(vb)[mesh.faces[TRI[band]]] * BARY[band], 1))
        return out

    nv = len(v)
    f0 = L.march(ph.cortex, ph.wm, band, PHI, FOOT, NRM, bb(np.full(nv, 15.0)), ph.n, mode=L.LENGTH)
    tp = L.tops(f0, ph.fluid)
    vt = cKDTree(v)
    vtx = vt.query(tp.R)[1]
    y = np.full(nv, np.nan)
    for i in np.unique(vtx):
        y[i] = np.median(np.log(tp.s[vtx == i] * 0.16))
    k1, k2 = __import__("baillarger.frame", fromlist=["mesh_curvatures"]).mesh_curvatures(mesh.mm, fa, nr, 0.8)
    H = np.clip(0.5 * (k1 + k2), -0.8, 0.8)
    fld = LM.MeshField(mesh, H, alpha=2)
    r = fld.fit(y, trend=True)
    b = np.exp(r["mu"]) / 0.16
    nob = LM.pieces_without_evidence(mesh, np.isfinite(y))
    b[nob] = np.median(b[~nob])
    f = L.march(ph.cortex, ph.wm, band, PHI, FOOT, NRM, bb(b), ph.n, mode=L.LENGTH)
    sm = L.seams(f)
    reg = ph.extra["region"]
    yz = np.round(sm.pos[:, 1:]).astype(int)
    inr = (sm.axis == 0) & reg[yz[:, 0], yz[:, 1]]
    err = sm.pos[inr, 0] - ph.extra["known"]
    # budgets at the plates' vertices facing the patch
    vA = (np.abs(v[:, 0] - 9.5) < 0.6) & (v[:, 2] > 30)
    vB = (np.abs(v[:, 0] - 55.5) < 0.6) & (v[:, 2] > 30)
    sA = tp.s[(np.abs(tp.R[:, 0] - 9.5) < 1) & (tp.R[:, 2] > 30)]
    return dict(vertices=nv, mesh_face_x=[float(np.median(v[np.abs(v[:, 0] - 9.5) < 1.0, 0])),
                                          float(np.median(v[np.abs(v[:, 0] - 55.5) < 1.0, 0]))],
                tops_plateA_median=float(np.median(sA)),
                budget_A_B=[float(np.median(b[vA])), float(np.median(b[vB]))],
                a0=r["a0"], seam_offset_median=float(np.median(err)), seam_offset_max=float(np.abs(err).max()),
                lines=[int(inr.sum()), int(reg.sum())], rule_prediction_from_budgets=None)


def figure(panels):
    n = len(panels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for a, (title, ph, f, sm) in zip(axs.ravel(), panels):
        y0 = ph.wm.shape[1] // 2
        tau = np.where(L.claimed(f) & ph.cortex, f.tau, np.nan)[:, y0, :]
        lab = np.zeros(ph.wm.shape[::2])
        lab[ph.wm[:, y0, :]] = 1
        lab[ph.fluid[:, y0, :]] = 0.3
        a.imshow(lab.T, origin="lower", cmap="Greys", vmin=0, vmax=1.6)
        im = a.imshow(tau.T, origin="lower", cmap="viridis", vmin=0, vmax=1.5)
        a.contour(tau.T, levels=[0.25, 0.5, 0.75, 1.0], colors="w", linewidths=0.6, origin="lower")
        if sm is not None and sm.p.size:
            k = np.abs(sm.pos[:, 1] - y0) <= 0.5
            a.plot(sm.pos[k, 0], sm.pos[k, 2], ".", color="#eb6834", ms=3)
        a.set_title(title, fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    for a in axs.ravel()[n:]:
        a.axis("off")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5, label="tau")
    fig.suptitle("Level-set filling on phantoms (3D, one y plane shown): tau with contours at 0.25 steps, "
                 "seam orange; WM dark, fluid light", fontsize=10)
    fig.savefig(FIG / "levelset_synth_panels.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
