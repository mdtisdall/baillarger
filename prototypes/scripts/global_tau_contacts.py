"""Contact measures of a global-τ run, for a person (not pass/fail). The
measures are those of tube_contacts.py (branch feature/tube-solver), on the
contact faces of the ownership field instead of the tube bands.

usage: global_tau_contacts.py SITE_NPZ RUN_DIR
Reads RUN_DIR/result.npz and log.json; writes RUN_DIR/contacts.json.

Per contact pair (a, b) of owner patches in the final state (a contact face
between a voxel of a and a voxel of b, the patches not in the same bank):
- root_mm: geodesic distance of the patch centres along the cropped white
  surface (inf when not connected inside the crop);
- n_faces, area_mm2; labels of the face voxels; the fraction of label 0 in
  their 6-neighbourhoods; echo 2 at the face voxels relative to the crop's
  cortex median;
- signature: echo-2 mean profile of each patch in τ bins (0 to -3 mm,
  0.16 mm) and the Pearson correlation of the two profiles;
- tau_face_mm: median τ of the face voxels on each side (about -h/2 when the
  ghost value holds);
- thickness_mm, residual (log thickness - log median), imbalance_mm (the
  thickness minus the median of the open (no contact) mesh neighbours within
  2 mm);
- component, rim: the connected contact region (patch adjacency) and whether
  the patch borders an open patch;
- conflicts: label0_in_gap, thickness_outlier (|residual| > 2 SD),
  signature_mismatch (correlation < 0);
- rule_only: mean coherence of both patches' top 1 mm < 0.1 and no label 0.
The face units of the face test (log.json face_patches) carry the signed
bank disagreement (mm; positive = a gap, negative = the two sides' evidence
disagrees; never read as overlap).
"""

import json
import sys
from pathlib import Path

import global_tau_energy as GE
import global_tau_prep as TP
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

BINS = np.arange(0.0, -3.0001, -0.16)
AX = np.eye(3, dtype=np.int64)


def main() -> None:
    site_p, run = Path(sys.argv[1]), Path(sys.argv[2])
    z = np.load(site_p)
    r = np.load(run / "result.npz")
    log = json.loads((run / "log.json").read_text())
    labels = z["labels"]
    e2 = z["echoes"][1].astype(float)
    shape = np.array(labels.shape)
    tau, owner, inner = r["tau"].astype(float), r["owner"].astype(np.int64), r["inner"]
    faces = r["faces"]
    vox = np.abs(np.diag(z["affine"])[:3])
    cm = faces[:, :3]
    cp = cm + AX[faces[:, 3]]
    oa, ob = owner[tuple(cm.T)], owner[tuple(cp.T)]
    th = r["thickness"]
    resid = np.log(np.maximum(th, 0.02)) - r["logmed"]
    ntube = len(th)
    cmed = float(np.median(e2[labels == TP.CORTEX]))
    coh = r["coh"]

    flat_t, flat_o = tau.ravel(), owner.ravel()
    m = (flat_o >= 0) & np.isfinite(flat_t) & (flat_t < 0) & ~inner.ravel()
    ti, tt = flat_o[m], flat_t[m]
    bi = np.clip(np.digitize(-tt, -BINS) - 1, 0, len(BINS) - 2)
    nb_ = len(BINS) - 1
    cnt = np.zeros((ntube, nb_))
    sm = np.zeros((ntube, nb_))
    np.add.at(cnt, (ti, bi), 1)
    np.add.at(sm, (ti, bi), e2.ravel()[m] / cmed)
    prof = np.where(cnt >= 3, sm / np.maximum(cnt, 1), np.nan)
    top = tt > -1.0
    ctop = np.bincount(
        ti[top], weights=coh.ravel()[m][top], minlength=ntube
    ) / np.maximum(np.bincount(ti[top], minlength=ntube), 1)

    pairs: dict[tuple[int, int], list[int]] = {}
    for k, (a, b) in enumerate(zip(oa, ob)):
        key = (int(min(a, b)), int(max(a, b)))
        pairs.setdefault(key, []).append(k)
    contact = np.zeros(ntube, bool)
    for a, b in pairs:
        contact[a] = contact[b] = True

    # geodesic root distance on the cropped surface
    v = r["vertex_vox"]
    f = z["white_f"].astype(np.int64)
    vmm = v * vox
    g = TP.mesh_graph(vmm, f, np.ones(len(vmm), bool))
    cen = r["centre"]
    keys = sorted(pairs)
    pa = np.array(keys, np.int64).reshape(-1, 2)
    root = np.full(len(pa), np.inf)
    for s0 in range(0, len(pa), 64):
        srcs = np.unique(pa[s0 : s0 + 64, 0])
        dm = csgraph.dijkstra(g, indices=cen[srcs], limit=20.0)
        pos = {int(x): k for k, x in enumerate(srcs)}
        for k in range(s0, min(s0 + 64, len(pa))):
            root[k] = dm[pos[int(pa[k, 0])], cen[pa[k, 1]]]

    adj = r["adj"]
    keep = contact[adj[:, 0]] & contact[adj[:, 1]]
    A = sparse.coo_matrix(
        (np.ones(keep.sum()), (adj[keep, 0], adj[keep, 1])), shape=(ntube, ntube)
    )
    _, comp = csgraph.connected_components(A, directed=False)
    nbr: list[list[int]] = [[] for _ in range(ntube)]
    for i, j in adj:
        nbr[i].append(int(j))
        nbr[j].append(int(i))
    rim = np.array(
        [contact[t] and any(not contact[j] for j in nbr[t]) for t in range(ntube)]
    )
    cen_mm = vmm[cen]

    def imbalance(t: int) -> float:
        js = [j for j in nbr[t] if not contact[j] and np.isfinite(th[j])]
        js = [j for j in js if np.linalg.norm(cen_mm[j] - cen_mm[t]) <= 2.0]
        return float(th[t] - np.median(th[js])) if js else float("nan")

    def label0_frac(c: np.ndarray) -> float:
        n0 = 0
        for ax in range(3):
            for sg in (-1, 1):
                q = np.clip(c + sg * AX[ax], 0, shape - 1)
                n0 += int((labels[tuple(q.T)] == TP.OUTSIDE).sum())
        return n0 / (6 * len(c))

    out = []
    for k, (a, b) in enumerate(keys):
        idx = np.array(pairs[(a, b)])
        vox_all = np.concatenate([cm[idx], cp[idx]])
        lab = labels[tuple(vox_all.T)]
        frac0 = label0_frac(vox_all)
        side_a = np.where(oa[idx] == a, 0, 1)
        ta = np.concatenate(
            [tau[tuple(cm[idx][side_a == 0].T)], tau[tuple(cp[idx][side_a == 1].T)]]
        )
        tb = np.concatenate(
            [tau[tuple(cp[idx][side_a == 0].T)], tau[tuple(cm[idx][side_a == 1].T)]]
        )
        pa_, pb_ = prof[a], prof[b]
        ok = np.isfinite(pa_) & np.isfinite(pb_)
        corr = (
            float(np.corrcoef(pa_[ok], pb_[ok])[0, 1])
            if ok.sum() >= 4
            else float("nan")
        )
        conf = []
        if frac0 > 0:
            conf.append("label0_in_gap")
        if max(abs(resid[a]), abs(resid[b])) > 2 * GE.THICK_SD:
            conf.append("thickness_outlier")
        if np.isfinite(corr) and corr < 0:
            conf.append("signature_mismatch")
        out.append(
            {
                "a": a,
                "b": b,
                "root_mm": float(root[k]),
                "n_faces": int(len(idx)),
                "area_mm2": float(len(idx) * np.mean(vox) ** 2),
                "labels": {int(u): int((lab == u).sum()) for u in np.unique(lab)},
                "label0_frac_nbhd": frac0,
                "echo2_rel": float(np.mean(e2[tuple(vox_all.T)]) / cmed),
                "tau_face_mm": [float(np.median(ta)), float(np.median(tb))],
                "signature_corr": corr,
                "profile_a": pa_.tolist(),
                "profile_b": pb_.tolist(),
                "thickness_mm": [float(th[a]), float(th[b])],
                "residual": [float(resid[a]), float(resid[b])],
                "imbalance_mm": [imbalance(a), imbalance(b)],
                "component": [int(comp[a]), int(comp[b])],
                "rim": [bool(rim[a]), bool(rim[b])],
                "in_region": [bool(r["in_region"][a]), bool(r["in_region"][b])],
                "conflicts": conf,
                "rule_only": bool(ctop[a] < 0.1 and ctop[b] < 0.1 and frac0 == 0),
            }
        )
    comp_ids = np.unique(comp[contact])
    units = log.get("face_patches", [])
    dis = np.array([u["disagreement_mm"] for u in units]) if units else np.zeros(0)
    summary = {
        "contact_pairs": len(out),
        "contact_faces": int(len(faces)),
        "contact_area_mm2": float(len(faces) * np.mean(vox) ** 2),
        "contact_components": int(len(comp_ids)),
        "pairs_root_inf": int(np.sum(~np.isfinite(root))),
        "pairs_with_label0": int(sum("label0_in_gap" in o["conflicts"] for o in out)),
        "pairs_thickness_outlier": int(
            sum("thickness_outlier" in o["conflicts"] for o in out)
        ),
        "pairs_signature_mismatch": int(
            sum("signature_mismatch" in o["conflicts"] for o in out)
        ),
        "pairs_rule_only": int(sum(o["rule_only"] for o in out)),
        "tau_face_mm_median": float(
            np.median([t for o in out for t in o["tau_face_mm"]])
        )
        if out
        else float("nan"),
        "patches": int(ntube),
        "contact_patches": int(contact.sum()),
        "thickness_mm_region_p10_50_90": np.nanpercentile(
            th[r["in_region"]], [10, 50, 90]
        ).tolist(),
        "thickness_mm_contact_p10_50_90": np.nanpercentile(
            th[contact], [10, 50, 90]
        ).tolist()
        if contact.any()
        else [],
        "face_units": len(units),
        "disagreement_mm_p10_50_90": np.percentile(dis, [10, 50, 90]).tolist()
        if len(dis)
        else [],
        "disagreement_censored": int(sum(u["censored"] for u in units)),
    }
    (run / "contacts.json").write_text(
        json.dumps({"summary": summary, "contacts": out}, default=float)
    )
    print(json.dumps(summary, default=float))


if __name__ == "__main__":
    main()
