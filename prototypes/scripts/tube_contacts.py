"""Contact, stop and retreat diagnostics of a tube-solver run (for a person,
not pass/fail; the measures of the design notes' Outputs section).

usage: tube_contacts.py SITE_NPZ RUN_DIR
Reads RUN_DIR/result.npz and log.json; writes RUN_DIR/diagnostics.json.

Per contact pair (a, b) in the final state (a tissue voxel of a 6-adjacent to
a band voxel of b, the patches not in the same bank):
- root_mm: geodesic distance of the patch centres along the white surface
  (inf when not connected inside the crop);
- gap: contact voxels, their labels, the fraction of label 0 in their
  6-neighbourhoods, echo 2 at the contact relative to the crop's cortex median;
- signature: echo-2 mean profile of each tube in τ bins (0 to -3 mm, 0.16 mm)
  and the Pearson correlation of the two profiles;
- imbalance_mm: thickness minus the median of the open (no contact) mesh
  neighbours within 2 mm, for a and b (signed);
- component, rim: the connected contact region (patch adjacency) and whether
  the tube borders an open tube;
- conflicts: label0_in_gap, thickness_outlier (|log residual| > 2 SD),
  signature_mismatch (correlation < 0);
- rule_only: mean coherence of both tubes' top 1 mm < 0.1 and no label 0.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tube_energy as TE
import tube_kernels as K
import tube_prep as TP
from scipy import sparse
from scipy.sparse import csgraph

BINS = np.arange(0.0, -3.0001, -0.16)


def main() -> None:
    site_p, run = Path(sys.argv[1]), Path(sys.argv[2])
    z = np.load(site_p)
    r = np.load(run / "result.npz")
    log = json.loads((run / "log.json").read_text())
    labels = z["labels"]
    e2 = z["echoes"][1].astype(float)
    shape = labels.shape
    tau, owner, inside = r["tau"].astype(float), r["owner"].astype(np.int64), r["inside"]
    P = K.PAD
    sp = np.array(shape) + 2 * P
    st = np.array([sp[1] * sp[2], sp[2], 1], np.int64)
    taup = np.pad(tau, P, constant_values=np.nan).ravel()
    ownp = np.pad(owner, P, constant_values=-1).ravel()
    insp = np.pad(inside, P, constant_values=False).ravel()
    band = np.flatnonzero(ownp >= 0)
    rows = K.contacts(band, taup, ownp, insp, st, r["same_ptr"], r["same_idx"])
    unpad = lambda v: tuple((np.stack(np.unravel_index(v, sp), 1) - P).T)  # noqa: E731

    th = r["thickness"]
    resid = np.log(np.maximum(th, 0.02)) - r["logmed"]
    ntube = len(th)
    cmed = float(np.median(e2[labels == TP.CORTEX]))
    coh = np.zeros(shape)  # recompute coherence for rule_only
    _, coh = TP.laminar_normal(z["echoes"], labels)

    # per-tube profiles and top coherence
    flat_t = tau.ravel()
    flat_o = owner.ravel()
    m = (flat_o >= 0) & np.isfinite(flat_t) & (flat_t < 0) & ~inside.ravel()
    ti, tt = flat_o[m], flat_t[m]
    bi = np.clip(np.digitize(-tt, -BINS) - 1, 0, len(BINS) - 2)
    nb_ = len(BINS) - 1
    cnt = np.zeros((ntube, nb_))
    sm = np.zeros((ntube, nb_))
    np.add.at(cnt, (ti, bi), 1)
    np.add.at(sm, (ti, bi), e2.ravel()[m] / cmed)
    prof = np.where(cnt >= 3, sm / np.maximum(cnt, 1), np.nan)
    top = tt > -1.0
    ctop = np.bincount(ti[top], weights=coh.ravel()[m][top], minlength=ntube) / np.maximum(
        np.bincount(ti[top], minlength=ntube), 1)

    pairs = {}
    for a, b, v in rows:
        key = (int(a), int(b))
        pairs.setdefault(key, []).append(int(v))
    contact_tubes = np.zeros(ntube, bool)
    for a, b in pairs:
        contact_tubes[a] = contact_tubes[b] = True

    # geodesic root distance
    site = TP.Site(z["echoes"], labels, z["affine"], np.abs(np.diag(z["affine"])[:3]), r["vertex_vox"],
                   z["white_f"].astype(np.int64), None, None, None, None, inside, None, None)
    vmm = site.v * site.vox
    keys = sorted({(min(a, b), max(a, b)) for a, b in pairs})
    pa = np.array(keys, np.int64).reshape(-1, 2)
    g = TP.mesh_graph(vmm, site.f, np.ones(len(vmm), bool))
    root = np.full(len(pa), np.inf)
    cen = r["centre"]
    for s0 in range(0, len(pa), 64):
        srcs = np.unique(pa[s0 : s0 + 64, 0])
        dm = csgraph.dijkstra(g, indices=cen[srcs])
        pos = {int(x): k for k, x in enumerate(srcs)}
        for k in range(s0, min(s0 + 64, len(pa))):
            root[k] = dm[pos[int(pa[k, 0])], cen[pa[k, 1]]]

    # components over patch adjacency (contact tubes only)
    adj = r["adj"]
    keep = contact_tubes[adj[:, 0]] & contact_tubes[adj[:, 1]]
    A = sparse.coo_matrix((np.ones(keep.sum()), (adj[keep, 0], adj[keep, 1])), shape=(ntube, ntube))
    ncomp, comp = csgraph.connected_components(A, directed=False)
    nbr = [[] for _ in range(ntube)]
    for i, j in adj:
        nbr[i].append(j)
        nbr[j].append(i)
    rim = np.array([contact_tubes[t] and any(not contact_tubes[j] for j in nbr[t]) for t in range(ntube)])
    area = r["area"]
    comp_area = {int(c): float(area[(comp == c) & contact_tubes].sum()) for c in np.unique(comp[contact_tubes])}
    comp_n = {int(c): int(((comp == c) & contact_tubes).sum()) for c in comp_area}
    cen_mm = vmm[cen]

    def imbalance(t: int) -> float:
        js = [j for j in nbr[t] if not contact_tubes[j] and np.isfinite(th[j])]
        js = [j for j in js if np.linalg.norm(cen_mm[j] - cen_mm[t]) <= 2.0]
        return float(th[t] - np.median(th[js])) if js else float("nan")

    out = []
    for k, (a, b) in enumerate(keys):
        vox = pairs.get((a, b), []) + pairs.get((b, a), [])
        idx = unpad(np.array(vox))
        lab = labels[idx]
        l0 = 0
        for v in vox:
            for ax in range(3):
                for s in (-1, 1):
                    w = unpad(np.array([v + s * st[ax]]))
                    l0 += int(labels[w][0] == TP.OUTSIDE)
        frac0 = l0 / (6 * len(vox))
        pa_, pb_ = prof[a], prof[b]
        ok = np.isfinite(pa_) & np.isfinite(pb_)
        corr = float(np.corrcoef(pa_[ok], pb_[ok])[0, 1]) if ok.sum() >= 4 else float("nan")
        conf = []
        if frac0 > 0:
            conf.append("label0_in_gap")
        if max(abs(resid[a]), abs(resid[b])) > 2 * TE.THICK_SD:
            conf.append("thickness_outlier")
        if np.isfinite(corr) and corr < 0:
            conf.append("signature_mismatch")
        out.append({
            "a": a, "b": b, "root_mm": float(root[k]), "n_voxels": len(vox),
            "labels": {int(u): int((lab == u).sum()) for u in np.unique(lab)},
            "label0_frac_nbhd": frac0, "echo2_rel": float(np.mean(e2[idx]) / cmed),
            "signature_corr": corr, "profile_a": pa_.tolist(), "profile_b": pb_.tolist(),
            "thickness_mm": [float(th[a]), float(th[b])], "residual": [float(resid[a]), float(resid[b])],
            "imbalance_mm": [imbalance(a), imbalance(b)],
            "component": [int(comp[a]), int(comp[b])], "rim": [bool(rim[a]), bool(rim[b])],
            "in_region": [bool(r["in_region"][a]), bool(r["in_region"][b])],
            "conflicts": conf, "rule_only": bool(ctop[a] < 0.1 and ctop[b] < 0.1 and frac0 == 0),
            "first_step": None,
        })
    first = {}
    for ev in log["events"]["contact"]:
        first.setdefault((ev["a"], ev["b"]), ev["step"])
    for o in out:
        o["first_step"] = first.get((o["a"], o["b"]))

    # stops
    stops = []
    shell = (flat_o >= 0) & np.isfinite(flat_t) & (flat_t >= 0)
    sh_n = np.bincount(flat_o[shell], minlength=ntube)
    sh_0 = np.bincount(flat_o[shell], weights=(labels.ravel()[shell] == TP.OUTSIDE), minlength=ntube)
    last_stop = {}
    for ev in log["events"]["stop"]:
        last_stop[ev["tube"]] = ev["step"]
    for t, s in last_stop.items():
        stops.append({"tube": t, "step": s, "thickness_mm": float(th[t]), "residual": float(resid[t]),
                      "shell_label0_frac": float(sh_0[t] / max(sh_n[t], 1)), "coh_top": float(ctop[t]),
                      "contact": bool(contact_tubes[t]), "in_region": bool(r["in_region"][t]),
                      "stopped_at_end": bool(r["stopped"][t])})
    # retreats by step and term
    ret = {}
    for ev in log["events"]["retreat"]:
        d = ret.setdefault(ev["step"], {"n": 0, "dE_terms": np.zeros(len(TE.TERMS)), "n_terms": 0})
        d["n"] += 1
        if ev["dE_terms"] is not None:
            d["dE_terms"] += np.array(ev["dE_terms"])
            d["n_terms"] += 1
    retreats = [{"step": s, "n": d["n"], "dE_terms_sum": dict(zip(TE.TERMS, d["dE_terms"].tolist())),
                 "with_terms": d["n_terms"]} for s, d in sorted(ret.items())]
    summary = {
        "contact_pairs": len(out),
        "contact_pairs_in_region": int(sum(all(o["in_region"]) for o in out)),
        "contact_components": len(comp_area), "component_area_mm2": comp_area, "component_tubes": comp_n,
        "stops": len(stops), "tubes": int(ntube), "live_tubes": int(np.isfinite(th).sum()),
        "thickness_mm_region": np.nanpercentile(th[r["in_region"]], [10, 50, 90]).tolist(),
    }
    (run / "diagnostics.json").write_text(json.dumps(
        {"summary": summary, "contacts": out, "stops": stops, "retreats": retreats}, default=float))
    print(json.dumps(summary, default=float))


if __name__ == "__main__":
    main()
