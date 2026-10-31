"""The joint warp's model data for a saved result, for the correction tools (edit_priority.py,
edit_resolve.py). Rebuilds joint_levels.Data as joint_site.py does (shifts and weights between the
saved pairs, open-top anchors, ties with their lengths and facing factors) for the saved traces,
and caches it in <work>/edit_data_<tag>.npz. joint_site.py itself parses argv when imported, so
its steps 2-3 are repeated here; the ties are checked against the saved ones.

c0 (the conservation taper's pia): the saved c0 for a one-round result; for a result of several
rounds (tag _r<k>) the last round read its taper at the previous round's pia, which is not saved,
so the final pia stands in (the last two rounds moved it by a median 0.0 mm).
"""

import os
import re

import numpy as np
from scipy.spatial import cKDTree

import crossing_lib as X
import flatten_lib as FL
import joint_levels as JL

STEP = X.STEP_MM
OPEN, CROSS = 0, 1


def site_of(tag):
    return "M1S1" if tag.startswith("M1S1") else "V1V2_banks"


def load(work, tag, log=print):
    """Returns (D, J, Z): Data, the saved joint result (dict of arrays) and the column volume."""
    site = site_of(tag)
    Z = np.load(f"{work}/column_volume_{site}.npz")
    J = dict(np.load(f"{work}/joint_{tag}.npz"))
    Q = np.load(f"{work}/pia_{site}.npz")
    sub, pr = J["sub"], J["pairs"]
    cache = f"{work}/edit_data_{tag}.npz"
    if os.path.exists(cache):
        C = np.load(cache)
        S, W, tL, tw = C["S"], C["W"], C["tie_L"], C["tie_w"]
    else:
        keep = Q["keep"]
        fv = Z["fv"]
        F = fv[:, sub].astype(float)
        M_ = np.isfinite(F[0]) & np.isfinite(F[1])
        for k in range(2):
            F[k] /= np.nanstd(fv[k][keep])
        F = np.where(M_[None], F, 0.0)
        S = np.zeros((len(pr), F.shape[2]), np.float32)
        W = np.zeros_like(S)
        EN, NC, VA = np.zeros_like(S), np.zeros_like(S), np.zeros(S.shape, bool)
        for a0 in range(0, len(pr), 20000):
            p = pr[a0 : a0 + 20000]
            s_, ncc, en, va = FL.shifts(
                F[:, p[:, 0]], F[:, p[:, 1]], M_[p[:, 0]], M_[p[:, 1]], STEP
            )
            (
                S[a0 : a0 + 20000],
                NC[a0 : a0 + 20000],
                EN[a0 : a0 + 20000],
                VA[a0 : a0 + 20000],
            ) = s_, ncc, en, va
        W[:] = FL.pair_weights(NC, EN, VA, float(np.median(EN[VA])))
        tL, tw = ties_of(Z, sub, J["ties"], radius_of(tag))
        np.savez_compressed(cache, S=S, W=W, tie_L=tL, tie_w=tw)
        log(f"edit_data: rebuilt shifts for {len(pr):,} pairs -> {cache}")
    D = JL.Data(
        step=STEP,
        ns=Z["ns"][sub],
        top=J["top"],
        mu=Q["mu"][sub],
        pairs=pr,
        S=S,
        W=W,
        ties=J["ties"],
        tie_L=tL,
        tie_w=tw,
    )
    D.tau = JL.TAU
    D.c0 = J["z"][-1].copy() if re.search(r"_r\d+$", tag) else J["c0"]
    return D, J, Z


def radius_of(tag):
    """joint_site's default neighbour radius, 1.7 x the spacing in the tag."""
    return 1.7 * (0.16 if "_s0.16" in tag else 0.3)


def ties_of(Z, sub, saved, radius):
    """joint_site's ties (each crossing trace to the facing subsampled root nearest its far end):
    lengths and facing factors, in the saved order (checked)."""
    tv, ns, cls, v, nrm, vox = Z["tv"], Z["ns"], Z["cls"], Z["v"], Z["nrm"], Z["vox"]
    root, nr = v[tv] * vox, nrm[tv]
    tree = cKDTree(root[sub])
    pos, far_wm = Z["pos"], Z["H_far WM"]
    ties, tL, tw = [], [], []
    for i in np.flatnonzero(cls[sub] == CROSS):
        t = sub[i]
        d, j = tree.query(pos[t, ns[t] - 1].astype(float) * vox)
        if d <= radius:
            dot = float(nr[sub[j]] @ nr[t])
            if dot < -0.3 and j != i:
                ties.append((i, j))
                tL.append(float(far_wm[t]))
                tw.append(-dot)
    ties = np.array(ties, int).reshape(-1, 2)
    assert np.array_equal(ties, saved), "rebuilt ties differ from the saved ones"
    return np.array(tL), np.array(tw)


def point_traces(Z, sub, P=None, within=0.3):
    """As joint_site.score: each subsampled trace whose nearest sample to the marked points
    (ITK-SNAP voxels, 0-based box coordinates) lies within `within` mm goes to that point, at that
    sample's depth (mm). Returns per point (trace indices into sub, depths)."""
    pos, ns, vox = Z["pos"], Z["ns"], Z["vox"]
    P = Z["P"] if P is None else np.atleast_2d(P)
    tree = cKDTree((P * vox).astype(float))
    hit = [([], []) for _ in range(len(P))]
    for i, t in enumerate(sub):
        d, k = tree.query(pos[t, : ns[t]].astype(float) * vox)
        j = int(np.argmin(d))
        if d[j] <= within:
            hit[int(k[j])][0].append(i)
            hit[int(k[j])][1].append(j * STEP)
    return [(np.array(a, int), np.array(b, float)) for a, b in hit]


def point_errors(pia, hits):
    """Per point: median |pia - point depth| over its traces (nan where none), as joint_site.score."""
    return np.array(
        [np.median(np.abs(pia[i] - d)) if i.size else np.nan for i, d in hits]
    )


if __name__ == "__main__":
    import json
    import sys
    import time

    import baillarger.surfcut as WC

    work, tag = sys.argv[1], sys.argv[2]
    t0 = time.time()
    D, J, Z = load(work, tag)
    # check: the fine model centred on the saved levels, its energy there against the saved one
    z, cert = J["z"], J["cert"]
    M, o = JL.build(D, 0.04, z, np.where(cert, 0.16, 0.48))
    lab = np.round(z.ravel() / 0.04).astype(int) - np.asarray(M.o)
    js = json.load(open(f"{work}/joint_{tag}.json"))
    print(
        dict(
            tag=tag,
            seconds=round(time.time() - t0, 1),
            energy_here=round(WC.energy(M, lab), 4),
            energy_saved_fine=round(js["history"][-1]["energy"], 4),
        )
    )
