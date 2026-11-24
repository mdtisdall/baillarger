"""A local re-solve of the joint warp's levels after an edit, and the edit reach (plan:
reports/Crossing paths and contacts from laminar profiles.md, *Correction tools*).

An edit: a marked point (ITK-SNAP voxel, 0-based box coordinates as in column_volume) becomes a
strong unary w_edit |z - d*| on the pia of every subsampled trace passing within 0.3 mm of it
(d*: the depth of the trace's sample nearest the point; edit_data.point_traces). Every variable
outside a neighbourhood of radius R along the roots (geodesic on the white surface's mesh from the
edited traces' roots) is fixed at its current level, and only the neighbourhood is solved:

- local_solve: the sub-model of the neighbourhood's traces and the ring of traces one term away
  (pairs, ties), the ring fixed by baillarger.surfcut.reduce (one label per ring variable: every
  term between a free and a fixed variable becomes a unary on the free one; terms between fixed
  variables are constants and drop out, so this is the full model reduced, exactly); then
  qpbo_pruned from the current levels, windows re-centred where hit at an edge (as
  joint_levels.solve_pass).
- main: the edit reach. For each site and each of `--draws` random orders of the marked points:
  seed the first k (k = 3, 6, 12) one edit at a time from the saved result, for R = 1, 2, 4 mm,
  and score the pia on the held-out points (those not among the first 12, the same for every k),
  with k = 0 the saved result. Time per edit.
Writes <work>/edit_reach_<tag>.json.

Usage: edit_resolve.py <work> <tag> [--draws N]
Needs baillarger.surfcut from PR #31 only for edit_priority, not here.
"""

import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

import baillarger.surfcut as WC
import joint_levels as JL

W_EDIT = 50.0  # per mm of pia away from the point: 0.1 mm costs the anchor's cap (5)


def neighbourhood(G, root_vertex, seeds, R):
    """Traces (indices into root_vertex) whose root lies within R mm of a seed trace's root along
    the mesh graph G."""
    if not len(seeds):
        return np.zeros(len(root_vertex), bool)
    d = dijkstra(G, indices=np.unique(root_vertex[seeds]), min_only=True, limit=R)
    return np.isfinite(d[root_vertex])


def restrict(D, T):
    """Data on traces T (sorted indices): pairs and ties with both ends in T."""
    loc = -np.ones(D.ns.size, int)
    loc[T] = np.arange(T.size)
    pk = (loc[D.pairs[:, 0]] >= 0) & (loc[D.pairs[:, 1]] >= 0)
    tk = (loc[D.ties[:, 0]] >= 0) & (loc[D.ties[:, 1]] >= 0) if D.ties.size else np.zeros(0, bool)
    S = JL.Data(step=D.step, ns=D.ns[T], top=D.top[T], mu=D.mu[T], pairs=loc[D.pairs[pk]].reshape(-1, 2),
                S=D.S[pk], W=D.W[pk], ties=loc[D.ties[tk]].reshape(-1, 2) if D.ties.size else D.ties,
                tie_L=D.tie_L[tk] if D.ties.size else D.tie_L, tie_w=D.tie_w[tk] if D.ties.size else D.tie_w)
    S.c0 = D.c0[T]
    S.tau = D.tau
    return S


def ring_of(D, free):
    """Traces one term (a pair or a tie) away from the free ones, not free themselves."""
    ring = np.zeros(free.size, bool)
    for e in (D.pairs, D.ties):
        if e.size:
            a, b = e[:, 0], e[:, 1]
            ring[b[free[a]]] = True
            ring[a[free[b]]] = True
    return ring & ~free


def local_solve(D, z, free, h=0.04, half=0.48, prm=JL.Params(), edits=None, w_edit=W_EDIT, recentre=3,
                log=None):
    """z: (L, n) current levels (mm, on the h grid); free: (n,) traces to solve; edits: {trace:
    target pia (mm)}, every edit made so far (those on free traces enter; the others are fixed
    anyway). Returns the new levels (L, n) and info."""
    t0 = time.time()
    edits = {int(t): d for t, d in (edits or {}).items() if free[int(t)]}
    L = z.shape[0]
    ring = ring_of(D, free)
    T = np.flatnonzero(free | ring)
    Ds = restrict(D, T)
    fr = free[T]
    nT = T.size
    zs = z[:, T].copy()
    centre = zs.copy()
    hw = np.where(fr[None], half, 0.0) * np.ones((L, 1))
    loc = {int(t): i for i, t in enumerate(T)}
    for t, d in edits.items():
        i = loc[int(t)]
        hw[-1, i] = max(hw[-1, i], abs(d - zs[-1, i]) + 2 * h)
    fixed_var = np.repeat(~fr[None], L, 0).ravel()
    hist = []
    for it in range(recentre + 1):
        M, o = JL.build(Ds, h, centre, hw, prm)
        Ko = np.asarray(M.K)
        for t, d in edits.items():
            i = (L - 1) * nT + loc[int(t)]
            dep = h * (M.o[i] + np.arange(Ko[i]))
            M.unary[i] = M.unary[i] + w_edit * np.abs(dep - d)
        lab = np.clip(np.round(zs.ravel() / h).astype(int) - np.asarray(M.o), 0, Ko - 1)
        R, fv, _ = WC.reduce(M, fixed_var, lab)
        r = WC.qpbo_pruned(R, start_labels=lab[fv])
        lab[fv] = r.labels
        zs = h * (np.asarray(M.o) + lab).reshape(L, nT)
        hits = JL.edge_hits(M, lab).reshape(L, nT) & fr[None]
        hist.append(dict(edge_hits=int(hits.sum()), certified=float(np.mean(r.certified)), free_vars=int(fv.size)))
        if log:
            log(f"local: try {it}: {fv.size:,} free variables, certified {np.mean(r.certified):.1%}, "
                f"edge hits {int(hits.sum())}")
        if not hits.any():
            break
        centre = np.where(hits, zs, centre)
    out = z.copy()
    out[:, T[fr]] = zs[:, fr]
    return out, dict(free_traces=int(fr.sum()), ring_traces=int((~fr).sum()), seconds=time.time() - t0,
                     tries=hist)


def edits_of(hits, p):
    """The edit of marked point p: {trace: depth} for its traces."""
    i, d = hits[p]
    return {int(t): float(dd) for t, dd in zip(i, d)}


def main(work, tag, draws=3, ks=(3, 6, 12), Rs=(1.0, 2.0, 4.0), h=0.04):
    import json

    import edit_data as ED
    from volume_views import mesh_graph

    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    D, J, Z = ED.load(work, tag, log=say)
    sub, z0 = J["sub"], J["z"]
    G = mesh_graph(Z["v"], Z["f"], Z["vox"])
    rv = Z["tv"][sub]
    hits = ED.point_traces(Z, sub)
    pts = np.array([p for p in range(len(hits)) if hits[p][0].size])
    err0 = ED.point_errors(z0[-1], hits)
    signed = lambda pia, ps: round(float(np.median([np.median(pia[hits[q][0]] - hits[q][1]) for q in ps])), 3)  # noqa: E731
    say(f"{len(pts)} scored points; saved pia: median {np.nanmedian(err0[pts]):.3f} mm")
    runs = []
    for dr in range(draws):
        order = np.random.default_rng(dr).permutation(pts)
        held = order[max(ks):]
        base = dict(draw=dr, k=0, held_out=[int(p + 1) for p in held],
                    held_median=round(float(np.median(err0[held])), 3),
                    held_over_0p3=int(np.sum(err0[held] > 0.3)), held_signed_median=signed(z0[-1], held))
        for R in Rs:
            runs.append(dict(base, R=R))
            z = z0.copy()
            times, free_n, made = [], [], {}
            for j, p in enumerate(order[:max(ks)]):
                ed = edits_of(hits, p)
                made.update(ed)
                seeds = np.array(list(ed), int)
                free = neighbourhood(G, rv, seeds, R)
                z, info = local_solve(D, z, free, h=h, edits=made)
                times.append(info["seconds"])
                free_n.append(info["free_traces"])
                if j + 1 in ks:
                    e = ED.point_errors(z[-1], hits)
                    seeded = order[:j + 1]
                    rest = np.setdiff1d(pts, seeded)
                    runs.append(dict(draw=dr, k=j + 1, R=R, held_median=round(float(np.median(e[held])), 3),
                                     held_over_0p3=int(np.sum(e[held] > 0.3)),
                                     held_signed_median=signed(z[-1], held),
                                     all_unseeded_median=round(float(np.median(e[rest])), 3),
                                     seeded_median=round(float(np.median(e[seeded])), 3),
                                     held_changed=int(np.sum(np.abs(e[held] - err0[held]) > 1e-9)),
                                     held_per_point={int(q + 1): [round(float(err0[q]), 2), round(float(e[q]), 2)]
                                                     for q in held},
                                     seconds_per_edit_median=round(float(np.median(times)), 2),
                                     seconds_per_edit_max=round(float(np.max(times)), 2),
                                     free_traces_median=int(np.median(free_n))))
                    say(f"draw {dr} R {R} k {j + 1}: held-out median {runs[-1]['held_median']} "
                        f"(saved {base['held_median']}), over 0.3: {runs[-1]['held_over_0p3']} "
                        f"(saved {base['held_over_0p3']}); {np.median(times):.1f} s/edit, "
                        f"{int(np.median(free_n))} free traces")
    out = dict(tag=tag, w_edit_per_mm=W_EDIT, h=h, half=0.48, points=int(len(pts)), runs=runs,
               seconds=round(time.time() - t0, 1))
    json.dump(out, open(f"{work}/edit_reach_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("tag")
    ap.add_argument("--draws", type=int, default=3)
    A = ap.parse_args()
    sys.path.insert(0, A.work)
    main(A.work, A.tag, A.draws)
