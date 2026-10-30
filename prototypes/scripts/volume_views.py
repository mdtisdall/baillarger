"""Views shared by the column-volume scripts: the marked planes and the traces of a section
ordered along the white surface (as column_volume.sections)."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import crossing_lib as X
import mesh_smoothing as MS


def mesh_graph(v, f, vox):
    E = MS.edges(f)
    w = np.linalg.norm((v[E[:, 0]] - v[E[:, 1]]) * vox, axis=1)
    G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v),) * 2).tocsr()
    return G.maximum(G.T).tocsr()


def planes_of(P):
    ax = int(np.argmin([len(np.unique(np.round(P[:, a]))) for a in range(3)]))
    ks = np.unique(np.round(P[:, ax]).astype(int))
    cnt = [(k, int(np.sum(np.abs(P[:, ax] - k) <= 2))) for k in ks]
    chosen = []
    for k, _ in sorted(cnt, key=lambda t: -t[1]):
        if all(abs(k - c) > 4 for c in chosen):
            chosen.append(k)
        if len(chosen) == 3:
            break
    return ax, chosen


def section_traces(Z, G, ax, k, Pk, trace_of, tv_set, vox):
    """As column_volume.sections: per bank facing the points on plane k, the traces ordered along
    the white surface (trace indices into the column volume, distances in mm from the anchor)."""
    v, nrm = Z["v"], Z["nrm"]
    band = np.flatnonzero(np.abs(v[:, ax] - k) < 0.75)
    c0 = Pk[np.argmin(np.linalg.norm(Pk - np.median(Pk, 0), axis=1))]
    facing = band[(((c0 - v[band]) * vox) * nrm[band]).sum(1) > 0]
    a1 = facing[np.argmin(np.linalg.norm((c0 - v[facing]) * vox, axis=1))]
    g1 = dijkstra(G, indices=a1, limit=30.0)
    other = facing[~(g1[facing] < 5.0)]
    anchors = [a1] + (
        [other[np.argmin(np.linalg.norm((c0 - v[other]) * vox, axis=1))]]
        if other.size
        else []
    )
    _, comp = connected_components(G[band][:, band], directed=False)
    out = []
    for anc in anchors:
        piece = band[comp == comp[np.searchsorted(band, anc)]]
        o, dist = X.order_along(piece, G)
        da = dist[np.flatnonzero(o == anc)[0]]
        sel = (np.abs(dist - da) <= 8.0) & tv_set[o]
        out.append((trace_of[o[sel]], dist[sel] - da))
    return out
