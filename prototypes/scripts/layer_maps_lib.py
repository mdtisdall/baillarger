"""Where layers differ: seismic-attribute maps from the joint warp (plan: reports/Crossing paths and
contacts from laminar profiles.md, *Where layers differ*). Pure functions; the driver is
layer_maps.py, the exact-answer tests test_layer_maps.py.

The warp of each trace is z_v(tau) (mm from the root; tau 0 at the WM, 1 at the pia) on a grid tg.
- Isopachs: the thickness of a tau interval, z(tau_b) - z(tau_a) = the integral of dz/dtau, in mm.
- Curvature trend: thickness t = t0 exp(a H) with H the white-mesh mean curvature at the root (the
  log-linear trend of pia_horizon.py); removed by t exp(-a (H - H_ref)), which keeps mm and zeros.
- Pinch-out attribute: an interval's fraction of the whole thickness and its lateral gradient, on
  the traces' roots (a plane fit through each root, in its tangent plane).
- Facies: profiles at constant tau, clustered by a Gaussian mixture (soft memberships), k chosen
  by BIC.
"""

import numpy as np
from scipy.spatial import cKDTree


def monotone(zf):
    """Each warp made non-decreasing (running maximum): the fill is not forced to be monotone."""
    return np.maximum.accumulate(np.asarray(zf, float), axis=1)


def depth_at(zf, tg, tau):
    """z_v(tau) for every trace (N,) or (N, len(tau)), linear between grid levels."""
    zf = monotone(zf)
    t = np.atleast_1d(np.asarray(tau, float))
    j = np.clip(np.searchsorted(tg, t, side="right") - 1, 0, tg.size - 2)
    f = (t - tg[j]) / (tg[j + 1] - tg[j])
    out = zf[:, j] * (1 - f) + zf[:, j + 1] * f
    return out[:, 0] if np.ndim(tau) == 0 else out


def isopachs(zf, tg, edges):
    """Thickness (N, len(edges) - 1) in mm of each tau interval [edges[k], edges[k + 1]]."""
    return np.diff(depth_at(zf, tg, np.asarray(edges, float)), axis=1)


def curvature_trend(t, H, pairs, fit=None, floor=0.1):
    """The slope a of log t = m(x) + a H, from neighbouring traces: least squares of
    log t_j - log t_i = a (H_j - H_i) over `pairs` where both thicknesses exceed `floor` (a layer
    pinched out has no log) and both are in `fit` (default: all). A plain fit of log t on H, as
    pia_horizon.py's for the whole thickness, takes any areal change of a layer that correlates
    with curvature as the trend; differences between near neighbours cancel what changes slowly
    along the surface. Returns (a, n_pairs)."""
    ok = np.isfinite(t) & np.isfinite(H) & (t > floor)
    if fit is not None:
        ok &= fit
    p = pairs[ok[pairs[:, 0]] & ok[pairs[:, 1]]]
    if len(p) < 3:
        return 0.0, int(len(p))
    dH = H[p[:, 1]] - H[p[:, 0]]
    dl = np.log(t[p[:, 1]]) - np.log(t[p[:, 0]])
    return float(dH @ dl / (dH @ dH)), int(len(p))


def detrend(t, H, a, H_ref=0.0):
    """The thickness with the curvature trend removed, in mm: t exp(-a (H - H_ref))."""
    return t * np.exp(-a * (H - H_ref))


def neighbour_pairs(root, nrm, radius, ndot=0.7):
    """Pairs (i, j), i < j, of roots within `radius` mm whose normals agree (dot > ndot)."""
    pr = cKDTree(root).query_pairs(radius, output_type="ndarray")
    if len(pr) == 0:
        return pr.reshape(0, 2)
    return pr[(nrm[pr[:, 0]] * nrm[pr[:, 1]]).sum(1) > ndot]


def lateral_gradient(val, root, nrm, pairs, min_pairs=3):
    """The gradient (N, 3) of a value on the roots, in its tangent plane: per root, least squares of
    val_j - val_i = g . d_ij over its neighbours, with d_ij projected onto i's tangent plane and
    g . n_i = 0. Exact for a linear field on a plane. nan where fewer than min_pairs neighbours have
    a value."""
    n = len(root)
    val = np.asarray(val, float)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    A = np.zeros((n, 3, 3))
    b = np.zeros((n, 3))
    cnt = np.zeros(n, int)
    ok = np.isfinite(val[pairs[:, 0]]) & np.isfinite(val[pairs[:, 1]])
    p = pairs[ok]
    for i, j in ((p[:, 0], p[:, 1]), (p[:, 1], p[:, 0])):
        d = root[j] - root[i]
        d -= (d * nrm[i]).sum(1, keepdims=True) * nrm[i]
        dv = val[j] - val[i]
        np.add.at(A, i, d[:, :, None] * d[:, None, :])
        np.add.at(b, i, d * dv[:, None])
        np.add.at(cnt, i, 1)
    # g . n = 0 as a stiff extra equation, scaled to the neighbourhood
    # (and a tiny ridge, for neighbours on one line); roots with too few neighbours: nan
    s = np.trace(A, axis1=1, axis2=2)[:, None, None]
    A += s * (nrm[:, :, None] * nrm[:, None, :] + 1e-9 * np.eye(3))
    g = np.full((n, 3), np.nan)
    use = (cnt >= min_pairs) & np.isfinite(val)
    g[use] = np.linalg.solve(A[use], b[use][:, :, None])[:, :, 0]
    return g


def pinch_out(thick, total, root, nrm, pairs, min_total=0.2):
    """Per interval k (columns of thick): its fraction of the whole thickness r = t_k / total, and
    |grad r| on the roots (per mm). nan where total < min_total. Where a layer pinches out, r falls
    to 0 and |grad r| peaks on the taper."""
    r = np.where(total[:, None] >= min_total, thick / total[:, None], np.nan)
    gr = np.stack([np.linalg.norm(lateral_gradient(r[:, k], root, nrm, pairs), axis=1)
                   for k in range(r.shape[1])], 1)
    return r, gr


def smooth_on_roots(val, pairs, w):
    """Weighted mean of each value (N, ...) with its neighbours' (pair weights w), itself weight 1."""
    val = np.asarray(val, float)
    num = val.copy()
    den = np.ones(len(val))
    shp = (-1,) + (1,) * (val.ndim - 1)
    for i, j in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        np.add.at(num, i, w.reshape(shp) * val[j])
        np.add.at(den, i, w)
    return num / den.reshape(shp)


def fold_transitions(means, tol=0.25, amin=0.1):
    """Components whose mean is a blend of two others': for each component c (most blend-like
    first), the closest point of each segment between two remaining end-members a, b; c is a
    transition if its distance to it is under tol |a - b| with the blend weight in [amin, 1 - amin].
    Returns (ends: indices of the end-members, M (k, len(ends)): each component's share of each
    end-member; a transition splits by its blend weight), and the transitions as (c, a, b, alpha)."""
    k = len(means)
    ends, trans = list(range(k)), []
    while len(ends) > 2:
        best = None
        for c in ends:
            for a in ends:
                for b in ends:
                    if len({a, b, c}) < 3 or a > b:
                        continue
                    ab = means[a] - means[b]
                    al = float(np.clip((means[c] - means[b]) @ ab / (ab @ ab), 0, 1))
                    rel = np.linalg.norm(means[c] - (al * means[a] + (1 - al) * means[b])) / np.linalg.norm(ab)
                    if amin <= al <= 1 - amin and rel < tol and (best is None or rel < best[0]):
                        best = (rel, c, a, b, al)
        if best is None:
            break
        _, c, a, b, al = best
        ends.remove(c)
        trans.append((c, a, b, al))
    M = np.zeros((k, len(ends)))
    for i, e in enumerate(ends):
        M[e, i] = 1
    for c, a, b, al in reversed(trans):  # a parent of a later fold is itself an end-member here
        M[c] = al * M[a] + (1 - al) * M[b]
    return ends, M, trans


def facies(X, ks=(1, 2, 3, 4, 5, 6), n_pca=8, seed=0, fit_max=20000):
    """Soft facies of profiles X (N, L) (rows with nan are left out: memberships nan): PCA to
    n_pca components, Gaussian mixtures (full covariance) for each k, k chosen by the lowest ICL
    (BIC + 2 x the memberships' entropy: Biernacki et al. 2000; it penalizes overlapping components;
    BIC is reported too). A gradual border is a continuum of blends, which a mixture can give a
    component of its own: such transition components are folded into the two end-members they
    blend (fold_transitions). Returns dict(k, bic, icl {k: value}, memberships (N, k), labels (N,),
    means (k, L) in X's units, explained (the PCA's variance fractions), ends, transitions, and
    end_memberships (N, len(ends)): the memberships with transitions split). Components are ordered
    by their mean of PC 1, so labels are reproducible."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    ok = np.isfinite(X).all(1)
    Xo = X[ok]
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(Xo), min(fit_max, len(Xo)), replace=False)
    pca = PCA(n_components=min(n_pca, X.shape[1]), random_state=seed).fit(Xo[fit_idx])
    Y = pca.transform(Xo)
    gms, bic, icl = {}, {}, {}
    for k in ks:
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed).fit(Y[fit_idx])
        gms[k], bic[k] = gm, float(gm.bic(Y[fit_idx]))
        icl[k] = bic[k] + 2 * float(entropy(gm.predict_proba(Y[fit_idx])).sum())
    k = min(icl, key=icl.get)
    gm = gms[k]
    order = np.argsort(gm.means_[:, 0])
    P = gm.predict_proba(Y)[:, order]
    mem = np.full((len(X), k), np.nan)
    mem[ok] = P
    lab = np.full(len(X), -1)
    lab[ok] = P.argmax(1)
    mu = gm.means_[order]
    means = pca.inverse_transform(mu)
    ends, M, trans = fold_transitions(means) if k > 2 else (list(range(k)), np.eye(k), [])
    # a trace's share of a transition is split by its own blend weight: its projection onto the
    # segment between the two end-members (PCA space), not the component mean's
    R = np.broadcast_to(M, (len(Y), *M.shape)).copy()
    for c, a, b, _ in reversed(trans):
        ab = mu[a] - mu[b]
        al = np.clip((Y - mu[b]) @ ab / (ab @ ab), 0, 1)[:, None]
        R[:, c] = al * R[:, a] + (1 - al) * R[:, b]
    em = np.full((len(X), len(ends)), np.nan)
    em[ok] = np.einsum("nk,nke->ne", P, R)
    return dict(k=k, bic=bic, icl=icl, memberships=mem, labels=lab, means=means,
                explained=pca.explained_variance_ratio_, ends=ends, transitions=trans,
                end_memberships=em)


def entropy(mem):
    """Membership entropy (nats) per row: 0 for a pure trace, log k at an even mix."""
    p = np.clip(mem, 1e-12, 1)
    return -(p * np.log(p)).sum(1)


def facies_set_aside(X, small=0.05, **kw):
    """facies(X), then again without the traces of small end-members: an end-member that holds
    (by the largest end-membership) under `small` of the traces is a local anomaly (a few hundred
    traces of odd tissue or a failed warp) that would otherwise take the first principal
    components. Returns (the second fit, the first, set_aside (N,) bool)."""
    f0 = facies(X, **kw)
    em = f0["end_memberships"]
    ok = np.isfinite(em).all(1)
    own = np.full(len(X), -1)
    own[ok] = em[ok].argmax(1)
    cnt = np.bincount(own[ok], minlength=em.shape[1])
    tiny = np.flatnonzero(cnt < small * ok.sum())
    aside = np.isin(own, tiny) & ok
    if not aside.any():
        return f0, f0, aside
    X1 = np.where(aside[:, None], np.nan, X)
    return facies(X1, **kw), f0, aside
