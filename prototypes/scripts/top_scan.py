"""Coarse eigenvector rays over the whole hemisphere, one level-1 superblock
at a time. The same tests as the site's coarse run (coarse_regions.py), with
the constants of that run. Writes one npz per superblock.

usage: top_scan.py OUT [--only I,J,K]   (OUT holds top_tensor.py's tensor; inputs: top_paths.py)
SAVE_DIAG=1: write every superblock's ray paths to OUT/diag/ for top_features.py.

DEVICE: numpy (reference), cpu, cuda or mps, for tracing, profiles and
correlations. WORKERS: superblocks run at once, each in its own process.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).parent))
import bent_confidence as C  # noqa: E402
import coarse_kernels as K  # noqa: E402
import torch  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import multiprocessing as mp  # noqa: E402
from baillarger.backend import device as get_device  # noqa: E402

from baillarger import orient as O  # noqa: E402
from baillarger import rays as RY  # noqa: E402
from baillarger.orient_noise import superblocks  # noqa: E402
from baillarger.timing import peak_rss_gb  # noqa: E402

import top_paths as TP  # noqa: E402

DSEG, SURF, ECHOES = TP.dseg(), TP.surf(), TP.echoes()

# ---- constants: the site run's, set before the site run
STEP = 0.04
import os
TRACE_STEP = float(os.environ.get('TRACE_STEP', '0.16'))
EARLY_MM = float(os.environ.get('EARLY_MM', '1.0'))  # trace this far past the first outside sample; 0: off
DEV = get_device(os.environ.get('DEVICE', 'numpy'))
FDT = torch.float32 if DEV.name == 'mps' else torch.float64
WORKERS = int(os.environ.get('WORKERS', '1'))
PROF_K = int(os.environ.get('PROF_K', '1'))  # profile samples every PROF_K x 0.04 mm
GEOM_FAST = int(os.environ.get('GEOM_FAST', '0'))
S_MAX = 8.0
L1_MIN_C, L1_MAX_TURN = 0.3, 45.0
ALIGN_MIN_C = 0.7
TREND_MM = 0.5
FOLD_DEPTHS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
TRIM = 0.3
SHIFTS = np.arange(-4, 5)
SILENT_OWN = 0.85
DISAGREE = 0.10
GEOM_FAIL = 0.5
CELL_VOX = 2  # coarse cell: 2 voxels a side (about 0.32 mm)
HALO_MM = float(__import__('os').environ.get('HALO_MM', '1.0'))  # roots round the superblock, for the coarse neighbours
REACH_MM = S_MAX + 0.5  # the box read round the roots
LEVEL = 1
# outside medium: label 0 and the lateral ventricle (ex vivo, the ventricle holds outside medium)
OUTSIDE = (0, 43)
REASONS = ("pass", "pass, silent", "no top", "folded", "disagrees", "geometry")

s_grid = np.arange(int(round(S_MAX / STEP)) + 1) * STEP


def coarse_pick(vox, normal, vertex):
    """Indices of the chosen roots: one per (2-voxel cell, dominant signed
    axis of the normal), the root nearest the cell centre, ties by vertex."""
    cell = np.floor(vox / CELL_VOX).astype(np.int64)
    ax = np.argmax(np.abs(normal), 1)
    sg = (normal[np.arange(len(ax)), ax] > 0).astype(np.int64)
    key = np.c_[cell, 2 * ax + sg]
    d = np.linalg.norm(vox - (cell + 0.5) * CELL_VOX, axis=1)
    order = np.lexsort((vertex, d, key[:, 3], key[:, 2], key[:, 1], key[:, 0]))
    k = key[order]
    first = np.r_[True, np.any(k[1:] != k[:-1], axis=1)]
    return np.sort(order[first])


TIMES = {}
TRI_DIAG = {}
DIAG = {}


def tick(name, t0=[None]):
    now = time.perf_counter()
    if t0[0] is not None and name:
        TIMES[name] = TIMES.get(name, 0.0) + now - t0[0]
    t0[0] = now


def run_superblock(sb, s, rs, J, inv, A, filled, cortex):
    tick(None)
    roots = rs.roots(sb, LEVEL, HALO_MM)
    n = len(roots.vertex)
    if n == 0:
        return None
    R = roots.root.astype(np.float64)
    Nn = roots.normal.astype(np.float64)
    vox = R @ inv[:3, :3].T + inv[:3, 3]
    pick = coarse_pick(vox, Nn, roots.vertex)
    tick('roots and pick')
    # the fine mesh in the call, for the Voronoi cells
    F = rs.faces(sb, LEVEL, HALO_MM)
    vo = np.argsort(roots.vertex)
    vs = roots.vertex[vo]
    if len(F):
        jj = np.clip(np.searchsorted(vs, F), 0, n - 1)
        Fi = np.where(vs[jj] == F, vo[jj], -1)
    else:
        Fi = np.zeros((0, 3), np.int64)
    Fi = Fi[(Fi >= 0).all(1)]
    E = np.unique(np.sort(np.r_[Fi[:, [0, 1]], Fi[:, [1, 2]], Fi[:, [0, 2]]], 1), axis=0)
    L = np.linalg.norm(R[E[:, 0]] - R[E[:, 1]], axis=1)
    G = coo_matrix((np.r_[L, L], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    _, _, src = dijkstra(G, indices=pick, min_only=True, return_predecessors=True)
    pos = np.full(n, -1)
    pos[pick] = np.arange(len(pick))
    cell = np.where(src >= 0, pos[np.maximum(src, 0)], -1)
    ce = np.c_[cell[E[:, 0]], cell[E[:, 1]]]
    ce = ce[(ce[:, 0] != ce[:, 1]) & (ce >= 0).all(1)]
    ce = np.unique(np.sort(ce, 1), axis=0)
    fa = 0.5 * np.linalg.norm(np.cross(R[Fi[:, 1]] - R[Fi[:, 0]], R[Fi[:, 2]] - R[Fi[:, 0]]), axis=1)
    area = np.zeros(n)
    for k in range(3):
        np.add.at(area, Fi[:, k], fa / 3)
    tick('mesh graph and Voronoi')
    m = len(pick)
    carea = np.bincount(cell[(cell >= 0) & roots.owned], weights=area[(cell >= 0) & roots.owned], minlength=m)
    # the box read round the chosen roots
    Rc, Nc = R[pick], Nn[pick]
    vc = vox[pick]
    rv = REACH_MM / np.asarray(s.voxel_mm)
    lo = np.maximum(np.floor(vc.min(0) - rv).astype(int), 0)
    hi = np.minimum(np.ceil(vc.max(0) + rv).astype(int) + 1, s.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    tick('area')
    T = np.asarray(J[(slice(None), *box)], np.float64)
    lab = np.asarray(s.labels.read(box))
    ech, _ = s.read(box)
    vm = np.asarray(s.voxel_mm, np.float64)
    to_box = lambda x: (x @ inv[:3, :3].T + inv[:3, 3]) - lo  # noqa: E731
    tick('read field, labels, echoes')
    D = A[:3, :3] / vm

    def samp(vol, x, order=1):
        return map_coordinates(vol, to_box(x).T, order=order, mode="nearest", cval=np.nan)

    ns = len(s_grid)
    out0 = np.isin(lab, OUTSIDE).astype(np.float64)
    Ainv3, bvec = inv[:3, :3], inv[:3, 3] - lo
    if DEV.is_numpy:
        P, share = K.trace_np(T, out0, Rc, Nc, Ainv3, bvec, D, S_MAX, TRACE_STEP, EARLY_MM, s_grid)
    else:
        Tt = torch.as_tensor(T, dtype=FDT, device=DEV.torch)
        Ot = torch.as_tensor(out0[None], dtype=FDT, device=DEV.torch)
        P, share = K.trace_t(Tt, Ot, Rc, Nc, Ainv3, bvec, D, S_MAX, TRACE_STEP, EARLY_MM, s_grid)
        del Tt, Ot
    TIMES['traced steps (share of full)'] = share
    tick('trace')
    flat = P.reshape(-1, 3)
    f0 = samp(out0, flat).reshape(m, ns)
    top = np.full(m, np.nan)
    pos_ = f0 > 0.5
    anyp = pos_.any(1)
    k = np.argmax(pos_, 1)
    k0 = np.maximum(k - 1, 0)
    rr = np.arange(m)
    fa_, fb_ = f0[rr, k0] - 0.5, f0[rr, k] - 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        tt = np.clip(np.where(fb_ != fa_, -fa_ / (fb_ - fa_), 0.0), 0, 1)
    top[anyp] = (s_grid[k0] + tt * STEP)[anyp]
    top[anyp & (k == 0)] = 0.0
    below = C.below_top(top, s_grid)
    tick('top')
    # geometry
    if GEOM_FAST:
        # crop to the roots +- (largest finite top + 1 mm), float32 EDTs, and the
        # outward normal only at the top points (central differences)
        tmax = float(np.nanmax(top)) if np.isfinite(top).any() else 1.0
        rv2 = (tmax + 1.0) / vm
        lo2 = np.maximum(np.floor(vc.min(0) - rv2).astype(int) - lo, 0)
        hi2 = np.minimum(np.ceil(vc.max(0) + rv2).astype(int) + 1 - lo, np.array(lab.shape))
        cb = tuple(slice(int(a_), int(b_)) for a_, b_ in zip(lo2, hi2))
        labc = lab[cb]
        dist0 = np.zeros(lab.shape, np.float32)
        distw = np.zeros(lab.shape, np.float32)
        dist0[cb] = distance_transform_edt(~np.isin(labc, OUTSIDE), sampling=vm)
        distw[cb] = distance_transform_edt(~np.isin(labc, filled), sampling=vm)
        sm = np.zeros(lab.shape, np.float32)
        sm[cb] = gaussian_filter(out0[cb].astype(np.float32), 1.0)
        okt = np.flatnonzero(np.isfinite(top))
        kq = np.clip(np.round(top[okt] / STEP).astype(int), 1, ns - 1)
        tp = P[okt, kq]
        qv = to_box(tp)
        gv = np.zeros((len(okt), 3))
        for ax_ in range(3):
            e_ = np.zeros(3)
            e_[ax_] = 1.0
            gv[:, ax_] = 0.5 * (map_coordinates(sm, (qv + e_).T, order=1, mode="nearest") - map_coordinates(sm, (qv - e_).T, order=1, mode="nearest"))
        gmm = gv @ np.linalg.inv(A[:3, :3])
        tvec = P[okt, kq] - P[okt, kq - 1]
        tvec /= np.maximum(np.linalg.norm(tvec, axis=1, keepdims=True), 1e-12)
        gmm /= np.maximum(np.linalg.norm(gmm, axis=1, keepdims=True), 1e-12)
        arr = np.full(m, np.nan)
        arr[okt] = np.degrees(np.arccos(np.clip((tvec * gmm).sum(1), -1, 1)))
    else:
        dist0 = distance_transform_edt(~np.isin(lab, OUTSIDE), sampling=vm)
        distw = distance_transform_edt(~np.isin(lab, filled), sampling=vm)
        g = np.stack(np.gradient(gaussian_filter(out0, 1.0)), -1) @ np.linalg.inv(A[:3, :3])
        arr = C.arrival_deg(P, top, g, lambda z: to_box(z))
    d0 = samp(dist0, Rc)
    ratio = top / np.maximum(d0, vm.min())
    fall = C.largest_fall(samp(distw, flat).reshape(m, ns), top, s_grid)
    geom = (C.soft_low(top, *C.THICK_MIN) * C.soft(top, *C.THICK_MAX) * C.soft(ratio, *C.DIRECT)
            * C.soft(arr, *C.ARRIVAL_DEG) * C.soft(fall, *C.FALL_MM))
    tick('geometry (EDT etc.)')
    fold_ok = None
    tick('fold')
    # the fold test by signed area of the coarse triangles (dual of the
    # Voronoi cells: a fine face whose three roots lie in three cells)
    tc = cell[Fi]
    tri = tc[(tc >= 0).all(1) & (tc[:, 0] != tc[:, 1]) & (tc[:, 1] != tc[:, 2]) & (tc[:, 0] != tc[:, 2])]
    tri = np.unique(np.sort(tri, 1), axis=0)
    a, b, c = tri.T
    n0 = np.cross(Rc[b] - Rc[a], Rc[c] - Rc[a])
    # the root triangle's own normal is the reference: same vertex order at
    # depth, so a sign change of the cross product is an inversion
    tri_bad = np.zeros(len(tri), bool)
    RATIO = []
    tmin = np.fmin(np.fmin(top[a], top[b]), top[c])
    for d in FOLD_DEPTHS:
        kk = int(round(d / STEP))
        ok_d = np.isfinite(tmin) & (tmin >= d)
        nd = np.cross(P[b, kk] - P[a, kk], P[c, kk] - P[a, kk])
        rat = np.einsum("ij,ij->i", nd, n0) / np.maximum(np.einsum("ij,ij->i", n0, n0), 1e-18)
        RATIO.append(np.where(ok_d, rat, np.nan))
        bad_d = ok_d & (rat < 0)
        TRI_DIAG[d] = (int(ok_d.sum()), int(bad_d.sum()))
        tri_bad |= bad_d
    e1 = np.linalg.norm(Rc[b] - Rc[a], axis=1); e2 = np.linalg.norm(Rc[c] - Rc[b], axis=1); e3 = np.linalg.norm(Rc[a] - Rc[c], axis=1)
    ar = 0.5 * np.linalg.norm(n0, axis=1)
    TRI_DIAG["quality"] = 4 * np.sqrt(3) * ar / (e1**2 + e2**2 + e3**2)
    TRI_DIAG["bad"] = tri_bad.copy()
    TRI_DIAG["shortest"] = np.fmin(np.fmin(e1, e2), e3)
    TRI_DIAG["one_fine_face"] = None
    fold_tri = np.ones(m, bool)
    for q in range(3):
        fold_tri[tri[tri_bad, q]] = False
    tick("fold by triangles")
    fold_ok = fold_tri
    DIAG.update(tri=tri, ratio=np.stack(RATIO, 1), P=P.astype(np.float32), top=top, Rc=Rc, Nc=Nc, vert=roots.vertex[pick], owned=roots.owned[pick], fold_lsq=fold_ok, fold_tri=fold_tri, depths=np.array(FOLD_DEPTHS))
    # profiles, echo 2 and 3
    KP = PROF_K
    sp = s_grid[::KP]
    Pq = to_box(P[:, ::KP].reshape(-1, 3)).reshape(m, -1, 3)
    SH = np.arange(-(4 // KP), 4 // KP + 1)
    sig = TREND_MM / (STEP * KP)
    if DEV.is_numpy:
        f2, f3 = K.profile_np(ech[1], Pq, sig), K.profile_np(ech[2], Pq, sig)

        def curves(fa, gb, pairs):
            return K.curves_np(fa, gb, pairs, top, sp, TRIM, SH, 10, KP)
    else:
        Pqt = torch.as_tensor(Pq, dtype=FDT, device=DEV.torch)
        f2, f3 = (K.profile_t(torch.as_tensor(ech[e][None], dtype=FDT, device=DEV.torch), Pqt, sig) for e in (1, 2))

        def curves(fa, gb, pairs):
            return K.curves_t(fa, gb, pairs, top, sp, TRIM, SH, 10, KP)


    cur = 0.5 * (curves(f2, f3, ce) + curves(f3, f2, ce)) if len(ce) else np.zeros((0, len(SH)))
    acc = np.zeros((m, len(SH)))
    cnt = np.zeros((m, len(SH)))
    okc = np.isfinite(cur)
    v = np.where(okc, cur, 0.0)
    np.add.at(acc, ce[:, 0], v)
    np.add.at(cnt, ce[:, 0], okc)
    np.add.at(acc, ce[:, 1], v[:, ::-1])
    np.add.at(cnt, ce[:, 1], okc[:, ::-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        nbc = acc / cnt
    has = np.isfinite(nbc).any(1)
    rmax = np.full(m, np.nan)
    best = np.full(m, np.nan)
    rmax[has] = np.nanmax(nbc[has], 1)
    best[has] = SH[np.nanargmax(nbc[has], 1)] * STEP * KP
    own = curves(f2, f3, np.c_[np.arange(m), np.arange(m)])[:, len(SH) // 2]
    tick('profiles and coherence')
    reach = np.isfinite(top)
    silent = ~(own >= SILENT_OWN)
    disagree = ~silent & np.isfinite(rmax) & (own - rmax > DISAGREE)
    gfail = reach & (geom < GEOM_FAIL)
    reason = np.zeros(m, np.int8)  # pass
    reason[silent] = 1
    reason[gfail] = 5
    reason[disagree] = 4
    reason[reach & ~fold_ok] = 3
    reason[~reach] = 2
    tick('reasons')
    vert = roots.vertex[pick]
    owned = roots.owned[pick]
    # the cortex the ray crosses below the top: a guard for empty columns
    return dict(
        vertex=vert, owned=owned, root=Rc.astype(np.float32), normal=Nc.astype(np.float32),
        fold_tri=fold_tri, n_tri=len(tri), n_tri_bad=int(tri_bad.sum()),
        reason=reason, top=top.astype(np.float32), own=own.astype(np.float32), rmax=rmax.astype(np.float32),
        best=best.astype(np.float32), geom=geom.astype(np.float32), area=carea.astype(np.float32),
        edges=np.c_[vert[ce[:, 0]], vert[ce[:, 1]]].astype(np.int64) if len(ce) else np.zeros((0, 2), np.int64),
        edge_owned=(owned[ce[:, 0]] | owned[ce[:, 1]]) if len(ce) else np.zeros(0, bool),
        top_point=np.where(np.isfinite(top)[:, None], P[np.arange(m), np.clip(np.nan_to_num(np.round(top / STEP)).astype(int), 0, ns - 1)], np.nan).astype(np.float32),
        n_fine=int(roots.owned.sum()),
    )


_G = {}


def _init():
    out = Path(_G_OUT[0])
    meta = json.loads((out / "tensor.json").read_text())
    _G['J'] = np.memmap(out / "tensor.f32", np.float32, "r", shape=tuple(meta["shape"]))
    _G['s'] = O.OrientSampler(ECHOES, DSEG, cache_bytes=512 << 20)
    _G['rs'] = RY.RaySampler(SURF, DSEG, cache_bytes=512 << 20)


_G_OUT = []


def _one(sb, out, only):
    if not _G:
        _G_OUT[:] = [str(out)]
        _init()
    s, rs, J = _G['s'], _G['rs'], _G['J']
    A = s.affine
    inv = np.linalg.inv(A)
    parts = out / "superblocks"
    f = parts / f"{sb[0]}_{sb[1]}_{sb[2]}.npz"
    t = time.perf_counter()
    TIMES.clear()
    r = run_superblock(sb, s, rs, J, inv, A, s.filled, s.cortex)
    if r is None:
        np.savez(f, empty=True)
        return f"{sb}: empty"
    tmp = parts / f"{sb[0]}_{sb[1]}_{sb[2]}.tmp.npz"
    np.savez_compressed(tmp, seconds=time.perf_counter() - t, stage_seconds=json.dumps(TIMES), **r)
    tmp.rename(f)
    if only:
        np.savez_compressed(out / f"diag_{sb[0]}_{sb[1]}_{sb[2]}.npz", reason=r['reason'], **DIAG)
    elif os.environ.get("SAVE_DIAG") == "1":
        # every superblock's ray paths, for the top stage (uncompressed: zlib is slow)
        (out / "diag").mkdir(exist_ok=True)
        np.savez(out / "diag" / f"diag_{sb[0]}_{sb[1]}_{sb[2]}.npz", reason=r['reason'], **DIAG)
    own_ = r["owned"]
    return (f"{sb}: {int(own_.sum())} coarse owned of {r['n_fine']} fine; "
            f"{ {REASONS[k]: int(((r['reason'] == k) & own_).sum()) for k in range(6)} } "
            f"{time.perf_counter() - t:.0f} s, peak {peak_rss_gb():.1f} GB; stages "
            f"{ {k: round(v, 1) for k, v in TIMES.items()} }")


def main():
    out = Path(sys.argv[1])
    only = []
    if "--only" in sys.argv:
        only = [tuple(int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(","))]
    (out / "superblocks").mkdir(parents=True, exist_ok=True)
    _G_OUT[:] = [str(out)]
    _init()
    sbs = only or list(superblocks(_G['s'], LEVEL))
    todo = [sb for sb in sbs if not (out / "superblocks" / f"{sb[0]}_{sb[1]}_{sb[2]}.npz").exists()]
    print(f"device {DEV.name} ({FDT}), workers {WORKERS}, {len(todo)} of {len(sbs)} superblocks to do", flush=True)
    t0 = time.perf_counter()
    if WORKERS <= 1:
        for i, sb in enumerate(todo):
            print(f"{i + 1}/{len(todo)} {_one(sb, out, only)}", flush=True)
    else:
        _G.clear()
        with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_one, sb, out, only) for sb in todo]
            for i, fu in enumerate(as_completed(futs)):
                print(f"{i + 1}/{len(todo)} {fu.result()}", flush=True)
    print("done", time.perf_counter() - t0, flush=True)


if __name__ == "__main__":
    main()
