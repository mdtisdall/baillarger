"""Flattening of one region of the hemisphere (phase 3.1): the columns of a
superblock with a halo, from the samplers, with baillarger.flattening.

    flattening_region.py OUT --surf SURF --dseg DSEG --echoes E1 E2 E3
        --echo-dir DIR --superblock I J K --level M --halo-mm H [--name NAME]
        [--decay] [--proximity]

Writes OUT/<name>.npz (edge keys, owned, root, normal, vertex, pairs, the
warp r and tau, s_lo, the scales and the energy reference) and
OUT/<name>.json (counts, the solve's log, times, peak memory).

--decay: the linear response of the warp to a perturbation, at the final
linearisation (the matrix of the last outer pass): (a) one pair's shift
+delta at every level; (b) every pair of one column, in the consistent
direction (the column moves by delta against all its neighbours). For each,
max over levels of |dr| per column against the distance along the mesh
edges from the perturbed column. Saves OUT/<name>_decay.npz.

--proximity: also the prototype's pairs (0.25 mm, dot > 0.7), their count
against the mesh edges, and a second fit with them.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import cg

from baillarger import flatten as FL
from baillarger import flattening as FG
from baillarger import rays as R
from baillarger.blocks import ChunkedSource, open_volume
from baillarger.column_volume import ColumnVolumeSampler

T0 = time.time()


def say(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def peak_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)


def response(M, pre, rhs):  # type: ignore[no-untyped-def]
    x, flag = cg(M, rhs, rtol=1e-12, maxiter=20000, M=pre)
    return x, flag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--surf", type=Path, required=True)
    ap.add_argument("--dseg", type=Path, required=True)
    ap.add_argument("--echoes", type=Path, nargs=3, required=True)
    ap.add_argument("--echo-dir", type=Path, required=True)
    ap.add_argument("--superblock", type=int, nargs=3, required=True)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--halo-mm", type=float, default=0.0)
    ap.add_argument("--name", default="region")
    ap.add_argument("--decay", action="store_true")
    ap.add_argument("--proximity", action="store_true")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--warp-from", type=Path, default=None,
                    help="the region's warp from an earlier run's npz (skips the fit)")
    ap.add_argument("--n-outer", type=int, default=4)
    ap.add_argument("--precond", default="jacobi", choices=["jacobi", "line"])
    ap.add_argument(
        "--local",
        type=float,
        nargs="+",
        default=None,
        help="CUBE_MM H1 [H2 ...]: local solves of a centred cube of CUBE_MM with halos H",
    )
    A = ap.parse_args()
    A.out.mkdir(parents=True, exist_ok=True)
    sb = tuple(A.superblock)

    rs = R.RaySampler(A.surf, A.dseg)
    echoes = []
    for f in A.echoes:
        src = open_volume(f, A.echo_dir / f.with_suffix("").name)
        echoes.append(ChunkedSource(src, 64, 1 << 30))
        aff = src.affine
    smp = ColumnVolumeSampler(echoes, rs.affine, [aff] * 3, rs.shape)  # type: ignore[arg-type]
    r0 = rs.sample(sb, A.level, e_in=0.0, e_out=0.0, halo_mm=A.halo_mm)  # type: ignore[arg-type]
    e_in, e_out = FG.extents(r0)
    del r0
    rays = rs.sample(sb, A.level, e_in=e_in, e_out=e_out, halo_mm=A.halo_mm)  # type: ignore[arg-type]
    n = len(rays.roots.vertex)
    say(
        f"{n:,} columns ({int(rays.roots.owned.sum()):,} owned), {len(rays.s):,} samples; "
        f"e_in p50/p99/max {np.percentile(e_in, [50, 99, 100]).round(2).tolist()}, "
        f"e_out p50/p99/max {np.percentile(e_out, [50, 99, 100]).round(2).tolist()}"
    )
    cv = smp.sample(rays, None)
    say(f"column volume; peak {peak_gb():.2f} GB")
    p = FG.pad(cv)
    del cv
    F, M, scales = FG.features(p)
    say(f"padded {p.echo.shape}, s_lo {p.s_lo}; scales {scales.round(4).tolist()}")
    vox = (rs.inverse[:3, :3] @ rays.roots.root.astype(np.float64).T).T + rs.inverse[
        :3, 3
    ]
    sf = FG.SurfaceFaces.load(A.surf)
    pairs = FG.orient(
        FG.mesh_edge_pairs(sf, rays.roots.vertex, vox), rays.roots.edge_key
    )
    say(f"{len(pairs):,} mesh-edge pairs ({2 * len(pairs) / n:.2f} per column)")
    stats: dict[str, object] = dict(
        name=A.name,
        superblock=list(sb),
        level=A.level,
        halo_mm=A.halo_mm,
        columns=n,
        owned=int(rays.roots.owned.sum()),
        samples=int(len(rays.s)),
        grid=list(p.echo.shape[1:]),
        s_lo=p.s_lo,
        scales=scales.tolist(),
        mesh_pairs=int(len(pairs)),
        e_in_p50_p99=np.percentile(e_in, [50, 99]).tolist(),
        e_out_p50_p99=np.percentile(e_out, [50, 99]).tolist(),
    )
    if A.proximity:
        pp = FG.orient(
            FG.proximity_pairs(rays.roots.root, rays.roots.normal), rays.roots.edge_key
        )
        a = {tuple(x) for x in pairs.tolist()}
        b = {tuple(x) for x in pp.tolist()}
        stats["proximity_pairs"] = len(pp)
        stats["proximity_and_mesh"] = len(a & b)
        say(f"proximity pairs {len(pp):,}; shared with the mesh edges {len(a & b):,}")
    A.step_mm = p.step
    t = time.time()
    ps = FG.pair_shifts(F, M, pairs, p.step)
    stats["seconds_shifts"] = round(time.time() - t, 1)
    stats["energy_ref"] = ps.energy_ref
    say(f"shifts; energy ref {ps.energy_ref:.4g}; peak {peak_gb():.2f} GB")
    t = time.time()
    if A.warp_from is not None:
        z = np.load(A.warp_from)
        assert np.array_equal(z["edge_key"], rays.roots.edge_key)
        rr = z["r"].astype(np.float64)
        warp = FG.Warp(float(z["s_lo"]), z["tau"], rr, int(z["anchor_level"]), [])
        say("warp from " + str(A.warp_from))
    else:
        warp = FG.fit(ps, n, p.s_lo, p.step, log=say, precond=A.precond, n_outer=A.n_outer)
    stats["seconds_fit"] = round(time.time() - t, 1)
    stats["fit"] = warp.info
    say(f"fit; peak {peak_gb():.2f} GB")
    np.savez_compressed(
        A.out / f"{A.name}.npz",
        edge_key=rays.roots.edge_key,
        owned=rays.roots.owned,
        vertex=rays.roots.vertex,
        root=rays.roots.root,
        normal=rays.roots.normal,
        pairs=pairs,
        r=warp.r.astype(np.float32),
        tau=warp.tau,
        s_lo=p.s_lo,
        anchor_level=warp.anchor_level,
        scales=scales,
        energy_ref=ps.energy_ref,
    )
    if A.proximity:
        psp = FG.pair_shifts(F, M, pp, p.step)
        wp = FG.fit(psp, n, p.s_lo, p.step, log=say)
        stats["fit_proximity"] = wp.info
        d = np.abs(wp.r - warp.r)
        stats["proximity_minus_mesh_abs_dr_p50_p99_max"] = np.percentile(
            d, [50, 99, 100]
        ).tolist()
        del psp
    if A.local:
        local(A, ps, warp, pairs, vox, rs.voxel_mm, stats)
    if A.decay:
        decay(A, rays, ps, warp, pairs, vox, rs.voxel_mm, stats)
    stats["peak_gb"] = peak_gb()
    stats["total_seconds"] = round(time.time() - T0, 1)
    (A.out / f"{A.name}.json").write_text(json.dumps(stats, indent=1))
    say(json.dumps({k: v for k, v in stats.items() if k != "fit"}))


def decay(A, rays, ps, warp, pairs, vox, voxel_mm, stats) -> None:  # type: ignore[no-untyped-def]
    """The linear response of the warp at its final linearisation."""
    N, K = warp.r.shape
    Rm = FL.prior_matrix(N, K, FG.EPS, 10.0, warp.anchor_level)
    Ad, bd, sw = FL.data_rows(
        pairs, ps.shift, ps.weight, warp.r, warp.tau, ps_step(rays)
    )
    Mx = (Ad.T @ Ad + Rm).tocsr()
    dg = Mx.diagonal()
    pre = diags(1.0 / np.where(dg > 0, dg, 1.0))
    # the rows of each pair: rebuild their pair index as data_rows does
    v = pairs[:, 0]
    q = (warp.tau[None] + warp.r[v]) / rays.step
    wt, ins = FL.interp_last(ps.weight, q)
    wt = np.where(ins, np.maximum(wt, 0.0), 0.0)
    p_, _ = np.nonzero(wt > 0)
    # the graph of mesh edges in mm
    xm = vox * voxel_mm
    L = np.linalg.norm(xm[pairs[:, 0]] - xm[pairs[:, 1]], axis=1)
    G = coo_matrix((L, (pairs[:, 0], pairs[:, 1])), shape=(N, N)).tocsr()
    # the perturbed column: the one nearest the centre of the region's roots,
    # among the columns with the most weight
    wsum = np.bincount(pairs[p_, 0], minlength=N) + np.bincount(
        pairs[p_, 1], minlength=N
    )
    cen = xm.mean(0)
    cand = np.flatnonzero(wsum >= np.percentile(wsum, 75))
    c = int(cand[np.argmin(np.linalg.norm(xm[cand] - cen, axis=1))])
    dist = dijkstra(G, directed=False, indices=c)
    out: dict[str, np.ndarray] = dict(dist=dist, column=np.array(c))
    mine = np.flatnonzero((pairs[:, 0] == c) | (pairs[:, 1] == c))
    say(f"decay: column {c}, {len(mine)} pairs, {int(wsum[c])} data rows")
    # (a) one pair; (b) all the column's pairs, consistent
    cases = {
        "pair": (mine[:1], np.ones(1)),
        "column": (mine, np.where(pairs[mine, 1] == c, 1.0, -1.0)),
    }
    for name, (sel, sign) in cases.items():
        db = np.zeros(len(bd))
        for pi, sg in zip(sel, sign, strict=True):
            rows = np.flatnonzero(p_ == pi)
            db[rows] = sg * A.delta * sw[rows]
        x, flag = response(Mx, pre, Ad.T @ db)
        dr = np.abs(x.reshape(N, K)).max(1)
        out[name] = dr
        say(f"decay {name}: cg flag {flag}; |dr| at the column {dr[c]:.4g} mm")
    # the curve: max |dr| in distance bins
    edges_ = np.arange(0, np.nanmax(dist[np.isfinite(dist)]) + 0.5, 0.25)
    curve: dict[str, list[float]] = {"bin_mm": edges_[:-1].tolist()}
    for name in cases:
        b = np.digitize(dist, edges_) - 1
        curve[name] = [
            float(out[name][b == i].max()) if np.any(b == i) else float("nan")
            for i in range(len(edges_) - 1)
        ]
    stats["decay"] = curve
    np.savez_compressed(A.out / f"{A.name}_decay.npz", **out)


def local(A, ps, warp, pairs, vox, voxel_mm, stats) -> None:  # type: ignore[no-untyped-def]
    """Local solves: the columns whose roots lie in a cube of CUBE_MM round the
    region's centre are owned; the halo adds the roots in the cube widened by h
    on each side. The pairs are the region's mesh edges among those columns,
    with the same shifts and weights (a shift depends only on its two columns,
    with the scales and the energy reference fixed). Compares r of the owned
    columns with the region's solve."""
    cube, halos = A.local[0], A.local[1:]
    xm = vox * voxel_mm
    cen = 0.5 * (xm.min(0) + xm.max(0))
    d = np.abs(xm - cen).max(1)
    own = d <= cube / 2
    rows = []
    for h in halos:
        sel = np.flatnonzero(d <= cube / 2 + h)
        loc = -np.ones(len(xm), np.int64)
        loc[sel] = np.arange(len(sel))
        keep = np.all(loc[pairs] >= 0, axis=1)
        sub = FG.PairShifts(
            loc[pairs[keep]], ps.shift[keep], ps.weight[keep], ps.energy_ref
        )
        t = time.time()
        w = FG.fit(
            sub, len(sel), warp.s_lo, float(A.step_mm), precond=A.precond, n_outer=A.n_outer
        )
        dt = time.time() - t
        o = own[sel]
        dd = w.r[o] - warp.r[sel[o]]
        diff = np.abs(dd)
        # the same after the per-level constant (the gauge: fit_warp gives
        # mean_v r_v(tau) = 0 over the solve's columns, at each level)
        dg = np.abs(dd - dd.mean(0, keepdims=True))
        gauge = np.abs(dd.mean(0))
        # per connected piece of the local graph (a piece has its own gauge)
        from scipy.sparse import coo_matrix as _coo
        from scipy.sparse.csgraph import connected_components

        lp = loc[pairs[keep]]
        ncomp, comp = connected_components(
            _coo((np.ones(len(lp)), (lp[:, 0], lp[:, 1])), shape=(len(sel), len(sel))),
            directed=False,
        )
        co = comp[o]
        dp = dd.copy()
        for k in np.unique(co):
            dp[co == k] -= dd[co == k].mean(0)
        # the same horizon family: for each local level, the region's horizon
        # with the same mean depth over the owned columns (a reparametrisation
        # of tau: z(tau) -> z(phi(tau)) leaves the data rows unchanged)
        zl = warp.tau[None] + w.r[o]
        zg = warp.tau[None] + warp.r[sel[o]]
        mg = np.maximum.accumulate(zg.mean(0))
        ml = zl.mean(0)
        tp = np.interp(ml, mg, warp.tau)
        zgm = np.stack([np.interp(tp, warp.tau, row) for row in zg])
        drp = np.abs(zl - zgm)
        inside = (ml > mg[0]) & (ml < mg[-1])
        sizes = np.bincount(co)
        pieces = dict(
            reparam_abs_dz_p50_p99_max=np.percentile(drp[:, inside], [50, 99, 100]).tolist(),
            reparam_abs_dz_cortex_p50_p99_max=np.percentile(
                drp[:, inside & (warp.s_lo + warp.tau > 0) & (warp.s_lo + warp.tau < 3)],
                [50, 99, 100],
            ).tolist(),
            pieces_with_owned=int((sizes > 0).sum()),
            owned_in_largest=int(sizes.max()),
            degauged_per_piece_abs_dr_p50_p99_max=np.percentile(
                np.abs(dp), [50, 99, 100]
            ).tolist(),
        )
        row = dict(
            halo_mm=h,
            columns=int(len(sel)),
            owned=int(o.sum()),
            seconds=round(dt, 1),
            cg_iters=[i["cg_iters"] for i in w.info],
            abs_dr_p50_p99_max=np.percentile(diff, [50, 99, 100]).tolist(),
            per_column_max_p50_p99=np.percentile(diff.max(1), [50, 99]).tolist(),
            gauge_abs_max=float(gauge.max()),
            degauged_abs_dr_p50_p99_max=np.percentile(dg, [50, 99, 100]).tolist(),
            local_mean_of_region_r_abs_max=float(np.abs(warp.r[sel].mean(0)).max()),
            **pieces,
        )
        say(f"local h={h}: {row}")
        rows.append(row)
    stats["local"] = dict(cube_mm=cube, rows=rows)


def ps_step(rays: R.Rays) -> float:
    return float(rays.step)


if __name__ == "__main__":
    main()
