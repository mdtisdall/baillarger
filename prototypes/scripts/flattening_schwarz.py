"""Overlapping Schwarz with a coarse gauge, against one region solve (phase
3.1, option B').

    flattening_schwarz.py OUT --surf SURF --dseg DSEG --echoes E1 E2 E3
        --echo-dir DIR --superblock I J K --level M --reference NPZ
        --mode linear|gn [--sub-cells 32] [--halo-mm 2] [--sweeps 5]
        [--outer 1] [--reverse] [--name NAME]

The region is one superblock with no halo (as flattening_region.py). It is
cut into sub-cubes of --sub-cells voxels (32 at level 0: 8 sub-cubes of
5.12 mm). A sweep visits the sub-cubes in Morton order (--reverse: the
other way). For each sub-cube: its owned columns are free; the other
columns in its box widened by the halo are fixed at their owners' current
warp (a Dirichlet condition); the warp of the owned columns is solved and
stored. After each sweep, gauge_offsets puts the sub-cubes in one gauge
(the overlaps: each sub-cube's halo values as used in its solve).

--mode linear: the warp linearised at r = 0 (one Gauss-Newton pass); each
sub-cube solve is the exact block solve of that one linear system, and the
reference is the region's one-pass solve.
--mode gn: each sub-cube solve is fit_warp with --outer passes from the
current warp, with the fixed columns; the reference is the region's
four-pass solve.

Prints and writes OUT/<name>.json: per sweep, |r - r_reference| over all
columns at the levels in the cortex (s from 0 to 3 mm), p50/p99/max, and
the change from the last sweep.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

from baillarger import flatten as FL
from baillarger import flattening as FG
from baillarger import rays as R
from baillarger.blocks import ChunkedSource, morton, open_volume
from baillarger.column_volume import ColumnVolumeSampler

T0 = time.time()


def say(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--surf", type=Path, required=True)
    ap.add_argument("--dseg", type=Path, required=True)
    ap.add_argument("--echoes", type=Path, nargs=3, required=True)
    ap.add_argument("--echo-dir", type=Path, required=True)
    ap.add_argument("--superblock", type=int, nargs=3, required=True)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--mode", choices=["linear", "gn"], required=True)
    ap.add_argument("--sub-cells", type=int, default=32)
    ap.add_argument("--halo-mm", type=float, default=2.0)
    ap.add_argument("--sweeps", type=int, default=5)
    ap.add_argument("--outer", type=int, default=1)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--name", default="schwarz")
    ap.add_argument("--no-gauge", action="store_true", help="no gauge_offsets after a sweep")
    ap.add_argument(
        "--exact-iters", type=int, default=0,
        help="linear: first run CG from the reference to rtol 1e-11 for this many "
        "iterations, report the difference, and use that as the reference",
    )
    A = ap.parse_args()
    sb = tuple(A.superblock)

    rs = R.RaySampler(A.surf, A.dseg)
    echoes = []
    for f in A.echoes:
        src = open_volume(f, A.echo_dir / f.with_suffix("").name)
        echoes.append(ChunkedSource(src, 64, 1 << 30))
        aff = src.affine
    smp = ColumnVolumeSampler(echoes, rs.affine, [aff] * 3, rs.shape)  # type: ignore[arg-type]
    r0 = rs.sample(sb, A.level, e_in=0.0, e_out=0.0)  # type: ignore[arg-type]
    e_in, e_out = FG.extents(r0)
    rays = rs.sample(sb, A.level, e_in=e_in, e_out=e_out)  # type: ignore[arg-type]
    n = len(rays.roots.vertex)
    ref = np.load(A.reference)
    assert np.array_equal(ref["edge_key"], rays.roots.edge_key)
    p = FG.pad(smp.sample(rays, None))
    assert abs(p.s_lo - float(ref["s_lo"])) < 1e-9
    F, M, _ = FG.features(p, ref["scales"])
    vox = (rs.inverse[:3, :3] @ rays.roots.root.astype(np.float64).T).T + rs.inverse[
        :3, 3
    ]
    pairs = FG.orient(
        FG.mesh_edge_pairs(FG.SurfaceFaces.load(A.surf), rays.roots.vertex, vox),
        rays.roots.edge_key,
    )
    ps = FG.pair_shifts(F, M, pairs, p.step, float(ref["energy_ref"]))
    del F, M
    tau = ref["tau"]
    K = len(tau)
    ja = int(ref["anchor_level"])
    r_ref = ref["r"].astype(np.float64)
    cortex = (p.s_lo + tau > 0) & (p.s_lo + tau < 3)
    say(f"{n:,} columns, {len(pairs):,} pairs, K {K}; mode {A.mode}")

    # sub-cubes in Morton order
    sub = np.floor(vox / A.sub_cells).astype(np.int64)
    codes = morton(sub)
    order = np.unique(codes)
    if A.reverse:
        order = order[::-1]
    h = A.halo_mm / rs.voxel_mm

    if A.mode == "linear":
        Ad, bd, _ = FL.data_rows(pairs, ps.shift, ps.weight, np.zeros((n, K)), tau, p.step)
        Mx = (Ad.T @ Ad + FL.prior_matrix(n, K, FG.EPS, 10.0, ja)).tocsr()
        rhs = Ad.T @ bd
        del Ad, bd
        nb = float(np.linalg.norm(rhs))

        def relres(x: np.ndarray) -> float:
            return float(np.linalg.norm(Mx @ x.ravel() - rhs) / nb)

        say(f"reference relative residual {relres(r_ref):.3g}")
        if A.exact_iters:
            dg0 = Mx.diagonal()
            xe, fl = cg(
                Mx, rhs, x0=r_ref.ravel(), rtol=1e-11, maxiter=A.exact_iters,
                M=diags(1.0 / np.where(dg0 > 0, dg0, 1.0)),
            )
            xe = xe.reshape(n, K)
            dd = np.abs(xe - r_ref)[:, cortex]
            mlev = np.abs((xe - r_ref).mean(0))
            say(
                f"long CG (flag {fl}): residual {relres(xe):.3g}; |exact - reference| "
                f"p50/p99/max {np.percentile(dd, [50, 99, 100]).round(4).tolist()}; "
                f"|mean per level| max {mlev.max():.4f}"
            )
            np.savez_compressed(A.out / f"{A.name}_exact.npz", r=xe.astype(np.float32))
            r_ref = xe

    Rw = np.zeros((n, K))
    rows = []
    prev = Rw.copy()
    for sweep in range(A.sweeps):
        t = time.time()
        results = []
        for code in order:
            own = codes == code
            c3 = sub[np.flatnonzero(own)[0]]
            lo, hi = c3 * A.sub_cells - h, (c3 + 1) * A.sub_cells + h
            sel = np.flatnonzero(np.all((vox >= lo) & (vox <= hi), axis=1))
            o = own[sel]
            if A.mode == "linear":
                fr = np.repeat(o, K)
                ix = (sel[:, None] * K + np.arange(K)[None]).ravel()
                Msel = Mx[ix][:, ix]
                Mff = Msel[fr][:, fr].tocsr()
                b = rhs[ix][fr] - Msel[fr][:, ~fr] @ Rw[sel[~o]].ravel()
                dg = Mff.diagonal()
                x, flag = cg(
                    Mff, b, x0=Rw[sel[o]].ravel(), rtol=1e-7, maxiter=3000,
                    M=diags(1.0 / np.where(dg > 0, dg, 1.0)),
                )
                Rw[sel[o]] = x.reshape(-1, K)
            else:
                loc = -np.ones(n, np.int64)
                loc[sel] = np.arange(len(sel))
                keep = np.all(loc[pairs] >= 0, axis=1)
                tau_, r_, info = FL.fit_warp(
                    loc[pairs[keep]], ps.shift[keep], ps.weight[keep], len(sel),
                    p.step, dtau=FG.DTAU, tau_max=float(tau[-1]), eps=FG.EPS,
                    n_outer=A.outer, r0=Rw[sel], anchor_level=ja, fixed=~o,
                )
                Rw[sel[o]] = r_[o]
            results.append(
                FG.Flattened(
                    edge_key=rays.roots.edge_key[sel[o]], s_lo=p.s_lo, step=p.step,
                    tau=tau, r=Rw[sel[o]].copy(), anchor_level=ja,
                    pair_key=np.zeros((0, 2), np.int64),
                    shift=np.zeros((0, 1), np.float32),
                    weight=np.zeros((0, 1), np.float32), scales=ref["scales"],
                    energy_ref=float(ref["energy_ref"]), info=[], n_columns=len(sel),
                    n_pairs=0, misfit=np.zeros(int(o.sum())),
                    vertex=rays.roots.vertex[sel[o]],
                    halo_key=rays.roots.edge_key[sel[~o]], halo_r=Rw[sel[~o]].copy(),
                )
            )
        lev, c = FG.gauge_offsets(results)
        assert len(lev) == K  # one s_lo and one tau for all the sub-cubes
        if not A.no_gauge:
            for f, cb in zip(results, c, strict=True):
                idx = np.searchsorted(rays.roots.edge_key, f.edge_key)
                Rw[idx] += cb[None]
        d = np.abs(Rw - r_ref)[:, cortex]
        ch = np.abs(Rw - prev)[:, cortex]
        prev = Rw.copy()
        row = dict(
            sweep=sweep,
            seconds=round(time.time() - t, 1),
            offsets_abs_max=float(np.abs(c).max()),
            abs_dr_p50_p99_max=np.percentile(d, [50, 99, 100]).tolist(),
            change_p50_p99_max=np.percentile(ch, [50, 99, 100]).tolist(),
            mean_per_level_abs_max=float(np.abs((Rw - r_ref).mean(0)).max()),
        )
        if A.mode == "linear":
            row["relative_residual"] = relres(Rw)
        rows.append(row)
        say(json.dumps(row))
    A.out.mkdir(parents=True, exist_ok=True)
    (A.out / f"{A.name}.json").write_text(
        json.dumps(dict(args={k: str(v) for k, v in vars(A).items()}, sweeps=rows), indent=1)
    )
    np.savez_compressed(A.out / f"{A.name}.npz", r=Rw.astype(np.float32), edge_key=rays.roots.edge_key)
    say("done")


if __name__ == "__main__":
    sys.exit(main())
