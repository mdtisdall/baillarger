"""Toy checks of the thin-plate antialiasing (antialias.py) where the answer is known: a
voxelised tilted plane (true normal known, true surface flat) and a voxelised sphere (true
normals radial, true radius known). For each: the raw (binary) mesh against the antialiased one,
with the same triangles; normal errors from the mesh (area-weighted, unsmoothed) and from phi's
gradient; distance of the vertices from the true surface; self-intersections; the solver's
convergence.

Usage: antialias_toy.py [--device mps] [--band B] [--eps E] [--tol T] [--iters N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import antialias as AA  # noqa: E402

from baillarger import mesh_intersect as mi  # noqa: E402
from baillarger import mesh_smoothing as S  # noqa: E402
from baillarger.mesh import marching_tetrahedra_blocks  # noqa: E402


def field_normals(phi, v):
    g = np.zeros((len(v), 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = 0.5
        g[:, k] = ndi.map_coordinates(phi, (v + e).T, order=1) - ndi.map_coordinates(
            phi, (v - e).T, order=1
        )
    return -g / np.linalg.norm(g, axis=1, keepdims=True)


def angle(n, ref):
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n, ref), -1, 1)))


def stats(x):
    return f"median {np.median(x):7.3f}  p99 {np.percentile(x, 99):7.3f}  max {np.max(x):7.3f}"


def run(name, mask, true_normal, true_dist, a):
    din = ndi.distance_transform_edt(mask)
    dout = ndi.distance_transform_edt(~mask)
    sdf = np.where(mask, din - 0.5, 0.5 - dout).astype(np.float32)
    np.clip(sdf, -(a.band + 3), a.band + 3, out=sdf)
    phi, solve = AA.thin_plate(
        sdf,
        mask,
        a.band,
        a.eps,
        a.device,
        a.iters,
        a.tol,
        fused=not a.unfused,
        stop=a.stop,
        start=a.start,
        levels=a.levels,
        quiet=True,
        restart=a.restart,
    )
    vr, fr, br = marching_tetrahedra_blocks(mask.view(np.uint8), 0.5, 16, np.float32)
    v, f, b = marching_tetrahedra_blocks(phi, 0.0, 16, np.float32)
    fr, f = fr[:, [0, 2, 1]], f[:, [0, 2, 1]]
    same = np.array_equal(f, fr) and np.array_equal(b.vertex_key, br.vertex_key)
    _, nx = mi.intersecting_faces(v, f)
    # leave out vertices near the volume's faces (the problem's own border)
    keep = np.all((v > a.band + 3) & (v < np.array(mask.shape) - a.band - 4), axis=1)
    vr64, v64 = vr.astype(float), v.astype(float)
    print(
        f"\n== {name}: {len(v)} vertices ({keep.sum()} away from the border); "
        f"same triangles {same}; self-intersecting faces {nx}; solver {solve['iterations']} "
        f"iterations, last change {solve['last_change']:.1e}; levels {solve['levels']}"
    )
    for label, vv, ff, nn in (
        ("raw, mesh normals", vr64, fr, S.vertex_normals(vr64, fr)),
        ("antialiased, mesh normals", v64, f, S.vertex_normals(v64, f)),
        ("antialiased, gradient normals", v64, f, field_normals(phi, v64)),
    ):
        err = angle(nn, true_normal(vv))[keep]
        print(f"  {label:32s} normal error (deg): {stats(err)}")
    for label, vv in (("raw", vr64), ("antialiased", v64)):
        print(
            f"  {label:32s} distance to true surface (vox): {stats(np.abs(true_dist(vv))[keep])}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--band", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--levels", type=int, default=0)
    ap.add_argument("--start", choices=("sdf", "blur"), default="sdf")
    ap.add_argument("--stop", choices=("surface", "band"), default="band")
    ap.add_argument("--unfused", action="store_true")
    ap.add_argument("--restart", action="store_true")
    a = ap.parse_args()
    N = 48
    x = np.indices((N, N, N)).astype(float)
    # a plane through the centre, tilted off every axis
    n = np.array([0.42, 0.29, 0.86])
    n /= np.linalg.norm(n)
    c = np.array([23.3, 24.1, 23.7])
    plane = np.einsum("i...,i->...", x - c[:, None, None, None], n) < 0
    run(
        "tilted plane",
        plane,
        lambda v: np.tile(n, (len(v), 1)),
        lambda v: (v - c) @ n,
        a,
    )
    # a sphere of radius 12
    R = 12.0
    sphere = np.sqrt(((x - c[:, None, None, None]) ** 2).sum(0)) < R
    run(
        "sphere r=12",
        sphere,
        lambda v: (v - c) / np.linalg.norm(v - c, axis=1, keepdims=True),
        lambda v: np.linalg.norm(v - c, axis=1) - R,
        a,
    )


if __name__ == "__main__":
    main()
