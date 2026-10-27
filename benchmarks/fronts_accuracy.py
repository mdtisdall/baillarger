"""Accuracy and scale of baillarger.fronts, against exact answers.

The tests in tests/test_fronts.py check correctness with loose bounds; this
script prints the measured numbers behind those bounds and behind the limits
stated in the march() docstring, so a change to the solver can be compared
with them. Run it in the devShell:

  nix develop --command uv run python benchmarks/fronts_accuracy.py [part ...]

Parts (default: all but scale):
  selling   Selling reconstruction error and offset size at anisotropy 1-30.
  point     Arrival from a point source, isotropic and at anisotropy 3 and 10:
            relative error, and the angle between computed and exact gradients.
  manuf     A smoothly varying metric with an exact distance (the pull-back of
            the Euclidean metric by a smooth map), at 41^3 and 81^3, against
            the same run with the metric frozen at the source.
  length    Arc length: from a plane source (exact) and from a point source.
  ball      Arc length from a ball source (radius 6) at 41^3, 81^3 and 121^3:
            it should converge under refinement.
  scale     A 141^3 box at the data's voxel size with a direction field, a wall
            and a plane of 141^2 labelled sources: time and peak memory
            (about 9 s and 2.6 GB). Run it on its own, under a memory cap.
"""

import resource
import sys
import time

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from baillarger.fronts import march, riemann_from_direction, selling

type Float = npt.NDArray[np.float64]

# an oblique axis for the anisotropic metrics
AXIS = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])


def grid(n: int, h: float = 1.0) -> Float:
    """Voxel positions (n, n, n, 3) with spacing h."""
    axes = [np.arange(n) * h] * 3
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1)


def angles(u: Float, exact: Float, keep: npt.NDArray[np.bool_], h: float) -> Float:
    """Angle (degrees) between the gradients of u and exact, where keep."""
    gu = np.moveaxis(np.asarray(np.gradient(u, h)), 0, -1)
    ge = np.moveaxis(np.asarray(np.gradient(exact, h)), 0, -1)
    ok = keep & np.all(np.isfinite(gu), -1)
    cos = np.sum(gu[ok] * ge[ok], -1)
    cos /= np.linalg.norm(gu[ok], axis=-1) * np.linalg.norm(ge[ok], axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def part_selling() -> None:
    rng = np.random.default_rng(0)
    for ratio in (1, 3, 10, 30):
        q, _ = np.linalg.qr(rng.standard_normal((200, 3, 3)))
        ev = np.stack(
            [np.ones(200), np.full(200, ratio**2), rng.uniform(1, ratio**2, 200)], 1
        )
        d = q @ (ev[:, :, None] * np.eye(3)) @ q.transpose(0, 2, 1)
        off, w = selling(d)
        o = off.astype(float)
        rec = np.einsum("np,npi,npj->nij", w, o, o)
        err = np.max(np.abs(rec - d)) / np.max(np.abs(d))
        print(
            f"selling, anisotropy {ratio}: max relative error {err:.1e}, "
            f"max |offset| {np.abs(off).max()}"
        )


def part_point() -> None:
    n = 61
    c = n // 2
    lab = -np.ones((n, n, n), np.int64)
    lab[c, c, c] = 0
    g = grid(n) - c
    far = np.linalg.norm(g, axis=-1) > 5
    for ratio in (1.0, 3.0, 10.0):
        m = riemann_from_direction(AXIS, ratio)
        t = time.time()
        f = march(m, lab)
        el = time.time() - t
        ex = np.sqrt(np.einsum("...i,ij,...j->...", g, m, g))
        rel = np.abs(f.arrival[far] - ex[far]) / ex[far]
        ang = angles(f.arrival, ex, far, 1.0)
        print(
            f"point source {n}^3, anisotropy {ratio:g} ({el:.1f} s): arrival "
            f"relative error median {np.median(rel):.3f}, p95 "
            f"{np.percentile(rel, 95):.3f}; gradient angle median "
            f"{np.median(ang):.2f}, p95 {np.percentile(ang, 95):.2f} deg"
        )


def part_manuf() -> None:
    """Metric J^T J for phi(x) = x + (a/k) sin(k x_rotated): distance |phi(x) - phi(s)|.

    That is exact only along straight lines in phi's image, which is all of it
    for a = 0 and close to it for small a; the frozen-metric run separates the
    solver's error from the metric's variation."""
    for n in (41, 81):
        for frozen in (False, True):
            a = 0.12
            h = 40.0 / (n - 1)
            k = np.pi / 40.0
            x = grid(n, h)
            c = (n // 2,) * 3
            cy, cz, cx = (np.cos(k * x[..., i]) for i in (1, 2, 0))
            jac = np.zeros((n, n, n, 3, 3))
            jac[..., 0, 0] = jac[..., 1, 1] = jac[..., 2, 2] = 1
            jac[..., 0, 1] = a * cy
            jac[..., 1, 2] = a * cz
            jac[..., 2, 0] = a * cx
            if frozen:
                jac = np.broadcast_to(jac[c], jac.shape)
                phi = x @ jac[c].T
            else:
                sines = [np.sin(k * x[..., i]) for i in (1, 2, 0)]
                phi = x + (a / k) * np.stack(sines, -1)
            m = np.einsum("...ki,...kj->...ij", jac, jac)
            lab = -np.ones((n, n, n), np.int64)
            lab[c] = 0
            f = march(m, lab, spacing=(h, h, h))
            ex = np.linalg.norm(phi - phi[c], axis=-1)
            r = np.linalg.norm(x - x[c], axis=-1)
            keep = (r > 5) & (r < 15)
            rel = np.abs(f.arrival[keep] - ex[keep]) / ex[keep]
            ang = angles(f.arrival, ex, keep, h)
            kind = "frozen at the source" if frozen else "varying"
            print(
                f"manufactured metric {n}^3, {kind}: arrival relative error median "
                f"{np.median(rel):.3f}; gradient angle median {np.median(ang):.2f}, "
                f"p95 {np.percentile(ang, 95):.2f} deg"
            )


def part_length() -> None:
    n = 41
    for ratio in (3.0, 10.0):
        m = riemann_from_direction(AXIS, ratio)
        # plane source x = 0: the characteristics run along V = M^-1 e_x, so
        # the arc length at x is x |V| / V_x, wherever the characteristic from
        # the plane stays inside the box
        v = np.linalg.solve(m, [1.0, 0.0, 0.0])
        lab = -np.ones((n, n, n), np.int64)
        lab[0] = 0
        f = march(m, lab)
        x, y, z = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
        ex = x * np.linalg.norm(v) / v[0]
        yy = y - v[1] / v[0] * x
        zz = z - v[2] / v[0] * x
        ok = (x >= 5) & (yy >= 0) & (yy <= n - 1) & (zz >= 0) & (zz <= n - 1)
        rel = np.abs(f.length[ok] - ex[ok]) / ex[ok]
        print(
            f"plane source, anisotropy {ratio:g}: length relative error median "
            f"{np.median(rel):.3f}, p95 {np.percentile(rel, 95):.3f}"
        )
        for np_ in (41, 81):
            h = 40.0 / (np_ - 1)
            c = (np_ // 2,) * 3
            lab = -np.ones((np_, np_, np_), np.int64)
            lab[c] = 0
            f = march(m, lab, spacing=(h, h, h))
            r = np.linalg.norm(grid(np_, h) - np.array(c) * h, axis=-1)
            shell = (r > 10) & (r < 18)
            rel = np.abs(f.length[shell] - r[shell]) / r[shell]
            print(
                f"  point source {np_}^3: length relative error median "
                f"{np.median(rel):.3f}"
            )


def part_ball() -> None:
    """Arc length from a ball: exact = the Euclidean length of the metric-closest path,
    a straight segment to the ball point nearest in the metric (constant M)."""
    radius = 6.0
    for ratio in (3.0, 10.0):
        m = riemann_from_direction(AXIS, ratio)
        chol = np.linalg.cholesky(m)
        for n in (41, 81, 121):
            h = 40.0 / (n - 1)
            g = grid(n, h)
            r = np.linalg.norm(g - 20.0, axis=-1)
            ball = r <= radius
            f = march(m, np.where(ball, 0, -1), spacing=(h, h, h))
            _, idx = cKDTree(g[ball] @ chol).query(g.reshape(-1, 3) @ chol)
            ex = np.linalg.norm(g.reshape(-1, 3) - g[ball][idx], axis=-1)
            ex = ex.reshape(r.shape)
            far = (r > radius + 3) & (r < 18)
            rel = np.abs(f.length[far] - ex[far]) / ex[far]
            print(
                f"ball source {n}^3, anisotropy {ratio:g}: length relative error "
                f"median {np.median(rel):.3f}, p95 {np.percentile(rel, 95):.3f}"
            )


def part_scale() -> None:
    n = 141
    x = grid(n, 1.0 / (n - 1))
    y = x[..., 1]
    v = np.stack([np.cos(3 * y), np.sin(3 * y), np.full_like(y, 0.3)], -1)
    m = riemann_from_direction(v, 5.0)
    mask = np.ones((n, n, n), bool)
    mask[:, 70, 20:120] = False
    lab = -np.ones((n, n, n), np.int64)
    lab[0] = np.arange(n * n).reshape(n, n)
    lab[~mask] = -1
    t = time.time()
    f = march(m, lab, mask=mask, spacing=(0.16028, 0.16, 0.16028))
    el = time.time() - t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / (1e9 if sys.platform == "darwin" else 1e6)
    print(
        f"scale {n}^3 ({mask.sum():,} voxels, {(lab >= 0).sum():,} labels): march "
        f"{el:.1f} s (including compilation), peak memory {peak_gb:.1f} GB; "
        f"reached {np.isfinite(f.arrival).mean():.1%}"
    )


PARTS = {
    "selling": part_selling,
    "point": part_point,
    "manuf": part_manuf,
    "length": part_length,
    "ball": part_ball,
    "scale": part_scale,
}

if __name__ == "__main__":
    names = sys.argv[1:] or [p for p in PARTS if p != "scale"]
    unknown = [p for p in names if p not in PARTS]
    if unknown:
        sys.exit(f"unknown parts {unknown}; choose from {list(PARTS)}")
    for p in names:
        PARTS[p]()
        sys.stdout.flush()
