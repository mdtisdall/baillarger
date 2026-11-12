"""The truth under the stop rule (docs/plans/flow-field-phantoms.md, *Truth under the stop rule*;
PR 1, task 1.3a).

This module has three parts:

1. The registry of the plan's phantoms: PHANTOMS, build_phantom (the phantom with its image) and
   geometry_of (the geometry and tau_scale only, with no image).
2. `truth`: a tracer of the true columns over the `ff_types.Geometry` protocol. It does not know
   the phantom families, except for the stop kinds (see `stop_kinds`).
3. A CLI that writes OUT_DIR/<name>/truth.npz and truth.json for each phantom except "noise".

The truth is a definition
-------------------------
The true columns are the orthogonal trajectories of the phantom's depth function φ. This is a
choice of definition, as `bent_phantom.py` says for its depth function: another depth function
with the same white surface and stop surface gives other true columns. Each true column starts at
a root on φ = 0 (the white surface) and follows ∇φ / |∇φ| toward larger φ. It stops at the FIRST
point where `stop_sdf` reaches 0 (the true stop surface). Thus the columns never enter the stopped
region. The true τ is (φ − 1) · tau_scale (`ff_types.true_tau`) for every phantom.

Per family:

- **slab**: φ = x / t. The columns are straight lines along +x. They stop on the pia φ = 1.
- **crown**: φ = (r − w) / t, r the distance from the half-line. The columns are straight and
  radial. They stop on the pia φ = 1 (r = w + t).
- **tau**: φ = x / t(u). ∇φ has a u part where t changes, so the columns bend a small amount.
  They stop on the pia x = t(u), which is φ = 1.
- **fundus** (`fold_phantoms` fundus): φ = 1 − (r − g) / t, r the distance from the half-plane
  K. The columns are straight. They stop on the slot's edge r = g, which is φ = 1. As g goes to
  0, the columns at the round end converge on the slot's edge line. That line is part of the stop
  surface, so the columns that meet there do not make a focal point.
- **curved** (`bent_phantom.Curved`): φ = 1 − (r − g) / t(θ). The columns bend round the fundus.
  They stop on r = g (φ = 1). Where tA ≠ tB, ∂φ/∂θ > 0 at the round end, so the columns there
  move toward bank A: a column from a bank-B root near θ = 0 can stop at θ > 0. The bank of a
  column is the bank of its root.
- **ckiss** (the connected kiss): φ = 1 − (r − g(z)) / t(θ). ∇φ has a z part where g changes.
  The columns stop on the slot's face r = g(z) (φ = 1). Where g < VOXEL_MM the two faces are a
  fused contact: each bank's column stops on its own face, so no column crosses the contact.
- **abraded**: the curved fundus (g = 0.16) with discs of bank A removed. φ is that of the curved
  fundus. A column that meets a patch stops on the patch's face, with φ < 1 there. Its true jump
  is τ_top − τ(φ at the face) = (1 − φ_face) · tau_scale. On the flat bank this is
  depth / tA · tau_scale.

Roots
-----
The roots are points on φ = 0, from a cubic lattice in the local frame (spacing
root_spacing_mm, with a fixed sub-spacing offset so that no lattice plane is on a closed-form
surface). Each lattice point within one spacing of φ = 0 (first order: |φ| < spacing · |∇φ|) moves
to φ = 0 by Newton steps along ∇φ. A point is kept if it moved less than one spacing and reached
|φ| < 1e-10. Then a greedy thinning in the lattice order keeps a point only if no kept point is
closer than one spacing. So no two roots are closer than root_spacing_mm. At spacing 0.32 mm, the
distance to the nearest root is the spacing (to 5 %) for at least 95 % of the roots of each
phantom, and at most 1.4 spacings. On a flat white surface the roots are a square grid. Reason for this choice: it uses only φ and ∇φ, so one code
serves every phantom, and the order is fixed, so the result is deterministic. A per-family
parametrisation of the white surface would give more regular roots, but needs code for each family.

A root is kept only if its start point (the root − START_DEPTH_MM along the normal), every point
of its true column and its stop are inside the box, at least margin_mm from the range of the voxel
centres.

Stop kinds (`ff_types.StopKind`), in this order:

1. ABRADED: the geometry has `patch_of` and the stop is on a patch (patch_of of the point 1e-3 mm
   beyond the stop, along the column, is ≥ 0, or φ at the stop is < 1 − 1e-6).
2. HAIRPIN: the slot families (fundus, curved, ckiss, abraded), and the stop's local y < 0 (the
   round end of the slot).
3. CONTACT: the slot families, and the slot's half-width at the stop is below VOXEL_MM (ckiss:
   geometry.g(z); fundus and curved: p["g"]).
4. PIA: all other stops. Slab, crowns and tau have only PIA stops.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bent_phantom as BP  # noqa: E402
import ff_ckiss as CK  # noqa: E402
import ff_simple as S  # noqa: E402
import ff_tau as TAU  # noqa: E402
import ff_types as T  # noqa: E402

H = T.VOXEL_MM
SMAX_MM = 12.0  # the longest true column (mm); a longer one is an error
STOP_TOL_MM = 1e-12  # the bisection of the last step stops when its bracket is below this
BEYOND_MM = 1e-3  # the test point beyond a stop, for the ABRADED kind
_LATTICE_OFFSET = np.array([0.37, 0.41, 0.29])  # in spacings: no lattice plane on a closed-form surface
_NEWTON = 40

PHANTOMS: tuple[str, ...] = (
    "slab", "noise",
    "crown_w0.32", "crown_w0.64", "crown_w1.28",
    "ckiss_smooth", "ckiss_sharp", "ckiss_smooth_flat", "ckiss_sharp_flat",
    "fundus_g0.48", "fundus_g0.16", "fundus_g0.04",
    "curved_g0.16", "curved_g0.04",
    "abraded", "tau",
)
SLOT_PREFIXES = ("fundus_", "curved_", "ckiss_", "abraded")


# ---------------------------------------------------------------- the registry


def build_phantom(name: str, feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
                  shape: tuple[int, int, int] | None = None) -> T.Phantom:
    """The phantom `name` of PHANTOMS, with its image (from ff_simple, ff_ckiss or ff_tau)."""
    if name not in PHANTOMS:
        raise KeyError(name)
    if name in CK.NAMES:
        return CK.ckiss(name, feats, site=site, noise=noise, seed=seed, shape=shape)
    if name == "tau":
        return TAU.tau(feats, site=site, noise=noise, seed=seed, shape=shape)
    return S.build(name, feats, site=site, noise=noise, seed=seed, shape=shape)


def geometry_of(name: str, shape: tuple[int, int, int] | None = None) -> tuple[Any, float]:
    """(geometry, tau_scale) of the phantom `name`, with no image. The geometry is the one that
    build_phantom gives. "noise" has no geometry: KeyError."""
    if name not in PHANTOMS or name == "noise":
        raise KeyError(name)
    if name in CK.NAMES:
        geo = CK.geometry("sharp" in name, shape)
        geo.p["flat"] = name.endswith("_flat")
        return geo, CK.TAU_SCALE
    if name == "tau":
        return TAU.geometry(TAU.SHAPE if shape is None else tuple(int(s) for s in shape)), TAU.T0  # type: ignore[arg-type]
    geo, ts, _ = S.geometry(name, shape)
    return geo, float(ts)


# ---------------------------------------------------------------- roots


def _box_limits(geo: Any, margin_mm: float) -> tuple[T.Array, T.Array]:
    shp = np.array(geo.shape, float)
    return np.full(3, margin_mm / H), shp - 1.0 - margin_mm / H


def _inside(X: T.Array, lo: T.Array, hi: T.Array) -> T.BoolArray:
    return np.asarray(np.all((X >= lo) & (X <= hi), axis=-1))


def _unit_grad(geo: Any, Y: T.Array) -> T.Array:
    G = geo.grad_phi(Y)
    return np.asarray(G / np.linalg.norm(G, axis=1)[:, None])


def root_candidates(geo: Any, spacing: float, margin_mm: float) -> T.Array:
    """Local mm points on φ = 0, inside the box by margin_mm, no two closer than `spacing`
    (the module docstring, *Roots*). Deterministic."""
    lo, hi = _box_limits(geo, margin_mm)
    corners = np.array([[lo[0] if i == 0 else hi[0], lo[1] if j == 0 else hi[1],
                         lo[2] if k == 0 else hi[2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    C = geo.to_local(corners)
    a, b = C.min(0) - spacing, C.max(0) + spacing
    axes = [a[d] + spacing * (np.arange(int(np.ceil((b[d] - a[d]) / spacing)) + 1) + _LATTICE_OFFSET[d])
            for d in range(3)]
    cands = []
    for x in axes[0]:  # one plane at a time, to bound the memory
        Y = np.stack(np.meshgrid([x], axes[1], axes[2], indexing="ij"), -1).reshape(-1, 3)
        ph = geo.phi(Y)
        gn = np.linalg.norm(geo.grad_phi(Y), axis=1)
        near = np.abs(ph) < spacing * gn
        if near.any():
            cands.append(Y[near])
    if not cands:
        return np.zeros((0, 3))
    Y0 = np.concatenate(cands)
    Y = Y0.copy()
    for _ in range(_NEWTON):
        G = geo.grad_phi(Y)
        Y = Y - (geo.phi(Y) / np.sum(G * G, 1))[:, None] * G
    ok = (np.abs(geo.phi(Y)) < 1e-10) & (np.linalg.norm(Y - Y0, axis=1) < spacing)
    ok &= _inside(geo.to_vox(Y), lo, hi)
    return _thin(Y[ok], spacing)


def _thin(Y: T.Array, r: float) -> T.Array:
    """Greedy thinning in the given order: keep a point if no kept point is closer than r."""
    cell: dict[tuple[int, int, int], list[int]] = {}
    keep: list[int] = []
    key = np.floor(Y / r).astype(np.int64)
    r2 = (r * (1.0 - 1e-6)) ** 2
    nb = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for n in range(len(Y)):
        kx, ky, kz = (int(v) for v in key[n])
        y = Y[n]
        bad = False
        for dx, dy, dz in nb:
            for m in cell.get((kx + dx, ky + dy, kz + dz), ()):
                d = Y[m] - y
                if d @ d < r2:
                    bad = True
                    break
            if bad:
                break
        if not bad:
            cell.setdefault((kx, ky, kz), []).append(n)
            keep.append(n)
    return Y[np.array(keep, int)] if keep else np.zeros((0, 3))


# ---------------------------------------------------------------- the trace


def _last_step(geo: Any, Y: T.Array, h: float) -> T.Array:
    """The RK4 step from Y (stop_sdf < 0) of the length s in [0, h] at which stop_sdf crosses 0
    first, by bisection on s, so that the end stays on the trajectory. Returns the end on the
    tissue side of the bracket: stop_sdf ≤ 0 and |stop_sdf| below about STOP_TOL_MM."""
    n = len(Y)
    lo, hi = np.zeros(n), np.full(n, h)
    one = np.ones(n)
    while (hi - lo).max() > STOP_TOL_MM:
        mid = 0.5 * (lo + hi)
        s = geo.stop_sdf(BP._rk4(geo, Y, one, mid[:, None]))
        tissue = s < 0
        lo = np.where(tissue, mid, lo)
        hi = np.where(tissue, hi, mid)
    return np.asarray(BP._rk4(geo, Y, one, lo[:, None]))


def trace(geo: Any, Y0: T.Array, h: float = T.TRUTH_STEP_MM, smax: float = SMAX_MM) -> tuple[T.Array, T.IntArray]:
    """The orthogonal trajectories of φ from the local points Y0 toward larger φ, RK4 at step h,
    each to the first crossing of stop_sdf = 0. All points step together. A full step that ends
    within STOP_TOL_MM of the stop surface (on the tissue side) is the last step. Returns the local
    points (n, m, 3), nan-padded, and the count of points of each trace (root and stop included).
    A RuntimeError if a trace is longer than smax."""
    n = len(Y0)
    one = np.ones(n)
    rows = [np.asarray(Y0, float).copy()]
    Y = rows[0].copy()
    counts = np.ones(n, np.int64)
    active = np.arange(n)
    k = 0
    while active.size:
        k += 1
        if k * h > smax:
            raise RuntimeError(f"{active.size} true columns longer than {smax} mm")
        Yn = BP._rk4(geo, Y[active], one[active], h)
        sd = geo.stop_sdf(Yn)
        cross = sd >= 0
        on = ~cross & (sd > -STOP_TOL_MM)  # a full step that ends on the stop surface
        go = active[~cross & ~on]
        row = np.full((n, 3), np.nan)
        Y[go] = Yn[~cross & ~on]
        row[go] = Y[go]
        row[active[on]] = Yn[on]
        end = active[cross]
        if end.size:
            row[end] = _last_step(geo, Y[end], h)
        counts[active] += 1
        rows.append(row)
        active = go
    return np.stack(rows, axis=1), counts


# ---------------------------------------------------------------- stop kinds


def stop_kinds(geo: Any, name: str, stops: T.Array, beyond: T.Array) -> Any:
    """The StopKind of each stop (local mm), by the rule of the module docstring. beyond: the
    points BEYOND_MM past each stop along its column."""
    n = len(stops)
    kind = np.full(n, int(T.StopKind.PIA), np.int8)
    slot = name.startswith(SLOT_PREFIXES)
    abr = np.zeros(n, bool)
    if hasattr(geo, "patch_of"):
        abr = (np.asarray(geo.patch_of(beyond)) >= 0) | (geo.phi(stops) < 1.0 - 1e-6)
    if slot:
        hair = stops[:, 1] < 0
        gw = geo.g(stops[:, 2]) if callable(getattr(geo, "g", None)) else np.full(n, geo.p["g"])
        contact = ~hair & (np.asarray(gw) < T.VOXEL_MM)
        kind[contact] = int(T.StopKind.CONTACT)
        kind[hair] = int(T.StopKind.HAIRPIN)
    kind[abr] = int(T.StopKind.ABRADED)
    return kind


# ---------------------------------------------------------------- the truth


def truth(geometry: Any, tau_scale: float, name: str, root_spacing_mm: float = 0.32,
          margin_mm: float = 1.0, step_mm: float = T.TRUTH_STEP_MM) -> T.Truth:
    """The truth of one phantom under the stop rule (the module docstring)."""
    geo = geometry
    lo, hi = _box_limits(geo, margin_mm)
    Yr = root_candidates(geo, root_spacing_mm, margin_mm)
    Nl = _unit_grad(geo, Yr)
    roots = geo.to_vox(Yr)
    normals = geo.dir_to_vox(Nl)
    starts = roots - (T.START_DEPTH_MM / H) * normals
    ok = _inside(starts, lo, hi)
    Yr, Nl, roots, normals, starts = Yr[ok], Nl[ok], roots[ok], normals[ok], starts[ok]
    P, counts = trace(geo, Yr, step_mm)
    m = P.shape[1]
    Pv = geo.to_vox(P.reshape(-1, 3)).reshape(P.shape)
    valid = np.arange(m)[None, :] < counts[:, None]
    ok = np.all(_inside(Pv, lo, hi) | ~valid, axis=1)
    Yr, roots, normals, starts, P, Pv, counts = Yr[ok], roots[ok], normals[ok], starts[ok], P[ok], Pv[ok], counts[ok]
    mm = int(counts.max()) if len(counts) else 1
    P, Pv = P[:, :mm], Pv[:, :mm]
    idx = np.arange(len(counts))
    stops_l = P[idx, counts - 1]
    beyond = stops_l + BEYOND_MM * _unit_grad(geo, stops_l) if len(idx) else stops_l
    kind = stop_kinds(geo, name, stops_l, beyond)
    stops = Pv[idx, counts - 1]
    tau_arr = T.true_tau(geo, tau_scale, stops)
    return T.Truth(
        phantom=name, roots=roots, normals=normals, starts=starts, paths=Pv, counts=counts,
        stops=stops, stop_kind=np.asarray(kind, np.int8), bank=np.asarray(geo.bank(Yr), np.int8),
        tau_arrival=tau_arr, jump=T.TAU_TOP - tau_arr, offset=T.true_tau(geo, tau_scale, starts),
        tau_scale=float(tau_scale), step_mm=float(step_mm), root_spacing_mm=float(root_spacing_mm))


# ---------------------------------------------------------------- the CLI


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--names", nargs="+", default=[n for n in PHANTOMS if n != "noise"])
    ap.add_argument("--spacing", type=float, default=0.32)
    a = ap.parse_args(argv)
    for name in a.names:
        if name == "noise":
            continue
        t0 = time.perf_counter()
        geo, ts = geometry_of(name)
        tr = truth(geo, ts, name, root_spacing_mm=a.spacing)
        dt = time.perf_counter() - t0
        d = a.out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        tr.to_npz(d / "truth.npz")
        kinds = {k.name: int((tr.stop_kind == k).sum()) for k in T.StopKind}
        rec = dict(phantom=name, n_roots=tr.n, stop_kinds=kinds,
                   banks={str(b): int((tr.bank == b).sum()) for b in (0, 1)},
                   max_count=int(tr.counts.max()), root_spacing_mm=a.spacing, step_mm=tr.step_mm,
                   tau_scale=ts, shape=list(geo.shape), time_s=round(dt, 2),
                   peak_rss_bytes_so_far=_peak_rss_bytes())
        (d / "truth.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
