"""Test the radius of a local preview solve of the antialiased field (docs/preview-radius.md).

A preview of a dseg edit solves the field of baillarger.antialias only within R voxels of the
changed voxels, and holds it at the previous solution beyond R. This script measures, on the
real white mask, how the preview's error near the edit falls with R, with and without the sign
constraints, for edits of several sizes at sites of several band half-widths c.

Two steps:

  solve DSEG_A OUT.npy [--box-with DSEG]
      The converged field of DSEG_A's WM mask on white_surf's box (the joint box with
      --box-with, if given), as antialias.thin_plate computes it.
  test DSEG_A PHI_A.npy OUT.json [--box-with DSEG]
      Sites: 20,000 random boundary voxels (seed 0); c at a site is the largest distance from a
      free (band) voxel to a non-free one within 4 voxels; two sites in each class of c
      (CLASSES). At each site, a ball of radius 1, 2, 3, 5 and 8 is set inside (bump) and
      outside (dent) the mask. For each edit and R, the field is solved (antialias.thin_plate)
      in the box of the changed voxels plus R + S + 2, the voxels farther than R from the
      changed voxels held at the previous field; the reference is R = 56. For the ball of radius 3 the same sweep is also
      run without the sign constraints (the linear problem; previous field = its own solution
      for A in the site's box; reference R = 40). Recorded: the largest |phi_R - phi_ref| over
      band voxels within S = BAND + 3 of the changed voxels (field error, voxels), the largest
      vertex shift along the grid edges there, iterations, and the share of active sign
      constraints near the edit (|phi| < EPS + 1e-4, within 14 voxels).

--box-with: the runs in docs/preview-radius.md solved phi_A on the joint box of the current
dseg and the dseg from the three-echo excisions; the sites depend on the box.
Run each step under scripts/memguard (about 22 GB peak on the right hemisphere).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from baillarger import antialias as AA
from baillarger import backend
from baillarger.bids import read_sidecar, sidecar_path
from baillarger.blocks import open_volume
from baillarger.white_dseg import DSEG_SPEC

T0 = time.perf_counter()
DEV = backend.device(None)
S = int(AA.BAND + 3)  # sd changes only within S voxels of the changed voxels
CLASSES = ((4.5, 5.5), (6.5, 7.5), (8.5, 11.0))  # classes of c (voxels)
PER_CLASS = 2
SITE_MARGIN = 56 + S + 2 + 3 + 2
RS = (8, 10, 12, 14, 16, 20, 24, 32)
RS_LIN = (8, 10, 12, 14, 16, 20, 24)
R_REF, R_REF_LIN = 56, 40
RADII = (1, 2, 3, 5, 8)
ACTIVE = 1e-4


def log(msg):
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


def wm_mask(dseg):
    side = read_sidecar(sidecar_path(Path(dseg)), DSEG_SPEC)
    vol = open_volume(Path(dseg))
    return np.isin(vol.read(tuple(slice(0, n) for n in vol.shape)), side["FilledLabels"])


def box(masks):
    """white_surf's crop: the masks' joint box plus BAND + 4, as slices."""
    m = int(np.ceil(AA.BAND)) + 4
    anyv = np.logical_or.reduce(masks)
    out = []
    for i in range(3):
        t = np.flatnonzero(anyv.any(axis=tuple(j for j in range(3) if j != i)))
        out.append(slice(max(int(t[0]) - m, 0), min(int(t[-1]) + 1 + m, anyv.shape[i])))
    return tuple(out)


def cropped_mask(dseg, box_with):
    ma = wm_mask(dseg)
    masks = [ma] if box_with is None else [ma, wm_mask(box_with)]
    return np.ascontiguousarray(ma[box(masks)])


def iterate(m, start=None, held=None, signs=True):
    """antialias.thin_plate on m: from start (default sd), with held voxels at start;
    with signs, phi >= EPS on the mask and <= -EPS off it (without, eps = -inf removes
    the sign constraints: the linear problem). Returns phi and the iterations."""
    eps = AA.EPS if signs else -math.inf
    phi, solve = AA.thin_plate(m, DEV, eps=eps, start=start, held=held)
    return phi, solve.iterations


def take(a, sl, fill):
    """a[sl], with fill beyond a's edges."""
    out = np.full(tuple(t.stop - t.start for t in sl), fill, a.dtype)
    src = tuple(slice(max(t.start, 0), min(t.stop, n)) for t, n in zip(sl, a.shape, strict=True))
    dst = tuple(slice(u.start - t.start, u.stop - t.start) for t, u in zip(sl, src, strict=True))
    out[dst] = a[src]
    return out


def sd_of(m, big, mb, p):
    """sd of mb (the edited box big of m), with p voxels of m round it (no truncation)."""
    pad = tuple(slice(s.start - p, s.stop + p) for s in big)
    mp = take(m, pad, False)
    mp[tuple(slice(p, -p) for _ in range(3))] = mb
    return AA.signed_distance(mp, AA.BAND + 3.0)[tuple(slice(p, -p) for _ in range(3))]


def vertex_t(m, phi):
    """Per axis: the boundary edges and their vertex positions t (from the lower voxel)."""
    out = []
    for ax in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        e = m[tuple(a)] != m[tuple(b)]
        pa, pb = phi[tuple(a)][e], phi[tuple(b)][e]
        out.append((np.argwhere(e), pa / (pa - pb)))
    return out


def vertex_shift(m, phi, ref, near):
    """The largest |t - t_ref| over the boundary edges whose lower voxel is near."""
    d = [np.abs(t - tr)[near[tuple(ij.T)]]
         for (ij, t), (_, tr) in zip(vertex_t(m, phi), vertex_t(m, ref), strict=True)]
    d = np.concatenate(d)
    return float(d.max()) if d.size else 0.0


def sweep(mb, sdb, dist, prev, rs, rref, signs):
    """The preview at each R (and rref) from prev: {R: (phi in the site's box, iterations)}."""
    out = {}
    e = np.argwhere(dist == 0)
    for R in (*rs, rref):
        # the margin of antialias.local_solve: sd of the box is exact within R
        lo = np.maximum(e.min(0) - R - S - 2, 0)
        hi = np.minimum(e.max(0) + R + S + 3, np.array(mb.shape))
        sb = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        hold = dist[sb] > R
        start = np.asarray(prev[sb], np.float32)
        phi, it = iterate(np.ascontiguousarray(mb[sb]), start, hold, signs)
        full = prev.copy()
        full[sb] = phi
        out[R] = (full, it)
    return out


def sites_of(m):
    sd = AA.signed_distance(m, AA.BAND + 3.0)
    half = ndi.distance_transform_edt(np.abs(sd) <= AA.BAND).astype(np.float32)
    del sd
    bnd = np.argwhere(AA.next_to_boundary(m) & m)
    rng = np.random.default_rng(0)
    cand = bnd[rng.choice(len(bnd), 20000, replace=False)]
    keep = np.all((cand >= SITE_MARGIN) & (cand < np.array(m.shape) - SITE_MARGIN), axis=1)
    cand = cand[keep]
    cw = np.array([half[tuple(slice(p - 4, p + 5) for p in q)].max() for q in cand])
    out = []
    for lo_c, hi_c in CLASSES:
        idx = np.flatnonzero((cw >= lo_c) & (cw <= hi_c))
        for i in rng.choice(idx, min(PER_CLASS, len(idx)), replace=False):
            out.append((cand[i], float(cw[i])))
    return out


def run_solve(a):
    m = cropped_mask(a.dseg, a.box_with)
    log(f"box {m.shape}")
    phi, it = iterate(m)
    np.save(a.out, phi)
    log(f"converged in {it} iterations: {a.out}")


def run_test(a):
    m = cropped_mask(a.dseg, a.box_with)
    phi_a = np.load(a.phi, mmap_mode="r")
    if phi_a.shape != m.shape:
        raise ValueError(f"phi {phi_a.shape} is not on the mask's box {m.shape}")
    sites = sites_of(m)
    log(f"sites: {[(q.tolist(), round(w, 2)) for q, w in sites]}")
    res = dict(S=S, active_tol=ACTIVE, edits=[])
    for q, cwid in sites:
        h = R_REF + S + 2 + max(RADII) + 2
        big = tuple(slice(p - h, p + h + 1) for p in q)
        ma = take(m, big, False)
        pa = take(phi_a, big, np.float32(-(AA.BAND + 3.0))).astype(np.float32)
        sda = sd_of(m, big, ma, S + 2)
        for rad in RADII:
            g = np.mgrid[-rad : rad + 1, -rad : rad + 1, -rad : rad + 1]
            ball = np.sum(g**2, axis=0) <= rad * rad
            bs = tuple(slice(h - rad, h + rad + 1) for _ in range(3))
            for kind in ("bump", "dent"):
                mb = ma.copy()
                if kind == "bump":
                    mb[bs] |= ball
                else:
                    mb[bs] &= ~ball
                changed = ma != mb
                if not changed.any():
                    continue
                dist = ndi.distance_transform_edt(~changed)
                sdb = sd_of(m, big, mb, S + 2)
                near = (dist <= S) & (np.abs(sdb) <= AA.BAND)
                rec = dict(site=q.tolist(), c=cwid, kind=kind, radius=rad,
                           changed=int(changed.sum()), con={}, lin={})
                sol = sweep(mb, sdb, dist, pa, RS, R_REF, True)
                ref = sol[R_REF][0]
                zone = (dist <= 14) & (np.abs(sdb) <= AA.BAND)
                act = zone & (np.abs(ref) < AA.EPS + ACTIVE)
                nb = zone & AA.next_to_boundary(mb)
                rec["active"] = dict(band_voxels=int(zone.sum()), active=int(act.sum()),
                                     next_to_boundary=int(nb.sum()),
                                     active_next_to_boundary=int((act & nb).sum()))
                for R in RS:
                    phi, it = sol[R]
                    rec["con"][R] = dict(field=float(np.abs(phi - ref)[near].max()),
                                         vertex=vertex_shift(mb, phi, ref, dist <= S),
                                         iterations=it)
                if rad == 3:
                    # the rim where sd of the box is not exact is held at sda
                    rim = np.ones(ma.shape, bool)
                    rim[S + 1 : -S - 1, S + 1 : -S - 1, S + 1 : -S - 1] = False
                    lin_a, _ = iterate(ma, sda, rim, False)
                    lsol = sweep(mb, sdb, dist, lin_a, RS_LIN, R_REF_LIN, False)
                    lref = lsol[R_REF_LIN][0]
                    for R in RS_LIN:
                        phi, it = lsol[R]
                        rec["lin"][R] = dict(field=float(np.abs(phi - lref)[near].max()),
                                             iterations=it)
                res["edits"].append(rec)
                log(f"{kind} r{rad} at {q.tolist()} c {cwid:.2f}: "
                    + " ".join(f"{R}:{rec['con'][R]['field']:.1e}" for R in RS))
                Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    s = sub.add_parser("solve")
    s.add_argument("dseg")
    s.add_argument("out")
    s.add_argument("--box-with")
    t = sub.add_parser("test")
    t.add_argument("dseg")
    t.add_argument("phi")
    t.add_argument("out")
    t.add_argument("--box-with")
    a = ap.parse_args()
    log(f"device {DEV}")
    run_solve(a) if a.step == "solve" else run_test(a)


if __name__ == "__main__":
    main()
