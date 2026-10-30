"""Prototype: the white mesh by constrained (sign-preserving) smoothing of a field, in
place of smoothing the mesh and capping its moves.

Whitaker (2000), "Reducing aliasing artifacts in iso-surfaces of binary volumes" (ITK's
AntiAliasBinaryImageFilter): a binary mask fixes only which voxel centres are inside, so smooth a
field phi while keeping every voxel's sign, then contour it. Here phi starts as the clipped signed
distance of the WM mask; each iteration blurs it (Gaussian, sigma voxels) and clamps it back:
phi >= eps inside the mask, phi <= -eps outside. Marching tetrahedra of phi at 0 then has the
binary mesh's triangles (they follow from the signs alone) with each vertex moved along its own
grid edge: no self-intersections by construction (one piecewise-linear level set).

Compares with the binary (raw) mesh and the current pipeline's mesh (smoothed, capped): the edge
keys, self-intersections, the angle between neighbouring vertices' normals (mesh normals, and
normals from phi's gradient), the area, the distance from the raw mesh and from the current one,
and the time of each step.

Method thinplate (the default; Lempitsky, CVPR 2010, "Surface extraction from binary volumes
with higher-order smoothness"): minimise the thin-plate energy of phi (its squared second
derivatives) in a band round the boundary under the same sign constraints, so that phi is nearly
linear near the surface, which is what makes marching tetrahedra's triangles smooth; solved by
accelerated projected gradient in PyTorch. Method diffusion: blur and clamp (iters times), the
first attempt, which leaves much of the staircase.

Usage: antialias.py <dseg.nii.gz> <current white.surf.gii> <out dir>
                    [--method thinplate|diffusion] [--device mps] [--iters N] [--tol T]
                    [--sigma S] [--eps E] [--band B]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from baillarger import mesh_intersect as mi
from baillarger import mesh_smoothing as S
from baillarger.bids import read_sidecar, sidecar_path
from baillarger.blocks import open_volume
from baillarger.mesh import edge_key, edge_points, marching_tetrahedra_blocks
from baillarger.white import DSEG_SPEC, WhiteSurface

T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


def _stencils():
    """The six second-derivative stencils of the thin-plate energy (3 x 3 x 3) and their
    weights: phi_xx, phi_yy, phi_zz (1), phi_xy, phi_xz, phi_yz (2)."""
    out = []
    for a in range(3):
        k = np.zeros((3, 3, 3), np.float32)
        idx = [1, 1, 1]
        for t, w in ((0, 1.0), (1, -2.0), (2, 1.0)):
            idx[a] = t
            k[tuple(idx)] = w
        out.append((k, 1.0))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        k = np.zeros((3, 3, 3), np.float32)
        for sa in (-1, 1):
            for sb in (-1, 1):
                idx = [1, 1, 1]
                idx[a] += sa
                idx[b] += sb
                k[tuple(idx)] = sa * sb / 4.0
        out.append((k, 2.0))
    return out


def fused_taps():
    """The gradient's one stencil, 2 sum_s w_s (k_s * k_s) over the six second-derivative
    stencils (5 x 5 x 5), as (offset, weight) for its nonzero taps. Applying it equals applying
    each D then D^T except within 2 voxels of the volume's edge (there zero padding differs),
    which lies in the fixed region."""
    from scipy.signal import convolve

    K = np.zeros((5, 5, 5))
    for k, w in _stencils():
        K += 2.0 * w * convolve(k.astype(np.float64), k.astype(np.float64))
    idx = np.argwhere(np.abs(K) > 1e-12)
    return [(tuple(int(t) - 2 for t in i), float(K[tuple(i)])) for i in idx]


def thin_plate(
    sdf,
    mask,
    band,
    eps,
    device,
    iters,
    tol,
    fixed=None,
    fused=True,
    stop="surface",
    start="sdf",
    levels=0,
    pos=None,
    neg=None,
    quiet=False,
    restart=False,
):
    """Minimise the thin-plate energy sum (phi_xx^2 + ... + 2 phi_xy^2 + ...) over the band
    |fixed| <= band, subject to phi >= eps where pos and phi <= -eps where neg (default: pos
    the mask, neg the rest), phi = fixed outside the band: accelerated projected gradient
    (FISTA) with PyTorch.

    fused: one 5 x 5 x 5 stencil (fused_taps) for the gradient, else each stencil then its
    transpose. stop: "surface" stops when the largest change at the voxels next to the
    boundary (they set the vertices) is under tol; "band" when the largest change anywhere in
    the band is. start: "sdf" or "blur" (sdf blurred once, sigma 1 voxel). levels: solve first
    on a grid 2 times coarser (recursively), a coarse voxel constrained only where its 8 fine
    voxels agree, and start from that solution upsampled. restart: adaptive restart of the
    momentum (O'Donoghue and Candes 2015, gradient scheme): when the step taken goes against
    the momentum, t starts again at 1. The stopping test is on the mean change per iteration
    over each window of 25 iterations (a single step can be tiny just after a restart; and
    reading back waits for the device)."""
    import torch
    import torch.nn.functional as Fn

    dev = torch.device(device)
    ref = sdf if fixed is None else fixed
    pos = mask if pos is None else pos
    neg = ~mask if neg is None else neg
    shape = tuple(int(n) for n in np.shape(ref))
    info = dict(levels=[])

    init = np.asarray(sdf, np.float32)
    if start == "blur":
        init = ndi.gaussian_filter(init, 1.0).astype(np.float32)
    if levels > 0:
        # the coarse problem: pad to even sizes, average 2 x 2 x 2
        pad = [(0, n % 2) for n in shape]

        def down(arr, how):
            q = np.pad(arr, pad, mode="edge")
            r = q.reshape(q.shape[0] // 2, 2, q.shape[1] // 2, 2, q.shape[2] // 2, 2)
            return how(r, axis=(1, 3, 5))

        cref = down(np.asarray(ref, np.float32), np.mean).astype(np.float32)
        cpos = down(np.asarray(pos), np.all)
        cneg = down(np.asarray(neg), np.all)
        cphi, cinfo = thin_plate(
            cref, None, band, eps, device, iters, tol, fixed=cref, fused=fused,
            stop="band", start=start, levels=levels - 1, pos=cpos, neg=cneg, quiet=quiet,
            restart=restart,
        )
        up = Fn.interpolate(
            torch.as_tensor(cphi)[None, None], scale_factor=2, mode="trilinear",
            align_corners=False,
        )[0, 0].numpy()
        init = up[: shape[0], : shape[1], : shape[2]].astype(np.float32)
        info["levels"] = cinfo["levels"]

    if fused:
        taps = fused_taps()

        def grad(x):
            p = Fn.pad(x[None, None], (2, 2, 2, 2, 2, 2))[0, 0]
            D, H, W = x.shape
            g = torch.zeros_like(x)
            for (di, dj, dk), c in taps:
                g.add_(p[2 + di : 2 + di + D, 2 + dj : 2 + dj + H, 2 + dk : 2 + dk + W], alpha=c)
            return g

    else:
        weights = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

        def stencil(x, i):
            """Stencil i of x (zero outside): shifted slices of a padded copy (on MPS about
            120 times faster than conv3d; antialias_profile.py)."""
            p = Fn.pad(x[None, None], (1, 1, 1, 1, 1, 1))[0, 0]
            D, H, W = x.shape

            def sl(d):
                return p[1 + d[0] : 1 + d[0] + D, 1 + d[1] : 1 + d[1] + H, 1 + d[2] : 1 + d[2] + W]

            if i < 3:
                e = [0, 0, 0]
                e[i] = 1
                return sl(e) - 2.0 * x + sl([-t for t in e])
            a, b = ((0, 1), (0, 2), (1, 2))[i - 3]

            def at(sa, sb):
                d = [0, 0, 0]
                d[a], d[b] = sa, sb
                return sl(d)

            return 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))

        def grad(x):
            # each stencil is symmetric, so D^T is D
            g = torch.zeros_like(x)
            for i, w in enumerate(weights):
                g += (2.0 * w) * stencil(stencil(x, i), i)
            return g

    # the Lipschitz constant of grad: power iteration on a small volume
    z = torch.randn((24, 24, 24), device=dev)
    for _ in range(60):
        z = grad(z)
        lam = float(torch.linalg.vector_norm(z))
        z = z / lam
    L = 1.05 * lam

    free = torch.as_tensor(np.abs(ref) <= band, device=dev)
    lo = torch.as_tensor(np.asarray(pos), device=dev) & free
    hi = torch.as_tensor(np.asarray(neg), device=dev) & free
    fixed_t = torch.as_tensor(np.asarray(ref, np.float32), device=dev)
    if stop == "surface":
        # voxels with a 6-neighbour of the other sign: they set the vertices
        P = np.pad(np.asarray(pos), 1)
        N = np.pad(np.asarray(neg), 1)
        near = np.zeros(shape, bool)
        c = (slice(1, -1),) * 3
        for ax in range(3):
            for d in (-1, 1):
                sh = [slice(1, -1)] * 3
                sh[ax] = slice(1 + d, P.shape[ax] - 1 + d)
                sh = tuple(sh)
                near |= (P[c] & N[sh]) | (N[c] & P[sh])
        watch = torch.as_tensor(near, device=dev) & free
    else:
        watch = free

    def project(x):
        x = torch.where(free, x, fixed_t)
        x = torch.where(lo, torch.clamp(x, min=eps), x)
        return torch.where(hi, torch.clamp(x, max=-eps), x)

    x = project(torch.as_tensor(init, device=dev))
    y, t = x.clone(), 1.0
    snap = x.clone()  # x 25 iterations ago: the stopping test is on the change over the window
    step = float("inf")
    t0 = time.perf_counter()
    restarts = 0
    for k in range(iters):
        xn = project(y - grad(y) / L)
        if restart and k > 0 and float(torch.sum((y - xn) * (xn - x))) > 0:
            t = 1.0
            restarts += 1
        tn = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        check = k % 25 == 0
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
        if check and k > 0:
            # the mean change per iteration over the last 25: a single step can be tiny just
            # after a restart, far from the optimum
            win = torch.abs(x - snap)[watch]
            step = float(torch.max(win)) / 25.0 if win.numel() else 0.0
            snap = x.clone()
        if check:
            if not quiet and k % 100 == 0:
                log(f"  thin plate {shape} "
                    f"iteration {k}: max change {step:.2e}")
            if k > 0 and step < tol:
                break
    if dev.type == "mps":
        torch.mps.synchronize()
    info["levels"] = info["levels"] + [
        dict(shape=list(np.shape(ref)), iterations=k + 1, last_change=step,
             seconds=round(time.perf_counter() - t0, 2), restarts=restarts)
    ]
    info.update(L=L, iterations=k + 1, last_change=step)
    return x.cpu().numpy(), info


def thin_plate_band(
    sdf, mask, band, eps, device, iters, tol, block=64, chunk=48, quiet=False, fixed=None
):
    """thin_plate (separate stencils, restart, surface stopping, start from sdf), computed only
    on the blocks (block voxels a side, from voxel 0) that hold band voxels: the same problem,
    since the other blocks never change. State is kept for the active blocks only; each
    iteration gathers them with a 2-voxel halo (two stencil passes reach 2) from a global
    field of the current momentum point, takes the projected, accelerated step on their
    interiors, and writes the interiors back. Blocks are gathered by index in chunks of
    chunk blocks. Outside the volume is zero, as in thin_plate."""
    import torch
    import torch.nn.functional as Fn

    dev = torch.device(device)
    ref = np.asarray(sdf if fixed is None else fixed, np.float32)
    shape = ref.shape
    B, h = block, 2
    nb = [-(-n // B) for n in shape]
    gshape = [n * B + 2 * h for n in nb]  # the global field: blocks plus a halo of zeros
    free_v = np.abs(ref) <= band

    def pad_global(arr, fill):
        out = np.full(gshape, fill, arr.dtype)
        out[h : h + shape[0], h : h + shape[1], h : h + shape[2]] = arr
        return out

    free_g = pad_global(free_v, False)
    # the active blocks: those with a free voxel
    fb = free_g[h : h + nb[0] * B, h : h + nb[1] * B, h : h + nb[2] * B]
    fb = fb.reshape(nb[0], B, nb[1], B, nb[2], B).any(axis=(1, 3, 5))
    act = np.argwhere(fb)
    N = len(act)
    stride = np.array([gshape[1] * gshape[2], gshape[2], 1], np.int64)
    # flat offsets of a block's window (with halo) and of its interior, from its origin
    w = np.indices((B + 2 * h,) * 3).reshape(3, -1).T
    win = torch.as_tensor(w @ stride, device=dev)
    inn = np.indices((B,) * 3).reshape(3, -1).T + h
    inner = torch.as_tensor(inn @ stride, device=dev)
    base = torch.as_tensor((act * B) @ stride, device=dev)  # window origin = block origin - h + h

    def compact(arr_g):
        """The interiors of the active blocks of a global array: (N, B^3)."""
        flat = torch.as_tensor(arr_g.ravel(), device=dev)
        return flat[base[:, None] + inner[None, :]]

    fixed_g = pad_global(ref, np.float32(0.0))
    # inside the volume: the first stencil's output is zero outside it (as thin_plate's
    # zero padding between the two passes)
    vol_flat = torch.as_tensor(pad_global(np.ones(shape, bool), False).ravel(), device=dev)
    free_c = compact(free_g)
    lo_c = compact(pad_global(np.asarray(mask), False)) & free_c
    hi_c = compact(pad_global(~np.asarray(mask), False)) & free_c
    fixed_c = compact(fixed_g)
    # the voxels next to the boundary (they set the vertices)
    P = np.pad(np.asarray(mask), 1)
    near = np.zeros(shape, bool)
    c = (slice(1, -1),) * 3
    for ax in range(3):
        for d in (-1, 1):
            sh = [slice(1, -1)] * 3
            sh[ax] = slice(1 + d, P.shape[ax] - 1 + d)
            near |= P[c] != P[tuple(sh)]
    watch_c = compact(pad_global(near, False)) & free_c
    weights = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    S3 = (B + 2 * h,) * 3

    def stencil(x, i):
        """Stencil i of a batch x (n, a, b, c), zero beyond each window."""
        p = Fn.pad(x[:, None], (1, 1, 1, 1, 1, 1))[:, 0]
        _, D, H, W = x.shape

        def sl(d):
            return p[:, 1 + d[0] : 1 + d[0] + D, 1 + d[1] : 1 + d[1] + H, 1 + d[2] : 1 + d[2] + W]

        if i < 3:
            e = [0, 0, 0]
            e[i] = 1
            return sl(e) - 2.0 * x + sl([-t for t in e])
        a, b = ((0, 1), (0, 2), (1, 2))[i - 3]

        def at(sa, sb):
            d = [0, 0, 0]
            d[a], d[b] = sa, sb
            return sl(d)

        return 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))

    def grad_c(gflat):
        """The gradient at the interiors of the active blocks, from the global field."""
        out = torch.empty((N, B**3), device=dev)
        for s0 in range(0, N, chunk):
            idx = base[s0 : s0 + chunk, None] + win[None, :]
            xw = gflat[idx].reshape(-1, *S3)
            inv = vol_flat[idx].reshape(-1, *S3)
            g = torch.zeros_like(xw)
            for i, wt in enumerate(weights):
                g += (2.0 * wt) * stencil(torch.where(inv, stencil(xw, i), 0.0), i)
            out[s0 : s0 + chunk] = g[:, h : h + B, h : h + B, h : h + B].reshape(-1, B**3)
        return out

    # the Lipschitz constant: as thin_plate (the same stencils)
    z = torch.randn((1, 24, 24, 24), device=dev)
    for _ in range(60):
        gz = torch.zeros_like(z)
        for i, wt in enumerate(weights):
            gz += (2.0 * wt) * stencil(stencil(z, i), i)
        lam = float(torch.linalg.vector_norm(gz))
        z = gz / lam
    L = 1.05 * lam

    def project(xc):
        xc = torch.where(free_c, xc, fixed_c)
        xc = torch.where(lo_c, torch.clamp(xc, min=eps), xc)
        return torch.where(hi_c, torch.clamp(xc, max=-eps), xc)

    gflat = torch.as_tensor(fixed_g.ravel(), device=dev).clone()
    x = project(compact(fixed_g))  # the start: sdf, projected
    y, t = x.clone(), 1.0
    snap = x.clone()
    put = base[:, None] + inner[None, :]
    gflat[put] = y
    step, restarts = float("inf"), 0
    first_done = np.full(N, -1)  # the first check at which each block was below tol
    t0 = time.perf_counter()
    for k in range(iters):
        xn = project(y - grad_c(gflat) / L)
        if k > 0 and float(torch.sum((y - xn) * (xn - x))) > 0:
            t, restarts = 1.0, restarts + 1
        tn = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
        gflat[put] = y
        if k % 25 == 0 and k > 0:
            win_c = torch.abs(x - snap)[watch_c]
            step = float(torch.max(win_c)) / 25.0 if win_c.numel() else 0.0
            snap_prev = snap
            snap = x.clone()
            # per block: its largest change over the window, on its watched voxels
            per = torch.amax(torch.where(watch_c, torch.abs(x - snap_prev), 0.0), dim=1) / 25.0
            done_now = (per < tol).cpu().numpy()
            first_done[(first_done < 0) & done_now] = k
            if not quiet and k % 100 == 0:
                slow = np.argsort(-per.cpu().numpy())[:3]
                log(f"  thin plate (band, {N} blocks) iteration {k}: max change {step:.2e}; "
                    f"blocks below tol {int(done_now.sum())}/{N}; slowest at (voxels) "
                    f"{[(act[i] * B + B // 2).tolist() for i in slow]}")
            if step < tol:
                break
    gflat[put] = x
    if dev.type == "mps":
        torch.mps.synchronize()
    out = gflat.reshape(gshape)[h : h + shape[0], h : h + shape[1], h : h + shape[2]]
    info = dict(
        L=L, iterations=k + 1, last_change=step, restarts=restarts, blocks=N,
        blocks_total=int(np.prod(nb)), seconds=round(time.perf_counter() - t0, 2),
        blocks_converged_by=dict(
            (int(q), int(np.percentile(np.where(first_done < 0, k + 1, first_done), q)))
            for q in (25, 50, 75, 90, 99, 100)
        ),
    )
    return out.cpu().numpy(), info


def edge_angles(n, f):
    """Angle (degrees) between the unit normals at the two ends of every mesh edge."""
    E = S.edges(f)
    c = np.einsum("ij,ij->i", n[E[:, 0]], n[E[:, 1]])
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def pct(x, qs=(50, 90, 99, 99.9)):
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs} | {
        "max": round(float(np.max(x)), 4)
    }


def area_mm2(v, f, vox):
    p = v[f] * vox
    return float(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dseg", type=Path)
    ap.add_argument("current", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--band", type=float, default=4.0)
    ap.add_argument("--method", choices=("diffusion", "thinplate"), default="thinplate")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--levels", type=int, default=1)
    ap.add_argument("--start", choices=("sdf", "blur"), default="blur")
    ap.add_argument("--stop", choices=("surface", "band"), default="surface")
    ap.add_argument("--unfused", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--blocked", action="store_true", help="thin_plate_band: active blocks only")
    ap.add_argument("--block", type=int, default=64)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    side = read_sidecar(sidecar_path(a.dseg), DSEG_SPEC)
    vol = open_volume(a.dseg)
    full = vol.shape
    vox = np.array(vol.img.header.get_zooms()[:3], float)
    mask_full = np.isin(vol.read(tuple(slice(0, n) for n in full)), side["FilledLabels"])
    nz = [np.flatnonzero(mask_full.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    m = int(a.band) + 2
    lo = np.array([max(int(t[0]) - m, 0) for t in nz])
    hi = np.array([min(int(t[-1]) + 1 + m, n) for t, n in zip(nz, full)])
    mask = np.ascontiguousarray(mask_full[tuple(slice(p, q) for p, q in zip(lo, hi))])
    del mask_full
    log(f"mask {mask.shape} at {lo.tolist()}")

    # phi: the clipped signed distance, +-0.5 on the voxels next to the boundary
    din = ndi.distance_transform_edt(mask).astype(np.float32)
    dout = ndi.distance_transform_edt(~mask).astype(np.float32)
    phi = np.where(mask, din - 0.5, 0.5 - dout).astype(np.float32)
    del din, dout
    np.clip(phi, -(a.band + 3), a.band + 3, out=phi)
    log("signed distance")

    if a.method == "diffusion":
        changes = []
        for it in range(a.iters):
            new = ndi.gaussian_filter(phi, a.sigma, output=np.float32, mode="nearest")
            np.maximum(new, a.eps, out=new, where=mask)
            np.minimum(new, -a.eps, out=new, where=~mask)
            changes.append(float(np.abs(new - phi).max()))
            phi = new
        solve = dict(first=changes[0], last=changes[-1])
        log(f"{a.iters} constrained diffusion iterations")
    else:
        if a.blocked:
            phi, solve = thin_plate_band(
                phi, mask, a.band, a.eps, a.device, a.iters, a.tol, block=a.block
            )
        else:
            phi, solve = thin_plate(
                phi, mask, a.band, a.eps, a.device, a.iters, a.tol, fused=not a.unfused,
                stop=a.stop, start=a.start, levels=a.levels, restart=a.restart,
            )
        log(f"thin plate: {solve['iterations']} iterations, last change {solve['last_change']:.2e}")
    v, f, b = marching_tetrahedra_blocks(phi, 0.0, 64, np.float32)
    vr, fr, br = marching_tetrahedra_blocks(mask.view(np.uint8), 0.5, 64, np.float32)
    f, fr = f[:, [0, 2, 1]].copy(), fr[:, [0, 2, 1]].copy()  # out of the WM
    log(f"marching: {len(v)} vertices, {len(f)} faces")
    same = np.array_equal(f, fr) and np.array_equal(b.vertex_key, br.vertex_key)
    log(f"same triangles and keys as the binary mesh: {same}")

    _, n_x = mi.intersecting_faces(v + lo, f)
    log(f"self-intersecting faces: {n_x}")

    # full-grid keys, to line up with the current mesh vertex by vertex
    p0, p1 = edge_points(b.vertex_key, tuple(int(x) for x in mask.shape))
    key = edge_key(np.ravel_multi_index(tuple((p0 + lo).T), full), p1 - p0)
    cur = WhiteSurface.load(a.current)
    o_new, o_cur = np.argsort(key), np.argsort(cur.vertex_key)
    assert np.array_equal(key[o_new], cur.vertex_key[o_cur]), "the meshes cross other edges"
    to_cur = np.empty(len(key), np.int64)
    to_cur[o_new] = o_cur  # new vertex i is current vertex to_cur[i]
    vc = cur.v[to_cur].astype(np.float64) - lo
    nc = cur.normals[to_cur].astype(np.float64)

    # normals: phi's gradient (trilinear samples a voxel apart), and the mesh's
    h = 0.5
    g = np.zeros((len(v), 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        up = ndi.map_coordinates(phi, (v + e).T, order=1, mode="nearest")
        dn = ndi.map_coordinates(phi, (v - e).T, order=1, mode="nearest")
        g[:, k] = (up - dn) / (2 * h)
    n_grad = -g / np.linalg.norm(g, axis=1, keepdims=True)  # phi > 0 inside: out of the WM
    vv = v.astype(np.float64)
    n_mesh = S.vertex_normals(vv, f)
    n_mesh_s = S.smooth_normals(vv, f)
    n_raw = S.vertex_normals(vr.astype(np.float64), fr)
    log("normals")

    def angle(n1, n2):
        return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n1, n2), -1, 1)))

    rep = dict(
        params=dict(method=a.method, iters=a.iters, sigma=a.sigma, eps=a.eps, band=a.band, tol=a.tol,
                    levels=a.levels, start=a.start, stop=a.stop, fused=not a.unfused,
                    restart=a.restart, blocked=a.blocked, block=a.block),
        solve=solve,
        vertices=len(v),
        faces=len(f),
        same_triangles_as_binary=bool(same),
        self_intersecting_faces=int(n_x),
        area_mm2=dict(
            raw=round(area_mm2(vr.astype(float), fr, vox), 1),
            antialiased=round(area_mm2(vv, f, vox), 1),
            current=round(area_mm2(vc, f, vox), 1),
        ),
        moved_from_raw_vox=pct(np.linalg.norm(vv - vr, axis=1)),
        distance_to_current_vox=pct(np.linalg.norm(vv - vc, axis=1)),
        neighbour_normal_angle_deg=dict(
            raw_mesh=pct(edge_angles(n_raw, fr)),
            antialiased_mesh=pct(edge_angles(n_mesh, f)),
            antialiased_mesh_smoothed=pct(edge_angles(n_mesh_s, f)),
            antialiased_gradient=pct(edge_angles(n_grad, f)),
            current=pct(edge_angles(nc, f)),
        ),
        normal_vs_current_deg=dict(
            gradient=pct(angle(n_grad, nc)),
            mesh_smoothed=pct(angle(n_mesh_s, nc)),
        ),
    )
    (a.out / "antialias_report.json").write_text(json.dumps(rep, indent=1))
    np.savez_compressed(a.out / "antialias_mesh.npz", v=v + lo, f=f, n_grad=n_grad.astype(np.float32),
                        key=key)
    log("done")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
