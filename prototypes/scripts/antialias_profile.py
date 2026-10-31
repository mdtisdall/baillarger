"""Profile the thin-plate energy's gradient (antialias.py) on the GPU and the CPU, to choose
its implementation: how the time per step scales with the volume, and so what the hemisphere
(217 M voxels) and a whole brain would take.

The gradient is 2 sum_s w_s D_s^T D_s phi over the six second-derivative stencils (phi_xx,
phi_yy, phi_zz with w 1; phi_xy, phi_xz, phi_yz with w 2), zero outside the volume. Each stencil is
symmetric, so D^T is D. Three implementations:
  A  dense 3 x 3 x 3 conv3d per stencil: 12 calls (the first prototype);
  B  one 6-channel conv3d, then one conv3d summing the 6 channels back: 2 calls;
  C  shifted slices of a padded copy: each stencil is 3 or 4 slices added (elementwise only).
First a check that all three give the same gradient; then, per size and device, the median time
of a gradient and of a whole solver iteration (gradient, projection, the accelerated step),
with and without reading the largest change back to the host every iteration.

Usage: antialias_profile.py [--sizes 64 128 192 256 320] [--devices mps cpu] [--reps 5]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).parent))
from antialias import _stencils  # noqa: E402

PURE = [((1, 0, 0), 0), ((0, 1, 0), 1), ((0, 0, 1), 2)]
MIXED = [(0, 1), (0, 2), (1, 2)]


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def make_A(dev):
    ks = [(torch.as_tensor(k, device=dev)[None, None], w) for k, w in _stencils()]

    def grad(x):
        g = torch.zeros_like(x)
        for k, w in ks:
            r = Fn.conv3d(x[None, None], k, padding=1)
            g += (2.0 * w) * Fn.conv3d(r, k, padding=1)[0, 0]
        return g

    return grad


def make_B(dev):
    st = _stencils()
    k6 = torch.as_tensor(np.stack([k for k, _ in st]), device=dev)[
        :, None
    ]  # (6,1,3,3,3)
    w = torch.as_tensor([2.0 * w for _, w in st], device=dev)
    kback = (k6[:, 0] * w[:, None, None, None])[None]  # (1,6,3,3,3): weighted sum back

    def grad(x):
        r = Fn.conv3d(x[None, None], k6, padding=1)  # (1,6,...)
        return Fn.conv3d(r, kback, padding=1)[0, 0]

    return grad


def _apply_shift(x):
    """The six stencils of x (zero outside), by shifted slices of a padded copy."""
    p = Fn.pad(x[None, None], (1, 1, 1, 1, 1, 1))[0, 0]
    return [_one(x, p, i) for i in range(6)]


def make_C(dev):
    w = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

    def grad(x):
        r = _apply_shift(x)
        g = torch.zeros_like(x)
        for i, ri in enumerate(r):
            # D^T = D for these symmetric stencils: stencil i of r_i
            p = Fn.pad(ri[None, None], (1, 1, 1, 1, 1, 1))[0, 0]
            g += (2.0 * w[i]) * _one(ri, p, i)
        return g

    return grad


def _one(x, p, i):
    """Stencil i of x, from its zero-padded copy p."""
    D, H, W = x.shape

    def sl(di, dj, dk):
        return p[1 + di : 1 + di + D, 1 + dj : 1 + dj + H, 1 + dk : 1 + dk + W]

    if i < 3:
        e = PURE[i][0]
        return sl(*e) - 2.0 * x + sl(*[-t for t in e])
    a, b = MIXED[i - 3]

    def at(sa, sb):
        d = [0, 0, 0]
        d[a], d[b] = sa, sb
        return sl(*d)

    return 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))


def fused_taps():
    """The gradient's one stencil: 2 sum_s w_s (k_s * k_s), 5 x 5 x 5, as (offset, weight) for
    its nonzero taps (equal to applying D then D^T, except within 2 voxels of the volume's
    edge)."""
    from scipy.signal import convolve

    K = np.zeros((5, 5, 5))
    for k, w in _stencils():
        K += 2.0 * w * convolve(k.astype(np.float64), k.astype(np.float64))
    idx = np.argwhere(np.abs(K) > 1e-12)
    return [(tuple(int(t) - 2 for t in i), float(K[tuple(i)])) for i in idx]


def make_F(dev):
    taps = fused_taps()

    def grad(x):
        p = Fn.pad(x[None, None], (2, 2, 2, 2, 2, 2))[0, 0]
        D, H, W = x.shape
        g = torch.zeros_like(x)
        for (di, dj, dk), c in taps:
            g += c * p[2 + di : 2 + di + D, 2 + dj : 2 + dj + H, 2 + dk : 2 + dk + W]
        return g

    return grad


def time_it(fn, dev, reps):
    fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def iteration(grad, dev, x, inside, free, fixed, readback):
    """One accelerated projected-gradient step (as antialias.thin_plate), state in a dict."""
    st = {"x": x.clone(), "y": x.clone(), "t": 1.0}

    def step():
        y = st["y"]
        xn = y - grad(y) / 400.0
        xn = torch.where(free, xn, fixed)
        xn = torch.where(inside & free, torch.clamp(xn, min=0.02), xn)
        xn = torch.where(~inside & free, torch.clamp(xn, max=-0.02), xn)
        tn = (1.0 + (1.0 + 4.0 * st["t"] ** 2) ** 0.5) / 2.0
        if readback:
            float(torch.max(torch.abs(xn - st["x"])))
        st["y"] = xn + ((st["t"] - 1.0) / tn) * (xn - st["x"])
        st["x"], st["t"] = xn, tn

    return step


def solve_crops(a):
    """Iterations to converge (antialias.thin_plate, MPS) on crops of a real mask, from the
    signed distance and from the signed distance blurred once (sigma 1)."""
    from scipy import ndimage as ndi

    import antialias as AA
    from baillarger.bids import read_sidecar, sidecar_path
    from baillarger.blocks import open_volume
    from baillarger.white import DSEG_SPEC

    side = read_sidecar(sidecar_path(a.solve), DSEG_SPEC)
    vol = open_volume(a.solve)
    n = a.crop
    for lo in ((180, 120, 400), (250, 60, 650), (300, 200, 850)):
        box = tuple(slice(o, o + n) for o in lo)
        mask = np.isin(vol.read(box), side["FilledLabels"])
        din = ndi.distance_transform_edt(mask)
        dout = ndi.distance_transform_edt(~mask)
        sdf = np.where(mask, din - 0.5, 0.5 - dout).astype(np.float32)
        np.clip(sdf, -8, 8, out=sdf)
        variants = (
            ("baseline: surface stop", dict(restart=False, levels=0)),
            ("+ restart", dict(restart=True, levels=0)),
            ("+ restart, 1 level", dict(restart=True, levels=1)),
            ("+ restart, 2 levels", dict(restart=True, levels=2)),
            ("+ restart, unfused", dict(restart=True, levels=0, fused=False)),
        )
        ref = None
        for label, kw in variants:
            t = time.perf_counter()
            phi, info = AA.thin_plate(
                sdf,
                mask,
                5.0,
                0.02,
                "mps",
                8000,
                1e-4,
                stop="surface",
                quiet=True,
                **kw,
            )
            dt = time.perf_counter() - t
            if ref is None:
                ref = phi
            near = np.abs(ref) < 1.0
            diff = float(np.abs(phi - ref)[near].max())
            lv = [(d["iterations"], d["seconds"]) for d in info["levels"]]
            print(
                f"crop at {lo} {label:24s}: {dt:6.1f} s; levels (iterations, s) {lv}; "
                f"max |phi - baseline| near the surface {diff:.1e}",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 192, 256, 320])
    ap.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cpu-max", type=int, default=192)
    ap.add_argument("--skip-conv", action="store_true", help="leave out A and B")
    ap.add_argument(
        "--solve", type=Path, help="a dseg: iterations to converge on crops of it"
    )
    ap.add_argument("--crop", type=int, default=192)
    a = ap.parse_args()
    print(
        f"torch {torch.__version__}, CPU threads {torch.get_num_threads()}", flush=True
    )

    # correctness: the three gradients agree (on the CPU, small)
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.standard_normal((20, 23, 17)).astype(np.float32))
    dev = torch.device("cpu")
    gA, gB, gC, gF = make_A(dev)(x), make_B(dev)(x), make_C(dev)(x), make_F(dev)(x)
    inner = (slice(2, -2),) * 3
    print(
        f"check: |B - A| max {float((gB - gA).abs().max()):.2e}, "
        f"|C - A| max {float((gC - gA).abs().max()):.2e}, "
        f"|F - A| max (2 voxels from the edge) {float((gF - gA)[inner].abs().max()):.2e}, "
        f"|A| max {float(gA.abs().max()):.2f}; fused taps {len(fused_taps())}",
        flush=True,
    )
    if a.solve is not None:
        solve_crops(a)
        return

    print("\nms per call (and ns per voxel); iteration = gradient + projection + step")
    for name in a.devices:
        dev = torch.device(name)
        for n in a.sizes:
            if name == "cpu" and n > a.cpu_max:
                continue
            nv = n**3
            x = torch.as_tensor(
                rng.standard_normal((n, n, n)).astype(np.float32), device=dev
            )
            inside = x > 0
            free = torch.ones_like(inside)
            fixed = x.clone()
            row = []
            for label, make in (
                ("A", make_A),
                ("B", make_B),
                ("C", make_C),
                ("F", make_F),
            ):
                if label in ("A", "B") and a.skip_conv:
                    continue
                try:
                    g = make(dev)
                    tg = time_it(lambda: g(x), dev, a.reps)
                    row.append(f"{label} grad {1e3 * tg:8.1f} ({1e9 * tg / nv:5.1f})")
                except Exception as e:  # e.g. an op MPS does not support
                    row.append(f"{label} grad FAILED {type(e).__name__}: {str(e)[:60]}")
            gC = make_C(dev)
            it_rb = time_it(
                iteration(gC, dev, x, inside, free, fixed, True), dev, a.reps
            )
            it_no = time_it(
                iteration(gC, dev, x, inside, free, fixed, False), dev, a.reps
            )
            row.append(
                f"C iteration {1e3 * it_rb:8.1f} (readback) {1e3 * it_no:8.1f} (none)"
            )
            print(
                f"{name:4s} {n:4d}^3 = {nv / 1e6:6.2f} M: " + " | ".join(row),
                flush=True,
            )
            del x, inside, free, fixed
            if name == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
