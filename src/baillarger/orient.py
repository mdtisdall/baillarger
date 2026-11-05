"""orient: orientation from the image, a sampler (spec: docs/formats/orient.md,
orient/1).

For a superblock and its halo, from the FLASH echoes and the corrected
segmentation (wm-dseg/1), on demand; nothing is stored:

- the local cortex level of each log echo: the normalized convolution of
  log I_e over the cortex mask with a Gaussian of sigma mm (level);
- the gradients of each log echo minus its level, at the derivative scale
  SD voxels (gradients), and the structure tensor of one or more of them
  (tensor), the tools that the noise measurement (baillarger.orient_noise)
  and the data normal use.

Every convolution is separable, with taps of a Gaussian sampled at the
voxel offsets and truncated at TRUNCATE sigma, and zero outside the grid.
A value at a voxel uses only the voxels within the taps' radius, in the same
order whatever box holds it, so on numpy it is the same, bit for bit, from
every superblock. numpy (float64) is the reference; PyTorch (float32, on
cpu, cuda or mps) gives the same values up to float32 rounding.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from baillarger import labels as L
from baillarger.backend import Device
from baillarger.backend import device as get_device
from baillarger.bids import read_sidecar, sidecar_path
from baillarger.blocks import (
    CACHE_BYTES,
    CHUNK,
    ChunkedSource,
    open_volume,
    superblock_box,
    voxel_size,
)
from baillarger.mesh import BLOCK
from baillarger.white_dseg import DSEG_SPEC
from baillarger.wm_distance import WMDistanceSampler

VERSION = "orient/1"


#: the derivative and the integration scales of the tensor, voxels
SD = 1.0
SI = 2.0
#: the default sigma of the local cortex level, mm (provisional: phase 1.2,
#: Part D chooses it)
SIGMA = 0.64
#: the Gaussian taps end at TRUNCATE sigma (rounded up to whole voxels)
TRUNCATE = 3.0
#: the floor of an echo before the log
FLOOR = 1.0
#: the smallest weight of the cortex mask in a window: below it the
#: normalized convolution is not defined (NaN)
MIN_WEIGHT = 1e-6
#: the cap (mm) of the WM distance that signs the normal (cortex is at most
#: about 4.5 mm thick)
SIGN_CAP = 5.0
#: |cos| of the normal and grad D below which the sign is 0: the two are
#: perpendicular, and rounding could flip the sign
SIGN_EPS = 1e-3

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]
Bool = npt.NDArray[np.bool_]
Box = tuple[slice, slice, slice]
#: an array on numpy or on a PyTorch device
Arr = Any


def taps(sigma_vox: float) -> F64:
    """The Gaussian taps of sigma_vox voxels, at the integer offsets -r to r
    (r = ceil(TRUNCATE sigma_vox)), normalized to a sum of 1."""
    if not sigma_vox > 0:
        raise ValueError(f"sigma {sigma_vox} voxels: not > 0")
    r = math.ceil(TRUNCATE * sigma_vox - 1e-9)
    k = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (k / sigma_vox) ** 2)
    out: F64 = w / w.sum()
    return out


def radius(sigma_vox: float) -> int:
    """The radius in voxels of the taps of sigma_vox."""
    return (len(taps(sigma_vox)) - 1) // 2


def separable(x: Arr, kernels: Sequence[F64]) -> Arr:
    """x (..., n0, n1, n2) convolved with kernels[a] along each of the last
    three axes, zero outside the array. numpy (float64) or a torch tensor
    (float32, on its device)."""
    if isinstance(x, torch.Tensor):
        y = x
        nd = y.ndim
        for a, k in enumerate(kernels):
            ax = nd - 3 + a
            r = (len(k) - 1) // 2
            w = torch.as_tensor(k, dtype=torch.float32, device=y.device).view(1, 1, -1)
            z = y.movedim(ax, -1)
            shp = z.shape
            z = F.conv1d(z.reshape(-1, 1, shp[-1]), w, padding=r)
            y = z.reshape(shp).movedim(-1, ax)
        return y
    out = np.asarray(x, np.float64)
    nd = out.ndim
    for a, k in enumerate(kernels):
        out = ndi.correlate1d(out, k, axis=nd - 3 + a, mode="constant", cval=0.0)
    return out


def _kernels(sigma_mm: float, voxel_mm: npt.ArrayLike) -> list[F64]:
    vs = np.asarray(voxel_mm, np.float64)
    return [taps(sigma_mm / float(v)) for v in vs]


def masked_smooth(x: Arr, mask: Arr, kernels: Sequence[F64]) -> Arr:
    """The normalized convolution G(x m) / G(m) over mask (a 0/1 array of
    the last three axes of x); NaN where G(m) < MIN_WEIGHT."""
    if isinstance(x, torch.Tensor):
        m = mask.to(torch.float32)
        num = separable(x * m, kernels)
        den = separable(m, kernels)
        return torch.where(
            den >= MIN_WEIGHT, num / torch.clamp(den, min=MIN_WEIGHT), torch.nan
        )
    m = np.asarray(mask, np.float64)
    num = separable(np.asarray(x, np.float64) * m, kernels)
    den = separable(m, kernels)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den >= MIN_WEIGHT, num / den, np.nan)


def log_echo(x: Arr) -> Arr:
    """log max(I, FLOOR)."""
    if isinstance(x, torch.Tensor):
        return torch.log(torch.clamp(x.to(torch.float32), min=FLOOR))
    out: F64 = np.log(np.maximum(np.asarray(x, np.float64), FLOOR))
    return out


def local_level(
    echoes: Arr, cortex: Arr, sigma_mm: float, voxel_mm: npt.ArrayLike
) -> Arr:
    """The local cortex level (E, ...) of each echo (E, ...): the normalized
    convolution of log max(I, FLOOR) over the cortex with a Gaussian of
    sigma_mm; NaN outside the cortex. On the device of echoes."""
    lv = masked_smooth(log_echo(echoes), cortex, _kernels(sigma_mm, voxel_mm))
    if isinstance(lv, torch.Tensor):
        return torch.where(cortex.to(torch.bool), lv, torch.nan)
    out: F64 = np.where(np.asarray(cortex, bool), lv, np.nan)
    return out


def gradients(det: F64, cortex: Bool, sd: float = SD) -> F64:
    """The gradients (E, 3, ...) per voxel step of each detrended log echo
    det (E, ...) over the cortex: masked_smooth at sd voxels, then central
    differences (np.gradient, one-sided at the box's edge); 0 outside the
    cortex."""
    k = [taps(sd)] * 3
    s = masked_smooth(np.where(cortex, det, 0.0), cortex, k)
    s = np.where(cortex, s, 0.0)
    g = np.stack([np.gradient(s, axis=a) for a in range(1, 4)], 1)
    out: F64 = np.where(cortex, g, 0.0)
    return out


#: the six unique components of a symmetric 3 x 3 tensor, in this order
PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def tensor(g: F64, weights: F64, cortex: Bool, si: float = SI) -> F64:
    """The structure tensor (6, ...) J = sum_ef W_ef g_e g_f^T of gradients
    g (E, 3, ...) with the echo weights W (E, E) (for example C^-1),
    averaged over the cortex at si voxels (masked_smooth)."""
    w = np.asarray(weights, np.float64)
    h = np.einsum("ef,fa...->ea...", w, g)
    k = [taps(si)] * 3
    comp = [np.einsum("e...,e...->...", g[:, i], h[:, j]) for i, j in PAIRS]
    out: F64 = np.stack([masked_smooth(c, cortex, k) for c in comp], 0)
    return out


def leading(t: F64) -> tuple[F64, F64]:
    """The leading unit eigenvector (..., 3) and the coherence (...) of a
    tensor field (6, ...): c = (l1 - l2) / (l1 + l2), clipped to [0, 1].
    NaN where the tensor is NaN."""
    shp = t.shape[1:]
    flat = t.reshape(6, -1)
    ok = np.all(np.isfinite(flat), 0)
    M = np.zeros((flat.shape[1], 3, 3))
    for c, (i, j) in enumerate(PAIRS):
        M[:, i, j] = M[:, j, i] = np.where(ok, flat[c], 0.0)
    w, V = np.linalg.eigh(M)
    vec = np.where(ok[:, None], V[:, :, 2], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.clip((w[:, 2] - w[:, 1]) / (w[:, 2] + w[:, 1]), 0.0, 1.0)
    coh = np.where(ok & (w[:, 2] + w[:, 1] > 0), coh, np.nan)
    return vec.reshape(*shp, 3), coh.reshape(shp)


def gradient_gain(sd: float = SD) -> float:
    """The variance of one component of `gradients` per unit variance of
    white noise, away from the cortex's edge: the sum of the squared taps of
    the derivative filter (the Gaussian taps of sd, then central
    differences). Exact for the discrete filters."""
    k = taps(sd)
    r = (len(k) - 1) // 2
    d = np.zeros((2 * r + 3,) * 3)
    d[r + 1, r + 1, r + 1] = 1.0
    s = separable(d, [k, k, k])
    return float(np.sum(np.gradient(s, axis=0) ** 2))


def central(s: Arr, axis: int) -> Arr:
    """The central difference (s[i + 1] - s[i - 1]) / 2 along axis (an axis
    of the last three, 0 to 2), 0 at the array's first and last points."""
    ax = s.ndim - 3 + axis
    n = s.shape[ax]
    lo = [slice(None)] * s.ndim
    hi = [slice(None)] * s.ndim
    mid = [slice(None)] * s.ndim
    lo[ax], hi[ax], mid[ax] = slice(0, n - 2), slice(2, n), slice(1, n - 1)
    if isinstance(s, torch.Tensor):
        out = torch.zeros_like(s)
    else:
        out = np.zeros_like(s)
    out[tuple(mid)] = (s[tuple(hi)] - s[tuple(lo)]) / 2
    return out


def _noise_c(level0: Arr, noise: "NoiseModel") -> Arr:
    """C (E, E, ...) at each voxel: g diag(sigma/I) R diag(sigma/I) + rest,
    I = exp(level0)."""
    if isinstance(level0, torch.Tensor):
        dev = level0.device
        sig = torch.as_tensor(noise.sigma, dtype=torch.float32, device=dev)
        R = torch.as_tensor(noise.R, dtype=torch.float32, device=dev)
        rest = torch.as_tensor(noise.rest, dtype=torch.float32, device=dev)
        d = sig.view(-1, 1, 1, 1) * torch.exp(-level0)
        c = noise.gain * torch.einsum("e...,ef,f...->ef...", d, R, d)
        return c + rest.view(*rest.shape, 1, 1, 1)
    d = np.asarray(noise.sigma).reshape(-1, 1, 1, 1) * np.exp(-level0)
    c = noise.gain * np.einsum("e...,ef,f...->ef...", d, noise.R, d)
    out: F64 = c + np.asarray(noise.rest).reshape(*noise.rest.shape, 1, 1, 1)
    return out


def _inv(c: Arr) -> Arr:
    """The inverse of C (E, E, ...) at each voxel."""
    if isinstance(c, torch.Tensor):
        m = c.movedim(0, -1).movedim(0, -1)  # (..., E, E)
        return torch.linalg.inv(m).movedim(-1, 0).movedim(-1, 0)
    m = np.moveaxis(np.moveaxis(c, 0, -1), 0, -1)
    out: F64 = np.moveaxis(np.moveaxis(np.linalg.inv(m), -1, 0), -1, 0)
    return out


class NoiseModel(Protocol):
    """The parts of C (baillarger.orient_noise.Noise has them)."""

    @property
    def sigma(self) -> F64: ...
    @property
    def R(self) -> F64: ...
    @property
    def gain(self) -> float: ...
    @property
    def rest(self) -> F64: ...


def orient_arrays(
    echoes: Arr,
    cortex: Arr,
    noise: NoiseModel,
    voxel_mm: npt.ArrayLike,
    sigma_mm: float = SIGMA,
    echo_tensors: bool = False,
) -> dict[str, Arr]:
    """Part C on one box, numpy (float64) or torch (float32): the level, the
    gradients G_e (mm^-1) of the detrended log echoes at SD voxels, the
    whitened tensor J = sum_ef (C^-1)_ef G_e G_f^T over the cortex at SI
    voxels ("tensor", (6, ...), NaN outside the cortex), and on request each
    echo's own tensor G_e G_e^T at SI ("echo_tensor", (E, 6, ...)). Valid
    where the box holds the whole reach (reach_voxels)."""
    vs = np.asarray(voxel_mm, np.float64)
    is_t = isinstance(echoes, torch.Tensor)
    lv = local_level(echoes, cortex, sigma_mm, vs)
    ctx: Arr
    lv0: Arr
    det: Arr
    sm: Arr
    g: Arr
    h: Arr
    J: Arr
    if is_t:
        ctx = cortex.to(torch.bool)
        lv0 = torch.nan_to_num(lv)
        det = torch.where(ctx, log_echo(echoes) - lv0, 0.0)
    else:
        ctx = np.asarray(cortex, bool)
        lv0 = np.nan_to_num(lv)
        det = np.where(ctx, log_echo(echoes) - lv0, 0.0)
    sm = masked_smooth(det, ctx, [taps(SD)] * 3)
    if is_t:
        sm = torch.where(ctx, sm, 0.0)
        g = torch.stack([central(sm, a) / float(vs[a]) for a in range(3)], 1)
        g = torch.where(ctx, g, 0.0)
    else:
        sm = np.where(ctx, sm, 0.0)
        g = np.stack([central(sm, a) / float(vs[a]) for a in range(3)], 1)
        g = np.where(ctx, g, 0.0)
    w = _inv(_noise_c(lv0, noise))
    if is_t:
        h = torch.einsum("ef...,fa...->ea...", w, g)
        comp = [(g[:, i] * h[:, j]).sum(0) for i, j in PAIRS]
        J = torch.stack([masked_smooth(c, ctx, [taps(SI)] * 3) for c in comp], 0)
        J = torch.where(ctx, J, torch.nan)
    else:
        h = np.einsum("ef...,fa...->ea...", w, g)
        comp = [(g[:, i] * h[:, j]).sum(0) for i, j in PAIRS]
        J = np.stack([masked_smooth(c, ctx, [taps(SI)] * 3) for c in comp], 0)
        J = np.where(ctx, J, np.nan)
    out: dict[str, Arr] = {"level": lv, "tensor": J}
    if echo_tensors:
        k = [taps(SI)] * 3
        et: Arr
        if is_t:
            et = torch.stack(
                [
                    torch.stack(
                        [masked_smooth(g[e, i] * g[e, j], ctx, k) for i, j in PAIRS]
                    )
                    for e in range(g.shape[0])
                ]
            )
            out["echo_tensor"] = torch.where(ctx, et, torch.nan)
        else:
            et = np.stack(
                [
                    np.stack(
                        [masked_smooth(g[e, i] * g[e, j], ctx, k) for i, j in PAIRS]
                    )
                    for e in range(g.shape[0])
                ]
            )
            out["echo_tensor"] = np.where(ctx, et, np.nan)
    return out


def leading_any(t: Arr) -> tuple[F64, F64]:
    """leading on numpy, or on a torch tensor (eigh on its device; results
    to numpy float64). The eigenvector is canonical: its component of
    largest magnitude is positive."""
    if isinstance(t, torch.Tensor):
        shp = t.shape[1:]
        flat = t.reshape(6, -1)
        ok = torch.all(torch.isfinite(flat), 0)
        M = torch.zeros((flat.shape[1], 3, 3), dtype=t.dtype, device=t.device)
        for q, (i, j) in enumerate(PAIRS):
            v = torch.where(ok, flat[q], 0.0)
            M[:, i, j] = v
            M[:, j, i] = v
        w, V = torch.linalg.eigh(M)
        vec = V[:, :, 2].cpu().numpy().astype(np.float64)
        wn = w.cpu().numpy().astype(np.float64)
        okn = ok.cpu().numpy()
        vec = np.where(okn[:, None], vec, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            coh = np.clip((wn[:, 2] - wn[:, 1]) / (wn[:, 2] + wn[:, 1]), 0.0, 1.0)
        coh = np.where(okn & (wn[:, 2] + wn[:, 1] > 0), coh, np.nan)
        n, c = vec.reshape(*shp, 3), coh.reshape(shp)
    else:
        n, c = leading(t)
    return canonical(n), c


def canonical(n: F64) -> F64:
    """n (..., 3) with the sign that makes its component of largest
    magnitude positive."""
    k = np.argmax(np.abs(np.nan_to_num(n)), -1)
    s = np.take_along_axis(n, k[..., None], -1)
    out: F64 = np.where(s < 0, -n, n)
    return out


def distance_gradient(d: F64, voxel_mm: npt.ArrayLike) -> F64:
    """grad D (..., 3) in mm^-1... per mm, at each voxel of d (n0, n1, n2):
    on each axis the central difference where both neighbours and the voxel
    are finite, else the one-sided difference with the finite neighbour,
    else 0. NaN where D or every axis is not defined."""
    vs = np.asarray(voxel_mm, np.float64)
    x = np.asarray(d, np.float64)
    fin = np.isfinite(x)
    out = np.zeros((*x.shape, 3))
    anyax = np.zeros(x.shape, bool)
    for a in range(3):
        f = np.moveaxis(np.where(fin, x, 0.0), a, 0)
        m = np.moveaxis(fin, a, 0)
        nxt = np.zeros_like(f)
        prv = np.zeros_like(f)
        mn = np.zeros_like(m)
        mp = np.zeros_like(m)
        nxt[:-1], mn[:-1] = f[1:], m[1:]
        prv[1:], mp[1:] = f[:-1], m[:-1]
        both = mn & mp
        g = np.where(
            both,
            (nxt - prv) / 2,
            np.where(mn, nxt - f, np.where(mp, f - prv, 0.0)),
        )
        out[..., a] = np.moveaxis(g, 0, a) / vs[a]
        anyax |= np.moveaxis(mn | mp, 0, a)
    ok = fin & anyax
    out[~ok] = np.nan
    return out


def sign_of(n: F64, grad_d: F64) -> npt.NDArray[np.int8]:
    """+1 where n points away from the WM (n . grad D / |grad D| > SIGN_EPS),
    -1 where it points towards it (< -SIGN_EPS), 0 where grad D is not
    defined, is 0, or is within SIGN_EPS of perpendicular to n."""
    nd = np.linalg.norm(grad_d, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("...a,...a->...", n, grad_d) / nd
    out = np.zeros(cos.shape, np.int8)
    out[np.nan_to_num(cos) > SIGN_EPS] = 1
    out[np.nan_to_num(cos) < -SIGN_EPS] = -1
    return out


# ---------------------------------------------------------------- the sampler


@dataclass(frozen=True)
class Level:
    """The local cortex level of each echo on the output box."""

    #: (E, box shape) float32: the level of log echo e; NaN outside cortex
    level: F32
    #: the output box, in grid points
    box: Box
    #: sigma of the call, mm
    sigma_mm: float


@dataclass(frozen=True)
class Orientation:
    """orient's output on the output box."""

    #: (3, box) float32: the data normal, a unit vector in scanner axes of
    #: the grid (mm), away from the WM where sign is not 0; NaN outside cortex
    normal: F32
    #: (box) float32: the coherence (l1 - l2) / (l1 + l2) in [0, 1]; NaN
    #: outside cortex
    coherence: F32
    #: (box) int8: the factor (+1 or -1) that turned the canonical
    #: eigenvector away from the WM; 0 where the sign is not defined (the
    #: normal is the canonical eigenvector there) and outside cortex
    sign: npt.NDArray[np.int8]
    #: (6, box) float32 or None: J's six components (PAIRS), on request
    tensor: F32 | None
    #: (E, 6, box) float32 or None: each echo's own tensor, on request
    echo_tensor: F32 | None
    #: the output box, in grid points
    box: Box
    #: sigma of the level, mm
    sigma_mm: float


class OrientSampler:
    """orient on one acquisition: echoes, the echo images (uncompressed or
    gzipped NIfTI, in echo order); dseg, the _desc-wmtopo_dseg of wm-dseg/1
    on the same grid; noise, its _desc-orientnoise_noise.json (needed by
    orient, not by level). Superblocks are 2^level blocks of `cells` voxels
    a side. The echoes and the labels are read through chunk caches of
    cache_bytes each."""

    def __init__(
        self,
        echoes: Sequence[Path],
        dseg: Path,
        cells: int = BLOCK,
        cache_bytes: int = CACHE_BYTES,
        noise: Path | None = None,
    ) -> None:
        if not echoes:
            raise ValueError("no echoes")
        if cells < 1:
            raise ValueError(f"cells {cells}: not >= 1")
        self.cells = int(cells)
        self.dseg = Path(dseg)
        side = read_sidecar(sidecar_path(self.dseg), DSEG_SPEC)
        self.filled = np.asarray(side["FilledLabels"], np.int64)
        self.cortex = L.CORTEX_LABEL[side["Hemisphere"]]
        lab = open_volume(self.dseg)
        self.shape: tuple[int, int, int] = tuple(int(n) for n in lab.shape)  # type: ignore[assignment]
        self.affine = np.asarray(lab.affine, np.float64)
        self.voxel_mm = voxel_size(self.affine)
        self.labels = ChunkedSource(lab, CHUNK, cache_bytes)
        self.echo_paths = [Path(p) for p in echoes]
        self.echoes = []
        for p in self.echo_paths:
            v = open_volume(p)
            if tuple(v.shape) != self.shape:
                raise ValueError(f"{p.name}: grid {v.shape}, not {self.shape}")
            if not np.allclose(v.affine, self.affine, atol=1e-5):
                raise ValueError(f"{p.name}: its affine is not the segmentation's")
            self.echoes.append(ChunkedSource(v, CHUNK, cache_bytes))
        self.noise: NoiseModel | None = None
        if noise is not None:
            from baillarger.orient_noise import read_noise

            nm = read_noise(Path(noise))
            if len(nm.sigma) != len(self.echoes):
                raise ValueError(f"{Path(noise).name}: {len(nm.sigma)} echoes")
            self.noise = nm
        self.distance = WMDistanceSampler(self.dseg, self.cells, cache_bytes)

    def output_box(
        self, superblock: tuple[int, int, int], level: int, halo_mm: float
    ) -> Box:
        _check_call(superblock, level, halo_mm)
        return superblock_box(  # type: ignore[return-value]
            superblock, level, self.cells, halo_mm, self.affine, self.shape
        )

    def grow(self, box: Box, reach: Sequence[int]) -> Box:
        """box grown by reach[a] voxels on axis a, clipped to the grid."""
        return tuple(  # type: ignore[return-value]
            slice(max(s.start - r, 0), min(s.stop + r, n))
            for s, r, n in zip(box, reach, self.shape, strict=True)
        )

    def read(self, box: Box) -> tuple[F64, Bool]:
        """The echoes (E, box) as float64 and the cortex mask on box."""
        e = np.stack([np.asarray(s.read(box), np.float64) for s in self.echoes], 0)
        return e, np.asarray(self.labels.read(box) == self.cortex)

    def level_reach(self, sigma_mm: float) -> list[int]:
        """The reach in voxels on each axis of the level's taps."""
        return [radius(sigma_mm / float(v)) for v in self.voxel_mm]

    def level(
        self,
        superblock: tuple[int, int, int],
        level: int,
        halo_mm: float = 0.0,
        sigma_mm: float = SIGMA,
        device: str | Device = "numpy",
    ) -> Level:
        """The local cortex level of each echo on the output box of
        superblock (I, J, K) at level m with its halo."""
        if not (sigma_mm > 0 and math.isfinite(sigma_mm)):
            raise ValueError(f"sigma_mm {sigma_mm}: not a finite number > 0")
        dev = device if isinstance(device, Device) else get_device(device)
        out = self.output_box(superblock, level, halo_mm)
        read = self.grow(out, self.level_reach(sigma_mm))
        e, ctx = self.read(read)
        crop = tuple(
            slice(o.start - r.start, o.stop - r.start)
            for o, r in zip(out, read, strict=True)
        )
        if dev.is_numpy:
            lv = local_level(e, ctx, sigma_mm, self.voxel_mm)
            res = np.asarray(lv[(slice(None), *crop)], np.float32)
        else:
            te = torch.as_tensor(e.astype(np.float32), device=dev.torch)
            tc = torch.as_tensor(ctx, device=dev.torch)
            lv = local_level(te, tc, sigma_mm, self.voxel_mm)
            res = lv[(slice(None), *crop)].cpu().numpy().astype(np.float32)
        return Level(level=np.ascontiguousarray(res), box=out, sigma_mm=float(sigma_mm))

    def orient(
        self,
        superblock: tuple[int, int, int],
        level: int,
        halo_mm: float = 0.0,
        sigma_mm: float = SIGMA,
        device: str | Device = "numpy",
        tensor: bool = False,
        echo_tensor: bool = False,
    ) -> Orientation:
        """The data normal, its coherence and its sign on the output box of
        superblock (I, J, K) at level m with its halo; J and each echo's tensor
        on request."""
        if self.noise is None:
            raise ValueError("orient needs the noise file (OrientSampler(noise=...))")
        if not (sigma_mm > 0 and math.isfinite(sigma_mm)):
            raise ValueError(f"sigma_mm {sigma_mm}: not a finite number > 0")
        dev = device if isinstance(device, Device) else get_device(device)
        out = self.output_box(superblock, level, halo_mm)
        read = self.grow(out, orient_reach(self.voxel_mm, sigma_mm))
        e, ctx = self.read(read)
        crop = tuple(
            slice(o.start - r.start, o.stop - r.start)
            for o, r in zip(out, read, strict=True)
        )
        if dev.is_numpy:
            res = orient_arrays(
                e, ctx, self.noise, self.voxel_mm, sigma_mm, echo_tensor
            )
            J = res["tensor"][(slice(None), *crop)]
        else:
            te = torch.as_tensor(e.astype(np.float32), device=dev.torch)
            tc = torch.as_tensor(ctx, device=dev.torch)
            res = orient_arrays(
                te, tc, self.noise, self.voxel_mm, sigma_mm, echo_tensor
            )
            J = res["tensor"][(slice(None), *crop)].contiguous()
        n, c = leading_any(J)
        cx = ctx[crop]
        # the sign: grad D of the WM distance (cap SIGN_CAP), one voxel round
        step = float(np.max(self.voxel_mm))
        dist = self.distance.sample(superblock, level, halo_mm + step, SIGN_CAP)
        db = dist.box
        pad = tuple(
            slice(
                max(o.start - 1, b.start) - b.start, min(o.stop + 1, b.stop) - b.start
            )
            for o, b in zip(out, db, strict=True)
        )
        dd = dist.distance[pad].astype(np.float64)
        gd = distance_gradient(dd, self.voxel_mm)
        inner = tuple(
            slice(o.start - (b.start + p.start), o.stop - (b.start + p.start))
            for o, b, p in zip(out, db, pad, strict=True)
        )
        sgn = np.where(cx, sign_of(n, gd[inner]), 0).astype(np.int8)
        signed = np.where((sgn < 0)[..., None], -n, n)
        normal = np.where(cx[..., None], signed, np.nan)
        et = None
        if echo_tensor:
            et = np.asarray(
                _to_numpy(res["echo_tensor"][(slice(None), slice(None), *crop)]),
                np.float32,
            )
        return Orientation(
            normal=np.ascontiguousarray(np.moveaxis(normal, -1, 0), np.float32),
            coherence=np.where(cx, c, np.nan).astype(np.float32),
            sign=sgn,
            tensor=np.asarray(_to_numpy(J), np.float32) if tensor else None,
            echo_tensor=et,
            box=out,
            sigma_mm=float(sigma_mm),
        )


def _to_numpy(a: Arr) -> Any:
    return a.cpu().numpy() if isinstance(a, torch.Tensor) else a


def orient_reach(voxel_mm: npt.ArrayLike, sigma_mm: float) -> list[int]:
    """The voxels orient needs round its output box on each axis: the
    level's taps, SD's taps and the central difference, SI's taps."""
    extra = radius(SD) + 1 + radius(SI)
    return [radius(sigma_mm / float(v)) + extra for v in np.asarray(voxel_mm)]


def _check_call(superblock: tuple[int, int, int], level: int, halo_mm: float) -> None:
    if len(superblock) != 3 or min(superblock) < 0:
        raise ValueError(f"superblock {superblock}: not 3 indices >= 0")
    if level < 0:
        raise ValueError(f"level {level}: not >= 0")
    if not (halo_mm >= 0 and math.isfinite(halo_mm)):
        raise ValueError(f"halo_mm {halo_mm}: not a finite number >= 0")
