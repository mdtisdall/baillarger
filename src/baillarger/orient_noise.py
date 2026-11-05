"""The noise covariance C of orient, measured on one acquisition (spec:
docs/formats/orient.md, "The noise file", orient-noise/1).

C is the covariance across echoes of one component of the gradients of the
log echoes minus their local cortex level (baillarger.orient.gradients, per
voxel step), everything in the gradients that is not laminar signal. Its
model at a voxel:

    C = g diag(sigma_e / I_e) R diag(sigma_e / I_e) + C_rest

- The thermal part: sigma_e, the noise of echo e's real and imaginary
  channels before the log, and R, its correlation across echoes, measured
  in background far from tissue; I_e = exp(the echo's local cortex level);
  g = orient.gradient_gain(SD), the derivative filter's gain on white noise.
- The rest, C_rest: the gradients' residual in cortex after the local level
  and the laminar direction are removed (the part of G_e normal to J's
  leading direction, in high-coherence voxels), minus the thermal part
  there. In the log it is about constant relative to the signal, so it does
  not scale with I_e.

The numbers are this acquisition's: nothing here builds them in.

The command:

    python -m baillarger.orient_noise --echoes E1 E2 E3 --dseg DSEG --out DERIV
"""

import argparse
import itertools
import json
import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger import orient as O
from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    dataset_description,
    derivative_path,
    entities,
    provenance,
    sources,
    write_sidecar,
)
from baillarger.mesh import BLOCK as BLOCK_CELLS

log = logging.getLogger("baillarger")

#: the file's SpecVersion (docs/formats/orient.md, "The noise file")
NOISE_VERSION = "orient-noise/1"
#: the background blocks: voxels a side, and the smallest distance (voxels,
#: a box: no tissue within FAR voxels on each axis) to any label not 0
BLOCK = 32
FAR = 20
#: the scale (voxels) of the local mean removed from the background
BG_SCALE = 4.0
#: the coherence thresholds at which the rest is measured; the first is
#: the one that the file's ResidualCovariance uses
COHERENCE = (0.5, 0.3, 0.7)
#: the smallest weight of the cortex in a voxel's whole window (SD and SI)
#: for the rest: away from the cortex's edge
INTERIOR = 0.95

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

NOISE_SPEC = SidecarSpec(
    NOISE_VERSION,
    COMMON_KEYS
    + (
        Key("Echoes", "integer", True, "E, the number of echoes"),
        Key("GradientGain", "number", True, "g = orient.gradient_gain(SD)"),
        Key("DerivativeScale", "number", True, "SD, voxels"),
        Key("ThermalSigma", "array of numbers", True, "sigma_e per echo"),
        Key("ThermalChannels", "number", True, "N of the chi fit"),
        Key("ThermalCorrelation", "array of numbers", True, "R, E x E row-major"),
        Key("ThermalSquaredCorrelation", "array of numbers", True, "c, E x E"),
        Key("ThermalRegion", "object", True, "where and how it was measured"),
        Key("ResidualCovariance", "array of numbers", True, "C_rest, E x E"),
        Key("ResidualTotal", "array of numbers", True, "residual, E x E"),
        Key("ResidualThermal", "array of numbers", True, "thermal there, E x E"),
        Key("ResidualRegion", "object", True, "where and how it was measured"),
        Key("ResidualByCoherence", "array of objects", True, "the rest by c"),
        Key("Model", "string", True, "which parts whitening uses"),
    ),
)


# ---------------------------------------------------------------- thermal


def chi_ratio(n: float) -> float:
    """mu^2 / E[M^2] of a scaled chi distribution with 2n degrees of freedom:
    Gamma(n + 1/2)^2 / (n Gamma(n)^2). Increases with n, from pi/4 at n = 1
    to 1."""
    return math.exp(2 * (math.lgamma(n + 0.5) - math.lgamma(n))) / n


def fit_chi(mean: float, var: float) -> tuple[float, float]:
    """The channels n (2n degrees of freedom, n >= 0.5) and the channel sigma
    of a scaled chi distribution with this mean and variance:
    mu^2 / (mu^2 + v) = chi_ratio(n), sigma^2 = (mu^2 + v) / (2 n). n is
    found by bisection on log n; it is clipped to [0.5, 1e4]."""
    m2 = mean * mean + var
    target = mean * mean / m2
    lo, hi = math.log(0.5), math.log(1e4)
    if target <= chi_ratio(0.5):
        n = 0.5
    elif target >= chi_ratio(1e4):
        n = 1e4
    else:
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if chi_ratio(math.exp(mid)) < target:
                lo = mid
            else:
                hi = mid
        n = math.exp(0.5 * (lo + hi))
    return n, math.sqrt(m2 / (2 * n))


def _local_residual(x: F64, scale: float) -> F64:
    """x minus its Gaussian local mean (scale voxels) on the last three
    axes; valid where the taps lie within x."""
    k = [O.taps(scale)] * 3
    one = np.ones(x.shape[-3:])
    out: F64 = x - O.separable(x, k) / O.separable(one, k)
    return out


@dataclass
class BlockStats:
    """The statistics of one background block."""

    corner: tuple[int, int, int]
    #: (E,) the mean of each echo
    mean: F64
    #: (E,) the variance of each echo's residual about its local mean
    var: F64
    #: (E, E) the correlation of the residuals of the squared magnitudes
    sqcorr: F64


def block_stats(x: F64, halo: int, corner: tuple[int, int, int]) -> BlockStats:
    """The statistics of the block x[:, halo:-halo, ...] (x (E, ...) with a
    halo of halo voxels on each side, all background)."""
    c = (slice(None),) + (slice(halo, -halo),) * 3
    r = _local_residual(x, BG_SCALE)[c]
    q = _local_residual(x * x, BG_SCALE)[c]
    e = x.shape[0]
    qf = q.reshape(e, -1)
    qf = qf - qf.mean(1, keepdims=True)
    cov = qf @ qf.T
    d = np.sqrt(np.diag(cov))
    return BlockStats(
        corner=corner,
        mean=x[c].reshape(e, -1).mean(1),
        var=r.reshape(e, -1).var(1),
        sqcorr=cov / np.outer(d, d),
    )


def far_blocks(
    s: O.OrientSampler, block: int = BLOCK, far: int = FAR
) -> Iterator[tuple[int, int, int]]:
    """The corners of the blocks (block voxels a side, on a grid from voxel
    0) with no label other than 0 within far voxels on each axis, and whose
    box with that margin lies within the grid."""
    sh = s.shape
    rng = [range(far, n - block - far + 1, block) for n in sh]
    for i, j, k in itertools.product(*rng):
        box = (
            slice(i - far, i + block + far),
            slice(j - far, j + block + far),
            slice(k - far, k + block + far),
        )
        if not np.any(s.labels.read(box)):
            yield (i, j, k)


def thermal_from_blocks(stats: list[BlockStats]) -> dict[str, Any]:
    """sigma_e, N and R from the background blocks: medians over the blocks,
    so that the few blocks with signal (the slab's ends) do not move them.
    N: the median over blocks and echoes of the chi fit's n. sigma_e: the
    median over blocks of sqrt((mu^2 + v) / (2 N)). R: for a scaled chi with
    channels independent and correlation R across echoes in each channel,
    corr(M_e^2, M_f^2) = R_ef^2; so R_ef = sign(c) sqrt(|c|), c the median
    over blocks of the correlation of the squared magnitudes' residuals.
    The sign of R cannot be measured; it is taken from c."""
    if not stats:
        raise ValueError("no background blocks")
    mean = np.stack([b.mean for b in stats])
    var = np.stack([b.var for b in stats])
    ns = np.array(
        [
            [fit_chi(m, v)[0] for m, v in zip(mm, vv, strict=True)]
            for mm, vv in zip(mean, var, strict=True)
        ]
    )
    n = float(np.median(ns))
    sigma = np.median(np.sqrt((mean**2 + var) / (2 * n)), 0)
    c = np.median(np.stack([b.sqcorr for b in stats]), 0)
    np.fill_diagonal(c, 1.0)
    R = np.sign(c) * np.sqrt(np.abs(c))
    signal = (
        float(np.mean(np.stack([b.sqcorr[0, 1] for b in stats]) > 0.5))
        if mean.shape[1] > 1
        else 0.0
    )
    return dict(
        sigma=sigma,
        channels=n,
        R=R,
        sqcorr=c,
        blocks=len(stats),
        channels_iqr=[float(np.percentile(ns, 25)), float(np.percentile(ns, 75))],
        signal_fraction=signal,
    )


def measure_thermal(
    s: O.OrientSampler, block: int = BLOCK, far: int = FAR
) -> dict[str, Any]:
    """The thermal part on the acquisition of s (thermal_from_blocks)."""
    halo = O.radius(BG_SCALE)
    stats = []
    for corner in far_blocks(s, block, far):
        box = tuple(slice(c - halo, c + block + halo) for c in corner)
        x = np.stack([np.asarray(e.read(box), np.float64) for e in s.echoes], 0)
        stats.append(block_stats(x, halo, corner))
    out = thermal_from_blocks(stats)
    out["region"] = dict(
        Kind="background",
        BlockVoxels=block,
        FarVoxels=far,
        LocalMeanScaleVoxels=BG_SCALE,
        Blocks=len(stats),
        ChannelsIQR=out["channels_iqr"],
        SignalBlockFraction=out["signal_fraction"],
    )
    return out


def thermal_c(level: F64, sigma: F64, R: F64, gain: float) -> F64:
    """The thermal part of C (E, E, ...) at each voxel of level (E, ...):
    g diag(sigma_e / I_e) R diag(sigma_e / I_e), I_e = exp(level_e)."""
    d = sigma.reshape((-1,) + (1,) * (level.ndim - 1)) * np.exp(-level)
    out: F64 = gain * np.einsum("e...,ef,f...->ef...", d, R, d)
    return out


# ---------------------------------------------------------------- the rest


@dataclass
class RestSums:
    """Sums over the measured voxels, per coherence threshold: the in-plane
    residual products (E, E), the thermal part (E, E), and the voxels."""

    total: dict[float, F64] = field(default_factory=dict)
    thermal: dict[float, F64] = field(default_factory=dict)
    voxels: dict[float, int] = field(default_factory=dict)

    def add(self, c0: float, total: F64, thermal: F64, n: int) -> None:
        self.total[c0] = self.total.get(c0, 0.0) + total
        self.thermal[c0] = self.thermal.get(c0, 0.0) + thermal
        self.voxels[c0] = self.voxels.get(c0, 0) + n


def rest_sums(
    det: F64,
    cortex: Bool,
    level: F64,
    sigma: F64,
    R: F64,
    gain: float,
    keep: Bool | None = None,
    thresholds: tuple[float, ...] = COHERENCE,
    sums: RestSums | None = None,
) -> RestSums:
    """Add one box's sums: det (E, ...) the log echoes minus their level,
    level (E, ...) the level, cortex the mask, keep the voxels to count (the
    output box). The laminar direction n is the leading eigenvector of J
    with the thermal C's inverse at the box's median level as the weights
    (the identity if that C is singular);
    the residual of echo e is G_e - (G_e . n) n, whose two components each
    add r_e . r_f / 2."""
    sums = sums or RestSums()
    g = O.gradients(det, cortex)
    lv = np.where(cortex, level, np.nan)
    if not np.any(cortex):
        return sums
    med = np.nanmedian(lv.reshape(lv.shape[0], -1), 1)
    cth = thermal_c(med[:, None], sigma, R, gain)[..., 0]
    # the weights only find the direction: the identity if C is singular
    w = np.linalg.inv(cth) if np.linalg.cond(cth) < 1e12 else np.eye(len(sigma))
    J = O.tensor(g, w, cortex)
    n, c = O.leading(J)
    k = [O.taps(math.hypot(O.SD, O.SI))] * 3
    inner = cortex & (O.separable(cortex.astype(np.float64), k) > INTERIOR)
    if keep is not None:
        inner &= keep
    gn = np.einsum("ea...,...a->e...", g, np.nan_to_num(n))
    r = g - gn[:, None] * np.moveaxis(np.nan_to_num(n), -1, 0)[None]
    for c0 in thresholds:
        sel = inner & (np.nan_to_num(c) >= c0)
        m = int(sel.sum())
        if m == 0:
            sums.add(c0, np.zeros((len(sigma),) * 2), np.zeros((len(sigma),) * 2), 0)
            continue
        rs = r[:, :, sel]  # (E, 3, m)
        tot = np.einsum("eam,fam->ef", rs, rs) / 2.0
        th = thermal_c(level[:, sel], sigma, R, gain).sum(-1)
        sums.add(c0, tot, th, m)
    return sums


def superblocks(s: O.OrientSampler, level: int) -> Iterator[tuple[int, int, int]]:
    """The superblocks at level whose output box holds cortex, in Morton
    order."""
    from baillarger.blocks import morton

    side = s.cells << level
    grid = [math.ceil(n / side) for n in s.shape]
    all_sb = [
        (int(i), int(j), int(k))
        for i, j, k in itertools.product(*(range(g) for g in grid))
    ]
    all_sb.sort(key=lambda t: int(morton(np.asarray(t))))
    for sb in all_sb:
        box = s.output_box(sb, level, 0.0)
        if np.any(s.labels.read(box) == s.cortex):
            yield sb


def rest_reach(s: O.OrientSampler, sigma_mm: float) -> list[int]:
    """The voxels the rest needs round the output box on each axis: the
    level's taps, then SD's taps and the central difference, then SI's."""
    extra = O.radius(O.SD) + 1 + O.radius(O.SI)
    return [r + extra for r in s.level_reach(sigma_mm)]


def measure_rest(
    s: O.OrientSampler,
    thermal: dict[str, Any],
    level: int = 1,
    sigma_mm: float = O.SIGMA,
    report: list[dict[str, Any]] | None = None,
) -> RestSums:
    """The rest over the cortex of the acquisition of s, superblock by
    superblock."""
    gain = O.gradient_gain(O.SD)
    sums = RestSums()
    reach = rest_reach(s, sigma_mm)
    for sb in superblocks(s, level):
        t = time.perf_counter()
        out = s.output_box(sb, level, 0.0)
        read = s.grow(out, reach)
        e, ctx = s.read(read)
        lv = O.local_level(e, ctx, sigma_mm, s.voxel_mm)
        det = np.where(ctx, O.log_echo(e) - np.nan_to_num(lv), 0.0)
        keep = np.zeros(ctx.shape, bool)
        keep[
            tuple(
                slice(o.start - r.start, o.stop - r.start)
                for o, r in zip(out, read, strict=True)
            )
        ] = True
        one = rest_sums(
            det, ctx, np.nan_to_num(lv), thermal["sigma"], thermal["R"], gain, keep
        )
        for c0 in one.voxels:
            sums.add(c0, one.total[c0], one.thermal[c0], one.voxels[c0])
        if report is not None:
            c0 = COHERENCE[0]
            m = max(one.voxels[c0], 1)
            report.append(
                dict(
                    superblock=list(sb),
                    voxels=one.voxels[c0],
                    total_diag=(np.diag(one.total[c0]) / m).tolist(),
                    thermal_diag=(np.diag(one.thermal[c0]) / m).tolist(),
                    seconds=time.perf_counter() - t,
                )
            )
        log.info(
            f"rest {sb}: {one.voxels[COHERENCE[0]]} voxels, "
            f"{time.perf_counter() - t:.1f} s"
        )
    return sums


def _flat(a: F64) -> list[float]:
    return [float(x) for x in np.asarray(a).ravel()]


def noise_file(
    s: O.OrientSampler,
    thermal: dict[str, Any],
    sums: RestSums,
    level: int,
    sigma_mm: float,
) -> dict[str, Any]:
    """The noise file's content (orient-noise/1)."""
    c0 = COHERENCE[0]
    m = sums.voxels[c0]
    if m == 0:
        raise ValueError("no cortex voxels with coherence above the threshold")
    tot = sums.total[c0] / m
    th = sums.thermal[c0] / m
    by = []
    for c in COHERENCE:
        k = max(sums.voxels[c], 1)
        by.append(
            dict(
                Coherence=c,
                Voxels=sums.voxels[c],
                ResidualCovariance=_flat(sums.total[c] / k - sums.thermal[c] / k),
                ResidualTotal=_flat(sums.total[c] / k),
                ResidualThermal=_flat(sums.thermal[c] / k),
            )
        )
    return dict(
        SpecVersion=NOISE_VERSION,
        Description=(
            "The noise covariance C of orient on one acquisition: "
            "the thermal part and the rest."
        ),
        **sources(*s.echo_paths, s.dseg),
        Provenance=provenance("baillarger.orient_noise.main"),
        Echoes=len(s.echoes),
        GradientGain=O.gradient_gain(O.SD),
        DerivativeScale=O.SD,
        ThermalSigma=_flat(thermal["sigma"]),
        ThermalChannels=float(thermal["channels"]),
        ThermalCorrelation=_flat(thermal["R"]),
        ThermalSquaredCorrelation=_flat(thermal["sqcorr"]),
        ThermalRegion=thermal["region"],
        ResidualCovariance=_flat(tot - th),
        ResidualTotal=_flat(tot),
        ResidualThermal=_flat(th),
        ResidualRegion=dict(
            Kind="cortex",
            Coherence=c0,
            InteriorWeight=INTERIOR,
            Voxels=m,
            Level=level,
            BlockCells=s.cells,
            SigmaMM=sigma_mm,
            IntegrationScale=O.SI,
        ),
        ResidualByCoherence=by,
        Model="thermal+rest",
    )


@dataclass(frozen=True)
class Noise:
    """A read noise file: the model's parts."""

    sigma: F64
    R: F64
    gain: float
    rest: F64

    def c(self, level: F64) -> F64:
        """C (E, E, ...) at each voxel of level (E, ...)."""
        out: F64 = thermal_c(level, self.sigma, self.R, self.gain) + self.rest.reshape(
            self.rest.shape + (1,) * (level.ndim - 1)
        )
        return out


def read_noise(path: Path) -> Noise:
    """The model of a noise file (orient-noise/1)."""
    from baillarger.bids import read_sidecar

    d = read_sidecar(Path(path), NOISE_SPEC)
    e = int(d["Echoes"])
    return Noise(
        sigma=np.asarray(d["ThermalSigma"], np.float64),
        R=np.asarray(d["ThermalCorrelation"], np.float64).reshape(e, e),
        gain=float(d["GradientGain"]),
        rest=np.asarray(d["ResidualCovariance"], np.float64).reshape(e, e),
    )


def noise_path(out: Path, echo: Path) -> Path:
    """<out>/sub-*/ses-*/anat/sub-*_ses-*[_acq-*][_run-*]_desc-orientnoise_noise.json,
    with the entities of the first echo."""
    ents = entities(echo)
    extra = {k: ents[k] for k in ("acq", "run") if k in ents}
    return derivative_path(out, ents, "noise", ".json", desc="orientnoise", **extra)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m baillarger.orient_noise", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument(
        "--echoes",
        nargs="+",
        required=True,
        type=Path,
        help="the echoes, in echo order",
    )
    ap.add_argument(
        "--dseg", required=True, type=Path, help="the _desc-wmtopo_dseg (wm-dseg/1)"
    )
    ap.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    ap.add_argument("--level", type=int, default=1, help="superblock level of the rest")
    ap.add_argument("--cells", type=int, default=BLOCK_CELLS, help="block size, voxels")
    ap.add_argument(
        "--sigma", type=float, default=O.SIGMA, help="sigma of the local level, mm"
    )
    ap.add_argument(
        "--cache-mb", type=int, default=256, help="chunk cache per volume, MB"
    )
    ap.add_argument(
        "--report", type=Path, help="a JSON file for the per-superblock sums"
    )
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = O.OrientSampler(a.echoes, a.dseg, cells=a.cells, cache_bytes=a.cache_mb << 20)
    t = time.perf_counter()
    th = measure_thermal(s)
    log.info(f"thermal: {th['blocks']} blocks, {time.perf_counter() - t:.0f} s")
    rows: list[dict[str, Any]] = []
    sums = measure_rest(s, th, a.level, a.sigma, rows)
    d = noise_file(s, th, sums, a.level, a.sigma)
    path = noise_path(a.out, a.echoes[0])
    dataset_description(a.out)
    write_sidecar(path, NOISE_SPEC, d)
    log.info(f"wrote {path}")
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
