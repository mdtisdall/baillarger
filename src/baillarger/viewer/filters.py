"""Filters of a site's column data for the viewer, run on demand and cached (plan:
reports/Crossing paths and contacts from laminar profiles.md, *Filters in the viewer*).

A filter is a set of parameters (Params); its result (Result) holds, for every trace of
the column volume that is not near a box face, the profile to show and, when
flattened, its warp. Steps, in order:

1. Echo: echo 2, echo 3 or their mean, each divided by its SD over the traces'
   samples (the column volume's echoes are already trend-removed).
2. Smoothing along the column: normalized Gaussian, sigma `smooth_mm`, over each
   trace's own samples.
3. Flattening (`flatten` lk or dyn): baillarger.flatten on both echoes, as
   prototypes/scripts/flatten_volume.py: pairs of roots within 0.25 mm whose normals
   agree (dot > 0.7), shifts by the gradient estimator or dynamic warping (bound
   `smax`), the warp z_v(tau) by fit_warp (`eps`), then the profile resampled at
   constant tau. The depth axis is then tau (mm: the mean depth of its horizon).
4. Stacking across columns (`stack_mm` > 0): each profile averaged with its
   neighbours' (roots within `stack_mm`, normals agreeing) in the frame of step 3,
   weighted by exp(-d^2 / 2 (stack_mm/2)^2) times the positive correlation of the two
   profiles, as prototypes/scripts/tau_bands.py.

Per-trace layers (names start "filter: "): with flattening, the rms of its pairs'
misfit over levels, the fraction of its levels that fold (dz/dtau <= 0) and the
fraction of its weighted shift samples at the bound; with stacking, the sum of its
neighbours' weights.

The filters take their inputs (Inputs) from the caller; this module reads no data
files. Its job runner and the page code that read a column volume's .npz went with the
site mode (phase 4.3); flattening as a filter of the viewer is phase 4.4's (it reads
the published flattening), and runs on the job pattern of baillarger.viewer.jobs.
"""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from baillarger import flatten as FL
from baillarger.streamlines import STEP_MM

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Arr = npt.NDArray[Any]

PAIR_MM, NDOT = 0.25, 0.7
DTAU, TAU_MAX = 0.08, 6.0
CHUNK = 20000
ECHOES = ("2", "3", "2+3")
FLATTEN = ("off", "lk", "dyn")
PREFIX = "filter: "


@dataclass(frozen=True)
class Params:
    echo: str = "2"
    smooth_mm: float = 0.0
    flatten: str = "off"
    smax: float = 0.3
    eps: float = 0.2
    stack_mm: float = 0.0

    @classmethod
    def parse(cls, d: dict[str, Any]) -> "Params":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown filter parameters: {sorted(unknown)}")
        p = cls(
            echo=str(d.get("echo", "2")),
            smooth_mm=float(d.get("smooth_mm", 0.0)),
            flatten=str(d.get("flatten", "off")),
            smax=float(d.get("smax", 0.3)),
            eps=float(d.get("eps", 0.2)),
            stack_mm=float(d.get("stack_mm", 0.0)),
        )
        if p.echo not in ECHOES:
            raise ValueError(f"echo must be one of {ECHOES}")
        if p.flatten not in FLATTEN:
            raise ValueError(f"flatten must be one of {FLATTEN}")
        if not (p.smooth_mm >= 0 and p.stack_mm >= 0 and p.smax > 0 and p.eps > 0):
            raise ValueError("smooth_mm, stack_mm >= 0; smax, eps > 0")
        return p

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def key(self, data_key: str) -> str:
        h = hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode())
        h.update(data_key.encode())
        return h.hexdigest()[:16]

    def label(self) -> str:
        parts = [f"echo {self.echo}"]
        if self.smooth_mm:
            parts.append(f"smoothed {self.smooth_mm:g} mm")
        if self.flatten != "off":
            est = "gradient" if self.flatten == "lk" else "dynamic"
            parts.append(f"flattened ({est}, ±{self.smax:g} mm, eps {self.eps:g})")
        if self.stack_mm:
            parts.append(f"stacked {self.stack_mm:g} mm")
        return ", ".join(parts)


@dataclass
class Inputs:
    """What the filters read: the profiles along the traces and their roots."""

    fv: Arr  # (2, N, n): echoes 2 and 3 along every trace, nan past its end
    ns: Int  # (N,) samples per trace
    keep: Int  # traces filtered: rooted, not near a box face, more than one sample
    root_mm: Float  # (len(keep), 3) box mm
    nrm: Float  # (len(keep), 3) white-mesh normal at the root


@dataclass
class Result:
    params: Params
    key: str
    keep: Int  # column-volume trace of each row
    prof: Float  # (N, m) the profile shown, nan where none
    frame: str  # "depth" (mm from the root) or "tau" (mm)
    tau: Float | None = None  # (K,) with flattening
    z: Float | None = None  # (N, K) depth of each tau level
    layers: dict[str, Float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        arrays: dict[str, Arr] = dict(keep=self.keep, prof=self.prof.astype(np.float32))
        if self.tau is not None and self.z is not None:
            arrays["tau"] = self.tau
            arrays["z"] = self.z.astype(np.float32)
        for k, v in self.layers.items():
            arrays["layer_" + k] = v.astype(np.float32)
        meta = dict(
            params=asdict(self.params),
            key=self.key,
            frame=self.frame,
            summary=self.summary,
        )
        arrays["meta"] = np.array(json.dumps(meta, default=float))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type]
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Result":
        with np.load(path, allow_pickle=False) as Q:
            meta = json.loads(str(Q["meta"]))
            return cls(
                params=Params.parse(meta["params"]),
                key=meta["key"],
                keep=Q["keep"].astype(np.int64),
                prof=Q["prof"].astype(float),
                frame=meta["frame"],
                tau=Q["tau"].astype(float) if "tau" in Q.files else None,
                z=Q["z"].astype(float) if "z" in Q.files else None,
                layers={
                    k.removeprefix("layer_"): Q[k].astype(float)
                    for k in Q.files
                    if k.startswith("layer_")
                },
                summary=meta["summary"],
            )

    def depth_to_frame(self, row: int, depth: Arr) -> Float:
        """Depths (mm from the root) on row `row` in this result's frame."""
        d = np.asarray(depth, float)
        if self.frame == "depth" or self.z is None or self.tau is None:
            return d
        out: Float = FL.tau_of(
            d, np.broadcast_to(self.z[row], (d.size, self.z.shape[1])), self.tau
        )
        return out


def neighbour_pairs(root_mm: Float, nrm: Float, radius: float) -> tuple[Int, Float]:
    """Pairs (P, 2) of roots within `radius` mm whose normals agree, and distances."""
    pairs = np.array(sorted(cKDTree(root_mm).query_pairs(radius)), np.int64)
    if not len(pairs):
        return np.zeros((0, 2), np.int64), np.zeros(0)
    pairs = pairs[(nrm[pairs[:, 0]] * nrm[pairs[:, 1]]).sum(1) > NDOT]
    d: Float = np.linalg.norm(root_mm[pairs[:, 0]] - root_mm[pairs[:, 1]], axis=1)
    return pairs, d


def correlation(G: Float, pairs: Int) -> Float:
    """Pearson correlation of each pair's two profiles over the samples both have."""
    a, b = G[pairs[:, 0]], G[pairs[:, 1]]
    m = np.isfinite(a) & np.isfinite(b)
    cnt = np.maximum(m.sum(1), 1)
    a0 = np.where(m, a, 0.0)
    b0 = np.where(m, b, 0.0)
    a0 = np.where(m, a0 - (a0.sum(1) / cnt)[:, None], 0.0)
    b0 = np.where(m, b0 - (b0.sum(1) / cnt)[:, None], 0.0)
    den = np.sqrt((a0 * a0).sum(1) * (b0 * b0).sum(1))
    out: Float = np.where(den > 0, (a0 * b0).sum(1) / np.where(den > 0, den, 1.0), 0.0)
    return out


def stack(G: Float, pairs: Int, d: Float, radius: float) -> tuple[Float, Float]:
    """Each profile averaged with its neighbours', sample by sample over the samples
    each has: weight 1 for itself, exp(-d^2 / 2 (radius/2)^2) max(corr, 0) for a
    neighbour. Returns the stacked profiles and each trace's sum of neighbour
    weights."""
    w = np.exp(-(d**2) / (2 * (radius / 2) ** 2)) * np.maximum(correlation(G, pairs), 0)
    ok = np.isfinite(G)
    G0 = np.where(ok, G, 0.0)
    num = G0.copy()
    den = ok.astype(float)
    for a, b in ((0, 1), (1, 0)):
        i, j = pairs[:, a], pairs[:, b]
        np.add.at(num, i, w[:, None] * G0[j])
        np.add.at(den, i, w[:, None] * ok[j])
    wsum = np.zeros(len(G))
    np.add.at(wsum, pairs[:, 0], w)
    np.add.at(wsum, pairs[:, 1], w)
    out: Float = np.where(ok, num / np.maximum(den, 1e-12), np.nan)
    return out, wsum


def compute(inp: Inputs, p: Params, key: str, log: Callable[[str], object]) -> Result:
    """The filter `p` on the traces of `inp`; see the module docstring."""
    t0 = time.time()
    N, n = inp.fv.shape[1], inp.fv.shape[2]
    F = inp.fv.copy()
    M = np.isfinite(F[0]) & np.isfinite(F[1])
    for k in range(F.shape[0]):
        sd = float(np.nanstd(F[k][M])) if M.any() else 1.0
        F[k] /= sd if sd > 0 else 1.0
    sel = {"2": [0], "3": [1], "2+3": [0, 1]}[p.echo]
    G = np.where(M, F[sel].mean(0), np.nan)
    if p.smooth_mm > 0:
        G = np.where(
            M, FL.masked_smooth(np.nan_to_num(G), M, p.smooth_mm / STEP_MM), np.nan
        )
    layers: dict[str, Float] = {}
    summary: dict[str, Any] = dict(traces=int(N))
    frame, tau, z = "depth", None, None
    if p.flatten != "off":
        pairs, _ = neighbour_pairs(inp.root_mm, inp.nrm, PAIR_MM)
        log(f"flattening: {N:,} traces, {len(pairs):,} pairs, shifts ({p.flatten})")
        Fz = np.where(M[None], F, 0.0)
        est = FL.dynamic_shifts if p.flatten == "dyn" else FL.shifts
        S = np.zeros((len(pairs), n), np.float32)
        W = np.zeros((len(pairs), n), np.float32)
        NC = np.zeros((len(pairs), n), np.float32)
        EN = np.zeros((len(pairs), n), np.float32)
        VA = np.zeros((len(pairs), n), bool)
        for a in range(0, len(pairs), CHUNK):
            q = pairs[a : a + CHUNK]
            s, ncc, en, va = est(
                Fz[:, q[:, 0]],
                Fz[:, q[:, 1]],
                M[q[:, 0]],
                M[q[:, 1]],
                STEP_MM,
                smax=p.smax,
            )
            S[a : a + CHUNK], NC[a : a + CHUNK], EN[a : a + CHUNK] = s, ncc, en
            VA[a : a + CHUNK] = va
            log(f"shifts: {min(a + CHUNK, len(pairs)):,} of {len(pairs):,} pairs")
        eref = float(np.median(EN[VA])) if VA.any() else 1.0
        W[:] = FL.pair_weights(NC.astype(float), EN.astype(float), VA, eref)
        del NC, EN, VA
        tau, r, info = FL.fit_warp(
            pairs, S, W, N, STEP_MM, dtau=DTAU, tau_max=TAU_MAX, eps=p.eps, log=log
        )
        z = tau[None] + r
        wt, mis = FL.residuals(pairs, S, W, r, tau, STEP_MM)
        num, den = np.zeros((N, tau.size)), np.zeros((N, tau.size))
        bound = np.zeros(N)
        wcount = np.zeros(N)
        heavy = W > 0.3
        at = heavy & (np.abs(S) >= p.smax - 0.005)
        for c in (0, 1):
            np.add.at(num, pairs[:, c], wt * mis**2)
            np.add.at(den, pairs[:, c], wt)
            np.add.at(bound, pairs[:, c], at.sum(1))
            np.add.at(wcount, pairs[:, c], heavy.sum(1))
        res = np.sqrt(num / np.maximum(den, 1e-9))
        res[den < 0.05] = np.nan
        inside = z[:, 1:] < ((inp.ns[:, None] - 1) * STEP_MM)
        fold = (np.diff(z, axis=1) <= 0) & inside
        cnt = np.isfinite(res).sum(1)
        layers["residual rms (mm)"] = np.where(
            cnt > 0, np.sqrt(np.nansum(res**2, axis=1) / np.maximum(cnt, 1)), np.nan
        )
        layers["fold fraction"] = fold.sum(1) / np.maximum(inside.sum(1), 1)
        layers["shifts at the bound (fraction)"] = np.where(
            wcount > 0, bound / np.maximum(wcount, 1), np.nan
        )
        summary["fit"] = info
        # the profile at constant tau, on a grid of one sample
        tf = np.arange(0, TAU_MAX + 1e-9, STEP_MM)
        zf = np.stack([np.interp(tf, tau, z[i]) for i in range(N)])
        G = FL.resample(G[None], inp.ns, zf, STEP_MM)[0]
        frame = "tau"
    if p.stack_mm > 0:
        sp, sd_ = neighbour_pairs(inp.root_mm, inp.nrm, p.stack_mm)
        log(f"stacking: {len(sp):,} pairs within {p.stack_mm:g} mm")
        G, wsum = stack(G, sp, sd_, p.stack_mm)
        layers["stack weight"] = wsum
    summary["seconds"] = round(time.time() - t0, 1)
    log(f"filter done: {p.label()} ({summary['seconds']} s)")
    return Result(p, key, inp.keep, G, frame, tau, z, layers, summary)
