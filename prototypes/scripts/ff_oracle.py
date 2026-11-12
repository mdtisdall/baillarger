"""The oracle stop field, the oracle tensor and the image tensor of the flow-field phantoms
(docs/plans/flow-field-phantoms.md, *Inputs*; PR 2, task 2.2).

The classes implement `ff_types.StopField` and `ff_types.Tensor`. The image stop field is in
ff_stop.py (task 2.1); `oracle_inputs` here does not import it.

The oracle stop field (provisional decision)
--------------------------------------------
The plan says: "the true stopped region, blurred by a Gaussian of 0.5 voxel SD, as a probability
clipped to [0.02, 0.98]". This module reads that as the closed form

    p(X) = Phi(stop_sdf(X) / sigma),   sigma = oracle_blur_vox * VOXEL_MM   (0.08 mm),

then `ff_types.log_odds(p, *oracle_clip)`. Phi is the standard normal CDF and stop_sdf the
signed distance of the geometry (positive in the stopped region). Reasons:

- Phi(d / sigma) is the exact Gaussian blur of the indicator of a flat stopped half-space, d the
  distance to its surface.
- It can be evaluated at any point, so an octree finer than the image reads it.
- p = 0.5 on the true stop surface, also at a fused contact. A literal blur of the indicator of
  the stopped region, when the region is a sheet thinner than the blur, has a peak below 0.5. For
  a sheet 0.04 mm thick and sigma 0.08 mm, a flat sheet gives a peak of about 0.2 (computed from
  the 1-D blur, not measured on a phantom), so the contact would have no stop at all.

Where the stop surface is curved, or where two parts of the stopped region are closer than a few
sigma, the closed form is not the literal blur. The coordinator accepted this reading as the
provisional decision CD-4. Measured on the PR 1 truth (200 CONTACT stops each, a Monte-Carlo blur
of `geometry.stopped`): the literal blur has a median p of 0.193 at the true stop, and its peak
along the column reaches 0.5 at 10 % of the contact stops of ckiss_smooth and at none of
ckiss_sharp. The literal blur stays available as an option (`literal=True`, `literal_blur`), not
the default.

stop_sdf on its positive side: `Geometry.stop_sdf` promises its value only where it is negative.
Task 2.2 compared it with a brute-force distance (a 0.01 mm grid, a 0.25 mm window) at points
0.01 to 0.164 mm behind the true stops of slab, crown_w0.64, fundus_g0.04, ckiss_smooth and
abraded, taken along the column direction and kept where `stopped` is true. It agreed with the
distance to within 0.014 mm, which is the grid's resolution. Beyond 0.164 mm the clip (0.98)
makes p constant anyway.

The oracle tensor
-----------------
The direction is `ff_types.true_direction` (the unit gradient of phi), the coherence 1, at every
point of the box (phi is defined everywhere).

The image tensor
----------------
`baillarger.orient.orient_arrays` on the phantom's echoes, with the steps and settings of cost 7
of bent_fit.py (its module docstring): the cortex label less the dark outliers of
`bent_texture.outliers` (k = OUTLIER_K), the phantom's noise model `bent_fit.PhantomNoise` (the
rest term alone, gain `orient.gradient_gain(orient.SD)`), orient's SD, SI and SIGMA defaults, the
tensor set to NaN where the box does not hold orient's reach (`bent_texture.valid_mask`,
`orient.orient_reach`).

`Phantom.image` holds only the detrended echoes, so `phantom_log_echoes` rebuilds the phantom's
own noisy log echoes with the same generator calls as the phantom's `make`
(`fold_phantoms.make`, `ff_ckiss.make`, `ff_tau.make_tau`). `ImageTensor.of` detrends them as
`make` does and compares with `image["det"]`. If they differ by more than `REBUILD_TOL` it raises
`ValueError`: it never goes on with echoes that are not the phantom's.

`ImageTensor.at(X)` interpolates the six tensor components trilinearly with a validity weight
(weight 1 at a voxel where the tensor is finite, 0 elsewhere; the weighted components are
interpolated and divided by the interpolated weight). It then takes the leading eigenvector and
the coherence as orient does. Where the interpolated weight is below `orient.MIN_WEIGHT`, or the
tensor has no positive eigenvalue sum, the coherence is 0 and the direction is
`NO_DIRECTION` = (1, 0, 0) (a fixed unit vector in the voxel axes).

NumPy or Torch
--------------
`ImageTensor.at` is NumPy float64 in chunks of CHUNK points (the reference): per point it is 7
trilinear lookups of 8 corners and one 3 x 3 symmetric eigen-decomposition. The callers (the
terms, the integrator) hold NumPy points. The analysis is in the report of task 2.2.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr, ndtri

sys.path.insert(0, str(Path(__file__).parent))
import bent_fit as BF  # noqa: E402
import bent_texture as BTX  # noqa: E402
import ff_ckiss as CK  # noqa: E402
import ff_simple as S  # noqa: E402
import ff_tau as TAU  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

from baillarger import orient as O  # noqa: E402

#: the direction where the image tensor has no valid voxel (the coherence is 0 there)
NO_DIRECTION = np.array([1.0, 0.0, 0.0])
#: the largest |difference| of the rebuilt, detrended echoes and image["det"] that
#: `ImageTensor.of` accepts (the arrays are of order 1)
REBUILD_TOL = 1e-12
#: points per chunk in `ImageTensor.at` (and blur samples per chunk in `literal_blur`)
CHUNK = 1 << 20
#: the offsets per axis of the literal blur (an option, not the default)
LITERAL_NODES = 16


# ---------------------------------------------------------------- the oracle


class OracleStopField:
    """The oracle stop field: p = Phi(stop_sdf / sigma), clipped, as log-odds (module docstring).

    `geometry` is an `ff_types.Geometry`; `settings.oracle_blur_vox` and
    `settings.oracle_clip` give sigma and the clip.
    """

    source: T.InputSource = "oracle"

    def __init__(
        self,
        geometry: T.Geometry,
        settings: T.SolverSettings = T.DEFAULTS,
        literal: bool = False,
    ) -> None:
        self.geometry = geometry
        self.settings = settings
        self.sigma_mm = settings.oracle_blur_vox * T.VOXEL_MM
        self.literal = literal

    def probability(self, X: T.Array) -> T.Array:
        """The unclipped stop probability at voxel positions X: Phi(stop_sdf / sigma), or
        with `literal` the Gaussian blur of the indicator of `geometry.stopped`."""
        Y = self.geometry.to_local(np.asarray(X, float))
        if self.literal:
            return literal_blur(self.geometry, Y, self.sigma_mm)
        return np.asarray(ndtr(self.geometry.stop_sdf(Y) / self.sigma_mm))

    def log_odds(self, X: T.Array) -> T.Array:
        return T.log_odds(self.probability(X), *self.settings.oracle_clip)

    def terms(self, X: T.Array) -> dict[str, T.Array]:
        return {"oracle": self.log_odds(X)}


def literal_blur(geometry: T.Geometry, Y: T.Array, sigma_mm: float) -> T.Array:
    """The Gaussian blur (SD sigma_mm) of the indicator of `geometry.stopped` at local mm
    points Y, on a product grid of LITERAL_NODES offsets per axis at the Gaussian's
    mid-quantiles, equal weights (a stratified rule: on a flat surface the error is at most
    1 / (2 LITERAL_NODES)). Not the default (module docstring): at a fused contact its peak is
    near 0.2."""
    x = ndtri((np.arange(LITERAL_NODES) + 0.5) / LITERAL_NODES)
    G = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3) * sigma_mm
    Y = np.asarray(Y, float)
    out = np.empty(len(Y))
    step = max(1, CHUNK // len(G))
    for i in range(0, len(Y), step):
        P = (Y[i : i + step, None, :] + G[None]).reshape(-1, 3)
        st = np.asarray(geometry.stopped(P), float).reshape(-1, len(G))
        out[i : i + step] = st.mean(1)
    return out


class OracleTensor:
    """The oracle tensor: the true column direction, coherence 1, at every point."""

    source: T.InputSource = "oracle"

    def __init__(self, geometry: T.Geometry) -> None:
        self.geometry = geometry

    def at(self, X: T.Array) -> tuple[T.Array, T.Array]:
        d = T.true_direction(self.geometry, np.asarray(X, float))
        return d, np.ones(len(d))


def oracle_inputs(phantom: T.Phantom) -> T.Inputs:
    """The oracle stop field and tensor of a phantom, with its contrast."""
    if phantom.geometry is None:
        raise ValueError(f"phantom {phantom.name!r} has no geometry, so no oracle")
    return T.Inputs(
        stop=OracleStopField(phantom.geometry),
        tensor=OracleTensor(phantom.geometry),
        contrast=phantom.det,
    )


# ---------------------------------------------------------------- the phantom's echoes


def _render(phantom: T.Phantom, feats: Mapping[str, Any]) -> tuple[Any, list[Any]]:
    """The labels and the two noise-free log echoes of the phantom's family."""
    geo = phantom.geometry
    if geo is None:  # the noise phantom: its renderer is the NoiseFold
        geo = S.geometry("noise", phantom.shape)[0]
    if isinstance(geo, CK.ConnectedKiss):
        return CK.render(geo, feats, bool(geo.p["flat"]))
    if isinstance(geo, TAU.Tau):
        return TAU.render_tau(geo, feats)
    return FP.render(geo, feats)


def phantom_log_echoes(
    phantom: T.Phantom, feats: Mapping[str, Any] | None = None
) -> tuple[Any, T.Array, list[float]]:
    """The labels, the noisy log echoes (2, X, Y, Z) of echo 2 and 3, and the noise SDs of the
    phantom, by the steps and the generator calls of the phantom's `make`: the noise is
    noise * sd * standard_normal from default_rng(seed), drawn echo by echo; sd is matched to the
    site's high-pass statistic. `feats` defaults to phantom.features."""
    feats = phantom.features if feats is None else feats
    labels, imgs = _render(phantom, feats)
    ctx = labels == 0
    rng = np.random.default_rng(phantom.seed)
    out, sds = [], []
    for e, L0 in zip(("echo2", "echo3"), imgs, strict=True):
        s0 = FP.hp_stat(FP.detrend(L0, ctx), ctx)
        z = np.random.default_rng(phantom.seed + 99).standard_normal(L0.shape)
        k1 = FP.hp_stat(FP.detrend(z, ctx), ctx)
        sd = np.sqrt(max(feats[e]["hp_stat"] ** 2 - s0**2, 0.0)) / k1
        sds.append(float(sd))
        out.append(L0 + phantom.noise * sd * rng.standard_normal(L0.shape))
    return labels, np.stack(out), sds


def rebuild_error(phantom: T.Phantom, L: T.Array, labels: Any) -> float:
    """The largest |difference| between the detrended rebuilt echoes and image["det"]."""
    ctx = labels == 0
    return max(
        float(np.max(np.abs(FP.detrend(L[k], ctx) - phantom.det[k + 1])))
        for k in range(2)
    )


# ---------------------------------------------------------------- the image tensor


class ImageTensor:
    """The structure tensor of `baillarger.orient` on the phantom's echoes (module docstring).

    `tensor` is the field (6, X, Y, Z), NaN where it is not valid. The voxel arrays `direction`
    (X, Y, Z, 3) and `coherence` (X, Y, Z) are orient's leading eigenvector (canonical sign) and
    coherence at the voxels, NaN where the tensor is NaN, for the views. Use `of` to build it
    from a phantom.
    """

    source: T.InputSource = "image"

    def __init__(self, tensor: T.Array) -> None:
        self.tensor = np.asarray(tensor, float)
        self.direction, self.coherence = O.leading_any(self.tensor)
        self.valid = np.all(np.isfinite(self.tensor), axis=0)
        # the interpolated channels: the valid components, then the validity weight
        self._chan = np.concatenate(
            [np.where(self.valid, self.tensor, 0.0), self.valid[None].astype(float)]
        )

    @classmethod
    def of(
        cls, phantom: T.Phantom, feats: Mapping[str, Any] | None = None
    ) -> ImageTensor:
        labels, L, sds = phantom_log_echoes(phantom, feats)
        err = rebuild_error(phantom, L, labels)
        if not err <= REBUILD_TOL:
            raise ValueError(
                f"the rebuilt echoes of {phantom.name!r} differ from image['det'] by {err:g}"
            )
        echoes = np.exp(L + BF.LOG_OFFSET)
        cortex = labels == 0
        vs = np.full(3, T.VOXEL_MM)
        odd, _ = BTX.outliers(echoes, cortex, vs, BF.OUTLIER_K)
        a = O.orient_arrays(echoes, cortex & ~odd, BF.PhantomNoise.of(sds), vs)
        J = np.asarray(a["tensor"], float)
        reach = O.orient_reach(vs, O.SIGMA)
        return cls(np.where(BTX.valid_mask(cortex.shape, reach), J, np.nan))

    def at(self, X: T.Array) -> tuple[T.Array, T.Array]:
        X = np.asarray(X, float)
        direction = np.empty((len(X), 3))
        coherence = np.empty(len(X))
        for i in range(0, len(X), CHUNK):
            direction[i : i + CHUNK], coherence[i : i + CHUNK] = self._chunk(
                X[i : i + CHUNK]
            )
        return direction, coherence

    def _chunk(self, X: T.Array) -> tuple[T.Array, T.Array]:
        P = X.T
        v = np.stack(
            [
                ndi.map_coordinates(c, P, order=1, mode="constant", cval=0.0)
                for c in self._chan
            ]
        )
        w = v[6]
        ok = w >= O.MIN_WEIGHT
        t = v[:6, ok] / w[ok]
        direction = np.tile(NO_DIRECTION, (len(X), 1))
        coherence = np.zeros(len(X))
        if t.shape[1]:
            vec, coh = O.leading_any(t)
            good = np.isfinite(coh)
            idx = np.flatnonzero(ok)[good]
            direction[idx] = vec[good]
            coherence[idx] = coh[good]
        return direction, coherence
