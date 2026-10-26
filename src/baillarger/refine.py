"""One run of the model on a patch (plan steps 1-6), in memory.

The merge loop (baillarger.regions) gives the regions; at each re-fit level
(the numbers of regions in levels) the model is fitted again:

- step 3b: the bands of each region (baillarger.bands), which also start the
  columns' boundaries at the region's Z_R;
- n_inner rounds of: templates (step 3), gain/offset (step 2), templates,
  and the boundaries of each region by optimal surfaces (step 4), with the TV
  on the boundaries only between columns of the same region;
- step 5: the bottom of each region (baillarger.bottom);
- step 6: the TV on the band values (baillarger.tv), which feeds nothing back.

Before the first re-fit the gain and offset of each column are the mean and
SD of its cortex, per echo. Each level's result keeps the regions, bands,
boundaries, templates, gains and offsets, the energy after each inner round
(data, gain, TV on the boundaries, pull) and the bottoms; the merge tree
keeps every merge.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from baillarger import bands as bd
from baillarger import bottom as bt
from baillarger import gain as gn
from baillarger import regions as rg
from baillarger import surfaces as sf
from baillarger import tv
from baillarger.columns import ColumnData

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

DEFAULT_LEVELS = [256, 128, 64, 32, 16, 8, 5, 4, 3, 2]


@dataclass(frozen=True)
class Params:
    """The weights and settings of the model (plan: Weights)."""

    h: float = 1 / 50  # depth grid bin (fraction of thickness)
    zstep: float = 1 / 200  # depth step of the min-cut
    k_max: int = 8
    t_min: float = 0.05  # minimum band thickness
    kappa: float = 1.0  # gain/offset smoothness
    # TV on the boundaries; 5 since the edge zones are out: on V1V2 it halves
    # the step between linked columns (0.010 to 0.005 of r) for 0.4 % data
    lambda_z: float = 5.0
    eta: float = 10.0  # pull to the region's bands
    lambda_g: float = 0.5  # TV on the band values, group term
    lambda_l: float = 0.0  # TV on the band values, per-depth term
    delta_a: float = 0.1
    delta_b: float = 0.1
    border_hops: int = 2
    n_inner: int = 3
    min_cols_cv: int = 6
    gain_frac: float = 0.05
    seed: int = 0
    border: float = 0.0  # Potts term of the merge cost (off)
    merge_m_max: float = 0.85
    window: Literal["surface", "region"] = "surface"
    tv_iter: int = 2000
    tv_tol: float = 1e-5


@dataclass
class Level:
    """The model at one re-fit level."""

    regions: dict[int, list[int]]
    reg: I64  # region (0..nR-1) of each column
    K: dict[int, int]
    Z: dict[int, F64]
    cv: dict[int, F64]
    z: F64  # boundaries of each column (n, K_max - 1), inf where none
    mus: dict[int, F64]  # template (K_R, E) of each region
    a: F64
    s: F64
    energies: list[dict[str, float]]
    bottoms: dict[int, tuple[float, str]]
    u: F64 = field(default_factory=lambda: np.zeros(0))  # TV band values
    xhat: F64 = field(default_factory=lambda: np.zeros(0))


class Refiner:
    """The model on one voxel table (baillarger.columns.ColumnData)."""

    def __init__(self, data: ColumnData, params: Params | None = None) -> None:
        self.D = data
        self.p = params or Params()
        self.rng = np.random.default_rng(self.p.seed)
        D = data
        cx = D.ctx
        c = D.col[cx]
        n, E = D.n, D.n_echoes
        cnt = np.bincount(c, minlength=n)
        m1 = np.stack(
            [np.bincount(c, D.f[cx, e], n) / np.maximum(cnt, 1) for e in range(E)], 1
        )
        m2 = np.stack(
            [np.bincount(c, D.f[cx, e] ** 2, n) / np.maximum(cnt, 1) for e in range(E)],
            1,
        )
        self.a = m1
        self.s = np.sqrt(np.maximum(m2 - m1**2, 1e-12))

    # the cortex voxels, as arrays
    def _cortex(self) -> tuple[I64, F64, F64]:
        D = self.D
        cx = D.ctx
        return D.col[cx], D.m[cx], D.f[cx]

    def stats(self) -> tuple[F64, F64]:
        """The per-column profiles for the merge costs, from the current gain
        and offset."""
        c, m, f = self._cortex()
        return rg.column_grid_stats(
            c,
            m,
            f,
            self.a,
            self.s,
            self.D.sigma2,
            self.D.n,
            self.p.h,
            self.p.merge_m_max,
        )

    def refit(self, regions: dict[int, list[int]]) -> Level:
        """Steps 3b, 2-4 (n_inner rounds), 5 and 6 at the given regions."""
        D, p = self.D, self.p
        n = D.n
        c, m, f = self._cortex()
        rid = {R: i for i, R in enumerate(sorted(regions))}
        reg = np.empty(n, np.int64)
        for R, cols in regions.items():
            reg[cols] = rid[R]
        nR = len(rid)
        # 3b: the bands of each region
        xh = (f - self.a[c]) / np.where(self.s[c] != 0, self.s[c], 1.0)
        w = self.s[c] ** 2 / D.sigma2[c]
        order = np.argsort(reg[c], kind="stable")
        bounds = np.searchsorted(reg[c][order], np.arange(nR + 1))
        vox = [order[bounds[R] : bounds[R + 1]] for R in range(nR)]
        K: dict[int, int] = {}
        Z: dict[int, F64] = {}
        CV: dict[int, F64] = {}
        for R in range(nR):
            v = vox[R]
            K[R], Z[R], CV[R] = bd.bands_for_region(
                c[v], m[v], f[v], xh[v], w[v], D.sigma2, n, p.h, p.k_max, p.t_min,
                self.rng, p.min_cols_cv, gain_frac=p.gain_frac,
            )  # fmt: skip
        kmax = max(K.values())
        z = np.full((n, max(kmax - 1, 1)), np.inf)
        for R in range(nR):
            if K[R] > 1:
                z[reg == R, : K[R] - 1] = Z[R]
        offs = np.concatenate([[0], np.cumsum([K[R] for R in range(nR)])])
        energies = []
        mus: dict[int, F64] = {}
        for _ in range(p.n_inner):
            bandv = bd.band_of(m, z[c])
            # 3: templates, 2: gain/offset, 3 again
            mus = gn.update_templates(c, reg, bandv, f, self.a, self.s, D.sigma2, K)
            t = np.concatenate([mus[R] for R in range(nR)])[offs[reg[c]] + bandv]
            self.a, self.s = gn.solve_gain(
                c, f, t, D.sigma2, D.edges, D.w, p.kappa, self.a, self.s
            )
            mus = gn.update_templates(c, reg, bandv, f, self.a, self.s, D.sigma2, K)
            # 4: boundaries, one min-cut per region (TV only inside regions)
            for R in range(nR):
                if K[R] < 2:
                    continue
                cr = np.flatnonzero(reg == R)
                e = D.edges
                inside = (reg[e[:, 0]] == R) & (reg[e[:, 1]] == R)
                v = vox[R]
                z[cr, : K[R] - 1] = sf.surfaces_region(
                    cr, c[v], m[v], f[v], D.sigma2, Z[R], mus[R], self.a, self.s,
                    e[inside], D.w[inside], p.zstep, p.eta, p.lambda_z, p.t_min,
                    p.window,
                )  # fmt: skip
            bandv = bd.band_of(m, z[c])
            mus = gn.update_templates(c, reg, bandv, f, self.a, self.s, D.sigma2, K)
            energies.append(self.energy(reg, z, mus, K, Z, bandv))
        level = Level(
            regions={R: list(v) for R, v in regions.items()},
            reg=reg,
            K=K,
            Z=Z,
            cv=CV,
            z=z,
            mus=mus,
            a=self.a.copy(),
            s=self.s.copy(),
            energies=energies,
            bottoms={},
        )
        level.bottoms = self.bottoms(level)
        if p.lambda_g > 0 or p.lambda_l > 0:
            Kc = np.array([K[reg[i]] for i in range(n)])
            bandv = bd.band_of(m, z[c])
            level.u, level.xhat, _ = tv.tv_bands(
                c, m, f, self.a, self.s, bandv, z, Kc, D.edges, D.w, p.h,
                p.lambda_g, p.lambda_l, p.tv_iter, p.tv_tol,
            )  # fmt: skip
        return level

    def energy(
        self,
        reg: I64,
        z: F64,
        mus: dict[int, F64],
        K: dict[int, int],
        Z: dict[int, F64],
        bandv: I64,
    ) -> dict[str, float]:
        """The terms of the energy (plan: Model)."""
        D, p = self.D, self.p
        c, _, f = self._cortex()
        nR = len(K)
        offs = np.concatenate([[0], np.cumsum([K[R] for R in range(nR)])])
        t = np.concatenate([mus[R] for R in range(nR)])[offs[reg[c]] + bandv]
        data = float(np.sum((f - self.a[c] - self.s[c] * t) ** 2 / D.sigma2[c]))
        i, j = D.edges[:, 0], D.edges[:, 1]
        sig2 = np.median(D.sigma2, axis=0)
        da = (self.a[i] - self.a[j]) ** 2 + (self.s[i] - self.s[j]) ** 2
        gain = float(p.kappa * np.sum(D.w[:, None] * da / sig2))
        same = reg[i] == reg[j]
        tvz = 0.0
        pull = 0.0
        for R in range(nR):
            k = K[R] - 1
            if k < 1:
                continue
            sel = same & (reg[i] == R)
            tvz += p.lambda_z * float(
                np.sum(D.w[sel, None] * np.abs(z[i[sel], :k] - z[j[sel], :k]))
            )
            pull += p.eta * float(np.sum((z[reg == R, :k] - Z[R]) ** 2))
        return dict(
            data=data, gain=gain, tv_z=tvz, pull=pull, total=data + gain + tvz + pull
        )

    def bottoms(self, level: Level) -> dict[int, tuple[float, str]]:
        """Step 5 at one level."""
        D, p = self.D, self.p
        reg = level.reg
        nR = len(level.K)
        cx = D.ctx
        # template value of every voxel (cortex; the WM's are replaced)
        tmpl = np.zeros((len(D.m), D.n_echoes))
        c = D.col[cx]
        bandv = bd.band_of(D.m[cx], level.z[c])
        offs = np.concatenate([[0], np.cumsum([level.K[R] for R in range(nR)])])
        allmu = np.concatenate([level.mus[R] for R in range(nR)])
        tmpl[cx] = allmu[offs[reg[c]] + bandv]
        e = D.edges
        nbrs: dict[int, set[int]] = {R: set() for R in range(nR)}
        for i, j in e.tolist():
            if reg[i] != reg[j]:
                nbrs[int(reg[i])].add(int(reg[j]))
                nbrs[int(reg[j])].add(int(reg[i]))
        dense = {R: [int(x) for x in np.flatnonzero(reg == R)] for R in range(nR)}
        return bt.bottom_depths(
            D.col, D.m, D.f, self.a, self.s, D.sigma2, reg, dense, nbrs, tmpl, e,
            p.h, p.border_hops, p.delta_a, p.delta_b,
        )  # fmt: skip

    def run(
        self, levels: list[int] | None = None
    ) -> tuple[list[tuple[int, int, int, int, float, int]], dict[int, Level]]:
        """The merge loop with re-fits at levels (default DEFAULT_LEVELS):
        the merge tree and the model at each level."""
        tree, out = rg.merge_regions(
            self.D.n,
            self.D.edges,
            self.D.w,
            self.stats,
            DEFAULT_LEVELS if levels is None else levels,
            self.refit,
            self.p.border,
        )
        return tree, out
