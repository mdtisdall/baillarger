"""Tests of ff_kiss_runs (the extra runs of PR 4): the shifted bank B and the anchor point.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_kiss_runs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_ckiss as C  # noqa: E402
import ff_kiss_runs as KR  # noqa: E402
import ff_truth as FTR  # noqa: E402

FEATS = {e: dict(profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))), wm_level=-0.3, fluid_level=-2.0, hp_stat=0.0)
         for e in ("echo2", "echo3")}
SMALL = (40, 40, 64)


def test_shift_changes_bank_b_only():
    base = C.ckiss("ckiss_smooth", FEATS, noise=0.0, shape=SMALL)
    sh = KR.shifted_phantom("ckiss_smooth", FEATS, "synthetic", 0.0, 0, shape=SMALL)
    assert np.array_equal(base.image["labels"], sh.image["labels"])
    geo = base.geometry
    X = np.indices(geo.shape).reshape(3, -1).T.astype(float)
    lab, rel, bank, *_ = geo.geom(geo.to_local(X))
    # whole voxels of one bank's cortex, away from the other bank and from tissue borders
    # (the render averages 27 sub-samples over each voxel)
    for e in (1, 2):
        d = (sh.image["det"][e] - base.image["det"][e]).reshape(-1)
        raw_b = np.abs(d[(lab == 0) & (bank == 1)])
        assert raw_b.max() > 1e-3  # bank B changed
    # the raw render (before detrending) of bank A is unchanged at points far from bank B
    Y = geo.to_local(X)
    a = (lab == 0) & (bank == 0) & (Y[:, 0] > 0.5)
    sgeo = KR.ShiftedB(geo.kind, geo.shape, dict(geo.p), geo.R.copy(), geo.origin_mm)
    va = C._values(geo, FEATS, "echo2", Y[a], False)
    vs = C._values(sgeo, FEATS, "echo2", Y[a], False)
    assert np.array_equal(va, vs)
    b = (lab == 0) & (bank == 1)
    rel_s = sgeo.geom(Y[b])[1]
    assert np.allclose(rel_s, rel[b] - KR.SHIFT_MM / C.TB)
    assert type(sh.geometry) is C.ConnectedKiss  # truth and oracle inputs stay those of the base


def test_anchor_point_is_on_bank_a_far_from_the_fundus():
    geo, ts = FTR.geometry_of("ckiss_smooth", (64, 64, 64))  # SMALL has no root inside the margin
    tr = FTR.truth(geo, ts, "ckiss_smooth")
    i, X = KR.anchor_point(tr, geo)
    assert tr.bank[i] == 0
    assert abs(float(geo.phi(geo.to_local(X[None]))[0]) - KR.ANCHOR_PHI) < 0.05
    y = geo.to_local(tr.roots)[tr.bank == 0, 1]
    assert geo.to_local(tr.roots[i : i + 1])[0, 1] >= np.quantile(y, 1 - KR.ANCHOR_FAR_FRACTION)
