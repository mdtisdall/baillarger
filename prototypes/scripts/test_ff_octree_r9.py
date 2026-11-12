"""Tests of the provisional design change for R9 (`ff_octree_r9.py`): the acceptance
of a change of the octree against a control solve of the current grid.

The same boxes and inputs as the adapt tests of `test_ff_octree.py` (PR 5): 24³
voxels, the slab with the oracle inputs and noise 0, the noise phantom with the null
inputs, noise 1.0, seed 0. PR 5's tests and its xfail stay as they are; they test the
original rule.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_octree_r9.py

The verdicts are boolean (accepted or not). There is no numeric tolerance. The tests
were fixed before the first run of the changed rule.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import ff_octree as O  # noqa: E402
import ff_octree_r9 as R9  # noqa: E402
import ff_oracle as FO  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402
from test_ff_octree import FEATS, SHAPE, assert_same_tree  # noqa: E402


@functools.cache
def slab_adapt_control() -> tuple[O.OctreeGrid, dict[str, Any]]:
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=SHAPE)
    return R9.adapt_control(ph, FO.oracle_inputs(ph))


@functools.cache
def noise_adapt_control() -> tuple[O.OctreeGrid, dict[str, Any]]:
    """octree_rounds = 1, as PR 5's noise test: round 0 of the base octree has no
    coarsening, and a refused round 0 ends adapt."""
    ph = FTR.build_phantom("noise", FEATS, noise=1.0, seed=0, shape=SHAPE)
    settings = dataclasses.replace(T.DEFAULTS, octree_rounds=1)
    return R9.adapt_control(ph, O.null_inputs(ph), settings)


def _assert_refused_against_control(g: O.OctreeGrid, rec: dict[str, Any]) -> None:
    json.dumps(rec, allow_nan=True)
    print("\n" + "\n".join(R9.trial_lines(rec)))
    assert rec["acceptance"] == "control"
    trials = [t for r in rec["rounds"] for t in r["trials"] if t["kind"] == "refine"]
    assert trials, "round 0 nominated no refinement"
    for t in trials:
        assert t["solve"]["defects"] == []
        assert t["control"]["defects"] == []
        assert t["free_energy_after"] >= t["free_energy_control"]
        assert not t["accepted"]
    assert_same_tree(g, O.OctreeGrid.base(SHAPE))


def test_noise_control_refuses_every_refinement() -> None:
    """The noise phantom, null inputs: the changed rule refuses every refinement of
    round 0, and the octree stays at 0.64 mm."""
    _assert_refused_against_control(*noise_adapt_control())


def test_slab_control_stays_at_coarsest() -> None:
    """The slab, oracle inputs, noise 0, all defaults: the changed rule refuses every
    refinement, and the octree stays at 0.64 mm."""
    _assert_refused_against_control(*slab_adapt_control())


def test_unknown_acceptance_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        O.adapt((8, 8, 8), None, 0.0, acceptance="other")  # type: ignore[arg-type]
