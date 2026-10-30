"""The pia editor's replay gives the same levels as the live steps, whatever the C rand() state
(QPBO-I's visiting order). Uses the V1V2 data; skipped without them.
    nix develop --command uv run --with thinqpbo pytest -q prototypes/scripts/test_pia_editor.py
"""

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
TAG = "V1V2_banks_s0.16"
pytestmark = pytest.mark.skipif(
    not (HERE / f"joint_{TAG}.npz").exists(), reason="no V1V2 joint result"
)


def test_replay_matches_live_steps_under_any_rand_state(tmp_path):
    import pia_editor as PE

    E = PE.Editor(
        str(HERE), TAG, edits_path=str(tmp_path / "e.json"), log=lambda m: None
    )
    libc = ctypes.CDLL(None)
    for seed, (t, d) in zip((3, 11), ((22649, 2.02), (22343, 2.1))):
        libc.srand(
            seed
        )  # a rand() state that, unseeded, changes the first step (measured)
        E.add_edit(t, d)
        E.resolve(2.0)
    libc.srand(3)
    assert E.replay_check()["replay_max_diff_mm"] == 0.0
    E.save()
    live = E.states[-1].copy()
    libc.srand(5)
    E.load()
    assert np.array_equal(E.states[-1], live)
    assert np.any(live[-1] != E.z0[-1])
