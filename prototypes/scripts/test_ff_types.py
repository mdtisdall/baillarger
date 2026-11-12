"""Tests of the contracts in ff_types.py: the helpers and the files of a run."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402


def test_trilinear_weights_sum_to_one_and_hit_corners():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 1, (50, 3))
    w = T.trilinear_weights(u)
    assert np.allclose(w.sum(1), 1.0)
    for c in range(8):
        corner = np.array([[(c >> a) & 1 for a in range(3)]], float)
        wc = T.trilinear_weights(corner)[0]
        assert wc[c] == 1.0 and wc.sum() == 1.0
    # reproduces a linear function exactly
    corners = np.array([[(c >> a) & 1 for a in range(3)] for c in range(8)], float)
    f = corners @ np.array([1.0, -2.0, 0.5]) + 3.0
    assert np.allclose(w @ f, u @ np.array([1.0, -2.0, 0.5]) + 3.0)


def test_sample_volume_is_exact_on_linear_volumes():
    idx = np.indices((6, 7, 8)).astype(float)
    vol = 2 * idx[0] - idx[1] + 0.25 * idx[2]
    X = np.random.default_rng(1).uniform(0, 5, (30, 3))
    assert np.allclose(T.sample_volume(vol, X), X @ np.array([2.0, -1.0, 0.25]))


def test_log_odds_and_sigmoid():
    p = np.array([0.0, 0.5, 0.9, 1.0])
    z = T.log_odds(p)
    assert z[1] == 0.0 and z[0] == pytest.approx(np.log(0.02 / 0.98))
    assert np.allclose(T.sigmoid(z)[1:3], [0.5, 0.9])


class _G:
    kind = "uniform"
    nodes = np.zeros((3, 3))

    def arrays(self):
        return {"grid_nodes": self.nodes}


def test_field_means():
    n, K = 3, 4
    tau = np.full((n, K), np.nan)
    D = np.zeros((n, K, 3))
    q = np.zeros((n, K))
    tau[0, :2] = [-1.0, -0.5]
    D[0, :2] = [[1, 0, 0], [0, 1, 0]]
    q[0, :2] = [0.3, 0.1]
    q_stop = np.array([0.6, 1.0, 0.0])
    tau[2, 0], D[2, 0], q[2, 0] = -2.0, [0, 0, 1], 1.0
    F = T.Field(_G(), tau, D, q, q_stop, np.array([2, 0, 1]))
    tm = F.tau_mean()
    assert tm[0] == pytest.approx(-0.875) and np.isnan(tm[1]) and tm[2] == -2.0
    dm = F.direction_mean()
    assert np.allclose(dm[0], np.array([0.3, 0.1, 0]) / np.hypot(0.3, 0.1)) and np.allclose(dm[1], 0)
    assert list(F.tau_map()) == [0.0, 0.0, -2.0]


def test_run_record_round_trip(tmp_path):
    r = T.RunRecord(phantom=dict(name="slab"), source="oracle", grid=dict(kind="uniform", spacing_mm=0.16))
    r.free_energy = [[3.0, 2.0], [1.5]]
    r.time_s = 1.5
    p = tmp_path / "run.json"
    r.to_json(p)
    r2 = T.RunRecord.from_json(p)
    assert r2.settings == T.DEFAULTS.record() and r2.free_energy == [[3.0, 2.0], [1.5]]
    assert r2.settings["widths"] == [4, 2, 1] and r2.settings["n_tau"] == 9


def test_truth_npz_round_trip(tmp_path):
    n, m = 2, 4
    rng = np.random.default_rng(2)
    paths = rng.normal(size=(n, m, 3))
    paths[1, 3] = np.nan
    tr = T.Truth("x", roots=paths[:, 0], normals=np.eye(3)[:2], starts=paths[:, 0] - 1, paths=paths,
                 counts=np.array([4, 3]), stops=np.stack([paths[0, 3], paths[1, 2]]),
                 stop_kind=np.array([0, 3], np.int8), bank=np.zeros(2, np.int8),
                 tau_arrival=np.array([0.0, -0.4]), jump=np.array([0.0, 0.4]),
                 offset=np.array([-3.0, -3.1]), tau_scale=1.85, step_mm=0.01, root_spacing_mm=0.32)
    tr.to_npz(tmp_path / "t.npz")
    t2 = T.Truth.from_npz(tmp_path / "t.npz")
    assert t2.phantom == "x" and t2.tau_scale == 1.85
    assert np.array_equal(t2.path(1), tr.path(1)) and np.array_equal(t2.stop_kind, tr.stop_kind)


def test_run_dir(tmp_path):
    d = T.run_dir(tmp_path, "slab", "oracle", 1.0, "u0.16")
    assert d == tmp_path / "slab" / "oracle_1_u0.16" and d.is_dir()
