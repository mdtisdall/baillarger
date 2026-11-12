"""Tests of the helpers of ff_pr6_views (PR 6, hairpins, jumps and the grid). Small
synthetic inputs only: no solver run, no site data, no scratch paths.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_pr6_views.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_pr6_views as V  # noqa: E402
import ff_simple as S  # noqa: E402
import ff_truth as FTR  # noqa: E402


def test_bank_check_compares_with_the_true_stop_bank() -> None:
    solved = np.array([0, 0, 1, 1, 0])
    true = np.array([0, 1, 1, 1, 0])
    root = np.array([0, 0, 1, 1, 1])  # column 1 and 4 differ from their true stop
    stopped = np.array([True, True, True, False, True])
    x = np.array([0.5, 0.01, -0.5, 0.2, 0.3])
    r = V.bank_check(solved, true, stopped, solved_x=x, root_bank=root)
    assert r["n_stopped"] == 4
    assert r["n_wrong"] == 1  # column 1
    assert r["n_wrong_clear"] == 0  # its x is 0.01 mm: within the tolerance
    assert r["n_true_stop_not_in_root_bank"] == 2  # columns 1 and 4
    r2 = V.bank_check(solved, true, stopped, solved_x=np.full(5, 0.5))
    assert r2["n_wrong_clear"] == 1


def test_bank_of_the_fundus_geometry() -> None:
    geo, _ = FTR.geometry_of("fundus_g0.16")
    Y = np.array([[0.5, 1.0, 0.0], [-0.5, 1.0, 0.0], [0.1, -0.2, 0.0]])
    assert list(geo.bank(Y)) == [0, 1, 0]


def test_patch_split_on_a_hand_made_truth() -> None:
    patches = S.abraded_patches(0.16)  # centres y = 3 and 8, radius 2, z = 0
    stops = np.array(
        [
            [0.7, 3.0, 0.5],  # patch 0
            [1.1, 8.5, -1.0],  # patch 1
            [0.16, 5.5, 0.0],  # intact (between the patches)
            [0.16, 3.0, 0.5],  # in patch 0's cylinder but true jump 0: intact
        ]
    )
    jt = np.array([0.3, 0.9, 0.0, 0.0])
    assert list(V.patch_index(stops, jt, patches)) == [0, 1, -1, -1]


def test_relative_to_intact_and_group_statistics() -> None:
    jt = np.array([0.0, 0.0, 0.0, 0.3, 0.3, 0.9])
    js = np.array([0.04, 0.04, 0.08, 0.34, 0.38, 0.5])
    stopped = np.array([True, True, True, True, True, False])
    patch = np.array([-1, -1, -1, 0, 0, 1])
    m, rel = V.relative_to_intact(js, patch < 0)
    assert abs(m - 0.04) < 1e-12 and abs(rel[3] - 0.30) < 1e-12
    r = V.jump_groups(js, jt, stopped, patch, np.zeros(6, int), dtau=0.04)
    assert abs(r["median_intact_jump"] - 0.04) < 1e-12
    p0 = r["groups"]["patch0"]
    assert p0["n_stopped"] == 2
    assert abs(p0["err_raw_dtau"]["median"] - 1.5) < 1e-9  # errors 1 and 2 dtau
    assert abs(p0["err_raw_dtau"]["max"] - 2.0) < 1e-9
    assert abs(p0["err_rel_dtau"]["median"] - 0.5) < 1e-9  # 0.30-0.3=0, 0.34-0.3=1
    assert r["groups"]["patch1"]["n_stopped"] == 0  # not stopped: masked
    assert r["groups"]["intact"]["n_stopped"] == 3


def test_nearest_moving_value_skips_stopped_nodes() -> None:
    nodes = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    q_stop = np.array([0.9, 0.1, 0.2])
    vals = np.array([10.0, 20.0, 30.0])
    v, d = V.nearest_moving_value(
        nodes, q_stop, vals, np.array([[0.1, 0, 0], [2.9, 0, 0]])
    )
    assert list(v) == [20.0, 30.0]
    assert abs(d[0] - 0.9) < 1e-12
    v0, _ = V.nearest_moving_value(nodes, np.ones(3), vals, np.array([[0.0, 0, 0]]))
    assert np.isnan(v0[0])


def test_fine_cell_distance_to_the_tip() -> None:
    g = 0.04
    voxel = 0.16
    # centres in local mm: on K, at g + voxel (not farther), and farther
    cen = np.array(
        [[0.0, 0.0, 0.0], [0.0, -(g + voxel), 0.0], [0.0, -0.5, 1.0], [0.3, 0.4, 0.0]]
    )
    size = np.array([0.04, 0.08, 0.08, 0.64])
    fine = size < voxel
    r = V.fine_cell_report(cen, size, fine, g, voxel=voxel)
    assert r["n_fine"] == 3
    # distances 0, 0.2, hypot(0, -0.5) = 0.5: only the last is farther than 0.2
    assert r["n_fine_far"] == 1
    assert r["by_size_mm"]["0.08"] == dict(n=2, n_far=1)
    # the nearest point is the centre distance minus 0.87 * size: 0.5 - 0.069 is still far
    assert r["n_fine_far_nearest_point"] == 1


def test_focal_place_report() -> None:
    g = 0.16
    places = np.array([[0.0, -0.1, 0.0], [0.0, -2.0, 3.0]])
    r = V.place_report(places, g)
    assert r["n"] == 2 and r["n_within_g_plus_voxel"] == 1
    assert abs(r["r_min_mm"] - 0.1) < 1e-12
    assert V.place_report(np.zeros((0, 3)), g)["n"] == 0


def test_bank_a_surface_coordinates() -> None:
    s, z = V.bank_a_surface(
        np.array([[2.0, 3.0, 1.0], [2.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    )
    assert list(s[:2]) == [3.0, 0.0] and z[0] == 1.0
    assert abs(s[2] + 2.0 * np.pi / 2) < 1e-12  # the bottom of the hairpin, r = 2


def test_section_polygon_of_an_axis_aligned_cell() -> None:
    geo, _ = FTR.geometry_of("fundus_g0.48")
    origin = np.asarray(geo.to_vox(np.array([[0.0, 0.0, -0.1]])))
    # A small cell: it is cut by the plane z = 0 only if it reaches it.
    polys, idx = V.section_polygons(
        geo, origin, np.array([0.64]), 0.0, (-5.0, 5.0), (-5.0, 5.0)
    )
    assert len(polys) == len(idx) <= 1
    for p in polys:
        assert p.shape[1] == 2 and len(p) >= 3


def test_missing_run_directories_are_skipped_and_listed(tmp_path: Path) -> None:
    ids = V.expected_ids(["fundus_g0.04"], ["uniform0.16", "octree"], ["oracle"], [0.0])
    d = tmp_path / "fundus_g0.04" / "oracle_0_octree"
    d.mkdir(parents=True)
    (d / "run.json").write_text("{}")
    (tmp_path / "fundus_g0.04" / "oracle_0_uniform0.16").mkdir()  # no run.json
    found, skipped = V.find_runs(tmp_path, ids)
    assert [r.grid for r in found] == ["octree"]
    assert [r.grid for r in skipped] == ["uniform0.16"]
    assert "fundus_g0.04/oracle_0_uniform0.16" in V.skipped_md(skipped)
    # the noise phantom is present with an adapt.json only
    n = V.RunId("noise", "oracle", 1.0, "octree")
    dn = tmp_path / "noise" / n.dirname
    dn.mkdir(parents=True)
    (dn / "adapt.json").write_text("{}")
    f2, s2 = V.find_runs(tmp_path, [n], V._need(n))
    assert list(f2) == [n] and not s2


def test_adapt_summary_counts_accepted_trials() -> None:
    adapt = {
        "final": {
            "level_counts": {"n_nodes": 10, "nodes_per_level": {"0": 10}},
            "prior_shaped": {"count": 0},
        },
        "rounds": [
            {
                "round": 0,
                "trials": [
                    {"kind": "refine", "accepted": True, "delta_free_energy": -3.0},
                    {"kind": "coarsen", "accepted": False, "delta_free_energy": 1.0},
                ],
            },
            {"round": 1, "trials": [{"kind": "refine", "accepted": False}]},
        ],
    }
    s = V.adapt_summary(json.loads(json.dumps(adapt)))
    assert (
        s["n_rounds"] == 2
        and s["n_refine_accepted"] == 1
        and s["n_coarsen_accepted"] == 0
    )
    assert s["rounds"][1]["coarsen_accepted"] is None
