"""Smoke test of ff_run: the whole pipeline on a small slab (noise 0, oracle inputs,
grid 0.32 mm), with a truth made in the test. It checks that the files exist, that
run.json loads as a RunRecord and that the measures have their keys. It does not check
any value.

The solver is limited to `max_sweeps=3` to stay under about 60 s (the setting is
recorded in `changed_settings`).

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_run.py
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_meanfield as MF  # noqa: E402
import ff_run as FR  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402

FEATS = {
    e: dict(
        profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))),
        wm_level=-0.3,
        fluid_level=-2.0,
        hp_stat=3.0,
    )
    for e in ("echo2", "echo3")
}
SHAPE = (44, 20, 20)  # holds a start point 2 mm behind the white surface
MEASURE_KEYS = {
    "stops", "paths", "levels", "speeds", "spacing_near_stop", "topology", "direction",
    "jumps", "exact", "inference", "cost", "inputs", "views", "origin",
}  # fmt: skip


def test_parse_settings_and_grid() -> None:
    s, ch = FR.parse_settings(["max_sweeps=3", "widths=[2,1]", "dtau=0.05"])
    assert s.max_sweeps == 3 and s.widths == (2, 1) and s.dtau == 0.05
    assert ch == dict(max_sweeps=3, widths=(2, 1), dtau=0.05)
    with pytest.raises(ValueError):
        FR.parse_settings(["nonsense=1"])
    assert FR.grid_label("uniform:0.16") == "uniform0.16"
    g = FR.make_grid("uniform:0.32", (20, 20, 20))
    assert g.kind == "uniform"
    with pytest.raises(ValueError):
        FR.make_grid("octree", (20, 20, 20))


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One smoke run of the slab, shared by the tests of this module."""
    tmp_path = tmp_path_factory.mktemp("smoke")
    feats = tmp_path / "features.json"
    feats.write_text(json.dumps({"synthetic": FEATS}))
    geo, ts = FTR.geometry_of("slab", SHAPE)
    truth = FTR.truth(geo, ts, "slab", root_spacing_mm=0.48, margin_mm=0.0)
    assert truth.n >= 4
    tdir = tmp_path / "truth" / "slab"
    tdir.mkdir(parents=True)
    truth.to_npz(tdir / "truth.npz")
    rec = FR.main(
        [
            "slab", "--source", "oracle", "--noise", "0", "--grid", "uniform:0.32",
            "--out", str(tmp_path / "out"), "--truth", str(tmp_path / "truth"),
            "--features", str(feats), "--site", "synthetic", "--shape", "44,20,20",
            "--set", "max_sweeps=3", "--quiet",
        ]
    )  # fmt: skip
    d = T.run_dir(tmp_path / "out", "slab", "oracle", 0.0, "uniform0.32")
    return dict(tmp=tmp_path, feats=feats, truth=truth, rec=rec, dir=d)


def test_slab_smoke(smoke: dict[str, Any]) -> None:
    truth, rec, d = smoke["truth"], smoke["rec"], smoke["dir"]
    for f in ("field.npz", "columns.npz", "column_errors.npz", "run.json",
        "section.png",
              "levels.png", "errors.png", "free_energy.png"):  # fmt: skip
        assert (d / f).is_file(), f
        if f.endswith(".png"):
            assert (d / f).stat().st_size < 1_000_000
    back = T.RunRecord.from_json(d / "run.json")
    assert back.source == "oracle"
    assert back.origin == "top"
    assert back.measures["origin"] == dict(
        origin="top", anchor_node=None, anchor_position_vox=None, init=False
    )
    assert back.changed_settings == {"max_sweeps": 3}
    assert back.versions["truth_sha256"]
    assert back.grid["kind"] == "uniform" and back.grid["spacing_mm"] == 0.32
    assert back.sweeps == rec.sweeps and len(back.free_energy) == 3
    assert back.peak_mem_bytes > 0 and back.time_s > 0
    assert MEASURE_KEYS <= set(back.measures)
    m = back.measures
    for k in ("abs_mm", "signed_mm", "end_reasons"):
        assert k in m["stops"]
    for k in ("max_mm", "mean_mm", "end_mm"):
        assert k in m["paths"]
    for k in ("median", "p90", "p99"):
        assert k in m["stops"]["abs_mm"]
    assert "slab" in m["exact"]
    for k in ("columns_straight_normal", "levels_planes", "pass"):
        assert k in m["exact"]["slab"]
    assert isinstance(m["exact"]["slab"]["pass"], bool)
    z = np.load(d / "columns.npz")
    assert z["paths"].shape[0] == truth.n
    assert len(z["levels"]) == 9
    assert "to_true_stop_mm" in m["stops"]
    assert "stop_to_true_stop_mm" in np.load(d / "column_errors.npz").files


def test_anchor_run_records_the_origin(tmp_path: Path) -> None:
    """run passes init, origin and anchor_node to the solver and records them."""
    feats = tmp_path / "features.json"
    feats.write_text(json.dumps({"synthetic": FEATS}))
    geo, ts = FTR.geometry_of("slab", SHAPE)
    truth = FTR.truth(geo, ts, "slab", root_spacing_mm=0.96, margin_mm=0.0)
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=SHAPE)
    grid = FR.make_grid("uniform:0.64", SHAPE)
    inputs = FR.make_inputs(ph, "oracle", None)
    lo = inputs.stop.log_odds(grid.nodes)
    anchor = int(np.argmin(lo))
    s = replace(T.DEFAULTS, max_sweeps=2, widths=(1,))
    coarse, _, _ = MF.solve(grid, inputs, ph.image["noise_sd"], s)
    out = tmp_path / "anchor"
    rec = FR.run(
        ph, inputs, truth, grid, out, source="oracle", grid_spec="uniform:0.64",
        settings=s, changed_settings=dict(max_sweeps=2, widths=(1,)), init=coarse,
        origin="anchor", anchor_node=anchor,
    )  # fmt: skip
    back = T.RunRecord.from_json(out / "run.json")
    assert rec.origin == back.origin == "anchor"
    assert back.measures["origin"] == dict(
        origin="anchor", anchor_node=anchor,
        anchor_position_vox=grid.nodes[anchor].tolist(), init=True,
    )  # fmt: skip
    z = np.load(out / "field.npz")
    assert z["q_stop"][anchor] == 0.0


def test_default_start_is_the_root(smoke: dict[str, Any]) -> None:
    d, truth = smoke["dir"], smoke["truth"]
    back = T.RunRecord.from_json(d / "run.json")
    assert back.measures["columns_from"] == "root"
    z = np.load(d / "columns.npz")
    assert np.array_equal(z["paths"][:, 0], truth.roots)
    with pytest.raises(SystemExit):  # the option exists and has two values
        FR.main(["slab", "--start", "middle"])


_SAME = (
    "stops", "paths", "levels", "speeds", "spacing_near_stop", "topology",
    "direction", "jumps", "exact", "columns_from",
)  # fmt: skip


def test_remeasure_gives_the_runs_measures(smoke: dict[str, Any]) -> None:
    d0: Path = smoke["dir"]
    d1 = smoke["tmp"] / "copy"
    shutil.copytree(d0, d1)
    for f in ("columns.npz", "column_errors.npz", "section.png"):
        (d1 / f).unlink()
    old_field = (d1 / "field.npz").read_bytes()
    rec = FR.main(
        [
            "remeasure", str(d1), "--truth", str(smoke["tmp"] / "truth"),
            "--features", str(smoke["feats"]),
        ]
    )  # fmt: skip
    for f in ("columns.npz", "column_errors.npz", "section.png", "levels.png"):
        assert (d1 / f).is_file(), f
    assert (d1 / "field.npz").read_bytes() == old_field
    a = T.RunRecord.from_json(d0 / "run.json")
    b = T.RunRecord.from_json(d1 / "run.json")
    for k in _SAME:
        assert json.dumps(a.measures[k], sort_keys=True) == json.dumps(
            b.measures[k], sort_keys=True
        ), k
    assert np.array_equal(
        np.load(d0 / "columns.npz")["paths"],
        np.load(d1 / "columns.npz")["paths"],
        equal_nan=True,
    )
    assert b.measures["remeasured"] is True and "remeasured" not in a.measures
    assert b.versions["remeasured"] == "true"
    assert b.versions["remeasure_git_commit"]
    assert b.versions["git_commit"] == a.versions["git_commit"]
    assert b.free_energy == a.free_energy and b.sweeps == a.sweeps == rec.sweeps
    assert b.time_s == a.time_s and b.settings == a.settings
    assert "remeasure" in b.measures["cost"]
