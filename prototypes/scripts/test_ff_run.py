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
import sys
from dataclasses import replace
from pathlib import Path

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


def test_slab_smoke(tmp_path: Path) -> None:
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
