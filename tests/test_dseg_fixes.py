"""Tests for the fixes of the dseg command as a table and a diff
(baillarger.white_dseg, baillarger.fix_diff; docs/formats/wm-dseg.md): ids,
order, replay, revert, the counts and the kept runs."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from test_white_dseg import VOX, synthetic_aseg

from baillarger import bids, white_dseg
from baillarger import fix_diff as D
from baillarger import wm_edits as E


def two_blocks() -> npt.NDArray[np.int16]:
    """The synthetic aseg twice, the second copy across the block border at
    64 on axis 0, with a speck of WM in its cortex (a small piece)."""
    s = synthetic_aseg()
    a = np.zeros((90, 40, 30), np.int16)
    a[:40, :36] = s
    a[45:85, :36] = s
    a[45 + 12, 14, 4] = 41  # in the cortex shell of the second block
    return a


def test_ids_kinds_and_counts() -> None:
    t = white_dseg.wm_topo(two_blocks(), VOX)
    kinds = [f.kind for f in t.fix_list]
    # handle moves, then cavities, then the small piece: ids in that order
    assert kinds[-3:] == ["cavity_fill", "cavity_fill", "small_piece"]
    assert set(kinds[:-3]) == {"handle_cut", "handle_fill"}
    assert [f.id for f in t.fix_list] == list(range(1, len(kinds) + 1))
    # the table's counts are the image's counts, kind by kind
    code = {v: k for k, v in white_dseg.FIXES.items()}
    for kind, c in code.items():
        n = sum(len(f.index) for f in t.fix_list if f.kind == kind)
        assert n == int((t.fixes == c).sum()), kind
    # every changed voxel is in the diff, once
    old = two_blocks()
    assert len(t.diff) == int((t.dseg != old).sum())
    assert {f.kind: f.wall is not None for f in t.fix_list}["handle_cut"]


def test_the_same_input_gives_the_same_ids_and_diff() -> None:
    a = two_blocks()
    t1, t2 = white_dseg.wm_topo(a, VOX), white_dseg.wm_topo(a.copy(), VOX)
    assert t1.diff.tobytes() == t2.diff.tobytes()
    assert [(f.id, f.kind, f.index.tolist()) for f in t1.fix_list] == [
        (f.id, f.kind, f.index.tolist()) for f in t2.fix_list
    ]


def test_the_diff_is_in_morton_order() -> None:
    a = two_blocks()
    t = white_dseg.wm_topo(a, VOX)
    D.check_diff(t.diff, a.shape)
    blocks = np.unravel_index(t.diff["index"], a.shape)[0] // 64
    assert set(blocks.tolist()) == {0, 1}
    assert (np.diff(blocks) >= 0).all()  # block 0 first, then block 1


def test_replay_reproduces_desc_wmtopo_byte_for_byte() -> None:
    a = two_blocks()
    t = white_dseg.wm_topo(a, VOX)
    assert white_dseg.replay_dseg(a, t.diff).tobytes() == t.dseg.tobytes()


def test_replay_with_manual_edits() -> None:
    a = synthetic_aseg()
    rod = [(27, 16, 25), (27, 17, 25)]
    e = E.Edits().with_answer(
        E.Answer(1, rod[0], "remove-strand"), rod, [41, 41], [0, 0]
    )
    t = white_dseg.wm_topo(a, VOX, e)
    assert t.fix_list[0].kind == "manual_edit" and len(t.fix_list[0].index) == 2
    assert white_dseg.replay_dseg(a, t.diff, e).tobytes() == t.dseg.tobytes()
    assert (t.fixes == 4).sum() == 2


def test_revert_changes_only_that_fix() -> None:
    a = two_blocks()
    t = white_dseg.wm_topo(a, VOX)
    for f in t.fix_list:
        r = D.revert_fix(t.dseg, t.diff, f.id)
        changed = np.flatnonzero(r.reshape(-1) != t.dseg.reshape(-1))
        assert changed.tolist() == f.index.tolist()
        assert (r.reshape(-1)[changed] == a.reshape(-1)[changed]).all()


def written(tmp_path: Path, a: npt.NDArray[np.int16]) -> Path:
    src = tmp_path / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(exist_ok=True)
    nib.save(nib.nifti1.Nifti1Image(a, np.diag([*VOX, 1.0])), src)
    return src


def clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs as if from a commit with no uncommitted change."""
    real = bids.provenance

    def prov(function: str) -> dict[str, Any]:
        return dict(real(function), Commit="c0ffee", Dirty=False)

    monkeypatch.setattr(white_dseg, "provenance", prov)


def test_the_command_writes_the_table_the_diff_and_the_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean(monkeypatch)
    a = two_blocks()
    src = written(tmp_path, a)
    out = tmp_path / "deriv"
    p = white_dseg.write_wm_topo(src, out)
    assert p["table"].name == "sub-T01_ses-S1_desc-wmtopofixes_fixes.tsv"
    assert p["diff"].name == "sub-T01_ses-S1_desc-wmtopofixes_diff.npy"
    side = bids.read_sidecar(bids.sidecar_path(p["diff"]), white_dseg.FIXES_SPEC)
    assert side == bids.read_sidecar(
        bids.sidecar_path(p["table"]), white_dseg.FIXES_SPEC
    )
    assert side["DiffInputs"]["Labels"] == bids.file_sha256(src)
    assert side["DiffInputs"]["Code"].endswith("+c0ffee")
    assert side["DiffSHA256"] == bids.file_sha256(p["diff"])
    diff = D.read_diff(p["diff"], a.shape)
    assert side["DiffVoxels"] == len(diff)
    assert p["diff"].stat().st_size <= 256 + 16 * len(diff)
    rows = D.read_table(p["table"])
    assert sum(side["Fixes"].values()) == len(rows)
    img = nib.load(p["dseg"])
    assert isinstance(img, nib.nifti1.Nifti1Image)
    dseg = np.asarray(img.dataobj)
    assert white_dseg.replay_dseg(a, diff).tobytes() == dseg.tobytes()
    hist = white_dseg.history_dir(p["dseg"])
    sha = side["DiffSHA256"]
    assert sorted(q.name for q in hist.iterdir()) == [
        f"{sha}.json",
        f"{sha}.npy",
        f"{sha}.tsv",
    ]
    # a rerun on the same input reproduces the kept diff
    white_dseg.write_wm_topo(src, out)
    assert len(list(hist.glob("*.npy"))) == 1


def test_a_rerun_that_does_not_reproduce_the_kept_diff_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean(monkeypatch)
    src = written(tmp_path, two_blocks())
    out = tmp_path / "deriv"
    p = white_dseg.write_wm_topo(src, out)
    hist = white_dseg.history_dir(p["dseg"])
    (js,) = hist.glob("*.json")
    side = bids.read_sidecar(js, white_dseg.FIXES_SPEC)
    side["DiffSHA256"] = "0" * 64  # as if the kept run had another diff
    bids.write_sidecar(js, white_dseg.FIXES_SPEC, side)
    before = p["dseg"].stat().st_mtime_ns
    with pytest.raises(RuntimeError, match="does not reproduce"):
        white_dseg.write_wm_topo(src, out)
    assert p["dseg"].stat().st_mtime_ns == before  # nothing written


def test_a_run_with_uncommitted_changes_is_not_checked(tmp_path: Path) -> None:
    hist = tmp_path / "h"
    hist.mkdir()
    white_dseg.check_reproduced(hist, dict(Labels="x", Code="y"), True, "z")
