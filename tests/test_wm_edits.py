"""Tests for the answer log (baillarger.wm_edits): its files, its records and
the checks on append. The dseg command does not read the log; its edits are
in baillarger.label_edits (tests/test_label_edits.py)."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from baillarger import bids
from baillarger import wm_edits as E

SHAPE = (12, 10, 8)
SHA = "ab" * 32


@pytest.fixture
def log(tmp_path: Path) -> E.EditLog:
    a = np.zeros(SHAPE, np.int16)
    a[2:10, 2:8, 2:6] = 41
    dseg = tmp_path / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(a, np.diag([0.16, 0.16, 0.16, 1.0])), dseg)
    p = E.edits_paths(tmp_path / "deriv", dict(sub="T01", ses="S1"))
    assert p["log"].name == "sub-T01_ses-S1_desc-wmedits_edits.jsonl"
    return E.EditLog.create(p["log"], dseg, "R")


def session(log: E.EditLog, *sites: tuple[int, int, int]) -> int:
    s = log.start_session(0.25, 7, SHA, SHA)
    for k, site in enumerate(sites, 1):
        log.shown(s, site, "misfacing", k, None)
    return s


def test_the_files_and_the_sidecar(log: E.EditLog) -> None:
    p = log.paths
    assert p["voxels"].name.endswith("_desc-wmedits_edits.bin")
    assert p["sidecar"].name.endswith("_desc-wmedits_edits.json")
    side = bids.read_sidecar(p["sidecar"], E.EDITS_SPEC)
    assert side["Shape"] == list(SHAPE) and side["Hemisphere"] == "R"
    assert len(side["Affine"]) == 16 and side["Affine"][0] == pytest.approx(0.16)
    with pytest.raises(FileExistsError):
        E.EditLog.create(p["log"], Path(side["Sources"][0]), "R")
    test = E.edits_paths(Path("d"), dict(sub="T01", ses="S1"), desc="wmeditstest")
    assert test["log"].name == "sub-T01_ses-S1_desc-wmeditstest_edits.jsonl"


def test_records_round_trip(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3), (5, 5, 5))
    log.skipped(s, (5, 5, 5))
    a = log.answer(
        s,
        (3, 3, 3),
        "remove-strand",
        [(3, 3, 3), (3, 4, 3)],
        [0, 42],
        SHA,
        12.5,
        params=dict(radius=1, reach=4),
    )
    b = log.scribble(s, [(41, [(1, 1, 1)]), (42, [(0, 0, 0), (0, 0, 1)])], SHA, 3.0)
    L = E.read_log(log.path)
    assert [r["Record"] for r in L.records] == [
        "session",
        "shown",
        "shown",
        "skipped",
        "answer",
        "answer",
    ]
    ra = L.answer(a)
    assert ra["Kind"] == "site" and ra["Choice"] == "remove-strand"
    assert ra["SiteKind"] == "misfacing" and ra["How"] == "ranked" and ra["Rank"] == 1
    assert ra["ReviewerSeconds"] == 12.5 and ra["DecidedAgainst"] == SHA
    assert ra["Params"] == dict(radius=1, reach=4)
    v, lab, st = L.voxels(a)
    assert v.tolist() == [[3, 3, 3], [3, 4, 3]] and lab.tolist() == [0, 42]
    assert st.tolist() == [-1, -1]
    rb = L.answer(b)
    assert rb["Kind"] == "scribble" and rb["Site"] is None and rb["How"] == "none"
    assert rb["Strokes"] == [dict(Stroke=0, Class=41), dict(Stroke=1, Class=42)]
    assert L.voxels(b)[2].tolist() == [0, 1, 1]
    assert [x["Session"] for x in L.sessions()] == [1]
    assert L.sessions()[0]["RandomShare"] == 0.25
    assert L.sessions()[0]["RandomSeed"] == 7
    assert log.answered(s) == {(3, 3, 3)}
    assert log.sessions() == L.sessions()
    # 16 bytes per voxel
    assert log.paths["voxels"].stat().st_size == 16 * 5


def test_how_a_site_was_chosen_comes_from_the_shown_record(log: E.EditLog) -> None:
    s = log.start_session(0.5, 3, SHA, SHA)
    log.shown(s, (4, 4, 4), "random", None)
    a = log.answer(s, (4, 4, 4), "leave", [], [], SHA, 1.0)
    r = E.read_log(log.path).answer(a)
    assert r["How"] == "random" and r["Rank"] is None and r["SiteKind"] == "random"


def test_an_answer_at_a_site_not_shown_is_refused(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    with pytest.raises(ValueError, match="not shown"):
        log.answer(s, (9, 9, 7), "leave", [], [], SHA, 1.0)
    t = session(log)
    with pytest.raises(ValueError, match="not shown"):
        log.answer(t, (3, 3, 3), "leave", [], [], SHA, 1.0)  # another session
    with pytest.raises(ValueError, match="not shown"):
        log.skipped(t, (3, 3, 3))
    with pytest.raises(ValueError, match="not shown"):
        log.scribble(t, [(42, [(1, 1, 1)])], SHA, 1.0, site=(3, 3, 3))
    with pytest.raises(ValueError, match="not started"):
        log.shown(99, (1, 1, 1), "misfacing", 1)
    assert [r["Record"] for r in E.read_log(log.path).records] == [
        "session",
        "shown",
        "session",
    ]


def test_bad_answers_are_refused(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    with pytest.raises(ValueError, match="unknown choice"):
        log.answer(s, (3, 3, 3), "paint", [(1, 1, 1)], [41], SHA, 1.0)
    with pytest.raises(ValueError, match="leave has no voxels"):
        log.answer(s, (3, 3, 3), "leave", [(1, 1, 1)], [41], SHA, 1.0)
    with pytest.raises(ValueError, match="outside the grid"):
        log.answer(s, (3, 3, 3), "thicken", [(12, 0, 0)], [41], SHA, 1.0)
    with pytest.raises(ValueError, match="twice"):
        log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)] * 2, [41, 41], SHA, 1.0)
    with pytest.raises(ValueError, match="label set"):
        log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)], [7], SHA, 1.0)
    with pytest.raises(ValueError, match="supersedes unknown"):
        log.answer(s, (3, 3, 3), "leave", [], [], SHA, 1.0, supersedes=[5])
    assert E.read_log(log.path).answers() == []
    assert log.paths["voxels"].stat().st_size == 0


def test_supersession_is_a_new_record_and_the_old_one_stays(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    a = log.answer(s, (3, 3, 3), "fill-gap", [(1, 1, 1)], [41], SHA, 1)
    before = log.path.read_text()
    b = log.answer(s, (3, 3, 3), "leave", [], [], SHA, 1, supersedes=[a])
    after = log.path.read_text()
    assert after.startswith(before)  # nothing is rewritten
    L = E.read_log(log.path)
    assert [x["Id"] for x in L.answers()] == [a, b]
    assert L.answer(b)["Supersedes"] == [a]


def test_revert_for_inspection_does_not_change_the_file(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    a = log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)], [41], SHA, 1)
    text = log.path.read_text()
    L = E.read_log(log.path).without(a)
    assert L.answers() == [] and log.path.read_text() == text
    with pytest.raises(KeyError):
        L.without(a)


def test_an_append_that_did_not_finish_is_ignored(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)], [41], SHA, 1)
    # a crash: voxels appended, the line cut off
    with log.paths["voxels"].open("ab") as fh:
        fh.write(b"\0" * 20)
    with log.path.open("a") as fh:
        fh.write('{"Record":"answ')
    L = E.read_log(log.path)
    assert len(L.answers()) == 1
    b = log.answer(s, (3, 3, 3), "fill-gap", [(2, 2, 2)], [41], SHA, 1)
    L = E.read_log(log.path)
    assert [x["Id"] for x in L.answers()] == [1, b]
    assert L.voxels(b)[0].tolist() == [[2, 2, 2]]
    assert log.paths["voxels"].stat().st_size == 32


def test_a_changed_voxel_file_is_refused(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)], [41], SHA, 1)
    raw = bytearray(log.paths["voxels"].read_bytes())
    raw[8] ^= 1  # the label
    log.paths["voxels"].write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256"):
        E.read_log(log.path)


def test_the_log_is_json_lines(log: E.EditLog) -> None:
    session(log, (3, 3, 3))
    lines = log.path.read_text().splitlines()
    assert [json.loads(x)["Record"] for x in lines] == ["session", "shown"]


def _count_voxel_reads(monkeypatch: pytest.MonkeyPatch, log: E.EditLog) -> list[str]:
    """Each read of the .bin file, from now on: its open mode."""
    reads: list[str] = []
    real_open, real_bytes = Path.open, Path.read_bytes

    def open_(self: Path, mode: str = "r", *a: object, **k: object) -> object:
        if self == log.paths["voxels"] and "r" in mode:
            reads.append(mode)
        return real_open(self, mode, *a, **k)  # type: ignore[call-overload]

    def read_bytes(self: Path) -> bytes:
        if self == log.paths["voxels"]:
            reads.append("read_bytes")
        return real_bytes(self)

    monkeypatch.setattr(Path, "open", open_)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return reads


def test_a_new_record_does_not_read_the_log_again(
    log: E.EditLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = session(log, (3, 3, 3), (5, 5, 5))
    log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1), (1, 1, 2)], [41, 41], SHA, 1)

    def refuse(path: Path) -> None:
        raise AssertionError("the whole log is read again")

    monkeypatch.setattr(E, "read_log", refuse)
    monkeypatch.setattr(E, "_read_all", refuse)
    reads = _count_voxel_reads(monkeypatch, log)
    b = log.answer(s, (5, 5, 5), "fill-gap", [(2, 2, 2)], [41], SHA, 1)
    log.skipped(s, (3, 3, 3))
    t = log.start_session(0, None, SHA, SHA)
    L = log.read()
    log.sessions(), log.answered(s)
    # the .bin file is written (r+b), not read
    assert reads == ["r+b"]
    assert [r["Record"] for r in L.records][-3:] == ["answer", "skipped", "session"]
    assert t == 2 and b == 2
    assert L.voxels(b)[0].tolist() == [[2, 2, 2]]  # mapped, read when asked
    assert L.voxels(1)[0].tolist() == [[1, 1, 1], [1, 1, 2]]


def test_check_log_verifies_every_record(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)], [41], SHA, 1)
    raw = bytearray(log.paths["voxels"].read_bytes())
    raw[8] ^= 1  # the label of the first answer, which the memory copy has
    log.paths["voxels"].write_bytes(bytes(raw))
    # an append checks the new record only
    b = log.answer(s, (3, 3, 3), "fill-gap", [(2, 2, 2)], [41], SHA, 1)
    assert b == 2
    with pytest.raises(ValueError, match="line 3: the SHA-256"):
        log.check_log()
    with pytest.raises(ValueError, match="SHA-256"):
        E.read_log(log.path)


def test_check_log_on_a_good_log(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    a = log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)], [41], SHA, 1)
    L = log.check_log()
    assert L.records == E.read_log(log.path).records == log.read().records
    assert L.voxels(a)[0].tolist() == [[1, 1, 1]]


def test_the_appends_of_another_writer_are_read_and_checked(
    log: E.EditLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = session(log, (3, 3, 3))
    other = E.EditLog(log.path)
    a = other.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)], [41], SHA, 1)
    other.shown(s, (5, 5, 5), "misfacing", 2)
    # this writer reads only the new part: the lines and their voxels
    reads = _count_voxel_reads(monkeypatch, log)
    b = log.answer(s, (5, 5, 5), "fill-gap", [(2, 2, 2)], [41], SHA, 1)
    assert reads == ["rb", "r+b"]
    assert (a, b) == (1, 2)
    L = log.read()
    assert L.answer(b)["Voxels"]["Offset"] == 1  # after the other's voxels
    assert L.answer(b)["Rank"] == 2  # from the other's shown record
    assert L.records == E.read_log(log.path).records
    # a bad append by another writer is refused, with its line
    c = other.answer(s, (3, 3, 3), "leave", [], [], SHA, 1)
    with log.path.open("a") as fh:
        fh.write(json.dumps(dict(Record="what")) + "\n")
    with pytest.raises(ValueError, match="line 7: unknown record"):
        log.read()
    with pytest.raises(ValueError, match="line 7: unknown record"):
        log.read()  # no memory copy is left half-updated
    assert c == 3


def test_a_replaced_log_is_read_again_in_full(log: E.EditLog) -> None:
    s = session(log, (3, 3, 3))
    log.answer(s, (3, 3, 3), "thicken", [(1, 1, 1)], [41], SHA, 1)
    text = log.path.read_text().splitlines(keepends=True)
    tmp = log.path.with_name("new.jsonl")
    tmp.write_text("".join(text[:2]))  # the session and the shown record
    tmp.replace(log.path)
    L = log.read()
    assert [r["Record"] for r in L.records] == ["session", "shown"]
    assert log.answer(s, (3, 3, 3), "leave", [], [], SHA, 1) == 1
