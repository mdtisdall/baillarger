"""Tests of baillarger.viewer.review through its routes on a real server: a
phantom queue, a temporary log, phase 1.3's log writer and reader, and stand-in
choices (the routes do not depend on what a choice does)."""

import json
import multiprocessing
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from edits_helpers import as_dict, edits, write_dseg_sidecar, write_edits

from baillarger import bids, blocks, fix_diff, white_dseg
from baillarger import label_edits as LE
from baillarger import wm_edits as E
from baillarger import wm_sites as WS
from baillarger.viewer import review as VR
from baillarger.viewer import server as VS

SHAPE = (20, 18, 16)
A = np.diag([0.16, 0.16, 0.16, 1.0])
QUEUE = [
    (1, "override", (5, 5, 5)),
    (2, "misfacing", (8, 6, 7)),
    (3, "handle-fix", (12, 9, 4)),
]
RANDOM = [(3, 12, 10), (14, 4, 9), (6, 14, 12), (9, 9, 9)]
#: the fix of the override site, and the voxels that it changes to 0
FIX = 1
FIX_VOXELS = [(5, 5, 6), (5, 5, 7)]


def row(rank: int, kind: str, c: tuple[int, int, int]) -> dict[str, Any]:
    r: dict[str, Any] = {k: 0 for k in WS.COLUMNS}
    r.update(rank=rank, kind=kind, i=c[0], j=c[1], k=c[2], tissue="grey matter")
    r.update(fix=FIX if kind == "override" else 0)
    r.update(wm_thickness=1.0, gap_width=0.0)
    return r


class Fake:
    """Stand-in choices: the answer sets the site to 42 and its +x neighbour to
    41; the correction then changes the voxel +y of the site; fill-gap
    overrides. A preview with a change (manual-paint) gives the change's
    voxels as the answer. delay: seconds a preview takes."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seen: list[tuple[Any, ...]] = []

    def half(self, params: dict[str, Any]) -> int:
        return 4

    def answer(
        self, labels: Any, lo: Any, site: Any, choice: str, params: dict[str, Any]
    ) -> VR.Answer:
        self.seen.append(("answer", labels.shape, lo, site, choice, params))
        v = np.array([site, (site[0] + 1, site[1], site[2])], np.int64)
        lab = np.array([42, 41]) if choice != "leave" else np.zeros(0, np.int64)
        return VR.Answer(v if choice != "leave" else v[:0], lab, dict(params))

    def preview(
        self,
        labels: Any,
        lo: Any,
        site: Any,
        choice: str,
        params: dict[str, Any],
        change: VR.Answer | None = None,
    ) -> VR.Preview:
        time.sleep(self.delay)
        if change is not None:
            a = change
        else:
            a = self.answer(labels, lo, site, choice, params)
        c = np.array([(site[0], site[1] + 1, site[2])], np.int64)
        return VR.Preview(a.voxels, c, choice == "fill-gap")


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    lab = np.zeros(SHAPE, np.int16)
    lab[3:16, 3:15, 3:13] = 41
    dseg = tmp_path / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(lab, A), dseg)
    q = tmp_path / "sub-T01_ses-S1_hemi-R_desc-wmsites_sites.tsv"
    with q.open("w") as fh:
        fh.write("\t".join(WS.COLUMNS) + "\n")
        for r in QUEUE:
            fh.write("\t".join(str(row(*r)[c]) for c in WS.COLUMNS) + "\n")
    bids.write_sidecar(
        q.with_suffix(".json"),
        WS.SITES_SPEC,
        dict(
            SpecVersion=WS.SITES_SPEC.version,
            Description="a test queue",
            **bids.sources(dseg),
            Provenance=bids.provenance("tests"),
            Hemisphere="R",
            FilledLabels=[41],
            MisfacingFaces=1,
            Sites=len(QUEUE),
            GroupVoxels=3.0,
            FixVoxels=4,
            TissueVoxels=3,
            ContactVoxels=4,
            CoreVoxels=2,
            MinWallVoxels=10,
            SitesByKind={k: 1 for k in WS.KINDS},
        ),
    )
    log = E.edits_paths(
        tmp_path / "deriv", dict(sub="T01", ses="S1"), desc="wmeditstest"
    )
    # the input labels: the dseg's, except two voxels that the correction changed
    inp = lab.copy()
    inp[6, 5, 5] = 0
    inp[9, 6, 7] = 0
    inp[8, 6, 7] = 42
    src = tmp_path / "sub-T01_ses-S1_desc-input_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(inp, A), src)
    # the fix diff: fix FIX sets FIX_VOXELS to 0, fix 2 sets (12, 9, 5) to 0
    new = inp.copy()
    fv = [*FIX_VOXELS, (12, 9, 5)]
    for v in fv:
        new[v] = 0
    idx = np.ravel_multi_index(tuple(np.array(fv).T), SHAPE)
    fd = tmp_path / "sub-T01_ses-S1_desc-wmtopofixes_diff.npy"
    fix_diff.write_diff(fd, fix_diff.make_diff(inp, new, idx, [FIX, FIX, 2]))
    edits = LE.bids_path(tmp_path / "deriv", dict(sub="T01", ses="S1"), "R")
    # the dseg's copy of the edits that it used: none
    copy = white_dseg.output_paths(dseg)["edits"]
    LE.write(copy, LE.empty(), hemisphere="R", shape=SHAPE, affine=A)
    return dict(
        dseg=dseg,
        queue=q,
        log=log["log"],
        edits=edits,
        input=src,
        fixdiff=fd,
        copy=copy,
    )


def randoms(n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(RANDOM))[:n]
    return [row(0, "random", RANDOM[i]) for i in pick]


def make(files: dict[str, Path], choices: Fake | None = None) -> VR.Review:
    lab = np.asarray(nib.nifti1.load(files["dseg"]).dataobj)
    return VR.Review(
        files["queue"],
        files["log"],
        files["edits"],
        files["dseg"],
        lambda: lab,
        A,
        randoms,
        choices or Fake(),
        inputs=files["input"],
    )


@pytest.fixture
def server(files: dict[str, Path]) -> Iterator[tuple[str, VR.Review, dict[str, Path]]]:
    rv = make(files)
    srv = VS.serve(None, 0, white=object(), wmreview=rv)  # type: ignore[arg-type]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", rv, files
    finally:
        srv.shutdown()
        rv.close()
        VS.Handler.white = None
        VS.Handler.wmreview = None


def get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def post(url: str, body: dict[str, Any]) -> Any:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def records(files: dict[str, Path]) -> list[dict[str, Any]]:
    return list(E.read_log(files["log"]).records)


def test_a_session_records_its_share_seed_and_each_site_shown(server: Any) -> None:
    base, rv, files = server
    q = get(base + "/review/queue")
    assert q["session"] is None and [s["id"] for s in q["sites"]] == [
        "5,5,5",
        "8,6,7",
        "12,9,4",
    ]
    assert q["classes"][0]["key"] == "wm" and q["classes"][1]["value"] == 42
    s = post(base + "/review/session", dict(random_share=0.5, seed=3))["session"]
    # share 0.5 of the session's sites: 3 ranked, 3 random
    assert s["n_ranked"] == 3 and s["n_random"] == 3
    shown = [get(base + "/review/next")["site"] for _ in range(6)]
    assert get(base + "/review/next")["site"] is None
    r = records(files)
    assert r[0]["Record"] == "session"
    assert r[0]["RandomShare"] == 0.5 and r[0]["RandomSeed"] == 3
    assert r[0]["DsegSHA256"] == bids.file_sha256(files["dseg"])
    assert r[0]["QueueSHA256"] == bids.file_sha256(files["queue"])
    sh = [x for x in r if x["Record"] == "shown"]
    assert [VR.site_id(x["Site"]) for x in sh] == [s_["id"] for s_ in shown]
    assert sorted(x["How"] for x in sh) == ["random"] * 3 + ["ranked"] * 3
    # the ranked sites keep their order; a random site shows no kind or rank
    ranked = [x for x in shown if x["kind"] is not None]
    assert [x["id"] for x in ranked] == ["5,5,5", "8,6,7", "12,9,4"]
    assert all(x["rank"] is None for x in shown if x["kind"] is None)
    for x in sh:
        assert (x["Rank"] is None) == (x["How"] == "random")


def test_the_share_needs_a_value_and_a_seed(server: Any) -> None:
    base, _, _ = server
    for body in ({}, dict(random_share=1.5, seed=1), dict(random_share=0.2)):
        with pytest.raises(urllib.error.HTTPError) as e:
            post(base + "/review/session", body)
        assert e.value.code == 400
    # share 0: no seed needed, no random site
    s = post(base + "/review/session", dict(random_share=0))["session"]
    assert s["n_random"] == 0 and s["n_ranked"] == 3


def test_a_skip_is_recorded_and_the_site_stays_unanswered(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    site = get(base + "/review/next")["site"]["id"]
    post(base + "/review/skip", dict(site=site))
    r = records(files)
    assert r[-1]["Record"] == "skipped" and VR.site_id(r[-1]["Site"]) == site
    assert E.read_log(files["log"]).answered(rv.session) == set()
    st = {s["id"]: s["state"] for s in get(base + "/review/queue")["sites"]}
    assert st == {"5,5,5": "skipped", "8,6,7": "not shown", "12,9,4": "not shown"}
    # next goes on to the site not shown
    assert get(base + "/review/next")["site"]["id"] == "8,6,7"


def test_an_answer_appends_one_record_with_its_time_and_how(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    site = get(base + "/review/next")["site"]["id"]
    n = len(records(files))
    out = post(
        base + "/review/answer",
        dict(site=site, choice="thicken", params=dict(radius=2), reviewer_seconds=7.25),
    )
    r = records(files)
    assert len(r) == n + 1
    a = r[-1]
    assert a["Record"] == "answer" and a["Id"] == out["answer"]
    assert (
        a["Kind"] == "site"
        and a["Choice"] == "thicken"
        and a["Params"] == dict(radius=2)
    )
    assert a["ReviewerSeconds"] == 7.25
    assert a["How"] == "ranked" and a["Rank"] == 1 and a["SiteKind"] == "override"
    assert a["DecidedAgainst"] == bids.file_sha256(files["dseg"])
    v, lab, st = E.read_log(files["log"]).voxels(a["Id"])
    assert v.tolist() == [[5, 5, 5], [6, 5, 5]] and lab.tolist() == [42, 41]
    assert out["kind"] == "override" and out["how"] == "ranked"
    st_ = {s["id"]: s["state"] for s in get(base + "/review/queue")["sites"]}
    assert st_[site] == "answered"


def test_an_answer_at_a_site_not_shown_is_refused(server: Any) -> None:
    base, _, files = server
    with pytest.raises(urllib.error.HTTPError) as e:  # no session
        post(
            base + "/review/answer",
            dict(site="5,5,5", choice="leave", reviewer_seconds=1),
        )
    assert e.value.code == 400
    post(base + "/review/session", dict(random_share=0, seed=None))
    with pytest.raises(urllib.error.HTTPError) as e:  # the log refuses it
        post(
            base + "/review/answer",
            dict(site="5,5,5", choice="leave", reviewer_seconds=1),
        )
    assert e.value.code == 400 and "not shown" in json.loads(e.value.read())["error"]
    assert [r["Record"] for r in records(files)] == ["session"]


def test_a_random_site_shows_its_kind_after_the_answer(server: Any) -> None:
    base, _, files = server
    post(base + "/review/session", dict(random_share=0.5, seed=3))
    while True:
        s = get(base + "/review/next")["site"]
        if s["kind"] is None:
            break
    q = {x["id"]: x for x in get(base + "/review/queue")["sites"]}
    assert q[s["id"]]["kind"] is None and q[s["id"]]["state"] == "shown"
    out = post(
        base + "/review/answer",
        dict(site=s["id"], choice="leave", reviewer_seconds=2.0),
    )
    assert out["kind"] == "random" and out["how"] == "random"
    q = {x["id"]: x for x in get(base + "/review/queue")["sites"]}
    assert q[s["id"]]["kind"] == "random"
    a = records(files)[-1]
    assert a["How"] == "random" and a["SiteKind"] == "random" and a["Rank"] is None


def test_a_scribble_appends_its_strokes_and_classes(server: Any) -> None:
    base, _, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    strokes = [
        {"class": 41, "voxels": [[4, 4, 4], [4, 5, 4]]},
        {
            "class": 42,
            "voxels": [[4, 5, 4], [7, 7, 7]],
        },  # (4, 5, 4) again: the later wins
    ]
    out = post(base + "/review/scribble", dict(strokes=strokes, reviewer_seconds=4.5))
    a = records(files)[-1]
    assert a["Kind"] == "scribble" and a["Id"] == out["answer"]
    assert a["Strokes"] == [{"Stroke": 0, "Class": 41}, {"Stroke": 1, "Class": 42}]
    assert a["ReviewerSeconds"] == 4.5 and a["How"] == "none"
    v, lab, st = E.read_log(files["log"]).voxels(a["Id"])
    got = sorted(zip(map(tuple, v.tolist()), lab.tolist(), st.tolist(), strict=True))
    assert got == [((4, 4, 4), 41, 0), ((4, 5, 4), 42, 1), ((7, 7, 7), 42, 1)]


def test_two_answers_stay_in_order_and_the_log_reads_back(server: Any) -> None:
    base, _, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    s1 = get(base + "/review/next")["site"]["id"]
    s2 = get(base + "/review/next")["site"]["id"]
    a1 = post(
        base + "/review/answer", dict(site=s2, choice="leave", reviewer_seconds=1.0)
    )
    a2 = post(
        base + "/review/answer",
        dict(site=s1, choice="remove-strand", reviewer_seconds=3.0),
    )
    L = E.read_log(files["log"])  # phase 1.3's reader checks every record
    ans = L.answers()
    assert [a["Id"] for a in ans] == [a1["answer"], a2["answer"]]
    assert [VR.site_id(a["Site"]) for a in ans] == [s2, s1]
    assert [a["Choice"] for a in ans] == ["leave", "remove-strand"]
    assert VR.check_log(files["log"]) == len(L.records)


def test_the_preview_route_equals_the_choices_preview(server: Any) -> None:
    base, rv, _ = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    out = post(
        base + "/review/preview", dict(site="5,5,5", choice="fill-gap", params={})
    )
    t = time.monotonic() + 60
    while out["status"] == "running":  # the first preview starts the worker
        assert time.monotonic() < t
        time.sleep(0.05)
        out = get(base + "/review/preview?job=" + out["job"])
    box, lo = rv.window((5, 5, 5), {})
    ref = Fake().preview(box, lo, (5, 5, 5), "fill-gap", {})
    assert out["status"] == "done"
    assert (
        out["answer"] == ref.answer.tolist()
        and out["correction"] == ref.correction.tolist()
    )
    assert out["overrides"] is True
    # the box round the site: BOX voxels each way, clipped to the grid
    assert lo == (1, 1, 1) and box.shape == (9, 9, 9)


def test_a_slow_preview_runs_as_a_job(review: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(VR, "WAIT_S", 0.05)
    rv = review(Fake(delay=0.3))
    out = rv.preview("5,5,5", "thicken", dict(radius=1))
    assert out["status"] == "running"
    out = poll(rv, out)
    assert out["status"] == "done" and out["answer"] == [[5, 5, 5], [6, 5, 5]]


def test_a_restart_resumes_the_last_session(files: dict[str, Path]) -> None:
    rv = make(files)
    rv.start(0.5, 3)
    first = [rv.next()["site"]["id"] for _ in range(2)]
    rv.skip(first[0])
    again = make(files)  # the server starts again on the same log
    assert again.session == rv.session
    assert [r["i"] for r in again.order] == [r["i"] for r in rv.order]
    st = {s["id"]: s["state"] for s in again.queue()["sites"]}
    assert st[first[0]] == "skipped" and st[first[1]] == "shown"
    assert again.next()["site"]["id"] not in first


def test_no_review_and_unknown_routes_are_404(server: Any) -> None:
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + "/review/nothing")
    assert e.value.code == 404
    srv = VS.serve(None, 0, white=object())  # type: ignore[arg-type]
    b2 = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            get(b2 + "/review/queue")
        assert e.value.code == 404
    finally:
        srv.shutdown()


def test_random_count() -> None:
    assert VR.random_count(44, 0) == 0
    assert VR.random_count(44, 0.2) == 11  # 11 of 55
    assert VR.random_count(44, 1) == 44
    with pytest.raises(ValueError):
        VR.random_count(44, -0.1)


def test_the_real_choices_through_the_route(files: dict[str, Path]) -> None:
    from baillarger import wm_choices as wc

    lab = np.asarray(nib.nifti1.load(files["dseg"]).dataobj).astype(np.int64)
    rv = VR.Review(
        files["queue"],
        files["log"],
        files["edits"],
        files["dseg"],
        lambda: lab,
        A,
        randoms,
        inputs=files["input"],
    )
    rv.start(0, None)
    rv.next()
    for choice, params in [("thicken", dict(radius=1)), ("remove-strand", {})]:
        out = rv.preview("5,5,5", choice, params)
        while out["status"] == "running":
            time.sleep(0.1)
            out = rv.job(out["job"])
        assert out["status"] == "done", out
        ref = wc.preview(lab, (5, 5, 5), "R", choice, voxel_mm=[0.16] * 3, **params)
        assert out["answer"] == ref.answer.voxels.tolist()
        assert out["correction"] == ref.correction.voxels.tolist()
        assert out["overrides"] == ref.overrides
        assert out["cut_by_box"] == ref.cut_by_box
    # the answer writes the voxels and labels of wm_choices.run
    a = rv.answer("5,5,5", "thicken", dict(radius=2), 3.0)
    ch = wc.run("thicken", lab, (5, 5, 5), "R", radius=2)
    v, labels, _ = E.read_log(files["log"]).voxels(a["answer"])
    order = np.lexsort(ch.voxels.T[::-1])
    assert v.tolist() == ch.voxels[order].tolist()
    assert labels.tolist() == ch.labels[order].tolist()
    rv.close()


# ------------------------------------------------------- answers into the diff


def diff(files: dict[str, Path]) -> dict[tuple[int, ...], int]:
    """The label edits as {voxel: label}."""
    return as_dict(LE.read(files["edits"]), SHAPE)


def test_an_answer_writes_its_labels_into_the_diff(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    out = post(
        base + "/review/answer",
        dict(site="5,5,5", choice="thicken", params={}, reviewer_seconds=1.0),
    )
    # (5, 5, 5): 42 on dseg 41; (6, 5, 5): 41 is its dseg label (the
    # correction's, on input 0): no edit
    assert diff(files) == {(5, 5, 5): 42}
    assert out["edited"] == 1
    side = json.loads(LE.edits_paths(files["edits"])["sidecar"].read_text())
    assert side["Hemisphere"] == "R" and side["Shape"] == list(SHAPE)
    assert np.allclose(np.reshape(side["Affine"], (4, 4)), A)
    # the log still records the answer, for the queue
    a = records(files)[-1]
    assert a["Record"] == "answer" and a["Id"] == out["answer"]


def test_an_answer_writes_only_the_voxels_whose_label_changes(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    get(base + "/review/next")
    post(
        base + "/review/answer",
        dict(site="5,5,5", choice="thicken", params={}, reviewer_seconds=1.0),
    )
    # (9, 6, 7): 41 is its dseg label, and it has no entry: not written
    out = post(
        base + "/review/answer",
        dict(site="8,6,7", choice="thicken", params={}, reviewer_seconds=1.0),
    )
    assert out["edited"] == 1
    assert diff(files) == {(5, 5, 5): 42, (8, 6, 7): 42}


def test_an_answer_at_a_corrected_voxel_is_an_edit(server: Any) -> None:
    """(8, 6, 7) is 42 in the input labels and 41 in the dseg (the
    correction's). An answer that sets it back to 42 writes an entry 42, so
    the next regeneration does not make the same correction there."""
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    get(base + "/review/next")
    out = post(
        base + "/review/answer",
        dict(site="8,6,7", choice="thicken", params={}, reviewer_seconds=1.0),
    )
    assert out["edited"] > 0
    assert diff(files) == {(8, 6, 7): 42}


def test_a_later_answer_replaces_the_labels_of_an_earlier_one(
    files: dict[str, Path],
) -> None:
    rv = make(files)
    rv.start(0, None)
    rv.next()
    write_edits(files["edits"], edits({(5, 5, 5): 0}, SHAPE), SHAPE, A)
    rv.answer("5,5,5", "thicken", {}, 1.0)
    assert diff(files) == {(5, 5, 5): 42}


def test_leave_and_a_refused_answer_leave_no_edit(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    with pytest.raises(urllib.error.HTTPError):  # not shown: the log refuses it
        post(
            base + "/review/answer",
            dict(site="8,6,7", choice="thicken", reviewer_seconds=1),
        )
    # the diff was written first, then written back: empty, with no undo
    assert diff(files) == {} and rv.undo == []
    get(base + "/review/next")
    out = post(
        base + "/review/answer", dict(site="5,5,5", choice="leave", reviewer_seconds=1)
    )
    assert out["edited"] == 0 and diff(files) == {} and rv.undo == []


def test_a_diff_on_another_grid_is_refused(files: dict[str, Path]) -> None:
    LE.write(files["edits"], LE.empty(), hemisphere="R", shape=(4, 4, 4), affine=A)
    with pytest.raises(ValueError, match="grid"):
        make(files)
    LE.write(files["edits"], LE.empty(), hemisphere="L", shape=SHAPE, affine=A)
    with pytest.raises(ValueError, match="hemisphere"):
        make(files)


def test_the_input_labels_default_to_the_dseg_sources(files: dict[str, Path]) -> None:
    lab = np.asarray(nib.nifti1.load(files["dseg"]).dataobj)
    write_dseg_sidecar(files["dseg"], files["input"])
    assert white_dseg.input_labels(files["dseg"]) == files["input"]
    rv = VR.Review(
        files["queue"],
        files["log"],
        files["edits"],
        files["dseg"],
        lambda: lab,
        A,
        randoms,
        Fake(),
    )
    rv.start(0, None)
    rv.next()
    rv.answer("5,5,5", "thicken", {}, 1.0)
    assert diff(files) == {(5, 5, 5): 42}


# ------------------------------------------------------- accept the fix


def test_accept_fix_writes_the_fix_labels_into_the_diff(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    p = post(base + "/review/preview", dict(site="5,5,5", choice=VR.ACCEPT))
    assert p["status"] == "done" and p["answer"] == [list(v) for v in FIX_VOXELS]
    assert p["correction"] == [] and p["overrides"] is False
    out = post(
        base + "/review/answer",
        dict(site="5,5,5", choice=VR.ACCEPT, reviewer_seconds=2.0),
    )
    assert out["edited"] == 2
    assert diff(files) == {v: 0 for v in FIX_VOXELS}
    a = records(files)[-1]
    assert a["Choice"] == VR.ACCEPT and a["Params"] == {"Fix": FIX}
    v, lab, _ = E.read_log(files["log"]).voxels(a["Id"])
    assert v.tolist() == [list(x) for x in FIX_VOXELS] and lab.tolist() == [0, 0]


def test_accept_fix_is_only_a_choice_of_an_override_site(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    get(base + "/review/next")
    for route_ in ("/review/answer", "/review/preview"):
        with pytest.raises(urllib.error.HTTPError) as e:
            post(
                base + route_,
                dict(site="8,6,7", choice=VR.ACCEPT, reviewer_seconds=1.0),
            )
        assert e.value.code == 400 and "override" in json.loads(e.value.read())["error"]
    assert not files["edits"].exists()
    assert [r["Record"] for r in records(files)][-1] == "shown"


# ------------------------------------------------------- the command line


def test_sites_need_the_log_and_the_edits(files: dict[str, Path]) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    VR.add_arguments(ap)
    for argv in (
        ["--sites", "q.tsv", "--log", "l.jsonl"],
        ["--sites", "q.tsv", "--edits", "e.npy"],
        ["--log", "l.jsonl"],
        ["--log", "l.jsonl", "--edits", "e.npy"],
    ):
        a = ap.parse_args(argv)
        with pytest.raises(ValueError):
            VR.for_white(a, Path("s"), Path("d"), lambda: np.zeros(1), A)
    a = ap.parse_args([])
    assert VR.for_white(a, Path("s"), Path("d"), lambda: np.zeros(1), A) is None
    # --edits alone: paint only, no queue and no log
    paint_only(files)
    a = ap.parse_args(["--edits", str(files["edits"])])
    rv = VR.for_white(a, Path("s"), files["dseg"], lambda: np.zeros(1), A)
    assert rv is not None and rv.queue()["paint_only"] is True


def test_the_fix_diff_path_is_the_dseg_commands() -> None:
    ents, root = dict(sub="T01", ses="S1"), Path("/d")
    d = bids.derivative_path(root, ents, "dseg", ".nii.gz", desc="wmtopo")
    want = bids.derivative_path(root, ents, "diff", ".npy", desc="wmtopofixes")
    assert white_dseg.dseg_path(root, ents) == d
    assert white_dseg.output_paths(d)["diff"] == want
    with pytest.raises(ValueError, match="not a corrected segmentation"):
        white_dseg.output_paths(d.with_name("sub-T01_ses-S1_aseg.nii.gz"))


# ------------------------------------------------------- paint and undo


def paint_strokes(*strokes: tuple[Any, list[tuple[int, int, int]]]) -> Any:
    return [dict(label=lab, voxels=[list(v) for v in vs]) for lab, vs in strokes]


def test_the_shared_strokes_function_keeps_a_voxel_in_its_later_stroke() -> None:
    strokes = [
        {"class": 41, "voxels": [[4, 4, 4], [4, 5, 4]]},
        {"class": 42, "voxels": [[4, 5, 4], [7, 7, 7]]},
        {"class": 0, "voxels": [[7, 7, 7]]},
    ]
    out = VR.stroke_voxels(strokes, "class")
    assert [(lab, v.tolist()) for lab, v in out] == [
        (41, [[4, 4, 4]]),
        (42, [[4, 5, 4]]),
        (0, [[7, 7, 7]]),
    ]
    # a stroke left with no voxel is dropped
    out = VR.stroke_voxels(strokes[1:2] + strokes[1:2], "class")
    assert [(lab, v.tolist()) for lab, v in out] == [(42, [[4, 5, 4], [7, 7, 7]])]


def test_a_paint_writes_the_painted_voxels_and_labels_into_the_diff(
    server: Any,
) -> None:
    base, rv, files = server
    # no session is needed: paint is not a queue answer
    out = post(
        base + "/review/paint",
        dict(
            strokes=paint_strokes(
                (42, [(4, 4, 4), (4, 5, 4)]), (0, [(4, 5, 4), (10, 10, 10)])
            )
        ),
    )
    assert out == dict(voxels=3, undo_depth=1)
    assert diff(files) == {(4, 4, 4): 42, (4, 5, 4): 0, (10, 10, 10): 0}
    # the paint is not a record of the log
    assert all(r["Record"] != "answer" for r in records(files))


def test_a_label_not_in_the_menu_is_refused_and_nothing_is_written(
    server: Any,
) -> None:
    base, rv, files = server
    for bad in (2, 3, 17, "wm", None):  # 2 and 3 are the left hemisphere's
        with pytest.raises(urllib.error.HTTPError) as e:
            post(
                base + "/review/paint",
                dict(strokes=paint_strokes((42, [(4, 4, 4)]), (bad, [(5, 5, 5)]))),
            )
        assert e.value.code == 400
    assert not files["edits"].exists()
    assert rv.undo == []


def test_a_voxel_with_its_current_label_is_not_written(server: Any) -> None:
    base, rv, files = server
    # (4, 4, 4) is 41 in the dseg; (8, 6, 7) is 41 there (its input label: 42)
    out = post(
        base + "/review/paint",
        dict(strokes=paint_strokes((41, [(4, 4, 4)]), (41, [(8, 6, 7)]))),
    )
    assert out == dict(voxels=0, undo_depth=0)
    assert not files["edits"].exists()
    post(base + "/review/paint", dict(strokes=paint_strokes((0, [(4, 4, 4)]))))
    out = post(base + "/review/paint", dict(strokes=paint_strokes((0, [(4, 4, 4)]))))
    assert out == dict(voxels=0, undo_depth=1)


def test_the_eraser_gives_a_voxel_its_input_label_again(server: Any) -> None:
    base, rv, files = server
    post(
        base + "/review/paint",
        dict(strokes=paint_strokes((0, [(4, 4, 4), (5, 4, 4)]))),
    )
    out = post(
        base + "/review/paint",
        dict(strokes=paint_strokes(("input", [(4, 4, 4), (6, 6, 6)]))),
    )
    # (6, 6, 6) has no entry: erasing it changes nothing
    assert out == dict(voxels=1, undo_depth=2)
    assert diff(files) == {(5, 4, 4): 0}


def test_a_voxel_off_the_grid_is_refused(server: Any) -> None:
    base, rv, files = server
    for v in ((20, 0, 0), (-1, 4, 4), (4, 4, 16)):
        with pytest.raises(urllib.error.HTTPError) as e:
            post(
                base + "/review/paint",
                dict(strokes=paint_strokes((42, [(4, 4, 4), v]))),
            )
        assert e.value.code == 400 and "grid" in json.loads(e.value.read())["error"]
    assert not files["edits"].exists() and rv.undo == []


def test_undo_restores_the_diff_exactly_after_overlapping_paints(
    server: Any,
) -> None:
    base, rv, files = server
    write_edits(files["edits"], edits({(4, 4, 4): 0}, SHAPE), SHAPE, A)
    d0 = diff(files)
    post(
        base + "/review/paint",
        dict(strokes=paint_strokes((42, [(4, 4, 4), (4, 5, 4), (4, 6, 4)]))),
    )
    d1 = diff(files)
    out = post(
        base + "/review/paint",
        dict(
            strokes=paint_strokes((0, [(4, 5, 4), (4, 7, 4)]), ("input", [(4, 4, 4)]))
        ),
    )
    assert out["undo_depth"] == 2
    assert diff(files) == {(4, 5, 4): 0, (4, 6, 4): 42, (4, 7, 4): 0}
    out = post(base + "/review/undo", {})
    assert out == dict(voxels=3, undo_depth=1) and diff(files) == d1
    out = post(base + "/review/undo", {})
    assert out == dict(voxels=3, undo_depth=0) and diff(files) == d0
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base + "/review/undo", {})
    assert e.value.code == 400 and "undo" in json.loads(e.value.read())["error"]
    assert diff(files) == d0


def test_an_undo_after_another_writer_changed_the_diff_is_refused(
    files: dict[str, Path],
) -> None:
    rv = make(files)
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    rv.paint(paint_strokes((0, [(5, 4, 4)])))
    # another writer (a second viewer process) changes the diff
    other = LE.read(files["edits"])
    LE.write(
        files["edits"],
        LE._set(other, np.array([np.ravel_multi_index((6, 6, 6), SHAPE)]), [42]),
        hemisphere="R",
        shape=SHAPE,
        affine=A,
    )
    d = diff(files)
    with pytest.raises(ValueError, match="another writer"):
        rv.undo_last()
    assert diff(files) == d and len(rv.undo) == 2
    # a new write of this run is on top of the other writer's: it can be undone
    rv.paint(paint_strokes((0, [(7, 4, 4)])))
    assert rv.undo_last() == dict(voxels=1, undo_depth=2) and diff(files) == d
    # the write below it was before the other writer's: still refused
    with pytest.raises(ValueError, match="another writer"):
        rv.undo_last()
    assert diff(files) == d


def test_a_failed_log_append_leaves_the_diff_and_the_undo_stack(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    rv = make(files)
    rv.start(0, None)
    rv.next()
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    before = files["edits"].read_bytes(), len(rv.undo), diff(files)
    n_log = len(records(files))

    def full(*a: Any, **k: Any) -> str:
        raise OSError("no space left on the device")

    monkeypatch.setattr(rv.log, "answer", full)
    with pytest.raises(OSError, match="no space"):
        rv.answer("5,5,5", "thicken", {}, 1.0)
    assert (files["edits"].read_bytes(), len(rv.undo), diff(files)) == before
    assert len(records(files)) == n_log
    # the undo stack still undoes the paint
    rv.undo_last()
    assert diff(files) == {} and rv.undo == []


def test_a_refused_log_append_holds_the_lock_against_another_writer(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log append is under the diff's lock: a second writer (another
    viewer process) cannot write between the answer's write and its log
    append, so the roll-back is not refused and the error is the log's."""
    rv = make(files)
    rv.start(0, None)
    rv.next()
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    before = files["edits"].read_bytes(), len(rv.undo), diff(files)
    n_log = len(records(files))
    started, done = threading.Event(), threading.Event()
    other_voxel = (6, 6, 6)

    def other() -> None:
        started.set()
        i = np.array([np.ravel_multi_index(other_voxel, SHAPE)], np.int64)
        LE.update(
            files["edits"],
            lambda c: LE._set(c.edits, i, [42]),
            hemisphere="R",
            shape=SHAPE,
            affine=A,
        )
        done.set()

    t = threading.Thread(target=other)

    def full(*a: Any, **k: Any) -> str:
        t.start()
        assert started.wait(10)
        assert not done.wait(0.5)  # the other writer waits for the lock
        raise OSError("no space left on the device")

    monkeypatch.setattr(rv.log, "answer", full)
    with pytest.raises(OSError, match="no space"):
        rv.answer("5,5,5", "thicken", {}, 1.0)
    t.join(10)
    assert done.is_set()
    assert len(rv.undo) == before[1] and len(records(files)) == n_log
    # the other writer wrote on the diff as it was before the answer
    assert diff(files) == {**before[2], other_voxel: 42}


def test_a_write_reads_only_the_sidecar_when_its_copy_is_current(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    rv = make(files)
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    reads: list[Path] = []
    real = LE._read_records

    def counted(q: Path) -> bytes:
        reads.append(q)
        return real(q)

    monkeypatch.setattr(LE, "_read_records", counted)
    rv.paint(paint_strokes((0, [(5, 4, 4)])))
    rv.undo_last()
    assert reads == []
    assert diff(files) == {(4, 4, 4): 0}  # diff() reads: not a write
    # another writer: the next write reads the records again
    reads.clear()
    LE.write(files["edits"], LE.empty(), hemisphere="R", shape=SHAPE, affine=A)
    rv.paint(paint_strokes((0, [(6, 4, 4)])))
    assert reads == [files["edits"]] and diff(files) == {(6, 4, 4): 0}


def test_undo_of_a_first_paint_leaves_an_empty_diff(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/paint", dict(strokes=paint_strokes((42, [(4, 4, 4)]))))
    post(base + "/review/undo", {})
    assert diff(files) == {}


def test_manual_paint_at_a_site_writes_the_diff_and_the_log(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    # no strokes and no pending edit: nothing to change
    p = post(base + "/review/preview", dict(site="5,5,5", choice=VR.MANUAL, strokes=[]))
    assert p["status"] == "done" and p["answer"] == [] and "command" not in p
    with pytest.raises(urllib.error.HTTPError):  # a manual paint needs strokes
        post(
            base + "/review/answer",
            dict(site="5,5,5", choice=VR.MANUAL, reviewer_seconds=1),
        )
    post(base + "/review/paint", dict(strokes=paint_strokes((0, [(5, 6, 5)]))))
    out = post(
        base + "/review/answer",
        dict(
            site="5,5,5",
            choice=VR.MANUAL,
            strokes=paint_strokes((42, [(5, 5, 5), (4, 4, 4)]), ("input", [(5, 6, 5)])),
            reviewer_seconds=7.5,
        ),
    )
    assert out["edited"] == 3 and out["undo_depth"] == 2
    assert diff(files) == {(5, 5, 5): 42, (4, 4, 4): 42}
    a = records(files)[-1]
    assert a["Record"] == "answer" and a["Choice"] == VR.MANUAL
    assert a["Id"] == out["answer"] and a["ReviewerSeconds"] == 7.5
    v, lab, _ = E.read_log(files["log"]).voxels(a["Id"])
    # the eraser's voxel is recorded with the label it takes: its input label
    got = sorted(zip(map(tuple, v.tolist()), lab.tolist(), strict=True))
    assert got == [((4, 4, 4), 42), ((5, 5, 5), 42), ((5, 6, 5), 41)]
    post(base + "/review/undo", {})
    assert diff(files) == {(5, 6, 5): 0}


def test_strokes_are_refused_with_another_choice(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    with pytest.raises(urllib.error.HTTPError):
        post(
            base + "/review/answer",
            dict(
                site="5,5,5",
                choice="thicken",
                strokes=paint_strokes((42, [(4, 4, 4)])),
                reviewer_seconds=1,
            ),
        )
    assert not files["edits"].exists()


# ------------------------------------------------------- paint only


def paint_only(files: dict[str, Path]) -> VR.Review:
    lab = np.asarray(nib.nifti1.load(files["dseg"]).dataobj)
    write_dseg_sidecar(files["dseg"], files["input"])
    return VR.Review(None, None, files["edits"], files["dseg"], lambda: lab, A, None)


def test_the_review_runs_with_the_edits_alone(files: dict[str, Path]) -> None:
    rv = paint_only(files)
    q = rv.queue()
    assert q["paint_only"] is True and q["sites"] == [] and q["session"] is None
    assert q["log"] is None and q["edits"] == str(files["edits"])
    assert [m["value"] for m in q["menu"]] == [0, 41, 42, "input"]
    out = rv.post("/review/paint", dict(strokes=paint_strokes((0, [(4, 4, 4)]))))
    assert out == dict(voxels=1, undo_depth=1) and diff(files) == {(4, 4, 4): 0}
    assert rv.post("/review/undo", {}) == dict(voxels=1, undo_depth=0)
    for path, req in (
        ("/review/session", dict(random_share=0)),
        ("/review/scribble", dict(strokes=[], reviewer_seconds=1)),
        ("/review/answer", dict(site="5,5,5", choice="leave", reviewer_seconds=1)),
    ):
        with pytest.raises(ValueError, match="paint only"):
            rv.post(path, req)
    assert not files["log"].exists()


def test_the_menu_of_the_left_hemisphere() -> None:
    assert [m["value"] for m in VR.menu("L")] == [0, 2, 3, "input"]
    assert [m["key"] for m in VR.menu("R")] == ["outside", "wm", "cortex", "input"]


def test_undo_reverses_the_last_write_also_a_queue_answer(server: Any) -> None:
    """Every write into the diff goes on one undo stack: an undo after a queue
    answer takes back the answer, not the paint under it."""
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    # the paint sets (5, 5, 5) to 0; thicken at 5,5,5 then sets it to 42
    post(base + "/review/paint", dict(strokes=paint_strokes((0, [(5, 5, 5)]))))
    after_paint = diff(files)
    out = post(
        base + "/review/answer",
        dict(site="5,5,5", choice="thicken", params={}, reviewer_seconds=1.0),
    )
    assert out["undo_depth"] == 2
    assert diff(files) == {(5, 5, 5): 42}
    assert post(base + "/review/undo", {}) == dict(voxels=1, undo_depth=1)
    assert diff(files) == after_paint
    # the log keeps the undone answer
    assert records(files)[-1]["Id"] == out["answer"]
    assert post(base + "/review/undo", {}) == dict(voxels=1, undo_depth=0)
    assert diff(files) == {}


def test_accept_fix_goes_on_the_undo_stack(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    post(
        base + "/review/answer",
        dict(site="5,5,5", choice=VR.ACCEPT, reviewer_seconds=2.0),
    )
    assert rv.undo and diff(files) == {v: 0 for v in FIX_VOXELS}
    post(base + "/review/undo", {})
    assert diff(files) == {}


# ------------------------------------------------- reads of the input labels


def test_the_input_labels_are_opened_once_for_a_review(
    server: Any, monkeypatch: Any
) -> None:
    base, rv, files = server
    opened: list[Path] = []
    real = blocks.open_volume

    def counting(path: Path, *a: Any, **k: Any) -> Any:
        opened.append(Path(path))
        return real(path, *a, **k)

    def no_load(*a: Any, **k: Any) -> Any:
        raise AssertionError("the input labels are read again with nib.load")

    monkeypatch.setattr(blocks, "open_volume", counting)
    post(base + "/review/paint", dict(strokes=paint_strokes((0, [(4, 4, 4)]))))
    # after the first open, nothing loads the input labels with nibabel again
    monkeypatch.setattr(nib.nifti1, "load", no_load)
    monkeypatch.setattr(nib, "load", no_load)
    post(base + "/review/paint", dict(strokes=paint_strokes((42, [(5, 4, 4)]))))
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    out = post(
        base + "/review/answer",
        dict(
            site="5,5,5",
            choice=VR.MANUAL,
            strokes=paint_strokes((42, [(5, 5, 5)]), ("input", [(4, 4, 4)])),
            reviewer_seconds=1,
        ),
    )
    assert out["edited"] == 2
    assert opened == [files["input"]]
    # the eraser's voxel is logged with its input label, read from the same map
    a = records(files)[-1]
    v, lab, _ = E.read_log(files["log"]).voxels(a["Id"])
    got = sorted(zip(map(tuple, v.tolist()), lab.tolist(), strict=True))
    assert got == [((4, 4, 4), 41), ((5, 5, 5), 42)]


def test_far_apart_strokes_read_only_their_voxels(tmp_path: Path) -> None:
    import tracemalloc

    # a grid whose box of two strokes on its first and last slices is all of it
    shape = (160, 160, 120)
    lab = np.zeros(shape, np.int16)
    lab[20:140, 20:140, 10:110] = 41
    dseg = tmp_path / "sub-T02_ses-S1_desc-wmtopo_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(lab, A), dseg)
    inp = tmp_path / "sub-T02_ses-S1_desc-input_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(lab, A), inp)
    write_dseg_sidecar(dseg, inp)
    edits = LE.bids_path(tmp_path / "deriv", dict(sub="T02", ses="S1"), "R")
    rv = VR.Review(None, None, edits, dseg, lambda: lab, A, None)
    rv.paint(paint_strokes((42, [(80, 80, 60)])))  # opens the input labels
    a = [(i, 0, 0) for i in range(shape[0])]
    b = [(i, shape[1] - 1, shape[2] - 1) for i in range(shape[0])]
    box_bytes = lab.nbytes  # the box of the two strokes is the whole grid
    tracemalloc.start()
    try:
        out = rv.paint(paint_strokes((42, a), (42, b)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert out["voxels"] == 2 * shape[0]
    # the read is of the order of the 320 voxels, not of their box (6.1 MB)
    assert peak < box_bytes // 20, (peak, box_bytes)
    e = LE.read(edits)
    got = set(map(tuple, np.stack(np.unravel_index(e.index, shape), 1).tolist()))
    assert got == {(80, 80, 60), *a, *b} and set(e.label.tolist()) == {42}


# ------------------------------------------------------- outputs replaced


def replace_on_disk(p: Path) -> None:
    """p replaced as regenerate replaces it: a new file with the same bytes,
    renamed over it (a new inode)."""
    tmp = p.with_name(p.name + ".new")
    tmp.write_bytes(p.read_bytes())
    os.replace(tmp, p)


@pytest.mark.parametrize("which", ["dseg", "queue", "fixdiff", "copy"])
def test_writes_are_refused_after_the_outputs_are_replaced(
    files: dict[str, Path], which: str
) -> None:
    rv = make(files)
    rv.start(0, None)
    rv.next()
    rv.paint(paint_strokes((0, [(4, 4, 4)])))
    replace_on_disk(files[which])
    before = files["edits"].read_bytes(), diff(files), len(rv.undo)
    n_log = len(records(files))
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.paint(paint_strokes((0, [(5, 4, 4)])))
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.answer("5,5,5", "thicken", {}, 1.0)
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.answer("5,5,5", VR.ACCEPT, {}, 1.0)
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.undo_last()
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.pending()
    assert (files["edits"].read_bytes(), diff(files), len(rv.undo)) == before
    assert len(records(files)) == n_log


def test_a_fix_diff_that_appears_after_the_start_refuses_writes(
    files: dict[str, Path],
) -> None:
    keep = files["fixdiff"].read_bytes()
    files["fixdiff"].unlink()
    rv = make(files)
    files["fixdiff"].write_bytes(keep)
    with pytest.raises(ValueError, match="restart the viewer"):
        rv.paint(paint_strokes((0, [(4, 4, 4)])))
    assert not LE.edits_paths(files["edits"])["sidecar"].exists()


# ------------------------------------------------------- preview jobs


class SlowThicken(Fake):
    """A Fake whose thicken preview takes 60 s; the other choices are quick."""

    def preview(
        self,
        labels: Any,
        lo: Any,
        site: Any,
        choice: str,
        params: dict[str, Any],
        change: VR.Answer | None = None,
    ) -> VR.Preview:
        if choice == "thicken":
            time.sleep(60)
        return super().preview(labels, lo, site, choice, params, change)


class Broken(Fake):
    """A Fake whose preview fails."""

    def preview(self, *a: Any, **k: Any) -> VR.Preview:
        raise RuntimeError("no WM in the box")


class Dies(Fake):
    """A Fake whose preview ends its process without a result (as the memory
    cap's kill does)."""

    def preview(self, *a: Any, **k: Any) -> VR.Preview:
        os._exit(9)


def poll(rv: VR.Review, out: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
    t = time.monotonic() + timeout
    while out["status"] == "running":
        assert time.monotonic() < t, "the preview did not end"
        time.sleep(0.05)
        out = rv.job(out["job"])
    return out


@pytest.fixture
def review(files: dict[str, Path]) -> Iterator[Any]:
    """make(files, choices), started at the first site; closed at the end."""
    made: list[VR.Review] = []

    def mk(choices: Fake) -> VR.Review:
        rv = make(files, choices)
        made.append(rv)
        rv.start(0, None)
        rv.next()
        return rv

    yield mk
    for rv in made:
        rv.close()


def test_a_newer_preview_stops_the_running_one(
    review: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(VR, "WAIT_S", 0.05)
    rv = review(SlowThicken())
    first = rv.preview("5,5,5", "thicken", {})
    assert first["status"] == "running"
    t = time.monotonic() + 60
    while rv._worker is None:  # the job starts its worker
        assert time.monotonic() < t
        time.sleep(0.02)
    old = rv._worker.proc
    second = poll(rv, rv.preview("5,5,5", "fill-gap", {}))
    assert not old.is_alive()
    assert second["status"] == "done" and second["overrides"] is True
    assert second["answer"] == [[5, 5, 5], [6, 5, 5]]
    # the stopped preview: failed, or no longer a job
    try:
        out = rv.job(first["job"])
    except ValueError as exc:
        assert "no preview job" in str(exc)
    else:
        assert out["status"] == "failed" and "newer preview" in out["error"]


def test_the_running_preview_again_starts_no_second_worker(
    review: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(VR, "WAIT_S", 0.05)
    rv = review(SlowThicken())
    out = rv.preview("5,5,5", "thicken", {})
    t = time.monotonic() + 60
    while rv._worker is None:
        assert time.monotonic() < t
        time.sleep(0.02)
    proc = rv._worker.proc
    assert rv.preview("5,5,5", "thicken", {}) == out
    assert out["status"] == "running" and rv._worker.proc is proc
    assert rv.job(out["job"]) == out
    assert [p for p in multiprocessing.active_children() if p.name == "preview"] == [
        proc
    ]


def test_a_worker_that_ends_without_a_result_fails_its_job(review: Any) -> None:
    rv = review(Dies())
    out = poll(rv, rv.preview("5,5,5", "thicken", {}))
    assert out["status"] == "failed" and "without a result" in out["error"]
    # the next preview starts a new worker
    rv.choices = Fake()
    out = poll(rv, rv.preview("5,5,5", "fill-gap", {}))
    assert out["status"] == "done"


def test_a_finished_preview_stays_available_for_polling(
    review: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(VR, "WAIT_S", 0.0)
    monkeypatch.setattr(VR, "PREVIEWS_KEPT", 2)
    rv = review(Fake())
    first = rv.preview("5,5,5", "thicken", {})
    done = poll(rv, first)
    # read more than once, and after a later preview finished
    assert done["status"] == "done" and rv.job(first["job"]) == done
    poll(rv, rv.preview("5,5,5", "fill-gap", {}))
    assert rv.job(first["job"]) == done
    # bounded: PREVIEWS_KEPT finished previews are kept, the oldest goes
    assert poll(rv, rv.preview("5,5,5", "remove-strand", {}))["status"] == "done"
    with pytest.raises(ValueError, match="no preview job"):
        rv.job(first["job"])


def test_a_failed_preview_is_reported_to_the_poll(review: Any) -> None:
    rv = review(Broken())
    out = poll(rv, rv.preview("5,5,5", "thicken", {}))
    assert out["status"] == "failed" and "no WM in the box" in out["error"]


# ------------------------------------------------------- pending edits


def pending(out: dict[str, Any]) -> dict[tuple[int, ...], int]:
    """A /review/pending response as {voxel: label}, checked to be in
    linear-index order."""
    v = [tuple(x) for x in out["voxels"]]
    assert v == sorted(v) and len(out["labels"]) == len(v)
    return dict(zip(v, out["labels"], strict=True))


def test_pending_gives_the_edits_not_in_the_dseg_with_their_labels(
    files: dict[str, Path],
) -> None:
    """The voxels where the edits and the dseg's copy differ: a new entry, a
    changed entry and an erased entry (its input label, not its dseg label);
    after an undo of all paints there are none."""
    # the copy that the dseg used; the edits start equal to it
    used = {(5, 4, 4): 0, (6, 5, 5): 42, (9, 9, 9): 42}
    write_edits(files["copy"], edits(used, SHAPE), SHAPE, A)
    write_edits(files["edits"], edits(used, SHAPE), SHAPE, A)
    rv = make(files)
    assert pending(rv.get("/review/pending", {})) == {}
    rv.paint(paint_strokes((42, [(7, 7, 7)])))  # new: dseg 41
    rv.paint(paint_strokes((41, [(5, 4, 4)])))  # changed: 0 in the copy
    rv.paint(paint_strokes(("input", [(6, 5, 5)])))  # erased: input 0, dseg 41
    want = {(5, 4, 4): 41, (6, 5, 5): 0, (7, 7, 7): 42}
    assert pending(rv.get("/review/pending", {})) == want
    # the same as label_edits.differ of the two diffs, the eraser's voxel
    # with its input label
    idx, lab = LE.differ(LE.read(files["edits"]), LE.read(files["copy"]))
    v = np.stack(np.unravel_index(idx, SHAPE), 1)
    inp = np.asarray(nib.nifti1.load(files["input"]).dataobj)
    lab = np.where(lab == LE.INPUT, inp[tuple(v.T)], lab)
    assert dict(zip(map(tuple, v.tolist()), lab.tolist(), strict=True)) == want
    for _ in range(3):
        rv.undo_last()
    assert pending(rv.get("/review/pending", {})) == {}
    rv.close()


def test_pending_through_the_route(server: Any) -> None:
    base, rv, files = server
    assert get(base + "/review/pending") == dict(voxels=[], labels=[])
    post(
        base + "/review/paint",
        dict(strokes=paint_strokes((0, [(4, 5, 4), (4, 4, 4)]))),
    )
    out = get(base + "/review/pending")
    assert out == dict(voxels=[[4, 4, 4], [4, 5, 4]], labels=[0, 0])
    post(base + "/review/undo", {})
    assert get(base + "/review/pending") == dict(voxels=[], labels=[])


def test_pending_without_the_dseg_copy_is_refused(files: dict[str, Path]) -> None:
    for p in LE.edits_paths(files["copy"]).values():
        p.unlink()
    rv = make(files)
    with pytest.raises(ValueError, match="no copy of the edits"):
        rv.pending()
    rv.close()


# ------------------------------------------------------- preview of strokes

#: strokes of WM on the top face (z 12) of the phantom's WM block: an arch
#: whose legs stand on the face, so it closes a loop (a handle) with the face
ARCH = [(5, 8, 13), (5, 8, 14), *[(x, 8, 14) for x in range(6, 10)], (10, 8, 14)]
ARCH = [*ARCH, (10, 8, 13)]
#: a flat sheet of WM on the same face: no new loop
SHEET = [(x, y, 13) for x in range(5, 11) for y in range(5, 11)]


def real_review(files: dict[str, Path]) -> tuple[VR.Review, Any]:
    """A review with the real choices (WmChoices), a session and its first
    site (5,5,5) shown; and the dseg's labels."""
    lab = np.asarray(nib.nifti1.load(files["dseg"]).dataobj).astype(np.int64)
    rv = VR.Review(
        files["queue"],
        files["log"],
        files["edits"],
        files["dseg"],
        lambda: lab,
        A,
        randoms,
        inputs=files["input"],
    )
    rv.start(0, None)
    rv.next("5,5,5")
    return rv, lab


def change_of(vox: dict[tuple[int, int, int], int]) -> Any:
    from baillarger import wm_choices as wc

    v = np.array(sorted(vox), np.int64).reshape(-1, 3)
    return wc.Change(
        v, np.array([vox[(x[0], x[1], x[2])] for x in v.tolist()], np.int64)
    )


def test_a_preview_of_strokes_equals_the_preview_of_the_same_change(
    files: dict[str, Path],
) -> None:
    """Strokes that close a loop: the correction changes voxels round them.
    The preview of the strokes, with a pending edit near them, equals
    wm_choices.preview of the same Change (the pending edit and the
    strokes, in linear-index order)."""
    from baillarger import wm_choices as wc

    rv, lab = real_review(files)
    rv.paint(paint_strokes((0, [(7, 8, 12)])))  # pending: under the arch
    out = poll(rv, rv.preview("5,5,5", VR.MANUAL, {}, paint_strokes((41, ARCH))))
    assert out["status"] == "done", out
    want = {v: 41 for v in ARCH} | {(7, 8, 12): 0}
    ref = wc.preview(lab, (5, 5, 5), "R", change_of(want), voxel_mm=[0.16] * 3)
    assert out["answer"] == ref.answer.voxels.tolist()
    assert sorted(map(tuple, out["answer"])) == sorted(want)
    assert out["correction"] == ref.correction.voxels.tolist()
    assert out["overrides"] == ref.overrides
    assert out["cut_by_box"] == ref.cut_by_box
    # the correction is round the arch: inside its box, one voxel larger
    c = np.array(out["correction"]).reshape(-1, 3)
    assert len(c)
    a = np.array(ARCH)
    assert np.all((c >= a.min(0) - 1) & (c <= a.max(0) + 1))
    # the preview wrote nothing
    assert pending(rv.pending()) == {(7, 8, 12): 0}
    rv.close()


def test_a_preview_of_strokes_on_a_flat_sheet_gives_no_correction(
    files: dict[str, Path],
) -> None:
    rv, _ = real_review(files)
    out = poll(rv, rv.preview("5,5,5", VR.MANUAL, {}, paint_strokes((41, SHEET))))
    assert out["status"] == "done", out
    assert len(out["answer"]) == len(SHEET)
    assert out["correction"] == [] and out["overrides"] is False
    rv.close()


def test_the_change_of_a_strokes_preview(server: Any) -> None:
    """The change: the pending edits after a paint of the strokes (the later
    stroke wins; the eraser gives the input label), near the site and the
    strokes only, less the voxels that keep their dseg label."""
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    post(
        base + "/review/paint",
        dict(strokes=paint_strokes((0, [(4, 4, 4), (6, 6, 6), (15, 14, 12)]))),
    )
    strokes = paint_strokes(
        (42, [(4, 5, 4), (6, 6, 6)]),  # (6, 6, 6) pending 0, now 42
        ("input", [(4, 4, 4)]),  # erased: its input label 41 is the dseg's
        (41, [(5, 6, 5)]),  # the dseg's label: no change
    )
    out = poll(
        rv,
        post(
            base + "/review/preview",
            dict(site="5,5,5", choice=VR.MANUAL, strokes=strokes),
        ),
    )
    assert out["status"] == "done", out
    # (15, 14, 12) is pending but far (Fake.half: 4) from the site and strokes
    assert out["answer"] == [[4, 5, 4], [6, 6, 6]]
    # the preview wrote nothing
    assert diff(files) == {(4, 4, 4): 0, (6, 6, 6): 0, (15, 14, 12): 0}
    # the same request again: the same job
    again = post(
        base + "/review/preview",
        dict(site="5,5,5", choice=VR.MANUAL, strokes=strokes),
    )
    assert again["answer"] == out["answer"]
    # strokes go with manual-paint only, and manual-paint needs them
    for body in (
        dict(site="5,5,5", choice="thicken", strokes=strokes),
        dict(site="5,5,5", choice=VR.MANUAL),
    ):
        with pytest.raises(urllib.error.HTTPError):
            post(base + "/review/preview", body)
