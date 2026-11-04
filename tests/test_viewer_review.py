"""Tests of baillarger.viewer.review through its routes on a real server: a
phantom queue, a temporary log, phase 1.3's log writer and reader, and stand-in
choices (the routes do not depend on what a choice does)."""

import json
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

from baillarger import bids
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


def row(rank: int, kind: str, c: tuple[int, int, int]) -> dict[str, Any]:
    r: dict[str, Any] = {k: 0 for k in WS.COLUMNS}
    r.update(rank=rank, kind=kind, i=c[0], j=c[1], k=c[2], tissue="grey matter")
    r.update(wm_thickness=1.0, gap_width=0.0)
    return r


class Fake:
    """Stand-in choices: the answer sets the site to 42 and its +x neighbour to
    41; the correction then changes the voxel +y of the site; fill-gap
    overrides. delay: seconds a preview takes."""

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
        self, labels: Any, lo: Any, site: Any, choice: str, params: dict[str, Any]
    ) -> VR.Preview:
        time.sleep(self.delay)
        a = self.answer(labels, lo, site, choice, params)
        c = np.array([(site[0], site[1] + 1, site[2])], np.int64)
        return VR.Preview(a.voxels, c, choice == "fill-gap", None)


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
    return dict(dseg=dseg, queue=q, log=log["log"])


def randoms(n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(RANDOM))[:n]
    return [row(0, "random", RANDOM[i]) for i in pick]


def make(files: dict[str, Path], choices: Fake | None = None) -> VR.Review:
    lab = np.asarray(nib.nifti1.load(files["dseg"]).dataobj)
    return VR.Review(
        files["queue"],
        files["log"],
        files["dseg"],
        lambda: lab,
        A,
        randoms,
        choices or Fake(),
        echoes=[files["dseg"]],
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


def test_a_slow_preview_runs_as_a_job(files: dict[str, Path], monkeypatch: Any) -> None:
    monkeypatch.setattr(VR, "WAIT_S", 0.05)
    rv = make(files, Fake(delay=0.3))
    rv.start(0, None)
    rv.next()
    out = rv.preview("5,5,5", "thicken", dict(radius=1))
    assert out["status"] == "running"
    while out["status"] == "running":
        time.sleep(0.05)
        out = rv.job(out["job"])
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


def test_manual_paint_gives_the_commands_and_no_answer(server: Any) -> None:
    base, rv, files = server
    post(base + "/review/session", dict(random_share=0, seed=None))
    get(base + "/review/next")
    out = post(base + "/review/preview", dict(site="5,5,5", choice="manual-paint"))
    assert out["status"] == "done" and out["answer"] == [] and out["correction"] == []
    cmd = out["command"].splitlines()
    assert cmd[0].startswith(f"python -m baillarger.wm_paint export {files['dseg']}")
    assert "--site 5 5 5" in cmd[0] and f"--echo {files['dseg']}" in cmd[0]
    assert cmd[1].startswith("itksnap -g ")
    assert f"--log {files['log']} --session {rv.session}" in cmd[2]
    with pytest.raises(urllib.error.HTTPError) as e:
        post(
            base + "/review/answer",
            dict(site="5,5,5", choice="manual-paint", reviewer_seconds=1),
        )
    assert e.value.code == 400


def test_the_real_choices_through_the_route(files: dict[str, Path]) -> None:
    from baillarger import wm_choices as wc

    lab = np.asarray(nib.nifti1.load(files["dseg"]).dataobj).astype(np.int64)
    rv = VR.Review(files["queue"], files["log"], files["dseg"], lambda: lab, A, randoms)
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
