"""Tests of baillarger.viewer.jobs (the job pattern), with exact answers: the
tests of the pattern that tests/test_viewer_filters.py has for FilterJobs, on a
job that reads no data files."""

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from baillarger.viewer.jobs import LOG_LINES, Jobs


def quiet(m: str) -> None:
    pass


def save(r: dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r))


def load(p: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(p.read_text())
    return out


def jobs(cache: Path | None) -> Jobs[dict[str, Any]]:
    if cache is None:
        return Jobs(log=quiet)
    return Jobs(cache, "job_synth", ".json", save=save, load=load, log=quiet)


def test_compute_once_then_load_from_the_cache(tmp_path: Path) -> None:
    calls: list[int] = []

    def work(log: Any) -> dict[str, Any]:
        log("working")
        calls.append(1)
        return dict(x=7)

    j = jobs(tmp_path / "cache")
    out = j.run("k1", "seven", work, wait=True)
    assert out == dict(key="k1", label="seven", status="done")
    assert j.status()["status"] == "done" and j.status()["log"] == ["working"]
    assert j.files() == [tmp_path / "cache" / "job_synth_k1.json"]
    assert j.get("k1") == dict(x=7) and calls == [1]

    def fail(log: Any) -> dict[str, Any]:
        raise AssertionError("recomputed")

    again = jobs(tmp_path / "cache")
    assert again.run("k1", "seven", fail)["status"] == "done"
    assert again.get("k1") == dict(x=7)
    with pytest.raises(KeyError):
        again.get("k2")


def test_memory_only_jobs_keep_the_last_results() -> None:
    j = jobs(None)
    for k in range(5):

        def work(log: Any, k: int = k) -> dict[str, Any]:
            return dict(x=k)

        j.run(f"k{k}", "", work, wait=True)
    assert j.files() == [] and j.path("k0") is None
    assert list(j.results) == ["k2", "k3", "k4"]
    assert j.get("k4") == dict(x=4)
    with pytest.raises(KeyError):
        j.get("k0")
    with pytest.raises(ValueError, match="save and a load"):
        Jobs(Path("."), save=save)


def test_one_job_at_a_time() -> None:
    gate = threading.Event()

    def slow(log: Any) -> dict[str, Any]:
        gate.wait(5)
        return dict(x=1)

    j = jobs(None)
    assert j.run("a", "slow", slow)["status"] == "running"
    with pytest.raises(RuntimeError, match="one at a time"):
        j.run("b", "other", slow)
    gate.set()
    assert j.thread is not None
    j.thread.join(10)
    assert j.status()["status"] == "done"
    assert j.run("b", "other", slow, wait=True)["status"] == "done"


def test_an_error_is_reported_and_the_log_is_bounded() -> None:
    def bad(log: Any) -> dict[str, Any]:
        for k in range(LOG_LINES + 5):
            log(str(k))
        raise ValueError("no data")

    j = jobs(None)
    out = j.run("e", "bad", bad, wait=True)
    assert out["status"] == "error" and out["error"] == "ValueError: no data"
    st = j.status()
    assert st["status"] == "error" and st["error"] == "ValueError: no data"
    assert st["log"] == [str(k) for k in range(5, LOG_LINES + 5)]
    assert not j.has("e")


def test_a_failed_job_is_reported_once_then_runs_again() -> None:
    """The request that finds a failed job gets its error and does not start it
    again; the error is then taken, so the next request with that key runs the
    job again (a failure can be transient)."""
    calls: list[int] = []

    def flaky(log: Any) -> dict[str, Any]:
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("transient")
        return dict(x=1)

    j = jobs(None)
    gate = threading.Event()

    def held(log: Any) -> dict[str, Any]:
        gate.wait(5)
        return flaky(log)

    out = j.run("f", "flaky", held)
    assert out["status"] == "running"
    gate.set()
    # the request waits for its job: the job fails, the error is reported
    assert j.wait("f", 10) == dict(
        key="f", label="flaky", status="error", error="ValueError: transient"
    )
    assert calls == [1]
    # the state keeps the error for the page, but it was reported: the next
    # request runs the job again
    assert j.status()["status"] == "error"
    assert j.run("f", "flaky", flaky, wait=True)["status"] == "done"
    assert calls == [1, 1] and j.get("f") == dict(x=1)


def test_the_next_request_gets_the_error_and_does_not_rerun() -> None:
    """A job that fails after its request stopped waiting (202): the next
    request with the same key gets the error, and the work does not run again
    in that request; the one after it runs the job again."""
    calls: list[int] = []

    def bad(log: Any) -> dict[str, Any]:
        calls.append(1)
        raise ValueError("over the cap")

    j = jobs(None)
    gate = threading.Event()

    def held(log: Any) -> dict[str, Any]:
        gate.wait(5)
        return bad(log)

    assert j.run("e", "bad", held)["status"] == "running"
    assert j.wait("e", 0)["status"] == "running"
    gate.set()
    assert j.thread is not None
    j.thread.join(10)
    out = j.run("e", "bad", bad)
    assert out == dict(
        key="e", label="bad", status="error", error="ValueError: over the cap"
    )
    assert calls == [1]
    # an error of another key is not this key's: it runs
    assert j.run("other", "bad", bad, wait=True)["status"] == "error"
    assert calls == [1, 1]
    # retry: the error of "e" was replaced and is not reported again
    assert j.run("e", "bad", bad, wait=True)["status"] == "error"
    assert calls == [1, 1, 1]
