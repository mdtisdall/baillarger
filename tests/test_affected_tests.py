"""Tests of scripts/affected-tests: a stale map is an error, and a change that
reaches no test gives a warning. Each test makes a small git repository."""

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "affected-tests"


def load_script() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("affected_tests", str(SCRIPT))
    spec = importlib.util.spec_from_loader("affected_tests", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def make_repo(tmp_path: Path, mapping: str) -> Path:
    """A repository with a package, a test, a page script, and this map."""
    repo = tmp_path / "repo"
    files = {
        "pyproject.toml": f"[tool.affected-tests.map]\n{mapping}\n",
        "src/pkg/__init__.py": "",
        "src/pkg/mod.py": "X = 1\n",
        "src/pkg/lonely.py": "Y = 2\n",
        "src/pkg/static/page.js": "// page\n",
        "tests/test_mod.py": (
            "from pkg import mod\n\n\ndef test_x():\n    assert mod.X\n"
        ),
        "tests/test_page.py": "def test_page():\n    pass\n",
    }
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def run(
    repo: Path, *args: str, actions: bool = False
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"}
    if actions:
        env["GITHUB_ACTIONS"] = "true"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base", "HEAD", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def touch(repo: Path, name: str) -> None:
    with open(repo / name, "a") as f:
        f.write("# changed\n" if name.endswith(".py") else "// changed\n")


def test_a_map_entry_selects_its_tests(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, '"src/pkg/static/*" = "tests/test_page.py"')
    touch(repo, "src/pkg/static/page.js")
    r = run(repo)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["tests/test_page.py"]


@pytest.mark.parametrize("actions", [False, True])
def test_a_map_target_that_does_not_exist_is_an_error(
    tmp_path: Path, actions: bool
) -> None:
    repo = make_repo(tmp_path, '"src/pkg/static/*" = "tests/test_gone.py"')
    touch(repo, "src/pkg/static/page.js")
    for args in ([], ["--run", "--", sys.executable, "-c", "pass"]):
        r = run(repo, *args, actions=actions)
        assert r.returncode == 2
        assert "tests/test_gone.py, which does not exist" in r.stderr
        assert ("::error title=affected-tests::" in r.stderr) == actions


def test_a_stale_map_entry_is_an_error_when_no_mapped_file_changed(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path, '"src/pkg/static/*" = "tests/test_gone.py"')
    touch(repo, "src/pkg/mod.py")
    r = run(repo)
    assert r.returncode == 2
    assert "tests/test_gone.py, which does not exist" in r.stderr


def test_a_map_pattern_that_matches_no_file_is_an_error(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, '"src/pkg/old/*.js" = "tests/test_page.py"')
    touch(repo, "src/pkg/mod.py")
    r = run(repo)
    assert r.returncode == 2
    assert "'src/pkg/old/*.js' matches no file" in r.stderr


def test_code_that_reaches_no_test_gives_a_warning(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, '"src/pkg/static/*" = "tests/test_page.py"')
    touch(repo, "src/pkg/lonely.py")
    r = run(repo)
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""
    assert "WARNING: code changed, but no test reaches it" in r.stderr
    assert "src/pkg/lonely.py" in r.stderr


def test_code_that_reaches_a_test_gives_no_warning(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, '"src/pkg/static/*" = "tests/test_page.py"')
    touch(repo, "src/pkg/mod.py")
    r = run(repo)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["tests/test_mod.py"]
    assert "WARNING" not in r.stderr


def test_this_projects_map_is_not_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    at = load_script()
    monkeypatch.chdir(ROOT)
    settings, _ = at.load_settings()
    files = at.git("ls-files", "-z").split("\0")
    at.check_map(settings.get("map", {}), [p for p in files if p])
