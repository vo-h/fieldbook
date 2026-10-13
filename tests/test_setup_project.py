"""`fieldbook init` (setup_project.py): project files, the author, and the IAM role prompt."""

import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from helpers import runner

from fieldbook import config, setup_project
from fieldbook.cli import app
from fieldbook.clouds import aws
from fieldbook.config import settings


def test_init_offers_roles_only_for_cloud_runs(mock_aws, monkeypatch: pytest.MonkeyPatch) -> None:
    from fieldbook import setup_project

    checked = []
    monkeypatch.setattr(aws, "role_state", lambda role: checked.append(role) or "ok")
    monkeypatch.delenv("FIELDBOOK_IMAGE_REGISTRY")  # an S3 bucket, local runs only
    setup_project.iam_roles(ask=lambda question: True, say=print)
    assert checked == []
    monkeypatch.setenv("FIELDBOOK_IMAGE_REGISTRY", "1.dkr.ecr.us-east-1.amazonaws.com/fieldbook")
    setup_project.iam_roles(ask=lambda question: True, say=print)
    assert len(checked) == 2


def test_init_creates_roles_only_when_confirmed(mock_aws) -> None:
    """D3: declined (or not a terminal), confirmed, and without permission to read IAM."""
    from fieldbook import setup_project

    said: list[str] = []
    setup_project.iam_roles(ask=lambda question: False, say=said.append)
    assert "not changed" in said[-1] and "Statement" in said[0]  # the policy is shown first
    run, build = aws.roles(settings())
    assert aws.role_state(run) == aws.role_state(build) == "missing"
    setup_project.iam_roles(ask=lambda question: True, say=said.append)
    assert said[-1].endswith(": ready") and aws.role_state(run) == aws.role_state(build) == "ok"
    setup_project.iam_roles(ask=lambda question: pytest.fail("asked again"), say=said.append)
    assert said[-1] == "IAM roles: up to date"


def test_init_without_permission_to_read_iam(mock_aws, monkeypatch: pytest.MonkeyPatch) -> None:
    from fieldbook import setup_project

    monkeypatch.setattr(aws, "role_state", lambda role: "unknown")
    said: list[str] = []
    setup_project.iam_roles(ask=lambda question: pytest.fail("asked"), say=said.append)
    assert "may not read IAM" in said[-1]


def test_init_is_idempotent(clean_env: Path) -> None:
    subprocess.run(["git", "init", "-q", "my repo!"], cwd=clean_env, check=True)
    repo = clean_env / "my repo!"
    (repo / ".gitignore").write_text("data/")  # no trailing newline
    first = setup_project.init(repo)
    assert any(line.startswith("created") and "fieldbook.toml" in line for line in first), first
    assert tomllib.loads((repo / "fieldbook.toml").read_text()) == {"project": "my-repo"}
    assert (repo / ".gitignore").read_text() == "data/\n.env\n.fieldbook/\n"
    assert "mlruns/" in (repo / ".fieldbookignore").read_text()
    files = {
        p: p.read_text()
        for p in (repo / "fieldbook.toml", repo / ".gitignore", repo / ".fieldbookignore")
    }
    second = setup_project.init(repo)
    assert not any("created" in line or "added" in line for line in second), second
    assert files == {p: p.read_text() for p in files}


def test_init_extends_pyproject(clean_env: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=clean_env, check=True)
    (clean_env / "pyproject.toml").write_text('[project]\nname = "x"')
    setup_project.init(clean_env)
    parsed = tomllib.loads((clean_env / "pyproject.toml").read_text())
    assert parsed["tool"]["fieldbook"] == {"project": "work"} and parsed["project"]["name"] == "x"
    assert not (clean_env / "fieldbook.toml").exists() and settings().project == "work"


def test_init_outside_a_repository(clean_env: Path) -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0 and "not in a git repository" in result.output


def test_init_records_your_github_login_as_author(
    clean_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings never ask GitHub; `fieldbook init` does once (sharing.md §4)."""
    subprocess.run(["git", "init", "-q"], cwd=clean_env, check=True)
    monkeypatch.delenv("FIELDBOOK_AUTHOR")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text("#!/bin/sh\necho Octo-Cat\n")
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    lines = setup_project.init(clean_env)
    assert "author: octo-cat (your GitHub login; in .fieldbook/config.toml)" in lines
    assert config.settings(clean_env).author == "octo-cat"
    (bin_dir / "gh").write_text("#!/bin/sh\nexit 1\n")
    assert not any(line.startswith("author") for line in setup_project.init(clean_env))


def test_init_adds_and_refreshes_agent_instructions(clean_env: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=clean_env, check=True)
    notes, other = (clean_env / name for name in setup_project.AGENT_FILES)
    setup_project.init(clean_env)
    assert not notes.exists()  # only when asked for
    notes.write_text("# My project\n\nUse tabs.")
    lines = setup_project.init(clean_env, agent=True)
    assert any("added fieldbook's instructions" in line for line in lines), lines
    text = notes.read_text()
    assert text.startswith("# My project\n\nUse tabs.\n\n<!-- fieldbook:start -->")
    assert "fieldbook log --json" in text and "fieldbook log diff" in text and "--smoke" in text
    assert "fieldbook log note" in text and "fieldbook run --rerun" in text
    assert other.read_text() == text.removeprefix("# My project\n\nUse tabs.\n\n")  # AGENTS.md
    assert notes.read_text() == text and "has fieldbook's" in "".join(setup_project.init(clean_env))
    notes.write_text(text.replace("fieldbook log --json", "an old command") + "\nMine, below.\n")
    lines = setup_project.init(clean_env)  # a newer fieldbook: its section is replaced
    assert any("updated fieldbook's instructions" in line for line in lines), lines
    assert notes.read_text() == text + "\nMine, below.\n"
