"""`fieldbook config set`, `unset`, and `get` (fieldbook/config/edit.py)."""

import subprocess
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fieldbook.cli import app
from fieldbook.config import ConfigError, edit, settings

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repository with a pyproject.toml, and no FIELDBOOK_* settings in the environment."""
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "pyproject.toml").write_text('[project]\nname = "p"\n')
    monkeypatch.chdir(root)
    for var in (
        "FIELDBOOK_STORAGE_ROOT",
        "FIELDBOOK_STATE_DIR",
        "FIELDBOOK_PROJECT",
        "FIELDBOOK_AUTHOR",
    ):
        monkeypatch.delenv(var, raising=False)
    return root


def _local(project: Path) -> dict:
    return tomllib.loads((project / ".fieldbook" / "config.toml").read_text())


def _shared(project: Path) -> dict:
    return tomllib.loads((project / "pyproject.toml").read_text())["tool"]["fieldbook"]


def test_settings_go_to_the_project_file_or_with_local_to_this_machine(project: Path) -> None:
    edit.set_value("project", "vision")
    edit.set_value("storage_root", "s3://lab-bucket/vision")
    edit.set_value("storage_root", "~/big-disk", local=True)
    assert _shared(project) == {"project": "vision", "storage_root": "s3://lab-bucket/vision"}
    assert _local(project) == {"storage_root": "~/big-disk"}
    assert settings().storage_root.endswith("/big-disk")  # this machine's wins
    edit.set_value("experiment", "baseline")  # the same project file, not a new one
    assert not (project / "fieldbook.toml").exists()
    assert edit.unset_value("storage_root", local=True).startswith("storage_root removed")
    assert settings().storage_root == "s3://lab-bucket/vision"


def test_without_pyproject_a_project_file_is_created(project: Path) -> None:
    (project / "pyproject.toml").unlink()
    edit.set_value("project", "vision")
    assert tomllib.loads((project / "fieldbook.toml").read_text()) == {"project": "vision"}


def test_local_needs_a_project(project: Path) -> None:
    with pytest.raises(ConfigError, match="--local needs a project"):
        edit.set_value("mlflow_port", "6000", local=True)


def test_values_take_the_settings_type(project: Path) -> None:
    edit.set_value("image_apt", "ffmpeg, git")
    edit.set_value("mlflow_port", "6000")
    edit.set_value("state_dir", "local-state", local=True)  # relative: to the working directory
    assert _shared(project)["image_apt"] == ["ffmpeg", "git"]
    assert _shared(project)["mlflow_port"] == 6000
    assert _local(project)["state_dir"] == str(project / "local-state")
    assert edit.get_value("image_apt") == "ffmpeg,git" and edit.get_value("mlflow_port") == "6000"


def test_invalid_values_are_refused_and_nothing_is_written(project: Path) -> None:
    before = (project / "pyproject.toml").read_text()
    for key, value, message in [
        ("mlflow_port", "abc", "whole number"),
        ("storage_root", "gs://bucket", "unsupported storage root"),
        ("project", "../escape", "invalid project name"),
        ("tracking_uri", "postgresql://bob:pw@db/mlflow", "PGPASSWORD"),
        ("no_such_setting", "1", "unknown setting"),
    ]:
        with pytest.raises(ConfigError, match=message):
            edit.set_value(key, value)
    assert (project / "pyproject.toml").read_text() == before


def test_comments_and_layout_are_kept(project: Path) -> None:
    path = project / "pyproject.toml"
    path.write_text('[project]\nname = "p"\n\n# shared\n[tool.fieldbook]\nproject = "v"  # name\n')
    edit.set_value("author", "ada")
    text = path.read_text()
    assert "# shared" in text and "# name" in text and 'author = "ada"' in text


def test_unset_and_get(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    edit.set_value("author", "ada")
    assert edit.get_value("author") == "ada"
    assert edit.unset_value("author").startswith("author removed")
    assert "is not set" in edit.unset_value("author")
    assert edit.get_value("experiment") is None
    monkeypatch.setenv("FIELDBOOK_AUTHOR", "grace")
    assert "FIELDBOOK_AUTHOR is set and overrides it" in edit.set_value("author", "ada")


def test_a_broken_settings_file_can_be_fixed_with_the_cli(project: Path) -> None:
    edit.set_value("storage_root", "s3://lab-bucket/fieldbook")
    path = project / "pyproject.toml"
    path.write_text(path.read_text().replace("s3://lab-bucket/fieldbook", "gs://not-supported"))
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 1 and "fix it with `fieldbook config set`" in result.output
    result = runner.invoke(app, ["config", "set", "storage_root", "s3://lab-bucket/fieldbook"])
    assert result.exit_code == 0, result.output
    assert (
        runner.invoke(app, ["config", "get", "storage_root"]).output.strip()
        == "s3://lab-bucket/fieldbook"
    )
