"""Settings (fieldbook/config/): sources, precedence, validation, `fieldbook config`."""

import subprocess
from pathlib import Path

import pytest
from helpers import runner

from fieldbook import config
from fieldbook.cli import app
from fieldbook.config import ConfigError, settings


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def write_local(root: Path, text: str) -> Path:
    path = root / ".fieldbook" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_defaults(clean_env: Path) -> None:
    """Outside any project: local data and state in the working directory's .fieldbook."""
    s = settings()
    assert s.project == "default" and s.experiment is None
    assert s.storage_root == "file://" + str((clean_env / ".fieldbook" / "data").resolve())
    assert s.state_dir == (clean_env / ".fieldbook").resolve()


def test_state_dir_is_separate_from_a_bucket_root(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (clean_env / "fieldbook.toml").write_text(
        'project = "vision"\nstorage_root = "s3://bucket/fieldbook"\n'
    )
    s = settings()
    assert s.storage_root == "s3://bucket/fieldbook" and s.state_dir == clean_env / ".fieldbook"
    monkeypatch.setenv("FIELDBOOK_STATE_DIR", str(clean_env / "elsewhere"))
    assert settings().state_dir == (clean_env / "elsewhere").resolve()


def test_layers_and_precedence(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (clean_env / "pyproject.toml").write_text(
        '[tool.fieldbook]\nproject = "vision"\nstorage_root = "s3://shared/vision"\n'
    )
    sub = clean_env / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)  # found by searching upward
    s = settings()
    assert (s.project, s.storage_root) == ("vision", "s3://shared/vision")
    assert s.state_dir == clean_env / ".fieldbook"  # next to the project file
    write_local(clean_env, 'storage_root = "data"\nmlflow_port = 6000\n')
    s = settings()  # this machine's file wins over the project file; paths relative to it
    assert s.storage_root == "file://" + str((clean_env / ".fieldbook" / "data").resolve())
    assert (s.project, s.mlflow_port) == ("vision", 6000)
    (clean_env / "fieldbook.toml").write_text('project = "dedicated"\nexperiment = "exp"\n')
    assert (settings().project, settings().experiment) == (
        "dedicated",
        "exp",
    )  # fieldbook.toml wins
    monkeypatch.setenv("FIELDBOOK_PROJECT", "from-env")
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "/tmp/x")
    s = settings()
    assert s.project == "from-env" and s.storage_root == "file://" + str(Path("/tmp/x").resolve())


def test_a_nested_project_has_its_own_folder(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (clean_env / "fieldbook.toml").write_text('project = "outer"\n')
    inner = clean_env / "inner"
    inner.mkdir()
    (inner / "fieldbook.toml").write_text('project = "inner"\n')
    monkeypatch.chdir(inner)
    assert (settings().project, settings().state_dir) == ("inner", inner / ".fieldbook")


@pytest.mark.parametrize(
    ("local", "project", "error"),
    [
        ("", 'projct = "x"\n', "unknown settings ['projct']"),
        ("root = 1\n", 'project = "x"\n', "unknown settings ['root']"),
        ("", "project = \n", "fieldbook.toml"),  # invalid TOML
        ("", 'project = "../x"\n', "invalid project name"),
        ('storage_root = "gs://b"\n', 'project = "x"\n', "unsupported storage root"),
    ],
)
def test_invalid_settings(clean_env: Path, local: str, project: str, error: str) -> None:
    if local:
        write_local(clean_env, local)
    (clean_env / "fieldbook.toml").write_text(project)
    with pytest.raises(ConfigError, match=error.replace("[", r"\[").replace("]", r"\]")):
        settings()
    result = runner.invoke(app, ["data", "ls"])
    assert result.exit_code == 1 and "Traceback" not in result.output


def test_data_and_run_commands_need_a_project(clean_env: Path) -> None:
    result = runner.invoke(app, ["data", "ls"])
    assert result.exit_code == 1 and "run `fieldbook init`" in result.output
    assert runner.invoke(app, ["config"]).exit_code == 0  # works anywhere
    with pytest.raises(ConfigError, match="not in a fieldbook project"):
        config.require_project()
    (clean_env / "fieldbook.toml").write_text('project = "vision"\n')
    config.require_project()


def test_the_author_is_git_user_name(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIELDBOOK_AUTHOR")  # set by conftest
    repo = clean_env / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Ada Lovelace")
    monkeypatch.chdir(repo)
    assert settings().author == "Ada Lovelace"
    (repo / "fieldbook.toml").write_text('project = "vision"\nauthor = "ada"\n')
    assert settings().author == "ada"


def test_worktrees_share_the_main_trees_folder(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs and local data are per project, not per checkout."""
    repo = clean_env / "repo"
    (repo / "sub").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "fieldbook.toml").write_text('project = "vision"\n')
    (repo / "sub" / "fieldbook.toml").write_text('project = "nested"\n')
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    git(repo, "worktree", "add", "-q", str(clean_env / "feature"))
    monkeypatch.chdir(clean_env / "feature")
    assert settings().state_dir == (repo / ".fieldbook").resolve()
    monkeypatch.chdir(clean_env / "feature" / "sub")
    assert settings().state_dir == (repo / "sub" / ".fieldbook").resolve()


def test_every_setting_has_a_variable_and_a_source(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (clean_env / "fieldbook.toml").write_text('project = "vision"\nauthor = "alice"\n')
    write_local(clean_env, "mlflow_port = 6000\n")
    monkeypatch.setenv("FIELDBOOK_LAUNCH_TIMEOUT", "40")
    monkeypatch.delenv("FIELDBOOK_AUTHOR")  # set by conftest
    s = settings()
    assert (s.author, s.mlflow_port, s.launch_timeout) == ("alice", 6000, 40)
    sources = {name: source for name, _, source in config.describe()}
    assert sources["author"] == sources["project"] == "fieldbook.toml"
    assert sources["mlflow_port"] == ".fieldbook/config.toml" and sources["experiment"] == "default"
    assert sources["launch_timeout"] == "FIELDBOOK_LAUNCH_TIMEOUT"
    monkeypatch.setenv("FIELDBOOK_MLFLOW_PORT", "not-a-port")
    with pytest.raises(ConfigError, match="invalid setting mlflow_port"):
        settings()


def test_fieldbook_config_and_env_file(clean_env: Path) -> None:
    (clean_env / "pyproject.toml").write_text(
        '[tool.fieldbook]\ntracking_uri = "postgresql://bob@db.example.com/mlflow"\n'
    )
    (clean_env / ".env").write_text("PGPASSWORD=s3cret\nFIELDBOOK_PROJECT=from-dotenv\n")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "postgresql://db.example.com/mlflow" in result.output  # no user name or password
    assert "s3cret" not in result.output and ".env: FIELDBOOK_PROJECT, PGPASSWORD" in result.output
    assert "from-dotenv" in result.output  # .env is loaded for fieldbook's own commands
    assert "pyproject.toml [tool.fieldbook]" in result.output  # brackets are not rich markup


def test_guide_lists_every_setting() -> None:
    guide = (Path(__file__).parents[2] / "docs" / "guide.md").read_text()
    assert config.settings_table() in guide, "update docs/guide.md §8 from config.settings_table()"


def test_fieldbook_config_shows_where_runs_log(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    result = runner.invoke(app, ["config"])
    assert "mlflow.db" in result.output and "default (this machine)" in result.output
    (clean_env / ".env").write_text("MLFLOW_TRACKING_URI=http://mlflow.example.com:5000\n")
    result = runner.invoke(app, ["config"])
    assert "http://mlflow.example.com:5000" in result.output
    assert ".env (MLFLOW_TRACKING_URI)" in result.output


def test_every_setting_has_exactly_one_group() -> None:
    """G2: `fieldbook config` and the docs show settings by group; the files keep flat keys."""
    from fieldbook.config import GROUPS, Settings

    grouped = [name for _, names in GROUPS for name in names]
    assert sorted(grouped) == sorted(Settings.model_fields) and len(grouped) == len(set(grouped))
    assert [g for g, _ in GROUPS] == [
        "Storage", "Cloud runs", "Project", "Experiments (MLflow)",
    ]  # fmt: skip
