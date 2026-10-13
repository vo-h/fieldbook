import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from helpers import SCRIPT

import fieldbook as fb
from fieldbook.config import settings
from fieldbook.tracking import store as tracking_store


@pytest.fixture(autouse=True)
def restore_environ():
    """`fieldbook` commands load the project's `.env` into os.environ; undo it after each test."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the platform at an isolated storage root for every test."""
    root = tmp_path / "storage"
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", str(root))
    monkeypatch.setenv("FIELDBOOK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("FIELDBOOK_PROJECT", "test")
    monkeypatch.setenv("FIELDBOOK_AUTHOR", "tester")
    for var in (
        "PLATFORM_RUN_ID", "PLATFORM_CODE", "PLATFORM_ENV", "MLFLOW_TRACKING_URI",
        "FIELDBOOK_EXPERIMENT",
    ):  # fmt: skip
        monkeypatch.delenv(var, raising=False)
    return root


# The real uv cache, found before HOME is replaced, so builds in tests stay offline-fast.
UV_CACHE = os.environ.get("UV_CACHE_DIR") or str(Path.home() / ".cache" / "uv")


@pytest.fixture(autouse=True)
def own_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home folder per test (D4): no real credentials or tool state (~/.aws, ~/.runpod,
    ~/.sky, ~/.gitconfig, ~/.config/gh) is visible, and no test can change them."""
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("UV_CACHE_DIR", UV_CACHE)
    for var in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                "AWS_DEFAULT_REGION", "AWS_REGION", "RUNPOD_API_KEY", "PGPASSWORD", "PGUSER",
                "GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR"):  # fmt: skip
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def files(tmp_path: Path) -> Path:
    """A small source directory with nested files."""
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "1.txt").write_text("one")
    (src / "a" / "2.txt").write_text("two")
    (src / "3.jpg").write_bytes(b"\xff\xd8three")
    return src


@pytest.fixture(scope="session")
def empty_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An MLflow SQLite store with its tables created (slow, so built once)."""
    from mlflow.tracking import MlflowClient

    db = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    MlflowClient(f"sqlite:///{db}").search_experiments()
    return db


@pytest.fixture
def store(empty_store: Path) -> Path:
    db = tracking_store.mlflow_db(settings())
    db.parent.mkdir(parents=True)
    shutil.copy(empty_store, db)
    return db


class MlflowRuns:
    """Creates and deletes runs in the test's local MLflow store."""

    def __init__(self, db: Path):
        from mlflow.tracking import MlflowClient

        self.db = db
        self.client = MlflowClient(f"sqlite:///{db}")

    def log(self, inputs: list[str], run_id: str = "", tags: dict | None = None) -> str:
        run = self.client.create_run(
            "0", tags={"fieldbook.inputs": json.dumps(inputs), **(tags or {})}
        )
        if run_id:
            self.client.set_tag(run.info.run_id, "fieldbook.run_id", run_id)
        return run.info.run_id

    def delete(self, mlflow_run_id: str) -> None:
        self.client.delete_run(mlflow_run_id)


@pytest.fixture
def mlflow_runs(store: Path) -> MlflowRuns:
    return MlflowRuns(store)


@pytest.fixture(scope="session", autouse=True)
def no_leaked_mlflow_servers(tmp_path_factory: pytest.TempPathFactory):
    """Fail if a test leaves an MLflow server running."""
    yield
    ours = str(tmp_path_factory.getbasetemp())  # this worker's tests only (pytest-xdist)
    import psutil

    leaked = []
    for process in psutil.process_iter(["cmdline"]):
        args = process.info["cmdline"] or []
        # Only `python -m mlflow server ...` with a store in a pytest temporary folder.
        is_server = args[1:4] == ["-m", "mlflow", "server"]
        if is_server and any(ours in arg for arg in args):
            leaked.append(" ".join(args))
            process.terminate()
    assert not leaked, f"tests left MLflow servers running (now stopped): {leaked}"


# Stands in for gitleaks: reports a finding for any file containing FAKE_SECRET.
FAKE_GITLEAKS = """#!/bin/bash
report=""
while [ $# -gt 0 ]; do [ "$1" = "--report-path" ] && report="$2"; shift; done
hits=$(grep -rl FAKE_SECRET . || true)
if [ -z "$hits" ]; then echo "[]" > "$report"; exit 0; fi
python3 -c 'import json,sys; json.dump([{"File": f[2:], "StartLine": 1, "RuleID": "fake"}
  for f in sys.argv[2:]], open(sys.argv[1], "w"))' "$report" $hits
exit 1
"""


@pytest.fixture
def gitleaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gitleaks").write_text(FAKE_GITLEAKS)
    (bin_dir / "gitleaks").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return bin_dir


@pytest.fixture
def repo(tmp_path: Path, files: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "proj"
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / ".gitignore").write_text("data/\nargs.txt\n")
    (path / "data").mkdir()
    (path / "data" / "big.bin").write_text("ignored")
    (path / "clean.py").write_text(SCRIPT)
    monkeypatch.chdir(path)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    with fb.dataset("raw").edit() as edit:
        edit.add_dir(files)
    return path


@pytest.fixture
def mock_aws(monkeypatch: pytest.MonkeyPatch):
    for name, value in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                        "AWS_DEFAULT_REGION": "us-east-1"}.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://data-bucket/fieldbook")
    monkeypatch.setenv("FIELDBOOK_IMAGE_REGISTRY", "1.dkr.ecr.us-east-1.amazonaws.com/fieldbook")
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        yield


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No FIELDBOOK_* variables, and a working directory outside any project."""
    for var in ("FIELDBOOK_STORAGE_ROOT", "FIELDBOOK_STATE_DIR", "FIELDBOOK_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.fixture(autouse=True)
def no_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests give the same results whether a person or a coding agent runs them: the
    variables an agent's shell sets (fieldbook/launch/agents.py) are cleared."""
    from fieldbook.launch import agents

    for variable in ("FIELDBOOK_AGENT", *agents.KNOWN):
        monkeypatch.delenv(variable, raising=False)
