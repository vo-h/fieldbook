"""MLflow setup for runs (fieldbook/tracking/setup.py), including a shared Postgres database."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
    run_foreground_ui,
    worker_port,
)

import fieldbook as fb
from fieldbook.code import environments
from fieldbook.config import settings
from fieldbook.tracking import records, setup
from fieldbook.tracking import store as tracking_store


@pytest.mark.slow
@pytest.mark.mlflow
def test_mlflow_run_is_tagged(repo: Path, gitleaks: Path) -> None:
    (repo / "train.py").write_text(
        "import mlflow, fieldbook as fb\n"
        "fb.dataset('raw', version=1).files()\n"  # read before the run starts
        "with mlflow.start_run():\n"
        "    mlflow.log_metric('loss', 0.5)\n"
        "    with fb.dataset('clean').edit() as e:\n"
        "        e.copy_from(fb.dataset('raw'))\n"
    )
    code, out = run_cli("run", "--experiment", "exp", "train.py", "lr=0.01", "--api-key", "s3cr3t")
    assert code == 0, out
    import mlflow

    db = tracking_store.mlflow_db(settings())
    client = mlflow.MlflowClient(f"sqlite:///{db}")
    experiment = client.get_experiment_by_name("exp")
    (logged,) = client.search_runs([experiment.experiment_id])
    tags = logged.data.tags
    version = fb.dataset("clean").log()[0]
    assert tags["fieldbook.run_id"] == version.run_id and tags["fieldbook.code"] == version.code
    assert json.loads(tags["fieldbook.inputs"]) == ["test/raw@1"]
    assert json.loads(tags["fieldbook.outputs"]) == ["test/clean@1"]
    assert tags["fieldbook.command"] == "train.py lr=0.01 --api-key '***'"  # secrets masked
    assert logged.info.artifact_uri.startswith(fb.storage.local_path("mlflow").as_uri())
    assert not (repo / "mlruns").exists()
    client.delete_experiment(experiment.experiment_id)
    assert "is deleted" in run_cli("run", "--experiment", "exp", "train.py")[1]


def test_system_metrics_only_when_collectable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", raising=False)
    assert setup.system_metrics_env({"psutil": True}) == {
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true"
    }
    assert setup.system_metrics_env({"psutil": False}) == {}  # MLflow would refuse to start
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")  # the user's choice wins
    assert setup.system_metrics_env({"psutil": True}) == {}


ENV_PRINTER = """import json, os
json.dump({k: os.environ.get(k) for k in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME")},
          open("mlflow_env.json", "w"))
"""


@pytest.mark.mlflow
@pytest.mark.parametrize(
    ("mlflow", "mlflow_sql", "note"),
    [(True, False, "mlflow-skinny, which cannot use the local store"), (False, False, None)],
)
def test_mlflow_is_checked_in_the_project_interpreter(
    repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch, mlflow, mlflow_sql, note
) -> None:
    real = environments.query_interpreter
    monkeypatch.setattr(
        environments,
        "query_interpreter",
        lambda py: {**real(py), "mlflow": mlflow, "mlflow_sql": mlflow_sql},
    )
    (repo / "show.py").write_text(ENV_PRINTER)
    (repo / ".gitignore").write_text("data/\nargs.txt\nmlflow_env.json\n")
    code, out = run_cli("run", "show.py")
    assert code == 0, out
    assert json.loads((repo / "mlflow_env.json").read_text())["MLFLOW_TRACKING_URI"] is None
    assert (note in out) if note else ("mlflow-skinny" not in out)
    assert not tracking_store.mlflow_db(settings()).exists()  # no store set up


LOG_ARTIFACT = """import mlflow
with mlflow.start_run() as run:
    mlflow.log_text("hello", "a.txt")
    open("artifact_uri.txt", "w").write(run.info.artifact_uri)
"""


@pytest.mark.slow
@pytest.mark.mlflow
def test_runs_log_to_the_store_and_the_ui_shows_them(
    repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    (repo / "log.py").write_text(LOG_ARTIFACT)
    (repo / ".gitignore").write_text("data/\nargs.txt\nartifact_uri.txt\n")
    code, out = run_cli("run", "log.py")  # no server needed
    assert code == 0, out
    artifacts = fb.storage.local_path("mlflow/artifacts")
    assert (repo / "artifact_uri.txt").read_text().startswith(artifacts.as_uri())
    assert list(artifacts.rglob("a.txt")) and not (repo / "mlruns").exists()

    monkeypatch.setenv("FIELDBOOK_MLFLOW_PORT", str(worker_port(0)))
    path = "/api/2.0/mlflow/experiments/get-by-name?experiment_name=test"
    assert run_foreground_ui("log", "ui", "--no-open", path=path)


@pytest.fixture(scope="module")
def postgres(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A throwaway local Postgres server (trust authentication); skipped without one."""

    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not (initdb and pg_ctl):
        pytest.skip("Postgres is not installed")
    data = tmp_path_factory.mktemp("pg") / "data"
    subprocess.run(
        [initdb, "-D", data, "-A", "trust", "-U", "fieldbook"], check=True, capture_output=True
    )
    port = worker_port(150)
    options = f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories=''"
    log = data.parent / "postgres.log"
    subprocess.run([pg_ctl, "-D", data, "-o", options, "-l", log, "-w", "start"], check=True)
    yield f"postgresql://fieldbook@127.0.0.1:{port}/postgres"
    subprocess.run([pg_ctl, "-D", data, "-m", "fast", "-w", "stop"], capture_output=True)


SHARED_DB_SCRIPT = """import os, mlflow
import fieldbook as fb
fb.dataset("raw").files()  # an input, recorded on the MLflow run
with mlflow.start_run() as run:
    mlflow.log_param("lr", 0.1)
    mlflow.log_text("hello", "a.txt")
open("seen.txt", "w").write(os.environ.get("MLFLOW_ENABLE_ASYNC_LOGGING", ""))
"""


@pytest.mark.mlflow
@pytest.mark.slow
def test_shared_postgres_tracking_database(
    repo: Path, gitleaks: Path, postgres: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    import mlflow

    from fieldbook import setup_project

    # A short storage root: MLflow's database allows 200 characters per artifact address.
    short_root = tempfile.mkdtemp(prefix="fb-", dir="/tmp")
    shutil.copytree(settings().storage_root.removeprefix("file://"), short_root, dirs_exist_ok=True)
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", short_root)
    # As the guide says: the address in the settings, the password in .env (trust auth here).
    (repo / ".fieldbook").mkdir(exist_ok=True)
    (repo / ".fieldbook" / "config.toml").write_text(f'tracking_uri = "{postgres}"\n')
    (repo / ".env").write_text("PGPASSWORD=pw-in-env\n")
    (repo / ".gitignore").write_text("data/\nargs.txt\nseen.txt\n.env\n")
    (repo / "log.py").write_text(SHARED_DB_SCRIPT)
    line = setup_project._tracking(repo)
    assert "reachable, schema up to date" in line and "pw-in-env" not in line
    code, out = run_cli("run", "log.py")
    assert code == 0, out
    assert (repo / "seen.txt").read_text() == "true"  # async logging to a remote database
    (logged,) = mlflow.MlflowClient(postgres).search_runs(
        [mlflow.MlflowClient(postgres).get_experiment_by_name("test").experiment_id]
    )
    assert logged.data.params == {"lr": "0.1"} and logged.data.tags["fieldbook.run_id"]
    assert logged.info.artifact_uri.startswith(fb.storage.uri("mlflow/artifacts/test"))
    assert not tracking_store.mlflow_db(settings()).exists()  # nothing logged on this machine only
    run_id = logged.data.tags["fieldbook.run_id"]  # cleanup sees runs from every machine
    assert records.referenced_versions(settings()) == {"test/raw": {1: [run_id]}}

    monkeypatch.setenv("FIELDBOOK_MLFLOW_PORT", str(worker_port(0)))
    path = "/api/2.0/mlflow/experiments/get-by-name?experiment_name=test"
    assert run_foreground_ui("log", "ui", "--no-open", path=path)  # the shared database


def test_artifact_addresses_too_long_for_the_database_stop_early() -> None:
    """A13: MLflow's database allows 200 characters per run's artifact address."""
    from fieldbook.launch import LaunchError
    from fieldbook.tracking import setup

    short = "s3://bucket/fieldbook/mlflow/artifacts/exp"
    setup._check_artifact_length("postgresql://db/mlflow", short)
    setup._check_artifact_length("sqlite:///x.db", "file:///" + "x" * 300)  # SQLite: no limit
    with pytest.raises(LaunchError, match="shorten the storage root or the experiment name by"):
        setup._check_artifact_length("postgresql://db/mlflow", "file:///" + "x" * 200)
