"""The MLflow variables of a run: which store and experiment it logs to (creating the
experiment with its artifacts under the storage root), and system metrics."""

import os
import sqlite3
import subprocess
from pathlib import Path

from fieldbook.config import Settings
from fieldbook.launch import LaunchError, note
from fieldbook.tracking import store


def system_metrics_env(found: dict) -> dict[str, str]:
    """Turn on MLflow's system metrics (CPU, GPU, memory) if the run's interpreter can
    collect them. MLflow refuses to start runs when they are on and `psutil` is missing,
    so they stay off then. An explicit `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING` wins."""
    if "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING" in os.environ or not found.get("psutil"):
        return {}
    return {"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true"}


# Run by the script's interpreter: create the experiment if missing; print its stage, or
# "read-only" for a viewer's login, which could start the script but not record its run.
_CREATE_EXPERIMENT = """
import logging, os, sys
logging.getLogger("mlflow").setLevel(logging.WARNING)  # database creation messages
from fieldbook.tracking.store import read_only
if read_only(os.environ["MLFLOW_TRACKING_URI"]):
    print("read-only")
    sys.exit(0)
from mlflow.tracking import MlflowClient
name, artifacts = sys.argv[1:]
client = MlflowClient()  # MLFLOW_TRACKING_URI: kept off the command line, as it may hold a password
found = client.get_experiment_by_name(name)
if found is None:
    client.create_experiment(name, artifact_location=artifacts)
print(found.lifecycle_stage if found is not None else "active")
"""


def mlflow_env(
    cfg: Settings, experiment: str, found: dict, python: str, uri: str | None = None
) -> dict[str, str]:
    """MLflow settings for the run, creating the experiment with its artifacts under the
    storage root (tracking.md §2). `found` and `python` describe the interpreter the
    script runs with (`environments.query_interpreter`), which is the one that must have MLflow.
    Empty if it has no MLflow, or only mlflow-skinny, which cannot use a database.

    `uri` (default: `MLFLOW_TRACKING_URI`) may name a shared database, such as Postgres,
    which every machine logs to directly (tracking.md §1); the experiment is created
    there too, or MLflow would put its artifacts in `./mlruns` on whichever machine ran
    first. A URI naming an MLflow server is left to the server.
    """

    uri = uri or os.environ.get("MLFLOW_TRACKING_URI") or cfg.tracking_uri
    if uri and not store.is_database(uri):  # a server sets artifact locations itself
        return {
            "MLFLOW_TRACKING_URI": uri,
            "MLFLOW_EXPERIMENT_NAME": os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment),
        }
    if not found.get("mlflow"):
        return {}
    if not found.get("mlflow_sql"):
        note(
            f"MLflow in {python} is mlflow-skinny, which cannot use the local store: "
            "runs are not tracked. Install full MLflow (pip install 'fieldbook[tracking]')."
        )
        return {}
    if uri is None:
        db = store.mlflow_db(cfg)
        db.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db}"
    local = Path(uri.removeprefix("sqlite:///")) if uri.startswith("sqlite:///") else None
    stage = _experiment_stage(local, experiment) if local else None
    if stage is None:
        # Creating it takes seconds (MLflow import, database setup), so a local database is
        # checked directly first; a shared one is asked through MLflow on every run.
        artifacts = f"{store.artifact_root(cfg)}/{experiment}"
        _check_artifact_length(uri, artifacts)
        result = subprocess.run(
            [python, "-c", _CREATE_EXPERIMENT, experiment, artifacts],
            capture_output=True, text=True, env={**os.environ, "MLFLOW_TRACKING_URI": uri},
        )  # fmt: skip
        if result.returncode != 0:
            error = (result.stderr.strip().splitlines()[-1:] or ["unknown error"])[0]
            if "psycopg" in error:
                error += " (Postgres needs pip install 'psycopg[binary]' in the project)"
            error += store.login_hint(result.stderr)
            raise LaunchError(f"could not set up MLflow experiment {experiment!r}: {error}")
        stage = result.stdout.strip().splitlines()[-1]
    if stage == "read-only":
        raise LaunchError(f"runs cannot be recorded: {store.VIEWER}")
    if stage != "active":
        raise LaunchError(f"MLflow experiment {experiment!r} is deleted; restore it or rename")
    env = {"MLFLOW_TRACKING_URI": uri, "MLFLOW_EXPERIMENT_NAME": experiment}
    if local is None and "MLFLOW_ENABLE_ASYNC_LOGGING" not in os.environ:
        env["MLFLOW_ENABLE_ASYNC_LOGGING"] = "true"  # a remote database: do not wait per call
    return env


def _experiment_stage(db: Path, experiment: str) -> str | None:
    """An experiment's lifecycle stage read directly from MLflow's SQLite store, or None if
    it does not exist or cannot be read (then the MLflow client decides)."""
    if not db.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            row = con.execute(
                "SELECT lifecycle_stage FROM experiments WHERE name = ?", (experiment,)
            ).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


# MLflow's database schema stores a run's artifact address in 200 characters; each run adds
# "/<32-character run ID>/artifacts" to the experiment's location. SQLite does not enforce it.
ARTIFACT_URI_LIMIT = 200
RUN_SUFFIX = 1 + 32 + len("/artifacts")


def _check_artifact_length(uri: str, artifacts: str) -> None:
    """Stop before MLflow fails every run with "value too long"."""
    if uri.startswith("sqlite:") or len(artifacts) + RUN_SUFFIX <= ARTIFACT_URI_LIMIT:
        return
    over = len(artifacts) + RUN_SUFFIX - ARTIFACT_URI_LIMIT
    raise LaunchError(
        f"the MLflow artifact location {artifacts} is too long for the tracking database "
        f"(at most {ARTIFACT_URI_LIMIT} characters with the run ID): shorten the storage root "
        f"or the experiment name by {over} characters"
    )
