"""Where MLflow runs and artifacts are kept: the tracking store (`MLFLOW_TRACKING_URI` or
the `tracking_uri` setting, else this machine's SQLite database) and the artifact root
under the storage root."""

import os
import re
from pathlib import Path

from fieldbook import credentials, storage
from fieldbook.config import Settings


def mlflow_db(cfg: Settings) -> Path:
    """This machine's MLflow run database, in the local state folder: SQLite needs a local
    disk, even when the storage root is a bucket (storage.md §4)."""
    return cfg.state_dir / "mlflow" / "mlflow.db"


def artifact_root(cfg: Settings) -> str:
    """Where MLflow artifacts go: under the storage root, so they sit next to the data."""
    return storage.uri("mlflow/artifacts", cfg)


# MLflow tracking URIs that name a database (`postgresql+psycopg2://...` included), which
# clients use directly; other URIs (http, https, databricks) name an MLflow server.
DATABASE_SCHEMES = ("sqlite", "postgresql", "mysql", "mssql")


def configured_uri(workdir: Path | None = None) -> str | None:
    """A shared tracking store (tracking.md §1): `MLFLOW_TRACKING_URI` from the environment
    or the project's `.env`, else the `tracking_uri` setting (without its password, which
    is `PGPASSWORD` in `.env`). None if none is set."""
    from fieldbook import config

    if uri := os.environ.get("MLFLOW_TRACKING_URI"):
        return uri
    _, values = credentials.project_values(workdir or Path.cwd())
    if uri := values.get("MLFLOW_TRACKING_URI"):
        return uri
    return config.settings(workdir).tracking_uri


def tracking_uri(cfg: Settings) -> str:
    """The MLflow store runs log to: the configured URI, else this machine's database."""
    return configured_uri() or f"sqlite:///{mlflow_db(cfg)}"


POSTGRES = re.compile(r"^postgres(ql)?(\+\w+)?://")


def is_postgres(uri: str | None) -> bool:
    return bool(uri) and POSTGRES.match(uri) is not None


def read_only(uri: str | None) -> bool:
    """Whether this login can read a shared Postgres run database but not record runs: a
    project's viewer (sharing.md §2). False when it cannot be told (no tables yet,
    no driver, unreachable): the real attempt then reports the problem."""
    if not is_postgres(uri):
        return False
    try:
        import psycopg

        with psycopg.connect(POSTGRES.sub("postgresql://", uri), connect_timeout=10) as conn:
            row = conn.execute(
                "SELECT has_table_privilege(current_user, 'runs', 'INSERT')"
            ).fetchone()
        return row is not None and row[0] is False
    except Exception:
        return False


def login_hint(error: str) -> str:
    """What to do when the database refuses the login: on a shared project the owner may
    have changed this person's password (their SSH keys changed; sharing.md §2)."""
    if "password authentication failed" not in error:
        return ""
    return " (your project credentials may have changed: `git pull`, then `fieldbook init`)"


VIEWER = (
    "your database login can read this project's runs but not record them: you are a viewer. "
    "For Write access the owner changes your GitHub permission and runs `fieldbook access sync`"
)


def is_database(uri: str) -> bool:
    return uri.split("://", 1)[0].split("+", 1)[0] in DATABASE_SCHEMES
