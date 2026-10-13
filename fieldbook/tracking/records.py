"""Run records in MLflow, read without importing MLflow where possible.

Cleanup uses them to keep the dataset versions that runs read (data.md §7):
every MLflow run started in a launched process carries an `fieldbook.inputs` tag listing them
(tracking.md §3).
"""

import json
import sqlite3
from pathlib import Path

from fieldbook import credentials
from fieldbook.config import Settings
from fieldbook.data.dataset import DatasetError
from fieldbook.data.lineage import _REF_RE
from fieldbook.tracking import store

# "project/name" -> {version: [run IDs]}
References = dict[str, dict[int, list[str]]]


class RunRecordsError(DatasetError):
    """Run records could not be read, so nothing is deleted."""


def referenced_versions(cfg: Settings) -> References:
    """Dataset versions read by active MLflow runs. Runs deleted in MLflow do not count."""
    references: References = {}
    rows = _tag_rows(cfg, "fieldbook.inputs")
    for run_id, inputs in rows:
        try:
            refs = json.loads(inputs)
        except json.JSONDecodeError:
            raise RunRecordsError(
                f"run {run_id} has an invalid fieldbook.inputs tag; nothing was deleted"
            ) from None
        for ref in refs:
            match = _REF_RE.match(ref)
            if match:
                key, version = f"{match.group(1)}/{match.group(2)}", int(match.group(3))
                references.setdefault(key, {}).setdefault(version, []).append(run_id)
    for versions in references.values():
        for run_ids in versions.values():
            run_ids.sort()
    return references


def referenced_environment(cfg: Settings) -> tuple[set[str], set[str]]:
    """Code snapshot hashes and package-list hashes that active MLflow runs refer to
    (`fieldbook.code`, and `packages` in `fieldbook.env`), so storage retention keeps them."""
    code = {value for _, value in _tag_rows(cfg, "fieldbook.code")}
    packages = set()
    for run_id, value in _tag_rows(cfg, "fieldbook.env"):
        try:
            packages.add(json.loads(value).get("packages"))
        except (json.JSONDecodeError, AttributeError):
            raise RunRecordsError(
                f"run {run_id} has an invalid fieldbook.env tag; nothing was deleted"
            ) from None
    return code, packages - {None}


def _tag_rows(cfg: Settings, key: str) -> list[tuple[str, str]]:
    """(run ID, value) of tag `key` on active MLflow runs."""
    uri = store.tracking_uri(cfg)
    if uri.startswith("sqlite:///"):
        return _sqlite_rows(Path(uri.removeprefix("sqlite:///")), key)
    return _client_rows(uri, key)


def _sqlite_rows(db: Path, key: str) -> list[tuple[str, str]]:
    """(run ID, value of tag `key`) for active runs, read directly from MLflow's SQLite
    store."""
    if not db.exists():
        return []
    query = """
        SELECT coalesce(r.value, i.run_uuid), i.value
        FROM tags i
        JOIN runs ON runs.run_uuid = i.run_uuid AND runs.lifecycle_stage = 'active'
        LEFT JOIN tags r ON r.run_uuid = i.run_uuid AND r.key = 'fieldbook.run_id'
        WHERE i.key = ?
    """
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            return con.execute(query, (key,)).fetchall()
    except sqlite3.Error as e:
        raise RunRecordsError(
            f"cannot read run records from {db} ({e}); nothing was deleted"
        ) from None


def _client_rows(uri: str, key: str) -> list[tuple[str, str]]:
    """(run ID, value of tag `key`) for active runs, through the MLflow client (e.g. a
    server)."""
    try:
        from mlflow.entities import ViewType
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise RunRecordsError(
            f"MLFLOW_TRACKING_URI is {credentials.display_uri(uri)}, but MLflow is not "
            "installed, so the dataset versions runs used cannot be checked; nothing was deleted"
        ) from None
    try:
        client = MlflowClient(uri)
        experiments = [e.experiment_id for e in client.search_experiments(ViewType.ALL)]
        rows, token = [], None
        while experiments:
            page = client.search_runs(
                experiments,
                filter_string=f"tags.`{key}` LIKE '%'",
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=1000,
                page_token=token,
            )
            rows += [
                (r.data.tags.get("fieldbook.run_id", r.info.run_id), r.data.tags[key]) for r in page
            ]
            token = page.token
            if not token:
                break
        return rows
    except Exception as e:  # network, auth, and server errors vary by MLflow version
        raise RunRecordsError(
            f"cannot read run records from {credentials.display_uri(uri)} "
            f"({credentials.redact(str(e), uri)}); nothing was deleted"
        ) from None
