"""Run records that cleanup reads (fieldbook/tracking/records.py)."""

import sys
from datetime import timedelta
from pathlib import Path

import pytest

import fieldbook as fb
from fieldbook.config import settings
from fieldbook.data import DatasetError
from fieldbook.data.cleanup import prune
from fieldbook.tracking import records

NOW = timedelta(0)


def _cats(files: Path, versions: int = 3) -> fb.data.Dataset:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files)
    for i in range(versions - 1):
        with ds.edit() as edit:
            edit.update("3.jpg", step=i)
    return ds


def test_referenced_versions(files: Path, store: Path, mlflow_runs) -> None:
    _cats(files)
    mlflow_runs.log(["test/cats@1", "hf://datasets/x"], run_id="r-1")
    mlflow_runs.log(["test/cats@1", "test/cats@2"], run_id="r-2")
    mlflow_runs.delete(mlflow_runs.log(["test/cats@3"]))
    expected = {"test/cats": {1: ["r-1", "r-2"], 2: ["r-2"]}}
    assert records.referenced_versions(settings()) == expected
    found = records._client_rows(
        f"sqlite:///{store}", "fieldbook.inputs"
    )  # the path used for MLflow servers
    assert sorted(run_id for run_id, _ in found) == ["r-1", "r-2"]


def test_cleanup_keeps_versions_used_by_runs(files: Path, mlflow_runs) -> None:
    ds = _cats(files)
    run = mlflow_runs.log(["test/cats@1"], run_id="r-1")
    with pytest.raises(DatasetError, match=r"v1 \(run r-1\).*delete those runs"):
        ds.delete_versions([1], grace=NOW)
    with pytest.raises(DatasetError, match=r"runs used v1 \(run r-1\)"):
        ds.delete(grace=NOW)
    report = prune(keep_latest=1, retention=NOW, grace=NOW)
    assert report.versions == {"test/cats": [2]} and report.kept_for_runs == {"test/cats": [1]}
    mlflow_runs.delete(run)  # releases the version
    ds.delete_versions([1], grace=NOW)
    assert [v.version for v in ds.log()] == [3]


def test_unreadable_run_records_stop_cleanup(files: Path, store: Path) -> None:
    ds = _cats(files)
    store.write_bytes(b"not a database")
    with pytest.raises(records.RunRecordsError, match="nothing was deleted"):
        ds.delete_versions([1], grace=NOW)
    with pytest.raises(records.RunRecordsError):
        prune(grace=NOW)
    assert len(ds.log()) == 3


def test_remote_store_needs_mlflow(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _cats(files)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.setitem(sys.modules, "mlflow.entities", None)
    with pytest.raises(records.RunRecordsError, match="MLflow is not installed"):
        ds.delete_versions([1], grace=NOW)


def test_no_run_records(files: Path) -> None:
    ds = _cats(files)
    assert records.referenced_versions(settings()) == {}
    ds.delete_versions([1], grace=NOW)


def test_unreadable_tracking_store_error_hides_the_password(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = "postgresql://bob:s3cret@db.example.com/mlflow"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    class Unreachable:  # fails at once, quoting the address, as database errors can
        def __init__(self, tracking_uri):
            raise ConnectionError(f"could not connect to {tracking_uri}")

    import mlflow.tracking

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", Unreachable)
    with pytest.raises(Exception) as error:
        records.referenced_versions(settings())
    assert "s3cret" not in str(error.value)
