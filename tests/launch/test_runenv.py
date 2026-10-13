"""What a run's process is given (fieldbook/launch/runenv.py)."""

from pathlib import Path

import pytest

from fieldbook import storage
from fieldbook.clouds import aws
from fieldbook.config import settings
from fieldbook.launch import runenv


def test_runs_sign_s3_requests_for_the_buckets_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aws, "bucket_region", lambda bucket: "us-west-2")  # no network
    monkeypatch.setattr(storage, "_regions", {})  # other tests cached this bucket's region
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://bucket/fieldbook")
    region = storage.s3_region(settings())
    assert aws.default_region_env({}, region) == {"AWS_DEFAULT_REGION": "us-west-2"}
    assert aws.default_region_env({"AWS_REGION": "eu-west-1"}, region) == {}  # yours
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", str(Path.cwd()))  # local storage
    assert aws.default_region_env({}, storage.s3_region(settings())) == {}


def test_every_kind_of_run_gets_the_resolved_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aws, "bucket_region", lambda bucket: "us-west-2")
    monkeypatch.setattr(storage, "_regions", {})
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://bucket/fieldbook")
    monkeypatch.setenv("FIELDBOOK_TRACKING_URI", "postgresql://db.example.com/mlflow")
    here = runenv.settings_env(settings())  # local runs and fb.init()
    assert (
        here["FIELDBOOK_STORAGE_ROOT"] == "s3://bucket/fieldbook"
        and here["FIELDBOOK_STORAGE_REGION"] == "us-west-2"
    )
    assert here["FIELDBOOK_STATE_DIR"] and here["FIELDBOOK_TRACKING_URI"].startswith(
        "postgresql://"
    )
    there = runenv.settings_env(settings(), portable=True)  # cloud machines
    assert "FIELDBOOK_STATE_DIR" not in there and "FIELDBOOK_TRACKING_URI" not in there
