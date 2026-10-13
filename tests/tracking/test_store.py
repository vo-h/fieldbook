"""Where runs log (fieldbook/tracking/store.py)."""

from pathlib import Path

import pytest

from fieldbook import credentials
from fieldbook.config import settings
from fieldbook.tracking import setup
from fieldbook.tracking import store as tracking_store


def test_shared_tracking_database_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    uri = "postgresql+psycopg2://alice:s3cret@db.example.com:5432/mlflow"
    assert credentials.display_uri(uri) == "postgresql+psycopg2://db.example.com:5432/mlflow"
    assert tracking_store.is_database(uri) and tracking_store.is_database("sqlite:///x.db")
    assert not tracking_store.is_database("https://mlflow.example.com")
    monkeypatch.chdir(tmp_path)
    assert tracking_store.configured_uri() is None
    assert (
        tracking_store.tracking_uri(settings())
        == f"sqlite:///{tracking_store.mlflow_db(settings())}"
    )
    (tmp_path / ".env").write_text(f"MLFLOW_TRACKING_URI={uri}\n")
    assert tracking_store.configured_uri() == uri  # a secret: read from .env
    server = "https://mlflow.example.com"  # an MLflow server sets up its own experiments
    assert setup.mlflow_env(settings(), "exp", {}, "python", server) == {
        "MLFLOW_TRACKING_URI": server,  # runs must still be told where to log
        "MLFLOW_EXPERIMENT_NAME": "exp",
    }
