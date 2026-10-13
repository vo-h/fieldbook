"""`fieldbook log ui` (fieldbook/cli/experiments.py), without starting MLflow."""

import pytest
from typer.testing import CliRunner

from fieldbook.cli import app
from fieldbook.code import environments
from fieldbook.tracking import costs, server


def test_opening_the_ui_records_cloud_costs_in_the_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D3: the costs of finished cloud runs are recorded when the UI opens, in the background,
    so the UI does not wait for them."""
    calls = []
    monkeypatch.setattr(environments, "query_interpreter", lambda python: {"mlflow_sql": True})
    monkeypatch.setattr(
        costs, "record_in_background", lambda cfg, report: calls.append(("costs", cfg.project))
    )
    monkeypatch.setattr(
        server, "serve", lambda cfg, python, open_browser: calls.append(("ui", open_browser)) or 0
    )
    result = CliRunner().invoke(app, ["log", "ui", "--no-open"])
    assert result.exit_code == 0, result.output
    assert calls == [("costs", "test"), ("ui", False)]  # costs started first, then the UI


def test_the_ui_needs_full_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environments, "query_interpreter", lambda python: {"mlflow": True})
    result = CliRunner().invoke(app, ["log", "ui", "--no-open"])
    assert result.exit_code == 1 and "pip install 'fieldbook[tracking]'" in result.output
