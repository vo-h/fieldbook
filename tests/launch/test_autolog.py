"""MLflow autologging switched on for training frameworks (fieldbook/launch/autolog.py)."""

import sys
import types
from pathlib import Path

import pytest

from fieldbook.config import settings
from fieldbook.launch import autolog, runenv


@pytest.fixture
def flavor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    """A training library `fakeboost` on the path, and the MLflow module that covers it."""
    calls: list[dict] = []
    module = types.ModuleType("fakeflavor")
    module.autolog = lambda log_models=True, silent=False: calls.append(  # type: ignore[attr-defined]
        {"log_models": log_models, "silent": silent}
    )
    monkeypatch.setitem(sys.modules, "fakeflavor", module)
    (tmp_path / "fakeboost.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(autolog, "FRAMEWORKS", {"fakeboost": "fakeflavor"})
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.delitem(sys.modules, "fakeboost", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x.db")
    monkeypatch.delenv("FIELDBOOK_AUTOLOG", raising=False)
    return calls


def test_autologging_starts_when_the_framework_is_imported(flavor: list[dict]) -> None:
    autolog.enable()
    assert flavor == []  # a script that never imports the framework pays nothing
    import fakeboost  # noqa: F401

    # only the options this flavor accepts; models are not uploaded
    assert flavor == [{"log_models": False, "silent": True}]


def test_autologging_is_off_when_switched_off_or_nothing_logs(
    flavor: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIELDBOOK_AUTOLOG", "0")
    autolog.enable()
    monkeypatch.delenv("FIELDBOOK_AUTOLOG")
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    autolog.enable()
    import fakeboost  # noqa: F401

    assert flavor == []


def test_a_failing_autolog_does_not_stop_the_script(
    flavor: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys.modules["fakeflavor"], "autolog", lambda: 1 / 0)
    autolog.enable()
    import fakeboost  # noqa: F401


def test_the_autolog_setting_reaches_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    assert runenv.settings_env(settings())["FIELDBOOK_AUTOLOG"] == "1"
    monkeypatch.setenv("FIELDBOOK_AUTOLOG", "false")
    assert runenv.settings_env(settings())["FIELDBOOK_AUTOLOG"] == "0"
