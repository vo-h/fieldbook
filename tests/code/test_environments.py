"""The project's Python environment (fieldbook/code/environments.py)."""

import os
from pathlib import Path

import pytest
from helpers import no_mlflow  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)

from fieldbook.code import environments


def test_interpreter_answers_are_kept_until_packages_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C5: asked once per interpreter; asked again after a package is installed."""
    venv = tmp_path / "venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (venv / "bin").mkdir()
    python = venv / "bin" / "python"
    python.write_text("")
    asked = []
    monkeypatch.setattr(environments, "_ask", lambda py: asked.append(py) or {"python": "3.12.1"})
    assert environments.query_interpreter(str(python)) == {"python": "3.12.1"}
    assert environments.query_interpreter(str(python)) == {"python": "3.12.1"}
    assert len(asked) == 1
    import time

    (site / "newpkg-1.0.dist-info").mkdir()  # installing a package changes the folder
    later = time.time() + 5
    os.utime(site, (later, later))
    environments.query_interpreter(str(python))
    assert len(asked) == 2
    assert environments._fingerprint(tmp_path / "no-env" / "bin" / "python") is None
