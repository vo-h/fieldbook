import os

import pytest

from fieldbook.launch import context
from fieldbook.launch.context import RunContext


def test_run_context_round_trips_through_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    run = RunContext(
        run_id="r1", code="abc", env="{}", seed=0, secrets=("A", "B"), deterministic=True,
        smoke=20, scratch="s1",
    )  # fmt: skip
    for name in run.to_env():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(os, "environ", {**os.environ, **run.to_env()})
    assert context.current() == run
    assert run.to_env()["PL_GLOBAL_SEED"] == "0" and run.to_env()["PL_TRAINER_FAST_DEV_RUN"] == "5"


def test_outside_a_run_the_context_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RunContext(run_id="x", seed=1, smoke=1, scratch="s").to_env():
        monkeypatch.delenv(name, raising=False)
    assert context.current() == RunContext()
    assert RunContext().to_env() == {}


def test_the_bootstrap_loads_almost_nothing() -> None:
    """C3: the bootstrap starts every run and every torchrun rank, so importing it must not
    load the rest of fieldbook or its dependencies."""
    import subprocess
    import sys

    code = (
        "import sys, fieldbook.launch.bootstrap; "
        "print(' '.join(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('fieldbook', 'pyarrow', 'pydantic', 'fsspec', 'lance', 'mlflow', 'yaml'))))"
    )
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True).stdout.split()  # fmt: skip
    assert loaded == [
        "fieldbook",
        "fieldbook.launch",
        "fieldbook.launch.autolog",
        "fieldbook.launch.autorecord",
        "fieldbook.launch.bootstrap",
        "fieldbook.launch.context",
        "fieldbook.launch.seeding",
    ]  # fmt: skip (autorecord, autolog: standard library only)
