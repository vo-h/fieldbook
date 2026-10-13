"""`fb.init()` and `fb.seed()` (fieldbook/launch/inprocess.py)."""

import json
import os
from pathlib import Path

import pytest
from helpers import (
    SECRET,
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
)

import fieldbook as fb
from fieldbook.code import snapshot
from fieldbook.tracking import setup


@pytest.fixture
def restore_process_state(monkeypatch: pytest.MonkeyPatch):
    import sys

    from fieldbook.launch import inprocess

    saved = dict(os.environ)
    monkeypatch.setattr(inprocess, "_started_here", False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_init_starts_a_run_in_this_process(
    repo: Path, gitleaks: Path, restore_process_state, monkeypatch: pytest.MonkeyPatch
) -> None:
    import random

    monkeypatch.setattr(setup, "mlflow_env", lambda cfg, experiment, *_: {})
    run = fb.init(seed=3)
    assert random.random() == random.Random(3).random()
    assert run.code == snapshot.tree_hash(repo) and os.environ["PLATFORM_RUN_ID"] == run.run_id
    with fb.dataset("clean").edit() as edit:
        edit.copy_from(fb.dataset("raw"))
    info = fb.dataset("clean").log()[0]
    assert (info.run_id, info.code, json.loads(info.env)["seed"]) == (run.run_id, run.code, 3)
    assert fb.init(seed=4).run_id != run.run_id  # calling again starts a new run


def test_init_inside_fieldbook_run_returns_the_launched_run(
    restore_process_state, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PLATFORM_RUN_ID", "launched")
    monkeypatch.setenv("PLATFORM_CODE", "abc")
    monkeypatch.setenv("PLATFORM_SEED", "9")
    from fieldbook.launch.inprocess import Run

    assert fb.init() == Run("launched", "abc", 9)


def test_init_loads_env_file(
    repo: Path, gitleaks: Path, restore_process_state, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup, "mlflow_env", lambda *args: {})
    (repo / ".env").write_text(f"HF_TOKEN={SECRET}\nSHARED=from-file\n")
    os.environ["SHARED"] = "already-set"
    fb.init()
    assert os.environ["HF_TOKEN"] == SECRET and os.environ["SHARED"] == "already-set"


def test_seed_per_trial() -> None:
    import random

    first = fb.seed(7, key={"lr": 0.1, "seed": 7})
    a = random.random()
    assert fb.seed(7, key={"seed": 7, "lr": 0.1}) == first and random.random() == a  # same trial
    assert fb.seed(7, key={"lr": 0.01, "seed": 7}) != first  # another trial
    assert fb.seed(7) == 7
