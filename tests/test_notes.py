"""Notes on runs (fieldbook/notes.py; `fieldbook log note`, `fb.note()`)."""

import json
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

import fieldbook as fb
from fieldbook import history, notes
from fieldbook.config import settings


def only_run() -> str:
    (run,) = history.runs(settings())
    return run.run_id


def test_notes_are_kept_with_the_run_and_shown_in_the_history(repo: Path, gitleaks: Path) -> None:
    assert run_cli("run", "clean.py")[0] == 0
    run = only_run()
    code, out = run_cli("log", "note", run[:15], "copies everything; fine as a baseline")
    assert code == 0 and "copies everything" in out
    code, out = run_cli("log", "note", run, "superseded by the dedup step", "--verdict", "bad")
    assert code == 0 and "[bad]" in out.splitlines()[0]  # the latest verdict is the run's
    code, out = run_cli("log")
    assert "[bad] superseded by the dedup step" in out
    (record,) = json.loads(run_cli("log", "--json")[1])
    assert [n["text"] for n in record["notes"]] == [
        "copies everything; fine as a baseline",
        "superseded by the dedup step",
    ]
    assert record["verdict"] == "bad" and record["notes"][0]["verdict"] is None
    assert json.loads(run_cli("log", "note", run, "--json")[1])["verdict"] == "bad"
    assert run_cli("log", "--verdict", "good")[1].startswith("no runs yet")
    assert run in run_cli("log", "--verdict", "bad")[1]
    code, out = run_cli("log", "note", run, "x", "--verdict", "great")
    assert code == 1 and "one of good, bad, inconclusive" in out
    code, out = run_cli("log", "note", "19990101", "x")
    assert code == 1 and "no run" in out


def test_a_script_notes_its_own_run(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="needs a run"):
        fb.note("nothing to attach this to")
    monkeypatch.setenv("PLATFORM_RUN_ID", "20260101-000000-abcdef")
    fb.note("loss plateaus from epoch 4", verdict="inconclusive")
    (note,) = notes.of(["20260101-000000-abcdef"])["20260101-000000-abcdef"]
    assert (note.text, note.verdict) == ("loss plateaus from epoch 4", "inconclusive")
