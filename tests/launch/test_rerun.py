"""`fieldbook run --rerun`: an earlier run again from its record (fieldbook/launch/rerun.py)."""

import json
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

import fieldbook as fb
from fieldbook import history, notes, storage
from fieldbook.config import settings
from fieldbook.launch import rerun


def only_run() -> str:
    (run,) = history.runs(settings())
    return run.run_id


def test_reads_are_pinned_to_the_versions_of_the_original_run(
    repo: Path, files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with fb.dataset("raw").edit() as edit:
        edit.add("new.txt", str(files / "3.jpg"))
    assert fb.dataset("raw").version == 2
    monkeypatch.setenv("FIELDBOOK_PIN_INPUTS", json.dumps({"raw": 1}))
    assert fb.dataset("raw").version == 1  # a read without a version
    assert fb.dataset("raw", version=2).version == 2  # the script's own choice wins
    assert "new.txt" not in fb.dataset("raw").files()["path"].to_pylist()
    with fb.dataset("raw").edit() as edit:  # edits start from the latest, as always
        edit.remove("new.txt")
    assert fb.dataset("raw", version=3).version == 3


def test_rerun_runs_the_recorded_code_and_command_again(repo: Path, gitleaks: Path) -> None:
    assert run_cli("run", "clean.py")[0] == 0
    first = only_run()
    (repo / "clean.py").write_text("raise SystemExit('the code has moved on')\n")
    (repo / "args.txt").unlink()  # written by the first run
    code, out = run_cli("run", "--rerun", first[:15])
    assert code == 0, out  # the snapshot's code ran, not the working tree's
    assert "reading the versions the original read: raw@1" in out
    # it started from the empty dataset the original started from, saved to a scratch area,
    # and what it saved is what the original saved
    assert f"rerun of {first} as " in out and "reproduced: clean@1" in out
    assert fb.dataset("clean").version == 1 and storage.fs().ls("scratch") == []
    assert not (repo / "args.txt").exists()  # it ran in a checkout, which is removed
    (note,) = notes.of([first])[first]  # the outcome is on the original's record
    assert note.text.startswith("rerun as ") and note.text.endswith("reproduced: clean@1")
    code, out = run_cli("run", "--rerun", first, "--json", "3")  # one change: an argument added
    assert code == 3 and "failed (exit code 3)" in out
    code, out = run_cli("run", "--rerun", "19990101")
    assert code == 1 and "no run" in out


def test_a_rerun_that_saves_something_else_is_reported(repo: Path, gitleaks: Path) -> None:
    (repo / "pick.py").write_text(
        "import sys\nimport fieldbook as fb\n"
        "with fb.dataset('picked').edit() as e:\n"
        "    e.copy_from(fb.dataset('raw'), prefix=sys.argv[-1])\n"
    )
    assert run_cli("run", "pick.py", "a/")[0] == 0
    first = only_run()
    code, out = run_cli("run", "--rerun", first, "3.jpg")  # the last argument is the prefix
    assert code == 0, out
    assert "differs: picked@1 0 files changed, " in out and " missing" in out
    assert fb.dataset("picked").version == 1


def test_a_rerun_is_compared_with_the_original_and_noted(repo: Path, mlflow_runs) -> None:
    old, new, changed = (f"20260101-00000{i}-abcdef" for i in range(3))
    for run_id, loss in ((old, 0.25), (new, 0.25), (changed, 0.31)):
        logged = mlflow_runs.log(["test/raw@1"], run_id=run_id)
        mlflow_runs.client.log_metric(logged, "val_loss", loss)
        mlflow_runs.client.set_terminated(logged)
    original = history.find(old)
    same = rerun._compare(original, rerun.Result(0, new), settings())
    assert same.same == ["val_loss"] and not same.differs
    assert same.verdict == "reproduced: 1 metrics (val_loss)"
    differs = rerun._compare(original, rerun.Result(0, changed, changed=["--lr", "1"]), settings())
    assert differs.verdict == "differs: val_loss 0.25 -> 0.31"
    found = notes.of([old, new, changed])
    assert found[new][0].text == f"rerun of {old}: {same.verdict}"
    assert found[changed][0].text == f"rerun of {old} with --lr 1: {differs.verdict}"
    assert [n.text for n in found[old]] == [
        f"rerun as {new}: {same.verdict}",
        f"rerun as {changed} with --lr 1: {differs.verdict}",
    ]


def test_rerun_is_an_option_of_run(repo: Path) -> None:
    """`--rerun` takes its settings from the record: other options and a missing target
    are refused with a line, and a new run does not let unknown options through."""
    code, out = run_cli("run", "--rerun", "19990101", "--smoke")
    assert code == 1 and "--smoke cannot be used with --rerun" in out
    code, out = run_cli("run", "--rerun", "19990101", "--on", "remote", "--gpus", "A10G")
    assert code == 1 and "--on, --gpus cannot be used with --rerun" in out
    code, out = run_cli("run")
    assert code == 1 and "or --rerun RUN" in out
    code, out = run_cli("run", "--no-such-option", "clean.py")
    assert code == 1 and "no such option: --no-such-option" in out
    assert run_cli("rerun", "19990101")[0] == 2  # the old command is gone
