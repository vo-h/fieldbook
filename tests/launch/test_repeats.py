"""Noticing that a run repeats an earlier one (fieldbook/launch/repeats.py)."""

from pathlib import Path

from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

import fieldbook as fb
from fieldbook import history, notes
from fieldbook.config import settings
from fieldbook.launch import repeats


def only_run() -> str:
    (run,) = history.runs(settings())
    return run.run_id


def test_the_fingerprint_is_the_same_however_the_script_is_started(tmp_path: Path) -> None:
    (tmp_path / "jobs").mkdir()
    same = {"code": "c0de", "environment": "pk9", "repo": tmp_path}
    mark = repeats.fingerprint(command=["python", "train.py", "--lr", "1"],
                               workdir=tmp_path / "jobs", **same)  # fmt: skip
    plain = repeats.fingerprint(command=[str(tmp_path / "jobs" / "train.py"), "--lr", "1"],
                                workdir=tmp_path / "jobs", **same)  # fmt: skip
    assert mark == plain and len(mark) == 32  # `fieldbook run train.py` and `python /abs/train.py`
    others = [
        repeats.fingerprint(command=["train.py", "--lr", "2"], workdir=tmp_path / "jobs", **same),
        repeats.fingerprint(command=["train.py", "--lr", "1"], workdir=tmp_path, **same),
        repeats.fingerprint(command=["train.py", "--lr", "1"], workdir=tmp_path / "jobs",
                            seed=7, **same),
        repeats.fingerprint(command=["train.py", "--lr", "1"], workdir=tmp_path / "jobs",
                            **(same | {"code": "0ther"})),
        repeats.fingerprint(command=["train.py", "--lr", "1"], workdir=tmp_path / "jobs",
                            **(same | {"environment": "pk10"})),
    ]  # fmt: skip
    assert len({mark, *others}) == 6


def test_a_repeated_run_is_reported_until_its_data_changes(
    repo: Path, gitleaks: Path, files: Path
) -> None:
    code, out = run_cli("run", "clean.py")
    assert code == 0 and "repeats an earlier run" not in out
    first = only_run()
    notes.add(first, "baseline copy", verdict="good")
    code, out = run_cli("run", "clean.py")  # it still runs: here a repeat costs only time
    assert "this repeats an earlier run" in out and f"run {first}" in out
    assert "saved clean@1" in out and "note: [good] baseline copy" in out
    assert "repeats an earlier run" not in run_cli("run", "clean.py", "--allow-repeat")[1]
    assert "repeats an earlier run" not in run_cli("run", "clean.py", "0")[1]  # another command
    assert "repeats an earlier run" not in run_cli("run", "--smoke", "clean.py")[1]
    with fb.dataset("raw").edit() as edit:  # the data it read has moved on
        edit.add("new.txt", str(files / "3.jpg"))
    assert "repeats an earlier run" not in run_cli("run", "clean.py")[1]
