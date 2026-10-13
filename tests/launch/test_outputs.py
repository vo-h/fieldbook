"""The files a run wrote, kept (fieldbook/launch/outputs.py)."""

import json
import os
import time
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

from fieldbook import history, storage
from fieldbook.config import settings
from fieldbook.launch import outputs

RUN = "20260101-000000-abcdef"


def aged(repo: Path) -> None:
    """Make the repository's files an hour old: there well before the run."""
    old = time.time() - 3600
    for path in repo.rglob("*"):
        os.utime(path, (old, old))


def test_new_and_changed_files_are_kept_and_the_rest_named(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIELDBOOK_OUTPUT_MAX_FILE_MB", "1")
    cfg, files = settings(), storage.fs()
    out = tmp_path / "out"
    out.mkdir()
    aged(repo)
    collector = outputs.Collector(RUN, cfg, {"project": repo, "output": out}, time.time())
    assert collector.sync() == {"kept": [], "left_out": []}
    assert not files.exists(f"runs/{RUN}/files.json")
    (repo / "logs").mkdir()
    (repo / "logs" / "train.log").write_text("epoch 1\n")
    (repo / "data" / "preds.csv").write_text("a,b\n")  # in an ignored folder: still an output
    (repo / "big.ckpt").write_bytes(b"0" * (2 * outputs.MB))
    for skipped in (".git/x", ".fieldbook/y", "__pycache__/z.pyc", ".venv/lib.py"):
        (repo / skipped).parent.mkdir(exist_ok=True)
        (repo / skipped).write_text("not an output")
    (out / "plot.txt").write_text("a plot")
    kept = collector.sync()
    assert [f["path"] for f in kept["kept"]] == [
        "output/plot.txt", "project/data/preds.csv", "project/logs/train.log",
    ]  # fmt: skip
    assert kept["left_out"] == [
        {"path": "project/big.ckpt", "size": 2 * outputs.MB, "why": "over 1 MB"}
    ]
    assert files.cat_file(f"runs/{RUN}/files/project/logs/train.log") == b"epoch 1\n"
    assert outputs.manifest(RUN, cfg) == kept and outputs.manifest("none", cfg)["kept"] == []
    (repo / "logs" / "train.log").write_text("epoch 1\nepoch 2\n")  # changed again: copied again
    collector.sync()
    assert files.cat_file(f"runs/{RUN}/files/project/logs/train.log") == b"epoch 1\nepoch 2\n"
    assert outputs.fetch(RUN, "output/plot.txt", tmp_path, cfg).read_text() == "a plot"


def test_a_runs_total_is_limited_and_what_is_past_it_named(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(outputs, "MAX_FILES", 2)
    aged(repo)
    for name in "abc":
        (repo / f"{name}.txt").write_text(name)
    collector = outputs.Collector(RUN, settings(), {"project": repo}, time.time())
    kept = collector.sync()
    assert [f["path"] for f in kept["kept"]] == ["project/a.txt", "project/b.txt"]
    assert kept["left_out"] == [{"path": "project/c.txt", "size": 1, "why": "past the run's total"}]


def test_fieldbook_run_keeps_what_the_script_wrote(
    repo: Path, gitleaks: Path, tmp_path: Path
) -> None:
    (repo / "write.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path('result.txt').write_text('0.91')\n"
        "pathlib.Path(os.environ['PLATFORM_OUTPUT_DIR'], 'elsewhere.txt').write_text('kept too')\n"
    )
    aged(repo)
    assert run_cli("run", "write.py")[0] == 0
    (run,) = history.runs(settings())
    code, out = run_cli("log", "files", run.run_id)
    assert code == 0 and "project/result.txt" in out and "output/elsewhere.txt" in out
    assert "write.py" not in out  # there before the run
    kept = json.loads(run_cli("log", "files", run.run_id, "--json")[1])
    assert {f["path"] for f in kept["kept"]} == {"project/result.txt", "output/elsewhere.txt"}
    code, out = run_cli("log", "files", run.run_id, "project/result.txt", "--to", str(tmp_path))
    assert code == 0 and (tmp_path / "result.txt").read_text() == "0.91"
    code, out = run_cli("log", "files", run.run_id, "project/nope.txt")
    assert code == 1 and "is not a kept file" in out


def test_files_are_not_kept_when_switched_off_or_in_smoke_runs(
    repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "write.py").write_text("open('result.txt', 'w').write('x')\n")
    assert run_cli("run", "--smoke", "write.py")[0] == 0
    monkeypatch.setenv("FIELDBOOK_KEEP_OUTPUTS", "false")
    assert run_cli("run", "write.py")[0] == 0
    for run in history.runs(settings()):
        assert outputs.manifest(run.run_id, settings()) == {"kept": [], "left_out": []}
