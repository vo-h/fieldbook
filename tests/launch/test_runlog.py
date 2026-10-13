"""A run's output kept in pieces (fieldbook/launch/runlog.py)."""

import gzip
import sys
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

from fieldbook import history, storage
from fieldbook.config import settings
from fieldbook.launch import runlog

RUN = "20260101-000000-abcdef"


def test_output_is_kept_in_pieces_and_joined_at_the_end(repo: Path) -> None:
    cfg, files = settings(), storage.fs()
    log = runlog.RunLog(RUN, cfg, secrets=("hunter2secret", "abc"))
    assert runlog.read(RUN, cfg) is None
    log.feed(b"epoch 1\nloss 0.5\nhalf a li")
    log.flush()
    assert files.ls(f"runs/{RUN}/log-output", detail=False) == [f"runs/{RUN}/log-output/000001.gz"]
    assert runlog.read(RUN, cfg) == ["epoch 1", "loss 0.5"]  # readable while it runs
    log.feed(b"ne\r\n")  # a line in two reads, with a terminal's line ending
    log.feed(b"\r 10%|#   |\r 50%|##  |\r100%|####|\n")  # a progress bar: its last state
    log.feed(b"\x1b[31mred\x1b[0m token=hunter2secret abc\n")  # colour dropped, secret masked
    log.feed(b"no newline at the end")
    log.flush()
    assert len(files.ls(f"runs/{RUN}/log-output")) == 2
    log.close()
    assert not files.exists(f"runs/{RUN}/log-output")  # joined: gzip members end to end
    joined = gzip.decompress(files.cat_file(f"runs/{RUN}/output.log.gz")).decode()
    assert (
        joined.splitlines()
        == runlog.read(RUN, cfg)
        == [
            "epoch 1",
            "loss 0.5",
            "half a line",
            "100%|####|",
            "red token=*** abc",
            "no newline at the end",
        ]
    )  # fmt: skip (a secret under 4 characters is not masked: it would mask ordinary text)


def test_a_progress_bar_without_a_line_end_does_not_grow(repo: Path) -> None:
    log = runlog.RunLog(RUN, settings())
    for step in range(5000):
        log.feed(f"\rstep {step}".encode())
    assert log._partial == b"step 4999"


def test_long_output_keeps_its_start_and_its_end(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runlog, "MAX_BYTES", 200)
    monkeypatch.setattr(runlog, "TAIL_LINES", 3)
    log = runlog.RunLog(RUN, settings())
    for step in range(40):
        log.feed(f"line {step} {'x' * (step % 7)} {hash(str(step))}\n".encode())
        log.flush()
    log.close()
    lines = runlog.read(RUN, settings())
    assert lines[0].startswith("line 0 ") and [line.split()[1] for line in lines[-3:]] == [
        "37", "38", "39",
    ]  # fmt: skip
    assert "lines left out (output over the size limit)" in lines[-4]
    assert len(lines) < 40


def test_a_commands_output_goes_on_as_before_and_to_the_log(repo: Path, capfd) -> None:
    log = runlog.RunLog(RUN, settings())
    script = "import sys; print('to out'); print('to err', file=sys.stderr); sys.exit(3)"
    copied = runlog.popen([sys.executable, "-c", script], log)
    assert copied.process.wait() == 3
    copied.drain()
    log.close()
    seen = capfd.readouterr()
    assert seen.out == "to out\n" and seen.err == "to err\n"  # each to its own stream
    assert sorted(runlog.read(RUN, settings())) == ["to err", "to out"]
    plain = runlog.popen([sys.executable, "-c", "print('not kept')"], None)
    assert plain.process.wait() == 0 and capfd.readouterr().out == "not kept\n"


def test_fieldbook_run_keeps_the_runs_output(repo: Path, gitleaks: Path) -> None:
    (repo / "loud.py").write_text(
        "import sys\nprint('hello from the run')\nprint('and a warning', file=sys.stderr)\n"
    )
    assert run_cli("run", "loud.py")[0] == 0
    (run,) = history.runs(settings())
    code, out = run_cli("log", "output", run.run_id)
    assert code == 0 and "hello from the run" in out and "and a warning" in out
    assert run_cli("log", "output", run.run_id, "--tail", "1")[1].count("\n") == 1
    code, out = run_cli("log", "output", run.run_id, "--follow")  # it has ended: prints, returns
    assert code == 0 and "hello from the run" in out
    code, out = run_cli("log", "output", run.run_id, "--launch")
    assert code == 1 and "no launch is kept" in out
