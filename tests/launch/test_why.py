"""`fieldbook log why`: how a run ended and what went wrong (fieldbook/launch/why.py)."""

import json
import time
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

from fieldbook import history
from fieldbook.config import settings
from fieldbook.launch import records, runlog, why

TRACE = """Epoch 3: 100%
Traceback (most recent call last):
  File "train.py", line 10, in <module>
    main()
  File "train.py", line 7, in main
    step(batch)
  File "train.py", line 3, in step
    {raise_line}
{error}
"""


def ended(exit_code: int, output: str, *, seconds_ago: float = 100, limit: int | None = None,
          launch: str | None = None, ran: float | None = None) -> str:  # fmt: skip
    cfg = settings()
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() - seconds_ago))
    run_id += f"-{exit_code:06x}"[:7]
    records.start(cfg, run_id, {"started": time.time() - seconds_ago, "limit": limit})
    for name, text in (("output", output), ("launch", launch)):
        if text:
            log = runlog.RunLog(run_id, cfg, name=name)
            log.feed(text.encode())
            log.close()
    if launch:
        records.end(cfg, run_id, exit_code, phase="launch")
    elif exit_code >= 0:
        records.end(cfg, run_id, exit_code, **({"seconds": ran} if ran is not None else {}))
    return run_id


@pytest.mark.parametrize(
    ("exit_code", "raise_line", "error", "cause"),
    [
        (1, "x = torch.zeros(n).cuda()",
         "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB",
         "it ran out of GPU memory"),
        (1, "big = np.zeros(n)", "MemoryError", "it ran out of memory"),
        (1, "import timm", "ModuleNotFoundError: No module named 'timm'",
         "the package 'timm' is not installed in the run's environment"),
        (1, "open(path)", "FileNotFoundError: [Errno 2] No such file or directory: 'cfg.yaml'",
         "a file it needs was not found"),
        (1, "assert loss > 0", "ValueError: loss went negative",
         "it raised ValueError: loss went negative"),
    ],
)  # fmt: skip
def test_the_cause_is_read_from_the_output(
    repo: Path, exit_code: int, raise_line: str, error: str, cause: str
) -> None:
    run_id = ended(exit_code, TRACE.format(raise_line=raise_line, error=error))
    verdict = why.why(run_id, settings())
    assert (verdict.status, verdict.cause, verdict.exit_code) == ("FAILED", cause, exit_code)
    assert verdict.detail[-1] == error and 90 < verdict.seconds < 130
    if "raised" in cause:  # the last two frames, not the whole traceback
        assert verdict.detail[0].startswith("Traceback") and len(verdict.detail) == 6


def test_causes_known_without_a_traceback(repo: Path) -> None:
    cfg = settings()
    limited = why.why(ended(143, "epoch 9\n", seconds_ago=3700, limit=3600), cfg)
    assert limited.cause == "it reached its time limit (1.0 h) and was stopped"
    assert why.why(ended(143, "epoch 9\n", limit=3600), cfg).cause == "it was terminated (SIGTERM)"
    # a slow launch does not count: the machine measured how long the process itself ran
    early = why.why(ended(143, "epoch 1\n", seconds_ago=3700, limit=3600, ran=600), cfg)
    assert early.cause == "it was terminated (SIGTERM)" and early.seconds == 600
    assert "usually by the system when memory runs out" in why.why(ended(137, "x\n"), cfg).cause
    assert why.why(ended(130, "x\n"), cfg).cause == "it was interrupted (Ctrl-C)"
    plain = why.why(ended(2, "usage: train.py [-h]\nerror: bad --lr\n"), cfg)
    assert plain.cause == "it exited with code 2" and plain.detail[-1] == "error: bad --lr"
    assert why.why(ended(0, "done\n"), cfg).cause == "it succeeded"
    lost = why.why(ended(-1, "epoch 2\n", seconds_ago=9000), cfg)  # no end, no sign of life
    assert lost.status == "KILLED" and "removed from outside" in lost.cause
    assert lost.detail == ["epoch 2"]
    assert "was not started with `fieldbook run`" in why.why("19990101-000000-000000", cfg).cause


def test_a_failed_launch_is_explained_from_the_launch_log(repo: Path) -> None:
    cfg = settings()
    none_free = "Trying RTX4090...\nsky.exceptions.ResourcesUnavailableError: Failed to provision"
    verdict = why.why(ended(1, "", launch=none_free), cfg)
    assert verdict.cause == (
        "the launch failed: no machine with the requested resources was available"
    )
    assert verdict.detail[0] == "Trying RTX4090..."
    slow = (
        "launch of fieldbook-x did not finish within 15 minutes (launch_timeout); tearing it down"
    )
    assert "did not finish in time" in why.why(ended(1, "", launch=slow), cfg).cause
    assert why.why(ended(1, "", launch="boom"), cfg).cause == (
        "the launch failed before the run started"
    )


def test_the_why_command(repo: Path, gitleaks: Path) -> None:
    (repo / "broken.py").write_text("import fieldbook\nraise RuntimeError('lr must be positive')\n")
    assert run_cli("run", "broken.py")[0] == 1
    (run,) = history.runs(settings())
    code, out = run_cli("log", "why", run.run_id[:15])
    assert code == 0 and f"run {run.run_id}  failed after " in out and "(exit code 1)" in out
    assert "cause: it raised RuntimeError: lr must be positive" in out
    assert 'File "broken.py", line 2' in out
    verdict = json.loads(run_cli("log", "why", run.run_id, "--json")[1])
    assert verdict["cause"] == "it raised RuntimeError: lr must be positive"
    assert verdict["exit_code"] == 1 and verdict["detail"][-1].startswith("RuntimeError")
