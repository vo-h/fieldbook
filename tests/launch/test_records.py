"""A run's record of starting, living, and ending (fieldbook/launch/records.py), and what
`fieldbook log wait` makes of it."""

import json
import time
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

from fieldbook import history, storage
from fieldbook.cli import history as commands
from fieldbook.config import settings
from fieldbook.launch import records

NOW = "20260101-000000-abcdef"


def run_id(seconds_ago: float) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() - seconds_ago)) + "-abcdef"


def test_a_runs_state_comes_from_one_listing(repo: Path) -> None:
    cfg = settings()
    assert records.states(cfg) == {}
    done, failed, going, launching, lost, gone = (run_id(s) for s in (50, 40, 30, 20, 10, 9000))
    for each in (done, failed, going, launching, lost, gone):
        records.start(cfg, each, {"command": "train.py", "where": "cloud"})
    records.end(cfg, done, 0)
    records.end(cfg, failed, 3)
    records.beat(cfg, going)
    records.beat(cfg, lost)
    states = records.states(cfg)
    stale = time.time() + records.STALE + 1  # as seen once signs of life have stopped
    assert states[done].status() == "FINISHED" and states[failed].status() == "FAILED"
    assert states[failed].exit_code == 3 and states[failed].ended is not None
    assert states[going].status() == "RUNNING" and states[lost].status(stale) == "KILLED"
    assert states[launching].status() == "LAUNCHING"  # a cloud machine not up yet
    assert states[gone].status() == "KILLED"  # no sign of life, long after launch
    assert set(records.states(cfg, done[:15])) >= {done}  # by prefix
    assert records.read(cfg, [done, "nope"]) == {
        done: {"run_id": done, "command": "train.py", "where": "cloud"}
    }
    records.start(cfg, done, {"command": "other"}, keep=True)  # the machine's side keeps it
    assert records.read(cfg, [done])[done]["command"] == "train.py"


def test_every_fieldbook_run_is_listed_with_how_it_ended(repo: Path, gitleaks: Path) -> None:
    (repo / "quiet.py").write_text("import sys\nsys.exit(int(sys.argv[1]))\n")
    assert run_cli("run", "quiet.py", "0")[0] == 0  # saves nothing, logs nothing
    assert run_cli("run", "quiet.py", "4")[0] == 4
    # by how they ended: started in the same second, the two are listed in either order
    failed, finished = sorted(history.runs(settings()), key=lambda run: -run.exit_code)
    assert (finished.status, finished.exit_code) == ("FINISHED", 0)
    assert (failed.status, failed.exit_code) == ("FAILED", 4) and failed.ended is not None
    assert failed.command == ""  # details are read only for the runs shown
    assert history.find(failed.run_id).command == "quiet.py 4"
    out = run_cli("log")[1].splitlines()
    assert "quiet.py 4" in out[1] and "failed" in out[1] and "finished" in out[2]
    (shown,) = json.loads(run_cli("log", "--json", "-n", "1")[1])
    assert shown["exit_code"] == 4 and shown["where"] == "local" and shown["ended"]
    assert "quiet.py" in run_cli("log", "--author", settings().author or "")[1]


def test_wait_returns_how_the_run_ended(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = settings()
    monkeypatch.setattr(commands, "WAIT_STEP", 0.05)
    code, out = run_cli("log", "wait")
    assert code == 1 and "no run to wait for" in out
    going = run_id(5)
    records.start(cfg, going, {"command": "train.py", "where": "local"})
    records.beat(cfg, going)
    code, out = run_cli("log", "wait", going, "--timeout", "0.002")  # still going
    assert code == 124 and "running" in out and "gave up waiting" in out
    import threading

    threading.Timer(0.3, records.end, (cfg, going, 2)).start()
    code, out = run_cli("log", "wait")  # no ID: the run that is going
    assert code == 1 and f"{going}  failed  exit code 2" in out and "fieldbook log why" in out
    done = run_id(1)
    records.start(cfg, done, {"command": "train.py", "where": "local"})
    records.end(cfg, done, 0)
    code, out = run_cli("log", "wait", done, "--json")
    assert code == 0 and json.loads(out)["status"] == "FINISHED"
    assert json.loads(out)["timed_out"] is False
    assert storage.fs(cfg).exists(f"runs/{done}.end.0")
