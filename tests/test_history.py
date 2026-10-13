"""What has been run, and how two runs differ (fieldbook/history.py; `fieldbook log`)."""

import json
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

from fieldbook import history
from fieldbook.config import settings
from fieldbook.launch import budget

OTHER = """import sys
import fieldbook as fb
raw = fb.dataset("raw")
with fb.dataset("small").edit() as edit:
    edit.copy_from(raw, prefix=sys.argv[1])
"""


@pytest.fixture
def two_runs(repo: Path, gitleaks: Path, mlflow_runs) -> tuple[str, str]:
    """A data run (clean.py), then a second run of changed code that also logged to MLflow."""
    code, out = run_cli("run", "clean.py")
    assert code == 0, out
    (repo / "other.py").write_text(OTHER)
    (repo / "clean.py").write_text((repo / "clean.py").read_text() + "\n# tuned\n")
    code, out = run_cli("run", "other.py", "kept/")
    assert code == 0, out
    first, second = (r.run_id for r in reversed(history.runs(settings())))
    logged = mlflow_runs.log(["test/raw@1"], run_id=second, tags={"fieldbook.author": "tester"})
    mlflow_runs.client.log_param(logged, "lr", "0.1")
    mlflow_runs.client.log_metric(logged, "val_loss", 0.25)
    spent = budget.Launch(second, f"fieldbook-{second}", "aws", "tester", 0.0, 1.0, 3600)
    budget.record(spent, settings())  # as a cloud run: its cost is in the launch ledger
    budget.priced(second, 1.5, "aws-cloudtrail", settings())
    mlflow_runs.client.set_terminated(logged)
    return first, second


def test_runs_merges_dataset_versions_and_mlflow_runs(two_runs: tuple[str, str]) -> None:
    first, second = two_runs
    newest, oldest = history.runs(settings())
    assert (newest.run_id, oldest.run_id) == (second, first)
    assert oldest.outputs == ["clean@1"] and oldest.inputs == ["raw@1"]
    assert oldest.command == "clean.py" and oldest.status == "FINISHED"  # from its own record
    assert newest.outputs == ["small@1"] and newest.status == "FINISHED"
    assert newest.params == {"lr": "0.1"} and newest.metrics == {"val_loss": 0.25}
    assert newest.cost_usd == 1.5 and newest.author == "tester"
    assert history.find(second[:19]).run_id == second  # past the second: runs can share one
    with pytest.raises(history.HistoryError, match="matches 2 runs"):
        history.find(second[:4])  # both started today
    with pytest.raises(history.HistoryError, match="no run"):
        history.find("19990101")


def test_the_runs_command(two_runs: tuple[str, str]) -> None:
    first, second = two_runs
    code, out = run_cli("log")
    assert code == 0, out
    lines = out.splitlines()
    assert lines[0].split()[:3] == ["run", "started", "author"]
    assert lines[1].startswith(second) and "val_loss=0.25" in lines[1]
    assert "cost" not in lines[0]  # costs have one place: `fieldbook log costs`
    assert f"{second}" in run_cli("log", "costs")[1] and "$1.50" in run_cli("log", "costs")[1]
    assert lines[2].startswith(first) and "clean@1" in lines[2]
    code, out = run_cli("log", "--json", "--limit", "1")
    (record,) = json.loads(out)
    assert record["run_id"] == second and record["metrics"] == {"val_loss": 0.25}
    assert record["env"]["seed"] and record["started"].startswith("20")
    assert run_cli("log", "--author", "nobody")[1].startswith("no runs yet")


def test_diff_shows_what_changed_between_two_runs(two_runs: tuple[str, str]) -> None:
    first, second = two_runs
    result = history.diff(history.find(first), history.find(second))
    assert result["command"] == {"a": "clean.py", "b": "other.py kept/"}
    assert result["outputs"] == {"clean": {"a": "1", "b": None}, "small": {"a": None, "b": "1"}}
    assert "inputs" not in result and "packages" not in result  # the same in both
    assert result["environment"].keys() == {"seed"}
    assert result["params"] == {"lr": {"a": None, "b": "0.1"}}
    code = result["code"]
    assert code["added"] == ["other.py"] and code["modified"] == ["clean.py"]
    assert "+# tuned" in code["patch"] and code["patch_lines_cut"] == 0
    exit_code, out = run_cli("log", "diff", first[:19], second)
    assert exit_code == 0, out
    assert "command\n  a: clean.py\n  b: other.py kept/" in out
    assert "added: other.py" in out and "+# tuned" in out and "val_loss: — -> 0.25" in out
    exit_code, out = run_cli("log", "diff", first, second, "--json")
    assert json.loads(out)["code"]["added"] == ["other.py"]
    exit_code, out = run_cli("log", "diff", first, "nope")
    assert exit_code == 1 and "no run 'nope'" in out


def test_run_points_to_runs_for_its_subcommands(repo: Path) -> None:
    code, out = run_cli("run", "note", "abc", "text")
    assert code == 1 and "did you mean `fieldbook log note`?" in out
    code, out = run_cli("run", "diff", "a", "b")  # `diff` is a real command: it is run
    assert "did you mean" not in out


def test_runs_are_tables_to_query(two_runs: tuple[str, str]) -> None:
    import fieldbook as fb
    from fieldbook import notes

    first, second = two_runs
    notes.add(first, "baseline copy", verdict="good")
    notes.add(
        first, f"rerun as {second}: reproduced: clean@1"
    )  # as `fieldbook run --rerun` notes it
    runs = fb.runs()  # one row a run, newest first
    assert runs.column("run_id").to_pylist() == [second, first]
    rows = {row["run_id"]: row for row in runs.to_pylist()}
    assert rows[second]["metrics"] == [("val_loss", 0.25)] and rows[second]["cost_usd"] == 1.5
    assert rows[second]["params"] == [("lr", "0.1")] and rows[second]["status"] == "finished"
    assert rows[first]["verified"] and not rows[second]["verified"]
    assert rows[first]["verdict"] == "good" and rows[first]["outputs"] == ["clean@1"]
    best = fb.runs(query="""
        select r.run_id, metrics['val_loss'] as loss, d.dataset
        from runs r join data d using (run_id)
        where d.role = 'saved' and r.status = 'finished' order by loss nulls last
    """)  # fmt: skip
    assert best.to_pylist() == [
        {"run_id": second, "loss": 0.25, "dataset": "small"},
        {"run_id": first, "loss": None, "dataset": "clean"},
    ]
    assert fb.runs(query="select count(*) n from notes").to_pylist() == [{"n": 2}]
    assert fb.runs(query="select name, value from params").to_pylist() == [
        {"name": "lr", "value": "0.1"}
    ]
    with pytest.raises(history.HistoryError, match="query failed"):
        fb.runs(query="select nope from runs")


def test_the_query_command_prints_tables_for_reading_and_for_tools(
    two_runs: tuple[str, str],
) -> None:
    first, second = two_runs
    # ordered by what they saved: two runs can start in the same second, and their IDs then
    # sort by their random endings
    sql = "select run_id, metrics['val_loss'] as loss, cost_usd, outputs from runs order by 4"
    code, out = run_cli("log", "query", sql)
    assert code == 0, out
    lines = out.splitlines()
    assert lines[0].split() == ["run_id", "loss", "cost_usd", "outputs"]
    assert lines[2].split() == [second, "0.25", "1.5", "small@1"]
    code, out = run_cli("log", "query", sql, "--format", "markdown")
    assert out.splitlines()[1] == "| --- | --- | --- | --- |"
    assert out.splitlines()[3] == f"| {second} | 0.25 | 1.5 | small@1 |"  # each row cites its run
    rows = json.loads(run_cli("log", "query", "select * from runs", "--format", "json")[1])
    row = next(r for r in rows if r["run_id"] == second)
    assert row["metrics"] == {"val_loss": 0.25} and row["started"].startswith("20")
    code, out = run_cli("log", "query", sql, "--format", "csv")
    assert out.splitlines()[0] == "run_id,loss,cost_usd,outputs"
    code, out = run_cli("log", "query", "select nope from runs")
    assert code == 1 and "query failed" in out
    assert run_cli("log", "query", sql, "--format", "xml")[0] == 1
