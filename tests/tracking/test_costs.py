"""Cost of cloud runs from the cloud's records (cloud-runs.md §10), without a cloud."""

import json
import time

import pytest

from fieldbook.clouds import aws, runpod
from fieldbook.config import settings
from fieldbook.tracking import costs

AWS = '{"cluster_name": "fieldbook-r1", "cloud": "AWS", "region": "us-west-2", "zone": null}'


@pytest.fixture(autouse=True)
def _fresh_machine_tags():
    costs.machine_tags.cache_clear()
    yield
    costs.machine_tags.cache_clear()


def test_aws_machines_tag_runs_with_their_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    from fieldbook.data import lineage

    monkeypatch.setenv("SKYPILOT_CLUSTER_INFO", AWS)
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
    monkeypatch.setattr(aws, "instance", lambda: ("i-0abc", "g5.xlarge"))
    tags = lineage.run_tags()
    shown = ("fieldbook.cl", "fieldbook.re", "fieldbook.no", "fieldbook.ma")
    assert {k: tags[k] for k in tags if k.startswith(shown)} == {
        "fieldbook.cluster": "fieldbook-r1",
        "fieldbook.cloud": "aws",
        "fieldbook.region": "us-west-2",
        "fieldbook.nodes": "2",
        "fieldbook.machine": "i-0abc",
        "fieldbook.machine_type": "g5.xlarge",
    }


def test_runpod_machines_tag_runs_with_their_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SKYPILOT_CLUSTER_INFO", '{"cluster_name": "fieldbook-r2", "cloud": "RunPod"}'
    )
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    assert costs.machine_tags()["fieldbook.machine"] == "pod123"


def test_no_tags_off_the_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKYPILOT_CLUSTER_INFO", raising=False)
    assert costs.machine_tags() == {}


def launched(run_id: str, ago: float, *, machine: dict | None = None, over: bool = True,
             author: str = "tester", agent: str = "", limit: int = 3600) -> str:  # fmt: skip
    """A cloud launch in the ledger, as `fieldbook run --on remote` and its machine leave it."""
    from fieldbook.launch import budget

    cfg = settings()
    launch = budget.Launch(run_id, f"fieldbook-{run_id}", "aws", author, time.time() - ago, 1.0,
                           limit, agent)  # fmt: skip
    budget.record(launch, cfg)
    if machine:
        budget.machine(run_id, machine, cfg)
    if over:
        budget.ended(run_id, cfg)
    return run_id


def recent(tail: str, seconds_ago: float = 4000) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() - seconds_ago)) + f"-{tail}"


def test_costs_come_from_the_clouds_records(mlflow_runs, monkeypatch: pytest.MonkeyPatch) -> None:
    from fieldbook.launch import budget

    cfg, now = settings(), time.time()
    ec2 = {
        "fieldbook.cloud": "aws",
        "fieldbook.region": "us-west-2",
        "fieldbook.machine_type": "g5.xlarge",
    }
    train = launched(
        recent("aaaaaa"), 4000, machine=ec2 | {"fieldbook.machine": "i-a", "fieldbook.nodes": "2"}
    )
    data = launched(
        recent("dddddd"), 9000, machine={"fieldbook.cloud": "runpod", "fieldbook.machine": "pod1"}
    )
    pending = launched(recent("bbbbbb"), 600, machine=ec2 | {"fieldbook.machine": "i-b"})
    broken = launched(recent("cccccc"), 600, machine=ec2 | {"fieldbook.machine": "i-c"})
    going = launched(recent("eeeeee"), 60, machine=ec2 | {"fieldbook.machine": "i-e"}, over=False)
    never = launched(recent("ffffff"), 600)  # the launch failed: it never got a machine
    logged = mlflow_runs.log([], run_id=train)  # the training run also logged to MLflow
    trial = mlflow_runs.client.create_run(
        mlflow_runs.client.get_run(logged).info.experiment_id,
        tags={"fieldbook.run_id": train, "mlflow.parentRunId": logged},
    ).info.run_id

    def trail(instance: str, region: str):
        assert region == "us-west-2" and instance != "i-e"  # a run still going is not priced
        if instance == "i-c":
            raise RuntimeError("throttled")
        return {"i-a": (now - 4200, now - 4200 + 1800)}.get(instance)  # i-b: not there yet

    monkeypatch.setattr(aws, "cloudtrail_times", trail)
    monkeypatch.setattr(aws, "hourly_price", lambda kind, region: 1.0)
    monkeypatch.setattr(runpod, "billed", lambda pod, start: 0.37)
    monkeypatch.setattr(runpod, "SETTLED", 0)
    warnings = []
    assert costs.record(cfg, warn=warnings.append) == 2
    ledger = budget.ledger(cfg)
    # 2 nodes x $1/h x 30 minutes from launch to termination
    assert ledger[train].cost == pytest.approx(1.0) and ledger[data].cost == pytest.approx(0.37)
    assert not ledger[pending].priced and not ledger[broken].priced and not ledger[going].priced
    assert ledger[never].priced and ledger[never].cost is None  # marked, and not tried again
    assert len(warnings) == 1 and "cccccc" in warnings[0]
    # a run that logged nothing to MLflow (a data run) has its cost all the same; one that
    # did has a copy on its first run, not on its trials
    client = mlflow_runs.client
    assert client.get_run(logged).data.metrics["fieldbook.cost_usd"] == pytest.approx(1.0)
    assert client.get_run(logged).data.tags["fieldbook.cost_basis"] == "aws-cloudtrail"
    assert "fieldbook.cost_usd" not in client.get_run(trial).data.metrics

    monkeypatch.setattr(aws, "cloudtrail_times", lambda *a: None)
    monkeypatch.setattr(runpod, "billed", lambda *a: pytest.fail("priced twice"))
    assert costs.record(cfg) == 0


def test_a_run_that_stopped_without_a_word_is_priced_too(
    repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine that was reclaimed never marked its job over: its run's record (no sign of
    life) or, failing that, its time limit says when it was."""
    from fieldbook.launch import budget, records

    cfg = settings()
    ec2 = {
        "fieldbook.cloud": "aws",
        "fieldbook.region": "us-west-2",
        "fieldbook.machine_type": "g5.xlarge",
    }
    killed = launched(recent("111111", 9000), 9000, machine=ec2 | {"fieldbook.machine": "i-k"},
                      over=False, limit=86400)  # fmt: skip
    records.start(cfg, killed, {"where": "cloud"})  # no sign of life, long after launch
    silent = launched(
        recent("222222", 9000),
        9000,
        machine=ec2 | {"fieldbook.machine": "i-s"},
        over=False,
        limit=600,
    )  # fmt: skip (no record at all; its limit is past)
    young = launched(
        recent("333333", 900),
        900,
        machine=ec2 | {"fieldbook.machine": "i-y"},
        over=False,
        limit=86400,
    )  # fmt: skip (no record, within its limit)
    monkeypatch.setattr(aws, "cloudtrail_times", lambda instance, region: (0.0, 3600.0))
    monkeypatch.setattr(aws, "hourly_price", lambda kind, region: 2.0)
    assert costs.record(cfg) == 2
    ledger = budget.ledger(cfg)
    assert ledger[killed].cost == ledger[silent].cost == 2.0 and not ledger[young].priced


def test_runpod_waits_for_billing_to_settle(repo, monkeypatch: pytest.MonkeyPatch) -> None:
    launched(
        recent("999999", 300), 300, machine={"fieldbook.cloud": "runpod", "fieldbook.machine": "p"}
    )
    monkeypatch.setattr(runpod, "billed", lambda *a: pytest.fail("billing not settled"))
    assert costs.record(settings()) == 0


def test_runpod_billing_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import json
    import urllib.parse
    import urllib.request

    from fieldbook.clouds import runpod

    seen = {}

    def urlopen(request, timeout):
        seen["query"] = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(request.full_url).query))
        seen["agent"] = request.get_header("User-agent")
        reply = {
            "records": [{"totalAmount": 0.0174}],
            "metadata": {"totals": {"totalAmount": 0.0174}},
        }
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert runpod.billed("pod1", time.time() - 3600) == pytest.approx(0.0174)
    assert seen["query"]["podId"] == "pod1" and {"startTime", "endTime"} <= seen["query"].keys()
    assert seen["agent"] == "fieldbook"  # Cloudflare refuses Python's default (error 1010)


def test_runs_carry_their_author(monkeypatch: pytest.MonkeyPatch) -> None:
    from fieldbook.data import lineage

    assert (
        lineage.run_tags()["fieldbook.author"] == "tester"
    )  # FIELDBOOK_AUTHOR, as `fieldbook run` sets it
    monkeypatch.delenv("FIELDBOOK_AUTHOR")
    monkeypatch.setenv("FIELDBOOK_AUTHOR", "")
    assert lineage.run_tags().get("fieldbook.author") == settings().author


def test_costs_by_month_author_and_agent(repo, mlflow_runs) -> None:
    from helpers import runner

    from fieldbook.cli import app
    from fieldbook.launch import budget

    def paid(author: str, day: str, dollars: float, agent: str = "") -> str:
        run_id = f"{day.replace('-', '')}-000000-{int(dollars * 100):06x}"
        launched(run_id, 0, author=author, agent=agent)
        budget.priced(run_id, dollars, "aws-cloudtrail", settings())
        return run_id

    paid("alice", "2026-09-03", 1.5)
    paid("alice", "2026-10-01", 2.0)
    paid("alice", "2026-10-02", 0.25)
    paid("alice", "2026-10-03", 3.0, agent="claude-code")
    paid("bob", "2026-10-05", 4.0)
    paid("", "2026-10-06", 0.1)
    launched("20261007-000000-000000", 0, author="carol", over=False)  # not priced yet
    budget.priced(
        launched("20261008-000000-000000", 0, author="dan"), None, "unavailable", settings()
    )
    assert costs.summary(settings()) == [
        ("2026-10", "bob", "", 1, 4.0), ("2026-10", "alice", "claude-code", 1, 3.0),
        ("2026-10", "alice", "", 2, 2.25), ("2026-10", "unknown", "", 1, 0.1),
        ("2026-09", "alice", "", 1, 1.5),
    ]  # fmt: skip
    result = runner.invoke(app, ["log", "costs", "--total"])
    assert result.exit_code == 0, result.output
    lines = [line.split() for line in result.output.splitlines()]
    assert lines[0] == ["month", "author", "by", "runs", "cost"]
    assert ["2026-10", "alice", "claude-code", "1", "$3.00"] in lines
    assert ["2026-10", "alice", "by", "hand", "2", "$2.25"] in lines
    assert lines[-1] == ["total", "6", "$10.85"]
    totals = json.loads(runner.invoke(app, ["log", "costs", "--total", "--json"]).output)
    assert totals["months"][1] == {"month": "2026-10", "author": "alice", "agent": "claude-code",
                                   "runs": 1, "cost_usd": 3.0}  # fmt: skip
    # without --total: what each run cost, newest first, from the same records
    result = runner.invoke(app, ["log", "costs", "-n", "2"])
    assert result.exit_code == 0, result.output
    rows = result.output.splitlines()
    assert rows[0].split() == ["run", "started", "author", "command", "cost"]
    assert rows[1].startswith("20261006-") and rows[1].endswith("$0.10")
    assert rows[2].startswith("20261005-") and rows[2].endswith("$4.00") and len(rows) == 3
    per_run = json.loads(runner.invoke(app, ["log", "costs", "--json"]).output)
    assert [r["cost_usd"] for r in per_run] == [0.1, 4.0, 3.0, 0.25, 2.0, 1.5]
