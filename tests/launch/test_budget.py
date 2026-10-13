"""Spending limits for cloud runs (fieldbook/launch/budget.py)."""

import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fieldbook import config
from fieldbook.cli import app
from fieldbook.config import settings
from fieldbook.launch import budget, cloud

HOUR = 3600


def with_budget(**values: float) -> config.Settings:
    return settings().model_copy(update=values)


def launch(
    run_id: str, *, price: float | None = 2.0, limit: int = HOUR, ago: float = 0
) -> budget.Launch:
    return budget.Launch(
        run_id, f"fieldbook-{run_id}", "aws", "me", time.time() - ago, price, limit
    )


def test_a_run_gets_a_time_limit_from_the_budget_or_a_default() -> None:
    assert budget.time_limit(settings(), 2.0, None) == 24 * HOUR  # nothing set
    assert budget.time_limit(settings(), None, None) == 24 * HOUR  # unpriced, no budget
    assert budget.time_limit(settings(), 2.0, 1.5) == 1.5 * HOUR
    capped = with_budget(budget_run_usd=10.0)  # 5 hours at $2, less the start-up
    assert budget.time_limit(capped, 2.0, None) == 5 * HOUR - budget.STARTUP
    assert budget.time_limit(capped, 2.0, 1) == HOUR  # the shorter of the two
    assert budget.time_limit(capped, 2.0, 100) == 5 * HOUR - budget.STARTUP
    with pytest.raises(budget.BudgetError, match="price is not known"):
        budget.time_limit(capped, None, None)
    with pytest.raises(budget.BudgetError, match="buys under a minute"):
        budget.time_limit(with_budget(budget_run_usd=0.3), 2.0, None)


def test_the_month_counts_recorded_costs_and_estimates(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = settings()
    month = time.strftime("%Y%m", time.gmtime())
    assert budget.month_spend(cfg) == 0.0  # nothing launched yet
    budget.record(launch(f"{month}01-priced", ago=5 * HOUR), cfg)  # its recorded cost counts
    budget.priced(f"{month}01-priced", 0.25, "aws-cloudtrail", cfg)
    budget.record(launch(f"{month}01-running", ago=HOUR / 2), cfg)  # $2 an hour, so far
    budget.record(launch(f"{month}01-forgot", ago=9 * HOUR), cfg)  # no end seen: to its limit
    budget.record(launch(f"{month}01-ended", ago=3 * HOUR), cfg)
    budget.record(launch("20200101-old", ago=9 * HOUR), cfg)  # another month
    budget.record(launch(f"{month}01-unpriced", price=None, ago=HOUR), cfg)
    budget.ended(f"{month}01-ended", cfg)  # ended now: 3 hours, capped at its limit
    most = 2.0 * (HOUR + budget.STARTUP) / HOUR
    assert budget.month_spend(cfg) == pytest.approx(0.25 + 1.0 + most + most, abs=0.01)


def test_a_launch_over_the_months_budget_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    month = time.strftime("%Y%m", time.gmtime())
    budget.check_month(settings(), launch("x", price=None))  # no budget: nothing to check
    cfg = with_budget(budget_monthly_usd=4.5)
    budget.check_month(cfg, launch("x"))  # $2.33 at most, of $4.50
    budget.record(launch(f"{month}01-earlier", ago=2 * HOUR), cfg)  # $2.33 spent
    budget.check_month(cfg, launch("x", limit=HOUR // 2))  # $1.33 more still fits
    with pytest.raises(budget.BudgetError) as refused:
        budget.check_month(cfg, launch("x"))
    message = str(refused.value)
    assert "up to $2.33" in message and "$2.33 of $4.50" in message and "--max-hours" in message
    with pytest.raises(budget.BudgetError, match="price is not known"):
        budget.check_month(cfg, launch("x", price=None))


def test_the_machine_stops_the_job_at_its_limit_and_notes_its_end() -> None:
    command = cloud._remote_command("r1", "abc", ["train.py"], [], 1, limit=5400)
    lines = command.splitlines()
    assert lines[1] == "trap 'fieldbook launch-ended r1 >/dev/null 2>&1 || true' EXIT"
    assert lines[-1].startswith(
        "timeout -k 60 5400 fieldbook run --run-id r1 --code abc train.py || "
    )
    assert "time limit of this run (1.5 h)" in lines[-1]


def test_the_machine_notes_the_end_of_its_job(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(app, ["launch-ended", "20260101-000000-abc"])
    assert result.exit_code == 0, result.output
    from fieldbook import storage

    assert storage.fs(settings()).exists("launches/20260101-000000-abc.ended")


def test_budgets_are_read_from_the_project_file_only(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (clean_env / "fieldbook.toml").write_text('project = "p"\nbudget_monthly_usd = 50\n')
    assert settings().budget_monthly_usd == 50 and settings().budget_run_usd is None
    monkeypatch.setenv("FIELDBOOK_BUDGET_MONTHLY_USD", "5000")  # a variable cannot raise it
    with pytest.raises(
        config.ConfigError, match="project file only, not in the variable FIELDBOOK_BUDGET_MONTHLY"
    ):
        settings()
    monkeypatch.delenv("FIELDBOOK_BUDGET_MONTHLY_USD")
    result = CliRunner().invoke(app, ["config", "set", "--local", "budget_run_usd", "9"])
    assert result.exit_code != 0 and "project file only" in result.output
    assert settings().budget_run_usd is None  # this machine's file was put back
    result = CliRunner().invoke(app, ["config", "set", "budget_run_usd", "9"])
    assert result.exit_code == 0 and settings().budget_run_usd == 9
    (clean_env / "fieldbook.toml").write_text('project = "p"\nbudget_monthly_usd = 0\n')
    with pytest.raises(config.ConfigError, match="budget_monthly_usd"):
        settings()
