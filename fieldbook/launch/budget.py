"""Spending limits for cloud runs (cloud-runs.md §11): a time limit on every run, enforced
on the machine, and a launch refused when the project's month would go over its budget.

A guardrail, not a wall: costs are known only after a machine is gone, so what is checked
is an estimate (the machine's hourly price times its time limit), and anyone with the
cloud's keys can start machines without `fieldbook`. The hard limit is the cloud's own (AWS
Budgets, a prepaid RunPod balance).

Every launch is written to the ledger, `launches/` in the project's folder of the storage
root, so that a member's launches count for every member, and so do runs that log nothing
to MLflow. The ledger is also where a launch's cost is recorded once the cloud's records
have it (cloud-runs.md §10): the one source of what cloud runs cost.

    launches/<run ID>.json            the launch: cluster, cloud, author, agent, price, limit
    launches/<run ID>.machine.json    which machine it got, written by the machine
    launches/<run ID>.ended           its job is over
    launches/<run ID>.cost.<USD>      what it cost (`none`: cannot be priced)

Side by side, with the cost in the file's name, so one listing gives every launch's state
and cost without reading a file.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from fieldbook import storage
from fieldbook.config import Settings, settings
from fieldbook.launch import LaunchError

DEFAULT_HOURS = 24  # a run's time limit when nothing else sets one
STARTUP = 600  # seconds a machine is paid for before its job starts (launch, image pull)
FOLDER = "launches"


class BudgetError(LaunchError):
    pass


def _unpriced(setting: str) -> BudgetError:
    return BudgetError(
        f"this machine's price is not known, so {setting} cannot be applied: name an "
        "instance type or GPU that SkyPilot's catalog has"
    )


@dataclass(frozen=True)
class Launch:
    run_id: str
    cluster: str
    cloud: str
    author: str
    launched: float  # seconds since the epoch
    price: float | None  # USD per hour, all nodes; None when the machine could not be priced
    limit: int  # seconds the job may run
    agent: str = ""  # the coding agent that launched it for the author (runs.md §20)

    def most(self) -> float:
        """The most this launch can cost, in USD."""
        return (self.price or 0.0) * (self.limit + STARTUP) / 3600


def span(seconds: float) -> str:
    """A time limit for reading: `9 min`, `1.5 h`."""
    return f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


def time_limit(cfg: Settings, price: float | None, max_hours: float | None) -> int:
    """Seconds the run's job may take: what `budget_run_usd` buys on this machine and
    `--max-hours`, whichever is less; 24 hours when neither is given."""
    limits = [max_hours * 3600] if max_hours else []
    if cap := cfg.budget_run_usd:
        if not price:
            raise _unpriced("budget_run_usd")
        bought = cap / price * 3600 - STARTUP
        if bought < 60:
            raise BudgetError(
                f"budget_run_usd (${cap:,.2f}) buys under a minute on this machine "
                f"(${price:,.2f} an hour, {STARTUP // 60} minutes to start)"
            )
        limits.append(bought)
    return int(min(limits)) if limits else DEFAULT_HOURS * 3600


def check_month(cfg: Settings, launch: Launch) -> None:
    """Refuse the launch if it could take the project's month over `budget_monthly_usd`."""
    cap = cfg.budget_monthly_usd
    if not cap:
        return
    if not launch.price:
        raise _unpriced("budget_monthly_usd")
    spent = month_spend(cfg)
    if spent + launch.most() > cap:
        month = datetime.now(UTC).strftime("%B")
        raise BudgetError(
            f"over the project's budget: this run could cost up to ${launch.most():,.2f} "
            f"({span(launch.limit)} at ${launch.price:,.2f} an hour), and about "
            f"${spent:,.2f} of ${cap:,.2f} is spent for {month} (budget_monthly_usd). "
            "Shorten it with --max-hours, or raise the budget in the project file"
        )


def record(launch: Launch, cfg: Settings) -> None:
    files = storage.fs(cfg)
    storage.makedirs(files, FOLDER)
    files.pipe_file(f"{FOLDER}/{launch.run_id}.json", json.dumps(asdict(launch)).encode())


def ended(run_id: str, cfg: Settings | None = None) -> None:
    """Mark a launch's job as over: its machine is deleted next. Written by the machine."""
    files = storage.fs(cfg or settings())
    storage.makedirs(files, FOLDER)
    files.pipe_file(f"{FOLDER}/{run_id}.ended", str(time.time()).encode())


def machine(run_id: str, tags: dict[str, str], cfg: Settings) -> None:
    """Record which machine a launch got (the cloud's ID for it, its type and region): what
    its cost is looked up by. Written by the machine, as its job starts."""
    files = storage.fs(cfg)
    storage.makedirs(files, FOLDER)
    files.pipe_file(f"{FOLDER}/{run_id}.machine.json", json.dumps(tags).encode())


def priced(run_id: str, usd: float | None, basis: str, cfg: Settings) -> None:
    """Record what a launch cost (`usd` None: it cannot be priced, and is not tried again)."""
    files = storage.fs(cfg)
    storage.makedirs(files, FOLDER)
    value = "none" if usd is None else f"{usd:.4f}"
    files.pipe_file(f"{FOLDER}/{run_id}.cost.{value}", json.dumps({"basis": basis}).encode())


@dataclass
class Entry:
    """A launch's state, as one listing of the ledger shows it."""

    run_id: str
    ended: float | None = None  # when its job was marked over
    has_machine: bool = False
    priced: bool = False
    cost: float | None = None  # USD; None while not priced, and when it cannot be


def ledger(cfg: Settings, prefix: str = "") -> dict[str, Entry]:
    """Every launch (or those whose run ID starts with `prefix`), from one listing."""
    from fieldbook.launch.records import _modified

    files = storage.fresh(storage.fs(cfg))
    try:
        listed = files.glob(f"{FOLDER}/{prefix}*", detail=True)
    except FileNotFoundError:
        return {}
    found: dict[str, Entry] = {}
    launched = set()
    for path, info in listed.items():
        run_id, _, kind = path.rsplit("/", 1)[-1].partition(".")
        entry = found.setdefault(run_id, Entry(run_id))
        if kind == "json":
            launched.add(run_id)
        elif kind == "ended":
            entry.ended = _modified(info)
        elif kind == "machine.json":
            entry.has_machine = True
        elif kind.startswith("cost."):
            entry.priced = True
            entry.cost = None if kind == "cost.none" else float(kind[5:])
    return {run_id: entry for run_id, entry in found.items() if run_id in launched}


def read(cfg: Settings, run_ids: list[str], suffix: str = "json") -> dict[str, dict]:
    """The launches of `run_ids` as written (or, with `machine.json`, their machines)."""
    files = storage.fs(cfg)
    found = {}
    for run_id in run_ids:
        try:
            found[run_id] = json.loads(files.cat_file(f"{FOLDER}/{run_id}.{suffix}"))
        except FileNotFoundError:
            continue
    return found


def most_paid(launch: dict, until: float) -> float:
    """An estimate for a launch without a recorded cost: its price times the time from its
    launch to `until`, never more than its limit."""
    paid = min(max(until - launch["launched"], 0.0), launch["limit"] + STARTUP)
    return (launch.get("price") or 0.0) * paid / 3600


def month_spend(cfg: Settings, now: float | None = None) -> float:
    """What the project's cloud runs launched this month (UTC) cost, in USD: the recorded
    cost where there is one, else the price times the time so far, or until the job ended
    or reached its limit."""
    now = now or time.time()
    month = datetime.fromtimestamp(now, UTC).strftime("%Y%m")  # run IDs begin with the date
    entries = ledger(cfg, month)
    estimated = [e.run_id for e in entries.values() if e.cost is None]
    launches = read(cfg, estimated)
    total = sum(e.cost for e in entries.values() if e.cost is not None)
    for run_id, launch in launches.items():
        total += most_paid(launch, entries[run_id].ended or now)
    return total
