"""The cost of cloud runs, from the cloud's own records (cloud-runs.md §10).

The machine writes which machine it is to the launch ledger (`launches/`, see
`fieldbook.launch.budget`). On the laptop, launches that are over are then priced by the
cloud's module (`fieldbook.clouds`; AWS: CloudTrail times the price list; RunPod: its
billing), and the cost is written to the ledger: the one place costs are read from, for
every cloud run, whether or not it logged to MLflow. A launch that never got a machine is
marked `unavailable`. Multi-node runs are charged the head machine's time for every node.

A copy goes on the run's first MLflow run, if it has one, as the metric `fieldbook.cost_usd`.
"""

import functools
import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable

from fieldbook import clouds
from fieldbook.config import Settings

LOOKBACK = 90 * 86400  # seconds: CloudTrail keeps events for 90 days


# --- on the machine -------------------------------------------------------------------


@functools.cache
def machine_tags() -> dict[str, str]:
    """MLflow tags naming this cloud machine; empty when not on one."""
    info = os.environ.get("SKYPILOT_CLUSTER_INFO")
    if not info:
        return {}
    info = json.loads(info)
    cloud = str(info.get("cloud") or "").lower()
    tags = {
        "fieldbook.cluster": info.get("cluster_name") or "",
        "fieldbook.cloud": cloud,
        "fieldbook.region": info.get("region") or "",
        "fieldbook.nodes": os.environ.get("SKYPILOT_NUM_NODES", "1"),
    }
    if module := clouds.get(cloud):
        tags |= module.machine()
    return {tag: value for tag, value in tags.items() if value}


# --- on the laptop --------------------------------------------------------------------


def record(cfg: Settings, warn: Callable[[str], None] = lambda message: None) -> int:
    """Record the cost of cloud launches that are over and whose cloud records are there;
    returns how many were recorded. A launch that cannot be priced yet is tried again next
    time; one that never can be (it got no machine) is marked so."""
    from fieldbook.launch import budget, records
    from fieldbook.tracking import store

    if store.read_only(store.tracking_uri(cfg)):
        return 0  # a viewer: members record the costs (sharing.md §2)
    now = time.time()
    waiting = [e for e in budget.ledger(cfg).values()
               if not e.priced and now - records.started_at(e.run_id) < LOOKBACK]  # fmt: skip
    if not waiting:
        return 0
    states = records.states(cfg)
    launches = budget.read(cfg, [e.run_id for e in waiting])
    recorded, failed = 0, []
    for entry in waiting:
        launch = launches.get(entry.run_id)
        over = launch and _over(entry, launch, states.get(entry.run_id), now)
        if not over:
            continue  # still going
        tags = budget.read(cfg, [entry.run_id], "machine.json").get(entry.run_id, {})
        module = clouds.get(tags.get("fieldbook.cloud"))
        if module is None or not tags.get("fieldbook.machine"):
            budget.priced(entry.run_id, None, "unavailable", cfg)  # it never got a machine
            continue
        try:
            result = module.cost(tags, launch["launched"], over)
        except Exception as e:  # one bad launch must not stop the others
            failed.append(f"{launch['cluster']} ({type(e).__name__}: {e})")
            continue
        if result is None:
            continue  # the cloud's records are not there yet
        cost, basis = result
        budget.priced(entry.run_id, cost, basis, cfg)
        _copy_to_mlflow(cfg, entry.run_id, cost, basis)
        recorded += 1
    if failed:
        warn(f"could not price {len(failed)} cloud runs: {'; '.join(failed[:3])}")
    return recorded


def _over(entry, launch: dict, state, now: float) -> float | None:  # noqa: ANN001
    """When a launch's job was over, or None while it may still be going: marked ended by
    its machine (or by the laptop, for a failed launch), or its run's own record says it
    ended or stopped giving signs of life, or its time limit is long past."""
    from fieldbook.launch import budget, records

    if entry.ended:
        return entry.ended
    if state is not None and state.status(now) not in (records.RUNNING, records.LAUNCHING):
        return state.ended or state.alive or now
    latest = launch["launched"] + launch["limit"] + budget.STARTUP + records.LAUNCH_WINDOW
    return latest if now > latest else None


def _copy_to_mlflow(cfg: Settings, run_id: str, cost: float, basis: str) -> None:
    """Also log the cost on the run's first MLflow run, if it has one, as `fieldbook.cost_usd`
    with `fieldbook.cost_basis`: a copy, for the MLflow UI. The ledger is what is read."""
    try:
        from mlflow.tracking import MlflowClient

        from fieldbook.tracking import store

        client = MlflowClient(store.tracking_uri(cfg))
        experiments = [e.experiment_id for e in client.search_experiments()]
        found = client.search_runs(
            experiments, filter_string=f"tags.`fieldbook.run_id` = '{run_id}'"
        )
        top = [r for r in found if "mlflow.parentRunId" not in r.data.tags]  # not the trials
        if top:
            holder = min(top, key=lambda r: r.info.start_time).info.run_id
            client.log_metric(holder, "fieldbook.cost_usd", round(cost, 4))
            client.set_tag(holder, "fieldbook.cost_basis", basis)
    except Exception:  # the copy is a convenience
        pass


def record_in_background(cfg: Settings, report: Callable[[str], None]) -> threading.Thread:
    """`record` in a daemon thread, reporting what happened through `report`."""

    def work() -> None:
        try:
            if count := record(cfg, warn=lambda m: report(f"warning: {m}")):
                report(f"recorded the cost of {count} cloud runs")
        except Exception as e:  # bookkeeping never stops anything
            report(f"warning: could not record cloud run costs: {type(e).__name__}: {e}")

    thread = threading.Thread(target=work, name="fieldbook-costs", daemon=True)
    thread.start()
    return thread


def summary(cfg: Settings) -> list[tuple[str, str, str, int, float]]:
    """(month, author, agent, runs, dollars) of the recorded cloud costs, newest month
    first (sharing.md §4); `agent` is the coding agent that launched the runs for the
    author, "" for the author's own (runs.md §20)."""
    from fieldbook.launch import budget

    paid = {e.run_id: e.cost for e in budget.ledger(cfg).values() if e.cost is not None}
    totals: dict[tuple[str, str, str], list] = defaultdict(lambda: [0, 0.0])
    for run_id, launch in budget.read(cfg, list(paid)).items():
        month = f"{run_id[:4]}-{run_id[4:6]}"  # run IDs begin with the UTC date
        total = totals[(month, launch.get("author") or "unknown", launch.get("agent", ""))]
        total[0] += 1
        total[1] += paid[run_id]
    rows = [(*key, n, cost) for key, (n, cost) in totals.items()]
    rows.sort(key=lambda row: row[4], reverse=True)  # within a month, the biggest first
    return sorted(rows, key=lambda row: row[0], reverse=True)
