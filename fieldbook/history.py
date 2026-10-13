"""What has been run in the project, and how two runs differ: `fieldbook log` and its `diff`
(runs.md §14).

A run leaves its record in three places: its own record of starting and ending, written
by `fieldbook run` (§18), the dataset versions it saved (data.md §11), and its MLflow runs
(tracking.md §3). None has every run: a script started with plain `python` has no record
of its own, a data transform logs nothing to MLflow, and a training run may save no
dataset. The history merges the three by run ID.
"""

from __future__ import annotations

import difflib
import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fieldbook.config import Settings, settings

MAX_DIFF_LINES = 400  # of code, per comparison, unless everything is asked for


class HistoryError(Exception):
    pass


@dataclass
class RunRecord:
    run_id: str
    started: datetime | None = None
    author: str = ""
    agent: str = ""  # the coding agent that ran it for the author (§20)
    command: str = ""
    git: str = ""  # `<commit> <branch> clean|dirty`
    code: str = ""  # code snapshot hash
    env: str = ""  # environment record (JSON)
    status: str = ""  # of its MLflow run: FINISHED, FAILED, RUNNING; empty without one
    experiment: str = ""
    smoke: bool = False
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    cost_usd: float | None = None
    inputs: list[str] = field(default_factory=list)  # dataset versions read
    outputs: list[str] = field(default_factory=list)  # dataset versions saved
    rerun_of: str = ""  # the run this one repeats (`fieldbook run --rerun`)
    exit_code: int | None = None  # of a run started by `fieldbook run`, once it ended
    ended: datetime | None = None
    where: str = ""  # local or cloud, for runs started by `fieldbook run`
    recorded: bool = False  # has a record of its own (§18), read by `with_details`
    verdict: str | None = None  # from its notes (runs.md §15)
    notes: list[dict[str, Any]] = field(default_factory=list)  # oldest first

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["started"] = self.started.isoformat() if self.started else None
        data["ended"] = self.ended.isoformat() if self.ended else None
        del data["recorded"]
        data["env"] = json.loads(self.env) if self.env else {}
        return data


def runs(cfg: Settings | None = None, *, limit: int | None = None) -> list[RunRecord]:
    """The project's runs, newest first."""
    cfg = cfg or settings()
    found: dict[str, RunRecord] = {}
    _from_datasets(cfg, found)
    _from_mlflow(cfg, found)
    _from_records(cfg, found)  # last: how the process ended decides the status
    _from_ledger(cfg, found)
    for run in found.values():
        run.inputs = sorted(set(run.inputs))
        run.outputs = sorted(set(run.outputs))
    ordered = sorted(found.values(), key=lambda r: (r.started or _EPOCH, r.run_id), reverse=True)
    return ordered[:limit] if limit else ordered


def with_details(found: list[RunRecord], cfg: Settings | None = None) -> list[RunRecord]:
    """Fill in what only a run's own record has (command, author, code of a run that saved
    nothing and logged nothing). One small file per run, so for the runs shown, not all."""
    from fieldbook.launch import records

    need = [r for r in found if r.recorded and not (r.command and r.code and r.where)]
    starts = records.read(cfg or settings(), [r.run_id for r in need])
    for run in need:
        start = starts.get(run.run_id, {})
        run.author = run.author or start.get("author", "")
        run.agent = run.agent or start.get("agent", "")
        run.command = run.command or start.get("command", "")
        run.git, run.code = run.git or start.get("git", ""), run.code or start.get("code", "")
        run.env = run.env or start.get("env", "")
        run.smoke = run.smoke or bool(start.get("smoke"))
        run.rerun_of = run.rerun_of or start.get("rerun_of", "")
        run.where = start.get("where", "")
    return found


def with_notes(found: list[RunRecord], cfg: Settings | None = None) -> list[RunRecord]:
    """Fill in the notes and verdicts of `found` (after choosing which runs to show: each
    note is a file to read)."""
    from fieldbook import notes

    by_run = notes.of([r.run_id for r in found], cfg)
    for run in found:
        mine = by_run.get(run.run_id, [])
        run.notes = [
            {"time": datetime.fromtimestamp(n.time, UTC).isoformat(), "author": n.author,
             "agent": n.agent, "text": n.text, "verdict": n.verdict}
            for n in mine
        ]  # fmt: skip
        run.verdict = notes.verdict(mine)
    return found


def find(prefix: str, cfg: Settings | None = None) -> RunRecord:
    """The run whose ID is `prefix` or starts with it."""
    matches = [r for r in runs(cfg) if r.run_id.startswith(prefix)]
    exact = [r for r in matches if r.run_id == prefix]
    if exact or len(matches) == 1:
        return with_details([(exact or matches)[0]], cfg)[0]
    if not matches:
        raise HistoryError(f"no run {prefix!r} (see `fieldbook log`)")
    raise HistoryError(f"{prefix!r} matches {len(matches)} runs: "
                       f"{', '.join(r.run_id for r in matches[:5])}")  # fmt: skip


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _from_datasets(cfg: Settings, found: dict[str, RunRecord]) -> None:
    from fieldbook.data.dataset import Dataset, list_datasets

    for name in list_datasets(cfg):
        for version in Dataset(name, cfg=cfg).log():
            if not version.run_id:
                continue
            run = found.setdefault(version.run_id, RunRecord(version.run_id))
            run.outputs.append(f"{name}@{version.version}")
            run.inputs += [ref.removeprefix(f"{cfg.project}/") for ref in version.inputs]
            saved = version.timestamp  # Lance gives local time without a timezone
            saved = saved if saved.tzinfo else saved.astimezone()
            if run.started is None or saved < run.started:
                run.started = saved
            run.author = run.author or version.author
            run.agent = run.agent or version.agent
            run.command = run.command or version.command
            run.git, run.code, run.env = version.git or run.git, version.code, version.env


def _from_mlflow(cfg: Settings, found: dict[str, RunRecord]) -> None:
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return
    from fieldbook.tracking import store

    client = MlflowClient(store.tracking_uri(cfg))
    experiments = {e.experiment_id: e.name for e in client.search_experiments()}
    token = None
    while experiments:
        page = client.search_runs(
            list(experiments), filter_string="tags.`fieldbook.run_id` LIKE '%'",
            max_results=1000, page_token=token,
        )  # fmt: skip
        for logged in page:
            tags = logged.data.tags
            run = found.setdefault(tags["fieldbook.run_id"], RunRecord(tags["fieldbook.run_id"]))
            started = datetime.fromtimestamp(logged.info.start_time / 1000, UTC)
            if run.started is None or started < run.started:
                run.started = started
            run.author = run.author or tags.get("fieldbook.author", "")
            run.agent = run.agent or tags.get("fieldbook.agent", "")
            run.command = run.command or tags.get("fieldbook.command", "")
            run.git = run.git or tags.get("fieldbook.git", "")
            run.code = run.code or tags.get("fieldbook.code", "")
            run.env = run.env or tags.get("fieldbook.env", "")
            run.smoke = run.smoke or "fieldbook.smoke" in tags
            run.rerun_of = run.rerun_of or tags.get("fieldbook.rerun_of", "")
            run.inputs += _refs(tags.get("fieldbook.inputs"), cfg)
            run.outputs += _refs(tags.get("fieldbook.outputs"), cfg)
            if "mlflow.parentRunId" in tags and run.params:
                continue  # a trial: the parent's values describe the run
            run.status = logged.info.status
            run.experiment = experiments.get(logged.info.experiment_id, "")
            run.params = dict(logged.data.params)
            run.metrics = {k: v for k, v in logged.data.metrics.items() if not _system(k)}
        token = page.token
        if not token:
            break


def _from_records(cfg: Settings, found: dict[str, RunRecord]) -> None:
    """Runs started by `fieldbook run` (§18): one listing gives each one's state."""
    from fieldbook.launch import records

    for run_id, state in records.states(cfg).items():
        run = found.setdefault(run_id, RunRecord(run_id))
        run.recorded = True
        if run.started is None:
            run.started = datetime.fromtimestamp(records.started_at(run_id), UTC)
        run.status, run.exit_code = state.status(), state.exit_code
        run.ended = datetime.fromtimestamp(state.ended, UTC) if state.ended else None


def _from_ledger(cfg: Settings, found: dict[str, RunRecord]) -> None:
    """What cloud runs cost (cloud-runs.md §10): one listing of the launch ledger."""
    from fieldbook.launch import budget

    for run_id, entry in budget.ledger(cfg).items():
        if entry.cost is not None:
            from fieldbook.launch import records

            run = found.setdefault(run_id, RunRecord(run_id))
            run.cost_usd = entry.cost
            run.started = run.started or datetime.fromtimestamp(records.started_at(run_id), UTC)


def _refs(tag: str | None, cfg: Settings) -> list[str]:
    try:
        return [ref.removeprefix(f"{cfg.project}/") for ref in json.loads(tag or "[]")]
    except ValueError:
        return []


def _system(metric: str) -> bool:
    return metric.startswith(("system/", "fieldbook."))


# --- comparing two runs ------------------------------------------------------------------


def diff(a: RunRecord, b: RunRecord, cfg: Settings | None = None, *,
         full_code: bool = False) -> dict[str, Any]:  # fmt: skip
    """What differs between two runs: command, git, environment, installed packages, data,
    parameters, metrics, and code. Only what differs is included."""
    cfg = cfg or settings()
    result: dict[str, Any] = {"a": a.run_id, "b": b.run_id}
    for name in ("command", "git", "author"):
        if getattr(a, name) != getattr(b, name):
            result[name] = {"a": getattr(a, name), "b": getattr(b, name)}
    env_a, env_b = (json.loads(r.env) if r.env else {} for r in (a, b))
    changed = {k: {"a": env_a.get(k), "b": env_b.get(k)}
               for k in sorted(set(env_a) | set(env_b))
               if k != "packages" and env_a.get(k) != env_b.get(k)}  # fmt: skip
    if changed:
        result["environment"] = changed
    if env_a.get("packages") != env_b.get("packages"):
        result["packages"] = _packages(env_a.get("packages"), env_b.get("packages"), cfg)
    for name in ("inputs", "outputs"):
        if data := _data(getattr(a, name), getattr(b, name)):
            result[name] = data
    params = {k: {"a": a.params.get(k), "b": b.params.get(k)}
              for k in sorted(set(a.params) | set(b.params))
              if a.params.get(k) != b.params.get(k)}  # fmt: skip
    if params:
        result["params"] = params
    metrics = {k: {"a": a.metrics.get(k), "b": b.metrics.get(k)}
               for k in sorted(set(a.metrics) | set(b.metrics))
               if a.metrics.get(k) != b.metrics.get(k)}  # fmt: skip
    if metrics:
        result["metrics"] = metrics
    if a.code != b.code and a.code and b.code:
        result["code"] = _code(a.code, b.code, cfg, full=full_code)
    return result


def _packages(a: str | None, b: str | None, cfg: Settings) -> dict[str, Any]:
    from fieldbook.code import snapshot

    def installed(digest: str | None) -> dict[str, str]:
        if not digest:
            return {}
        try:
            _, text = snapshot.read_packages(digest, cfg)
        except Exception:
            return {}
        return dict(line.split("==", 1) for line in text.splitlines() if "==" in line)

    before, after = installed(a), installed(b)
    return {
        "added": {k: after[k] for k in sorted(set(after) - set(before))},
        "removed": {k: before[k] for k in sorted(set(before) - set(after))},
        "changed": {k: {"a": before[k], "b": after[k]}
                    for k in sorted(set(before) & set(after)) if before[k] != after[k]},
    }  # fmt: skip


def _data(a: list[str], b: list[str]) -> dict[str, Any]:
    """Dataset versions by name: `raw@3` against `raw@5` is one changed input."""

    def by_name(refs: list[str]) -> dict[str, str]:
        return dict(ref.rsplit("@", 1) if "@" in ref and "://" not in ref else (ref, "")
                    for ref in refs)  # fmt: skip

    before, after = by_name(a), by_name(b)
    names = sorted(set(before) | set(after))
    return {k: {"a": before.get(k), "b": after.get(k)}
            for k in names if before.get(k) != after.get(k)}  # fmt: skip


def _code(a: str, b: str, cfg: Settings, *, full: bool) -> dict[str, Any]:
    from fieldbook.code import snapshot

    with tempfile.TemporaryDirectory(prefix="fieldbook-diff-") as tmp:
        left, right = Path(tmp, "a"), Path(tmp, "b")
        snapshot.checkout(a, left, cfg)
        snapshot.checkout(b, right, cfg)
        names_a = {p.relative_to(left).as_posix() for p in left.rglob("*") if p.is_file()}
        names_b = {p.relative_to(right).as_posix() for p in right.rglob("*") if p.is_file()}
        modified = sorted(n for n in names_a & names_b
                          if (left / n).read_bytes() != (right / n).read_bytes())  # fmt: skip
        patch: list[str] = []
        for name in modified:
            try:
                old = (left / name).read_text().splitlines()
                new = (right / name).read_text().splitlines()
            except UnicodeDecodeError:
                patch.append(f"Binary file {name} differs")
                continue
            patch += difflib.unified_diff(old, new, f"a/{name}", f"b/{name}", lineterm="")
        cut = None if full or len(patch) <= MAX_DIFF_LINES else len(patch) - MAX_DIFF_LINES
        return {
            "a": a, "b": b,
            "added": sorted(names_b - names_a), "removed": sorted(names_a - names_b),
            "modified": modified,
            "patch": patch if cut is None else patch[:MAX_DIFF_LINES],
            "patch_lines_cut": cut or 0,
        }  # fmt: skip


# --- the runs as tables, for queries (§21) -----------------------------------------------

VERIFIED = "rerun as "  # how a rerun's outcome is noted on the run it repeated (§17)


def tables(cfg: Settings | None = None) -> dict[str, Any]:
    """The project's runs as PyArrow tables: `runs` (one row a run, its parameters and
    metrics as maps), and `metrics`, `params`, `data`, `notes` (one row a value), which
    join to it on `run_id`. Built from the records each time: every run's own record and
    notes are read, so it costs more than a listing."""
    import pyarrow as pa

    cfg = cfg or settings()
    found = with_notes(with_details(runs(cfg), cfg), cfg)
    rows, metrics, params, data, noted = [], [], [], [], []
    for run in found:
        proved = any(n["text"].startswith(VERIFIED) and ": reproduced" in n["text"]
                     for n in run.notes)  # fmt: skip
        rows.append({
            "run_id": run.run_id, "started": run.started, "ended": run.ended,
            "author": run.author, "agent": run.agent, "command": run.command,
            "status": run.status.lower(), "exit_code": run.exit_code, "ran": run.where,
            "git": run.git, "code": run.code, "smoke": run.smoke, "experiment": run.experiment,
            "rerun_of": run.rerun_of, "cost_usd": run.cost_usd, "verdict": run.verdict,
            "note": run.notes[-1]["text"] if run.notes else None, "verified": proved,
            "params": list(run.params.items()), "metrics": list(run.metrics.items()),
            "inputs": run.inputs, "outputs": run.outputs,
        })  # fmt: skip
        metrics += [{"run_id": run.run_id, "name": k, "value": v} for k, v in run.metrics.items()]
        params += [{"run_id": run.run_id, "name": k, "value": v} for k, v in run.params.items()]
        for role, refs in (("read", run.inputs), ("saved", run.outputs)):
            for ref in refs:
                name, _, version = ref.rpartition("@") if "://" not in ref else (ref, "", "")
                data.append({"run_id": run.run_id, "dataset": name or ref, "role": role, "ref": ref,
                             "version": int(version) if version.isdigit() else None})  # fmt: skip
        noted += [{"run_id": run.run_id, **n, "time": datetime.fromisoformat(n["time"])}
                  for n in run.notes]  # fmt: skip
    text, moment = pa.string(), pa.timestamp("us", tz="UTC")
    schemas = {
        "runs": pa.schema([
            ("run_id", text), ("started", moment), ("ended", moment), ("author", text),
            ("agent", text), ("command", text), ("status", text), ("exit_code", pa.int64()),
            ("ran", text), ("git", text), ("code", text), ("smoke", pa.bool_()),
            ("experiment", text), ("rerun_of", text), ("cost_usd", pa.float64()),
            ("verdict", text), ("note", text), ("verified", pa.bool_()),
            ("params", pa.map_(text, text)), ("metrics", pa.map_(text, pa.float64())),
            ("inputs", pa.list_(text)), ("outputs", pa.list_(text)),
        ]),
        "metrics": pa.schema([("run_id", text), ("name", text), ("value", pa.float64())]),
        "params": pa.schema([("run_id", text), ("name", text), ("value", text)]),
        "data": pa.schema([("run_id", text), ("dataset", text), ("version", pa.int64()),
                           ("role", text), ("ref", text)]),
        "notes": pa.schema([("run_id", text), ("time", moment), ("author", text),
                            ("agent", text), ("verdict", text), ("text", text)]),
    }  # fmt: skip
    values = {"runs": rows, "metrics": metrics, "params": params, "data": data, "notes": noted}
    return {name: pa.Table.from_pylist(values[name], schema=schema)
            for name, schema in schemas.items()}  # fmt: skip


def table(query: str | None = None, cfg: Settings | None = None) -> Any:
    """`fb.runs()`: the project's runs as a PyArrow table, one row a run (`.to_pandas()`
    for a dataframe). With `query`, the result of that SQL (DuckDB's dialect) over the
    tables `runs`, `metrics`, `params`, `data`, and `notes`, as `fieldbook log query` runs it:

        fb.runs(query="select run_id, metrics['acc'] from runs where status = 'finished'")
    """
    found = tables(cfg)
    if query is None:
        return found["runs"]
    try:
        import duckdb
    except ImportError:
        raise HistoryError("queries need DuckDB: pip install 'fieldbook[sql]'") from None
    con = duckdb.connect()
    for name, rows in found.items():
        con.register(name, rows)
    try:
        result = con.sql(query)
        return result.to_arrow_table() if result is not None else found["runs"].slice(0, 0)
    except duckdb.Error as e:
        raise HistoryError(f"query failed: {e}") from None
