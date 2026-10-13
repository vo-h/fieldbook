"""`fieldbook log`: what has been run, how two runs differ, what was noted, and what each
run recorded (runs.md §14, §15)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from fieldbook.cli._common import fail

AsJson = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]


app = typer.Typer(
    invoke_without_command=True,
    help="List the project's runs; query, inspect, compare, or wait for one.",
)
# Things one might type after `fieldbook run` meaning `fieldbook log ...`
SUBCOMMANDS = ("diff", "note", "ls", "wait", "why", "output", "logs", "files", "costs", "query",
               "rm", "code", "packages", "ui")  # fmt: skip


@app.callback()
def runs(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many runs, newest first.")] = 20,
    author: Annotated[str | None, typer.Option(help="Only this person's runs.")] = None,
    verdict: Annotated[
        str | None, typer.Option(help="Only runs noted as good, bad, or inconclusive.")
    ] = None,
    agent: Annotated[
        bool | None,
        typer.Option(
            "--agent/--no-agent", help="Only runs a coding agent started, or only others."
        ),
    ] = None,
    as_json: AsJson = False,
) -> None:
    """List the project's runs; query, inspect, compare, or wait for one."""
    if ctx.invoked_subcommand is not None:
        return
    from fieldbook import history
    from fieldbook.code import snapshot
    from fieldbook.config import settings
    from fieldbook.launch import agents

    found = history.runs(settings())
    if author or agent is not None:  # of a run that saved and logged nothing: in its record
        found = [r for r in history.with_details(found)
                 if (not author or r.author == author)
                 and (agent is None or bool(r.agent) == agent)]  # fmt: skip
    if verdict:  # notes are read for every run only when filtering by them
        found = [r for r in history.with_notes(found) if r.verdict == verdict]
    found = history.with_notes(history.with_details(found[:limit]))
    if as_json:
        typer.echo(json.dumps([r.to_json() for r in found], indent=2))
        return
    if not found:
        typer.echo("no runs yet")
        return
    rows = []
    for r in found:
        metrics = "  ".join(f"{k}={v:.4g}" for k, v in sorted(r.metrics.items())[:3])
        rows.append([
            r.run_id,
            r.started.astimezone().strftime("%Y-%m-%d %H:%M") if r.started else "",
            agents.who(r.author, r.agent),
            _cut(r.command, 44),
            snapshot.short_git(r.git) or r.code[:12],
            (r.status or "").lower() + (" (smoke)" if r.smoke else ""),
            metrics,
            _cut(", ".join(r.outputs), 30),
            _cut(" ".join(filter(None, [f"[{r.verdict}]" if r.verdict else "",
                                        r.notes[-1]["text"] if r.notes else ""])), 50),
        ])  # fmt: skip
    headers = ["run", "started", "author", "command", "git", "status", "metrics", "saved", "note"]
    _table(headers, rows)


@app.command("diff")
def diff(
    a: Annotated[str, typer.Argument(metavar="RUN_A", help="A run ID, or the start of one.")],
    b: Annotated[str, typer.Argument(metavar="RUN_B")],
    patch: Annotated[
        bool, typer.Option("--patch", help="The whole code diff, however long.")
    ] = False,
    as_json: AsJson = False,
) -> None:
    """Show what differs between two runs."""
    from fieldbook import history
    from fieldbook.code import snapshot
    from fieldbook.config import settings

    cfg = settings()
    try:
        result = history.diff(history.find(a, cfg), history.find(b, cfg), cfg, full_code=patch)
    except (history.HistoryError, snapshot.SnapshotError) as e:
        raise fail(str(e)) from None
    if as_json:
        typer.echo(json.dumps(result, indent=2))
        return
    typer.echo(f"a: {result['a']}\nb: {result['b']}")
    if len(result) == 2:
        typer.echo("no differences recorded")
        return
    for name in ("command", "git", "author"):
        if name in result:
            typer.echo(f"\n{name}\n  a: {result[name]['a']}\n  b: {result[name]['b']}")
    for name in ("environment", "inputs", "outputs", "params", "metrics"):
        if name in result:
            typer.echo(f"\n{name}")
            for key, values in result[name].items():
                typer.echo(f"  {key}: {_show(values['a'])} -> {_show(values['b'])}{_delta(values)}")
    if packages := result.get("packages"):
        typer.echo("\npackages")
        for key, version in packages["added"].items():
            typer.echo(f"  + {key}=={version}")
        for key, version in packages["removed"].items():
            typer.echo(f"  - {key}=={version}")
        for key, versions in packages["changed"].items():
            typer.echo(f"  {key}: {versions['a']} -> {versions['b']}")
    if code := result.get("code"):
        typer.echo(f"\ncode  {code['a'][:12]} -> {code['b'][:12]}")
        for label in ("added", "removed", "modified"):
            for path in code[label]:
                typer.echo(f"  {label}: {path}")
        if code["patch"]:
            typer.echo("")
            typer.echo("\n".join(code["patch"]))
        if code["patch_lines_cut"]:
            typer.echo(f"... {code['patch_lines_cut']} more lines (--patch shows all)")


@app.command("note")
def note(
    run: Annotated[str, typer.Argument(metavar="RUN", help="A run ID, or the start of one.")],
    text: Annotated[
        str | None, typer.Argument(help="What was learned. Without it, the run's notes are shown.")
    ] = None,
    verdict: Annotated[
        str | None,
        typer.Option(help="Was the run worth building on: good, bad, or inconclusive."),
    ] = None,
    as_json: AsJson = False,
) -> None:
    """Note what a run showed, or read its notes."""
    from fieldbook import history, notes
    from fieldbook.config import settings

    cfg = settings()
    try:
        found = history.find(run, cfg)
        if text is not None or verdict is not None:
            notes.add(found.run_id, text or "", verdict=verdict, cfg=cfg)
    except (history.HistoryError, notes.NoteError) as e:
        raise fail(str(e)) from None
    history.with_notes([found], cfg)
    if as_json:
        typer.echo(json.dumps({"run_id": found.run_id, "verdict": found.verdict,
                               "notes": found.notes}, indent=2))  # fmt: skip
        return
    typer.echo(
        f"{found.run_id}  {found.command}" + (f"  [{found.verdict}]" if found.verdict else "")
    )
    for entry in found.notes:
        mark = f" [{entry['verdict']}]" if entry["verdict"] else ""
        when = datetime.fromisoformat(entry["time"]).astimezone().strftime("%Y-%m-%d %H:%M")
        by = entry["author"] + (f" [{entry['agent']}]" if entry.get("agent") else "")
        typer.echo(f"  {when}  {by}{mark}  {entry['text']}")
    if not found.notes:
        typer.echo("  no notes")


RunArgument = Annotated[str, typer.Argument(metavar="RUN", help="A run ID, or the start of one.")]


def _find(run: str):  # noqa: ANN202
    from fieldbook import history
    from fieldbook.config import settings

    try:
        return history.find(run, settings())
    except history.HistoryError as e:
        raise fail(str(e)) from None


@app.command("wait")
def wait(
    run: Annotated[
        str | None,
        typer.Argument(metavar="RUN", help="A run ID or its start (default: the latest run)."),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option(metavar="MINUTES", help="Give up after this long (exit code 124)."),
    ] = None,
    as_json: AsJson = False,
) -> None:
    """Wait until a run ends; the exit code says how it ended."""
    import time

    from fieldbook import history
    from fieldbook.config import settings
    from fieldbook.launch import records

    cfg = settings()
    if run is None:
        going = [r for r in history.runs(cfg) if r.recorded]
        if not going:
            raise fail("no run to wait for (runs started with `fieldbook run` can be waited for)")
        live = [r for r in going if r.status in (records.RUNNING, records.LAUNCHING)]
        run = (live or going)[0].run_id
    found = _find(run)
    if not found.recorded:
        raise fail(
            f"run {found.run_id} was not started with `fieldbook run`: it has no record to wait on"
        )
    start = records.read(cfg, [found.run_id]).get(found.run_id, {})
    # never forever: the run's own time limit and its launch, or a day
    limit = timeout * 60 if timeout else (start.get("limit") or 86400) + records.LAUNCH_WINDOW
    deadline = time.monotonic() + limit
    status = found.status
    while status in (records.RUNNING, records.LAUNCHING) and time.monotonic() < deadline:
        time.sleep(min(WAIT_STEP, max(deadline - time.monotonic(), 0)))
        state = records.states(cfg, found.run_id).get(found.run_id)
        status = state.status() if state else status
    found = _find(found.run_id)  # with its metrics, now that it ended
    waiting = status in (records.RUNNING, records.LAUNCHING)
    if as_json:
        typer.echo(json.dumps(found.to_json() | {"timed_out": waiting}, indent=2))
    else:
        metrics = "  ".join(f"{k}={v:.4g}" for k, v in sorted(found.metrics.items())[:6])
        typer.echo(f"{found.run_id}  {status.lower()}"
                   + (f"  exit code {found.exit_code}" if found.exit_code else "")
                   + (f"  {metrics}" if metrics else ""))  # fmt: skip
        if waiting:
            typer.echo("still going: gave up waiting (run `fieldbook log wait` again)", err=True)
        elif status != records.FINISHED:
            typer.echo(f"`fieldbook log why {found.run_id}` says what went wrong", err=True)
    raise typer.Exit(124 if waiting else 0 if status == records.FINISHED else 1)


WAIT_STEP = 10.0  # seconds between looks at a run's state


@app.command("why")
def why(run: RunArgument, as_json: AsJson = False) -> None:
    """Say how a run ended and, if it failed, what went wrong."""
    from fieldbook.config import settings
    from fieldbook.launch import why as reasons

    verdict = reasons.why(_find(run).run_id, settings())
    if as_json:
        typer.echo(json.dumps(verdict.to_json(), indent=2))
        return
    took = f" after {_duration(verdict.seconds)}" if verdict.seconds is not None else ""
    code = f"  (exit code {verdict.exit_code})" if verdict.exit_code else ""
    typer.echo(f"run {verdict.run_id}  {verdict.status.lower()}{took}{code}")
    typer.echo(f"cause: {verdict.cause}")
    for line in verdict.detail:
        typer.echo(f"  {line}")


def _duration(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m {rest}s" if minutes else f"{rest}s"


@app.command("output")
def logs(
    run: RunArgument,
    tail: Annotated[int | None, typer.Option("--tail", help="Only the last N lines.")] = None,
    follow: Annotated[
        bool, typer.Option("--follow", "-f", help="Keep printing until the run ends.")
    ] = False,
    launch: Annotated[
        bool, typer.Option("--launch", help="The cloud launch's output, not the run's.")
    ] = False,
) -> None:
    """Print what a run printed.

    Kept for runs started with `fieldbook run`."""
    import time

    from fieldbook.config import settings
    from fieldbook.launch import records, runlog

    cfg, run_id = settings(), _find(run).run_id
    name = "launch" if launch else "output"
    lines = runlog.read(run_id, cfg, name=name)
    if lines is None and not follow:
        other = "" if launch else " (`--launch` shows a cloud run's launch)"
        raise fail(f"no {name} is kept for run {run_id}{other}")
    shown = 0
    while True:
        lines = lines or []
        for line in lines[shown:] if tail is None or shown else lines[-tail:]:
            typer.echo(line)
        shown = len(lines)
        state = records.states(cfg, run_id).get(run_id)
        going = state is not None and state.status() in (records.RUNNING, records.LAUNCHING)
        if not (follow and going):
            break
        time.sleep(WAIT_STEP)
        lines = runlog.read(run_id, cfg, name=name)
    if follow and lines is not None:  # what arrived with the end
        for line in (runlog.read(run_id, cfg, name=name) or [])[shown:]:
            typer.echo(line)


@app.command("files")
def files(
    run: RunArgument,
    path: Annotated[str | None, typer.Argument(help="A kept file to copy here, as listed.")] = None,
    to: Annotated[Path, typer.Option("--to", help="Where to copy it (default: here).")] = Path(),
    as_json: AsJson = False,
) -> None:
    """List the files a run wrote, or copy one here."""
    from fieldbook.config import settings
    from fieldbook.launch import outputs

    cfg, run_id = settings(), _find(run).run_id
    kept = outputs.manifest(run_id, cfg)
    if path is not None:
        if path not in {f["path"] for f in kept["kept"]}:
            raise fail(
                f"{path!r} is not a kept file of run {run_id} (see `fieldbook log files {run_id}`)"
            )
        typer.echo(outputs.fetch(run_id, path, to, cfg))
        return
    if as_json:
        typer.echo(json.dumps(kept, indent=2))
        return
    if not kept["kept"] and not kept["left_out"]:
        typer.echo("no files kept (the run wrote none, or was not started with `fieldbook run`)")
    for entry in kept["kept"]:
        typer.echo(f"{entry['size']:>12,}  {entry['path']}")
    for entry in kept["left_out"]:
        typer.echo(f"{entry['size']:>12,}  {entry['path']}  (not kept: {entry['why']})")


@app.command("costs")
def costs(
    total: Annotated[
        bool, typer.Option("--total", help="Totals per month and author, and the budget.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many runs, newest first.")] = 20,
    as_json: AsJson = False,
) -> None:
    """Show what cloud runs cost, per run or (--total) per month."""
    from fieldbook import history
    from fieldbook.config import settings
    from fieldbook.launch import agents, budget
    from fieldbook.tracking import costs as recorded

    cfg = settings()
    try:  # price what has ended since the last look, so the numbers are current
        recorded.record(cfg, warn=lambda message: typer.echo(f"warning: {message}", err=True))
    except Exception as e:
        typer.echo(f"warning: could not record cloud run costs: {type(e).__name__}: {e}", err=True)
    if not total:
        paid = [r for r in history.runs(cfg) if r.cost_usd is not None][:limit]
        paid = history.with_details(paid, cfg)
        if as_json:
            typer.echo(json.dumps([
                {"run_id": r.run_id, "started": r.started.isoformat() if r.started else None,
                 "author": r.author, "agent": r.agent, "command": r.command,
                 "cost_usd": r.cost_usd} for r in paid], indent=2))  # fmt: skip
            return
        if not paid:
            typer.echo("no recorded cloud costs (a run's is recorded about 15 minutes after it)")
            return
        rows = [[r.run_id, r.started.astimezone().strftime("%Y-%m-%d %H:%M") if r.started else "",
                 agents.who(r.author, r.agent), _cut(r.command, 44), _usd(r.cost_usd)]
                for r in paid]  # fmt: skip
        _table(["run", "started", "author", "command", "cost"], rows)
        return
    sums = recorded.summary(cfg)
    spent = round(budget.month_spend(cfg), 2) if cfg.budget_monthly_usd else None
    if as_json:
        typer.echo(json.dumps({
            "months": [{"month": m, "author": a, "agent": g, "runs": n, "cost_usd": round(c, 2)}
                       for m, a, g, n, c in sums],
            "budget": {"monthly_usd": cfg.budget_monthly_usd, "spent_usd": spent},
        }, indent=2))  # fmt: skip
        return
    if sums:
        rows, month = [], None
        for row_month, author, agent, count, cost in sums:
            if month is not None and row_month != month:
                rows.append([""] * 5)
            month = row_month
            rows.append([row_month, author, agent or "by hand", str(count), _usd(cost)])
        rows.append(["total", "", "", str(sum(r[3] for r in sums)),
                     _usd(sum(r[4] for r in sums))])  # fmt: skip
        _table(["month", "author", "by", "runs", "cost"], rows)
    else:
        typer.echo("no recorded cloud costs (a run's is recorded about 15 minutes after it)")
    if spent is not None:
        typer.echo(f"\nthis month: about ${spent:,.2f} of the ${cfg.budget_monthly_usd:,.2f} "
                   "budget (recorded costs, and estimates for runs not priced yet)")  # fmt: skip


def _usd(amount: float) -> str:
    """Dollars, to the cent; an amount under a cent keeps the digits that say what it was."""
    return f"${amount:,.2f}" if amount >= 0.01 or amount == 0 else f"${amount:.4f}"


def _table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows, strict=True)]
    for row in [headers, *rows]:
        typer.echo("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())


@app.command("query")
def query(
    sql: Annotated[str, typer.Argument(metavar="SQL", help="A query in DuckDB's dialect.")],
    fmt: Annotated[
        str, typer.Option("--format", metavar="table|markdown|json|csv", help="How to print it.")
    ] = "table",
) -> None:
    """Run SQL over the runs: tables runs, metrics, params, data, notes.

    \b
    fieldbook log query "select run_id, metrics['acc'] from runs limit 5"
    fieldbook log query "select * from notes" --format markdown
    """
    from fieldbook import history

    if fmt not in ("table", "markdown", "json", "csv"):
        raise fail("--format is table, markdown, json, or csv")
    try:
        result = history.table(sql)
    except history.HistoryError as e:
        raise fail(str(e)) from None
    names, rows = result.column_names, result.to_pylist()
    if fmt == "json":
        typer.echo(json.dumps([{k: _plain(v) for k, v in row.items()} for row in rows], indent=2))
        return
    cells = [[_cell(row[name]) for name in names] for row in rows]
    if fmt == "csv":
        import csv
        import sys

        csv.writer(sys.stdout).writerows([names, *cells])
    elif fmt == "markdown":
        for line in [names, ["---"] * len(names), *cells]:
            typer.echo("| " + " | ".join(c.replace("|", "\\|") for c in line) + " |")
    else:
        _table(names, cells)


def _plain(value: Any) -> Any:
    """A query result's value as JSON: times as text, maps as objects."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list) and value and all(isinstance(v, tuple) for v in value):
        return {k: _plain(v) for k, v in value}
    return [_plain(v) for v in value] if isinstance(value, list) else value


def _cell(value: Any) -> str:
    """A query result's value for a table: short, on one line."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d %H:%M")
    if isinstance(value, list):
        return ", ".join(f"{v[0]}={_cell(v[1])}" if isinstance(v, tuple) else _cell(v)
                         for v in value)  # fmt: skip
    return str(value).replace("\n", " ")


@app.command("rm")
def rm(
    run: Annotated[
        str | None, typer.Argument(metavar="[RUN]", help="A run ID, or the start of one.")
    ] = None,
    old: Annotated[
        bool, typer.Option("--old", help="Every run older than --days, and unused images.")
    ] = False,
    days: Annotated[
        float | None, typer.Option("--days", min=0, help="With --old: how old (default: 30).")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be deleted and freed.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Free a run's checkpoints, output, and files; its record stays."""
    from datetime import timedelta

    from fieldbook.config import settings
    from fieldbook.launch import retention

    if (run is None) == (not old):
        raise fail("give a RUN, or --old for every run older than --days")
    if days is not None and not old:
        raise fail("--days applies to --old")
    cfg = settings()
    if old:
        keep = timedelta(days=days) if days is not None else retention.RETENTION
        report = retention.prune(cfg, retention=keep, dry_run=True)
    else:
        report = retention.of_run(_find(run).run_id, cfg, dry_run=True)
    typer.echo(_freed(report, final=dry_run))
    if dry_run or not report.total:
        return
    if not yes:
        typer.confirm("Delete these? This cannot be undone", abort=True)
    if old:
        report = retention.prune(cfg, retention=keep)
    else:
        report = retention.of_run(_find(run).run_id, cfg)
    typer.echo(f"deleted; freed {_size(report.total)}")


def _freed(report, *, final: bool) -> str:  # noqa: ANN001
    lines = []
    for what, found in (("checkpoints", report.checkpoints), ("output and files", report.kept),
                        ("images", report.images)):  # fmt: skip
        if found:
            names = ", ".join(sorted(found)[:4]) + (", ..." if len(found) > 4 else "")
            lines.append(f"{what}: {len(found)} ({_size(sum(found.values()))}): {names}")
    if report.spared:
        lines.append(f"kept: {len(report.spared)} old runs noted as good")
    if report.image_error:
        lines.append(f"images not looked at: {report.image_error}")
    if not report.total:
        lines.append("nothing to delete")
    elif final:
        lines.append(f"would free {_size(report.total)}")
    return "\n".join(lines)


def _size(count: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:,.0f} {unit}" if unit == "B" else f"{count:,.1f} {unit}"
        count /= 1024
    return ""


@app.command("code")
def code(
    run: Annotated[str, typer.Argument(metavar="RUN|HASH", help="A run, or a snapshot's hash.")],
    directory: Annotated[Path, typer.Argument(metavar="DIR", help="A new or empty folder.")],
) -> None:
    """Restore the code a run used into a new or empty folder."""
    from fieldbook.code import snapshot

    try:
        digest = snapshot.checkout(_recorded(run, "code"), directory)
    except snapshot.SnapshotError as e:
        raise fail(str(e)) from None
    typer.echo(f"{digest[:12]} -> {directory}")


@app.command("packages")
def packages(
    run: Annotated[str, typer.Argument(metavar="RUN|HASH", help="A run, or a list's hash.")],
) -> None:
    """Print the installed packages a run recorded."""
    from fieldbook.code import snapshot

    try:
        _, text = snapshot.read_packages(_recorded(run, "packages"))
    except snapshot.SnapshotError as e:
        raise fail(str(e)) from None
    typer.echo(text, nl=False)


def _recorded(run: str, what: str) -> str:
    """The hash of a run's code snapshot or package list; what was given, if it names no
    run (a hash, as dataset versions show for the code that saved them)."""
    from fieldbook import history
    from fieldbook.config import settings

    try:
        found = history.find(run, settings())
    except history.HistoryError:
        return run
    if what == "code":
        return found.code or run
    return (json.loads(found.env) if found.env else {}).get("packages") or run


@app.command("ui")
def ui(
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the UI in the browser.")
    ] = True,
) -> None:
    """Open the MLflow UI (metrics, charts, models) until Ctrl-C."""
    import sys

    from fieldbook import ui as browser
    from fieldbook.code import environments
    from fieldbook.config import settings
    from fieldbook.tracking import costs, server

    candidates = dict.fromkeys([str(environments.find(Path.cwd()).python), sys.executable])
    found = environments.query_interpreter
    python = next((p for p in candidates if found(p).get("mlflow_sql")), None)
    if python is None:
        raise fail("MLflow is not installed: pip install 'fieldbook[tracking]'")
    costs.record_in_background(settings(), lambda message: typer.echo(message, err=True))
    try:
        exit_code = server.serve(settings(), python, open_browser=open_browser)
    except browser.UIError as e:
        raise fail(str(e)) from None
    raise typer.Exit(exit_code)


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _show(value: Any) -> str:
    return "—" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def _delta(values: dict[str, Any]) -> str:
    a, b = values["a"], values["b"]
    if isinstance(a, float) and isinstance(b, float):
        return f"  ({b - a:+.4g})"
    return ""
