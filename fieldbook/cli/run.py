"""`fieldbook run` (and `--rerun`, runs.md §17) and `fieldbook build`."""

import json
from pathlib import Path
from typing import Annotated

import typer

from fieldbook.cli._common import fail

# Arguments after the target go to it (registered on the main app as `fieldbook run`).
# Options it does not know are let through for `--rerun`, whose changes are arguments of
# the script (`--rerun RUN --lr 0.01`); for a new run they are refused below.
RUN_SETTINGS = {
    "allow_extra_args": True,
    "allow_interspersed_args": False,
    "ignore_unknown_options": True,
}


def run(
    ctx: typer.Context,
    target: Annotated[
        str | None, typer.Argument(metavar="SCRIPT.py|TASK.yaml|COMMAND", show_default=False)
    ] = None,
    rerun: Annotated[
        str | None,
        typer.Option(
            "--rerun",
            metavar="RUN",
            help="Run an earlier run again from its record (a run ID, or the start of one); "
            "arguments that follow change it.",
        ),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Seed (default: random).")] = None,
    experiment: Annotated[
        str | None,
        typer.Option(help="MLflow experiment (default: the experiment setting, else the project)."),
    ] = None,
    max_snapshot_size: Annotated[
        int,
        typer.Option(
            "--max-snapshot-size", metavar="MB", help="Refuse code snapshots larger than this."
        ),
    ] = 100,
    smoke: Annotated[
        bool,
        typer.Option(
            "--smoke",
            help="Smoke run: datasets return a sample, new versions go to a scratch area, and "
            "MLflow runs go to the '<experiment>-smoke' experiment.",
        ),
    ] = False,
    sample: Annotated[
        int | None,
        typer.Option(
            "--sample", min=1, metavar="N", help="With --smoke: items per folder (default: 100)."
        ),
    ] = None,
    deterministic: Annotated[
        bool,
        typer.Option(
            "--deterministic",
            help="Make PyTorch use deterministic algorithms (slower; some operations fail).",
        ),
    ] = False,
    keep: Annotated[
        bool, typer.Option("--keep", help="With --smoke: keep the scratch area for inspection.")
    ] = False,
    on: Annotated[
        str, typer.Option("--on", metavar="local|remote", help="Where to run (default: local).")
    ] = "local",
    gpus: Annotated[
        str | None,
        typer.Option(
            "--gpus",
            metavar="SPEC",
            help="Remote: a GPU by name (A10G, RTX4090:2), by memory "
            "(24GB+, 48GB+:2, nvidia:24GB), or several separated by commas, tried "
            "in order. The cheapest that fits and has capacity is launched.",
        ),
    ] = None,  # fmt: skip
    max_price: Annotated[
        float | None,
        typer.Option(
            "--max-price", min=0.01, metavar="USD", help="Remote: most a machine may cost an hour."
        ),
    ] = None,  # fmt: skip
    cloud: Annotated[
        str | None, typer.Option(help="Remote: cloud, e.g. aws (default: remote_cloud).")
    ] = None,
    nodes: Annotated[int | None, typer.Option(min=1, help="Remote: number of machines.")] = None,
    max_hours: Annotated[
        float | None,
        typer.Option(
            min=0.05,
            metavar="HOURS",
            help="Remote: stop the run after this long (default: 24, or what budget_run_usd buys).",
        ),
    ] = None,  # fmt: skip
    secret: Annotated[
        list[str] | None,
        typer.Option(metavar="NAME", help="Remote: also send this variable (repeatable)."),
    ] = None,
    no_build: Annotated[
        bool, typer.Option("--no-build", help="Remote: stop if the environment image is missing.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Remote: show machines, price, time limit, and budget; launch nothing.",
        ),
    ] = False,  # fmt: skip
    as_json: Annotated[
        bool, typer.Option("--json", help="With --dry-run or --rerun: machine-readable output.")
    ] = False,
    again: Annotated[
        bool,
        typer.Option(
            "--allow-repeat",
            help="Run even if an earlier run had the same code, "
            "environment, command, and data (a cloud run is otherwise not launched).",
        ),
    ] = False,  # fmt: skip
    run_id: Annotated[str | None, typer.Option(hidden=True)] = None,  # set by cloud runs
    code: Annotated[str | None, typer.Option(hidden=True)] = None,
) -> None:
    """Run a script, task, or command, recorded; here or in the cloud.

    Arguments after the target are passed to it. Files matched by .gitignore or
    .fieldbookignore are left out of the code snapshot.

    \b
    fieldbook run clean.py --min-size 10
    fieldbook run --seed 1 train.py
    fieldbook run --smoke --sample 20 clean.py
                                         test on 20 items per folder
    fieldbook run task.yaml
    fieldbook run torchrun --nproc-per-node 4 train.py
                                         one process per GPU
    fieldbook run --on remote --gpus A10G:1 train.py
                                         on a cloud GPU
    fieldbook run --on remote --gpus 24GB+ --max-price 1 train.py
                                         any that fits
    fieldbook run --rerun 20261012-1432  the same code, seed, and data again
    fieldbook run --rerun 20261012-1432 --lr 0.01
                                         with one thing changed
    """
    import secrets as random

    from fieldbook.launch import LaunchError, local, tasks

    extra = [*([target] if target is not None else []), *ctx.args]
    if rerun is not None:
        given = {"--experiment": experiment, "--smoke": smoke, "--sample": sample, "--keep": keep,
                 "--deterministic": deterministic, "--on": on != "local", "--gpus": gpus,
                 "--max-price": max_price, "--cloud": cloud, "--nodes": nodes,
                 "--max-hours": max_hours, "--secret": secret, "--no-build": no_build,
                 "--dry-run": dry_run, "--allow-repeat": again}  # fmt: skip
        if used := [name for name, value in given.items() if value]:
            raise fail(f"{', '.join(used)} cannot be used with --rerun: a rerun takes its "
                       "settings from the run's record (only --seed and --json apply)")  # fmt: skip
        _rerun(rerun, extra, seed, as_json)
    if target is None:
        raise fail("give a script, a task file, or a command to run (or --rerun RUN)")
    if target.startswith("-"):
        raise fail(f"no such option: {target}")
    if (sample is not None or keep) and not smoke:
        raise fail("--sample and --keep apply to --smoke")
    if on not in ("local", "remote"):
        raise fail("--on is local or remote")
    remote_only = {"--gpus": gpus, "--max-price": max_price, "--cloud": cloud, "--nodes": nodes,
                   "--max-hours": max_hours, "--secret": secret, "--dry-run": dry_run,
                   "--no-build": no_build or None}  # fmt: skip
    if on == "local" and (used := [k for k, v in remote_only.items() if v]):
        raise fail(f"{', '.join(used)} apply to --on remote")
    try:
        if target.endswith((".yaml", ".yml")):
            task = tasks.load_task(Path(target), ctx.args)
        elif target.endswith(".py"):
            task = tasks.script_task(Path(target), ctx.args)
        else:
            import shutil

            from fieldbook.cli.history import SUBCOMMANDS

            if target in SUBCOMMANDS and not shutil.which(target):
                raise fail(f"`fieldbook run {target}` would run a command named {target!r}: "
                           f"did you mean `fieldbook log {target}`?")  # fmt: skip
            task = tasks.command_task([target, *ctx.args])
        if on == "remote":
            from fieldbook.config import settings
            from fieldbook.launch import cloud as cloud_launch  # `cloud` is the --cloud option

            cfg = settings()
            # Decided here, so every machine of a multi-node run uses the same values.
            run_args = ["--seed", str(seed if seed is not None else random.randbelow(2**31))]
            run_args += ["--experiment", experiment or cfg.experiment or cfg.project]
            if smoke:
                run_args += ["--smoke", "--sample", str(sample or 100)] + (["--keep"] * keep)
            if deterministic:
                run_args.append("--deterministic")
            options = cloud_launch.RemoteOptions(
                gpus=gpus, max_price=max_price, cloud=cloud, nodes=nodes, max_hours=max_hours,
                secrets=secret or [], build=not no_build, run_args=run_args, seed=seed, again=again,
                max_snapshot_mb=max_snapshot_size,
            )  # fmt: skip
            if dry_run:
                planned = cloud_launch.plan(task, [target, *ctx.args], options)
                typer.echo(json.dumps(planned, indent=2) if as_json else _show_plan(planned))
                raise typer.Exit(0 if planned["would_launch"] else 1)
            cloud_launch.launch(task, [target, *ctx.args], options)
            return
        code = local.run(
            task,
            seed=seed,
            experiment=experiment,
            max_snapshot_mb=max_snapshot_size,
            smoke=(sample or 100) if smoke else None,
            deterministic=deterministic,
            keep_scratch=keep,
            run_id=run_id,
            code=code,
            again=again,
        )
    except LaunchError as e:
        raise fail(str(e)) from None
    raise typer.Exit(code)


def _rerun(run: str, changes: list[str], seed: int | None, as_json: bool) -> None:
    """Run an earlier run again from its record, and say whether it came out the same."""
    from fieldbook import history
    from fieldbook.config import settings
    from fieldbook.launch import LaunchError
    from fieldbook.launch import rerun as again

    cfg = settings()
    try:
        original = history.find(run, cfg)
        result = again.rerun(original, changes, seed=seed, cfg=cfg)
    except (history.HistoryError, LaunchError) as e:
        raise fail(str(e)) from None
    if as_json:
        typer.echo(json.dumps({
            "original": original.run_id, "run_id": result.run_id, "exit_code": result.exit_code,
            "verdict": result.verdict, "changed": result.changed, "same": result.same,
            "differs": result.differs, "outputs": result.outputs,
        }, indent=2))  # fmt: skip
    else:
        typer.echo(f"rerun of {original.run_id} as {result.run_id}: {result.verdict}", err=True)
        for ref, how in result.outputs.items():
            typer.echo(f"  {ref}: {how}", err=True)
    raise typer.Exit(result.exit_code)


def _show_plan(plan: dict) -> str:
    """A dry run's plan, for reading."""
    from fieldbook.launch.budget import span

    lines = [f"on {plan['cloud']}" + (f", {plan['nodes']} machines" if plan["nodes"] > 1 else "")]
    for machine in plan["candidates"]:
        lines.append(f"  {machine['machine']:<16} ${machine['usd_per_hour']:,.2f} an hour")
    if not plan["candidates"]:
        lines.append("  no priced machine fits")
    if plan["time_limit_seconds"]:
        cost = f", at most ${plan['cost_max_usd']:,.2f}" if plan["cost_max_usd"] is not None else ""
        lines.append(f"time limit {span(plan['time_limit_seconds'])}{cost}")
    budget = plan["budget"]
    if budget["monthly_usd"]:
        lines.append(f"budget: about ${budget['spent_usd']:,.2f} of ${budget['monthly_usd']:,.2f} "
                     f"spent this month, ${budget['left_usd']:,.2f} left")  # fmt: skip
    else:
        lines.append("budget: none set for the month (budget_monthly_usd)")
    lines.append("image: built" if plan["image"]["built"] else
                 "image: not built yet; a launch builds it first (a few minutes)")  # fmt: skip
    if plan["repeats"]:
        lines.append(f"repeats an earlier run: {plan['repeats']['summary']}")
    for reason in plan["refused"]:
        lines.append(f"would not launch: {reason}")
    lines.append("nothing was launched" + ("" if plan["refused"] else "; it would be"))
    return "\n".join(lines)


def build(
    local: Annotated[
        bool, typer.Option("--local", help="Build with local Docker instead of a cloud machine.")
    ] = False,
) -> None:
    """Build the project's environment image for cloud runs.

    The image holds Python and the packages in uv.lock, not the code, for x86-64 machines
    (what cloud runs use). `fieldbook run --on remote` builds it when needed; run this to build
    ahead of time.
    """
    from fieldbook import images
    from fieldbook.code import snapshot
    from fieldbook.config import settings
    from fieldbook.launch import LaunchError

    try:
        repo = snapshot.repo_root(Path.cwd())
        typer.echo(images.image(repo, settings(), local=local).ref)
    except (LaunchError, snapshot.SnapshotError) as e:
        raise fail(str(e)) from None
    except Exception as e:  # registry and cloud errors: report, not a traceback
        raise fail(f"{type(e).__name__}: {e}") from None
