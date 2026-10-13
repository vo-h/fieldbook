"""Command-line interface: the `fieldbook` command. One module per command group."""

import os
from typing import Annotated

import typer

import fieldbook
from fieldbook.cli import access, history
from fieldbook.cli import config as config_commands
from fieldbook.cli import data as data_commands
from fieldbook.cli import run as run_commands
from fieldbook.cli._common import fail

app = typer.Typer(
    no_args_is_help=True, help="Lightweight toolkit for machine-learning experimentation."
)
# Commands that use a project's data or runs; `init`, `config`, and `version` work anywhere.
PROJECT_COMMANDS = {
    "data",
    "run",
    "build",
    "access",
    "log",
}
app.add_typer(data_commands.app, name="data")
app.add_typer(history.app, name="log")
app.add_typer(config_commands.app, name="config")
app.add_typer(access.app, name="access")


@app.callback()
def setup(ctx: typer.Context) -> None:
    """Lightweight toolkit for machine-learning experimentation."""
    from pathlib import Path

    from fieldbook import credentials
    from fieldbook.config import require_project, settings

    credentials.load_into_environ(
        Path.cwd()
    )  # credentials for fieldbook's own commands, as for runs
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # MLflow's start-up advertisement
    if ctx.invoked_subcommand == "config":
        return  # `fieldbook config set` and `unset` must work when a settings file is invalid
    try:
        settings()  # report invalid settings before any command runs
        if ctx.invoked_subcommand in PROJECT_COMMANDS and not _asks_for_help():
            require_project()
    except ValueError as e:
        raise fail(str(e)) from None


def _asks_for_help() -> bool:
    """`fieldbook data --help` works outside a project."""
    import sys

    return any(arg in ("--help", "-h") for arg in sys.argv[1:])


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(fieldbook.__version__)


@app.command("init")
def init_command(
    agents: Annotated[
        bool,
        typer.Option(
            "--agents",
            help="Also add instructions for coding agents to CLAUDE.md and AGENTS.md.",
        ),
    ] = False,
) -> None:
    """Set up the current repository as a project. Idempotent.

    Creates only what is missing: project settings, .fieldbookignore, and .env and .fieldbook/
    in .gitignore. Writes your project credentials to .env if the owner gave you access
    (`fieldbook access sync`). Checks gitleaks, the storage root, and the tracking database. When
    cloud runs are set up, shows the IAM roles for cloud machines and creates them after
    asking.
    """
    import sys

    from fieldbook import setup_project

    for line in setup_project.init(agent=agents):
        typer.echo(line)
    ask = lambda question: sys.stdin.isatty() and typer.confirm(question)  # noqa: E731
    try:
        setup_project.iam_roles(ask=ask, say=typer.echo)
    except setup_project.SetupError as e:
        raise fail(str(e)) from None


app.command("run", context_settings=run_commands.RUN_SETTINGS)(run_commands.run)
app.command("build")(run_commands.build)


@app.command("launch-ended", hidden=True)
def launch_ended(run_id: str) -> None:
    """Run by a cloud machine when its job is over, for the budget (cloud-runs.md §11)."""
    from fieldbook.launch import budget

    budget.ended(run_id)


def main() -> None:
    """The `fieldbook` command: errors from the setup or the outside world print one line, not a
    traceback (set FIELDBOOK_DEBUG=1 to see it); bugs keep their traceback for reports."""
    import sys

    from fieldbook.cli._common import operational_errors

    try:
        app()
    except operational_errors() as e:
        if os.environ.get("FIELDBOOK_DEBUG"):
            raise
        message = str(e) or type(e).__name__
        if not isinstance(e, (OSError,)) and type(e).__module__.startswith("botocore"):
            message = f"{type(e).__name__}: {message}"
        print(f"error: {message}", file=sys.stderr)
        sys.exit(1)
