"""`fieldbook config`: show settings, or change one (`set`, `unset`, `get`), like `git config`."""

from typing import Annotated

import typer

from fieldbook.cli._common import fail

app = typer.Typer(help="Show settings, or change one.", invoke_without_command=True)

Local = Annotated[
    bool,
    typer.Option("--local", help="This machine only: .fieldbook/config.toml, not committed."),
]


@app.callback(invoke_without_command=True)
def config_command(ctx: typer.Context) -> None:
    """Show every setting, its value, and its source; or change one with `set`, `unset`.

    \b
    fieldbook config                     every setting, its value, and its source
    fieldbook config get storage_root    one effective value, for scripts
    fieldbook config set storage_root s3://b/fieldbook
                                         write it to the right file
    fieldbook config set image_apt ffmpeg,git
                                         lists as a,b,c
    fieldbook config unset remote_cloud  remove it from its file

    Settings go to the project file (committed, shared by everyone on the project), or with
    --local to .fieldbook/config.toml (this machine only). FIELDBOOK_<NAME> variables override both
    for one command.
    """
    if ctx.invoked_subcommand is not None:
        return

    from fieldbook.config import ConfigError, settings

    try:
        settings()
    except ConfigError as e:
        raise fail(f"{e} (fix it with `fieldbook config set` or `unset`)") from None
    import sys
    from pathlib import Path

    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from fieldbook import config, credentials

    homes = {str(Path.home()), str(Path.home().resolve())}

    def short(text: str) -> str:
        """A path under the home folder as ~/..., escaped ("[tool.fieldbook]" is not markup)."""
        text = text.removeprefix("file://")
        home = next((h for h in homes if text == h or text.startswith(h + "/")), None)
        return escape("~" + text[len(home) :] if home else text)

    rows = config.effective()
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="bold")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_column("Source", no_wrap=True)
    table.add_column("Description", style="dim")
    group = None
    for name, value, source in rows:
        if config.group_of(name) != group:  # a heading row per group
            group = config.group_of(name)
            table.add_row(f"[bold]{escape(group)}[/bold]", "", "", "")
        table.add_row(
            f"  {name}",
            short(value) if value is not None else "[dim]—[/dim]",
            f"[dim]{source}[/dim]" if source == "default" else short(source),
            config.Settings.model_fields[name].description,
        )
    console = Console(width=None if sys.stdout.isatty() else 200, highlight=False)
    console.print(table)
    secrets, values = credentials.project_values(Path.cwd())
    names = sorted(values)
    found_project = config.project()
    files = Table(box=None, show_header=False, pad_edge=False)
    files.add_column(style="bold", no_wrap=True)
    files.add_column(no_wrap=True)
    none = "[dim]none (run `fieldbook init` here)[/dim]"
    files.add_row("Project file", short(str(found_project.file)) if found_project.file else none)
    local = found_project.local_file
    files.add_row("Local file", short(str(local)) if local.is_file() else "[dim]none[/dim]")
    found = (
        f"{short(str(secrets))}: {', '.join(names) or 'empty'}" if secrets else "[dim]none[/dim]"
    )
    files.add_row("Credentials", found)
    console.print(files)


@app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Setting name, e.g. storage_root.")],
    value: Annotated[str, typer.Argument(help="New value; lists as a,b,c.")],
    local: Local = False,
) -> None:
    """Set a setting in the project file (--local: this machine's)."""
    from fieldbook.config import ConfigError, edit

    try:
        typer.echo(edit.set_value(key, value, local=local))
    except ConfigError as e:
        raise fail(str(e)) from None


@app.command("unset")
def config_unset(
    key: Annotated[str, typer.Argument(help="Setting name.")], local: Local = False
) -> None:
    """Remove a setting from its file."""
    from fieldbook.config import ConfigError, edit

    try:
        typer.echo(edit.unset_value(key, local=local))
    except ConfigError as e:
        raise fail(str(e)) from None


@app.command("get")
def config_get(key: Annotated[str, typer.Argument(help="Setting name.")]) -> None:
    """Print a setting's effective value (nothing if it is unset)."""
    from fieldbook.config import ConfigError, edit

    try:
        value = edit.get_value(key)
    except ConfigError as e:
        raise fail(str(e)) from None
    if value is not None:
        typer.echo(value)
