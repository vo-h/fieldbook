"""`fieldbook access`: who may use the project, following its GitHub repository
(sharing.md §2)."""

from pathlib import Path
from typing import Annotated

import typer

from fieldbook.cli._common import fail

app = typer.Typer(help="Members: access follows the project's GitHub repository.")


@app.command("sync")
def sync_command(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would change.")] = False,
    renew: Annotated[
        list[str] | None,
        typer.Option(help="GitHub login to give a new database password; repeatable."),
    ] = None,
) -> None:
    """Match members' access to the GitHub repository's collaborators.

    \b
    Write and above on GitHub: member. Read or Triage: viewer.
    Database: a login per person, encrypted to their GitHub SSH keys in
    .fieldbook-members/. AWS: the IAM groups fieldbook-<project>-write and -read
    hold the IAM users named after their GitHub logins. RunPod: lists keys to
    create or delete.
    Run by the owner, with the owner's database login and AWS rights.
    """
    from fieldbook.code import snapshot
    from fieldbook.config import settings
    from fieldbook.members import MembersError, sync

    try:
        lines = sync.sync(
            snapshot.repo_root(Path.cwd()), settings(), dry_run=dry_run, renew=renew or ()
        )
        for line in lines:
            typer.echo(line)
    except MembersError as e:
        raise fail(str(e)) from None


@app.command("aws-policy")
def aws_policy(
    read: Annotated[bool, typer.Option("--read", help="The viewers' policy.")] = False,
) -> None:
    """Print the members' AWS policy."""
    import json

    from fieldbook.clouds import aws
    from fieldbook.config import settings

    cfg = settings()
    if not aws.roles_apply(cfg):
        raise fail("the project's storage root is not an S3 bucket")
    writers, readers = aws.member_policies(cfg)
    typer.echo(json.dumps(readers if read else writers, indent=2))


@app.command("runpod-key")
def runpod_key(
    login: Annotated[str, typer.Argument(help="The member's GitHub login.")],
) -> None:
    """Encrypt a member's RunPod key to them."""
    from fieldbook.code import snapshot
    from fieldbook.config import settings
    from fieldbook.members import MembersError, sync

    login = login.lower()
    name = sync.runpod_key_name(settings(), login)
    key = typer.prompt(f"RunPod key {name} for {login}", hide_input=True)
    try:
        path = sync.runpod_key(snapshot.repo_root(Path.cwd()), login, key.strip())
    except MembersError as e:
        raise fail(str(e)) from None
    typer.echo(f"encrypted to {path}; commit it so {login} receives it")
