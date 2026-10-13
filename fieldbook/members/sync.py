"""`fieldbook access sync`: the project's access follows its GitHub repository; and the member's
side, run by `fieldbook init` (sharing.md §2).

The owner (whoever runs `sync`, with the owner's database login and AWS rights) is skipped:
they already have access. Everything is idempotent; `dry_run` reports without changing.
"""

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fieldbook.config import Settings
from fieldbook.members import MembersError, database, github, sealed


@dataclass(frozen=True)
class Person:
    login: str
    writes: bool
    keys: tuple[str, ...]  # what their credentials are encrypted to
    stale: bool  # keys differ from what their files were encrypted to

    @property
    def role(self) -> str:
        return "member" if self.writes else "viewer"


def sync(
    repo: Path, cfg: Settings, *, dry_run: bool = False, renew: Sequence[str] = ()
) -> list[str]:
    """Bring database logins, AWS groups, and member folders in line with the repository's
    collaborators. `renew`: GitHub logins to give a new database password (e.g. one that
    leaked). Returns a line per change or finding."""
    collaborators = github.collaborators(repo)
    me = github.current_login(repo)
    renew = {login.lower() for login in renew}
    if unknown := sorted(renew - {c.login for c in collaborators} | renew & {me}):
        raise MembersError(f"cannot renew {', '.join(unknown)}: not a collaborator, or yourself")
    people = []
    for c in collaborators:
        if c.login == me:
            continue
        member = sealed.folder(repo, c.login)
        keys = tuple(sealed.recipients(list(c.keys)))
        people.append(Person(c.login, c.writes, keys, list(keys) != sealed.sealed_to(member)))
    everyone = {c.login for c in collaborators}
    members = sum(c.writes for c in collaborators)
    lines = [f"GitHub: {members} members, {len(collaborators) - members} viewers "
             f"(you, {me}, are skipped)"]  # fmt: skip
    lines += [f"{p.login}: no credentials yet: they need {sealed.KEY_HINT}"
              for p in people if not p.keys]  # fmt: skip
    before = _folder_state(repo)
    lines += _database(repo, cfg, people, everyone, dry_run, renew)
    lines += _aws(cfg, people, everyone, dry_run)
    lines += _runpod(repo, cfg, people, everyone, dry_run)
    lines += _leavers(repo, everyone, dry_run)
    if dry_run:
        lines.append("dry run: nothing changed")
    elif _folder_state(repo) != before:
        lines.append(f"commit {sealed.MEMBERS_DIR}/ so members receive their credentials")
    return lines


def runpod_key(repo: Path, login: str, key: str) -> Path:
    """Encrypt a member's RunPod key (created by the owner in the RunPod console) to them."""
    member = sealed.folder(repo, login)
    found = [c for c in github.collaborators(repo) if c.login == login]
    if not found:
        raise MembersError(f"{login} has no access to the repository on GitHub")
    keys = sealed.recipients(list(found[0].keys))
    if not keys:
        raise MembersError(f"{login} needs {sealed.KEY_HINT}")
    others = [p for p in member.glob("*.age") if p.name != "runpod.age"]
    if others and keys != sealed.sealed_to(member):  # all of a member's files share keys
        raise MembersError(f"{login}'s keys changed: run `fieldbook access sync` first")
    return sealed.seal(member, "runpod", {"RUNPOD_API_KEY": key}, keys)


def _has_database_login(repo: Path) -> bool:
    import os

    from fieldbook import credentials

    env = repo / ".env"
    values = credentials.load(env) if env.is_file() else {}
    return bool(os.environ.get("PGPASSWORD") or values.get("PGPASSWORD"))


def runpod_key_name(cfg: Settings, login: str) -> str:
    return f"fieldbook-{cfg.project}-{login}"


def receive(repo: Path, state: Path) -> tuple[str | None, dict[str, str]]:
    """The member's side (`fieldbook init`): decrypt this person's credentials into `.env`. Files
    already received are not decrypted again, so a key's passphrase is asked only when
    something changed. Returns a line (None when the project has no member folders) and the
    values written."""
    if not (repo / sealed.MEMBERS_DIR).is_dir():
        return None, {}
    found = sealed.find_own(repo)
    if found is None:
        if _has_database_login(repo):
            return None, {}  # the owner, or someone set up by hand: nothing to receive
        hint = "you need an SSH key that is on your GitHub profile and a file in ~/.ssh/; "
        hint += "then the owner runs `fieldbook access sync`"
        return f"project credentials: none for this machine's keys ({hint})", {}
    member, identity = found
    from dotenv import set_key

    from fieldbook import credentials

    seen_path = state / "members.json"
    try:
        seen = json.loads(seen_path.read_text())
    except (OSError, ValueError):
        seen = {}
    env = repo / ".env"
    present = credentials.load(env) if env.is_file() else {}
    received: dict[str, str] = {}
    for path in sorted(member.glob("*.age")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        known = seen.get(path.name)
        unchanged = isinstance(known, dict) and known.get("digest") == digest
        if unchanged and all(name in present for name in known["names"]):
            continue
        values = sealed.unseal(path, identity)
        for name, value in values.items():
            # A file seen before only puts back what `.env` lost; a new one replaces.
            if not unchanged or name not in present:
                set_key(env, name, value, quote_mode="never")
                received[name] = value
        seen[path.name] = {"digest": digest, "names": sorted(values)}
    if not received:
        return f"project credentials: up to date ({member.name})", {}
    state.mkdir(parents=True, exist_ok=True)
    seen_path.write_text(json.dumps(seen, indent=2))
    return f"project credentials: {', '.join(received)} written to .env ({member.name})", received


# --- each resource ---------------------------------------------------------------------


def _database(
    repo: Path, cfg: Settings, people: list[Person], everyone: set[str], dry_run: bool,
    renew: set[str] = frozenset(),
) -> list[str]:  # fmt: skip
    from fieldbook.tracking import store

    uri = store.configured_uri(repo)
    if not store.is_postgres(uri):
        return ["database: not a shared Postgres database; no logins to manage"]
    database.check_address(uri)
    db = database.Database(uri, cfg.project)
    lines = []
    with db.connect():
        db.check_tables()
        if not dry_run:
            db.prepare()
        existing = db.logins()
        for p in people:
            if not p.keys:
                continue
            user = database.login_name(cfg.project, p.login)
            group = db.writers if p.writes else db.readers
            sealed_file = sealed.folder(repo, p.login) / "postgres.age"
            if user not in existing:
                lines.append(f"{p.login}: database login created ({p.role})")
                if not dry_run:
                    _seal_login(repo, p, db.add(user, group))
                continue
            if existing[user] != group:
                lines.append(f"{p.login}: database login is now a {p.role}'s")
                if not dry_run:
                    db.move(user, group)
            if p.login in renew:
                lines.append(f"{p.login}: new database password encrypted (renewed)")
                if not dry_run:
                    _seal_login(repo, p, db.new_password(user))
            elif p.stale or not sealed_file.is_file():
                lines.append(f"{p.login}: keys changed: new database password encrypted")
                if not dry_run:
                    _seal_login(repo, p, db.new_password(user))
        for user in sorted(existing):
            login = user.partition("/")[2]
            if login not in everyone:
                lines.append(f"{login}: database login removed")
                if not dry_run:
                    db.remove(user)
    return lines


def _seal_login(repo: Path, person: Person, login: database.Login) -> None:
    values = {"PGUSER": login.user, "PGPASSWORD": login.password}
    sealed.seal(sealed.folder(repo, person.login), "postgres", values, list(person.keys))


def _aws(cfg: Settings, people: list[Person], everyone: set[str], dry_run: bool) -> list[str]:
    from fieldbook.clouds import aws

    if not aws.roles_apply(cfg):
        return []
    writers, readers = aws.group_names(cfg.project)
    write_policy, read_policy = aws.member_policies(cfg)
    if not dry_run:
        aws.ensure_group(writers, write_policy)
        aws.ensure_group(readers, read_policy)
    current = {group: _users(group) for group in (writers, readers)}  # lowercase: IAM's name
    lines = []
    for p in people:
        target, other = (writers, readers) if p.writes else (readers, writers)
        if p.login not in current[target]:
            name = aws.user_name(p.login)
            if name is None:
                lines.append(f"{p.login}: no IAM user named {p.login}: create one, or give "
                             "their AWS identity the policy `fieldbook access aws-policy` "
                             "prints")  # fmt: skip
                continue
            lines.append(f"{p.login}: added to AWS group {target}")
            if not dry_run:
                aws.add_to_group(target, name)
        if p.login in current[other]:
            lines.append(f"{p.login}: removed from AWS group {other}")
            if not dry_run:
                aws.remove_from_group(other, current[other][p.login])
    for group, users in current.items():
        for login in sorted(set(users) - everyone):
            lines.append(f"{users[login]}: removed from AWS group {group}")
            if not dry_run:
                aws.remove_from_group(group, users[login])
    return lines


def _users(group: str) -> dict[str, str]:
    """A group's users, by lowercase name (GitHub logins are compared in lowercase)."""
    from fieldbook.clouds import aws

    try:
        return {user.lower(): user for user in aws.group_users(group)}
    except Exception as e:  # a group not created yet (dry run)
        if "NoSuchEntity" in type(e).__name__ + repr(e):
            return {}
        raise


def _runpod(
    repo: Path, cfg: Settings, people: list[Person], everyone: set[str], dry_run: bool
) -> list[str]:
    root = repo / sealed.MEMBERS_DIR
    uses_runpod = (cfg.remote_cloud or "").startswith("runpod") or any(root.glob("*/runpod.age"))
    if not uses_runpod:
        return []
    lines = []
    for p in people:
        sealed_key = sealed.folder(repo, p.login) / "runpod.age"
        name = runpod_key_name(cfg, p.login)
        if p.writes and sealed_key.is_file() and p.stale:
            lines.append(f"{p.login}: keys changed: their RunPod key could be read with the old "
                         f"keys, so delete {name} in the RunPod console, create a new one, and "
                         f"run `fieldbook access runpod-key {p.login}`")  # fmt: skip
            if not dry_run:
                sealed_key.unlink()
                if not any(sealed_key.parent.glob("*.age")):  # nothing left on the old keys
                    sealed.record_recipients(sealed_key.parent, list(p.keys))
        elif p.writes and not sealed_key.is_file():
            lines.append(f"{p.login}: no RunPod key: create one named {name} in the RunPod "
                         f"console, then run `fieldbook access runpod-key {p.login}`")  # fmt: skip
        elif not p.writes and sealed_key.is_file():
            lines.append(f"{p.login}: is a viewer: delete RunPod key {name} in the console")
            if not dry_run:
                sealed_key.unlink()
    for member in sorted(root.glob("*/runpod.age")):
        login = member.parent.name
        if login not in everyone:
            lines.append(f"{login}: left: delete RunPod key {runpod_key_name(cfg, login)} "
                         "in the RunPod console")  # fmt: skip
    return lines


def _leavers(repo: Path, everyone: set[str], dry_run: bool) -> list[str]:
    root = repo / sealed.MEMBERS_DIR
    if not root.is_dir():
        return []
    lines = []
    for member in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in everyone):
        lines.append(f"{member.name}: credentials folder removed")
        if not dry_run:
            shutil.rmtree(member)
    return lines


def _folder_state(repo: Path) -> dict[str, str]:
    """The member folders' files and their contents, to tell whether a sync changed them."""
    root = repo / sealed.MEMBERS_DIR
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
