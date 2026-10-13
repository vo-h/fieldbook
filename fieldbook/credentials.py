"""Credentials: where they come from, which values are secret, and keeping them out of
records and messages (projects.md §4).

- **`.env`:** `fieldbook run`, `fb.init()`, and `fieldbook`'s own commands read the nearest `.env`
  (from the working directory up to the repository root). Variables already set in the
  environment win, so secret managers that wrap commands (`op run --`, `infisical run --`,
  `doppler run --`) work unchanged. `.env` files are left out of code snapshots.
- **A run's secrets:** the names of its `.env` variables, plus those its launcher passed
  in `FIELDBOOK_SECRETS` (on a cloud machine: `--secret` values, the database password, and so
  on). Their values are masked in recorded commands, as are flags that look like secrets.
- **Addresses:** user names, passwords, and secret query parameters in URIs are hidden in
  messages (`display_uri`, `redact`).
"""

import os
import re
import shlex
import subprocess
from pathlib import Path

from dotenv import dotenv_values

ENV_FILE = ".env"
TEMPLATE_SUFFIXES = (".example", ".sample", ".template")  # committed templates, no secrets
SECRETS_VAR = "FIELDBOOK_SECRETS"  # names (never values) of secrets a launcher gave a run
SECRET_FLAG_RE = re.compile(
    r"^--?[\w-]*(token|secret|password|passwd|api[-_]?key|access[-_]?key|credential)[\w-]*",
    re.IGNORECASE,
)
SECRET_PARAMETERS = {"password", "pwd", "pass", "passwd", "secret", "token", "access_token",
                     "sslpassword", "api_key", "apikey"}  # fmt: skip

# --- .env -----------------------------------------------------------------------------


def find(workdir: Path, repo: Path) -> Path | None:
    directory = workdir.resolve()
    while True:
        candidate = directory / ENV_FILE
        if candidate.is_file():
            return candidate
        if directory == repo.resolve() or directory.parent == directory:
            return None
        directory = directory.parent


def find_from(workdir: Path) -> Path | None:
    """The nearest `.env` from `workdir` up to its repository root (or only `workdir`
    outside a repository)."""
    from fieldbook.code import snapshot

    try:
        repo = snapshot.repo_root(workdir)
    except snapshot.SnapshotError:
        repo = workdir
    return find(workdir, repo)


def load(path: Path) -> dict[str, str]:
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def project_values(workdir: Path, repo: Path | None = None) -> tuple[Path | None, dict[str, str]]:
    """The project's `.env` and its variables (empty if there is none)."""
    path = find(workdir, repo) if repo is not None else find_from(workdir)
    return path, (load(path) if path else {})


def load_into_environ(workdir: Path) -> Path | None:
    """Load the project's `.env` into this process's environment, so `fieldbook`'s own commands
    use the same credentials as runs (e.g. bucket keys for `fieldbook data`). Variables already
    set win. Returns the file, or None."""
    path, values = project_values(workdir)
    for name, value in values.items():
        if name not in os.environ:
            os.environ[name] = value
            loaded.add(name)
    return path


loaded: set[str] = set()  # variables this process took from .env, for `fieldbook config`


def is_secret_file(name: str) -> bool:
    """`.env` and `.env.*` files hold secrets, except committed templates (`.env.example`)."""
    base = name.rsplit("/", 1)[-1]
    if base == ENV_FILE:
        return True
    return base.startswith(ENV_FILE + ".") and not base.endswith(TEMPLATE_SUFFIXES)


def is_tracked(path: Path, repo: Path) -> bool:
    """Whether git tracks the file, i.e. its secrets may be in the repository's history."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(repo))],
        cwd=repo, capture_output=True,
    )  # fmt: skip
    return result.returncode == 0


# --- a run's secrets ------------------------------------------------------------------


def run_secret_names(file_values: dict[str, str]) -> tuple[str, ...]:
    """Names of the secrets a run is given: its `.env` variables and `FIELDBOOK_SECRETS`."""
    passed = filter(None, os.environ.get(SECRETS_VAR, "").split(","))
    return tuple(sorted(set(file_values) | set(passed)))


def mask_command(argv: list[str], secret_values: list[str]) -> str:
    """`argv` as a shell command, with `secret_values` and the values of flags that look
    like secrets (`--token`, `--api-key`, ...) replaced by `***`."""
    for value in secret_values:
        if len(value) >= 4:  # shorter values would mask unrelated text
            argv = [arg.replace(value, "***") for arg in argv]
    masked, mask_next = [], False
    for arg in argv:
        if mask_next:
            masked.append("***")
            mask_next = False
        elif SECRET_FLAG_RE.match(arg):
            flag, sep, _ = arg.partition("=")
            masked.append(f"{flag}=***" if sep else arg)
            mask_next = not sep
        else:
            masked.append(arg)
    return shlex.join(masked)


# --- addresses ------------------------------------------------------------------------


def display_uri(uri: str) -> str:
    """The URI without its user name, password, or secret query parameters
    (`?password=...`), for messages."""
    from urllib.parse import parse_qsl, urlencode

    scheme, sep, rest = uri.partition("://")
    if not sep:
        return uri
    netloc, slash, path = rest.partition("/")
    path, question, query = path.partition("?")
    if question:
        pairs = [(k, "***" if k.lower() in SECRET_PARAMETERS else v) for k, v in parse_qsl(query)]
        query = urlencode(pairs, safe="*")
    return f"{scheme}://{netloc.rpartition('@')[2]}{slash}{path}{question}{query}"


def has_secret(uri: str) -> bool:
    return display_uri(uri) != uri


def redact(text: str, uri: str) -> str:
    """`text` (e.g. an error message) with `uri` shown without its secrets."""
    return text.replace(uri, display_uri(uri))
