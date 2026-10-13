"""`fieldbook init`: set up the current repository as a project. Idempotent: it only creates what
is missing, so it is safe to run again (projects.md §5)."""

import re
import shutil
from collections.abc import Callable
from pathlib import Path

from fieldbook import config
from fieldbook.code import snapshot

FIELDBOOKIGNORE = """\
# Files left out of code snapshots (gitignore syntax; runs.md §4).
# Outputs written inside the repository would otherwise be archived again by every run.
mlruns/
wandb/
lightning_logs/
outputs/
checkpoints/
# data/
"""

AGENT_FILES = ("CLAUDE.md", "AGENTS.md")  # Claude Code; Codex, Cursor, and others
AGENT_START, AGENT_END = "<!-- fieldbook:start -->", "<!-- fieldbook:end -->"
# What a coding agent working in the project needs to know (projects.md §5). Short: the file
# is read into every session.
AGENT_INSTRUCTIONS = f"""\
{AGENT_START}
## Runs and data (fieldbook)

This project records its runs and datasets with the `fieldbook` command. `fieldbook
<command> --help` has the details; the commands below take `--json`.

- **Scripts:** keep `import fieldbook as fb` at the top of every script. `python script.py`
  is then recorded (code, git state, packages, seed, datasets read and saved); nothing else
  is needed. Training with Lightning, XGBoost, LightGBM, or Keras logs its parameters and
  metrics to MLflow automatically: do not add logging code for those.
- **Before running anything, read what was run:** `fieldbook log --json` (newest first:
  command, git, parameters, metrics, cost, datasets, and the notes left on each run). A run
  noted as `bad` was a dead end: do not repeat it.
- **After a run, note what it showed:** `fieldbook log note RUN "what was learned"
  --verdict good|bad|inconclusive`. Write the conclusion, not the numbers (those are
  recorded). The next session starts from these notes.
- **Choosing and comparing runs:** `fieldbook log query "select ..." --format json` runs
  SQL over the tables `runs`, `metrics`, `params`, `data`, and `notes` (`metrics['acc']`,
  `params['lr']`); in Python, `fb.runs(query=...)`. For a write-up, use `--format markdown`
  and keep the `run_id` column, so every number can be traced to its run.
- **A repeat is reported:** when a run has the same code, environment, command, and data as
  an earlier one, `fieldbook` says so and shows that run's result. Use the earlier result
  instead of running again; a cloud run is not launched unless `--allow-repeat` is given.
- **Check a result before reporting it:** `fieldbook run --rerun RUN` runs it again from its
  record (same code, seed, and data) and says whether the metrics came out the same.
  `fieldbook run --rerun RUN --lr 0.01` changes one thing.
- **A run that failed:** `fieldbook log why RUN --json` gives the cause and the lines that
  show it; `fieldbook log output RUN --tail 100` the output; `fieldbook log files RUN` the
  files it wrote. Start scripts with `fieldbook run`, not plain `python`, so that these are
  kept.
- **A cloud run takes hours:** after launching, `fieldbook log wait --timeout 9` returns
  when it ends (exit code 0: finished, 1: failed, 124: still going, so call it again).
- **Files outside the repository** (`/tmp`, the home folder) are lost with a cloud machine:
  write them under `os.environ["PLATFORM_OUTPUT_DIR"]`.
- **Why two results differ:** `fieldbook log diff RUN_A RUN_B --json` (command, code,
  packages, data, parameters, metrics; only what differs). Run IDs can be shortened to a
  unique prefix.
- **Data:** read and save through `fb.dataset("name")`, never by path into the storage root.
  `fieldbook data ls` lists datasets and their kind: files (`files()`, `read_many(paths)`,
  `fieldbook data checkout NAME DIR` for real files) or records, the rows of tables
  (`to_table()`, `fieldbook data query -d NAME -q "select ..."`). `fieldbook data ls NAME
  --lineage` shows what made a version. Version numbers can skip.
- **Try before spending:** `fieldbook run --smoke script.py` runs on a sample and saves
  nothing. Do it before a long run and before any cloud run.
- **Cloud runs cost money:** `fieldbook run --on remote ...` starts a paid machine. First
  run it with `--dry-run --json` and show the user the machine, the price, and the most it
  can cost; launch only once they agree, and give it `--max-hours`. `fieldbook log costs
  --total` shows what has been spent. Never change `budget_run_usd` or `budget_monthly_usd`:
  those are the user's.
- **Say who you are:** runs, dataset versions, and notes you make are marked as an agent's.
  Claude Code, Codex, Cursor, and Gemini CLI are recognised by themselves; any other agent
  sets `FIELDBOOK_AGENT=<its name>` in the shell before using `fieldbook` or running
  scripts.
- **Never delete on your own:** `fieldbook log rm` and `fieldbook data rm` cannot be
  undone. Run them only when the user asks, and show `--dry-run` first.
- **Leave alone:** `.fieldbook/` (local data and run database), `.env` (secrets),
  `.fieldbook-members/` (members' encrypted credentials).
{AGENT_END}
"""


def init(cwd: Path | None = None, *, agent: bool = False) -> list[str]:
    """Create what is missing and return a line per check. `agent`: also write the
    instructions for coding agents (they are kept current once they exist)."""
    cwd = cwd or Path.cwd()
    lines = []
    try:
        repo = snapshot.repo_root(cwd)
    except snapshot.SnapshotError:
        lines.append("not in a git repository: project files skipped (run `git init` first)")
        repo = None
    if repo is not None:
        lines.append(_project_file(repo))
        lines.append(_create(repo / snapshot.IGNORE_FILE, FIELDBOOKIGNORE))
        lines.append(_gitignore(repo / ".gitignore", ".env", "local secrets"))
        lines.append(
            _gitignore(repo / ".gitignore", f"{config.LOCAL_DIR}/", "local data and state")
        )
        if line := _member_credentials(repo):
            lines.append(line)
        if line := _author(repo):
            lines.append(line)
        for name in AGENT_FILES:
            if line := _agent_instructions(repo / name, create=agent):
                lines.append(line)

    lines.append(_storage())
    lines.append(_tracking(cwd))

    if shutil.which("gitleaks"):
        lines.append("gitleaks: installed")
    else:
        lines.append(f"gitleaks: missing; {snapshot.GITLEAKS_HINT}")
    return lines


def project_name(repo: Path) -> str:
    """The repository folder name, made into a valid project name."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", repo.name).strip("-").lstrip("._-")
    return name or config.DEFAULT_PROJECT


def _project_file(repo: Path) -> str:
    found, values = config.find_project_file(repo)
    if found is not None and found.parent == repo:
        return f"project settings: {found} (project {values.get('project', '?')!r})"
    name = project_name(repo)
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text()
        separator = "" if text.endswith("\n") or not text else "\n"
        pyproject.write_text(f'{text}{separator}\n[tool.fieldbook]\nproject = "{name}"\n')
        return f"added [tool.fieldbook] to {pyproject} (project {name!r})"
    target = repo / config.PROJECT_FILE
    target.write_text(f'# fieldbook project settings (committed)\nproject = "{name}"\n')
    return f"created {target} (project {name!r})"


def _member_credentials(repo: Path) -> str | None:
    """This person's project credentials, if the owner gave them access (sharing.md
    §5). Loaded into this process too, so the checks below use them."""
    import os

    from fieldbook.members import MembersError
    from fieldbook.members.sync import receive

    try:
        line, received = receive(repo, config.project(repo).local)
    except MembersError as e:
        return f"project credentials: {e}"
    os.environ.update(received)
    return line


def _author(repo: Path) -> str | None:
    """Your name on runs and dataset versions: your GitHub login (sharing.md §4), the
    name the project's members know you by. Kept in this machine's settings, so finding it
    (from your member folder, else the GitHub CLI) happens once, not on every command."""
    from fieldbook.config import edit
    from fieldbook.members import MembersError, github, sealed

    found = config.project(repo)
    if found.file is None or any("author" in layer.values for layer in config.layers(repo)[1:]):
        return None  # not a project yet, or the author is set already
    own = sealed.find_own(repo)
    try:
        login = own[0].name if own else github.current_login(repo)
    except MembersError:
        return None  # no GitHub CLI or not logged in: git's user.name stays the default
    if not login:
        return None
    edit.set_value("author", login, local=True, start=repo)
    return f"author: {login} (your GitHub login; in {config.LOCAL_DIR}/{config.LOCAL_FILE})"


def _storage() -> str:
    from fieldbook import storage

    try:
        return storage.check()
    except storage.StorageError as e:
        return f"{e}\n  see design/storage.md §2 and §3 for settings and credentials"


# Run by the project's interpreter: connect to the tracking database (MLflow creates its
# tables in an empty one) and fail on a schema older than the installed MLflow.
_CHECK_TRACKING = """
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
from mlflow.tracking import MlflowClient
MlflowClient().search_experiments(max_results=1)
"""


def _tracking(cwd: Path) -> str:
    """Check a shared tracking database (tracking.md §1)."""
    import os
    import subprocess

    from fieldbook import credentials
    from fieldbook.code import environments
    from fieldbook.tracking import store

    uri = store.configured_uri(cwd)
    if uri is None:
        return f"tracking: this machine's database ({store.mlflow_db(config.settings())})"
    shown = credentials.display_uri(uri)
    if not store.is_database(uri):
        return f"tracking: MLflow server {shown}"
    python = str(environments.find(cwd).python)
    try:
        result = subprocess.run(
            [python, "-c", _CHECK_TRACKING], capture_output=True, text=True, timeout=120,
            env={**os.environ, "MLFLOW_TRACKING_URI": uri},
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return f"tracking: {shown}: no answer within 120 s"
    if result.returncode == 0:
        if store.read_only(uri):
            return f"tracking: shared database {shown} (reachable; read-only: {store.VIEWER})"
        return f"tracking: shared database {shown} (reachable, schema up to date)"
    error = (result.stderr.strip().splitlines()[-1:] or ["unknown error"])[0]
    if "out-of-date database schema" in result.stderr:
        error = "schema older than the installed MLflow; run: mlflow db upgrade <database address>"
    elif "psycopg" in error:
        error = "Postgres needs pip install 'psycopg[binary]' in the project"
    error += store.login_hint(result.stderr)
    return f"tracking: {shown}: {error}"


def _create(path: Path, content: str) -> str:
    if path.exists():
        return f"{path.name}: exists"
    path.write_text(content)
    return f"created {path}"


def _agent_instructions(path: Path, *, create: bool) -> str | None:
    """fieldbook's section of the agent's instruction file: added when asked for, replaced
    when out of date; the rest of the file is the user's."""
    text = path.read_text() if path.exists() else ""
    start, end = text.find(AGENT_START), text.find(AGENT_END)
    if start < 0 or end < start:
        if not create:
            return None
        separator = "" if not text else "\n" if text.endswith("\n") else "\n\n"
        path.write_text(f"{text}{separator}{AGENT_INSTRUCTIONS}")
        return f"added fieldbook's instructions for coding agents to {path}"
    section = AGENT_INSTRUCTIONS.rstrip("\n")
    if text[start : end + len(AGENT_END)] == section:
        return f"{path.name}: has fieldbook's instructions"
    path.write_text(text[:start] + section + text[end + len(AGENT_END) :])
    return f"updated fieldbook's instructions in {path}"


def _gitignore(gitignore: Path, entry: str, what: str) -> str:
    """Keep `entry` out of git."""
    lines = gitignore.read_text().splitlines() if gitignore.exists() else []
    bare = entry.rstrip("/")
    same = {entry, f"/{entry}", bare, f"/{bare}", *([".env*"] if entry == ".env" else [])}
    if any(line.strip() in same for line in lines):
        return f".gitignore: ignores {entry}"
    text = gitignore.read_text() if gitignore.exists() else ""
    separator = "" if text.endswith("\n") or not text else "\n"
    gitignore.write_text(f"{text}{separator}{entry}\n")
    return f"added {entry} to {gitignore} ({what})"


class SetupError(Exception):
    pass


def iam_roles(*, ask: Callable[[str], bool], say: Callable[[str], None]) -> None:
    """Create or update the IAM roles of cloud machines on AWS, after `ask` agrees. Only
    for cloud runs: data in S3 and an image registry set."""
    from fieldbook.clouds import aws

    cfg = config.settings()
    if not aws.roles_apply(cfg) or not cfg.image_registry:
        return
    try:
        pending = [
            (role, state) for role in aws.roles(cfg) if (state := aws.role_state(role)) != "ok"
        ]
    except Exception as e:  # no credentials, or IAM unreachable: report, go on
        say(f"IAM roles: could not check ({type(e).__name__}: {e})")
        return
    if not pending:
        say("IAM roles: up to date")
        return
    if any(state == "unknown" for _, state in pending):
        say("IAM roles: these credentials may not read IAM; an administrator creates the roles")
        return
    for role, state in pending:
        say(f"IAM role {role.name} ({role.purpose}): {state}; its policy:\n{role.document}")
    if not ask("Create or update these IAM roles?"):
        say("IAM roles: not changed; cloud runs need them (run fieldbook init again to create)")
        return
    try:
        for role, _ in pending:
            aws.ensure_role(role, cfg)
            say(f"IAM role {role.name}: ready")
    except Exception as e:
        raise SetupError(f"could not create IAM roles: {type(e).__name__}: {e}") from None
