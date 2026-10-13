# Projects, settings, and credentials

Status: Built · Last updated: 2026-10-09

Everything in fieldbook belongs to a **project**: one code repository with a name. Its settings, data, runs, models, cloud resources, and members are the project's. This document covers what a project is on one machine: its settings, its local folder, its credentials, and the commands that set it up. Sharing a project between people is in [sharing.md](sharing.md). Code: `fieldbook/config/`, `fieldbook/credentials.py`, `fieldbook/setup_project.py`.

## 1. A project

- **A repository with a name** (`project = "vision"`). Datasets, the MLflow experiment, the folder in the storage root, and cloud resources are named after it. One person may work on several projects; a personal project is one nobody else joins.
- **Settings travel with the code.** They are committed in the project file, so a clone is configured, changes are reviewed in pull requests, and every run's code snapshot records them.
- **Commands that use data or runs need a project** (`fieldbook data`, `run`, `log`, `build`, `access`, and `fb.dataset()` / `fb.init()`): a project file at or above the working directory. Outside one they stop with "run `fieldbook init` in the project's folder". `fieldbook config`, `fieldbook init`, and `--help` work anywhere. Cloud machines and tests have no project file and set the settings as `FIELDBOOK_*` variables instead (`FIELDBOOK_PROJECT` counts as a project).

## 2. Settings

One `Settings` class (pydantic-settings, `fieldbook/config/model.py`) lists every setting with its type, default, and description; the guide's settings table is generated from it. Settings are grouped for reading (`GROUPS`): **Storage** (`storage_root`, `storage_endpoint`, `storage_region`, `state_dir`); **Cloud runs** (`image_registry`, `image_base`, `image_apt`, `build_cloud`, `remote_cloud`, `launch_timeout`, `budget_run_usd`, `budget_monthly_usd`); **Project** (`project`, `author`, `keep_outputs`, `output_max_file_mb`, `output_max_total_mb`); **Experiments (MLflow)** (`experiment`, `tracking_uri`, `mlflow_port`, `autolog`). The keys stay flat in the files and in `FIELDBOOK_<NAME>` variables. Two settings are read from the project file only, `budget_run_usd` and `budget_monthly_usd` (`PROJECT_ONLY`): in the local file or a variable they are an error, since a budget that one machine could raise limits nobody (cloud-runs.md §11).

Every setting is a project setting; there is no per-user settings file and no profiles. Sources are layered as in uv, ruff, and git, with settings and credentials kept apart as in the AWS CLI; the first source that sets a value wins:

| Source | Holds | Example |
|---|---|---|
| Command-line options | One-off overrides (they set the variables below) | `fieldbook run --experiment ablation …` |
| Environment variables | Any setting as `FIELDBOOK_<NAME>` | `FIELDBOOK_PROJECT=test fieldbook run ...` |
| Local file, not committed: `.fieldbook/config.toml` in the project | This machine's changes to the project's settings, e.g. a storage root on a big local disk; the person's `author` | `storage_root = "~/big-disk/vision"` |
| Project file, committed: `fieldbook.toml`, or `[tool.fieldbook]` in `pyproject.toml`, nearest upward from the working directory | Every setting the project's members share: its name, storage root, run database address (no user name or password), image registry, clouds | `project = "vision"` |
| Defaults | | Local data and the run database in the project's `.fieldbook/`; author from git's `user.name` |

```toml
# pyproject.toml of github.com/acme/vision (committed)
[tool.fieldbook]
project = "vision"                                # names everything for this project
storage_root = "s3://acme-data"                   # the team's bucket; this project's folder is vision/
tracking_uri = "postgresql://db.acme.com/vision"  # this project's database; the login is in .env
image_registry = "1234.dkr.ecr.us-west-2.amazonaws.com/vision"
remote_cloud = "aws/us-west-2"
image_apt = ["ffmpeg"]
```

- Files are TOML. Unknown keys are errors, so typos are caught. Relative paths in a file are relative to that file.
- **`author`** is the person's name on runs and dataset versions. `fieldbook init` sets it once to their GitHub login (from their member folder, else the GitHub CLI) in the local file, so no command asks GitHub; without it, git's `user.name`.
- `fieldbook run` passes the resolved settings to the script as variables (runs.md §7).

## 3. The project's local folder

Each project keeps what it stores on a machine next to its code, as `.venv` sits next to `pyproject.toml`:

```
~/code/vision/
  pyproject.toml          # [tool.fieldbook]: the shared settings (committed)
  .env                    # credentials (never committed)
  .fieldbook-members/     # members' encrypted credentials (committed; sharing.md §2)
  .fieldbook/             # this machine only; in .gitignore, never in code snapshots
    config.toml           # this machine's changes to the settings (optional)
    data/                 # local data (when the project's storage root is not set)
    mlflow/mlflow.db      # local MLflow runs (when the project has no shared database)
    bucket-regions.json, environments/, members.json   # caches
```

- What belongs to a project is clear, and removing a project from a machine is deleting its folder. There is no folder in the home directory.
- **Git worktrees** of one repository share the main working tree's `.fieldbook/` (found with `git rev-parse --git-common-dir`), so every checkout sees the same runs and data.
- A folder with a project file below another one is a project of its own, with its own `.fieldbook/`.
- `state_dir` moves the databases and caches elsewhere.

## 4. Credentials

**Credentials are not settings.** They come from each provider's usual sources (e.g. AWS's, storage.md §3) and from the project's `.env` file (`KEY=value` lines; read with python-dotenv), as with Docker Compose and most frameworks; there is no keychain and no secrets command. The database login is `PGUSER` and `PGPASSWORD`, read by the Postgres driver itself.

- **Loaded for `fieldbook` and for runs:** `fieldbook` loads the nearest `.env` (from the working directory up to the repository root) into its own environment at start-up, so `fieldbook data` commands and runs see the same credentials, and `fieldbook run` and `fb.init()` pass its variables to the script; `fieldbook run` lists their names (never values). Variables already set in the environment win, so secret managers that wrap commands work unchanged: `op run -- fieldbook run ...` (1Password), `infisical run -- fieldbook run ...`, `doppler run -- fieldbook run ...`.
- **Never recorded:** `.env` and `.env.*` files are left out of code snapshots, except templates (`.env.example`, `.sample`, `.template`). Values of the run's secrets (4 characters or longer) are masked in recorded commands, next to secret-like flags. `fieldbook init` adds `.env` to `.gitignore`, and `fieldbook run` warns when git tracks `.env`.
- **Cloud runs:** the database login goes along automatically; `--secret NAME` selects which other secrets are sent (cloud-runs.md §7).
- **On a shared project** a member's database login and RunPod key are delivered by the owner and written to `.env` by `fieldbook init` (sharing.md §2). A person's own `.env` is theirs to protect.

## 5. `fieldbook init`

Sets up the current repository as a project and only creates what is missing, so it is safe to run again:

- the project settings (added to `pyproject.toml` if it exists, else `fieldbook.toml`; the project is named after the folder);
- a `.fieldbookignore` with common output folders, and `.env` and `.fieldbook/` in `.gitignore`;
- on a shared project, the person's credentials in `.env` (sharing.md §2), and their GitHub login as `author`;
- with `--agents`, a section for coding agents in the repository's `CLAUDE.md` (Claude Code) and `AGENTS.md` (Codex, Cursor, and others), the same text in both, between `<!-- fieldbook:start -->` and `<!-- fieldbook:end -->`. It tells an agent what a person would otherwise have to remember to tell it: keep `import fieldbook` at the top of scripts, read `fieldbook log --json` and its notes before running anything, note what each run showed (`fieldbook log note`), use an earlier result when a repeat is reported, check a result with `fieldbook run --rerun` before reporting it, compare with `fieldbook log diff`, read data through `fb.dataset`, smoke-test before long or cloud runs, ask before a cloud run, never change the budgets. The rest of each file is the user's; once the section exists in a file, every `fieldbook init` replaces it with the installed version's text. They are committed, so a clone has them.

It reports whether gitleaks is installed, checks a bucket storage root (storage.md §6) and a shared tracking database (reachable, schema up to date, or read-only for a viewer), and, when cloud runs are set up (an S3 bucket and `image_registry`), shows the machine roles' policies and creates them after asking (cloud-runs.md §7).

## 6. `fieldbook config`

`fieldbook config` shows every setting by group, its value, and its source (environment, local file, project file, default), the store runs log to, the credential names in `.env` (never values), and the file paths, as `aws configure list` and `git config --show-origin` do.

`fieldbook config set KEY VALUE`, `unset KEY`, and `get KEY` change and read one setting, as `git config` does (`fieldbook/config/edit.py`): a setting goes to the project file (created at the repository root if missing), or with `--local` to the project's `.fieldbook/config.toml`. Values are checked with the settings model before writing (types, storage-root schemes, project names; lists as `a,b,c`; relative paths made absolute), a `tracking_uri` with a password is refused, and the whole configuration is read back after writing, restoring the file if it no longer loads. Files are edited with tomlkit, which keeps comments and layout. `fieldbook config` commands skip the start-up settings check, so a broken file can be fixed with them.

## 7. Testing

Sources and precedence, the local file, worktrees sharing a folder, the author from git, commands refusing to run outside a project, `fieldbook config set`/`unset`/`get` (including a broken file being fixed), `fieldbook init` being idempotent, the agents' section added to both files only when asked for and replaced when out of date, and `.env` loading and masking.
