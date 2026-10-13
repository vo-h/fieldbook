# Working on fieldbook

Instructions for coding agents (and a quick orientation for people). fieldbook is a Python package and a command-line tool for ML experimentation: versioned datasets, recorded runs, experiment tracking, cloud GPUs. What it is and is not meant to do is in `design/README.md`.

## Commands

```bash
uv sync                                  # .venv with every extra and the development tools
uv run pytest tests/data                 # run the tests for the part you changed
uv run pytest                            # everything (about 3 minutes); not after every edit
uv run ruff check && uv run ruff format  # lint and format; lines up to 100 characters
uv run fieldbook --help                  # the CLI from this checkout
```

Tests need `git` and `gitleaks`; the shared-project tests also need `age`, and use a local Postgres when there is one.

## Where things are

| Path | What |
|---|---|
| `fieldbook/cli/` | The commands, one module per group: `run`, `data`, `log`, `config`, `access` |
| `fieldbook/data/` | Datasets: each is one Lance table (`dataset.py` files, `records.py` rows, `cleanup.py`) |
| `fieldbook/launch/` | `fieldbook run`: local and cloud runs, records, reruns, budgets |
| `fieldbook/clouds/`, `fieldbook/images/` | AWS and RunPod specifics; the environment image |
| `fieldbook/tracking/`, `fieldbook/history.py` | MLflow setup; the table of runs behind `fieldbook log` |
| `design/` | How fieldbook works and why. One document per topic; `design/README.md` is the index |
| `docs/guide.md` | The user guide |
| `tests/` | Mirrors the package; `design/testing.md` says how tests are isolated |

## Rules

- **Design documents are kept true.** When behaviour changes, update the document for that topic in the same change, and add a line to `CHANGELOG.md` for anything a user would notice.
- **Commit messages are Conventional Commits** (`fix(data): ...`, `feat(cli): ...`, `docs: ...`): the release version is derived from them (`docs/releasing.md`). Use `!` only for a real break of the CLI, the Python API, settings, or stored data.
- **Tests for what you change,** and run those tests, not the whole suite, while working.
- **No new server and no paid service.** Low overhead is the point. A new dependency should be open source, widely used, and an optional extra unless every user needs it.
- **Claim only what was run.** A cloud, storage provider, or framework is listed as supported once it has been exercised for real; measurements in the documents say how they were taken.
- **The top level of the CLI stays small:** new commands go under an existing group, and every `--help` description fits one line.
- **A file's bytes are never rewritten unless they changed** (`design/data.md` §6): edits to datasets append, mark deleted, or rewrite columns.

## Do not

- Commit credentials, or anything under `.fieldbook/` or `.env`.
- Launch cloud machines or change cloud accounts as part of a code change; cloud checks are run by a person (`design/testing.md` §5).
- Add attribution lines for AI tools to commits or files.
