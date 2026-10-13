# Contributing

Thank you for looking. fieldbook is in alpha and maintained by one person, so the most useful contributions right now are bug reports with a way to reproduce them, and small fixes.

## Before you write code

- **A bug:** open an issue with the command, what you expected, and what happened (`FIELDBOOK_DEBUG=1` shows the traceback).
- **A feature or a change of behaviour:** open an issue first. The design documents in `design/` say what fieldbook is and is not meant to do (`design/README.md` §5 lists what is out of scope); a change that goes against them needs agreeing before it is built.
- **A security problem:** see [SECURITY.md](SECURITY.md); do not open a public issue.

## Setting up

Requires [uv](https://docs.astral.sh/uv/), git, and [gitleaks](https://github.com/gitleaks/gitleaks). The tests for shared projects also need [age](https://github.com/FiloSottile/age), and those for a shared tracking database use a local Postgres and are skipped without one (`brew install age postgresql`, or `apt-get install age postgresql`).

```bash
uv sync                       # .venv with every extra and the development tools
uv run pytest                 # all tests, one worker per core
uv run pytest tests/data      # or only the part you changed
uv run ruff check && uv run ruff format --check
```

## What a change needs

- **Tests** for what it changes. How the tests are laid out is in `design/testing.md`.
- **The design documents updated** when behaviour changes: they are the description of how fieldbook works, and they are kept true. A line in `CHANGELOG.md` for anything a user would notice.
- **Lint and format clean** (`ruff`), lines up to 100 characters.
- **No new server to run and no paid service required.** fieldbook's point is low overhead; a dependency should be open source, widely used, and optional (an extra) unless every user needs it.
- **Nothing claimed that was not run.** A cloud, a storage provider, or a framework is listed as supported once it has been exercised for real, and the documents say how.

## Commit messages

Version numbers are derived from commit messages (`docs/releasing.md`), so write them as [Conventional Commits](https://www.conventionalcommits.org): `fix(data): ...`, `feat(cli): ...`, `docs: ...`. Mark a change that breaks something users rely on with `!` (`feat(cli)!: ...`); that is what moves the version past a patch, so do not use it lightly.

## Pull requests

Keep them small and about one thing. Say what the change does and how you checked it. By contributing you agree that your contribution is licensed under the project's licence (Apache-2.0).
