# Versions and Releases

## Version numbers

fieldbook follows [semantic versioning](https://semver.org). Nobody chooses the number: it is derived from the git history by [Git Semantic Version](https://github.com/marketplace/actions/git-semantic-version) (`PaulHatch/semantic-version`), from the last `v*` tag and what the commit messages since then say.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org): `type(scope): summary`, the scope optional.

| A commit since the last release | Example | Next version from 1.0 on | While the version is 0.x |
|---|---|---|---|
| Breaks something users rely on: `!` after the type, or `BREAKING CHANGE:` in the message | `feat(cli)!: log replaces runs` | Major: 1.4.2 → 2.0.0 | Minor: 0.4.2 → 0.5.0 |
| Adds a feature: `feat` | `feat(data): diff for records` | Minor: 1.4.2 → 1.5.0 | Patch: 0.4.2 → 0.4.3 |
| Anything else: `fix`, `docs`, `test`, `ci`, `refactor`, `chore`, or no type at all | `fix(data): creating a dataset twice at once` | Patch: 1.4.2 → 1.4.3 | Patch: 0.4.2 → 0.4.3 |

- The largest change since the last tag decides, and the version moves by one step however many commits there are.
- A message that does not follow the convention is not an error: it counts as a patch. Only the two markers above matter, so use them with care.
- **What counts as breaking:** a command, option, or its output format (`--json`) removed or changed; a function or argument of the Python API removed or changed; a setting renamed or its meaning changed; stored data (datasets, run records) that an older or newer fieldbook can no longer read. Adding to any of these is a feature.
- The `version` in `pyproject.toml` is the last release's; the release workflow writes it. Every CI run says what the next release would be.

## Dependency versions

`pyproject.toml` declares compatible ranges (lower bounds), so fieldbook can be installed next to other packages. The exact set of versions each release was tested with is published as `constraints.txt` at that release's tag, as Apache Airflow does:

```bash
pip install "fieldbook[all]==0.1.0" -c https://raw.githubusercontent.com/vo-h/fieldbook/v0.1.0/constraints.txt
```

The constraints file is exported from `uv.lock`, which the test suite runs against, and covers every extra. CI fails if `constraints.txt` no longer matches `uv.lock`; re-export it after changing dependencies:

```bash
uv export --frozen --no-hashes --all-extras --no-dev --no-emit-project \
  --format requirements-txt -o constraints.txt
```

The lower bounds are tested too: CI's `lowest` job runs the suite with the oldest version each bound allows (`uv run --isolated --resolution lowest-direct --all-extras --group dev pytest -m "not slow"`), so a bound that is too low fails CI instead of a user's install. Raise a bound when a test needs a newer version, and note why next to it in `pyproject.toml`.

## Releasing

1. Add a section for the release to `CHANGELOG.md` and push it to `main`.
2. Check the package by hand when the release is a large one: `uv build`, install the wheel with the constraints file into a new environment, and run the user guide's commands in a new project.
3. Run the release: on GitHub, **Actions > Release > Run workflow** (or `gh workflow run release.yml`).

`.github/workflows/release.yml` then runs the whole CI suite, derives the version, writes it into `pyproject.toml` and `uv.lock`, builds, and publishes to PyPI. Only once PyPI has the release does it commit the version as `chore(release): <version>`, tag it `v<version>`, and make a GitHub release, so a failed upload leaves nothing to undo and the workflow can simply be run again. It refuses to run when there is nothing new since the last tag.

- **The first release** has no tag to start from: give the workflow its `version` input (`0.1.0`). The same input overrides the derived version at any time, for example to release 1.0.0.
- It is off until the repository variable `RELEASE_ENABLED` is `true`. Before the first release, create the `pypi` environment in the repository settings and, on pypi.org, add this repository and `release.yml` as a trusted publisher of the `fieldbook` project (environment `pypi`). No PyPI token is stored in GitHub.

Linux check without CI:

```bash
docker run --rm -v "$PWD":/src:ro python:3.12 bash -c '
  cp -r /src /work && cd /work && rm -rf .venv && pip install -q uv &&
  curl -sSL https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_arm64.tar.gz | tar xz -C /usr/local/bin gitleaks &&
  uv sync --locked -q && uv run pytest -q'
```

(Use `linux_x64` on Intel machines.)
