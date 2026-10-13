# Versions and Releases

## Dependency versions

`pyproject.toml` declares compatible ranges (lower bounds), so fieldbook can be installed next to other packages. The exact set of versions each release was tested with is published as `constraints/<version>.txt`, as Apache Airflow does:

```bash
pip install "fieldbook[all]" -c constraints/0.1.0.txt
```

The constraints file is exported from `uv.lock`, which the test suite runs against, and covers every extra.

The lower bounds are tested too: CI's `lowest` job runs the suite with the oldest version each bound allows (`uv run --isolated --resolution lowest-direct --all-extras --group dev pytest -m "not slow"`), so a bound that is too low fails CI instead of a user's install. Raise a bound when a test needs a newer version, and note why next to it in `pyproject.toml`. CI also fails if `constraints/<version>.txt` no longer matches `uv.lock`.

## Releasing

1. Update `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
2. Refresh and test: `uv lock --upgrade` (optional), then `uv sync` and `uv run pytest`, on macOS and Linux (CI, or `docker run` as below).
3. Export the tested versions:
   ```bash
   uv export --frozen --no-hashes --all-extras --no-dev --no-emit-project \
     --format requirements-txt -o constraints/<version>.txt
   ```
4. Build and check the package: `uv build`, then install the wheel with the constraints file into a new environment and run the user guide's commands in a new project.
5. Commit, tag `v<version>`, and push the tag.

`.github/workflows/release.yml` builds and publishes to PyPI when a `v*` tag is pushed. It is off until the repository variable `RELEASE_ENABLED` is `true`. Before the first release:

- create the `pypi` environment in the repository settings;
- on pypi.org, add this repository and `release.yml` as a trusted publisher of the `fieldbook` project (environment `pypi`). No PyPI token is stored in GitHub.

Linux check without CI:

```bash
docker run --rm -v "$PWD":/src:ro python:3.12 bash -c '
  cp -r /src /work && cd /work && rm -rf .venv && pip install -q uv &&
  curl -sSL https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_arm64.tar.gz | tar xz -C /usr/local/bin gitleaks &&
  uv sync --locked -q && uv run pytest -q'
```

(Use `linux_x64` on Intel machines.)
