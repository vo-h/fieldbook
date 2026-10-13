# Testing

Last updated: 2026-10-09

How the tests are organized, kept apart from the real machine, and run. Each design document lists what its own tests cover.

## 1. Layout

The tests mirror the code. `tests/<package>/` holds the tests of `fieldbook/<package>/` (`data/`, `launch/`, `code/`, `images/`, `clouds/`, `tracking/`, `config/`, `members/`, `cli/`); `tests/test_<module>[_topic].py` those of a top-level module (`test_storage.py`, `test_storage_bucket.py`, `test_credentials.py`, `test_setup_project.py`). Helpers shared across folders are in `tests/helpers.py`, shared fixtures in `tests/conftest.py`.

`tests/test_layout.py` fails if a test folder has no package, a package has no tests, a top-level test file is not named after a module, or a `file.md §N` reference in the code or the documents names a section that does not exist.

## 2. Isolation

- Every test gets its own storage root, state folder, and home folder: no real `~/.aws`, `~/.runpod`, `~/.sky`, GitHub login, or database login is visible (the real uv cache is kept so builds stay offline).
- Cloud paths use fakes: moto for S3 and IAM, a fake `sky` executable, a fake registry, and a fake `gh`. Postgres tests start a throwaway server; where passwords matter it checks them, and its owner is not a superuser, as on managed Postgres.
- Servers bind port 0 where they can (moto), otherwise a port block per parallel worker (MLflow, Postgres); UI tests wait for the server to be gone instead of sleeping.
- Tests that send Ctrl-C start `fieldbook` with SIGINT enabled, as a terminal's foreground job has it: a test run started in the background of a non-interactive shell has SIGINT ignored, and every process it starts inherits that.
- `--strict-markers` makes a mistyped marker an error; `slow` marks tests that start real servers.

## 3. Running

```bash
uv run pytest                 # everything, in parallel
uv run pytest -m "not slow"   # without the tests that start real servers
uv run pytest tests/data      # one package's tests
```

Day to day, run the tests of the package being changed; the whole suite before a release.

## 4. CI

Python 3.11–3.13 on Linux (3.12 on macOS), with gitleaks, age, and Postgres installed; a job at the lowest allowed dependency versions (`--resolution lowest-direct`); a check that `constraints/<version>.txt` matches `uv.lock`. A release runs the same suite first.

## 5. Checks against real services

Done by hand, with the cheapest machines and a guard that terminates any `fieldbook-*` instance after 12 minutes: cloud runs on AWS and RunPod (cloud-runs.md §12), datasets on S3, concurrent saves included, and a `DataLoader` reading from a machine in the bucket's region (data.md §10, §13), and the GitHub collaborators query (sharing.md §8).
