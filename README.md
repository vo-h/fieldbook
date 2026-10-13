# fieldbook

fieldbook is a lightweight toolkit for machine-learning experimentation. It versions datasets, records how every result was produced (code, environment, data, seed), tracks experiments, and runs the same script on your machine or on a rented cloud GPU.

- **Low overhead.** There are no servers to deploy or operate. fieldbook is a Python package and a command-line tool; its state lives in your repository, a storage folder or bucket, and a database file. A shared project adds only a bucket and a managed database.
- **Cloud-agnostic by design.** Storage is addressed by URI and compute is launched through [SkyPilot](https://github.com/skypilot-org/skypilot), so no provider's services are built in, and research code does not change between a laptop and the cloud. Verified so far: AWS S3 for storage, AWS and RunPod for compute.
- **Agent-friendly.** Coding agents can use it as well as people can: every command has machine-readable output, what an agent runs is attributed to it, results carry notes for the next session, and a project can hold instructions that tell an agent how to work in it.

It is built from open-source components (Lance, MLflow, SkyPilot, DuckDB) and suits one person on one machine as well as a project shared between several. Status: alpha. See the [design](design/README.md).

## Quick start

Requires Python 3.11+, git, and [gitleaks](https://github.com/gitleaks/gitleaks) (`brew install gitleaks`).

```bash
pip install "fieldbook[all]"
cd my-project && fieldbook init

fieldbook data add pets ./images --set label=unknown    # a versioned dataset
fieldbook run train.py                                  # code, environment, and data recorded
fieldbook log                                           # what has been run
fieldbook log ui                                        # MLflow UI until Ctrl-C
```

Read the [user guide](docs/guide.md) next.

## Documentation

User documentation (`docs/`):

- [docs/guide.md](docs/guide.md): user guide
- [docs/releasing.md](docs/releasing.md): versions and releases
- [CHANGELOG.md](CHANGELOG.md)

Design (`design/`):

- [design/README.md](design/README.md): what fieldbook is, requirements, how the parts fit, and an index of the documents below
- [design/projects.md](design/projects.md): projects, settings, credentials
- [design/storage.md](design/storage.md): the storage root; local disk and S3
- [design/data.md](design/data.md): versioned datasets
- [design/runs.md](design/runs.md): `fieldbook run`, code snapshots, seeds, smoke runs
- [design/tracking.md](design/tracking.md): MLflow tracking, the model registry, costs
- [design/cloud-runs.md](design/cloud-runs.md): runs on cloud GPUs
- [design/sharing.md](design/sharing.md): projects shared between people
- [design/robotics.md](design/robotics.md): robot learning with LeRobot (designed, not built)
- [design/testing.md](design/testing.md): how the tests are organized

## Development

Requires [uv](https://docs.astral.sh/uv/) and gitleaks.

```bash
uv sync            # create .venv and install dependencies
uv run fieldbook --help  # run the CLI
uv run pytest      # run tests, one worker per core
uv run pytest -m "not slow"  # skip the MLflow server tests
uv run ruff check  # lint
uv run ruff format # format
```

CI (`.github/workflows/ci.yml`) runs lint and tests on Linux (Python 3.11–3.13, with Postgres) and macOS. It is off until the repository variable `CI_ENABLED` is `true`, and can be started by hand.

## License

Apache-2.0
