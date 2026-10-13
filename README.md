# fieldbook

[![PyPI](https://img.shields.io/pypi/v/fieldbook)](https://pypi.org/project/fieldbook/)
[![Python](https://img.shields.io/pypi/pyversions/fieldbook)](https://pypi.org/project/fieldbook/)
[![CI](https://github.com/vo-h/fieldbook/actions/workflows/ci.yml/badge.svg)](https://github.com/vo-h/fieldbook/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/fieldbook)](LICENSE)

fieldbook is a lightweight toolkit for machine-learning experimentation. It versions datasets, records how every result was produced (code, environment, data, seed), tracks experiments, and runs the same script on your machine or on a rented cloud GPU.

- **Low overhead.** There are no servers to deploy or operate. fieldbook is a Python package and a command-line tool; its state lives in your repository, a storage folder or bucket, and a database file. A shared project adds only a bucket and a managed database.
- **Cloud-agnostic by design.** Storage is addressed by URI and compute is launched through [SkyPilot](https://github.com/skypilot-org/skypilot), so no provider's services are built in, and research code does not change between a laptop and the cloud. Verified so far: AWS S3 for storage, AWS and RunPod for compute.
- **Agent-friendly.** Coding agents can use it as well as people can: every command has machine-readable output, what an agent runs is attributed to it, results carry notes for the next session, and a project can hold instructions that tell an agent how to work in it.

It is built from open-source components (Lance, MLflow, SkyPilot, DuckDB) and suits one person on one machine as well as a project shared between several. Status: alpha. See the [design](design/README.md).

## What it looks like

```console
$ fieldbook data add pets ./images
pets: added 6 items, v1

$ fieldbook run train.py
trained on 6 files: acc=0.92

$ fieldbook log
run                     started           author  command   git                   status    metrics
20261013-204949-7d0bc4  2026-10-13 13:50  ada     train.py  839c08c main+changes  finished  acc=0.92

$ fieldbook run --rerun 20261013-204949
trained on 6 files: acc=0.92
rerun of 20261013-204949-7d0bc4 as 20261013-205018-6d9911: reproduced: 1 metrics (acc)
```

`train.py` is an ordinary script that reads `fb.dataset("pets")` and logs to MLflow. Nothing in it says how to record the run: fieldbook kept the code, the installed packages, the seed, and the dataset version it read, which is what lets `--rerun` run it again and say whether the result holds. Add `--on remote --gpus A10G:1` and the same script runs on a rented GPU.

## How it compares

fieldbook is not a replacement for the tools below so much as the glue most projects end up writing around them. It uses MLflow for tracking and SkyPilot for cloud machines.

| | fieldbook | DVC | MLflow | Weights & Biases |
|---|---|---|---|---|
| Versioned datasets | Yes: each dataset is one [Lance](https://github.com/lancedb/lance) table, so a million small files are a handful of objects in a bucket | Yes: tied to git commits, one stored object per file | No: records which dataset a run used | Yes (Artifacts) |
| What a run records without extra code | Code snapshot, installed packages, seed, dataset versions read and saved | Git commit and pipeline dependencies | Git commit, parameters and metrics | Git commit, packages, parameters and metrics |
| Rerun a past result and check it still holds | `fieldbook run --rerun` | `dvc repro` for pipelines | No | No |
| Experiment tracking and UI | MLflow's | Through its VS Code extension or Studio | Yes | Yes |
| Launch the same script on a cloud GPU | Yes (AWS and RunPod verified) | No | No | Yes (Launch) |
| Servers or accounts needed | None | None | None locally; a server to share | An account, or a self-hosted server |
| Made for coding agents | Yes: machine-readable output, attribution, notes, project instructions | No | No | No |

As of October 2026 and to the best of our knowledge; [open an issue](https://github.com/vo-h/fieldbook/issues) if a cell is wrong or out of date.

## Quick start

Requires macOS or Linux (Windows is not supported; WSL should work but is untested), Python 3.11+, git, and [gitleaks](https://github.com/gitleaks/gitleaks) (`brew install gitleaks`).

```bash
pip install "fieldbook[all]"
cd my-project && fieldbook init          # add --agents to brief a coding agent

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

For coding agents: [AGENTS.md](AGENTS.md) (working on this repository) and [llms.txt](llms.txt) (a summary with links, for assistants answering questions about fieldbook).

Design (`design/`):

- [design/README.md](design/README.md): what fieldbook is, requirements, how the parts fit, and an index of the documents below
- [design/projects.md](design/projects.md): projects, settings, credentials
- [design/storage.md](design/storage.md): the storage root; local disk and S3
- [design/data.md](design/data.md): versioned datasets
- [design/runs.md](design/runs.md): `fieldbook run`, code snapshots, seeds, smoke runs
- [design/tracking.md](design/tracking.md): MLflow tracking, the model registry, costs
- [design/cloud-runs.md](design/cloud-runs.md): runs on cloud GPUs
- [design/sharing.md](design/sharing.md): projects shared between people
- [design/testing.md](design/testing.md): how the tests are organized

## Privacy

fieldbook collects nothing and sends nothing anywhere but the storage, database, and cloud accounts you configure. One of its dependencies does: SkyPilot, used for cloud runs (the `aws` and `runpod` extras), reports anonymous usage statistics to its developers unless `SKYPILOT_DISABLE_USAGE_COLLECTION=1` is set.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security problems as described in [SECURITY.md](SECURITY.md), not in a public issue.

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

CI (`.github/workflows/ci.yml`) runs lint and tests on Linux (Python 3.11–3.13, with Postgres) and macOS, and once more at the lowest dependency versions allowed.

## License

Apache-2.0
