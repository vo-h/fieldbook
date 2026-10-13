# Experiment tracking

Status: Built · Last updated: 2026-10-09

Experiment tracking and the model registry are MLflow. fieldbook points runs at the right store, tags them with what they used, shows the UI, and records what cloud runs cost. Code: `fieldbook/tracking/`.

**MLflow is used only by code that logs to it.** An MLflow run exists only if the script logs to MLflow; a pure data transform is recorded on the dataset versions it writes (runs.md §1).

## 1. Where runs are logged

- **This machine's store (default):** runs write to a SQLite database directly, `<project>/.fieldbook/mlflow/mlflow.db` (the local state folder, projects.md §3); no server is needed. Each machine then has its own runs.
- **A shared database:** the `tracking_uri` setting names a Postgres database every machine logs to directly (a managed one such as Neon needs nothing to run; a team uses one server with a database per project, sharing.md §3), so runs from laptops and cloud machines are in one place and remote metrics are live. The address holds no user name or password: the login is `PGUSER` and `PGPASSWORD` in `.env` (projects.md §4). `MLFLOW_TRACKING_URI` overrides the setting. The experiment is created there as below, and `MLFLOW_ENABLE_ASYNC_LOGGING` is turned on unless set, so log calls do not wait for the network. The URI is passed to MLflow in environment variables, never on a command line, and shown without its password. A URI naming an MLflow server (`http(s)://`, `databricks`) is used unchanged; the server sets artifact locations.
- **Read-only logins:** on a shared project, a viewer's database login can read runs but not record them. `store.read_only` asks Postgres whether the login may insert runs; `fieldbook run` stops before the script for a viewer, and their machine does not record costs (sharing.md §2). A refused login says the person's credentials may have changed (`git pull`, then `fieldbook init`).
- MLflow is not a dependency of `fieldbook`; it is installed with `fieldbook[tracking]` or by the research project. All checks and the experiment setup run with the interpreter the script uses (runs.md §3), not the one `fieldbook` runs on. Without MLflow there, the MLflow setup is skipped. With only `mlflow-skinny`, which cannot use the SQLite store, it is skipped with a note to install full MLflow, since runs would otherwise fail to start.

## 2. Experiments and artifacts

- **Experiment:** created on first use with artifact location `<root>/<project>/mlflow/artifacts/<experiment>`. MLflow's client otherwise defaults artifacts to `./mlruns` in the working directory. MLflow's database schema stores each run's artifact address in 200 characters (the location plus `/<run ID>/artifacts`); with Postgres or MySQL, a longer one stops the run before it starts, saying how much to shorten the storage root or experiment name (SQLite does not enforce the limit).
- The experiment defaults to the project's name (`experiment` setting, `--experiment`); smoke runs go to `<experiment>-smoke`.
- With a shared database the experiment is asked for through MLflow on every run; with this machine's database it is checked directly in SQLite first, since creating it takes seconds.

## 3. Run tags

- **Tags:** a run-context plugin (`mlflow.run_context_provider` entry point) adds tags to every MLflow run started in a launched process:

| Tag | Value |
|---|---|
| `fieldbook.run_id` | Platform run ID |
| `fieldbook.code` | Code snapshot hash |
| `fieldbook.git` | The commit, branch, and `clean` or `dirty` when the snapshot was taken (runs.md §4) |
| `fieldbook.env` | Environment record |
| `fieldbook.author` | Who ran it: the `author` setting (projects.md §2) |
| `fieldbook.agent` | The coding agent that ran it for the author, if one did (runs.md §20) |
| `fieldbook.smoke` | Items per folder, in smoke runs (runs.md §9) |
| `fieldbook.command` | The script's command line, secret-like flag values masked. With config files in the code snapshot, this reproduces the full configuration (e.g. Hydra overrides). Resolved configs are logged by the user's config tool if wanted. |
| `fieldbook.inputs` | Dataset versions read by the process, JSON list |
| `fieldbook.outputs` | Dataset versions written by the process, JSON list |

Reads before `mlflow.start_run()` are included; later reads and writes update the tags. A process with several MLflow runs attributes all its reads to each. Tags are used instead of params because params cannot change once set.

Runs on cloud machines also get `fieldbook.cluster`, `fieldbook.cloud`, `fieldbook.region`, `fieldbook.nodes`, and `fieldbook.machine` (with `fieldbook.machine_type` on AWS), and later the metric `fieldbook.cost_usd` (§7).

## 4. System metrics

MLflow's system metrics (CPU, GPU, memory, disk, network) are switched on for launched and `init()` runs when the interpreter has `psutil` (installed with `fieldbook[tracking]`, with `nvidia-ml-py` for GPUs). They stay off otherwise, because MLflow refuses to start runs when they are on and `psutil` is missing. An explicit `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING` is respected.

## 5. MLflow UI

**MLflow UI.** `fieldbook log ui` runs MLflow's server in the foreground, as `mlflow ui` does: it prints the address, opens the browser, and stops with Ctrl-C. Runs never need it; they write to the database directly (§1), which keeps `fieldbook run` fast (a background server cost 17 s on first start and 520 MB while running) and leaves no process or state file behind. The server reads the same store: `127.0.0.1`, port 5050 or the next free one (`mlflow_port`; MLflow's default 5000 is taken by AirPlay Receiver on macOS), artifacts read from `<root>/<project>/mlflow/artifacts` directly (`--no-serve-artifacts`). It runs one web worker, and MLflow's background job runner (used by its GenAI scorers) is off unless `MLFLOW_SERVER_ENABLE_JOB_EXECUTION` is set: 2 processes and 520 MB instead of MLflow's defaults of 15 processes and 1.9 GB.

## 6. Model registry

Models are registered from training runs with MLflow's own calls (`mlflow.<flavor>.log_model(..., registered_model_name=...)`) and promoted with aliases such as `prod`. Each model version links to its run, and the run to its code, environment, and dataset versions through its tags (§3), so a model's lineage needs nothing extra. fieldbook does not deploy models; demo and evaluation endpoints (`mlflow models serve`) are recipes in the guide.

## 7. Costs

What cloud runs cost is recorded in the launch ledger, not in MLflow (cloud-runs.md §10); a run that logged to MLflow gets a copy as the metric `fieldbook.cost_usd` with the tag `fieldbook.cost_basis`, for the MLflow UI. Costs are per project.

Costs are read in one place, `fieldbook log costs`, which first prices whatever has ended since the last look:

| Command | Shows |
|---|---|
| `fieldbook log costs [-n N]` | What each run cost, newest first: run, start, author, command, cost |
| `fieldbook log costs --total` | Totals per month and author, each author's split into runs started by hand and by a coding agent (runs.md §20), and where the month stands against the budget (cloud-runs.md §11) |

`fieldbook log` has no cost column and `fieldbook experiments` no `costs` command: two ways into the same numbers were one too many.

## 8. Autologging

Training frameworks log their parameters and metrics to MLflow with no logging code in the script (`fieldbook/launch/autolog.py`; setting `autolog`, on by default). When a run logs to MLflow and the script imports one of the frameworks below, MLflow's own autologging for it is switched on right after the import, by the same after-import hook that seeds NumPy and PyTorch (runs.md §8).

| Imported | MLflow autologging |
|---|---|
| `lightning`, `pytorch_lightning` | `mlflow.pytorch` |
| `xgboost` | `mlflow.xgboost` |
| `lightgbm` | `mlflow.lightgbm` |
| `keras`, `tensorflow` | `mlflow.keras`, `mlflow.tensorflow` |

- **Not `mlflow.autolog()`:** it also hooks scikit-learn, so a data transform that fits a scaler would start an MLflow run, and LLM client libraries. Only training frameworks are listed, so data runs still log nothing to MLflow.
- **What is logged:** parameters and metrics. Models, checkpoints, input examples, and MLflow's own dataset records are not (`log_models=False`, `checkpoint=False`, ...): uploads of that size are the script's decision, and datasets are already recorded as the run's inputs (§3).
- **Versions MLflow has not tested:** autologging is still switched on (not `disable_for_unsupported_versions`). MLflow's list lags the frameworks (Lightning 2.6 is "unsupported" in MLflow 3.16 and works), and a patch that fails falls back to the framework's plain call.
- **Never stops a run:** an error while switching it on is ignored.
- **Plain PyTorch loops** have nothing to hook: they call `mlflow.log_metric` themselves. HF Trainer has its own switch, `report_to="mlflow"` (runs.md §11).
- **With a logger of your own** (Lightning's `MLFlowLogger`), set `autolog = false`, or the run is logged twice.
- **Where it applies:** `fieldbook run` (the bootstrap), plain `python` on a script that imports fieldbook (runs.md §13), `fb.init()`, and cloud runs (`FIELDBOOK_AUTOLOG` travels with the run's settings).

## 9. Testing

Tags on runs, inputs read before `start_run`, the artifact location under the storage root, the shared Postgres database end to end (a throwaway server), costs from fake cloud records, the costs summary, and the UI started and stopped with Ctrl-C with nothing left running. Autologging: switched on at a framework's import with only the options its flavor accepts, off with the setting or without a tracking store, and a failure ignored; checked once by hand with Lightning 2.6 and MLflow 3.16 (parameters and `train_loss` logged with no logging code).
