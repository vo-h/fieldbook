# Changelog

## 0.1.0 (unreleased)

First release: local research, data in S3 and cloud runs on AWS and RunPod, and shared projects. Tested on macOS and Linux with Python 3.11–3.13; tested dependency versions in `constraints.txt`.

The command is `fieldbook`, settings are `FIELDBOOK_*` variables and `[tool.fieldbook]`, and MLflow tags are `fieldbook.*`.

### Local

- **Datasets:** versioned datasets (`fieldbook data`) of files with metadata, or of the rows of tables (Parquet, JSON lines, delimited text). Each is one Lance table holding the files or rows themselves, so a million small files are a handful of objects in a bucket. Add, update, remove, move, copy, tags, revert, diff, schemas (Pandera), SQL (`fieldbook data query`), `checkout` to real files, verification, and cleanup that never frees data in use.
- **Adding from S3:** `fieldbook data add NAME s3://bucket/prefix` reads a bucket's objects into the dataset and records the source.
- **Lineage:** every dataset version records its inputs, command, and, when saved in a run, the run, code snapshot, and environment.
- **Nothing to remember:** a project's script started with plain `python` is recorded like `fieldbook run` when it imports fieldbook (`FIELDBOOK_RECORD=0` switches it off).
- **Autologging:** Lightning, XGBoost, LightGBM, and Keras/TensorFlow runs log their parameters and metrics to MLflow with no logging code (`autolog` setting, on by default; models and checkpoints are not uploaded).
- **History:** `fieldbook log` lists what has been run (command, git state, metrics, cost, datasets saved), from dataset versions and MLflow together; `fieldbook log diff RUN_A RUN_B` shows what differs between two runs: command, code, packages, data, parameters, metrics. Both have `--json`. `diff` and `note` are subcommands of `fieldbook log`.
- **Queries:** `fieldbook log query "SQL"` and `fb.runs(query=...)` run SQL over the runs (tables `runs`, `metrics`, `params`, `data`, `notes`); `--format markdown` prints a table for a write-up, each row with its run ID.
- **Notes:** `fieldbook log note RUN "what it showed" --verdict good|bad|inconclusive` (or `fb.note()` in a script) keeps conclusions with the run; `fieldbook log` shows them.
- **Repeats:** a run with the same code, packages, command, and data as an earlier one is reported before it starts, with that run's result; a cloud run is then not launched without `--allow-repeat`. One hash is compared, not files.
- **Reruns:** `fieldbook run --rerun RUN` runs an earlier run again from its record (snapshot, command, seed, dataset versions) and says whether its metrics and saved files came out the same; `fieldbook run --rerun RUN --lr 0.01` changes one thing. Nothing is added to datasets.
- **Kept of every run:** `fieldbook run` keeps a run's output, the files it wrote, and how it ended, in the project's storage, also for cloud runs after their machine is gone: `fieldbook log output`, `fieldbook log files`, `fieldbook log why` (the cause of a failure: time limit, out of memory, missing package, the exception), and `fieldbook log wait` (block until a run ends). `fieldbook log` now lists every run started with `fieldbook run`, with its status.
- **Freeing space:** `fieldbook log rm --old` frees the checkpoints, output, and files of runs older than 30 days and the environment images not used since (`--dry-run` lists them; runs noted `good` are kept); `fieldbook log rm RUN` does one run. The runs stay in the history.
- **By whom:** runs, dataset versions, and notes made by a coding agent say so next to their author (`ada [claude-code]`; `fieldbook log --agent`); the author stays the person.
- **Every cloud run is priced:** costs are recorded in the project's storage for every cloud run, also ones that log nothing to MLflow, and totals, per-run costs, and the budget read the same record.
- **Costs in one place:** `fieldbook log costs` (what each run cost) and `fieldbook log costs --total` (per month and author, by hand and by agent, with the budget). `fieldbook experiments costs` is gone.
- **Coding agents:** `fieldbook init --agents` adds a section to `CLAUDE.md` and `AGENTS.md` telling an agent how the project records runs; later `fieldbook init` runs keep it current.
- **Git:** every run also records the commit and branch its code was on, and whether it had uncommitted changes (`fieldbook.git`); the snapshot remains what reproduces the run.
- **Runs:** `fieldbook run` snapshots code (gitleaks-scanned, `.fieldbookignore`, 100 MB limit, symlinks outside the repository refused for cloud runs), records installed packages, applies a seed, passes `.env` secrets (masked in records), and runs in the project's environment. `fb.init()` does the same for notebooks. Smoke runs (`--smoke`) use samples and a scratch area. Every run gets a checkpoint folder, `PLATFORM_CHECKPOINT_DIR`.
- **Experiments:** MLflow over a local store, with its UI in `fieldbook log ui`, run tags for code, data, command, and environment, and system metrics.
- **Settings:** `fieldbook init`; `fieldbook config` shows every setting, its source, and where runs log. Every setting belongs to the project: the project file (committed), the project's `.fieldbook/config.toml` (this machine only), or `FIELDBOOK_<NAME>`; credentials in the project's `.env`. Each project keeps its local data and databases in its own `.fieldbook/` folder, shared by its git worktrees. In the storage root (a folder or a bucket), each project keeps everything in a folder of its own, `<root>/<project>/`, so a team's projects can share one bucket with access per project. The author defaults to git's `user.name`.

### Cloud

- **Data in S3:** the storage root can be an AWS S3 bucket (extra `s3`); `fieldbook init` checks it, and its region is detected once per machine. Files are uploaded once, straight to their content-addressed name.
- **Shared tracking:** MLflow runs from the laptop and cloud machines go to a shared Postgres database (`tracking_uri`, password in `.env`), with no MLflow server.
- **Cloud runs:** `fieldbook run --on remote` runs on a fresh AWS machine (extra `aws`) or RunPod GPU (extra `runpod`), deleted when the job ends, with `--gpus`, `--nodes` (multi-node `torchrun`), `--cloud`, and `--secret`. Task files (`train.yaml`) work locally and in the cloud.
- **Environment images:** one Docker image per `uv.lock`, built on a cloud machine or locally (`fieldbook build`), stored in ECR, with the exact local fieldbook inside.
- **Permissions:** on AWS, machines get a minimal IAM role (created once by `fieldbook init`, after asking) instead of copied keys; RunPod pods get the laptop's keys as secrets.
- **GPUs by memory:** `--gpus 24GB+` (any GPU with at least that much memory), `48GB+:2`, or several choices in order (`--gpus RTX4090,24GB+`), with `--max-price` an hour; the cheapest machine with capacity is launched. Names (`--gpus A10G:1`) work as before.
- **Dry run:** `fieldbook run --on remote --dry-run` shows the machines that fit, their prices, the time limit, the most the run can cost, and the budget left, and launches nothing.
- **Spending limits:** every cloud run has a time limit enforced on the machine (24 hours by default, `--max-hours`); `budget_run_usd` and `budget_monthly_usd`, set in the project file only, cap a run and the project's month, and a launch that would go over is refused.
- **Cost:** each cloud run's cost, from the cloud's own records (AWS CloudTrail × price; RunPod billing), is recorded with the run; `fieldbook log costs` shows it.

### Shared projects

- **Members from GitHub:** the repository's collaborators are the project's members (Write and above) and viewers (Read). `fieldbook access sync` gives each a database login, encrypted with age to their GitHub SSH keys in `.fieldbook-members/`, and keeps the AWS groups `fieldbook-<project>-write`/`-read` in line; `fieldbook init` writes a member's credentials to `.env`. `fieldbook access aws-policy` and `runpod-key`.
- **Attribution:** runs are tagged `fieldbook.author`, which `fieldbook init` sets to the person's GitHub login; `fieldbook log costs --total` sums cloud costs per month and author.
- **Saves write only their changes:** one Lance merge-insert per save instead of rewriting the whole index; concurrent saves to different paths are redone cheaply, the same path still conflicts. Checked with concurrent writers and tag races on S3.
- **Projects are kept apart:** a project reads and copies only its own datasets; `fieldbook data --project` and `fb.dataset(..., project=...)` are gone.
