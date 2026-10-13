# Runs

Status: Built · Last updated: 2026-10-09

`fieldbook run`, plain `python` in a project, and `fb.init()`: what a run records before it starts (code, environment, seed), what its process is given, and smoke runs. Code: `fieldbook/launch/` (starting runs, and the run's own process), `fieldbook/code/` (snapshots, the environment record). Runs on cloud machines add to this: [cloud-runs.md](cloud-runs.md). What runs log to MLflow: [tracking.md](tracking.md).

## 1. Principles

- **Every `fieldbook run` snapshots its code**, whether it trains or transforms data.
- **Dataset versions carry their own lineage.** A version written in a run records the run ID, code snapshot, environment, inputs, and command. No MLflow access is needed to read it.
- **MLflow is used only for training.** An MLflow run exists only if the script logs to MLflow. A pure data transform is recorded only on the dataset versions it writes. A script that does both gets both.
- **Nothing to remember.** A project's script started on its own (`python clean.py`) is recorded like `fieldbook run clean.py`, because it imports fieldbook (§13). Only a process no run was started for (recording switched off, or a script outside a project) records just inputs and command on the versions it saves (data.md §11).
- **Nothing is required of research code.** Runs receive environment variables (§7) and the variables frameworks already read (§11).
- **The same run anywhere.** A script that runs on this machine runs on a cloud GPU with `--on remote`, with the same run ID, snapshot, seed, and records.

| Recorded | How | Stored as |
|---|---|---|
| Code | Tracked and untracked files, minus ignored ones; scanned for secrets (§4) | `code/<tree-hash>.tar.gz`, once per tree hash |
| Place in git | The commit, the branch, and whether the files differed from the commit (§4) | `git` on dataset versions; `fieldbook.git` tag |
| Environment | Python, installed packages, `uv.lock` hash, platform, CUDA driver (§5) | Environment record (JSON) |
| Data | Dataset versions read and written (data.md §11) | `inputs` on dataset versions; `fieldbook.inputs` and `fieldbook.outputs` MLflow tags |
| Configuration | The command line; config files are in the snapshot | `command` on dataset versions; `fieldbook.command` tag |
| Randomness | A seed applied to Python, NumPy, and PyTorch in every process (§8) | Environment record |
| Who | The `author` setting: the person's GitHub login (projects.md §2) | `author` on dataset versions; `fieldbook.author` tag |

## 2. Commands

```
fieldbook run <script.py> [args...]      # run a Python script in the project's environment (§3)
fieldbook run <task.yaml>                # run a SkyPilot task locally (§6)
fieldbook run <command> [args...]        # run any command, e.g. torchrun, in the project's environment
    --seed N                       # default: random; recorded either way
    --experiment NAME              # MLflow experiment (default: the project)
    --max-snapshot-size MB         # refuse larger code snapshots (default: 100)
    --smoke [--sample N] [--keep]  # smoke run: sampled reads, saves to scratch (§9)
    --deterministic                # deterministic PyTorch algorithms (§8)
    --on remote ...                # on a cloud machine (cloud-runs.md)
fieldbook log code <run|hash> <dir>   # restore the code a run used; a hash may be a unique prefix (min. 7)
fieldbook log packages <run|hash>     # print a run's installed packages (`packages` in §5)
```

Arguments after the script or task file are passed to it unchanged.

**Commands and multi-process training.** A target that is neither `.py` nor `.yaml` is a command, found on the project environment's `PATH` (§3); an unknown command is an error before anything runs. For `torchrun [OPTIONS] SCRIPT.py ARGS`, `fieldbook run` inserts `-m fieldbook.launch.bootstrap` before the script, so every worker applies the run's seed (§8), the same in every rank, as Lightning's `seed_everything()` does. With `torchrun -m MODULE` or `--no-python` the seed is not applied, with a note. All workers share the run's context (§7): dataset versions and MLflow runs should come from rank 0. `mp.spawn` needs nothing extra. Multi-node training is for cloud runs (cloud-runs.md §9). `fieldbook run` exits with the script's exit code, or 128 + N if the script was stopped by signal N.

**Signals.** The script never outlives `fieldbook run`: SIGTERM and SIGHUP sent to `fieldbook run` (e.g. by a scheduler or timeout) are passed on to the script, and `fieldbook run` waits for it to exit. Ctrl-C in a terminal reaches the script directly, so `fieldbook run` ignores SIGINT rather than sending it twice. SIGKILL cannot be caught; killing `fieldbook run` with it leaves the script running.

## 3. Run steps

1. **Run ID:** `YYYYMMDD-HHMMSS-xxxxxx` (UTC time plus 6 random hex digits). Sortable and unique per machine.
2. **Code snapshot** (§4). The run is aborted if the snapshot fails.
3. **Environment record** (§5).
4. **MLflow setup** (tracking.md §2), if MLflow is installed. A project's viewer is stopped here: their database login cannot record runs (sharing.md §2).
5. **Execute** with the environment variables in §7, in the project's environment: the active virtual environment (`$VIRTUAL_ENV`), else the nearest `.venv` from the working directory up to the repository root, else the interpreter running `fieldbook`. Scripts run with that environment's `python`; task commands get its `bin/` first on `PATH`. This matters when `fieldbook` is installed globally (`pipx`, `uv tool`), where its own interpreter lacks the project's packages. The run ID, code hash, seed, and interpreter are printed to stderr.

**No nested runs.** `fieldbook run` refuses to start inside another run (`PLATFORM_RUN_ID` set, by `fieldbook run` or `fb.init()`). A child run would inherit the parent's settings (seed, smoke mode, determinism) and record its outputs under a different run, so lineage would split. A script that runs several steps calls them directly. Multi-process training (`torchrun`, `mp.spawn`) is one run (§2).

## 4. Code snapshots

- **Repository required.** The snapshot covers the git repository containing the working directory (or the task's `workdir`). Outside a repository, `fieldbook run` fails with a hint to run `git init`.
- **Files:** tracked and untracked files, minus untracked files matched by `.gitignore` (as in git), any file matched by `.fieldbookignore`, `.env` files (projects.md §4), and the project's `.fieldbook/` folder (projects.md §3). `.fieldbookignore` is an optional file at the repository root, in gitignore syntax, for keeping outputs, checkpoints, and data out of snapshots without hiding them from git. Cloud runs ship the snapshot itself, so the same file controls what runs remotely. Symlinks are stored as links; those pointing outside the repository (or to absolute paths) cannot be reproduced, so cloud runs refuse them before launching and local runs warn.
- **Size limit:** if the files total more than 100 MB (`--max-snapshot-size`), the run is aborted before any content is read, listing the largest files. This catches outputs written inside the repository, which would otherwise be archived again by every later run.
- **Identity:** the git tree hash of exactly these files, computed with `git update-index` into a temporary index and `git write-tree`. The user's index and branch are untouched. Identical code gives the same hash; changes to ignored files do not change it.
- **Storage:** `code/<tree-hash>.tar.gz` in the project's folder of the storage root (storage.md §1), written to `tmp/` and moved into place. If the archive exists, scanning and archiving are skipped.
- **Secret scan:** the files are copied to a staging directory and scanned with `gitleaks dir` (redacted JSON report). Any finding aborts the run and lists rule, file, and line, never the secret. If `gitleaks` is not installed, the run is aborted with an install hint.
- **The snapshot reproduces the run; the commit says where it sits in git.** A commit hash alone would not do: research code usually runs with uncommitted edits, a saved diff leaves out untracked files (a new config, a helper script), commits disappear with rebases and deleted branches, and a cloud machine would need access to the git remote. So the snapshot is what a run is reproduced from. The run also records `<commit> <branch> clean|dirty` (`snapshot.git_state`; "dirty" when `git status` reports any change), shown as `git a1b2c3d main+changes`: a clean run links to a commit on GitHub and its snapshot equals that commit; a dirty one says which commit the local changes were made on. A cloud machine has no repository, so the laptop passes it along. Empty before the first commit.
- **Retention:** `fieldbook data rm --old` (all datasets) deletes code snapshots and package lists (§5) that no remaining dataset version and no active MLflow run refers to, once unused for the retention window (`--days`, default 7) rather than the 24-hour grace period, since a long run refers to its snapshot only when it saves a version or logs to MLflow. Reusing a snapshot or package list restarts its window.

## 5. Environment record

A JSON object passed to the run (`PLATFORM_ENV`) and stored with its outputs: in the `env` property of dataset versions it saves, and the `fieldbook.env` tag of its MLflow runs.

| Field | Source |
|---|---|
| `seed` | `--seed`, or the random seed chosen for the run |
| `cwd` | The folder the command was run in, relative to the repository root, so that the run can be started there again (§17) |
| `python` | Version of the interpreter the run uses (§3) |
| `platform` | OS and architecture |
| `packages` | SHA-256 of the installed packages (`name==version` per line), queried from the run's interpreter (cached per machine in the state folder, one file per interpreter, until its `site-packages` folders change; about 270 ms saved per run). The list is stored once at `packages/<hash>.txt`; `fieldbook log packages` prints it. Works for uv, pip, and conda environments. |
| `uv_lock` | SHA-256 of `uv.lock` at the repository root, or null. This is what should be installed; `packages` is what was. |
| `cuda_driver` | `nvidia-smi` driver version, or null |

Local runs use no container image; cloud runs add the image's digest (cloud-runs.md §2).

## 6. Task files

A SkyPilot task YAML, run locally: `setup`, then `run`, as `bash` commands, with `envs` added to the environment and `workdir` (relative to the YAML file) as the working directory. Other fields (`resources`, `num_nodes`, `file_mounts`, ...) are ignored locally with a note. `fieldbook run script.py args` is equivalent to a task whose `run` is `python script.py args`.

## 7. Environment variables

The `PLATFORM_*` variables are the run context (`fieldbook/launch/context.py`): `fieldbook run` and `fb.init()` write them, and the data module, MLflow plugin, and bootstrap read them through `context.current()`.

| Variable | Value |
|---|---|
| `PLATFORM_RUN_ID` | Run ID |
| `PLATFORM_CODE` | Code snapshot hash |
| `PLATFORM_GIT` | `<commit> <branch> clean|dirty` when the snapshot was taken (§4) |
| `PLATFORM_ENV` | Environment record (JSON) |
| `PLATFORM_SEED`, `PYTHONHASHSEED` | Seed |
| `PLATFORM_DETERMINISTIC` | `1` with `--deterministic` |
| `PLATFORM_SECRETS` | Names (not values) of the secrets from `.env`, for masking (projects.md §4) |
| `PL_GLOBAL_SEED` | Seed, for Lightning (§11) |
| `PL_TRAINER_FAST_DEV_RUN` | `5` in smoke runs, for Lightning (§11) |
| `PLATFORM_SMOKE` | Items per folder in a smoke run (§9) |
| `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING` | `true` if the interpreter has `psutil` (tracking.md §4) |
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME` | tracking.md §1–2, if MLflow is installed |
| `PLATFORM_CHECKPOINT_DIR` | A folder per run for checkpoints, `<root>/<project>/checkpoints/<run-id>` (cloud-runs.md §8) |
| `PLATFORM_OUTPUT_DIR` | A folder per run on the machine, whose files are kept with the run (§18); set by `fieldbook run` |
| `FIELDBOOK_PROJECT`, `FIELDBOOK_STORAGE_ROOT`, `FIELDBOOK_AUTHOR`, ... | The resolved settings, so the script's own fieldbook calls use them whatever its shell (`fieldbook/launch/runenv.py`) |

Dataset versions saved while `PLATFORM_RUN_ID` is set record `run_id`, `code`, and `env` in their transaction properties, next to `inputs` and `command`.

## 8. Seeds

Python scripts are started through `python -m fieldbook.launch.bootstrap SCRIPT ARGS`, which behaves like `python SCRIPT ARGS` (same `sys.argv`, `sys.path[0]`, and `__main__`, including for multiprocessing) after seeding: Python's `random` at once, and NumPy and PyTorch (including CUDA) when they are first imported, so scripts that do not use them pay no import time. `--deterministic` also switches PyTorch to deterministic algorithms (`torch.use_deterministic_algorithms`, cuDNN benchmarking off, `CUBLAS_WORKSPACE_CONFIG`); it is off by default because it can slow training and makes some operations raise errors. The seed and the flag are in the environment record (§5). Seeding needs fieldbook in the project's environment; otherwise the script runs directly and a note says the seed is not applied. Task commands get `PLATFORM_SEED` and `PYTHONHASHSEED` only.

## 9. Smoke runs

`fieldbook run --smoke [--sample N]` tests a script end to end on a sample, without side effects (README.md §2, T6):

- **Reads:** `files()` (and everything built on it: `stats`, `checkout`, `fieldbook data query`) returns up to N items (default 100) from each top-level folder under the prefix, so small splits such as `val/` are never empty. The items with the lowest hashes are taken: the same items every run. `open()` and `read_many()` still read any item. A records dataset gives its first N × 100 rows (`to_table`, `fieldbook data query`).
- **Also used by reruns** (§17), without the sampling.
- **Writes:** saving a version runs every check (schema, conflicts) and saves it to a scratch area, `<root>/<project>/scratch/<run-id>/`, instead of the dataset; later reads in the run see it, so a script can read back what it wrote. The scratch area is deleted when the run ends (`--keep` keeps it; `FIELDBOOK_SCRATCH=<id>` gives a read-only view). Tags, reverts, and deletions are skipped. Each is reported on stderr. The scratch area holds each saved dataset as `scratch/<run-id>/datasets/<name>.lance`, with versions starting at 1; file contents go to the object store as usual, and lineage refers to scratch versions as `scratch/<id>/<project>/<name>@<version>`, never confused with real ones. `fieldbook data rm --old` deletes scratch areas unchanged for longer than the grace period (24 hours), e.g. after a crash, and cleanup then frees files stored only by a smoke run.
- **MLflow:** runs go to the experiment `<experiment>-smoke` and are tagged `fieldbook.smoke`.
- **Step limits:** Lightning stops after 5 batches (`PL_TRAINER_FAST_DEV_RUN`, §11). Other scripts read `PLATFORM_SMOKE` to limit steps themselves.

A run on a subset whose results are kept (`--sample` without `--smoke`) is not supported: its lineage would have to record that inputs were sampled.

## 10. Notebooks

`run = fb.init(seed=None, experiment=None, deterministic=False)`, at the top of a notebook, does what `fieldbook run` does for the current process: run ID, code snapshot of the repository containing the working directory, environment record, seed, and MLflow settings. Dataset versions saved and MLflow runs started afterwards record the run. The snapshot holds files as they are on disk, so the notebook should be saved first; the notebook's outputs are not recorded (users can log the notebook to MLflow). Calling `init()` again starts a new run; inside `fieldbook run`, it returns the launched run.

`fb.seed(value, key=None, deterministic=False)` seeds Python, NumPy, and PyTorch in a process the launcher did not start, such as a trial of a hyperparameter search. With `key` (e.g. the trial's config), the seed is derived from `value` and `key`, so each trial gets its own seed and a rerun of the same trial the same one (docs/guide.md, hyperparameter search).

## 11. Framework integration

Frameworks are configured through the environment variables they are designed to read, not by patching their code (as SkyPilot, SageMaker, and Lightning's own cluster integrations do). Explicit arguments in user code always win. Verified with transformers 5.18 and Lightning 2.6:

| | Lightning | HF Trainer |
|---|---|---|
| MLflow | Logged automatically (tracking.md §8). A script's own `MLFlowLogger()` uses the run's tracking URI; its runs are tagged (tracking.md §3). It logs to experiment `lightning_logs` unless `experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"]` is passed. | Logs nothing by default since transformers 5; pass `report_to="mlflow"`. It then uses the run's tracking URI and experiment, and its runs are tagged (tracking.md §3). |
| Seed | `seed_everything()` reads `PL_GLOBAL_SEED`, set to the run's seed. Without it, Lightning re-seeds with 0. | Re-seeds with `TrainingArguments(seed=42)` by default; pass `seed=int(os.environ["PLATFORM_SEED"])` to use the run's seed. HF logs the seed it used to MLflow. |
| Smoke runs | `PL_TRAINER_FAST_DEV_RUN=5`: 5 batches, logging and checkpoints suppressed | No variable exists; e.g. `max_steps=5 if os.environ.get("PLATFORM_SMOKE") else -1` |

Experiments created by frameworks or user code (e.g. `lightning_logs`) get MLflow's default artifact location, `./mlruns` in the working directory, so the guide tells users to log to the run's experiment (`MLFLOW_EXPERIMENT_NAME`); `fieldbook init` adds `mlruns/` to `.fieldbookignore`. Checkpoints go under `PLATFORM_CHECKPOINT_DIR` (cloud-runs.md §8).

## 12. Evaluation, profiling, and search recipes

Both are runs like any other (README.md §2, T7 and T8); the recipes below are tested.

**Evaluation and benchmarks.** An evaluation is a script run with `fieldbook run` that reads a pinned dataset version and logs metrics to MLflow. The evaluated version is recorded as the run's input (`fieldbook.inputs`) without extra code, so evaluations of different models on the same version are compared in the MLflow UI. `mlflow.models.evaluate` adds standard metrics for common model types; benchmark tools with a command line (e.g. lm-evaluation-harness) run as a task file.

```python
test = fb.dataset("pets-test", version=1)        # pin the evaluated version
rows = test.files().to_pylist()
accuracy = sum(predict(r["path"]) == r["label"] for r in rows) / len(rows)
with mlflow.start_run(run_name="eval-cats-v1"):
    mlflow.log_param("model", "models:/cats/1")
    mlflow.log_metrics({"accuracy": accuracy, "n": len(rows)})
```

**Profiling.** System metrics (tracking.md §4) show whether the GPU is idle; PyTorch's profiler shows why. Traces are written outside the repository (or they would enter the next code snapshot) and logged as MLflow artifacts next to the run's system metrics; Perfetto (ui.perfetto.dev) opens them.

```python
with mlflow.start_run(), tempfile.TemporaryDirectory() as tmp:
    trace = Path(tmp) / "trace.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=1, active=3),
                 on_trace_ready=lambda p: p.export_chrome_trace(str(trace))) as prof:
        for batch in loader:
            train_step(batch)
            prof.step()
    mlflow.log_artifact(str(trace), "profile")
    mlflow.log_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10),
                    "profile/summary.txt")
```

**Hyperparameter search** is a recipe, not a feature: Ray Tune with Optuna and an early-stopping scheduler inside one run, trials logged to MLflow under a parent run by Tune's MLflow integration, seeds per trial with `fb.seed(..., key=config)` (docs/guide.md §6). Searches across several machines (a Ray cluster) and grids launched as separate cloud runs are out of scope. Endpoints for demos (`mlflow models serve`) and LLM evaluations (vLLM, llama.cpp) are recipes in the guide too.

## 13. Recording without `fieldbook run`

A run is recorded whether or not anyone remembered to type `fieldbook run`: when a project's script imports fieldbook, the import starts the run (`fieldbook/launch/autorecord.py`, called at the end of `import fieldbook`). Both kinds of run a project has, data transforms and training, use fieldbook's datasets, so both have the import; it is an ordinary, visible line of the script, not a hook hidden in the environment.

| Started as | Who creates the run | `import fieldbook` |
|---|---|---|
| `fieldbook run train.py`, also `--on remote` | `fieldbook run`, before the script starts | Joins that run |
| `python train.py` in a project | Nobody yet | Starts it: run ID, code snapshot, git state, environment record, seed, MLflow settings, as `fb.init()` does (§10) |
| Worker processes (data loaders, `torchrun` ranks, subprocesses) | The parent | Join the parent's run (they inherit `PLATFORM_RUN_ID`) |
| A notebook or interactive session, `python -c` | Nobody | Nothing: call `fb.init()` (§10) |
| Tools that import fieldbook (`pytest`, `fieldbook` itself) | Nobody | Nothing |

- **When it starts a run:** no run is in progress; `FIELDBOOK_RECORD` is not `0`; the program being run is a `.py` file inside the project the working directory is in, and not installed in the Python environment. These checks use the standard library only, so `import fieldbook` stays cheap everywhere else (0.09 s).
- **Why at import:** the seed has to be applied before the script's own code uses randomness, and the snapshot has to hold the code as it was when the run began; the import is at the top of the script. A start-up hook in the environment (a `.pth` file) would cover scripts that never import fieldbook, but it is invisible, runs for every Python process in the environment, and such scripts read and write nothing fieldbook tracks.
- **Cost:** about 0.6 s per script start (0.95 s for `fieldbook run`); a second or two more the first time after the code changes (archiving and the secret scan); several seconds once, when the project's run database is created.
- **A run that cannot be recorded still runs:** if the snapshot fails (gitleaks missing, a secret found, not a git repository), the script goes ahead with `fieldbook: this run is not recorded: <reason>` on stderr. `fieldbook run` stops instead, since it was asked to record.
- **Options belong to `fieldbook run`:** `--seed`, `--smoke`, `--deterministic`, `--experiment`, `--on remote`. Plain `python` uses a new random seed (recorded) and the project's experiment.
- **A script that launches runs itself** (calls `fieldbook run` for each step) would start a run by importing fieldbook, and its steps would be refused as nested (§3); it is started with `FIELDBOOK_RECORD=0`.
- **A script that only logs to MLflow** and never imports fieldbook: MLflow loads fieldbook's plugin when its first run starts, which starts the record then; but MLflow has already chosen its store by that point, so the run is tagged yet logged to MLflow's default store, and the seed comes late. Such scripts import fieldbook at the top, or use `fieldbook run`.

## 14. History and comparison

`fieldbook log` is the group for looking at runs (`fieldbook run` starts them, a rerun included: `--rerun`, §17): on its own it lists what has been run in the project, and `diff`, `note` (§15), `output`, `files` (§18), `code`, `packages` (§4), `ui` (the MLflow UI, tracking.md §2), `wait`, `why` (§19), `costs` (tracking.md §7), `query` (§21), and `rm` (§22) are its subcommands (`fieldbook run diff` and the like are refused with a pointer here unless such a command exists). `fieldbook log diff RUN_A RUN_B` shows how two runs differ (`fieldbook/history.py`). Both exist so that "what did I run last week, and why is this result different" is answered from the record, by a person or by a coding agent (`--json` on both).

- **Three sources, merged by run ID:** the run's own record of starting and ending, written by `fieldbook run` (§18); the dataset versions it saved (data.md §11); its MLflow runs (tracking.md §3). None has every run: a script started with plain `python` has no record of its own (it is listed once it saved a version or logged to MLflow), a data transform logs nothing to MLflow, and a training run may save no dataset. The status is how the process ended when that is known (`finished`, `failed`, `running`, `launching`, `killed`), else its MLflow run's.
- **`fieldbook log`** (`-n`, `--author`, `--json`): newest first; run ID, start, author, command, git state, status of its MLflow run, metrics, cloud cost, dataset versions saved. Several MLflow runs of one run (a search's trials) show as one row with the parent's values and the summed cost.
- **`fieldbook log diff A B`** (`--patch`, `--json`): only what differs, in this order: command, git state, author, environment record (Python, seed, platform, ...), installed packages (added, removed, changed versions, from the two package lists), inputs and outputs by dataset name (`raw@3` against `raw@5`), parameters, metrics, and code (files added, removed, modified, and a unified diff of the two snapshots, cut at 400 lines unless `--patch`). Run IDs can be shortened to a unique prefix.
- **Cost of a listing:** every dataset's version log, one MLflow search, and one listing of `runs/`; the small files of a run's own record and its notes are read only for the runs shown (`with_details`, `with_notes`). It grows with the project; fine for thousands of runs, and the place to add a cache if it ever is not.

## 15. Notes

`fieldbook log note RUN "text" [--verdict good|bad|inconclusive]` records what a run showed; `fieldbook log note RUN` reads a run's notes; `fb.note(text, verdict=None)` does it from the run's own script (`fieldbook/notes.py`). Metrics say what happened; a note says what it means ("lr 3e-4 diverges after epoch 2: dead end"). The reader is whoever picks the work up next, and above all a coding agent, which starts every session knowing nothing: `fieldbook log --json` gives it the conclusions with the numbers.

- **Storage:** one small file per note, `notes/<run ID>/<time>-<random>.json` (run, time, author, text, verdict), in the project's folder of the storage root. Not MLflow tags: a data run has no MLflow run. Not one file per run: two members noting the same run would overwrite each other. Nothing is rewritten; a later note adds to the record.
- **Verdict:** `good`, `bad`, or `inconclusive`: whether the run is worth building on. A run's verdict is that of its latest note that gives one. `fieldbook log --verdict bad` lists the dead ends.
- **Reading:** `fieldbook log` reads the notes of the runs it shows (one listing, then one small file per note), not of every run; filtering by verdict reads all.
- **Written by fieldbook too:** a rerun's outcome is noted on the original run and on the new one (§17).

## 16. Repeated runs

Before a run starts, fieldbook says so if an earlier run was the same: same code, environment, command, and data (`fieldbook/launch/repeats.py`). On this machine the run goes ahead with a notice showing the earlier run's result and latest note; a cloud run is not launched unless `--allow-repeat` is given, since it costs money.

Nothing is compared file by file, however large the repository or the datasets, because everything a run depends on is already a single value:

| What | Already one value | Where from |
|---|---|---|
| Code | The snapshot's hash, covering every file (§4) | Taken for the run anyway |
| Environment | The hash of the installed packages (§5); a cloud run's image | Recorded for the run anyway |
| Command | The script (relative to the repository), its arguments, the folder, a task file's variables | The command line |
| Data | Dataset versions: a version number never changes meaning | The earlier run's record |

- **Fingerprint:** the first three, hashed (SHA-256, 32 hex digits). `fieldbook run train.py --lr 1` and `python train.py --lr 1` give the same one. The seed is part of it only when one was asked for (`--seed`): a run with a random seed repeats an earlier run whatever seed that had, and seed studies pass `--seed`.
- **Index:** each run leaves an empty file, `fingerprints/<fingerprint>/<run ID>`, in the project's folder of the storage root. The check is one listing of one folder (2 ms on a local disk; one LIST request on S3), so a run that repeats nothing pays almost nothing.
- **Data cannot be in the fingerprint,** since what a script reads is known only once it has run. So when the listing finds earlier runs, the newest is taken whose record still stands: it is in the history (§14), did not fail, was not a smoke run, and every dataset version it read is still that dataset's latest. Only this path reads the history, and it replaces a whole run. A version the script pins is treated the same way, so a pinned read of an old version is not called a repeat once the dataset has moved on: the check errs towards running.
- **Not checked:** smoke runs, notebooks, the cloud machine's side of a run, and reruns (§17), which repeat on purpose. A script started with plain `python` that saved no version and logged nothing left no record (§18), so repeating it is not noticed.
- **Local and cloud runs are not matched** with each other: their environments differ (this machine's packages, the image).
- **Never stops a local run:** if the check fails (storage unreachable), the run goes ahead with a warning.

## 17. Running a run again

`fieldbook run --rerun RUN [ARGS...]` runs an earlier run again from its record (`fieldbook/launch/rerun.py`): to check that a result holds before it is reported or built on, or, with arguments added to the command, to change one thing and nothing else.

| | A rerun uses |
|---|---|
| Code | The original's snapshot, checked out to a temporary folder (deleted afterwards), in the folder the original ran in (`cwd` of the environment record; for older runs, where the script is found) |
| Command | The recorded one, plus `ARGS` |
| Seed | The original's (`--seed` for another) |
| Data read | The versions the original read: reads without a version are pinned to them (`FIELDBOOK_PIN_INPUTS`); a version the script names itself is used as written |
| Data saved | A scratch area, as in smoke runs (§9) but without sampling: an edit starts from the version before the first one the original saved (`FIELDBOOK_PIN_BASES`; an empty dataset if the original created it), so a transform that adds files does not find them already there. Tags, reverts, and deletions are skipped. Nothing is added to a dataset. |
| Environment | The project's current one. If its packages differ from the original's, the rerun says how many; `fieldbook log diff` lists them. Rebuilding the old environment is `uv sync` in a checkout, not done here. |
| MLflow | Real: the rerun is a run with its own ID, logged to the project's experiment and tagged `fieldbook.rerun_of`. |

When it ends, the rerun is compared with the original: metrics both logged (identical or `a -> b`), and for each dataset the original saved, the files (path and content hash) of its last saved version against the scratch area's. The outcome is one line (`reproduced: 2 metrics (loss, acc), clean@4`; `differs: acc 0.91 -> 0.89`; with `ARGS`, `nothing changed: ...`), also in `--json`, and it is noted on the original run, where someone asking "does this hold?" looks, and on the new run if it logged to MLflow (§15). The exit code is the script's.

- **Only what the snapshot holds:** a script that reads files left out of the snapshot (ignored local data, absolute paths elsewhere) finds them missing or changed. Data belongs in datasets, which the rerun pins.
- **Not reproduced exactly:** a run through a launcher (`fieldbook run torchrun ... train.py` records `train.py ...`, so the rerun is one process), a command given a secret on its command line (masked in the record; refused), and a notebook (no command). Cloud runs are rerun on this machine.
- **A rerun cannot start inside a run,** like any run (§3).

## 18. What is kept of a run

`fieldbook run` keeps three things of every run in the project's folder of the storage root, here and on a cloud machine, so that how a run went can be read from any member's machine, after the machine it ran on is gone (`fieldbook/launch/records.py`, `runlog.py`, `outputs.py`, `keeping.py`):

| Kept | Where | Read with |
|---|---|---|
| Its record: started (command, author, code, where, time limit), alive, ended (exit code) | `runs/<run ID>.json`, `.alive`, `.end.<code>` | `fieldbook log`, `fieldbook log wait`, `fieldbook log why` |
| Its output: stdout and stderr | `runs/<run ID>/output.log.gz` | `fieldbook log output RUN [--tail N] [--follow]` |
| A cloud run's launch: what SkyPilot printed on the laptop (machines tried, image pull). A task file's `setup` runs on the machine, so its output is in the run's output. | `runs/<run ID>/launch.log.gz` | `fieldbook log output RUN --launch` |
| The files it wrote | `runs/<run ID>/files/`, `files.json` | `fieldbook log files RUN [PATH --to DIR]` |

**Only `fieldbook run`.** A script started with plain `python` (§13) has no process around it to do the keeping; capturing output from inside a process (as W&B does) was left out as the part most likely to surprise. Its dataset versions and MLflow runs are recorded as before.

**Record.** Three small files side by side, so one listing of `runs/` gives every run's state and no file is read to list runs. The exit code is in the end file's name for the same reason. `.alive` is rewritten every 30 seconds; a run with no end and no sign of life for two minutes is `killed`: its process or machine was removed from outside (a reclaimed or deleted machine, `kill -9`). The end record also holds how long the run's own process ran, measured by its machine, so a slow launch is not counted. A cloud run's record is started by the laptop at launch (`launching` until the machine's first sign of life), and ended by the laptop if the launch fails. Of a cluster's machines, the first keeps the record.

**Output.** The run's process no longer writes to the terminal directly: `fieldbook run` reads what it prints and writes the same bytes on to its own stdout and stderr, each to its own, and to a buffer. Where `fieldbook run` itself writes to a terminal, the process gets a pseudo-terminal, so it prints as before (unbuffered, progress bars, colours); otherwise a pipe, with `PYTHONUNBUFFERED=1`.

- **Pieces:** S3 cannot append to an object, so every 30 seconds (or megabyte) the lines gathered are written as the next piece, `runs/<run ID>/log-output/000001.gz`. A machine that disappears leaves its output up to the last half minute. Each piece is a gzip member, and members end to end are a gzip file, so joining at the end is concatenation. AWS's services that do stream (CloudWatch Logs, Firehose, S3 Express One Zone, EFS) would add a second service and permissions per project, and none serves RunPod pods and local runs.
- **For reading, not replay:** a line a progress bar rewrote keeps its last state, colour codes are dropped, and the values of the run's secrets (`.env`, `--secret`) are masked. Anything else a script prints is kept as printed, readable by the project's members.
- **Limit:** 50 MB compressed per run; past it the start is kept, then only the last 5,000 lines, with the number left out.

**Files.** Every file under the repository that is new or changed since the run started is copied, whether or not git or `.fieldbookignore` ignores it: ignored folders are where outputs go (`outputs/`, `lightning_logs/`, a `train.log`, TensorBoard files). Left out: `.git`, `.fieldbook`, virtual environments, and caches.

- **Elsewhere on the machine:** files a script writes outside the repository cannot be found. Every run gets `PLATFORM_OUTPUT_DIR`, a folder of its own whose whole content is kept; scripts write there what would otherwise go to `/tmp` or the home folder.
- **Nothing is left out without being named:** a file over `output_max_file_mb` (100), or past the run's total (`output_max_total_mb`, 5,000; 10,000 files), is listed with its size by `fieldbook log files` and when the run ends. Checkpoints belong in `PLATFORM_CHECKPOINT_DIR` (cloud-runs.md §8); Lightning's default ones, written under the working folder, are kept when under the limit.
- **When:** at the end of the run; on a cloud machine also every 5 minutes, since the machine may vanish. A file that changed again is copied again, whole.
- **Not for smoke runs,** and switched off with `keep_outputs = false`. A rerun's files are kept under the rerun's ID (§17).
- **Cost:** one walk of the repository at the end of a run; a repository with a million ignored files adds seconds.

**Never stops a run.** An error while keeping (storage unreachable) is reported once and the run goes on.

**Two things a bucket needs** (both found on a real machine, cloud-runs.md §12): no folder is ever created in a bucket (`storage.makedirs`: under permissions for the project's folder only, s3fs tries to create the bucket instead), and whatever watches for another machine's files lists anew each time (`storage.fresh`: s3fs remembers listings for the life of the process).

## 19. Waiting for a run, and why it failed

- **`fieldbook log wait [RUN] [--timeout MINUTES] [--json]`** returns when the run has ended, printing its status and metrics; without `RUN`, the run that is going (else the latest). Exit code 0 if it finished, 1 if it failed or was killed, 124 if it is still going when the wait gives up. It looks at the run's record every 10 seconds (one listing by prefix), so it needs no SkyPilot and works from any machine, and it can be called again after giving up, which is how a coding agent, whose commands time out after minutes, follows a run of hours. Never forever: the default is the run's time limit plus the launch, or a day. There is no `fieldbook run --wait`: a wait tied to the launching command cannot be picked up again.
- **`fieldbook log why RUN [--json]`** says how the run ended, and for a failure the cause with the lines of output that show it (`fieldbook/launch/why.py`). Named causes, in this order: a failed launch (no machine available, launch timeout, image), the time limit (cloud-runs.md §11), out of GPU memory, out of memory, a package not installed, a file not found, a Python exception (its last two frames and the error), SIGKILL (usually the system's out-of-memory killer), Ctrl-C or SIGTERM. Otherwise the exit code and the last 15 lines. A `killed` run gets the last lines it left.

## 20. Runs started by a coding agent

A run, a dataset version, and a note say when a coding agent made them, next to their author: `ada [claude-code]` (`fieldbook/launch/agents.py`). It answers "what did the agent do and spend last night", and tells an agent's conclusion from a person's.

- **The author stays the person.** The agent is a second field, not part of the name: it acts with the person's credentials and spends their money, so costs by author and access per member mean what they meant, and either can be filtered (`fieldbook log --author alice`, `--agent`, `--no-agent`).
- **How it is known:** the variable the agent sets in the commands it runs (`KNOWN`), or `FIELDBOOK_AGENT=<name>` for any other agent; `FIELDBOOK_AGENT=0` says a person is working in the agent's terminal. A run passes it on as `FIELDBOOK_AGENT`, so the cloud machine's side knows too.

  | Agent | Variable | Checked in |
  |---|---|---|
  | Claude Code | `CLAUDECODE` | Its own shell |
  | Codex | `CODEX_SESSION_ID`, `CODEX_THREAD_ID` | Its source (`exec_env.rs`, `shell_environment.rs`); older versions set neither |
  | Cursor | `CURSOR_AGENT` | Its forum, by its staff; its command-line agent lost the variable for a time and got it back |
  | Gemini CLI | `GEMINI_CLI` | Its documentation of the shell tool |

  Only Claude Code's was seen at work; the others are as their makers describe them, and they change. An agent not known by itself is told to set `FIELDBOOK_AGENT` by the instructions `fieldbook init --agents` writes.
- **Where it is recorded:** the run's start record (§18), the `agent` property of dataset versions, the MLflow tag `fieldbook.agent`, and each note.
- **Bookkeeping, not a control:** a variable can be set or unset by anything. Nothing is allowed or refused because of it; a separate budget for agents was left out for that reason (cloud-runs.md §11).

## 21. Querying runs

Which runs to look at is a query, and no set of flags is ever enough for it, so the runs are tables and the language is SQL: `fieldbook log query "SQL"` and, in Python, `fb.runs(query=None)` (`history.tables`, `history.table`). DuckDB runs the SQL, as for `fieldbook data query` (data.md §10; extra `sql`); `fb.runs()` without a query needs no extra.

| Table | One row per | Columns |
|---|---|---|
| `runs` | Run | `run_id`, `started`, `ended`, `author`, `agent`, `command`, `status` (`finished`, `failed`, `running`, `launching`, `killed`; empty for a run without a record of its own), `exit_code`, `ran` (`local`, `cloud`), `git`, `code`, `smoke`, `experiment`, `rerun_of`, `cost_usd`, `verdict`, `note` (the latest), `verified`, `params` and `metrics` (maps: `metrics['acc']`), `inputs` and `outputs` (lists) |
| `metrics`, `params` | Run and name | `run_id`, `name`, `value` |
| `data` | Run and dataset version | `run_id`, `dataset`, `version`, `role` (`read`, `saved`), `ref` |
| `notes` | Note | `run_id`, `time`, `author`, `agent`, `verdict`, `text` |

- **Maps and long tables:** the maps on `runs` make the common question one line; the long tables are for joins and aggregates ("the best accuracy per dataset version").
- **`verified`:** a rerun reproduced the run (the note `fieldbook run --rerun` leaves on it, §17): a number measured once and a number confirmed look different.
- **Output:** a plain table, or `--format markdown` (for a write-up), `json`, `csv`. Every row can carry its `run_id`, so a table pasted into a report cites its runs. This replaced a separate report command: a report is a query printed as Markdown.
- **`fb.runs()`** returns the `runs` table as a PyArrow table (`.to_pandas()` for a dataframe), `fb.runs(query=...)` the result of the same SQL as the command, for analysis in a script or notebook.
- **`fieldbook log` keeps its few flags** (`--author`, `--agent`, `--verdict`) as shortcuts; anything more is a query.
- **Cost:** the tables are built from the records for each query, reading every run's own record and notes (§14): fine for hundreds to a few thousand runs, and the place for a cache beyond that.
- **Not built:** checking a write-up's numbers against the record.

## 22. Freeing what runs leave behind

`fieldbook log rm` frees the large things a run leaves that nothing else deletes (`fieldbook/launch/retention.py`): its checkpoints (`checkpoints/<run ID>/`, cloud-runs.md §8), its kept output and files (`runs/<run ID>/`, §18), and, with `--old`, environment images (cloud-runs.md §5). `fieldbook data rm --old` is the other half: dataset files, code snapshots, package lists (data.md §9).

| Command | Frees |
|---|---|
| `fieldbook log rm RUN` | That run's checkpoints, output, and files, whatever its age |
| `fieldbook log rm --old [--days D]` | The same for every run older than `D` days (default 30), and the images not used since |

- **The run stays in the history:** its record, notes, cost, and fingerprint are small and are kept; only the bulk goes. `fieldbook log why` then has the cause on record but no output to show.
- **Kept by `--old`:** runs still going; runs noted `good` (§15), whose checkpoints are the ones worth having; runs younger than `D` days.
- **Images:** an `env-*` image is deleted when it was neither pushed nor pulled for `D` days. ECR records each image's last pull, so an image a run still uses is not old. The newest image is always kept, as are the build cache and untagged images. Members' AWS policy does not allow deleting images, so this part is the owner's; without permission the rest is done and the reason is shown.
- **Asks first:** it lists what would go and how much is freed, then asks, since nothing here can be brought back; `--yes` skips the question, `--dry-run` only lists. A coding agent is told never to run it without the user's word (the instructions of `fieldbook init --agents`).
- **Not automatic:** nothing is deleted by a run or on a schedule.

