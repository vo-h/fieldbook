# fieldbook User Guide

fieldbook is a lightweight toolkit for machine-learning experimentation: it versions your datasets, records how every result was produced (code, environment, data, seed), and tracks experiments, with no servers to run. Everything works on your own machine with no paid services; when you need them, settings add a cloud bucket for data (§8), a shared tracking database (§6), and cloud GPUs on AWS or RunPod (§4).

## 1. Install and set up

Requirements: Python 3.11 or newer, git, and [gitleaks](https://github.com/gitleaks/gitleaks) (`brew install gitleaks`; it checks code snapshots for secrets).

```bash
pip install "fieldbook[all]"        # or pick extras from the table below
cd my-project                        # a git repository
fieldbook init
```

`fieldbook init` creates what is missing and is safe to run again:

- the project's settings: `[tool.fieldbook]` in `pyproject.toml` (or `fieldbook.toml`), with the project name (§8)
- `.fieldbookignore`: files left out of code snapshots (outputs, checkpoints)
- `.env` and `.fieldbook/` in `.gitignore`: local secrets, and this project's local data and run database

If you work with a coding agent, run `fieldbook init --agents` once: it adds a short section to the repository's `CLAUDE.md` (Claude Code) and `AGENTS.md` (Codex, Cursor, and others) telling the agent how this project records runs: read `fieldbook log` and its notes before running anything, note what each run showed, do not repeat a run that was already made, check a result with `fieldbook run --rerun`, smoke-test first, ask before a cloud run. Commit them; later `fieldbook init` runs keep that section current and leave the rest of each file alone.

Each project keeps everything it stores on this machine in its own `.fieldbook/` folder; removing a project from the machine is deleting its folder. `fieldbook data`, `fieldbook run`, and the other commands that use data or runs work inside a project (a folder with a project file, or below one).

To install exactly the versions a release was tested with, add its constraints file:
`pip install "fieldbook[all]==0.1.0" -c https://raw.githubusercontent.com/vo-h/fieldbook/v0.1.0/constraints.txt`.

| Extra | Adds |
|---|---|
| `tracking` | MLflow, with CPU/GPU metrics |
| `sql` | DuckDB, for `fieldbook data query` |
| `validate` | Pandera, for dataset schemas |
| `s3` | Data in an AWS S3 bucket |
| `aws` | Cloud runs on AWS (`s3`, plus SkyPilot and the AWS SDK) |
| `runpod` | Cloud runs on RunPod GPUs (`aws`, plus SkyPilot's RunPod support) |

## 2. Datasets

A dataset holds files with metadata (images, audio, checkpoints) or the rows of tables (Parquet, CSV, JSON lines). What you add first decides which: tables only, and their rows are imported; anything else, or a mix, and every file is kept as a file. Every change creates a new version; earlier versions stay readable. Version numbers go up but can skip one.

```bash
fieldbook data add pets ./images --set label=unknown -m "first batch"   # creates pets, version 1
fieldbook data add open-images s3://some-public-bucket/images/           # read from an S3 bucket
fieldbook data add sales 2025.parquet q1.csv          # tables: rows are imported (a records dataset)
fieldbook data ls                                     # datasets
fieldbook data ls pets -l                             # files and metadata
fieldbook data ls pets --stats --depth 1              # files and size per folder (train/, val/, ...)
fieldbook data update pets --from labels.csv          # metadata per file (CSV with a path column)
fieldbook data update pets --tag clean-v1             # name the latest version
fieldbook data ls pets --versions                     # history: who, when, inputs, command
fieldbook data ls pets@2 --diff 1                     # what changed from v1 to v2
fieldbook data cat pets train/cat/1.jpg > 1.jpg       # one file's content
fieldbook data checkout pets@clean-v1 ./pets          # write the files into a folder
fieldbook data rm pets train/cat/1.jpg                # remove a file (new version)
```

Versions are written `name@3` (version number) or `name@clean-v1` (tag).

**In Python:**

```python
import fieldbook as fb

pets = fb.dataset("pets", version="clean-v1")   # pin a version for training
rows = pets.files(prefix="train/")               # Arrow table: path, size, label, ...
with pets.open("train/cat/1.jpg") as f:         # one file
    data = f.read()
batch = pets.read_many(rows["path"].to_pylist()[:64])   # many at once: use this in training

with fb.dataset("pets-small").edit("subset") as e:   # build one dataset from another
    e.copy_from(pets, "label = 'cat'")                # the files are copied
```

**Tables** (Parquet, JSON lines, `.csv`, `.tsv`; `--delimiter '|'` for delimited text under any other extension) become rows you filter, change, and query:

```bash
fieldbook data ls sales --where "amount > 100" --limit 10
fieldbook data update sales --where "region = 'EU'" --set vat=0.2
fieldbook data rm sales --where "amount IS NULL"
fieldbook data ls sales --diff 3                      # like diff: - rows gone since v3, + rows new
fieldbook data query -d sales@3 -q "SELECT region, sum(amount) FROM sales GROUP BY 1"
fieldbook data query -d sales -q "COPY (SELECT * FROM sales) TO 'sales.parquet'"   # export
```

```python
sales = fb.dataset("sales")
table = sales.to_table("amount > 100", columns=["id", "amount"])   # Arrow; .to_pandas()
with sales.edit("june") as e:
    e.remove_records("amount IS NULL")   # operations apply in the order written
    e.update_records("region = 'EU'", vat=0.2)
    e.add_records("june.parquet")        # or a DataFrame, an Arrow table, a list of dicts
```

`fieldbook data query` also works on a files dataset: its table is the paths and metadata. Files are stored inside the dataset, not as files of their own in your storage, so a tool that needs real files gets them from `fieldbook data checkout`.

**Schemas:** `fieldbook data update pets --schema schema.yaml` (a Pandera schema) checks every later version; a version that fails is not saved.

**Adding from an S3 bucket.** `fieldbook data add NAME s3://bucket/prefix` reads the bucket's files into the dataset and records where they came from. The bytes pass through the machine running the command, so for a large bucket run it on a machine in that region. Public buckets work without extra setup; a different region may add AWS transfer charges, which the command points out.

## 3. Running code

Run scripts with `fieldbook run`, or just `python`: a script in your project that has `import fieldbook` at the top is recorded either way.

```bash
fieldbook run train.py --lr 0.01      # with options: --seed, --smoke, --on remote, ...
python train.py --lr 0.01       # the same record, with a new random seed
```

Before the script starts, `fieldbook run` snapshots your code (including uncommitted files), records the installed packages and a seed, and applies the seed to Python, NumPy, and PyTorch. Every dataset version the script saves records all of this, plus the datasets it read:

```bash
fieldbook data ls pets-small --lineage     # which data and code produced this version
fieldbook log code <run> ./restored   # restore the exact code of a run
fieldbook log packages <run>          # the packages it ran with
```

**What did I run?** `fieldbook log` lists the project's runs, newest first, and `fieldbook log diff` shows how two of them differ:

```bash
fieldbook log                        # run, started, author, command, git, status, metrics, cost, saved
fieldbook log -n 10 --author alice
fieldbook log diff 3f2a 9c41              # command, code, packages, data, parameters, metrics: only what differs
fieldbook log diff 3f2a 9c41 --patch      # with the whole code diff (otherwise cut at 400 lines)
```

Run IDs can be shortened to a unique prefix; `--json` on both is for scripts and coding agents. A run a coding agent started shows the agent next to your name (`ada [claude-code]`), as do the dataset versions and notes it made; `fieldbook log --agent` and `--no-agent` filter by it. Claude Code, Codex, Cursor, and Gemini CLI are recognised by the variable each sets; for another agent, set `FIELDBOOK_AGENT=<name>` in its shell. A run is listed once it has saved a dataset version or logged to MLflow.

**Which runs?** Anything the flags of `fieldbook log` cannot say is a query. The runs are tables (`runs`, `metrics`, `params`, `data`, `notes`) and the language is SQL (DuckDB's; `pip install "fieldbook[sql]"`):

```bash
fieldbook log query "select run_id, params['lr'] as lr, metrics['acc'] as acc, cost_usd
                from runs where status = 'finished' and verdict is distinct from 'bad'
                order by acc desc limit 5" --format markdown
fieldbook log query "select d.dataset, d.version, max(m.value) as best
                from data d join metrics m using (run_id)
                where d.role = 'read' and m.name = 'acc' group by all"
```

`--format markdown` gives a table to paste into a write-up, each row with its run ID; `json` and `csv` are for tools. `verified` says whether a rerun reproduced the run. In a script or notebook, `fb.runs()` is the same table (`fb.runs().to_pandas()`), and `fb.runs(query="select ...")` runs the same SQL.

**What did it show?** Leave the conclusion with the run, where you (or your coding agent's next session) will find it:

```bash
fieldbook log note 3f2a "lr 3e-4 diverges after epoch 2" --verdict bad    # good, bad, or inconclusive
fieldbook log note 3f2a                                                    # read a run's notes
fieldbook log --verdict bad                                           # the dead ends
```

Notes show in `fieldbook log` and `fieldbook log --json`. From inside a script: `fb.note("val loss plateaus from epoch 4", verdict="inconclusive")`.

**Did I already run this?** When a run has the same code, packages, command, and data as an earlier one, `fieldbook` says so before it starts and shows that run's result and note. On your machine it runs anyway; a cloud run is not launched unless you add `--allow-repeat`. The check costs nothing noticeable: it compares one hash, not your files. A different seed alone does not make a run new unless you pass `--seed`.

**How did it go?** `fieldbook run` keeps each run's output and the files it wrote, on your machine and on cloud machines alike:

```bash
fieldbook log why 3f2a               # how it ended; for a failure, the cause and the lines that show it
fieldbook log output 3f2a --tail 50    # what it printed (--follow while it runs)
fieldbook log files 3f2a             # the files it wrote: train.log, TensorBoard files, plots
fieldbook log files 3f2a project/outputs/preds.csv --to .
fieldbook log wait                   # block until the run that is going ends
```

- **Output:** stdout and stderr, as text: a progress bar keeps its last state, colours are dropped, and your `.env` secrets are masked. Anything else your script prints is stored where your teammates can read it.
- **Files:** everything new or changed under the repository, ignored or not. A file over 100 MB is not kept and is named (`output_max_file_mb`); save checkpoints to `PLATFORM_CHECKPOINT_DIR`. Files written elsewhere on the machine cannot be found: write those to `PLATFORM_OUTPUT_DIR`, a folder each run gets. `keep_outputs = false` switches file keeping off.
- **Scripts started with plain `python`** keep neither output nor files (nothing sits between them and the terminal); use `fieldbook run` when you want them.

**Does the result hold?** `fieldbook run --rerun` runs an earlier run again from its record: the same code (from its snapshot, whatever your working tree looks like now), command, seed, and dataset versions.

```bash
fieldbook run --rerun 3f2a                 # reproduced: 2 metrics (loss, acc), clean@4
fieldbook run --rerun 3f2a --lr 0.01       # the same run with one change
```

It compares the new metrics and saved files with the original's, prints one line, and notes it on the original run. A rerun never adds versions to your datasets (what it saves is compared, then discarded); what it logs to MLflow is kept as a new run. It uses your current environment and tells you if packages have changed since.

- **Plain `python`:** the record starts at `import fieldbook`, so keep that line at the top of your scripts. It is skipped for notebooks (call `fb.init()`), for tools such as `pytest`, and with `FIELDBOOK_RECORD=0 python scratch.py`. If the record cannot be taken (e.g. gitleaks is missing), the script still runs and says so; `fieldbook run` stops instead. A script that itself calls `fieldbook run` for its steps is started with `FIELDBOOK_RECORD=0`.
- **Large files:** snapshots over 100 MB stop the run. Add outputs and checkpoints to `.fieldbookignore` (or `.gitignore`).
- **Symlinks:** a snapshot keeps a symlink, not the file it points to. Links to files outside the repository stop cloud runs before launching (the machine does not have those files) and give a warning locally; copy such files into the repository.
- **Secrets:** put them in `.env` (`HF_TOKEN=...`); `fieldbook run` passes them to the script and never records them. Variables already set in your shell win, so `op run -- fieldbook run ...` works with a secret manager.
- **Several GPUs (DDP):** put the launcher after `fieldbook run`: `fieldbook run torchrun --standalone --nproc-per-node 4 train.py --lr 0.01`. Every process gets the run's seed. Save dataset versions and log to MLflow from rank 0 only (Lightning does this for you). Any other command works too, e.g. `fieldbook run bash prepare.sh`.
- **Options:** `--seed N` (default: a new random seed, recorded with the run), `--deterministic` (PyTorch also uses deterministic algorithms; slower, and some operations then raise an error), `--experiment NAME` (the MLflow experiment; default: the `experiment` setting, else the project name).
- **Task files:** `fieldbook run train.yaml` runs a task file in SkyPilot's format: `run` (the command), `setup` (run first), `envs`, and `workdir`. `resources` and `num_nodes` apply to cloud runs, so the same file works locally and with `--on remote`.
- **Checkpoints:** every run gets `PLATFORM_CHECKPOINT_DIR`, its own folder under the storage root (in the bucket when the storage root is one). Save checkpoints there; to continue a run, start a new one with the old checkpoint's path.
- **Smoke test:** `fieldbook run --smoke train.py` runs on a sample (100 files per top-level folder) and saves nothing to your datasets. Otherwise it behaves like a real run, so it catches the errors a real run would hit (e.g. adding files a dataset already has). Lightning stops after 5 batches; with HF Trainer, read `PLATFORM_SMOKE` to limit steps.

## 4. Running on cloud GPUs

`fieldbook run --on remote` runs a script on a rented cloud machine instead of your laptop, through [SkyPilot](https://docs.skypilot.ai). Nothing else changes: the run gets the same code snapshot, seed, lineage, and MLflow tags as a local run.

**Once:** the storage root must be a bucket (§8), runs should log to a shared tracking database (§6), and the project needs a `uv.lock` (`uv lock`). Then:

```bash
pip install "fieldbook[aws]"        # or "fieldbook[runpod]" for RunPod GPUs
sky check                                           # SkyPilot finds your cloud credentials
```

and set `image_registry` in the project's settings (`fieldbook config set image_registry ...`), e.g. an ECR repository: `image_registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com/fieldbook"` (created if missing).

```bash
fieldbook run --on remote --gpus A10G:1 train.py --lr 0.1      # deleted when the job ends
fieldbook run --on remote torchrun --nproc-per-node 4 train.py   # several GPUs; --nodes 2 for machines
fieldbook build                                                   # build the environment image ahead of time
```

- **Choosing a GPU:** name it (`--gpus A10G`, `--gpus RTX4090:2`), or describe it and take whatever is free: `--gpus 24GB+` is any GPU with at least 24 GB of memory, `48GB+:2` two of them, `nvidia:24GB+` one maker's. Several, separated by commas, are tried in your order (`--gpus RTX4090,24GB+`). `--max-price 1` leaves out machines over $1 an hour. `fieldbook run` lists the machines that fit with their prices, and the cheapest one with capacity is launched, so on RunPod, where supply changes by the hour, there is nothing to look up first:

  ```bash
  fieldbook run --on remote --cloud runpod --gpus 24GB+ --max-price 1 train.py
  ```

  With several candidates, the time limit and the "at most" cost are worked out from the dearest one. Add `--dry-run` to see the machines that fit, their prices, the time limit, the most the run can cost, and what is left of the month's budget, without launching anything (`--json` for scripts).
- **Environment:** an image with Python and the packages in `uv.lock`, built once per environment on a cloud machine and reused; code changes never rebuild it. `fieldbook run --on remote` builds it when `uv.lock` changes. Extra system packages: `image_apt = ["ffmpeg"]`.
- **Following a run:** `fieldbook log wait` returns when it ends (exit code 0 if it succeeded), `fieldbook log output RUN --follow` prints its output as it arrives, and `fieldbook log why RUN` says what went wrong if it failed. All three read the run's record in the bucket, so they work after the machine is deleted and from a teammate's machine. You can close the laptop once it has launched. `sky down <cluster>` stops a run now.
- **Credentials:** the shared database password (`PGPASSWORD`) goes along automatically; anything else only with `--secret NAME` (e.g. `--secret HF_TOKEN`). With data in S3, machines get a minimal IAM role instead of your AWS keys: once `image_registry` is set, `fieldbook init` shows its policy and creates it once, after asking (`fieldbook-run-…` and `fieldbook-build-…`; to remove them, delete these roles and instance profiles in the AWS console).
- **RunPod:** `--cloud runpod` (or `remote_cloud = "runpod"`) runs on a RunPod GPU. Images are still built on AWS and pulled from ECR. A pod cannot have an IAM role, so it gets your AWS keys as secrets to reach the bucket; temporary keys (SSO) expire, so long RunPod runs need long-lived keys.
- **Your own S3 access:** runs, on the laptop and in the cloud, get `AWS_DEFAULT_REGION` set to the bucket's region unless you set a region yourself, so boto3, s3fs, and Lightning reach the bucket.
- **Checkpoints:** save them under `PLATFORM_CHECKPOINT_DIR`, a folder in the bucket for each run, so they outlive the machine. To continue a run that failed or was cancelled, start a new run and pass the old checkpoint's path to your code (e.g. `--ckpt s3://<bucket>/<project>/checkpoints/<run ID>/last.ckpt`; Lightning: `trainer.fit(model, ckpt_path=args.ckpt)`). The command line, and so the link to the old run, is recorded with the new run.
- **Spending limits:** every cloud run has a time limit, enforced by the machine itself, so a forgotten or stuck run cannot run for weeks: 24 hours unless you say otherwise. `--max-hours N` sets it for one run. Two settings in the project file cap spending in dollars: `budget_run_usd` (one run: its time limit becomes what that buys on the machine it gets) and `budget_monthly_usd` (the project: a launch is refused, with the numbers, when the month's spending plus the most this run can cost would go over). `fieldbook run` prints the limit and the most the run can cost before it launches, and `fieldbook log costs --total` shows where the month stands.

  ```bash
  fieldbook config set budget_monthly_usd 200
  fieldbook config set budget_run_usd 20
  fieldbook run --on remote --gpus A10G:1 --max-hours 2 train.py
  ```

  Budgets are read from the committed project file only (not `.fieldbook/config.toml`, not a variable), so raising one is a change your team sees. They are estimates from list prices and stop only `fieldbook`'s own launches: for a hard limit, also set one in the cloud ([AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html); on RunPod, a prepaid balance with auto-reload off).
- **Stuck launches:** if a machine has not started the job within `launch_timeout` minutes (default 15), `fieldbook` deletes it.
- **Cost:** each cloud run's cost is recorded from the cloud's own records about 15 minutes after it ends on AWS (the machine's launch-to-termination time × its price) and about two hours after on RunPod (the amount billed for the pod), whether or not the run logged to MLflow. A run that did also shows it there as `fieldbook.cost_usd`.
- **What it cost:** `fieldbook log costs` lists what each run cost, and `fieldbook log costs --total` sums it per month and author, split into runs you started and runs a coding agent started for you, with where the month stands against the budget.

## 5. Notebooks

Call `fb.init()` at the top of a notebook (after saving it):

```python
import fieldbook as fb
run = fb.init()      # code snapshot, environment, seed, MLflow settings, .env secrets
```

## 6. Experiments

Scripts that log to MLflow need no setup: `fieldbook run` and `fb.init()` point them at the local MLflow store and tag each run with its code, data, command, and environment. Log to the run's experiment (the default); an experiment your code creates, such as Lightning's `lightning_logs`, keeps its artifacts in `./mlruns`.

```bash
fieldbook log ui          # opens MLflow in the browser; Ctrl-C stops it
```

- **Lightning, XGBoost, LightGBM, Keras:** nothing to add. Parameters and metrics are logged automatically (MLflow's autologging, switched on when your script imports the framework). Models and checkpoints are not uploaded: log the ones you want. To log with your own logger instead (e.g. Lightning's `MLFlowLogger(experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"])`), turn it off with `fieldbook config set autolog false`, or the run is logged twice. Lightning's `seed_everything()` picks up the run's seed.
- **HF Trainer:** pass `report_to="mlflow"`, and `seed=int(os.environ["PLATFORM_SEED"])` to use the run's seed.
- **Plain PyTorch loops:** call `mlflow.log_metric` yourself; there is nothing to hook.
- **Evaluation:** an evaluation is a script that reads a pinned dataset version and logs metrics; compare evaluations in the MLflow UI.
- **Run tags:** `fieldbook.run_id`, `fieldbook.author`, `fieldbook.code` (snapshot hash), `fieldbook.git` (the commit and branch the code was on, and `dirty` if it had uncommitted changes), `fieldbook.env` (environment record), `fieldbook.command`, `fieldbook.inputs` and `fieldbook.outputs` (dataset versions read and written), and `fieldbook.smoke` for smoke runs. Cloud runs add `fieldbook.cluster`, `fieldbook.cloud`, `fieldbook.region`, `fieldbook.nodes`, and `fieldbook.machine` (with `fieldbook.machine_type` on AWS), and later, as a copy of its recorded cost, the metric `fieldbook.cost_usd` (§4).

**Hyperparameter search.** Use [Ray Tune](https://docs.ray.io/en/latest/tune/) with [Optuna](https://optuna.org) choosing the values and an early-stopping scheduler (ASHA) cutting bad trials short, inside one run, typically on one multi-GPU cloud machine. Tune's [MLflow integration](https://docs.ray.io/en/latest/tune/examples/tune-mlflow.html) logs each trial as an MLflow run; filed under a parent run, a search shows as one row in the MLflow UI with its trials beneath. *This recipe has not yet been checked end to end.*

```python
# search.py, in a project with ray[tune] and optuna in its dependencies
import os

import mlflow
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import fieldbook as fb

RUN_SEED = int(os.environ["PLATFORM_SEED"])


def trial(config):
    fb.seed(RUN_SEED, key=config)  # each trial its own seed; the same trial, the same seed
    train = fb.dataset("clean", version=3)  # read pinned data as in any script
    for epoch in range(20):
        val_loss = ...  # train one epoch with config["lr"], config["batch_size"]
        tune.report({"val_loss": val_loss})  # ASHA stops trials that fall behind


with mlflow.start_run(run_name="lr-search") as search:  # the parent run
    tune.Tuner(
        tune.with_resources(trial, {"gpu": 1}),  # one GPU per trial: 4 at a time on 4 GPUs
        param_space={"lr": tune.loguniform(1e-5, 1e-2), "batch_size": tune.choice([32, 64])},
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(seed=RUN_SEED),  # the search itself repeats too
            scheduler=ASHAScheduler(),
            metric="val_loss", mode="min", num_samples=50,
        ),
        run_config=tune.RunConfig(
            storage_path=os.environ["PLATFORM_CHECKPOINT_DIR"],  # trial checkpoints in the bucket
            callbacks=[MLflowLoggerCallback(
                tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
                experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
                tags={"mlflow.parentRunId": search.info.run_id},  # trials under the parent
            )],
        ),
    ).fit()
```

```bash
fieldbook run --on remote --gpus A10G:4 search.py
```

- **What is recorded:** the search is one `fieldbook` run: one code snapshot, environment, and set of input versions. Each trial's MLflow run holds its values (as parameters) and metrics. The trial runs are created by Tune's callback in the search's own process, so they carry the same `fieldbook.*` tags (run ID, code, data) as any run.
- **Best trial:** sort the parent's trials by `val_loss` in the MLflow UI, or use `mlflow.search_runs` with `filter_string="tags.mlflow.parentRunId = '<id>'"`.
- **Scale:** one machine runs as many trials at a time as it has GPUs. Searches across several machines need a Ray cluster, which is out of scope.
- **Grids:** for a few fixed values, a loop in one run (or Hydra's `--multirun`) is enough; use `fb.seed(RUN_SEED, key=values)` per combination.

**Sharing runs between machines.** By default each machine keeps its own MLflow runs. To see runs from your laptop and from cloud GPUs in one place, log them to a shared Postgres database. A managed one needs nothing to run yourself: [Neon](https://neon.com) has a free tier, and companies can use Cloud SQL, RDS, or Azure Database.

1. Create a database and copy its connection string.
2. Put the address, without a user name or password, in the project's settings (`fieldbook config set tracking_uri ...`): `tracking_uri = "postgresql://host/dbname?sslmode=verify-full"`
3. Put your login in the project's `.env`: `PGUSER=...` and `PGPASSWORD=...` (Postgres's standard variables). On a shared project, members get theirs from `fieldbook init` (§9).
4. Run `fieldbook init`, which checks that the database is reachable. Postgres needs `pip install "fieldbook[tracking]"` (or `psycopg[binary]`) in the project.

`fieldbook run`, `fb.init()`, and `fieldbook log ui` then use the shared database; artifacts stay in the storage root. Runs already in the local database are not moved. Anyone with the password can read and change every experiment, so keep it in `.env` (never committed) and reset it if it leaks. Logging needs an internet connection. `MLFLOW_TRACKING_URI`, if set, overrides `tracking_uri`.

## 7. Models

Register models in MLflow from a training run (`mlflow.<flavor>.log_model(..., registered_model_name="cats")`) and point aliases such as `prod` at versions. Each model version links to its run, and the run to its data and code. The platform does not deploy models; for a demo or an evaluation, start an endpoint yourself:

**A quick endpoint for a demo.** Point MLflow at this machine's run database (in the project's `.fieldbook/` folder by default; run from the project's folder):

```bash
MLFLOW_TRACKING_URI=sqlite:///$PWD/.fieldbook/mlflow/mlflow.db mlflow models serve -m models:/cats@prod --env-manager local -p 5100
curl http://127.0.0.1:5100/invocations -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["width"], "data": [[620]]}}'
```

On macOS, avoid port 5000 (used by AirPlay Receiver). `--env-manager uv` builds the model's own environment from its recorded requirements instead of using yours.

**An LLM endpoint for evaluations.** Benchmark tools and LLM judges call an OpenAI-compatible API:

- Linux with NVIDIA GPUs: `vllm serve <model>` serves `http://127.0.0.1:8000/v1`.
- Mac: `llama-server -hf <user>/<model>-GGUF --port 8080` (`brew install llama.cpp`) serves `http://127.0.0.1:8080/v1`. llama.cpp runs GGUF files; convert a fine-tuned model with llama.cpp's `convert_hf_to_gguf.py`.

To serve a registered model, download its files first: `mlflow artifacts download -u models:/my-llm@prod -d ./my-llm`.

## 8. Settings

Every setting belongs to the project. They live in two files, and credentials in a third; `fieldbook config` shows every setting, its value, and where it comes from:

| File | Holds | Committed |
|---|---|---|
| `[tool.fieldbook]` in `pyproject.toml` (or `fieldbook.toml`) | The project's settings, shared by everyone who works on it: its name, bucket, run database address, image registry, clouds | Yes |
| `.fieldbook/config.toml` in the project | Changes for this machine only, e.g. a storage root on a big local disk | No |
| `.env` in the project | Credentials only (bucket keys, `PGPASSWORD`, `HF_TOKEN`); passed to runs and used by `fieldbook` itself | Never |

This machine's file wins over the project file, and a variable wins over both for one command, e.g. `FIELDBOOK_PROJECT=test fieldbook data ls`. Without a setting, local data and databases go in the project's `.fieldbook/` folder, the author is git's `user.name`, and the MLflow UI uses the first free port from 5050. Git worktrees of one repository share the main checkout's `.fieldbook/`, so they see the same runs and data.

Change settings with `fieldbook config set`, which checks the value first and keeps the file's comments:

```bash
fieldbook config set storage_root s3://lab-data            # the project file (commit it)
fieldbook config set storage_root ~/big-disk/vision --local # this machine only
fieldbook config set image_apt ffmpeg,git                   # lists as a,b,c
fieldbook config get storage_root                           # the effective value, for scripts
fieldbook config unset remote_cloud                         # back to the default
```

**Storage**

| Setting | Variable | Description |
|---|---|---|
| `storage_root` | `FIELDBOOK_STORAGE_ROOT` | Where data lives: a path or a bucket (default: .fieldbook/data). |
| `storage_endpoint` | `FIELDBOOK_STORAGE_ENDPOINT` | S3-compatible endpoint instead of AWS (e.g. a test server). |
| `storage_region` | `FIELDBOOK_STORAGE_REGION` | S3 region. |
| `state_dir` | `FIELDBOOK_STATE_DIR` | Local MLflow database and caches (default: .fieldbook). |

**Cloud runs**

| Setting | Variable | Description |
|---|---|---|
| `image_registry` | `FIELDBOOK_IMAGE_REGISTRY` | ECR repository for environment images. |
| `image_base` | `FIELDBOOK_IMAGE_BASE` | Base of environment images. |
| `image_apt` | `FIELDBOOK_IMAGE_APT` | Extra system packages in images. |
| `build_cloud` | `FIELDBOOK_BUILD_CLOUD` | Where image builds run (aws/REGION). |
| `remote_cloud` | `FIELDBOOK_REMOTE_CLOUD` | Where remote runs go: aws, aws/REGION, or runpod. |
| `launch_timeout` | `FIELDBOOK_LAUNCH_TIMEOUT` | Minutes before a stuck launch is torn down. |
| `budget_run_usd` |  | Most one cloud run may cost (project file only). |
| `budget_monthly_usd` |  | Most the project's cloud runs may cost a month (project file only). |

**Project**

| Setting | Variable | Description |
|---|---|---|
| `project` | `FIELDBOOK_PROJECT` | Project name; namespaces datasets. |
| `author` | `FIELDBOOK_AUTHOR` | Your name on runs (`init` sets your GitHub login; else git user.name). |
| `keep_outputs` | `FIELDBOOK_KEEP_OUTPUTS` | Keep the files a run writes (runs/ in storage). |
| `output_max_file_mb` | `FIELDBOOK_OUTPUT_MAX_FILE_MB` | Largest file kept from a run, in MB. |
| `output_max_total_mb` | `FIELDBOOK_OUTPUT_MAX_TOTAL_MB` | Most kept from one run, in MB. |

**Experiments (MLflow)**

| Setting | Variable | Description |
|---|---|---|
| `experiment` | `FIELDBOOK_EXPERIMENT` | MLflow experiment (default: project). |
| `tracking_uri` | `FIELDBOOK_TRACKING_URI` | Shared MLflow database or server; password in .env. |
| `mlflow_port` | `FIELDBOOK_MLFLOW_PORT` | First port for the MLflow UI. |
| `autolog` | `FIELDBOOK_AUTOLOG` | Log parameters and metrics of training frameworks automatically. |

**Data in a cloud bucket.** The storage root can be an AWS S3 bucket (other clouds are not supported yet). Datasets, stored files, code snapshots, checkpoints, and MLflow artifacts then live in the bucket, in a folder named after the project, so several projects can share one bucket; this machine's MLflow run database stays in `state_dir`. Install the S3 extra, set the storage root, and run `fieldbook init`, which checks that the bucket works:

```toml
# pyproject.toml                 # pip install "fieldbook[s3]"
[tool.fieldbook]
project = "vision"
storage_root = "s3://my-bucket"     # vision's data goes in s3://my-bucket/vision/
```

fieldbook never stores credentials; it uses AWS's usual login: `aws configure` or `aws sso login` (set `AWS_PROFILE` to pick a profile), or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`. The bucket's region is found automatically.

Reading data from a bucket on a laptop is slower than from disk, and AWS charges for data downloaded out of its cloud; heavy reads belong in cloud runs.

## 9. Working on a shared project

Who works on a project is who has access to its GitHub repository: Write (or Maintain, Admin) makes a **member**, who reads and changes data and runs and launches cloud runs; Read (or Triage) makes a **viewer**, who reads them (`fieldbook run` tells a viewer it cannot record runs). The project's settings (§8) are in the repository, so a clone is already configured; each person only needs their own credentials, which fieldbook delivers.

**Owner.** Once: set up the project's run database (a database on the team's Postgres server, `tracking_uri` in the project file, your login as `PGUSER`/`PGPASSWORD` in your `.env`), then run `fieldbook init`: it creates MLflow's tables with your login, which `fieldbook access sync` requires, and, for cloud runs, the IAM roles. Then, whenever you add or remove someone on GitHub:

```bash
gh auth login              # once: the GitHub CLI reads the repository's collaborators
fieldbook access sync --dry-run # what would change
fieldbook access sync           # database logins, AWS groups, credentials for new members
git add .fieldbook-members && git commit -m "Project access" && git push
```

`fieldbook access sync` creates a database login for each person and encrypts it with [age](https://github.com/FiloSottile/age) (`brew install age`) to the SSH keys on their GitHub profile, in `.fieldbook-members/<login>/`: only they can read it. For data in S3, it puts the IAM users named after their GitHub logins in the groups `fieldbook-<project>-write` and `-read`; with IAM Identity Center, give their identities the policy `fieldbook access aws-policy` prints (`--read` for viewers) instead. For RunPod, create a key per member in the RunPod console, named as `sync` lists, and run `fieldbook access runpod-key <login>`. A member who needs a new database password (e.g. it leaked) gets one with `fieldbook access sync --renew <login>`; a member with new SSH keys gets one on the next `sync` (replace their RunPod key in the console too, as `sync` says). When someone leaves, remove them on GitHub and run `sync`: their login, AWS access, and folder go; delete their RunPod key in the console as listed.

**Member.**

```bash
git clone git@github.com:acme/vision.git && cd vision
brew install age           # once
fieldbook init                   # writes your PGUSER, PGPASSWORD (and RUNPOD_API_KEY) to .env
```

`fieldbook init` opens your credentials with the SSH key you use for GitHub (`~/.ssh/id_ed25519` or `id_rsa`; it asks for the key's passphrase if it has one). Every member needs such a key, as a file in `~/.ssh/` and on their GitHub profile: a key that is only in a password manager (e.g. 1Password's SSH agent) or on a security key cannot be used. To make one: `ssh-keygen -t ed25519`, then `gh ssh-key add ~/.ssh/id_ed25519.pub` (or GitHub → Settings → SSH keys), and ask the owner to run `fieldbook access sync` again. Your runs and dataset versions carry your GitHub login, which `fieldbook init` records as your `author`. If the database later refuses your login, your credentials were renewed (e.g. your SSH keys changed): `git pull`, then `fieldbook init`.

## 10. Freeing space

```bash
fieldbook data rm pets@2,3              # delete versions; refused for tagged versions and those runs used
fieldbook data rm --old --dry-run       # what cleanup would delete
fieldbook data rm --old --keep 5 --days 1
```

`fieldbook data rm --old` keeps tagged versions, versions used by MLflow runs, the latest 10, and those from the last 7 days; it also frees unused code snapshots and files no version uses.

Runs leave their own bulk: checkpoints, kept output and files, and (for cloud runs) environment images.

```bash
fieldbook log rm --old --dry-run       # what would go, and how much it frees
fieldbook log rm --old                 # runs older than 30 days, and images not used since (asks first)
fieldbook log rm --old --days 7
fieldbook log rm 3f2a                  # one run's checkpoints, output, and files
```

The runs stay in `fieldbook log` with their notes and costs; only their checkpoints, output, and files go. Runs noted as `good` and runs still going are kept. Deleting images needs the project owner's AWS permissions.

## 11. Troubleshooting

| Message | Fix |
|---|---|
| `not in a fieldbook project` | `cd` into the project, or run `fieldbook init` in its folder |
| `project credentials: none for this machine's keys` | You need an SSH key that is on your GitHub profile and a file in `~/.ssh/` (`ssh-keygen -t ed25519`, `gh ssh-key add ~/.ssh/id_ed25519.pub`); then the owner runs `fieldbook access sync` (§9) |
| `age is required` | `brew install age` (or `apt install age`) |
| `gitleaks is required` | `brew install gitleaks` |
| `possible secrets found` | Remove the secret from the listed file, or add the file to `.gitignore` |
| `the code snapshot would be ... MB` | Add the listed files or folders to `.fieldbookignore` |
| `mlflow-skinny ... cannot use the local store` | `pip install "fieldbook[tracking]"` |

Design and internals: [design/README.md](../design/README.md).
