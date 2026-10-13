# fieldbook — Design

Last updated: 2026-10-13

fieldbook is a lightweight toolkit for machine-learning experimentation. It versions datasets, records how every result was produced (code, environment, data, seed), tracks experiments, and runs the same script on your machine or on a rented cloud GPU.

- **Low overhead.** There are no servers to deploy or operate. fieldbook is a Python package and a command-line tool; its state lives in your repository, a storage folder or bucket, and a database file. A shared project adds only a bucket and a managed database.
- **Cloud-agnostic by design.** Storage is addressed by URI and compute is launched through [SkyPilot](https://github.com/skypilot-org/skypilot), so no provider's services are built in, and research code does not change between a laptop and the cloud. Verified so far: AWS S3 for storage, AWS and RunPod for compute.
- **Agent-friendly.** Coding agents can use it as well as people can: every command has machine-readable output, what an agent runs is attributed to it, results carry notes for the next session, and a project can hold instructions that tell an agent how to work in it.

The package is `fieldbook` (CLI: `fieldbook`), imported as `import fieldbook as fb`. The user guide is [docs/guide.md](../docs/guide.md).

## 1. Who it is for, and the main idea

| Situation | What fieldbook gives |
|---|---|
| One researcher, one machine, no budget | Versioned data, runs with lineage and seeds, smoke runs, MLflow tracking and registry, notebooks; everything on the machine, no paid services |
| A researcher who rents GPUs | The same project with data in an S3 bucket, a shared tracking database, and `fieldbook run --on remote` on AWS or RunPod, with per-run cost |
| Several people on one project | The project's settings in its repository, members taken from its GitHub collaborators, per-project data, runs, and cloud resources, attribution per person |

**Settings change, research code does not.** A script that runs locally runs on a cloud GPU with `--on remote`, and in a shared project, with the same commands. What a project runs against is chosen by its settings ([projects.md](projects.md) §2), not by a mode: a `storage_root` that is a path or a bucket, a `tracking_uri` that is unset or a shared database, `fieldbook run` with or without `--on remote`.

**Everything belongs to a project:** one code repository with a name. Its settings, local state, storage folder, run database, cloud resources, and members are the project's ([projects.md](projects.md) §1).

## 2. Requirements

### 2.1 Functional

| ID | Area | Requirement | Where |
|---|---|---|---|
| D1 | Data | Version datasets; any run can be traced to the exact data it used. | data.md |
| D2 | Data | Ingestion, cleaning, and transformation as runs that save dataset versions. | data.md, runs.md |
| D3 | Data | Schema and quality validation when versions are saved. | data.md §9 |
| D5 | Data | Tabular and unstructured data (text, PDF documents, images, video) across versioning, validation, and exploration. | data.md |
| D6 | Data | Storage retention: dataset versions, code snapshots, and environment records no longer used are deleted by user-set rules. | data.md §7 |
| T1 | Training | Interactive work in Jupyter and VS Code, with runs recorded (`fb.init()`). | runs.md §10 |
| T2 | Training | Batch training from scripts, including several GPUs and machines. | runs.md §2, cloud-runs.md §9 |
| T3 | Training | Experiment tracking: parameters, metrics, artifacts, system metrics. | tracking.md |
| T4 | Training | Model registry with versions, aliases, and lineage to runs. | tracking.md §6 |
| T5 | Training | Automatic code snapshot per run, including uncommitted and untracked files. | runs.md §4 |
| T6 | Training | Smoke runs on a small data sample before a full or cloud run. | runs.md §9 |
| T7 | Training | Evaluations as tracked runs on pinned dataset versions, compared in MLflow. | runs.md §12 |
| T8 | Training | Profiling: training bottlenecks visible next to the run's system metrics. | runs.md §12 |
| M1 | Monitoring | Live training metrics and resource utilization. | tracking.md §4 |
| C1 | Compute | One job definition runs locally and on cloud GPUs (AWS and RunPod). | cloud-runs.md |
| S1 | Sharing | Several researchers share a project's data, runs, models, and compute, with access per person. | sharing.md |

### 2.2 Non-functional

| ID | Requirement |
|---|---|
| N2 | Reproducibility: code, environment, data, config, and seed are recorded for every run without manual steps. |
| N3 | Works on one machine (e.g. a 16 GB laptop), starts with one command, and needs no paid services. Cloud use adds only pay-as-you-go costs: GPU time, bucket and registry storage, short-lived build machines; a managed tracking database can stay within a free tier. |
| N6 | Cost control: cloud machines are deleted when done; per-run cost is visible. |
| N7 | Storage is addressed by URI (`file://` or `s3://`); the same code and data formats work with either. |
| N8 | Secrets are injected at runtime and never written to code, notebooks, snapshots, or logs. |
| N9 | All software is free and open source, with no dependency on paid tiers. Infrastructure (storage, GPUs, managed databases) may be paid if accessed through standard APIs and replaceable. |
| N10 | Every external component is widely adopted and actively maintained. |

## 3. How it fits together

Each part uses one tool or in-house module, and each part's code is one package or module, so a concern lives in one place.

| Part | Built on | Code | Design |
|---|---|---|---|
| Projects, settings, credentials | pydantic-settings, tomlkit, python-dotenv | `fieldbook/config/`, `credentials.py`, `setup_project.py` | [projects.md](projects.md) |
| Storage | fsspec; local disk or AWS S3 | `fieldbook/storage.py` | [storage.md](storage.md) |
| Datasets | In-house, on Lance: each dataset is one Lance table holding its files or rows and every version; DuckDB for queries; Pandera for schemas | `fieldbook/data/` | [data.md](data.md) |
| Runs | In-house launcher; git and gitleaks for code snapshots; uv for environments | `fieldbook/launch/`, `fieldbook/code/` | [runs.md](runs.md) |
| Experiment tracking and registry | MLflow (SQLite on the machine, or a shared Postgres database; no MLflow server) | `fieldbook/tracking/` | [tracking.md](tracking.md) |
| Cloud runs | SkyPilot (a cluster per run); Docker images in AWS ECR; AWS and RunPod | `fieldbook/launch/cloud.py`, `fieldbook/images/`, `fieldbook/clouds/` | [cloud-runs.md](cloud-runs.md) |
| Shared projects | GitHub (who is a member), Postgres logins, AWS IAM groups, age (delivering credentials) | `fieldbook/members/` | [sharing.md](sharing.md) |
| Commands | Typer | `fieldbook/cli/`, `ui.py` (web UIs in the foreground) | each document |

`import fieldbook` loads everything lazily (about 0.3 s); the run's own process (`launch/context`, `bootstrap`, seeding) uses the standard library only.

**A run, end to end.** `fieldbook run train.py` (or plain `python train.py`, which is recorded the same way: runs.md §13) finds the project and its settings ([projects.md](projects.md)), snapshots the code and records the environment and seed ([runs.md](runs.md)), points MLflow at the project's store ([tracking.md](tracking.md)), and starts the script in the project's environment. The script reads dataset versions and saves new ones, each recording the run that made it ([data.md](data.md)), all in the project's folder of the storage root ([storage.md](storage.md)). With `--on remote` the same happens on a cloud machine built from the project's `uv.lock` ([cloud-runs.md](cloud-runs.md)). On a shared project every member does the same against the same bucket and database, each under their own name ([sharing.md](sharing.md)).

## 4. Documents

| Document | Covers |
|---|---|
| [projects.md](projects.md) | What a project is; settings and where they come from; the project's `.fieldbook/` folder; credentials; `fieldbook init`; `fieldbook config` |
| [storage.md](storage.md) | The storage root and project folders; local disk and S3; the bucket check |
| [data.md](data.md) | Datasets: files and records as rows of one Lance table, versions and tags, the API and commands, saving and concurrent saves, cleanup, schemas, reading with other tools, lineage, and why the files are in the table |
| [runs.md](runs.md) | `fieldbook run` and `fb.init()`: code snapshots, the environment record, seeds, smoke runs, framework integration, recipes, recording without `fieldbook run`, `fieldbook log` and `fieldbook diff`, notes, repeated runs, `fieldbook run --rerun`, what is kept of a run (record, output, files), `fieldbook log wait` and `why`, runs started by a coding agent, querying runs with SQL, freeing what runs leave behind |
| [tracking.md](tracking.md) | MLflow: where runs log, run tags, the UI, the model registry, costs per person, autologging |
| [cloud-runs.md](cloud-runs.md) | `fieldbook run --on remote`: images, credentials and machine roles, checkpoints, cost, the checks done on real clouds |
| [sharing.md](sharing.md) | Members from GitHub (`fieldbook access sync`), each project's resources, the security model, central services for later |
| [testing.md](testing.md) | How the tests are laid out and isolated; CI |

## 5. Out of scope

| Item | Reason |
|---|---|
| Deploying models (serving, serving monitoring) | Researchers hand models off through the MLflow registry. Demo and evaluation endpoints are recipes in the guide. |
| Feature store | Point-in-time joins are a DuckDB query (data.md §10) |
| Labeling and annotation tools (Label Studio, CVAT) | Separate tools; labels are dataset columns |
| Orchestration: scheduled or data-triggered pipelines | Built with Dagster and removed. Research on one machine is started by a person, step by step; a background scheduler per project did not fit projects that live in their own folders and are shared (every member's machine would react to the same dataset change). A sequence of steps is a script of `fieldbook run` commands |
| Remote workspaces | Deferred. Idea for later: a local notebook whose kernel runs on a cloud GPU |
| Clouds other than AWS and RunPod (GCS, Azure Blob, Cloudflare R2 storage; GCP, Azure machines; other registries) | Deferred, so nothing is claimed that was never run (storage.md §8, cloud-runs.md §3) |
| Central services: an MLflow server with per-user permissions, a shared SkyPilot API server, single sign-on, a secrets manager | Later. Projects are shared without new servers; how the services would map onto projects is in sharing.md §6 |
| Spot instances, automatic recovery, managed jobs, job queues; keeping machines between runs; resuming a run under its old ID | Each run gets a fresh on-demand machine, deleted when it ends; a failed run is continued by a new run from its checkpoints (cloud-runs.md §8) |
| Distributed data processing engines (Ray Data, Daft) | Most datasets fit one machine or shard across independent runs |
| Read-through cache for bucket data | An optimisation; revisit if laptop reads are a problem |
| Dataset branches and merges; referencing external datasets in place; reading another project's datasets | Parallel work uses separate datasets or tags; external datasets are copied in; projects are kept apart (sharing.md §4) |
| Kubernetes, Helm charts, infrastructure modules | Too much to operate for the audience |
| A plugin system for backends | Backends are chosen by settings |
| Non-Python training; GPU quotas; caching Hugging Face models on cloud machines; publication bundles; license checks; LLM tracing beyond MLflow Tracing | Not needed for the audience |
| Migrating runs, models, or datasets between setups | Existing state stays where it was created |

## 6. Alternatives considered

| Option | Decision | Reason |
|---|---|---|
| **ClearML** (instead of MLflow) | Rejected | RBAC, SSO, autoscalers, and LLM serving require paid tiers. The open-source server has no per-user isolation. |
| **dstack** (instead of SkyPilot) | Rejected | Stronger RunPod integration, but fails N10: ~2.3k GitHub stars and ~13k monthly PyPI downloads versus ~10.7k and ~1.1M for SkyPilot (Sept 2026). |
| **lakeFS** (data versioning) | Rejected | Only its dataset-snapshot capability was needed. Adds a server, and stores objects under generated names, so versioned data cannot be read as plain paths. Its staging model is adopted for edits. |
| **DVC** (data versioning) | Rejected | Couples data versions to git commits and adds a separate CLI workflow, and stores one object per file, which is slow and costly for many small files in a bucket (data.md §14). |
| **Feast** (feature store) | Rejected | Not needed; point-in-time joins are a DuckDB query. |
| **Dagster** (pipelines, dataset triggers, a daemon as a login service) | Built, then removed | See §5. Its code is in the git history before 2026-10-09. |
| **Prefect / Flyte** | Rejected | Prefect has weaker data lineage; Flyte requires Kubernetes. |
| **Ray Data** (default processing engine) | Rejected | Adds a cluster runtime to operate and debug. |
| **MinIO** (local object store) | Rejected | Community edition in maintenance mode since Dec 2025; repository archived in 2026. |
| **Local S3 server** (SeaweedFS, Garage) | Rejected | Not needed; storage is addressed by URI (N7). |
| **An MLflow tracking server** | Rejected for now | An always-on server to run and pay for; a managed database needs neither. Returns with central services, for per-user permissions (sharing.md §6). |
| **A per-user settings file with profiles** (`~/.fieldbook/config.toml`) | Built, then removed | Shared locations are the same for every member, so they belong in the repository; the rare local path goes in the project's local file (projects.md §2). |
| **Infisical / OpenBao** (secrets manager) | Not used | Members' credentials are delivered encrypted to their GitHub SSH keys with age; a secrets manager is a server to run. Infisical's free edition lacks SSO and role-based access. |
| **Managed jobs and spot instances** | Deferred | A controller VM and recovery logic; a fresh on-demand cluster per run, continued from bucket checkpoints, suffices. |
| **Image per run** (instead of per environment) | Rejected | Code changes every run, so images would never be reused; an image per `uv.lock` is rebuilt rarely. |
| **MLflow built-in code tracking** | Insufficient | Run tags record only the commit hash; `mlflow.genai.enable_git_model_versioning()` records a diff of tracked files only. |
| **Git refs for code snapshots** | Rejected | Require a git remote, which a solo researcher may not have. |
| **Kubeflow** | Rejected | Kubernetes-first; high operational cost relative to team size. |

## 7. Risks

- **In-house data versioning:** no external community; bugs and maintenance fall on this project. Deletion safety has dedicated tests (data.md §13).
- **Framework settings:** frameworks may rename the variables the launcher sets or change defaults; the tested versions cover them, at an ongoing cost (runs.md §11).
- **macOS:** Apple GPUs offer MPS only; CUDA-only code runs with `--on remote`.
- **Laptop reads and egress:** reading a bucket from a laptop is slow, and AWS charges for data leaving its cloud; heavy reads belong in cloud runs (storage.md §7).
- **Cloud runs:** image size, RunPod's API changes, machine roles after a SkyPilot upgrade (cloud-runs.md §13).
- **Shared projects:** trust within a project; account-wide launch rights (sharing.md §7).

## 8. Limits

- Millions of files per dataset (three million measured), added a few million at a time: one add holds its staged rows in memory, about 2.3 GB per million (data.md §12).
- AWS and RunPod only; S3 only (§5).
- No enforced permissions within a project (sharing.md §5).

## 9. Open items

| Item | What is needed |
|---|---|
| The members' AWS policy has not been used by a real member identity | A real IAM user (or role) in a project's `fieldbook-<project>-write` group launching a cloud run. Temporary federation credentials cannot test it: AWS refuses them every IAM call. Part of the next item |
| A shared project has not been used by two real people end to end | One project, an owner and a member: `fieldbook access sync`, the member's `fieldbook init`, a run each, a cloud run by the member |
| One add of several million files needs the memory | 6.9 GB for three million (data.md §12): the staged rows are held as Python objects. Larger imports go in several edits; staging them more compactly is the likely change |
| Large files from a bucket or another dataset | A local file over 64 MB is streamed into the table; one added from an `s3://` source, or copied from another dataset, is still read whole into memory |
