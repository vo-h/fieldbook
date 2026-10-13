"""Cloud runs (cloud-runs.md): `fieldbook run --on remote`.

A run's code is its snapshot in the bucket, its environment an image built once per
environment (`fieldbook.images`), and the machine a SkyPilot cluster deleted when the job
ends. The cloud machine runs `fieldbook run` with the run ID and code hash chosen here, so
lineage, seeds, and run tags match local runs.
"""

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from fieldbook import credentials, images
from fieldbook.code import snapshot
from fieldbook.config import Settings, settings
from fieldbook.launch import (
    LaunchError,
    agents,
    budget,
    new_run_id,
    note,
    records,
    repeats,
    runenv,
    runlog,
    skypilot,
)
from fieldbook.launch.tasks import Task
from fieldbook.tracking import costs, store

TORCHRUN_NODES = (
    "--nnodes=$SKYPILOT_NUM_NODES --node-rank=$SKYPILOT_NODE_RANK "
    '--master-addr=$(echo "$SKYPILOT_NODE_IPS" | head -n1) --master-port=29500'
)


class RemoteError(LaunchError):
    pass


CLOUDS = ("aws", "runpod")  # images are built on AWS either way


@dataclass
class RemoteOptions:
    gpus: str | None = None  # "A10G:1", "24GB+", or several separated by commas (§9)
    max_price: float | None = None  # USD an hour per machine
    cloud: str | None = None
    nodes: int | None = None
    max_hours: float | None = None  # the job's time limit, if shorter than the budget's
    secrets: list[str] = field(default_factory=list)
    build: bool = True
    max_snapshot_mb: int = snapshot.MAX_SIZE_MB
    run_args: list[str] = field(default_factory=list)  # passed to `fieldbook run` on the machine
    seed: int | None = None  # the seed, if one was asked for (run_args always carries one)
    again: bool = False  # run even if an earlier run was the same


# --- runs -----------------------------------------------------------------------------


def launch(task: Task, target: list[str], options: RemoteOptions) -> str:
    """Launch a run on a cloud machine; returns the cluster name. The laptop can disconnect
    once this returns: the cluster deletes itself when the job ends."""
    cfg = settings()
    _check_ready(cfg, task.workdir)
    costing = costs.record_in_background(cfg, note)  # earlier runs, now that they ended
    snap = _snapshot(task.workdir, cfg, options.max_snapshot_mb)
    run_id = new_run_id()
    image = images.image(snap.repo, cfg, build=options.build)
    resources, nodes, cloud = _resources(task, options, cfg, image.ref)
    try:  # run from the folder this was launched from, so paths as typed resolve
        subdir = Path.cwd().resolve().relative_to(snap.repo).as_posix()
    except ValueError:
        raise RemoteError(
            f"run `fieldbook run --on remote` inside the repository {snap.repo}"
        ) from None
    cluster = f"fieldbook-{run_id}"
    mark = None
    if "--smoke" not in options.run_args:
        mark = repeats.fingerprint(
            code=snap.hash, environment=image.ref, command=target, workdir=Path.cwd(),
            repo=snap.repo, seed=options.seed, envs=task.envs,
        )  # fmt: skip
        if not options.again and (repeat := repeats.find(mark, cfg)):
            raise RemoteError(
                "this repeats an earlier run (same code, environment, command, and data), so "
                f"it was not launched: {repeat.summary}. To run it anyway: --allow-repeat"
            )
    price = _price(resources, nodes, cloud)
    limit = budget.time_limit(cfg, price, options.max_hours)
    spending = budget.Launch(run_id, cluster, cloud, cfg.author or "", time.time(), price, limit,
                             agents.current())  # fmt: skip
    budget.check_month(cfg, spending)
    command = _remote_command(run_id, snap.hash, target, options.run_args, nodes, subdir,
                              limit=limit)  # fmt: skip
    envs, secrets = _run_environment(cfg, options, image, cloud)
    git = snapshot.git_state(snap.repo)
    if git:
        envs["FIELDBOOK_GIT"] = git  # the machine has the snapshot, not the repository
    spec = {"name": cluster, "resources": resources, "num_nodes": nodes, "run": command}
    tag = image.ref.rsplit("@", 1)[0].rsplit(":", 1)[1]
    note(f"run {run_id}  code {snap.hash[:12]}  image {tag}")
    cost = f", at most ${spending.most():,.2f} (${price:,.2f} an hour)" if price else ""
    note(f"time limit {budget.span(limit)}{cost}")
    role = skypilot.machine_role(cfg, cloud)
    budget.record(spending, cfg)  # before launching, so the next launch counts this one
    if mark:
        repeats.mark(mark, run_id, cfg)
    # The run's record starts here (runs.md §18); the machine adds signs of life, its
    # output, and the end. What SkyPilot prints while launching is kept as the launch log.
    records.start(cfg, run_id, {
        "started": time.time(), "author": cfg.author or "", "agent": agents.current(),
        "code": snap.hash, "git": git or "",
        "command": credentials.mask_command(list(target), list(secrets.values())),
        "smoke": "--smoke" in options.run_args, "where": "cloud", "cloud": cloud,
        "cluster": cluster, "limit": limit,
    })  # fmt: skip
    launch_log = runlog.RunLog(run_id, cfg, name="launch", secrets=tuple(secrets.values()))
    try:
        skypilot.launch(spec, cluster, secrets=secrets, envs=envs, detach=True, cfg=cfg,
                        role=role, log=launch_log)  # fmt: skip
    except BaseException:
        budget.ended(run_id, cfg)  # torn down: it stops counting against the month
        records.end(cfg, run_id, 1, phase="launch")
        raise
    finally:
        launch_log.close()
    note(f"launched on cluster {cluster}; it is deleted when the job ends")
    note(f"wait: fieldbook log wait {run_id}   output: fieldbook log output {run_id} --follow   "
         f"stop now: sky down {cluster}")  # fmt: skip
    costing.join(timeout=60)
    return cluster


def plan(task: Task, target: list[str], options: RemoteOptions) -> dict:
    """What `launch` would do, for `fieldbook run --on remote --dry-run` (§11): the machines that
    fit and their prices, the time limit, the most the run can cost, where the month's
    budget stands, whether the image is built, whether it repeats an earlier run, and
    whether it would be launched at all. Nothing is launched, built, or recorded; the code
    snapshot is stored, as the launch would store it."""
    cfg = settings()
    _check_ready(cfg, task.workdir)
    snap = _snapshot(task.workdir, cfg, options.max_snapshot_mb)
    refused: list[str] = []
    try:
        image_ref = images.image(snap.repo, cfg, build=False).ref
    except images.ImageMissing:
        image_ref = None  # a launch builds it first
    resources, nodes, cloud = _resources(task, options, cfg, image_ref or "not/built:yet")
    fit = skypilot.candidates(resources, nodes)
    price = max(cost for _, cost in fit) if fit else None  # any of them may be launched
    if fit == [] and resources.get("accelerators"):
        refused.append(f"no machine on {cloud} has GPUs {resources['accelerators']}"
                       + (" under the price asked" if options.max_price else ""))  # fmt: skip
    repeat = None
    if image_ref and "--smoke" not in options.run_args:  # a new environment repeats nothing
        mark = repeats.fingerprint(
            code=snap.hash, environment=image_ref, command=target, workdir=Path.cwd(),
            repo=snap.repo, seed=options.seed, envs=task.envs,
        )  # fmt: skip
        repeat = repeats.find(mark, cfg)
        if repeat and not options.again:
            refused.append(f"it repeats run {repeat.run_id} (--allow-repeat runs it anyway)")
    limit = most = None
    try:
        limit = budget.time_limit(cfg, price, options.max_hours)
        spending = budget.Launch("dry-run", "", cloud, "", time.time(), price, limit)
        most = round(spending.most(), 2) if price else None
        budget.check_month(cfg, spending)
    except budget.BudgetError as e:
        refused.append(str(e))
    monthly = cfg.budget_monthly_usd
    spent = round(budget.month_spend(cfg), 2) if monthly else None
    return {
        "would_launch": not refused,
        "refused": refused,
        "command": target,
        "code": snap.hash,
        "cloud": str(resources.get("infra") or cloud),
        "nodes": nodes,
        "candidates": [{"machine": m, "usd_per_hour": round(cost, 4)} for m, cost in fit or []],
        "usd_per_hour_max": round(price, 4) if price else None,
        "time_limit_seconds": limit,
        "cost_max_usd": most,
        "budget": {
            "run_usd": cfg.budget_run_usd,
            "monthly_usd": monthly,
            "spent_usd": spent,
            "left_usd": round(monthly - spent, 2) if monthly else None,
        },
        "image": {"built": image_ref is not None, "ref": image_ref},
        "repeats": {"run_id": repeat.run_id, "summary": repeat.summary} if repeat else None,
    }  # fmt: skip


def _check_ready(cfg: Settings, workdir: Path) -> None:
    if cfg.storage_root.startswith("file://"):
        raise RemoteError(
            "cloud runs read and write data in a bucket: set storage_root to an S3 bucket, "
            "s3://... (see fieldbook config)"
        )
    uri = store.configured_uri(workdir)
    if uri is None or uri.startswith("sqlite:"):
        note("warning: no shared tracking database (tracking_uri), so MLflow runs on the "
              "cloud machine are lost when it is deleted")  # fmt: skip
    if shutil.which("uv"):
        check = subprocess.run(
            ["uv", "lock", "--check"], cwd=workdir, capture_output=True, text=True
        )
        if check.returncode != 0:
            raise RemoteError("uv.lock does not match pyproject.toml: run `uv lock` first")


def _snapshot(workdir: Path, cfg: Settings, max_size_mb: int) -> snapshot.Snapshot:
    try:
        return snapshot.take(workdir, cfg, max_size_mb=max_size_mb, refuse_outside_links=True)
    except snapshot.SnapshotError as e:
        raise RemoteError(str(e)) from None


def _resources(
    task: Task, options: RemoteOptions, cfg: Settings, image_ref: str
) -> tuple[dict, int, str]:
    """SkyPilot resources, nodes, and the run's cloud (`aws` or `runpod`). The cloud (and
    region) comes from `--cloud`, else the task file (`infra`, or the older
    `cloud`/`region`/`zone`), else the `remote_cloud` setting, else AWS."""
    resources = dict(task.resources or {})
    if options.gpus:  # SkyPilot's own syntax; several are tried in the order given
        wanted = [gpu.strip() for gpu in options.gpus.split(",") if gpu.strip()]
        resources["accelerators"] = wanted[0] if len(wanted) == 1 else wanted
    if options.max_price:
        resources["max_hourly_cost"] = options.max_price
    from_file = resources.get("infra") or resources.get("cloud")
    if options.cloud:  # the flag replaces the file's choice entirely
        for key in ("infra", "cloud", "region", "zone"):
            resources.pop(key, None)
        resources["infra"] = options.cloud
    elif not from_file:
        resources["infra"] = cfg.remote_cloud or "aws"  # never a cloud SkyPilot picks
    cloud = skypilot.cloud_name(resources.get("infra") or resources.get("cloud"))
    if cloud not in CLOUDS:
        raise RemoteError(f"cloud {cloud!r}: only {' and '.join(CLOUDS)} are supported for now")
    resources["image_id"] = f"docker:{image_ref}"
    nodes = options.nodes or task.num_nodes or 1
    return resources, nodes, cloud


def _price(resources: dict, nodes: int, cloud: str) -> float | None:
    """USD an hour, at most, for the run's machines; says which machines fit when several
    do (a GPU given by memory, or several GPUs), and stops when none does."""
    fit = skypilot.candidates(resources, nodes)
    wanted = resources.get("accelerators")
    if fit == [] and wanted:
        ceiling = resources.get("max_hourly_cost")
        under = f" under ${ceiling:,.2f} an hour" if ceiling else ""
        raise RemoteError(f"no machine on {cloud} has GPUs {wanted}{under} "
                          f"(`sky show-gpus --infra {cloud}` lists them)")  # fmt: skip
    if not fit:
        return None
    if len(fit) > 1:  # SkyPilot launches the first that has capacity
        shown = ", ".join(f"{name} ${cost:,.2f}" for name, cost in fit[:8])
        order = "tried in the order given" if isinstance(wanted, list) else "cheapest tried first"
        note(f"{len(fit)} machines fit ({order}): {shown}{', ...' if len(fit) > 8 else ''}")
    return max(cost for _, cost in fit)  # any of them may be the one launched


def _remote_command(run_id: str, code: str, target: list[str], run_args: list[str], nodes: int,
                    subdir: str = ".", *, limit: int | None = None) -> str:  # fmt: skip
    """What the cloud machine runs: check out the snapshot, install the project itself if
    it is a package, then `fieldbook run` with this run's ID and code from the same folder of the
    repository as on the laptop (design §2). With `limit` (seconds), the job is stopped at
    its time limit by the machine itself, and its end is noted for the budget (§11)."""
    words = [shlex.quote(w) for w in target]
    if nodes > 1 and Path(target[0]).name == "torchrun":
        words.insert(1, TORCHRUN_NODES)  # node rank and addresses from SkyPilot
    fieldbook_run = [
        "fieldbook",
        "run",
        "--run-id",
        run_id,
        "--code",
        code,
        *map(shlex.quote, run_args),
    ]
    run = " ".join([*fieldbook_run, *words])
    limited = []
    if limit is not None:
        limited = [f"trap 'fieldbook launch-ended {run_id} >/dev/null 2>&1 || true' EXIT"]
        hours = budget.span(limit)
        # TERM to the job's whole process group, KILL a minute later; then the cluster's
        # teardown runs as after any job
        said = f"echo 'fieldbook: stopped at the time limit of this run ({hours})' >&2"
        stopped = f"{{ code=$?; [ $code -ne 124 ] || {said}; exit $code; }}"
        run = f"timeout -k 60 {limit} {run} || {stopped}"
    return "\n".join([
        "set -eo pipefail",
        *limited,
        "mkdir -p ~/fieldbook-runs && cd ~/fieldbook-runs",
        f"[ -d {run_id} ] || fieldbook log code {code} {run_id} >/dev/null",
        f"cd {run_id}",
        "if grep -q '^\\[build-system\\]' pyproject.toml 2>/dev/null; then "
        "uv pip install -q --python /opt/venv --no-deps -e .; fi",
        'export FIELDBOOK_CODE_ROOT="$PWD"',  # the checkout's root, for the environment record
        f"cd {shlex.quote(subdir)}",
        run,
    ])  # fmt: skip


def _run_environment(
    cfg: Settings, options: RemoteOptions, image: images.Image, cloud: str = "aws"
) -> tuple[dict, dict]:
    """Settings (plain variables) and credentials (SkyPilot secrets) for the cloud machine."""
    secrets = {}
    for name in options.secrets:
        if name not in os.environ:
            raise RemoteError(f"--secret {name}: not set in the environment or .env")
        secrets[name] = os.environ[name]
    # SkyPilot takes the registry login as secrets when all three variables are secrets.
    user, password = image.login  # fetched once, with the image
    secrets |= {"SKYPILOT_DOCKER_USERNAME": user, "SKYPILOT_DOCKER_SERVER": image.repository.host,
                "SKYPILOT_DOCKER_PASSWORD": password}  # fmt: skip
    envs, secrets = runenv.machine_env(cfg, cloud=cloud, secrets=secrets)
    return {"FIELDBOOK_IMAGE": image.ref, **envs}, secrets
