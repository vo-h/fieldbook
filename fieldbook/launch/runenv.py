"""Everything a run's process is given as environment variables (runs.md §7):
the resolved settings, the run context, MLflow, the bucket's region, and, for cloud
machines, credentials. Local runs, `fb.init()`, and cloud machines all
take them from here, so a setting added once reaches every kind of run.
"""

import getpass
import os

from fieldbook import credentials, storage
from fieldbook.clouds import aws
from fieldbook.config import Settings
from fieldbook.launch import agents
from fieldbook.launch.context import RunContext
from fieldbook.tracking import setup

# Settings a run's own fieldbook calls use (datasets, MLflow, records); passed resolved, so
# the child does not depend on the shell's variables or the active profile.
RUN_SETTINGS = ("project", "experiment", "author", "storage_root", "storage_endpoint")


def settings_env(cfg: Settings, *, portable: bool = False) -> dict[str, str]:
    """The settings as `FIELDBOOK_*` variables, resolved, so the child uses them whatever its
    shell variables or active profile. `portable`: for another machine, so without this
    machine's folders and tracking setting (sent separately)."""
    env = {
        f"FIELDBOOK_{name.upper()}": str(v) for name in RUN_SETTINGS if (v := getattr(cfg, name))
    }
    env["FIELDBOOK_AUTHOR"] = cfg.author or getpass.getuser()  # a machine's user is root
    env["FIELDBOOK_AUTOLOG"] = "1" if cfg.autolog else "0"  # read by the run's own process
    if agent := agents.current():
        env["FIELDBOOK_AGENT"] = agent  # the coding agent that started it (runs.md §20)
    if region := storage.s3_region(cfg):
        env["FIELDBOOK_STORAGE_REGION"] = region  # detected once here, not in every child
    if not portable:
        env["FIELDBOOK_STATE_DIR"] = str(cfg.state_dir)
        if cfg.tracking_uri:
            env["FIELDBOOK_TRACKING_URI"] = cfg.tracking_uri
    return env


def run_env(cfg: Settings, run: RunContext, *, experiment: str, found: dict, python: str,
            uri: str | None = None) -> dict[str, str]:  # fmt: skip
    """The variables of a run on this machine (`fieldbook run`, `fb.init()`), on top of the
    inherited environment. `found` and `python` describe the run's interpreter."""
    env = {
        **run.to_env(),
        "PLATFORM_CHECKPOINT_DIR": storage.location(f"checkpoints/{run.run_id}", cfg),
        **setup.system_metrics_env(found),
        **settings_env(cfg),
        **setup.mlflow_env(cfg, experiment, found, python, uri),
    }
    if run.seed is not None:
        env["PYTHONHASHSEED"] = str(run.seed)
    return env | aws.default_region_env({**os.environ, **env}, storage.s3_region(cfg))


def machine_env(
    cfg: Settings, *, cloud: str, secrets: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    """(plain variables, secrets) for a cloud machine, which runs `fieldbook run` with them.
    `secrets` are the launch's own (e.g. `--secret` values, the registry login); the
    database login (`PGUSER`, plain) and password, the tracking address if it holds one,
    and bucket keys for machines without an IAM role are added. The names of all secrets
    go in `FIELDBOOK_SECRETS`, so the run masks their values in its records."""
    envs = settings_env(cfg, portable=True)
    secrets = dict(secrets)
    uri = _configured_tracking_uri()
    if uri and not uri.startswith("sqlite:"):
        if credentials.has_secret(uri):
            secrets["MLFLOW_TRACKING_URI"] = uri
        else:
            envs["FIELDBOOK_TRACKING_URI"] = uri
    if os.environ.get("PGPASSWORD"):
        secrets["PGPASSWORD"] = os.environ["PGPASSWORD"]
    if os.environ.get("PGUSER"):  # a member's own database login (sharing.md §2)
        envs["PGUSER"] = os.environ["PGUSER"]
    if cloud != "aws" and aws.roles_apply(cfg):  # no IAM role off AWS: the bucket needs keys
        secrets |= aws.credentials_env()
    envs[credentials.SECRETS_VAR] = ",".join(sorted(secrets))
    return envs, secrets


def _configured_tracking_uri() -> str | None:
    from fieldbook.tracking import store

    return store.configured_uri()
