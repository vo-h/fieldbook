"""The settings model: every setting, its type, default, description, and validation."""

import re
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PROJECT = "default"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # dataset, project, and tag names
PROJECT_FILE = "fieldbook.toml"
LOCAL_DIR = ".fieldbook"  # a project's local state and machine-only settings (not committed)
LOCAL_FILE = "config.toml"  # in LOCAL_DIR
STORAGE_SCHEMES = ("file", "s3")  # local, AWS S3 (other clouds are deferred)


class ConfigError(ValueError):
    """A settings file or value is invalid."""


class Settings(BaseSettings):
    """Every setting, each also settable as `FIELDBOOK_<NAME>`. Settings live in the project: the
    project file (committed) and `.fieldbook/config.toml` (this machine only); see `files`."""

    model_config = SettingsConfigDict(env_prefix="FIELDBOOK_", extra="forbid", frozen=True)

    project: str = Field(DEFAULT_PROJECT, description="Project name; namespaces datasets.")
    experiment: str | None = Field(None, description="MLflow experiment (default: project).")
    storage_root: str = Field(
        default_factory=lambda: str(Path.cwd() / LOCAL_DIR / "data"),  # `files` sets it
        description="Where data lives: a path or a bucket (default: .fieldbook/data).",
    )
    storage_endpoint: str | None = Field(
        None, description="S3-compatible endpoint instead of AWS (e.g. a test server)."
    )
    storage_region: str | None = Field(None, description="S3 region.")
    state_dir: Path = Field(
        default_factory=lambda: Path.cwd() / LOCAL_DIR,  # `files` sets it
        description="Local MLflow database and caches (default: .fieldbook).",
    )
    tracking_uri: str | None = Field(
        None, description="Shared MLflow database or server; password in .env."
    )
    author: str | None = Field(
        None,
        description="Your name on runs (`init` sets your GitHub login; else git user.name).",
    )
    mlflow_port: int = Field(5050, description="First port for the MLflow UI.")
    autolog: bool = Field(
        True, description="Log parameters and metrics of training frameworks automatically."
    )
    keep_outputs: bool = Field(True, description="Keep the files a run writes (runs/ in storage).")
    output_max_file_mb: int = Field(100, gt=0, description="Largest file kept from a run, in MB.")
    output_max_total_mb: int = Field(5000, gt=0, description="Most kept from one run, in MB.")
    image_registry: str | None = Field(None, description="ECR repository for environment images.")
    image_base: str = Field("ubuntu:24.04", description="Base of environment images.")
    image_apt: list[str] = Field([], description="Extra system packages in images.")
    build_cloud: str = Field("aws", description="Where image builds run (aws/REGION).")
    remote_cloud: str | None = Field(
        None, description="Where remote runs go: aws, aws/REGION, or runpod."
    )
    launch_timeout: int = Field(15, description="Minutes before a stuck launch is torn down.")
    budget_run_usd: float | None = Field(
        None, gt=0, description="Most one cloud run may cost (project file only)."
    )
    budget_monthly_usd: float | None = Field(
        None,
        gt=0,
        description="Most the project's cloud runs may cost a month (project file only).",
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, **others):  # noqa: ANN001
        return (init_settings,)  # `layers()` merges the sources, and reports where values are from

    @field_validator("storage_root")
    @classmethod
    def _root(cls, value: str) -> str:
        return normalize_root(value)

    @field_validator("project")
    @classmethod
    def _project(cls, value: str) -> str:
        return validate_project(value)

    @field_validator("state_dir")
    @classmethod
    def _state_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()


def normalize_root(root: str) -> str:
    """Return the root as a URI, expanding `~` and turning plain paths into `file://` URIs."""
    if root.startswith("file://"):
        return "file://" + str(Path(root.removeprefix("file://")).expanduser().resolve())
    if "://" not in root:
        return "file://" + str(Path(root).expanduser().resolve())
    scheme = root.split("://", 1)[0]
    if scheme not in STORAGE_SCHEMES:
        raise ConfigError(
            f"unsupported storage root {root!r}: use a local path or an AWS S3 bucket (s3://); "
            "other clouds are not supported yet"
        )
    return root.rstrip("/")


def validate_project(name: str) -> str:
    """Project names are folder names under the storage root, so they cannot contain
    "/" or start with "." (which would allow paths outside the root)."""
    if not NAME_RE.match(name):
        raise ConfigError(
            f"invalid project name: {name!r} (letters, digits, '.', '_', '-'; "
            "must start with a letter or digit)"
        )
    return name


# How settings are grouped in `fieldbook config` and the docs. Keys stay flat in the files and
# in `FIELDBOOK_<NAME>` variables; the groups are for reading only.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Storage", ("storage_root", "storage_endpoint", "storage_region", "state_dir")),
    ("Cloud runs", ("image_registry", "image_base", "image_apt", "build_cloud", "remote_cloud",
                    "launch_timeout", "budget_run_usd", "budget_monthly_usd")),
    ("Project", ("project", "author", "keep_outputs", "output_max_file_mb",
                 "output_max_total_mb")),
    ("Experiments (MLflow)", ("experiment", "tracking_uri", "mlflow_port", "autolog")),
)  # fmt: skip
# Read from the project file only: a budget that this machine's file or a variable could
# raise would not limit whoever is spending (cloud-runs.md §11).
PROJECT_ONLY = ("budget_run_usd", "budget_monthly_usd")
ORDER = tuple(name for _, names in GROUPS for name in names)


def group_of(name: str) -> str:
    return next(group for group, names in GROUPS if name in names)
