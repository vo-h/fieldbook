"""Showing settings: `fieldbook config` and the guide's settings table."""

import os
from pathlib import Path

from fieldbook.config.files import layers, settings
from fieldbook.config.model import GROUPS, ORDER, PROJECT_ONLY, Settings


def describe(start: Path | None = None) -> list[tuple[str, str | None, str]]:
    """(name, effective value or None, source) for every setting. Sources are the variable
    (`FIELDBOOK_STORAGE_ROOT`), a file and table, or `default`. In group order (`GROUPS`)."""
    found = layers(start)
    cfg = settings(start)
    rows = []
    for name in ORDER:
        layer = next((layer for layer in found if name in layer.values), None)
        if layer is None:
            source = "default"
        elif layer is found[0]:  # the environment
            source = f"FIELDBOOK_{name.upper()}"
        else:
            source = layer.source
        value = getattr(cfg, name)
        rows.append((name, None if value is None else str(value), source))
    return rows


def settings_table() -> str:
    """The settings as Markdown tables, one per group, for docs/guide.md (a test keeps the
    guide current)."""
    parts = []
    for group, names in GROUPS:
        lines = [f"**{group}**", "", "| Setting | Variable | Description |", "|---|---|---|"]
        for name in names:
            description = Settings.model_fields[name].description
            variable = "" if name in PROJECT_ONLY else f"`FIELDBOOK_{name.upper()}`"
            lines.append(f"| `{name}` | {variable} | {description} |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def effective(start: Path | None = None) -> list[tuple[str, str | None, str]]:
    """`describe`, with the tracking store runs actually log to: `MLFLOW_TRACKING_URI` (from
    the environment or `.env`) wins over the setting, and an unset one means this machine's
    database. Addresses are shown without passwords."""
    from fieldbook import credentials
    from fieldbook.tracking import store

    rows = []
    for name, value, source in describe(start):
        if name == "tracking_uri":
            if os.environ.get("MLFLOW_TRACKING_URI"):
                value = os.environ["MLFLOW_TRACKING_URI"]
                from_env_file = "MLFLOW_TRACKING_URI" in credentials.loaded
                source = ".env (MLFLOW_TRACKING_URI)" if from_env_file else "MLFLOW_TRACKING_URI"
            elif value is None:
                value = f"sqlite:///{store.mlflow_db(settings(start))}"
                source = "default (this machine)"
            value = credentials.display_uri(value)
        rows.append((name, value, source))
    return rows
