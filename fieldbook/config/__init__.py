"""Runtime settings (projects.md §2): the model (`model`), where values come from
(`files`), how they are shown (`display`), and changing them (`edit`).

Settings live in the project, as git's do: the project file (`[tool.fieldbook]` in
`pyproject.toml`, or `fieldbook.toml`), committed and shared by everyone on the project; and
`.fieldbook/config.toml` next to it, for this machine only. `FIELDBOOK_<NAME>` variables override
both. The project's `.fieldbook/` also holds its local data, MLflow database,
and caches. Credentials are not settings: they come from each provider's usual sources and
the project's `.env` (credentials.py).
"""

from fieldbook.config.display import describe, effective, settings_table
from fieldbook.config.files import (
    Layer,
    Project,
    find_project_file,
    layers,
    project,
    require_project,
    settings,
)  # fmt: skip
from fieldbook.config.model import (
    DEFAULT_PROJECT,
    GROUPS,
    LOCAL_DIR,
    LOCAL_FILE,
    NAME_RE,
    PROJECT_FILE,
    STORAGE_SCHEMES,
    ConfigError,
    Settings,
    group_of,
    normalize_root,
    validate_project,
)  # fmt: skip

__all__ = [
    "DEFAULT_PROJECT", "GROUPS", "LOCAL_DIR", "LOCAL_FILE", "NAME_RE", "PROJECT_FILE",
    "STORAGE_SCHEMES", "ConfigError", "Layer", "Project", "Settings", "describe", "effective",
    "find_project_file", "group_of", "layers", "normalize_root", "project", "require_project",
    "settings", "settings_table", "validate_project",
]  # fmt: skip
