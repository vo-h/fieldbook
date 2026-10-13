"""Optional dataset schemas, checked on every commit (data.md §9).

A schema is a Pandera DataFrameSchema (pandas API) stored as YAML in the index table's
schema metadata, so it is versioned with the data. Pandera and pandas are imported only
when a schema is set or checked (`pip install fieldbook[validate]`).
"""

import numbers
import warnings
from functools import lru_cache
from typing import Any

import pyarrow as pa

from fieldbook.data.dataset import DatasetError

SCHEMA_KEY = b"fieldbook.schema"
MAX_REPORTED = 20
_COLUMN_KEYS = {
    "dtype",
    "nullable",
    "unique",
    "coerce",
    "required",
    "regex",
    "description",
    "title",
    "checks",
    "name",
    "drop_invalid_rows",
}


class ValidationError(DatasetError):
    """A new version does not match the dataset's schema. `failures` lists every problem
    as {"path", "column", "check", "value"}; `path` is None for column-level problems."""

    def __init__(self, message: str, failures: list[dict[str, Any]]):
        super().__init__(message)
        self.failures = failures


def read_schema(schema: pa.Schema) -> str | None:
    """The dataset schema (YAML) stored in an index table's metadata, if any."""
    raw = (schema.metadata or {}).get(SCHEMA_KEY)
    return raw.decode() if raw else None


def to_yaml(schema: Any) -> str:
    """Normalize a schema (pandas DataFrameSchema or YAML text) to YAML for storage.

    Rejects what Pandera would silently drop: custom checks, unknown keys, and schemas
    for other dataframe libraries.
    """
    pa_ = _pandera()
    if isinstance(schema, str):
        import yaml

        try:
            raw = yaml.safe_load(schema)
        except yaml.YAMLError as e:
            raise DatasetError(f"schema is not valid YAML: {e}") from None
        _check_keys(raw)
        try:
            schema = pa_.DataFrameSchema.from_yaml(schema)
        except Exception as e:  # Pandera raises TypeError, ValueError, and its own errors
            raise DatasetError(f"invalid schema: {e}") from None
    if not isinstance(schema, pa_.DataFrameSchema):
        raise DatasetError("schema must be a pandera.pandas.DataFrameSchema or YAML text")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # dropped checks are reported below
        text = schema.to_yaml()
    if pa_.DataFrameSchema.from_yaml(text) != schema:
        raise DatasetError(
            "schema uses checks that cannot be stored (custom Python checks); "
            "use built-in checks, or run custom checks in pipeline code"
        )
    return text


def validate(table: pa.Table, schema_yaml: str, dataset_name: str) -> None:
    """Raise ValidationError if `table` does not match the schema."""
    pa_ = _pandera()
    df = table.to_pandas()
    try:
        _load(schema_yaml).validate(df, lazy=True)
    except pa_.errors.SchemaErrors as e:
        failures = []
        for case in e.failure_cases.to_dict("records"):
            index = case.get("index")
            path = None
            if _is_row(index):  # a file's path, or which row of the records checked
                path = df["path"].iloc[int(index)] if "path" in df else f"row {int(index) + 1}"
            failures.append(
                {
                    "path": path,
                    "column": case.get("column") or case.get("failure_case"),
                    "check": case["check"],
                    "value": None if path is None else case.get("failure_case"),
                }
            )
        raise ValidationError(_report(dataset_name, failures), failures) from None


def _report(dataset_name: str, failures: list[dict[str, Any]]) -> str:
    lines = [f"{len(failures)} schema check(s) failed in {dataset_name}; nothing was committed"]
    for f in failures[:MAX_REPORTED]:
        if f["path"] is None:
            lines.append(f"  {f['column']}: {f['check']}")
        else:
            lines.append(f"  {f['path']}: {f['column']} = {f['value']!r} fails {f['check']}")
    if len(failures) > MAX_REPORTED:
        lines.append(f"  ... and {len(failures) - MAX_REPORTED} more")
    return "\n".join(lines)


def _check_keys(raw: Any) -> None:
    if not isinstance(raw, dict) or not isinstance(raw.get("columns"), dict):
        raise DatasetError("schema must be a mapping with a 'columns' mapping")
    if raw.get("schema_type", "dataframe") != "dataframe" or raw.get("api", "pandas") != "pandas":
        raise DatasetError("only pandas schemas are supported (schema_type: dataframe)")
    check_names = {k for k in dir(_pandera().Check) if not k.startswith("_")}
    for column, spec in raw["columns"].items():
        unknown = set(spec or {}) - _COLUMN_KEYS - check_names
        if unknown:
            raise DatasetError(f"schema column {column!r}: unknown keys {sorted(unknown)}")


def _is_row(index: Any) -> bool:
    return isinstance(index, numbers.Real) and index == index  # NaN != NaN


@lru_cache(maxsize=32)
def _load(schema_yaml: str):
    return _pandera().DataFrameSchema.from_yaml(schema_yaml)


def _pandera():
    try:
        import pandera.pandas
    except ImportError:
        raise DatasetError(
            "dataset schemas need Pandera: pip install 'fieldbook[validate]'"
        ) from None
    return pandera.pandas
