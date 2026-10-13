"""SQL over datasets, for `fieldbook data query` (data.md §10).

Each dataset is a table in DuckDB: a records dataset's rows, read from its Lance table
as the query needs them; a files dataset's index (paths and metadata, without the files).
DuckDB is imported only here (`pip install fieldbook[sql]`).
"""

from typing import Any

from fieldbook.data import lineage, smoke
from fieldbook.data.dataset import Dataset, DatasetError
from fieldbook.data.records import RECORDS


def connect() -> Any:
    """An in-memory DuckDB connection."""
    try:
        import duckdb
    except ImportError:
        raise DatasetError("queries need DuckDB: pip install 'fieldbook[sql]'") from None
    return duckdb.connect()


def register(con: Any, ds: Dataset, view: str) -> None:
    """Make `ds` (at its version) available to SQL as `view`."""
    if ds.kind != RECORDS:
        con.register(view, ds.files())
    elif smoke.sample_size() is not None:  # a sample of the rows
        con.register(view, ds.to_table())
    else:
        index = ds.index()  # DuckDB reads the columns and rows a query asks for
        con.register(view, index)
        lineage.record_read(ds, ds.version)
