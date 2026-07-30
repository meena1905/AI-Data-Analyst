from __future__ import annotations
import re
import duckdb
import pandas as pd
from app.utils.errors import QueryExecutionError
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|PRAGMA|EXPORT|IMPORT|CALL)\b",
    re.IGNORECASE,
)
_MAX_ROWS_RETURNED = 5000
def is_read_only_select(sql: str) -> bool:
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return False
    if _FORBIDDEN.search(stripped):
        return False
    return bool(re.match(r"^\s*(WITH|SELECT)\b", stripped, re.IGNORECASE))
def run_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame, capped at
    _MAX_ROWS_RETURNED rows to keep responses bounded."""
    if not is_read_only_select(sql):
        raise QueryExecutionError(
            "Only read-only SELECT/WITH queries are allowed. "
            "The generated SQL contained a disallowed statement."
        )
    try:
        result = conn.execute(sql).fetch_df()
    except duckdb.Error as e:
        raise QueryExecutionError(f"SQL execution failed: {e}") from e

    truncated = len(result) > _MAX_ROWS_RETURNED
    if truncated:
        result = result.head(_MAX_ROWS_RETURNED)
    return result
