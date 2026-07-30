from __future__ import annotations
import re
import unicodedata
import duckdb
import pandas as pd
from app.utils.errors import ValidationError
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_]")
def sanitize_table_name(filename: str, existing: set[str]) -> str:
    """Turn an arbitrary filename into a safe, unique SQL identifier."""
    base = filename.rsplit(".", 1)[0]
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
    base = _INVALID_NAME_CHARS.sub("_", base).strip("_").lower() or "table"
    if base[0].isdigit():
        base = f"t_{base}"
    name = base
    i = 1
    while name in existing:
        i += 1
        name = f"{base}_{i}"
    return name
def sanitize_column_names(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    cleaned = []
    for col in columns:
        c = str(col).strip()
        c = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode()
        c = _INVALID_NAME_CHARS.sub("_", c).strip("_").lower() or "col"
        if c[0].isdigit():
            c = f"c_{c}"
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        cleaned.append(c)
    return cleaned
def validate_csv_bytes(raw: bytes, filename: str, max_bytes: int) -> None:
    if not filename.lower().endswith(".csv"):
        raise ValidationError(f"'{filename}' is not a .csv file.")
    if len(raw) == 0:
        raise ValidationError(f"'{filename}' is empty.")
    if len(raw) > max_bytes:
        mb = len(raw) / (1024 * 1024)
        raise ValidationError(
            f"'{filename}' is {mb:.1f}MB, which exceeds the {max_bytes / (1024 * 1024):.0f}MB limit."
        )
def read_csv_safely(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, trying a couple of encodings/
    delimiters since real-world CSVs are messy."""
    import io

    last_err: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        for sep in (None, ",", ";", "\t"):
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=encoding,
                    sep=sep,
                    engine="python" if sep is None else "c",
                    on_bad_lines="warn",
                )
                if df.shape[1] == 0:
                    raise ValueError("no columns parsed")
                return df
            except Exception as e:  # noqa: BLE001 - trying multiple strategies
                last_err = e
                continue
    raise ValidationError(
        f"Could not parse '{filename}' as CSV: {last_err}"
    )
def load_csv_into_duckdb(
    conn: duckdb.DuckDBPyConnection,
    raw: bytes,
    filename: str,
    existing_table_names: set[str],
) -> tuple[str, pd.DataFrame]:
    """Validate, parse, clean, and register a CSV as a DuckDB table.

    Returns (table_name, cleaned_dataframe).
    """
    df = read_csv_safely(raw, filename)

    if df.empty:
        raise ValidationError(f"'{filename}' has no data rows.")
    if len(df.columns) != len(set(str(c) for c in df.columns)):
        pass

    df.columns = sanitize_column_names(list(df.columns))
    for col in df.columns:
        if df[col].dtype == object:
            lowered = col.lower()
            if any(k in lowered for k in ("date", "time", "created", "updated", "month", "year")):
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.7:  # only convert if it mostly parses
                    df[col] = parsed

    table_name = sanitize_table_name(filename, existing_table_names)
    conn.register(f"_{table_name}_view", df)
    conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM "_{table_name}_view"')
    conn.unregister(f"_{table_name}_view")
    return table_name, df
