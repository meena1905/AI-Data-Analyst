from __future__ import annotations
import numpy as np
import pandas as pd
from app.models import ColumnInfo, TableProfile
def _dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string"
def profile_dataframe(df: pd.DataFrame, table_name: str, filename: str) -> TableProfile:
    columns: list[ColumnInfo] = []
    warnings: list[str] = []

    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        pct_missing = round((n_missing / n_rows) * 100, 2) if n_rows else 0.0
        n_unique = int(series.nunique(dropna=True))

        sample = series.dropna().unique()[:5]
        sample_values = [
            (v.isoformat() if isinstance(v, pd.Timestamp) else
             (float(v) if isinstance(v, (np.floating,)) else
              (int(v) if isinstance(v, (np.integer,)) else v)))
            for v in sample
        ]
        columns.append(ColumnInfo(
            name=col,
            dtype=_dtype_label(series),
            n_unique=n_unique,
            n_missing=n_missing,
            pct_missing=pct_missing,
            sample_values=sample_values,
        ))
        if pct_missing > 30:
            warnings.append(f"Column '{col}' is {pct_missing}% missing.")
        if n_unique == 1 and n_rows > 1:
            warnings.append(f"Column '{col}' has a single constant value across all rows.")
        if n_unique == n_rows and n_rows > 1 and _dtype_label(series) == "string":
            warnings.append(f"Column '{col}' looks like a unique identifier (every value is distinct).")
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        warnings.append(f"{dup_rows} duplicate row(s) detected.")

    return TableProfile(
        table_name=table_name,
        source_filename=filename,
        n_rows=n_rows,
        n_columns=len(df.columns),
        columns=columns,
        quality_warnings=warnings,
    )
def profile_to_prompt_text(profile: TableProfile) -> str:
    """Compact schema description for the LLM system prompt."""
    lines = [
        f"Table \"{profile.table_name}\" (from file: {profile.source_filename}) "
        f"— {profile.n_rows} rows, {profile.n_columns} columns:"
    ]
    for c in profile.columns:
        sample = ", ".join(str(v) for v in c.sample_values[:3])
        lines.append(
            f"  - {c.name} ({c.dtype}), {c.n_missing} missing ({c.pct_missing}%), "
            f"{c.n_unique} unique values, e.g. [{sample}]"
        )
    if profile.quality_warnings:
        lines.append("  Data quality notes: " + "; ".join(profile.quality_warnings))
    return "\n".join(lines)
