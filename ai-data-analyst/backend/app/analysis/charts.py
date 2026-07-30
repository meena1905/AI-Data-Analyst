from __future__ import annotations
import pandas as pd
from app.models import ChartSpec
_MAX_CHART_POINTS = 500
def build_chart_spec(
    df: pd.DataFrame,
    chart_type: str,
    title: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
) -> ChartSpec:
    if df.empty:
        raise ValueError("Cannot build a chart from an empty result set.")
    columns = list(df.columns)
    x = x if x in columns else columns[0]
    if y is None or y not in columns:
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c]) and c != x]
        y = numeric_cols[0] if numeric_cols else (columns[1] if len(columns) > 1 else columns[0])
    plot_df = df.copy()
    if len(plot_df) > _MAX_CHART_POINTS:
        plot_df = plot_df.iloc[:: max(1, len(plot_df) // _MAX_CHART_POINTS)]
    for col in plot_df.columns:
        if pd.api.types.is_datetime64_any_dtype(plot_df[col]):
            plot_df[col] = plot_df[col].dt.strftime("%Y-%m-%d")
    plot_df = plot_df.where(pd.notna(plot_df), None)
    return ChartSpec(
        title=title,
        x=x,
        y=y,
        color=color if color in columns else None,
        data=plot_df.to_dict(orient="records"),
    )
