from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    n_unique: int
    n_missing: int
    pct_missing: float
    sample_values: list[Any] = Field(default_factory=list)
class TableProfile(BaseModel):
    table_name: str
    source_filename: str
    n_rows: int
    n_columns: int
    columns: list[ColumnInfo]
    quality_warnings: list[str] = Field(default_factory=list)
class UploadResponse(BaseModel):
    session_id: str
    tables: list[TableProfile]
class ChatRequest(BaseModel):
    session_id: str
    message: str
class ChartSpec(BaseModel):
    type: Literal["bar", "line", "pie", "scatter", "histogram", "box", "area"]
    title: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    data: list[dict[str, Any]]
class ToolTrace(BaseModel):
    """A single tool invocation the agent made, shown to the user for
    transparency/explainability."""
    tool: str
    input: dict[str, Any]
    output_summary: str
class ChatResponse(BaseModel):
    session_id: str
    answer: str
    reasoning: str | None = None
    sql: str | None = None
    pandas_code: str | None = None
    chart: ChartSpec | None = None
    table_preview: list[dict[str, Any]] | None = None
    anomalies: list[dict[str, Any]] | None = None
    tool_trace: list[ToolTrace] = Field(default_factory=list)
class AnomalyRequest(BaseModel):
    session_id: str
    table_name: str | None = None
    method: Literal["zscore", "iqr", "isolation_forest"] = "zscore"
class ErrorResponse(BaseModel):
    error: str
    details: dict[str, Any] = Field(default_factory=dict)
