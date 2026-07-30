from __future__ import annotations
import json
import logging
import pandas as pd
from app.analysis import anomaly as anomaly_mod
from app.analysis import charts as charts_mod
from app.analysis.pandas_engine import run_pandas_code
from app.analysis.sql_engine import run_sql
from app.session import Session
from app.utils.errors import AppError
logger = logging.getLogger(__name__)
MAX_ROWS_IN_LLM_RESPONSE = 30
TOOL_SCHEMAS = [
    {
        "name": "run_sql",
        "description": (
            "Run a read-only SQL SELECT query against the uploaded tables (DuckDB dialect). "
            "Use this for aggregations, filters, joins across multiple uploaded files, sorting, "
            "grouping, and top-N questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single read-only SELECT/WITH statement."},
                "purpose": {"type": "string", "description": "One sentence: what this query is meant to answer."},
            },
            "required": ["sql", "purpose"],
        },
    },
    {
        "name": "run_pandas",
        "description": (
            "Run pandas code against the uploaded tables when a transformation is easier to express "
            "procedurally than in SQL. Dataframes are pre-loaded in variables named exactly like the "
            "table names. Assign the final answer to a variable called `result` (a DataFrame, Series, or scalar)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python/pandas code. Must set a `result` variable."},
                "purpose": {"type": "string", "description": "One sentence: what this code is meant to answer."},
            },
            "required": ["code", "purpose"],
        },
    },
    {
        "name": "build_chart",
        "description": (
            "Render a chart from a SQL query result. Call run_sql first isn't required -- provide the SQL "
            "here and it will be executed then charted in one step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query producing the data to chart."},
                "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter", "histogram", "box", "area"]},
                "title": {"type": "string"},
                "x": {"type": "string", "description": "Column name for the x-axis / category."},
                "y": {"type": "string", "description": "Column name for the y-axis / value."},
                "color": {"type": "string", "description": "Optional column to color/group by."},
            },
            "required": ["sql", "chart_type", "title"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Detect anomalous rows in a table using statistical methods. Use 'zscore' or 'iqr' for "
            "single-column outliers, 'isolation_forest' for unusual combinations across multiple numeric columns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "method": {"type": "string", "enum": ["zscore", "iqr", "isolation_forest"]},
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "get_table_profile",
        "description": "Get detailed schema, sample values, and data-quality notes for a specific table.",
        "input_schema": {
            "type": "object",
            "properties": {"table_name": {"type": "string"}},
            "required": ["table_name"],
        },
    },
]
def _df_preview_for_llm(df: pd.DataFrame) -> str:
    preview = df.head(MAX_ROWS_IN_LLM_RESPONSE)
    return preview.to_csv(index=False)
def execute_tool(session: Session, tool_name: str, tool_input: dict) -> tuple[str, dict]:
    """Dispatch a tool call. Returns (text_for_llm, structured_extra)."""
    try:
        if tool_name == "run_sql":
            df = run_sql(session.conn, tool_input["sql"])
            extra = {"sql": tool_input["sql"], "table_preview": df.head(MAX_ROWS_IN_LLM_RESPONSE).to_dict("records")}
            text = f"Query returned {len(df)} row(s). Preview (CSV):\n{_df_preview_for_llm(df)}"
            return text, extra
        if tool_name == "run_pandas":
            frames = {name: session.conn.table(name).to_df() for name in session.tables}
            df = run_pandas_code(tool_input["code"], frames)
            extra = {"pandas_code": tool_input["code"], "table_preview": df.head(MAX_ROWS_IN_LLM_RESPONSE).to_dict("records")}
            text = f"Pandas code executed. Result preview (CSV):\n{_df_preview_for_llm(df)}"
            return text, extra
        if tool_name == "build_chart":
            df = run_sql(session.conn, tool_input["sql"])
            spec = charts_mod.build_chart_spec(
                df,
                chart_type=tool_input["chart_type"],
                title=tool_input.get("title", "Chart"),
                x=tool_input.get("x"),
                y=tool_input.get("y"),
                color=tool_input.get("color"),
            )
            extra = {"sql": tool_input["sql"], "chart": json.loads(spec.model_dump_json())}
            text = (
                f"Chart built: {spec.type} chart '{spec.title}' with x={spec.x}, y={spec.y}, "
                f"{len(spec.data)} data point(s)."
            )
            return text, extra
        if tool_name == "detect_anomalies":
            table_name = tool_input["table_name"]
            if table_name not in session.tables:
                return f"Error: unknown table '{table_name}'.", {}
            df = session.conn.table(table_name).to_df()
            results = anomaly_mod.detect_anomalies(df, method=tool_input.get("method", "zscore"))
            extra = {"anomalies": results[:200]}
            if not results:
                text = f"No anomalies detected in '{table_name}' using method={tool_input.get('method', 'zscore')}."
            else:
                sample = results[:10]
                text = (
                    f"Detected {len(results)} anomalous point(s) in '{table_name}'. "
                    f"Top examples:\n" + "\n".join(f"- {a['explanation']}" for a in sample)
                )
            return text, extra
        if tool_name == "get_table_profile":
            table_name = tool_input["table_name"]
            profile = session.tables.get(table_name)
            if not profile:
                return f"Error: unknown table '{table_name}'.", {}
            from app.data.profiler import profile_to_prompt_text
            return profile_to_prompt_text(profile), {}
        return f"Error: unknown tool '{tool_name}'.", {}
    except AppError as e:
        logger.warning("Tool %s failed: %s", tool_name, e.message)
        return f"Error: {e.message}", {}
    except Exception as e:  # noqa: BLE001 - surface to LLM so it can self-correct
        logger.exception("Tool %s raised an unexpected error", tool_name)
        return f"Error: unexpected failure running {tool_name}: {e}", {}
