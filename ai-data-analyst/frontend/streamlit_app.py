"""Streamlit frontend for the AI Data Analyst — single-service edition.

This version runs entirely inside one Streamlit process: instead of calling
a separate FastAPI backend over HTTP, it imports the backend's app package
directly and calls it in-process. This lets the whole thing run on
Streamlit Community Cloud (which only hosts one Python process) with
nothing else to deploy.

If you *do* have a separately-deployed FastAPI backend (e.g. on Render) and
want the two-service architecture instead, use the HTTP-based version of
this file (see README) and set BACKEND_URL.

Each browser tab gets its own `Session` object (DuckDB connection + chat
history) stored in `st.session_state` — no shared global state between users.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Make the backend's `app` package importable -----------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_THIS_DIR, "..", "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# --- Wire up secrets as env vars *before* importing app.config ----------
# app.config.get_settings() is cached on first call, so env vars must be
# set before any `app.*` module is imported. st.secrets raises if no
# secrets.toml exists at all (e.g. running locally with plain env vars),
# so this is best-effort and falls back to whatever's already in os.environ.
try:
    for key in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "LLM_MODEL", "GROQ_API_KEY", "GROQ_MODEL"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # no secrets.toml configured; rely on real environment variables instead

from app.agent.orchestrator import AnalystAgent          # noqa: E402
from app.analysis import anomaly as anomaly_mod           # noqa: E402
from app.analysis.insights import generate_insights        # noqa: E402
from app.data.loader import load_csv_into_duckdb, validate_csv_bytes  # noqa: E402
from app.data.profiler import profile_dataframe             # noqa: E402
from app.session import Session                               # noqa: E402
from app.utils.errors import AppError                          # noqa: E402
import duckdb                                                    # noqa: E402
import uuid                                                        # noqa: E402


st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- state ----
if "session" not in st.session_state:
    st.session_state.session = None  # type: Session | None
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_agent() -> AnalystAgent:
    # One agent (and one LLM client) shared across reruns of this process.
    return AnalystAgent()


def new_session() -> Session:
    conn = duckdb.connect(database=":memory:")
    return Session(id=uuid.uuid4().hex, conn=conn)


# --------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("Upload CSVs, then ask questions in plain English.")

    uploaded = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
    if uploaded and st.button("Load data", type="primary", use_container_width=True):
        if st.session_state.session is None:
            st.session_state.session = new_session()
        session = st.session_state.session
        try:
            with st.spinner("Validating and profiling data..."):
                for f in uploaded:
                    raw = f.getvalue()
                    validate_csv_bytes(raw, f.name, max_bytes=50 * 1024 * 1024)
                    table_name, df = load_csv_into_duckdb(session.conn, raw, f.name, set(session.tables.keys()))
                    session.tables[table_name] = profile_dataframe(df, table_name, f.name)
            st.success(f"Loaded {len(uploaded)} file(s).")
        except AppError as e:
            st.error(e.message)
        except Exception as e:  # noqa: BLE001
            st.error(f"Upload failed: {e}")

    session = st.session_state.session
    if session and session.tables:
        st.divider()
        st.subheader("Loaded tables")
        for t in session.tables.values():
            with st.expander(f"{t.table_name} ({t.n_rows} rows × {t.n_columns} cols)"):
                st.caption(f"from {t.source_filename}")
                cols_df = pd.DataFrame([c.model_dump() for c in t.columns])
                st.dataframe(cols_df[["name", "dtype", "n_missing", "pct_missing", "n_unique"]], hide_index=True, use_container_width=True)
                for w in t.quality_warnings:
                    st.warning(w, icon="⚠️")

        st.divider()
        if st.button("Generate business insights", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Generate business insights and a summary of this dataset."})
            st.session_state._pending = True
            st.rerun()

        if st.button("Detect anomalies", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Detect anomalies in the dataset and explain why each was flagged."})
            st.session_state._pending = True
            st.rerun()

        st.divider()
        if st.button("🗑️ Reset session", use_container_width=True):
            st.session_state.session = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Example questions:")
    for ex in [
        "Which region generated the highest revenue?",
        "Show monthly sales trends.",
        "Which products are underperforming?",
        "What are the top five customers?",
        "Generate SQL for total revenue by category.",
        "Detect anomalies in the dataset.",
    ]:
        st.code(ex, language=None)


# ----------------------------------------------------------------- main ----
st.header("Chat with your data")

if not st.session_state.session or not st.session_state.session.tables:
    st.info("👈 Upload one or more CSV files to get started.")
    st.stop()


def render_chart(spec: dict):
    df = pd.DataFrame(spec["data"])
    if df.empty:
        return
    kind = spec["type"]
    x, y, color, title = spec.get("x"), spec.get("y"), spec.get("color"), spec.get("title")
    try:
        if kind == "bar":
            fig = px.bar(df, x=x, y=y, color=color, title=title)
        elif kind == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title, markers=True)
        elif kind == "area":
            fig = px.area(df, x=x, y=y, color=color, title=title)
        elif kind == "pie":
            fig = px.pie(df, names=x, values=y, title=title)
        elif kind == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, title=title)
        elif kind == "histogram":
            fig = px.histogram(df, x=x, color=color, title=title)
        elif kind == "box":
            fig = px.box(df, x=x, y=y, color=color, title=title)
        else:
            st.dataframe(df)
            return
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Couldn't render chart ({e}); showing raw data instead.")
        st.dataframe(df, use_container_width=True)


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        extras = msg.get("extras") or {}

        if extras.get("reasoning"):
            with st.expander("🧠 Reasoning"):
                st.markdown(extras["reasoning"])

        if extras.get("tool_trace"):
            with st.expander(f"🔧 Tool calls ({len(extras['tool_trace'])})"):
                for t in extras["tool_trace"]:
                    st.markdown(f"**{t['tool']}**")
                    st.json(t["input"], expanded=False)
                    st.caption(t["output_summary"])

        if extras.get("sql"):
            with st.expander("🗄️ Generated SQL", expanded=False):
                st.code(extras["sql"], language="sql")

        if extras.get("pandas_code"):
            with st.expander("🐍 Generated pandas code", expanded=False):
                st.code(extras["pandas_code"], language="python")

        if extras.get("table_preview"):
            with st.expander("📋 Data preview", expanded=False):
                st.dataframe(pd.DataFrame(extras["table_preview"]), use_container_width=True)

        if extras.get("chart"):
            render_chart(extras["chart"])

        if extras.get("anomalies"):
            with st.expander(f"🚨 Anomalies ({len(extras['anomalies'])})", expanded=True):
                st.dataframe(pd.DataFrame(extras["anomalies"]), use_container_width=True)


for msg in st.session_state.messages:
    render_message(msg)

prompt = st.chat_input("Ask a question about your data...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state._pending = True
    st.rerun()

if st.session_state.get("_pending"):
    last_user_msg = next(m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user")
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                agent = get_agent()
                result = agent.answer(st.session_state.session, last_user_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "extras": result.model_dump(),
                })
            except AppError as e:
                st.error(e.message)
            except Exception as e:  # noqa: BLE001
                st.error(f"Something went wrong: {e}")
    st.session_state._pending = False
    st.rerun()