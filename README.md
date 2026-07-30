# AI Data Analyst
https://drive.google.com/file/d/133vmxO7SpH8IyNaTSO09x9TUsczCkPQ4/view?usp=drivesdk

Upload one or more CSV files and interact with your data in plain English: ask questions,
get business insights, generate charts, get SQL/pandas code, and detect anomalies — with the
model's reasoning and tool calls shown for full transparency.

Built with **FastAPI + DuckDB** on the backend (agentic tool-calling with the Anthropic Claude
API) and a **Streamlit** chat frontend.

```
"Which region generated the highest revenue?"        -> grounded answer + the SQL that produced it
"Show monthly sales trends"                            -> line chart, rendered inline
"Which products are underperforming?"                  -> ranked table + explanation
"What are the top 5 customers?"                         -> SQL join across sales_data + customers
"Detect anomalies in the dataset"                       -> flagged rows + plain-language "why"
```

---

## Features
- Upload & validate one or more CSV files (encoding/delimiter recovery, size limits, schema sanitization)
- Natural language Q&A over the data, grounded in real tool calls (no fabricated numbers)
- Business insights / executive-summary generation
- Charts: bar, line, pie, scatter, histogram, box, area — rendered via Plotly from a structured `ChartSpec`
- SQL generation (DuckDB dialect) *and* pandas code generation, chosen automatically by the agent
- Anomaly detection (z-score, IQR, Isolation Forest) with a plain-language explanation per flagged point
- Reasoning shown: every response includes the tool calls made (`tool_trace`) and the model's
  intermediate reasoning text, not just a final answer
- Multi-turn conversation context (follow-up questions like "now break that down by region")
- Multi-file analysis (SQL joins across uploaded tables — see `sample_data/`)
- Data quality checks (missing %, constant columns, duplicate rows, ID-like columns) surfaced both
  in the upload response and to the LLM's context
- Tool calling / agentic workflow (Claude's native tool-use loop, bounded and observable)
- Docker support (backend + frontend, `docker-compose up`)
- Observability: structured request-scoped logging with request IDs and latency timers
- Testing: 39 unit tests covering loader, SQL guardrails, anomaly detection, chart building,
  profiling, and the agent loop (fully mocked LLM — runs offline, no API key needed)
- Security guardrails: read-only SQL enforcement, AST-checked + subprocess-sandboxed pandas execution

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full diagram and component breakdown.
Short version:

```
Streamlit UI  →  FastAPI  →  AnalystAgent (Claude tool-calling loop)  →  DuckDB (per-session, in-memory)
                                     │
                                     ├─ run_sql          (guarded read-only SQL)
                                     ├─ run_pandas        (sandboxed, AST-checked, subprocess + timeout)
                                     ├─ build_chart        (SQL → ChartSpec JSON)
                                     ├─ detect_anomalies    (zscore / IQR / Isolation Forest)
                                     └─ get_table_profile    (schema + data-quality lookup)
```

---

## Quick start (Docker)

**Prerequisites:** Docker + Docker Compose, and an [Groq API key](https://console.groq.com/).

```bash
git clone <this-repo-url>
cd ai-data-analyst
cp .env.example .env
docker compose up --build
```

- Backend API: http://localhost:8000
- Frontend UI: http://localhost:8501

Upload `sample_data/sales_data.csv` (and optionally `sample_data/customers.csv` for the
multi-file join demo) and start asking questions.

---

## Quick start (local, no Docker)

**Prerequisites:** Python 3.11+.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
streamlit run streamlit_app.py
```

---
## Project structure

```
ai-data-analyst/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, middleware, error handlers
│   │   ├── config.py               # env-driven settings
│   │   ├── models.py                # pydantic request/response schemas
│   │   ├── session.py                # in-memory session store (DuckDB conn + history)
│   │   ├── data/
│   │   │   ├── loader.py              # CSV validation + DuckDB ingestion
│   │   │   └── profiler.py             # schema + data-quality profiling
│   │   ├── analysis/
│   │   │   ├── sql_engine.py           # guarded read-only SQL execution
│   │   │   ├── pandas_engine.py         # sandboxed pandas code execution
│   │   │   ├── anomaly.py                # zscore / IQR / Isolation Forest
│   │   │   ├── charts.py                  # ChartSpec builder
│   │   │   └── insights.py                 # business-insights convenience wrapper
│   │   ├── agent/
│   │   │   ├── orchestrator.py             # Claude tool-calling loop
│   │   │   └── tools.py                     # tool schemas + dispatch
│   │   ├── llm/
│   │   │   ├── client.py                     # SDK wrapper (retries)
│   │   │   └── prompts.py                     # system prompt templates
│   │   ├── api/
│   │   │   ├── routes_upload.py
│   │   │   └── routes_chat.py
│   │   └── utils/
│   │       ├── errors.py                       # AppError hierarchy
│   │       └── logging.py                       # structured logging + Timer
│   ├── tests/                                    # 39 unit tests (see Testing above)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py                           # chat UI
│   ├── requirements.txt
│   └── Dockerfile
├── sample_data/
│   ├── sales_data.csv
│   └── customers.csv
├── docs/
│   └── ARCHITECTURE.md                             # diagram + design rationale
├── docker-compose.yml
├── .env.example
└── README.md
```
---
## DemoLink
https://drive.google.com/file/d/133vmxO7SpH8IyNaTSO09x9TUsczCkPQ4/view?usp=drivesdk

---

```
