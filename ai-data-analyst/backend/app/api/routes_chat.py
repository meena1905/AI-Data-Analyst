from __future__ import annotations
import logging
from fastapi import APIRouter
from app.agent.orchestrator import AnalystAgent
from app.analysis import anomaly as anomaly_mod
from app.analysis.insights import generate_insights
from app.models import AnomalyRequest, ChatRequest, ChatResponse
from app.session import session_store
from app.utils.errors import DatasetNotFoundError, SessionNotFoundError, ValidationError
logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
_agent = AnalystAgent()
def _require_session(session_id: str):
    session = session_store.get(session_id)
    if not session:
        raise SessionNotFoundError("Session not found or expired. Please upload your file(s) again.")
    return session
@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise ValidationError("Message cannot be empty.")
    session = _require_session(req.session_id)
    if not session.tables:
        raise ValidationError("Upload at least one CSV before asking questions.")
    return _agent.answer(session, req.message)
@router.post("/api/insights", response_model=ChatResponse)
async def insights(session_id: str, table_name: str | None = None) -> ChatResponse:
    session = _require_session(session_id)
    if not session.tables:
        raise ValidationError("Upload at least one CSV before requesting insights.")
    return generate_insights(session, _agent, table_name)
@router.post("/api/anomalies")
async def anomalies(req: AnomalyRequest):
    session = _require_session(req.session_id)
    table_name = req.table_name or (next(iter(session.tables), None))
    if not table_name or table_name not in session.tables:
        raise DatasetNotFoundError("No such table in this session.")
    df = session.conn.table(table_name).to_df()
    results = anomaly_mod.detect_anomalies(df, method=req.method)
    return {"table_name": table_name, "method": req.method, "count": len(results), "anomalies": results[:500]}
@router.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    session = _require_session(session_id)
    return {
        "session_id": session.id,
        "tables": [p.model_dump() for p in session.tables.values()],
        "turns": len([m for m in session.history if m["role"] == "user"]),
    }
