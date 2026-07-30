from __future__ import annotations
from app.agent.orchestrator import AnalystAgent
from app.llm.prompts import INSIGHTS_PROMPT
from app.models import ChatResponse
from app.session import Session
def generate_insights(session: Session, agent: AnalystAgent, table_name: str | None = None) -> ChatResponse:
    prompt = INSIGHTS_PROMPT
    if table_name:
        prompt += f"\n\nFocus specifically on the table '{table_name}'."
    return agent.answer(session, prompt)
