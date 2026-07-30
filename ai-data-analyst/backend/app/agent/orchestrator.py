from __future__ import annotations
import logging
from app.agent.tools import TOOL_SCHEMAS, execute_tool
from app.config import get_settings
from app.data.profiler import profile_to_prompt_text
from app.llm.client import LLMClient, ToolCallResult, get_llm_client
from app.llm.prompts import build_system_prompt
from app.models import ChartSpec, ChatResponse, ToolTrace
from app.session import Session
from app.utils.logging import Timer
logger = logging.getLogger(__name__)
def _schema_block(session: Session) -> str:
    if not session.tables:
        return "(no tables uploaded yet)"
    return "\n\n".join(profile_to_prompt_text(p) for p in session.tables.values())
class AnalystAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or get_llm_client()
        self.settings = get_settings()
    def answer(self, session: Session, user_message: str) -> ChatResponse:
        system = build_system_prompt(_schema_block(session))
        messages = list(session.history) + [{"role": "user", "content": user_message}]
        tool_trace: list[ToolTrace] = []
        reasoning_notes: list[str] = []
        extras: dict = {}
        final_text = ""
        with Timer(logger, "agent.answer"):
            for step in range(self.settings.llm_max_agent_steps):
                turn = self.llm.complete(system=system, messages=messages, tools=TOOL_SCHEMAS)
                messages.append(turn.assistant_message)
                if turn.stop_reason != "tool_use" or not turn.tool_calls:
                    final_text = turn.text.strip() or "I don't have an answer for that."
                    break
                if turn.text:
                    reasoning_notes.append(turn.text)
                results: list[ToolCallResult] = []
                for tc in turn.tool_calls:
                    logger.info("Executing tool=%s input=%s", tc.name, tc.input)
                    text_for_llm, extra = execute_tool(session, tc.name, tc.input)
                    tool_trace.append(ToolTrace(
                        tool=tc.name,
                        input=tc.input,
                        output_summary=text_for_llm[:500],
                    ))
                    extras.update({k: v for k, v in extra.items() if v})  # last write wins, fine for single-turn
                    results.append(ToolCallResult(id=tc.id, name=tc.name, content=text_for_llm))

                messages.extend(self.llm.build_tool_result_messages(results))
            else:
                final_text = (
                    "I wasn't able to fully resolve this within the allotted analysis steps. "
                    "Here's what I found so far: " + " ".join(reasoning_notes[-1:])
                )
        session.history = messages[-20:]
        chart_spec = ChartSpec(**extras["chart"]) if extras.get("chart") else None
        return ChatResponse(
            session_id=session.id,
            answer=final_text,
            reasoning=" \n".join(reasoning_notes) if reasoning_notes else None,
            sql=extras.get("sql"),
            pandas_code=extras.get("pandas_code"),
            chart=chart_spec,
            table_preview=extras.get("table_preview"),
            anomalies=extras.get("anomalies"),
            tool_trace=tool_trace,
        )
