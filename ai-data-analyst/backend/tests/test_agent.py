from app.agent.orchestrator import AnalystAgent
from app.data.loader import load_csv_into_duckdb
from app.data.profiler import profile_dataframe
from app.llm.client import LLMTurn, ToolCallRequest
from app.session import session_store
class FakeLLMClient:
    """Scripted sequence of LLMTurn responses, one per call to `complete`."""
    def __init__(self, script: list[LLMTurn]):
        self.script = script
        self.calls = 0
    def complete(self, system, messages, tools=None):
        turn = self.script[self.calls]
        self.calls += 1
        return turn
    def build_tool_result_messages(self, results):
        return [{"role": "tool_results", "content": [r.content for r in results]}]
def _make_session_with_data():
    session = session_store.create()
    raw = b"region,revenue\nEast,100\nWest,200\nEast,150\nNorth,300\n"
    table_name, df = load_csv_into_duckdb(session.conn, raw, "orders.csv", set())
    session.tables[table_name] = profile_dataframe(df, table_name, "orders.csv")
    return session
def test_agent_answers_directly_without_tools():
    session = _make_session_with_data()
    fake = FakeLLMClient([
        LLMTurn(text="Hello! Ask me about your data.", tool_calls=[], stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "Hello! Ask me about your data."}),
    ])
    agent = AnalystAgent(llm_client=fake)
    result = agent.answer(session, "hi")
    assert "Hello" in result.answer
    assert result.tool_trace == []
def test_agent_executes_sql_tool_then_answers():
    session = _make_session_with_data()
    fake = FakeLLMClient([
        LLMTurn(
            text="Let me check revenue by region.",
            tool_calls=[ToolCallRequest(
                id="tu_1", name="run_sql",
                input={"sql": "SELECT region, SUM(revenue) AS total FROM orders GROUP BY region ORDER BY total DESC",
                       "purpose": "find top region"},
            )],
            stop_reason="tool_use",
            assistant_message={"role": "assistant", "content": "Let me check revenue by region."},
        ),
        LLMTurn(
            text="North generated the highest revenue at 300.",
            tool_calls=[], stop_reason="end_turn",
            assistant_message={"role": "assistant", "content": "North generated the highest revenue at 300."},
        ),
    ])
    agent = AnalystAgent(llm_client=fake)
    result = agent.answer(session, "Which region generated the highest revenue?")
    assert "North" in result.answer
    assert result.sql is not None
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0].tool == "run_sql"
def test_agent_stops_after_max_steps(monkeypatch):
    session = _make_session_with_data()
    tool_call_turn = LLMTurn(
        text="",
        tool_calls=[ToolCallRequest(id="tu_x", name="run_sql",
                                     input={"sql": "SELECT COUNT(*) AS n FROM orders", "purpose": "count"})],
        stop_reason="tool_use",
        assistant_message={"role": "assistant", "content": ""},
    )
    fake = FakeLLMClient([tool_call_turn] * 10) 
    agent = AnalystAgent(llm_client=fake)
    monkeypatch.setattr(agent.settings, "llm_max_agent_steps", 3)
    result = agent.answer(session, "loop forever")
    assert result.answer  
    assert fake.calls == 3
def test_agent_maintains_conversation_history():
    session = _make_session_with_data()
    fake = FakeLLMClient([
        LLMTurn(text="Sure, noted.", tool_calls=[], stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "Sure, noted."}),
    ])
    agent = AnalystAgent(llm_client=fake)
    agent.answer(session, "remember I care about the North region")
    assert len(session.history) >= 2
    assert session.history[0]["role"] == "user"
