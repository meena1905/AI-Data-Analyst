from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from app.config import get_settings
from app.utils.errors import LLMError
logger = logging.getLogger(__name__)
@dataclass
class ToolCallRequest:
    id: str
    name: str
    input: dict[str, Any]
@dataclass
class ToolCallResult:
    id: str
    name: str
    content: str  
@dataclass
class LLMTurn:
    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "tool_use" | "end_turn"
    assistant_message: dict = field(default_factory=dict)  # opaque, provider-specific; appended to history as-is
class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn: ...
    def build_tool_result_messages(self, results: list[ToolCallResult]) -> list[dict]: ...
def _retry_sleep(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 8))
class AnthropicLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        import anthropic
        settings = get_settings()
        self.model = model or settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)
        self._anthropic = anthropic
    def complete(self, system: str, messages: list[dict], tools: list[dict] | None = None, max_retries: int = 2) -> LLMTurn:
        anthropic = self._anthropic
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                kwargs: dict = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages)
                if tools:
                    kwargs["tools"] = tools
                response = self._client.messages.create(**kwargs)
                break
            except anthropic.RateLimitError as e:
                last_err = e
                _retry_sleep(attempt)
            except anthropic.APIStatusError as e:
                last_err = e
                if e.status_code and e.status_code >= 500:
                    _retry_sleep(attempt)
                    continue
                raise LLMError(f"LLM request failed: {e.message}") from e
            except anthropic.APIConnectionError as e:
                last_err = e
                _retry_sleep(attempt)
        else:
            raise LLMError(f"LLM request failed after retries: {last_err}") from last_err

        assistant_content = [block.model_dump() for block in response.content]
        text = "\n".join(b["text"] for b in assistant_content if b["type"] == "text" and b.get("text")).strip()
        tool_calls = [
            ToolCallRequest(id=b["id"], name=b["name"], input=b["input"])
            for b in assistant_content if b["type"] == "tool_use"
        ]
        stop_reason = "tool_use" if (response.stop_reason == "tool_use" and tool_calls) else "end_turn"
        return LLMTurn(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            assistant_message={"role": "assistant", "content": assistant_content},
        )

    def build_tool_result_messages(self, results: list[ToolCallResult]) -> list[dict]:
        return [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.id, "content": r.content} for r in results
            ],
        }]
def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]
class GroqLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from groq import Groq

        settings = get_settings()
        self.model = model or settings.groq_model
        self.max_tokens = settings.llm_max_tokens
        self._client = Groq(api_key=api_key or settings.groq_api_key)

    def complete(self, system: str, messages: list[dict], tools: list[dict] | None = None, max_retries: int = 2) -> LLMTurn:
        import groq

        full_messages = [{"role": "system", "content": system}] + messages
        kwargs: dict = dict(model=self.model, max_tokens=self.max_tokens, messages=full_messages)
        if tools:
            kwargs["tools"] = _anthropic_tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except groq.RateLimitError as e:
                last_err = e
                _retry_sleep(attempt)
            except groq.APIStatusError as e:
                last_err = e
                if e.status_code and e.status_code >= 500:
                    _retry_sleep(attempt)
                    continue
                raise LLMError(f"LLM request failed: {e}") from e
            except groq.APIConnectionError as e:
                last_err = e
                _retry_sleep(attempt)
        else:
            raise LLMError(f"LLM request failed after retries: {last_err}") from last_err

        msg = response.choices[0].message
        text = (msg.content or "").strip()
        raw_tool_calls = msg.tool_calls or []
        tool_calls = []
        for tc in raw_tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, input=args))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        assistant_message = {
            "role": "assistant",
            "content": msg.content,
            **({"tool_calls": [tc.model_dump() for tc in raw_tool_calls]} if raw_tool_calls else {}),
        }
        return LLMTurn(text=text, tool_calls=tool_calls, stop_reason=stop_reason, assistant_message=assistant_message)

    def build_tool_result_messages(self, results: list[ToolCallResult]) -> list[dict]:
        return [
            {"role": "tool", "tool_call_id": r.id, "content": r.content}
            for r in results
        ]
def get_llm_client() -> LLMClient:
    settings = get_settings()
    if settings.llm_provider == "groq":
        return GroqLLMClient()
    return AnthropicLLMClient()
