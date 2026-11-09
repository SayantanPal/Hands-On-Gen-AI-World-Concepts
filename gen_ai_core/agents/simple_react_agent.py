from __future__ import annotations

import json
from typing import Any

from config.settings import Settings
from gen_ai_core.llm.langchain_llm import LangChainLLM


class ReActAgent:
    def __init__(self, tools: list[Any], settings: Settings | None = None, max_steps: int = 8):
        self.tool_map = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.llm = LangChainLLM(settings, tools=tools).model

    def run(self, user_question: str, system_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        messages: list[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=user_question)]
        for _ in range(self.max_steps):
            response = self.llm.invoke(messages)
            messages.append(response)
            if not getattr(response, "tool_calls", None):
                return getattr(response, "text", None) or response.content
            for tc in response.tool_calls:
                result = self.tool_map[tc["name"]].invoke(tc["args"])
                messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))
        return "Stopped: too many tool calls without a final answer."
