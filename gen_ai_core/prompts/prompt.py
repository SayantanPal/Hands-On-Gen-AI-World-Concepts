from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Prompt:
    role: str = "a helpful assistant"
    context: str = ""
    task: str = ""
    output_format: str = ""
    constraints: list[str] = field(default_factory=list)
    rag_context: str = ""
    assistant_priming: str = ""
    user_query: str = ""

    def system_text(self) -> str:
        parts = [f"[ROLE]\nYou are {self.role}."]
        if self.context:
            parts.append(f"[CONTEXT]\n{self.context}")
        if self.task:
            parts.append(f"[TASK]\n{self.task}")
        if self.output_format:
            parts.append(f"[FORMAT]\n{self.output_format}")
        if self.constraints:
            rules = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"[GUARDRAILS CONSTRAINTS]\n{rules}")
        if self.rag_context:
            parts.append(f"[RETRIEVED CONTEXT — answer using ONLY this when relevant]\n{self.rag_context}")
        return "\n\n".join(parts)

    def to_role_dicts(self) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_text()}]
        if self.assistant_priming:
            msgs.append({"role": "assistant", "content": self.assistant_priming})
        msgs.append({"role": "user", "content": self.user_query})
        return msgs

    def to_lc_messages(self) -> list[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        msgs: list[Any] = [SystemMessage(content=self.system_text())]
        if self.assistant_priming:
            msgs.append(AIMessage(content=self.assistant_priming))
        msgs.append(HumanMessage(content=self.user_query))
        return msgs
