from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gen_ai_core.prompts.rctfc_template import build_rctfc_system_prompt


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
        base = build_rctfc_system_prompt(
            role=self.role,
            task=self.task,
            context=self.context,
            output_format=self.output_format,
            constraints=self.constraints or None,
        )
        if self.rag_context:
            base += f"\n\n[RETRIEVED CONTEXT — answer using ONLY this when relevant]\n{self.rag_context}"
        return base

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
