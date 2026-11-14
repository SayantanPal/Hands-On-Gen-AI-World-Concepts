from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gen_ai_core.prompts.rctfc_template import build_rctfc_system_prompt


@dataclass
class Prompt:
    role: str = "a helpful assistant"
    context: str = ""
    task: str = ""
    rules: str = ""
    output_format: str = ""
    constraints: list[str] = field(default_factory=list)
    rag_context: str = ""
    assistant_priming: str = ""
    user_query: str = ""

    def get_base_system_prompt(self) -> str:
        base_sys_prompt = build_rctfc_system_prompt(
            role=self.role,
            task=self.task,
            rules=self.rules or None,
            context=self.context,
            output_format=self.output_format,
            constraints=self.constraints or None,
        )
        if self.rag_context:
            base_sys_prompt += f"\n\n[RETRIEVED CONTEXT — answer using ONLY this when relevant]\n{self.rag_context}"
        return base_sys_prompt

    def to_role_dicts(self) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.get_base_system_prompt()}]
        if self.assistant_priming:
            msgs.append({"role": "assistant", "content": self.assistant_priming})
        msgs.append({"role": "user", "content": self.user_query})
        return msgs

    def to_lc_messages(self) -> list[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        msgs: list[Any] = [SystemMessage(content=self.get_base_system_prompt())]
        if self.assistant_priming:
            msgs.append(AIMessage(content=self.assistant_priming))
        msgs.append(HumanMessage(content=self.user_query))
        return msgs
