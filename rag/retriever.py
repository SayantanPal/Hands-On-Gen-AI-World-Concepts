from __future__ import annotations

from typing import Callable

from config.settings import Settings
from gen_ai_core.llm.raw_llm import RawLLM
from gen_ai_core.models.result import LLMResult
from gen_ai_core.prompts.presets import HARDENED_TRAVEL
from gen_ai_core.prompts.prompt import Prompt

Retriever = Callable[[str], list[str]]


def rag_answer(
    question: str,
    retriever: Retriever | None,
    *,
    base_prompt: Prompt = HARDENED_TRAVEL,
    settings: Settings | None = None,
    k_join: str = "\n",
) -> LLMResult:
    rag_context = ""
    if retriever is not None:
        chunks = retriever(question)
        rag_context = k_join.join(chunks)
    prompt = Prompt(**{**base_prompt.__dict__, "rag_context": rag_context, "user_query": question})
    return RawLLM(settings).generate(prompt)