from __future__ import annotations

from typing import Any

from config.settings import Settings, get_settings
from gen_ai_core.prompts.prompt import Prompt


class LangChainLLM:
    def __init__(self, settings: Settings | None = None, tools: list[Any] | None = None):
        self.s = settings or get_settings()
        self._provider = self.s.llm_provider
        self._llm: str | None = self.s.default_llm
        self._tools = tools
        self._llm_temperature: float = self.s.llm_temperature or 0.0
        self._model = self._build_model()

    def _build_model(self) -> Any:
        p, m = self._provider, self._llm
        temp = self._llm_temperature or 0.0
        if p == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=m, temperature=temp, base_url=self.s.openai_base_url or None)
        elif p == "groq":
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=m, temperature=temp)
        elif p == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=m, temperature=temp)
        elif p == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=m, temperature=temp)
        elif p == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=m, temperature=temp)
        elif p == "sap":
            from gen_ai_hub.proxy.langchain.openai import ChatOpenAI as SAPChatOpenAI
            from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
            llm = SAPChatOpenAI(
                proxy_model_name=m,
                proxy_client=get_proxy_client("gen-ai-hub"),
                temperature=temp,
            )
        else:
            raise ValueError(f"LangChain layer needs a real provider, got {p!r}")
        return llm.bind_tools(self._tools) if self._tools else llm

    def invoke(self, prompt: Prompt) -> str:
        resp = self._model.invoke(prompt.to_lc_messages())
        return getattr(resp, "text", None) or resp.content

    def chain(self, chat_prompt_template: Any) -> Any:
        from langchain_core.output_parsers import StrOutputParser
        return chat_prompt_template | self._model | StrOutputParser()
