from __future__ import annotations

from typing import Any

from config.settings import Settings, get_settings
from gen_ai_core.prompts.prompt import Prompt
from langchain_core.prompts import ChatPromptTemplate


class LangChainLLM:
    def __init__(self, settings: Settings | None = None, 
                 tools: list[Any] | None = None,
                 provider: str | None = None, model: str | None = None):
        self.s = settings or get_settings()
        self._provider = provider or self.s.llm_provider
        self._model = model or self.s.default_llm
        self._llm: Any = None
        self._tools = tools
        self._llm_temperature: float = self.s.llm_temperature or 0.0
        if self._provider == "none":
            print("\n[LangChain SDK] skipped — no provider configured (set .env)")
        if self._model == "none":
            print("\n[LangChain SDK] skipped — no model configured (set .env)")
        else:
            self._build_model_from_llm_provider()
            print(f"Langchain Wrapper SDK: provider={self._provider}  model={self._model}")

    # def __repr__(self) -> str:
    #     return f"LangChainLLM({self._provider}:{self._model})" if self._model else "LangChainLLM(none)"


    def _build_model_from_llm_provider(self) -> Any:
        p, m = self._provider, self._model
        temp = self._llm_temperature or 0.0
        if p == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(model=m, temperature=temp, base_url=self.s.openai_base_url or None)
        elif p == "groq":
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(model=m, temperature=temp)
        elif p == "ollama":
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(model=m, temperature=temp)
        elif p == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(model=m, temperature=temp)
        elif p == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(model=m, temperature=temp)
        elif p == "sap":
            from gen_ai_hub.proxy.langchain.openai import ChatOpenAI as SAPChatOpenAI
            from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
            self._llm = SAPChatOpenAI(
                proxy_model_name=m,
                proxy_client=get_proxy_client("gen-ai-hub"),
                temperature=temp,
            )
        else:
            raise ValueError(f"LangChain layer needs a real provider, got {p!r}")
        return self._llm.bind_tools(self._tools) if self._tools else self._llm

    def invoke(self, prompt: Prompt) -> str:
        resp = self._llm.invoke(prompt.to_lc_messages())
        return getattr(resp, "text", None) or resp.content

    '''
        lets you define reusable prompt templates with named variables, 
        then fill them with different data each time 
        — useful when you have structured/computed data to inject
    '''
    def chain(self, prompt: ChatPromptTemplate) -> Any:
        from langchain_core.output_parsers import StrOutputParser
        return prompt | self._llm | StrOutputParser()
