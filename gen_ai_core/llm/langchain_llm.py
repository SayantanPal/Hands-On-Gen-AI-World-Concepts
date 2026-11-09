from __future__ import annotations

import json
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
        self._llm_base: Any = None
        self._tools = tools
        self._tool_fn_map: dict[str, Any] = {}
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


    def _build_model_from_llm_provider(self) -> None:
        p, m = self._provider, self._model
        temp = self._llm_temperature or 0.0
        if p == "openai":
            from langchain_openai import ChatOpenAI
            self._llm_base = ChatOpenAI(model=m, temperature=temp, base_url=self.s.openai_base_url or None)
        elif p == "groq":
            from langchain_groq import ChatGroq
            self._llm_base = ChatGroq(model=m, temperature=temp)
        elif p == "ollama":
            from langchain_ollama import ChatOllama
            self._llm_base = ChatOllama(model=m, temperature=temp)
        elif p == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._llm_base = ChatAnthropic(model=m, temperature=temp)
        elif p == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm_base = ChatGoogleGenerativeAI(model=m, temperature=temp)
        elif p == "sap":
            from gen_ai_hub.proxy.langchain.openai import ChatOpenAI as SAPChatOpenAI
            from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
            self._llm_base = SAPChatOpenAI(
                proxy_model_name=m,
                proxy_client=get_proxy_client("gen-ai-hub"),
                temperature=temp,
            )
        else:
            raise ValueError(f"LangChain layer needs a real provider, got {p!r}")

        # BUG FIX: previously .bind_tools() was called but the result was
        # returned (and discarded by __init__), so self._llm was always the
        # base model without tools.  Now we store both:
        #   _llm_base → base model (for invoke / chain — no tool awareness)
        #   _llm      → model with tools bound (if tools were provided)
        if self._tools:
            self._llm = self._llm_base.bind_tools(self._tools)
            # Build name → callable map if tools carry a "function" key
            self._tool_fn_map = {
                t["name"]: t["function"]
                for t in self._tools
                if isinstance(t, dict) and "function" in t
            }
        else:
            self._llm = self._llm_base

    def invoke(self, prompt: Prompt) -> str:
        """Plain text completion — uses base model (no tool calling)."""
        resp = self._llm_base.invoke(prompt.to_lc_messages())
        return getattr(resp, "text", None) or resp.content

    def invoke_with_tools(self, prompt: Prompt) -> dict:
        """Single-pass tool calling: model requests tools → we execute → model summarises.

        This is the LangChain equivalent of RawLLM.generate_response_content().
        Same single-pass limitation: if the model needs more than one round of
        tool calls, the second round won't happen (tools are not re-sent).

        Returns:
            dict with keys: text, tool_calls (list of {name, args, result})
        """
        from langchain_core.messages import ToolMessage

        if not self._tools:
            return {"text": self.invoke(prompt), "tool_calls": []}

        messages = prompt.to_lc_messages()
        response = self._llm.invoke(messages)
        messages.append(response)
        tool_calls_log: list[dict] = []

        if getattr(response, "tool_calls", None) and self._tool_fn_map:
            for tc in response.tool_calls:
                fn = self._tool_fn_map.get(tc["name"])
                if fn is None:
                    result = json.dumps({"error": f"Unknown tool: {tc['name']}"})
                else:
                    raw = fn(**tc["args"])
                    result = raw if isinstance(raw, str) else json.dumps(raw)
                tool_calls_log.append({"name": tc["name"], "args": tc["args"], "result": result})
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

            # Second call — model summarises the tool results into a text answer
            # (same single-pass pattern as RawLLM: tools still bound, but only
            # one round of execution — a multi-round question would need a loop)
            final = self._llm.invoke(messages)
            text = getattr(final, "text", None) or final.content
        else:
            text = getattr(response, "text", None) or response.content

        return {"text": (text or "").strip(), "tool_calls": tool_calls_log}

    '''
        lets you define reusable prompt templates with named variables,
        then fill them with different data each time
        — useful when you have structured/computed data to inject
    '''
    def chain(self, prompt: ChatPromptTemplate) -> Any:
        from langchain_core.output_parsers import StrOutputParser
        return prompt | self._llm_base | StrOutputParser()
