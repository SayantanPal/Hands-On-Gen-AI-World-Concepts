from __future__ import annotations

import json
import time
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
            # Build name → executable map, supporting BOTH tool shapes:
            #   • RawLLM-style dict      → {"name": ..., "function": callable}
            #   • bare LangChain tool    → a StructuredTool that carries its own .name
            self._tool_fn_map = {}
            for t in self._tools:
                if isinstance(t, dict) and "function" in t:
                    self._tool_fn_map[t["name"]] = t["function"]
                elif hasattr(t, "name"):          # BaseTool / StructuredTool
                    self._tool_fn_map[t.name] = t
        else:
            self._llm = self._llm_base

    def invoke(self, prompt: Prompt) -> str:
        """Plain text completion — uses base model (no tool calling)."""
        resp = self._llm_base.invoke(prompt.to_lc_messages())
        return getattr(resp, "text", None) or resp.content

    def invoke_with_tools(self, prompt: Any, variables: dict | None = None) -> dict:
        """Single-pass tool calling: model requests tools → we execute → model summarises.

        Flexible input — `prompt` may be any of:
          • Prompt             → our dataclass (uses .to_lc_messages())
          • ChatPromptTemplate → (or any LC prompt) formatted with `variables`
          • list[BaseMessage]  → already-built LangChain messages, used as-is
          • str                → treated as a single human message

        This is the LangChain equivalent of RawLLM.generate_response_content().
        Same single-pass limitation: if the model needs more than one round of
        tool calls, the second round won't happen (tools are not re-sent).

        Returns:
            dict with keys: text, tool_calls (list of {name, args, result})
        """
        from langchain_core.messages import ToolMessage

        messages = self._coerce_to_lc_messages(prompt, variables)

        if not self._tools:
            resp = self._llm_base.invoke(messages)
            text = getattr(resp, "text", None) or resp.content
            return {"text": (text or "").strip(), "tool_calls": []}

        response = self._llm.invoke(messages)
        messages.append(response)
        tool_calls_log: list[dict] = []

        if getattr(response, "tool_calls", None) and self._tool_fn_map:
            for tc in response.tool_calls:
                result = self._exec_tool(tc["name"], tc["args"])
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

    def invoke_with_multiple_tools_in_agentic_react_loop(
        self, prompt: Any, variables: dict | None = None, *, max_iterations: int = 5
    ) -> dict:
        """Multi-turn ReAct loop (LangChain): tools stay bound on EVERY round
        until the model stops calling them (or max_iterations is hit).

        This is the multi-round cousin of invoke_with_tools(). Where single-pass
        runs exactly one tool round (and fails on questions needing a second),
        this loop re-sends the tools each turn — so sequential / multi-entity
        questions ("compare A vs B") work. Mirrors react_loop_langchain() in
        demo_driver/my_trip_planner_demo/my_trip_planner.py.

        `prompt`: Prompt | ChatPromptTemplate (+ variables) | message list | str.
        Returns: dict with keys text, tool_calls, iterations, latency_ms.
        """
        from langchain_core.messages import ToolMessage

        t0 = time.perf_counter()
        messages = self._coerce_to_lc_messages(prompt, variables)

        # No tools → one plain text completion, framed like the loop result.
        if not self._tools:
            resp = self._llm_base.invoke(messages)
            text = getattr(resp, "text", None) or resp.content
            return {"text": (text or "").strip(), "tool_calls": [],
                    "iterations": 0, "latency_ms": (time.perf_counter() - t0) * 1000}

        tool_calls_log: list[dict] = []
        for iteration in range(1, max_iterations + 1):
            response = self._llm.invoke(messages)   # tools bound every round — never dropped
            messages.append(response)

            # No tool calls → model is done reasoning → final text answer.
            if not getattr(response, "tool_calls", None):
                text = getattr(response, "text", None) or response.content
                return {"text": (text or "").strip(), "tool_calls": tool_calls_log,
                        "iterations": iteration, "latency_ms": (time.perf_counter() - t0) * 1000}

            # Execute each requested tool, feed results back, loop again.
            for tc in response.tool_calls:
                result = self._exec_tool(tc["name"], tc["args"])
                tool_calls_log.append({"name": tc["name"], "args": tc["args"], "result": result})
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # Safety exit — model kept requesting tools past max_iterations.
        return {"text": "Stopped: too many tool-call rounds without a final answer.",
                "tool_calls": tool_calls_log, "iterations": max_iterations,
                "latency_ms": (time.perf_counter() - t0) * 1000}

    def _exec_tool(self, name: str, args: dict) -> str:
        """Execute one tool by name → JSON string. Duck-typed dispatch:
        LangChain tools use .invoke({args}); plain functions take kwargs."""
        fn = self._tool_fn_map.get(name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        raw = fn.invoke(args) if hasattr(fn, "invoke") else fn(**args)
        return raw if isinstance(raw, str) else json.dumps(raw)

    @staticmethod
    def _coerce_to_lc_messages(prompt: Any, variables: dict | None = None) -> list:
        """Normalise any supported prompt input into a mutable list of LangChain
        messages, so invoke_with_tools can consume several datatypes.

        Order matters — each branch tests a capability, most specific first.
        """
        # 1) our Prompt dataclass → knows how to emit LangChain messages
        if hasattr(prompt, "to_lc_messages"):
            return list(prompt.to_lc_messages())
        # 2) already a list of messages → copy so appends don't mutate the caller's list
        if isinstance(prompt, list):
            return list(prompt)
        # 3) an already-formatted PromptValue (e.g. ChatPromptValue) → has .to_messages()
        if hasattr(prompt, "to_messages"):
            return list(prompt.to_messages())
        # 4) a ChatPromptTemplate / Runnable prompt → format with variables, then flatten
        if hasattr(prompt, "invoke"):
            value = prompt.invoke(variables or {})
            return list(value.to_messages() if hasattr(value, "to_messages") else value)
        # 5) a plain string → single human message
        if isinstance(prompt, str):
            from langchain_core.messages import HumanMessage
            return [HumanMessage(content=prompt)]
        raise TypeError(f"invoke_with_tools: unsupported prompt type {type(prompt)!r}")

    '''
        lets you define reusable prompt templates with named variables,
        then fill them with different data each time
        — useful when you have structured/computed data to inject
    '''
    def chain(self, prompt: ChatPromptTemplate) -> Any:
        # Returns a LangChain pipe (RunnableSequence) → supports .invoke() only, NOT .invoke_with_tools().
        # Text-only (uses _llm_base, no tools). For tool calling, skip chain() and use invoke_with_tools().
        from langchain_core.output_parsers import StrOutputParser
        return prompt | self._llm_base | StrOutputParser()
