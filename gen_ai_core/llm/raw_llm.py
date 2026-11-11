from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

from config.settings import Settings, _env, get_settings
from gen_ai_core.models.result import LLMResult
from gen_ai_core.prompts.prompt import Prompt


class RawLLM:

    def __init__(self, settings: Settings | None = None,
                 tools: list[dict[str, Any]] | None = None):
        """
        Args:
            settings: Optional Settings override.
            tools:    Optional list of tool dicts, each with keys:
                      - name:        str — tool function name
                      - description: str — what the tool does
                      - parameters:  dict — JSON Schema of the arguments
                      - function:    Callable — the actual Python function to execute
        """
        self.s = settings or get_settings()
        self._provider = self.s.llm_provider
        self._model: str | None = self.s.default_llm
        self._llm_temperature: float = self.s.llm_temperature or 0.0
        self._max_tokens = self.s.max_tokens or 700
        self._reasoning_effort = self.s.reasoning_effort
        self._client: Any = None
        self._tools = tools or []
        self._tool_map: dict[str, Callable] = {
            t["name"]: t["function"] for t in self._tools
        }
        self._build_sdk_client_and_init_model_from_llm_provider()
        print(f"Raw LLM SDK: provider={self._provider}  model={self._model}")
        if self._provider == "none":
            print("\n[Raw SDK] skipped — no provider configured (set .env)")


    def _build_sdk_client_and_init_model_from_llm_provider(self) -> None:
        p = self._provider
        print(f"Building SDK client for provider: {p}")
        if p in {"openai", "groq", "ollama"}:
            from openai import OpenAI
            if p == "groq":
                # default_groq_llm = "openai/gpt-oss-120b"
                # print(f"Using Default Groq LLM: {default_groq_llm}" if not _env("GROQ_LLM") else f"Using Groq LLM: {_env("GROQ_LLM")}")                               
                # self._model = _env("GROQ_LLM") or default_groq_llm
                self._client = OpenAI(
                    base_url= _env("GROQ_OPENAI_BASE_URL") or "https://api.groq.com/openai/v1",
                    api_key= _env("GROQ_API_KEY"),
                )
            elif p == "ollama":
                # default_ollama_llm = "llama3.1:8b"
                # print(f"Using Default Ollama LLM: {default_ollama_llm}" if not _env("OLLAMA_LLM") else f"Using Ollama LLM: {_env("OLLAMA_LLM")}")
                # self._model = _env("OLLAMA_LLM") or default_ollama_llm
                self._client = OpenAI(
                    base_url= _env("OLLAMA_LOCAL_BASE_URL") or "http://localhost:11434/v1", 
                    api_key="ollama"
                )
            elif p == "openai":
                # default_openai_llm = "gpt-6-luna"
                # print(f"Using Default OpenAI LLM: {default_openai_llm}" if not _env("OPENAI_LLM") else f"Using OpenAI LLM: {_env("OPENAI_LLM")}")
                # self._model = _env("OPENAI_LLM") or default_openai_llm
                self._client = OpenAI(
                    base_url= _env("OPENAI_BASE_URL") or None
                )

        elif p == "gemini":
                    # default_gemini_llm = "gemini-2.5-flash"
                    # print(f"Using Default Gemini LLM: {default_gemini_llm}" if not _env("GEMINI_LLM") else f"Using Gemini LLM: {_env("GEMINI_LLM")}")
                    # self._model = _env("GEMINI_LLM") or default_gemini_llm
                    from google import genai
                    self._client = genai.Client(api_key=_env("GOOGLE_API_KEY"))

        elif p == "anthropic":
            # default_anthropic_llm = "claude-sonnet-4-6"
            # print(f"Using Default Anthropic LLM: {default_anthropic_llm}" if not _env("ANTHROPIC_LLM") else f"Using Anthropic LLM: {_env("ANTHROPIC_LLM")}")
            # self._model = _env("ANTHROPIC_LLM") or default_anthropic_llm
            import anthropic
            self._client = anthropic.Anthropic()

        elif p == "sap":
            # default_sap_llm = "sap-1"
            # print(f"Using Default SAP LLM: {default_sap_llm}" if not _env("SAP_LLM") else f"Using SAP LLM: {_env("SAP_LLM")}")
            # self._model = _env("SAP_LLM") or default_sap_llm
            from gen_ai_hub.proxy.native.openai import chat as sap_chat
            self._client = sap_chat

        elif p in {"none", "", None}:
            self._detect_sdk_client_and_init_model_and_provider_from_api_keys()
        else:
            print("No provider configured and no API keys detected")
            raise ValueError(f"Unknown provider {p!r}")


    def _detect_sdk_client_and_init_model_and_provider_from_api_keys(self) -> None:
        print("Detecting SDK client from provided API Keys")
        groq_key = _env("GROQ_API_KEY")
        if groq_key:
            # self._provider = "groq"
            # default_groq_llm = "openai/gpt-oss-120b"
            # self._model = _env("GROQ_LLM") or default_groq_llm
            from openai import OpenAI
            self._client = OpenAI(
                base_url=_env("GROQ_OPENAI_BASE_URL") or "https://api.groq.com/openai/v1",
                api_key=groq_key,
            )
            print(f"Auto-detected provider: groq (model: {self._model})")
            return

        ollama_url = _env("OLLAMA_LOCAL_BASE_URL") or "http://localhost:11434/v1"
        try:
            # import urllib.request
            # urllib.request.urlopen(ollama_url.replace("/v1", ""), timeout=1)
            # from openai import OpenAI
            # self._provider = "ollama"
            # self._model = _env("OLLAMA_LLM") or "llama3.1:8b"
            self._client = OpenAI(base_url=ollama_url, api_key="ollama")
            print(f"Auto-detected provider: ollama (model: {self._model})")
            return
        except Exception:
            pass

        google_key = _env("GOOGLE_API_KEY")
        if google_key:
            # from google import genai
            # self._provider = "gemini"
            # default_gemini_llm = "gemini-2.5-flash"
            # self._model = _env("GEMINI_LLM") or default_gemini_llm
            self._client = genai.Client(api_key=google_key)
            print(f"Auto-detected provider: gemini (model: {self._model})")
            return
        
        openai_key = _env("OPENAI_API_KEY")
        if openai_key:
            # self._provider = "openai"
            # default_openai_llm = "gpt-6-luna"
            # self._model = _env("OPENAI_LLM") or default_openai_llm
            from openai import OpenAI
            self._client = OpenAI(base_url=_env("OPENAI_BASE_URL") or None, api_key=openai_key)
            print(f"Auto-detected provider: openai (model: {self._model})")
            return

        anthropic_key = _env("ANTHROPIC_API_KEY")
        if anthropic_key:
            # self._provider = "anthropic"
            # default_anthropic_llm = "claude-sonnet-4-6"
            # self._model = _env("ANTHROPIC_LLM") or default_anthropic_llm
            import anthropic
            self._client = anthropic.Anthropic(api_key=anthropic_key)
            print(f"Auto-detected provider: anthropic (model: {self._model})")
            return
        print("No provider configured and no API keys detected")
        raise ValueError(f"Unknown provider {self._provider!r}")


    @property
    def is_sdk_client_available(self) -> bool:
        return self._client is not None


    def __repr__(self) -> str:
        return f"RawLLM({self._provider}:{self._model})" if self.is_sdk_client_available else "RawLLM(none)"


    def _openai_tool_defs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self._tools
        ]

    def _anthropic_tool_defs(self) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in self._tools
        ]

    def _execute_tool(self, name: str, arguments: dict) -> str:
        fn = self._tool_map.get(name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        # Duck-typed: LangChain @tool objects expose .invoke({args}) and are
        # not plain-callable; plain Python functions take kwargs directly.
        result = fn.invoke(arguments) if hasattr(fn, "invoke") else fn(**arguments)
        return result if isinstance(result, str) else json.dumps(result)

    @staticmethod
    def _lc_messages_to_role_dicts(msgs: list) -> list[dict]:
        """Convert LangChain message objects → OpenAI role dicts. Duck-typed on
        .type / .content, so no hard LangChain import is needed here."""
        role_map = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
        return [
            {"role": role_map.get(getattr(m, "type", "user"), "user"),
             "content": getattr(m, "content", str(m))}
            for m in msgs
        ]

    def _coerce_to_role_dicts(self, prompt: Any, variables: dict | None = None) -> list[dict]:
        """Normalise supported prompt inputs into OpenAI role dicts, so the
        ReAct loop accepts several datatypes (most-specific branch first).

        Accepts: Prompt | ChatPromptTemplate (+variables) | role-dict list | str.
        """
        # 1) our Prompt dataclass → already emits OpenAI role dicts
        if hasattr(prompt, "to_role_dicts"):
            return list(prompt.to_role_dicts())
        # 2) already a list of role dicts → copy so appends don't mutate caller's
        if isinstance(prompt, list):
            return list(prompt)
        # 3) an already-formatted PromptValue → .to_messages() then convert
        if hasattr(prompt, "to_messages"):
            return self._lc_messages_to_role_dicts(prompt.to_messages())
        # 4) a ChatPromptTemplate / Runnable prompt → format with variables, convert
        if hasattr(prompt, "invoke"):
            value = prompt.invoke(variables or {})
            msgs = value.to_messages() if hasattr(value, "to_messages") else value
            return self._lc_messages_to_role_dicts(msgs)
        # 5) a plain string → single user message
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        raise TypeError(f"react loop: unsupported prompt type {type(prompt)!r}")

    def invoke_with_multiple_tools_in_agentic_react_loop(
        self, prompt: Any, variables: dict | None = None, *, max_iterations: int = 5
    ) -> dict:
        """Multi-turn ReAct loop (raw OpenAI-compatible SDK): tools stay in the
        request on EVERY round (never popped), so the model can chain several
        sequential tool calls until it is done.

        This is the multi-round cousin of generate_response_content(), which is
        single-pass (drops tools after one round and thus crashes on questions
        needing a second). Mirrors react_loop() in
        demo_driver/my_trip_planner_demo/my_trip_planner.py.

        Supported providers: openai / groq / ollama (chat.completions API).
        For anthropic / gemini / sap, use generate_response_content (single pass)
        or add a provider-specific loop — their message formats differ.

        `prompt`: Prompt | ChatPromptTemplate (+ variables) | role-dict list | str.
        Returns: dict with keys text, tool_calls, iterations, latency_ms.
        """
        if self._provider not in {"openai", "groq", "ollama"}:
            raise NotImplementedError(
                f"ReAct loop here supports openai/groq/ollama; got {self._provider!r}. "
                "Use generate_response_content() for a single pass, or add a "
                "provider-specific loop for anthropic/gemini/sap."
            )

        t0 = time.perf_counter()
        messages = self._coerce_to_role_dicts(prompt, variables)
        tool_defs = self._openai_tool_defs() if self._tools else None
        tool_calls_log: list[dict] = []

        for iteration in range(1, max_iterations + 1):
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": self._llm_temperature,
                "max_tokens": self._max_tokens,
            }
            if tool_defs:
                kwargs["tools"] = tool_defs          # tools on EVERY round — never popped
            resp = self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            # No tool calls → model is done reasoning → final text answer.
            if not msg.tool_calls:
                return {"text": (msg.content or "").strip(), "tool_calls": tool_calls_log,
                        "iterations": iteration, "latency_ms": (time.perf_counter() - t0) * 1000}

            # Record the assistant's tool_calls (needed so tool results match ids).
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            # Execute each tool, feed results back, then loop with tools still present.
            for tc in msg.tool_calls:
                fn_args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, fn_args)
                tool_calls_log.append({"name": tc.function.name, "args": fn_args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Safety exit — model kept requesting tools past max_iterations.
        return {"text": "Stopped: too many tool-call rounds without a final answer.",
                "tool_calls": tool_calls_log, "iterations": max_iterations,
                "latency_ms": (time.perf_counter() - t0) * 1000}


    def generate_response_content(self, prompt: Prompt, *, json_mode: bool = False,
                 max_tokens: int | None = None, temperature: float | None = None) -> LLMResult:
        if not self.is_sdk_client_available:
            fallback = prompt.rag_context or prompt.context or "(no LLM configured and no context to quote)"
            return LLMResult(fallback.strip()[:1200], provider="none")

        max_tokens = max_tokens or self._max_tokens
        temperature = temperature or self._llm_temperature

        t0 = time.perf_counter()
        p = self._provider

        if p == "anthropic":
            return self._gen_anthropic(prompt, json_mode, max_tokens, t0, temperature)
        elif p == "gemini":
            return self._gen_gemini(prompt, json_mode, t0, max_tokens, temperature)
        elif p in {"openai", "groq", "ollama", "sap"}:
            return self._gen_openai_compatible(prompt, json_mode, t0, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider for generation: {p!r}")

    
    def _gen_openai_compatible(self, prompt: Prompt, json_mode: bool = False, t0: float = time.perf_counter(), max_tokens: int | None = None, temperature: float | None = None) -> LLMResult:
        kwargs: dict[str, Any] = {"model": self._model, "messages": prompt.to_role_dicts()}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._tools:
            kwargs["tools"] = self._openai_tool_defs()

        if self._provider == "sap":
            kwargs.pop("model")
            kwargs["model_name"] = self._model
            resp = self._client.completions.create(**kwargs)
            d = resp.model_dump()
            text = d["choices"][0]["message"]["content"] or ""
            usage = d.get("usage", {}) or {}
            return LLMResult(
                text.strip(),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                (time.perf_counter() - t0) * 1000,
                self._model, "sap",
            )

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls_log: list[dict] = []

        if msg.tool_calls and self._tools:
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
            messages = kwargs["messages"] + [assistant_msg]
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                result = self._execute_tool(fn_name, fn_args)
                tool_calls_log.append({"name": fn_name, "args": fn_args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            kwargs["messages"] = messages
            kwargs.pop("tools", None)
            resp = self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

        usage = getattr(resp, "usage", None)
        return LLMResult(
            (msg.content or "").strip(),
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            (time.perf_counter() - t0) * 1000,
            self._model, self._provider,
            tool_calls=tool_calls_log,
        )

    
    def _gen_anthropic(self, prompt: Prompt, json_mode: bool = False,t0: float = time.perf_counter(),max_tokens: int | None = None, temperature: float | None = None) -> LLMResult:
        user = prompt.user_query
        if json_mode:
            user += "\n\nReply with valid JSON only. No text before or after the JSON."
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "system": prompt.get_base_system_prompt(),
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self._tools:
            kwargs["tools"] = self._anthropic_tool_defs()

        resp = self._client.messages.create(**kwargs)
        tool_calls_log: list[dict] = []

        tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
        if tool_use_blocks and self._tools:
            tool_results = []
            for tb in tool_use_blocks:
                result = self._execute_tool(tb.name, tb.input)
                tool_calls_log.append({"name": tb.name, "args": tb.input, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": result,
                })
            kwargs["messages"] = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": [b.model_dump() for b in resp.content]},
                {"role": "user", "content": tool_results},
            ]
            kwargs.pop("tools", None)
            resp = self._client.messages.create(**kwargs)

        text = "".join(getattr(b, "text", "") for b in resp.content)
        return LLMResult(
            text.strip(), resp.usage.input_tokens, resp.usage.output_tokens,
            (time.perf_counter() - t0) * 1000, self._model, "anthropic",
            tool_calls=tool_calls_log,
        )

    
    def _gen_gemini(self, prompt: Prompt, json_mode: bool = False, t0: float = time.perf_counter(), max_tokens: int | None = None, temperature: float | None = None) -> LLMResult:
        from google.genai import types as gtypes

        cfg: dict[str, Any] = {}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if max_tokens is not None:
            cfg["max_output_tokens"] = max_tokens
        if temperature is not None:
            cfg["temperature"] = temperature
        if self._tools:
            cfg["tools"] = [gtypes.Tool(function_declarations=[
                gtypes.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["parameters"],
                )
                for t in self._tools
            ])]

        contents = prompt.get_base_system_prompt() + "\n\n" + prompt.user_query
        resp = self._client.models.generate_content(
            model=self._model, contents=contents, config=cfg or None,
        )
        tool_calls_log: list[dict] = []

        fn_calls = [
            p for p in (resp.candidates[0].content.parts or [])
            if p.function_call is not None
        ] if resp.candidates else []

        if fn_calls and self._tools:
            tool_parts = []
            for fc_part in fn_calls:
                fc = fc_part.function_call
                fn_args = dict(fc.args) if fc.args else {}
                result_str = self._execute_tool(fc.name, fn_args)
                tool_calls_log.append({"name": fc.name, "args": fn_args, "result": result_str})
                result_dict = json.loads(result_str) if result_str.startswith("{") else {"result": result_str}
                tool_parts.append(gtypes.Part.from_function_response(
                    name=fc.name, response=result_dict,
                ))
            history = [
                gtypes.Content(parts=[gtypes.Part.from_text(text=contents)], role="user"),
                resp.candidates[0].content,
                gtypes.Content(parts=tool_parts, role="user"),
            ]
            cfg.pop("tools", None)
            resp = self._client.models.generate_content(
                model=self._model, contents=history, config=cfg or None,
            )

        um = getattr(resp, "usage_metadata", None)
        return LLMResult(
            (resp.text or "").strip(),
            getattr(um, "prompt_token_count", 0) or 0,
            getattr(um, "candidates_token_count", 0) or 0,
            (time.perf_counter() - t0) * 1000,
            self._model, "gemini",
            tool_calls=tool_calls_log,
        )
