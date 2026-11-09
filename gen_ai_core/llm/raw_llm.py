from __future__ import annotations

import time
from typing import Any

from config.settings import Settings, _env, get_settings
from gen_ai_core.models.result import LLMResult
from gen_ai_core.prompts.prompt import Prompt


class RawLLM:

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._provider = self.s.llm_provider
        self._model: str | None = self.s.default_llm
        self._llm_temperature: float = self.s.llm_temperature or 0.0
        self._max_tokens = self.s.max_tokens or 700
        self._reasoning_effort = self.s.reasoning_effort
        self._client: Any = None
        # self._build_sdk_client_and_init_model_from_llm_provider()


    def _build_sdk_client_and_init_model_from_llm_provider(self) -> None:
        p = self._provider
        print(f"Building SDK client for provider: {p}")
        if p in {"openai", "groq", "ollama"}:
            from openai import OpenAI
            if p == "groq":
                default_groq_llm = "openai/gpt-oss-120b"
                print(f"Using Default Groq LLM: {default_groq_llm}" if not _env("GROQ_LLM") else f"Using Groq LLM: {_env("GROQ_LLM")}")                               
                self._model = _env("GROQ_LLM") or default_groq_llm
                self._client = OpenAI(
                    base_url= _env("GROQ_OPENAI_BASE_URL") or "https://api.groq.com/openai/v1",
                    api_key= _env("GROQ_API_KEY"),
                )
            elif p == "ollama":
                default_ollama_llm = "llama3.1:8b"
                print(f"Using Default Ollama LLM: {default_ollama_llm}" if not _env("OLLAMA_LLM") else f"Using Ollama LLM: {_env("OLLAMA_LLM")}")
                self._model = _env("OLLAMA_LLM") or default_ollama_llm
                self._client = OpenAI(
                    base_url= _env("OLLAMA_LOCAL_BASE_URL") or "http://localhost:11434/v1", 
                    api_key="ollama"
                )
            elif p == "openai":
                default_openai_llm = "gpt-6-luna"
                print(f"Using Default OpenAI LLM: {default_openai_llm}" if not _env("OPENAI_LLM") else f"Using OpenAI LLM: {_env("OPENAI_LLM")}")
                self._model = _env("OPENAI_LLM") or default_openai_llm
                self._client = OpenAI(
                    base_url= _env("OPENAI_BASE_URL") or None
                )

        elif p == "gemini":
                    from google import genai
                    default_gemini_llm = "gemini-2.5-flash"
                    print(f"Using Default Gemini LLM: {default_gemini_llm}" if not _env("GEMINI_LLM") else f"Using Gemini LLM: {_env("GEMINI_LLM")}")
                    self._model = _env("GEMINI_LLM") or default_gemini_llm
                    self._client = genai.Client(api_key=_env("GOOGLE_API_KEY"))

        elif p == "anthropic":
            import anthropic
            default_anthropic_llm = "claude-sonnet-4-6"
            print(f"Using Default Anthropic LLM: {default_anthropic_llm}" if not _env("ANTHROPIC_LLM") else f"Using Anthropic LLM: {_env("ANTHROPIC_LLM")}")
            self._model = _env("ANTHROPIC_LLM") or default_anthropic_llm
            self._client = anthropic.Anthropic()

        elif p == "sap":
            from gen_ai_hub.proxy.native.openai import chat as sap_chat
            default_sap_llm = "sap-1"
            print(f"Using Default SAP LLM: {default_sap_llm}" if not _env("SAP_LLM") else f"Using SAP LLM: {_env("SAP_LLM")}")
            self._model = _env("SAP_LLM") or default_sap_llm
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
            from openai import OpenAI
            self._provider = "groq"
            default_groq_llm = "openai/gpt-oss-120b"
            self._model = _env("GROQ_LLM") or default_groq_llm
            self._client = OpenAI(
                base_url=_env("GROQ_OPENAI_BASE_URL") or "https://api.groq.com/openai/v1",
                api_key=groq_key,
            )
            print(f"Auto-detected provider: groq (model: {self._model})")
            return

        ollama_url = _env("OLLAMA_LOCAL_BASE_URL") or "http://localhost:11434/v1"
        try:
            import urllib.request
            urllib.request.urlopen(ollama_url.replace("/v1", ""), timeout=1)
            from openai import OpenAI
            self._provider = "ollama"
            self._model = _env("OLLAMA_LLM") or "llama3.1:8b"
            self._client = OpenAI(base_url=ollama_url, api_key="ollama")
            print(f"Auto-detected provider: ollama (model: {self._model})")
            return
        except Exception:
            pass

        google_key = _env("GOOGLE_API_KEY")
        if google_key:
            from google import genai
            self._provider = "gemini"
            default_gemini_llm = "gemini-2.5-flash"
            self._model = _env("GEMINI_LLM") or default_gemini_llm
            self._client = genai.Client(api_key=google_key)
            print(f"Auto-detected provider: gemini (model: {self._model})")
            return
        
        openai_key = _env("OPENAI_API_KEY")
        if openai_key:
            from openai import OpenAI
            self._provider = "openai"
            default_openai_llm = "gpt-6-luna"
            self._model = _env("OPENAI_LLM") or default_openai_llm
            self._client = OpenAI(base_url=_env("OPENAI_BASE_URL") or None, api_key=openai_key)
            print(f"Auto-detected provider: openai (model: {self._model})")
            return

        anthropic_key = _env("ANTHROPIC_API_KEY")
        if anthropic_key:
            import anthropic
            self._provider = "anthropic"
            default_anthropic_llm = "claude-sonnet-4-6"
            self._model = _env("ANTHROPIC_LLM") or default_anthropic_llm
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
        usage = getattr(resp, "usage", None)
        return LLMResult(
            (resp.choices[0].message.content or "").strip(),
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            (time.perf_counter() - t0) * 1000,
            self._model, self._provider,
        )

    
    def _gen_anthropic(self, prompt: Prompt, json_mode: bool = False,t0: float = time.perf_counter(),max_tokens: int | None = None, temperature: float | None = None) -> LLMResult:
        user = prompt.user_query
        if json_mode:
            user += "\n\nReply with valid JSON only. No text before or after the JSON."
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "system": prompt.system_text(),
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return LLMResult(
            text.strip(), resp.usage.input_tokens, resp.usage.output_tokens,
            (time.perf_counter() - t0) * 1000, self._model, "anthropic",
        )

    
    def _gen_gemini(self, prompt: Prompt, json_mode: bool = False, t0: float = time.perf_counter(), max_tokens: int | None = None, temperature: float | None = None) -> LLMResult:
        cfg: dict[str, Any] = {}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if max_tokens is not None:
            cfg["max_output_tokens"] = max_tokens
        if temperature is not None:
            cfg["temperature"] = temperature
        contents = prompt.system_text() + "\n\n" + prompt.user_query
        resp = self._client.models.generate_content(model=self._model, contents=contents, config=cfg or None)
        um = getattr(resp, "usage_metadata", None)
        return LLMResult(
            (resp.text or "").strip(),
            getattr(um, "prompt_token_count", 0) or 0,
            getattr(um, "candidates_token_count", 0) or 0,
            (time.perf_counter() - t0) * 1000,
            self._model, "gemini",
        )
