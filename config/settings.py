from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
load_dotenv()

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _env_float(name: str, default: float | None) -> float | None:
    v = _env(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default

DEFAULT_MODELS: dict[str, str] = {
    "openai":    "gpt-6-luna", # cheap, fast OpenAI model (replaces gpt-4.1-mini / gpt-4o-mini)
    "groq":      "openai/gpt-oss-120b",
    "ollama":    "llama3.1:8b", # "llama3.2", # any model you have pulled locally with `ollama pull`
    "anthropic": "claude-haiku-5-5", # cheap, fast Anthropic model
    "gemini":    "gemini-2.5-flash",
    "sap":       "gpt-4o",
}

def _auto_provider() -> str:
    explicit = _env("LLM_PROVIDER").lower()
    if explicit:
        return explicit
    if _env("GROQ_API_KEY"):
        return "groq"
    if _env("OLLAMA_LOCAL_BASE_URL"):
        return "ollama"
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _env("GOOGLE_API_KEY"):
        return "gemini"
    return "none"


@dataclass
class Settings:
    # ---- LLM --------------------------------------------------------------
    llm_provider: str = field(default_factory=_auto_provider)
    default_llm: str = field(default_factory=lambda: _env("LLM_MODEL"))
    llm_temperature: float | None = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", None))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "700")))
    reasoning_effort: str = field(default_factory=lambda: _env("LLM_REASONING_EFFORT", "none"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))

    def __post_init__(self) -> None:
        if not self.default_llm and self.llm_provider in DEFAULT_MODELS:
            self.default_llm = DEFAULT_MODELS[self.llm_provider]


def get_settings(**overrides: Any) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    s.__post_init__()
    return s
