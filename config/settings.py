from __future__ import annotations

import os, sys
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


# Detect: api keys -(auto)-> api provider -(auto/default)-> llm model

# provides api provider from api keys
def _auto_provider() -> str:
    explicit = _env("LLM_PROVIDER").lower()
    if explicit:
        return explicit
    # Placed Priority-wise: prioritise free-tier API providers and local LLMs over paid-tiers 
    if _env("GROQ_API_KEY"):
        return "groq"
    if _env("OLLAMA_LOCAL_BASE_URL"):
        return "ollama"
    if _env("GOOGLE_API_KEY"):
        return "gemini"
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none" # no LLM: the labs fall back to an extractive answer

# provides default models from api providers {key: provider, value: model_name}
# Default model names. Check your provider's model page before you change these:
# model names get deprecated. These were current in November 2026.
DEFAULT_MODELS: dict[str, str] = {
    "openai":    "gpt-6-luna", # cheap, fast OpenAI model (replaces gpt-4.1-mini / gpt-4o-mini)
    "groq":      "openai/gpt-oss-120b",
    "ollama":    "llama3.1:8b", # "llama3.2", # any model you have pulled locally with `ollama pull`
    "anthropic": "claude-haiku-5-5", # cheap, fast Anthropic model
    "gemini":    "gemini-2.5-flash",
    "sap":       "gpt-4o",
    "local_transformers": "meta-llama/Llama-3.1-8B-Instruct", # default local transformers model
}

DEFAULT_LOCAL_TRANSFORMERS_MODEL = "microsoft/Phi-3-mini-4k-instruct" #"meta-llama/Llama-3.1-8B-Instruct"


from pathlib import Path
ROOT = Path(__file__).resolve().parents[0]

# try:  # python-dotenv is optional
#     from dotenv import load_dotenv

#     load_dotenv(ROOT / ".env")
# except Exception:  # pragma: no cover
#     pass

# # ---- folders ----------------------------------------------------------
data_dir: Path = ROOT / "demo_sample_data" / "unstructured_data" / "raw"
processed_dir: Path = ROOT / "demo_sample_data" / "unstructured_data" / "processed"
log_dir: Path = ROOT / "logs"
reports_dir: Path = ROOT / "reports"

@dataclass
class Settings:
    # ---- embeddings -------------------------------------------------------
    # fastembed  = local ONNX models, downloaded once from Hugging Face (default)
    # openai     = OpenAI embeddings API (needs OPENAI_API_KEY)
    # wordllama  = tiny model shipped inside the pip package (works fully offline)
    # hashing    = toy keyword "embedding" with no model at all (for tests)
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "fastembed").lower())
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    # ---- reranking --------------------------------------------------------
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))

    # ---- retrieval behaviour ---------------------------------------------
    # If the best vector similarity is below this, we refuse to answer.
    # None = no threshold. Lab 09 shows you how to pick a value for YOUR model.
    min_similarity: float | None = field(default_factory=lambda: _env_float("MIN_SIMILARITY", None))

    # ---- LLM --------------------------------------------------------------
    # If llm provider vendor is explicitly not provided, detect it from api keys using _auto_provider
    llm_provider: str = "ollama" if "--local" in sys.argv else field(default_factory=_auto_provider)
    default_llm: str = field(default_factory=lambda: _env("LLM_MODEL"))
    llm_temperature: float | None = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", None))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "700")))
    reasoning_effort: str = field(default_factory=lambda: _env("LLM_REASONING_EFFORT", "none"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    local_transformers_model: str = field(default_factory=lambda: _env("LOCAL_TRANSFORMER_MODEL", DEFAULT_LOCAL_TRANSFORMERS_MODEL))

    def __post_init__(self) -> None:
        # IF default LLM model is explicitly not provided, detect it from llm provider vendor
        if not self.default_llm and self.llm_provider in DEFAULT_MODELS:
            self.default_llm = DEFAULT_MODELS[self.llm_provider]
        # for d in (self.processed_dir, self.log_dir, self.reports_dir):
        #     d.mkdir(parents=True, exist_ok=True)


def get_settings(**overrides: Any) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    s.__post_init__()
    return s
