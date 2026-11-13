from __future__ import annotations

import time

from dotenv import load_dotenv
load_dotenv()

from config.settings import Settings, get_settings
from gen_ai_core.models.result import LLMResult
from gen_ai_core.prompts.prompt import Prompt

import torch
import transformers


class TransformersLLM:

    def __init__(self, model: str | None = None, purpose: str = "text-generation", settings: Settings | None = None, max_new_tokens: int = 1024):
        self.s = settings or get_settings()
        self._model = model or self.s.local_transformers_model
        self._purpose = purpose
        self._max_new_tokens = max_new_tokens
        self._pipe = self._build_pipeline()
        print(f"TransformersLLM ready: {self._model}")

    def _build_pipeline(self):

        return transformers.pipeline(
            self._purpose,
            model=self._model,                              # downloaded to YOUR machine — nothing leaves it
            model_kwargs={"torch_dtype": torch.float16},    # half-precision numbers → fits on a 16 GB GPU
            device_map="auto",                              # place it on the GPU automatically
        )

    def get_model(self) -> str:
        return self._model

    def get_simple_text_label_classifier_model(self, classifier_model_name: str="ProsusAI/finbert"):
        import transformers

        return transformers.pipeline("text-classification", model=classifier_model_name)

    def __repr__(self) -> str:
        return f"TransformersLLM({self._model})"

    def generate_response_content(
        self,
        prompt: Prompt,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        messages = prompt.to_role_dicts()

        gen_kwargs: dict = {"max_new_tokens": max_tokens or self._max_new_tokens}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        outputs = self._pipe(messages, **gen_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        # outputs[0] — the pipeline returns one result per input; we sent one conversation, so take the first.
        # ["generated_text"] — this holds the whole chat: your system message, your user message, and the model's new reply at the end.
        # [-1]["content"] — -1 means "last item", i.e. the model's reply; ["content"] is its text
        text = outputs[0]["generated_text"][-1]["content"]

        return LLMResult(
            text=(text or "").strip(),
            latency_ms=latency_ms,
            model=self._model,
            provider="local-transformers",
        )
