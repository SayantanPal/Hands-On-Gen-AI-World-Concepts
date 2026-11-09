from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    tool_calls: list[dict] = field(default_factory=list)
