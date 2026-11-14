from __future__ import annotations


def build_rctfc_system_prompt(
    *,
    role: str = "",
    task: str = "",
    rules: str = "",
    context: str = "",
    output_format: str = "",
    constraints: list[str] | None = None,
) -> str:
    """Build a system prompt string following the RCTFC scaffold.

    Role, Task, Context, Format, and Constraints
    are optional — sections are omitted when empty.

    Every field can contain {placeholders} for ChatPromptTemplate to fill
    at invoke-time.

    Args:
        role:          Who the LLM should act as.
        task:          What the LLM must do.
        context:       Background information the LLM needs (optional).
        output_format: How the answer should be structured (optional).
        constraints:   Guardrails / things the LLM must NOT do (optional).
    """
    parts = []
    if role:
        parts.append(f"[PERSONNA ROLE]\nYou are {role}.")
    if context:
        parts.append(f"[BACKGROUND CONTEXT]\n{context}")
    if task:
        parts.append(f"[TASK INSTRUCTIONS]\n{task}")
    if output_format:
        parts.append(f"[FORMAT]\n{output_format}")
    if rules:
        parts.append(f"[RULES]\n{rules}")
    if constraints:
        rules = "\n".join(f"- {c}" for c in constraints)
        parts.append(f"[GUARDRAILS CONSTRAINTS]\n{rules}")

    return "\n\n".join(parts)
