"""Turn a recall result into prompt-ready text.

Every consumer ends up writing this; here is one honest default. Ordering is
deliberate: synthesis rows (summary, insight) come FIRST because hosts that
trim prompts to a token budget shed from the tail — the reserved synthesis
slots are pointless if they are also the first thing trimmed. Facts follow in
recall order (best first), then the entity-graph expansion under its own
heading. Roll your own renderer if your prompt wants a different shape; the
data is all on :class:`~memory_sdk.models.RecallResult`.
"""

from __future__ import annotations

from .models import RecallResult

_TYPE_LABELS = {"summary": "(summary) ", "insight": "(insight) "}


def render_memories(
    result: RecallResult,
    *,
    title: str = "Long-term memory (most relevant first):",
    graph_title: str = "Related context:",
) -> str:
    """Render ``result`` as bulleted prompt text; empty string when there is
    nothing to say (so it contributes no prompt noise)."""
    if not result:
        return ""
    ordered = (
        [m for m in result.memories if m.mem_type == "summary"]
        + [m for m in result.memories if m.mem_type == "insight"]
        + [m for m in result.memories if m.mem_type not in ("summary", "insight")]
    )
    lines = [title]
    lines.extend(f"- {_TYPE_LABELS.get(m.mem_type, '')}{m.content}" for m in ordered)
    if result.graph:
        lines.append(graph_title)
        lines.extend(f"- {m.content}" for m in result.graph)
    return "\n".join(lines)
