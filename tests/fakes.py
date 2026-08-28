"""Shared test doubles.

The embedder fakes make cosine controllable so scoring paths can be isolated
without downloading a model; the LLM fakes script completions per call.
"""

from __future__ import annotations

import json

from memory_sdk import Embedder, LLMError


class ConstEmbedder(Embedder):
    """Every text maps to the same unit vector, so cosine is constant (1.0)
    and ranking is driven purely by recency/importance/lexical terms."""

    name = "const"

    async def encode(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class MapEmbedder(Embedder):
    """Returns a caller-supplied vector per exact text so cosine is
    controllable; unknown texts land on an orthogonal default."""

    name = "map"

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    async def encode(self, text: str) -> list[float]:
        return self.mapping.get(text.strip(), [0.0, 0.0, 1.0])


class ScriptedLLM:
    """Returns a queued response per complete() call (FIFO); repeats the last."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    async def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


class FailingLLM:
    """Every completion fails, the way a downed endpoint does."""

    async def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        raise LLMError("scripted failure")


class RecordingLLM:
    """Captures the user message of every call; replies from a script."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.captured: list[str] = []
        self.calls = 0

    async def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        self.calls += 1
        self.captured.append(messages[1]["content"])
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def fact_json(text, subject="user", importance=7, scope="user", entities=None) -> str:
    return json.dumps(
        [
            {
                "text": text,
                "subject": subject,
                "importance": importance,
                "scope": scope,
                "entities": entities or [],
            }
        ]
    )
