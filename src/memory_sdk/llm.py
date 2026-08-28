"""LLM backends.

The library needs an LLM for the consolidation stages (extract / summarize /
reflect / condense). Two ways to provide one:

- :class:`OpenAICompatibleLLM` — the built-in client for any
  ``/v1/chat/completions`` endpoint (OpenAI, llama.cpp server, vLLM, Ollama,
  LM Studio, OpenRouter, ...). Works out of the box with just a URL.
- Implement :class:`LLMBackend` yourself — a single ``async complete()``
  returning the reply text — to use any other provider or to route through
  your app's existing client, scheduler, retries, or fallbacks.

Failure is signalled by raising :class:`~memory_sdk.errors.LLMError`, never by
returning sentinel text: consolidation treats a raised ``LLMError`` as "this
stage produced nothing" and moves on, so fallback prose can never leak into
the memory store as if the model had said it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

import httpx

from .errors import LLMError

logger = logging.getLogger("memory_sdk")

Message = dict[str, str]


@runtime_checkable
class LLMBackend(Protocol):
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant reply text for ``messages`` (OpenAI-style
        ``{"role", "content"}`` dicts). Raise :class:`LLMError` on failure."""
        ...


class OpenAICompatibleLLM:
    """Minimal async client for OpenAI-compatible chat-completion endpoints.

    Bounded retries with exponential backoff; a generous default timeout
    because consolidation often runs against local models that are slow but
    not stuck. There is deliberately no fallback text and no streaming — this
    client serves background consolidation, not an interactive chat surface.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = max(0, retries)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport  # injectable for tests (httpx.MockTransport)
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url, headers=headers, timeout=self._timeout,
                transport=self._transport,
            )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        client = self._ensure_client()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                # 1s, 2s, 4s, ... — waiting is cheap for a background stage.
                await asyncio.sleep(2 ** (attempt - 1))
            try:
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                text = (data["choices"][0]["message"].get("content") or "").strip()
                if not text:
                    raise LLMError("LLM returned an empty completion")
                return text
            except LLMError as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                # 4xx (except 408/429) will not get better by retrying.
                code = exc.response.status_code
                if 400 <= code < 500 and code not in (408, 429):
                    break
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
        raise LLMError(f"LLM completion failed after {self.retries + 1} attempt(s): {last_error}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenAICompatibleLLM:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
