"""The built-in OpenAI-compatible client, exercised against a mock transport."""

from __future__ import annotations

import json

import httpx
import pytest

from memory_sdk import LLMError, OpenAICompatibleLLM


def _reply(text: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": text}}]}
    )


async def test_success_returns_text_and_sends_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        seen["auth"] = request.headers.get("Authorization")
        return _reply("hello there")

    llm = OpenAICompatibleLLM(
        "http://test/v1", model="test-model", api_key="sk-x",
        transport=httpx.MockTransport(handler),
    )
    async with llm:
        text = await llm.complete([{"role": "user", "content": "hi"}], temperature=0.1)
    assert text == "hello there"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["payload"]["model"] == "test-model"
    assert seen["payload"]["temperature"] == 0.1
    assert seen["auth"] == "Bearer sk-x"


async def test_empty_completion_raises_llm_error():
    llm = OpenAICompatibleLLM(
        "http://test/v1", model="m", retries=0,
        transport=httpx.MockTransport(lambda request: _reply("")),
    )
    async with llm:
        with pytest.raises(LLMError):
            await llm.complete([{"role": "user", "content": "hi"}])


async def test_client_errors_do_not_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    llm = OpenAICompatibleLLM(
        "http://test/v1", model="m", retries=3, transport=httpx.MockTransport(handler)
    )
    async with llm:
        with pytest.raises(LLMError):
            await llm.complete([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # a 4xx will not get better by retrying


async def test_server_errors_retry_then_succeed():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return _reply("recovered")

    llm = OpenAICompatibleLLM(
        "http://test/v1", model="m", retries=1, transport=httpx.MockTransport(handler)
    )
    async with llm:
        assert await llm.complete([{"role": "user", "content": "hi"}]) == "recovered"
    assert calls["n"] == 2
