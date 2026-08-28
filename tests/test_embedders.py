from __future__ import annotations

import math
import sys
import types

import httpx

from memory_sdk import HashEmbedder, OpenAICompatibleEmbedder, resolve_embedder
from memory_sdk.embedders import SentenceTransformerEmbedder, cosine, hash_embedding, tokens


def test_tokens_cover_words_and_cjk_grams():
    toks = tokens("Hello 世界")
    assert "hello" in toks
    assert "世" in toks and "界" in toks and "世界" in toks


def test_hash_embedding_is_deterministic_and_normalized():
    a = hash_embedding("user likes tea")
    b = hash_embedding("user likes tea")
    assert a == b
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-6)
    assert cosine(a, b) > 0.999


async def test_hash_embedder_is_not_semantic():
    embedder = HashEmbedder()
    assert embedder.semantic is False
    assert embedder.ready is True
    assert len(await embedder.encode("x")) == embedder.dims


def test_resolve_hash_and_explicit_model():
    assert isinstance(resolve_embedder("hash"), HashEmbedder)
    st = resolve_embedder("some/model")
    assert isinstance(st, SentenceTransformerEmbedder)
    assert st.name == "some/model"
    assert st.ready is False  # lazy — nothing loads at resolve time


def test_resolve_auto_falls_back_without_sentence_transformers(monkeypatch):
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert isinstance(resolve_embedder(None), HashEmbedder)


def test_resolve_auto_prefers_sentence_transformers_when_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.ModuleType("sentence_transformers"))
    assert isinstance(resolve_embedder(None), SentenceTransformerEmbedder)


async def test_openai_embedder_normalizes_and_names():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [3.0, 4.0]}]})

    embedder = OpenAICompatibleEmbedder(
        "http://test/v1", model="embed-1", transport=httpx.MockTransport(handler)
    )
    vector = await embedder.encode("anything")
    assert vector == [0.6, 0.8]  # L2-normalized
    assert embedder.name == "openai:embed-1"
    assert embedder.semantic is True
    await embedder.aclose()
