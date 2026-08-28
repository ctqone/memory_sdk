"""Embedding backends.

Three built-ins share one small async interface:

- :class:`HashEmbedder` — zero-dependency fallback. Deterministic token
  hashing; its cosine carries lexical overlap, NOT meaning. Always available,
  and flagged ``semantic = False`` so features that need real semantics (the
  consolidation novelty gate) can refuse to run on it instead of misfiring.
- :class:`SentenceTransformerEmbedder` — local CPU model via the optional
  ``sentence-transformers`` dependency (``pip install memory-sdk[local]``).
  Heavy models load lazily off the event loop; until resident, query-time
  callers fall back to the hash embedder so recall never blocks on a model
  download.
- :class:`OpenAICompatibleEmbedder` — any ``/v1/embeddings`` endpoint
  (OpenAI, llama.cpp server, vLLM, Ollama, LM Studio, ...).

Every stored vector is tagged with the ``name`` of the backend that produced
it, and cosine is only ever computed between vectors from the same backend —
switching backends gracefully degrades comparisons to the lexical leg instead
of comparing incompatible vector spaces.

All backends return L2-normalized vectors, so cosine == dot product.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re

import httpx

from .config import DEFAULT_ST_MODEL

logger = logging.getLogger("memory_sdk")

# Word tokens plus CJK handling: individual CJK chars and their 2/3-grams, so
# languages without spaces still produce overlappable token sets.
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[぀-ヿ㐀-鿿]")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿]")


def tokens(text: str) -> list[str]:
    lowered = text.lower()
    toks = _TOKEN_RE.findall(lowered)
    compact_cjk = "".join(_CJK_RE.findall(lowered))
    for size in (2, 3):
        toks.extend(compact_cjk[idx : idx + size] for idx in range(max(0, len(compact_cjk) - size + 1)))
    return [token for token in toks if token]


def hash_embedding(text: str, dims: int = 384) -> list[float]:
    vector = [0.0] * dims
    for token in tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % dims
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine between two vectors from the SAME backend (all backends
    normalize, so this is a plain dot product)."""
    # strict=False on purpose: callers length-check comparable vectors first,
    # and the one place lengths could differ (a foreign row slipping through)
    # should degrade to a low score, not raise mid-recall.
    return sum(a * b for a, b in zip(left, right, strict=False))


class Embedder:
    """Base interface. Subclass to plug in your own backend.

    - ``name`` tags stored vectors; two vectors compare only when their names
      match, so give each distinct model a distinct name.
    - ``semantic`` declares whether cosine over this backend's vectors carries
      meaning. Leave True for real models; the hash fallback sets it False.
    - ``ready`` must answer instantly: query-time embedding never blocks on a
      model load, it falls back for that call instead.
    - ``load()`` may block (it is awaited off the hot path) and returns whether
      the backend is usable; a permanently failed backend returns False rather
      than raising every call.
    """

    name: str = "none"
    semantic: bool = True

    @property
    def ready(self) -> bool:
        return True

    async def load(self) -> bool:
        return True

    async def encode(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbedder(Embedder):
    semantic = False

    def __init__(self, dims: int = 384) -> None:
        self.dims = dims
        self.name = f"hash-{dims}"

    async def encode(self, text: str) -> list[float]:
        return hash_embedding(text, self.dims)


class SentenceTransformerEmbedder(Embedder):
    """Lazy sentence-transformers backend (CPU). Loads on first use under a
    lock, off the event loop; if the import or model load fails it permanently
    latches into a failed state so callers fall back to the hash embedder
    rather than retrying the load on every call."""

    def __init__(self, model_name: str = DEFAULT_ST_MODEL) -> None:
        self.name = model_name
        self._model = None
        self._failed = False
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    async def load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        async with self._lock:
            if self._model is not None:
                return True
            if self._failed:
                return False

            def _load_sync():
                from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                    SentenceTransformer,
                )

                return SentenceTransformer(self.name, device="cpu")

            try:
                self._model = await asyncio.to_thread(_load_sync)
            except Exception as exc:  # noqa: BLE001 - any failure -> hash fallback
                self._failed = True
                logger.warning("embedding model %r unavailable (%s); using hash fallback", self.name, exc)
                return False
        return True

    async def encode(self, text: str) -> list[float]:
        if not await self.load():
            raise RuntimeError("embedding model not loaded")
        model = self._model
        vector = await asyncio.to_thread(model.encode, text, normalize_embeddings=True)  # type: ignore[union-attr]
        return [float(value) for value in vector]


class OpenAICompatibleEmbedder(Embedder):
    """Any OpenAI-compatible ``/v1/embeddings`` endpoint.

    The timeout is deliberately short: query-time recall runs through this on
    the hot path, and a downed endpoint should degrade recall to the lexical
    leg (via the caller's fallback), not hang the turn.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = f"openai:{model}"
        self.model = model
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

    async def encode(self, text: str) -> list[float]:
        client = self._ensure_client()
        response = await client.post("/embeddings", json={"model": self.model, "input": text})
        response.raise_for_status()
        data = response.json()
        vector = [float(v) for v in data["data"][0]["embedding"]]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def resolve_embedder(spec: str | None) -> Embedder:
    """Resolve ``MemoryConfig.embedder`` into a backend.

    ``None`` -> sentence-transformers with :data:`DEFAULT_ST_MODEL` if the
    package is installed, else the hash fallback with a loud warning (hash
    cosine is not semantics — recall degrades to lexical-ish and the novelty
    gate disables itself). ``"hash"``/``"none"`` -> hash. Anything else -> that
    sentence-transformers model id. An explicit :class:`Embedder` instance
    passed to ``Memory(embedder=...)`` bypasses this entirely.
    """
    spec = (spec or "").strip()
    if spec.lower() in ("hash", "none"):
        return HashEmbedder()
    if spec:
        return SentenceTransformerEmbedder(spec)
    try:
        import sentence_transformers  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed; falling back to the hash embedder. "
            "Recall will be lexical-only and the novelty gate stays off. "
            "Install memory-sdk[local] (or pass an Embedder) for semantic recall."
        )
        return HashEmbedder()
    return SentenceTransformerEmbedder(DEFAULT_ST_MODEL)
