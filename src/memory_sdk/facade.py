"""The :class:`Memory` client — the one object most applications need.

.. code-block:: python

    from memory_sdk import Memory, MemoryConfig, OpenAICompatibleLLM

    llm = OpenAICompatibleLLM("http://127.0.0.1:8000/v1", model="my-model")
    async with Memory(MemoryConfig(db_path="agent.sqlite3"), llm=llm) as mem:
        await mem.add("user-42", "I moved to Osaka last month", "Nice!")
        result = await mem.search("user-42", "where does the user live?")
        prompt_block = mem.render(result)

``scope`` is an opaque string naming who a memory belongs to; compose it
however your identity model works (``"user-42"``, ``"alice__support-bot"``,
...). Two scopes never see each other's memories.

Consolidation runs on a background asyncio task that starts with the first
``add()`` — that task IS the product, so it is on by default. Shut down
cleanly (``async with``, or ``await aclose()``) so buffered exchanges are
drained rather than lost; hosts that want manual control never call ``add()``
and drive :attr:`consolidator` directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .config import MemoryConfig
from .consolidation import Consolidator
from .embedders import Embedder, resolve_embedder
from .errors import ConfigurationError
from .llm import LLMBackend
from .models import MemoryItem, RecallResult
from .render import render_memories
from .store import SQLiteMemoryStore


class Memory:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        llm: LLMBackend | None = None,
        embedder: Embedder | None = None,
        turn_gate: Callable[[], object] | None = None,
    ) -> None:
        """``llm`` powers consolidation (extract/summarize/reflect/condense);
        without one the read/write API still works but ``add()`` refuses.
        ``embedder`` overrides ``config.embedder`` resolution entirely.
        ``turn_gate`` is entered around every consolidation LLM call — pass a
        factory for your scheduler/rate-limiter context manager if LLM access
        is contended in your app."""
        self.config = config or MemoryConfig()
        self._llm = llm
        self.store = SQLiteMemoryStore(
            self.config.db_path,
            self.config,
            embedder if embedder is not None else resolve_embedder(self.config.embedder),
        )
        self.consolidator = Consolidator(self.config, self.store, llm, turn_gate)

    # -- lifecycle -----------------------------------------------------------
    async def open(self) -> Memory:
        """Open (and create/migrate) the database. Idempotent; every public
        method calls it, so explicit use is optional."""
        await self.store.open()
        return self

    async def aclose(self, drain_timeout: float = 30.0) -> None:
        """Drain consolidation (bounded by ``drain_timeout``) and close."""
        await self.consolidator.aclose(drain_timeout)
        await self.store.close()

    async def __aenter__(self) -> Memory:
        return await self.open()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- the write path --------------------------------------------------------
    async def add(
        self, scope: str, user_text: str, assistant_text: str = "", *, session_id: str = ""
    ) -> None:
        """Hand one finished exchange to background consolidation.

        Returns immediately — extraction happens on the worker task, batched.
        ``session_id`` identifies the conversation: exchanges batch within a
        session, and a session switch flushes the previous session's buffer so
        two conversations are never extracted as one run of "consecutive"
        exchanges. Pass it if your app has multiple conversations per scope.
        """
        if self._llm is None:
            raise ConfigurationError(
                "add() consolidates through an LLM — construct Memory(llm=...), "
                "or write directly with remember()"
            )
        await self.store.open()
        self.consolidator.start()
        self.consolidator.enqueue(scope, user_text, assistant_text, session_id)

    async def remember(
        self,
        scope: str,
        content: str,
        *,
        mem_type: str = "fact",
        importance: float = 5.0,
        subject: str = "",
    ) -> MemoryItem:
        """Write one memory row directly, bypassing extraction (imports,
        user-stated facts, app events). To change a fact, ``remember()`` the
        new version and ``supersede()`` the old id — supersession, not
        mutation, is the model's way to change."""
        await self.store.open()
        return await self.store.add(
            scope, content, mem_type=mem_type, importance=importance, subject=subject
        )

    async def supersede(self, memory_id: str) -> None:
        await self.store.open()
        await self.store.supersede(memory_id)

    # -- the read path ----------------------------------------------------------
    async def search(
        self, scope: str, query: str, *, include_superseded: bool = False
    ) -> RecallResult:
        """Slot-reserved recall plus 1-hop entity-graph expansion.

        Slots come from ``config.recall`` (``top_k`` facts + ``summary_slots``
        + ``insight_slots``); the expansion is seeded from the recalled facts'
        subjects only (a summary's subject is "session" and an insight's is
        "insight" — neither is an entity).
        """
        await self.store.open()
        recall = self.config.recall
        items = await self.store.search(
            query,
            scope,
            include_superseded=include_superseded,
            per_type_limits={
                "fact": recall.top_k,
                "summary": recall.summary_slots,
                "insight": recall.insight_slots,
            },
        )
        graph: list[MemoryItem] = []
        if recall.graph_enabled and recall.graph_expand_limit > 0:
            subjects = list(
                {item.subject for item in items if item.mem_type == "fact" and item.subject}
            )
            if subjects:
                recalled_ids = {item.id for item in items}
                graph = [
                    item
                    for item in await self.store.graph_expand(
                        scope, subjects, recall.graph_expand_limit
                    )
                    if item.id not in recalled_ids
                ]
        return RecallResult(query=query, memories=items, graph=graph)

    def render(self, result: RecallResult) -> str:
        """Prompt-ready text for a recall result (see :func:`render_memories`)."""
        return render_memories(result)

    async def list(
        self,
        scope: str,
        *,
        limit: int = 40,
        mem_types: tuple[str, ...] = ("fact", "summary", "insight"),
        include_superseded: bool = False,
        since: float | None = None,
    ) -> list[MemoryItem]:
        """Most recent rows, oldest-first (inspection / UI surfaces)."""
        await self.store.open()
        return await self.store.recent(
            scope, limit=limit, mem_types=mem_types,
            include_superseded=include_superseded, since=since,
        )

    async def entities(self, scope: str, limit: int = 100) -> list[dict[str, object]]:
        await self.store.open()
        return await self.store.entities(scope, limit=limit)

    async def related(self, scope: str, subject: str, limit: int = 4) -> list[MemoryItem]:
        """Facts about entities one hop from ``subject`` in the graph."""
        await self.store.open()
        return await self.store.graph_expand(scope, [subject], limit=limit)

    # -- deletion -----------------------------------------------------------------
    async def delete(self, scope: str, ids: Iterable[str]) -> int:
        """Hard-delete rows (``secure_delete`` erasure, not soft-hide)."""
        await self.store.open()
        return await self.store.delete(scope, ids)

    async def clear(self, scope: str, *, include_core: bool = False) -> int:
        """Erase a scope: memories, entities, edges, counters, and any
        un-extracted buffered exchanges (they must not surface as memories
        minutes after the user erased the conversation). Core memory is kept
        unless ``include_core`` — it is user-visible state with its own
        lifecycle."""
        await self.store.open()
        self.consolidator.discard(scope)
        deleted = await self.store.clear(scope)
        if include_core:
            await self.store.clear_core(scope)
        return deleted

    def discard_buffer(self, scope: str, session_id: str | None = None) -> int:
        """Drop buffered, not-yet-extracted exchanges for a session (or the
        whole scope). Call when the host deletes a conversation."""
        return self.consolidator.discard(scope, session_id)

    # -- core memory ---------------------------------------------------------------
    async def get_core(self, scope: str) -> dict[str, str]:
        await self.store.open()
        return await self.store.get_core_blocks(scope)

    async def set_core(self, scope: str, blocks: dict[str, object]) -> dict[str, str]:
        """Merge ``blocks`` (only configured block names) over what's stored."""
        await self.store.open()
        return await self.store.set_core_blocks(scope, blocks)

    async def render_core(self, scope: str) -> str:
        """Core memory rendered for a system prompt; empty string when blank."""
        await self.store.open()
        return self.store.render_core(await self.store.get_core_blocks(scope))

    async def clear_core(self, scope: str) -> None:
        await self.store.open()
        await self.store.clear_core(scope)

    # -- maintenance ---------------------------------------------------------------
    async def flush(self, timeout: float = 60.0) -> bool:
        """Push every buffered exchange through consolidation and wait (up to
        ``timeout``). Returns True when fully drained — call before shutdown
        checkpoints, or after a burst of ``add()`` in tests."""
        await self.store.open()
        return await self.consolidator.drain(timeout)

    def status(self) -> dict[str, object]:
        """Live consolidation/gate counters plus which embedder is active."""
        status = self.consolidator.status()
        status["embedder"] = self.store.embedder.name
        status["embedder_ready"] = self.store.embedder.ready
        status["embedder_semantic"] = self.store.embedder.semantic
        return status
