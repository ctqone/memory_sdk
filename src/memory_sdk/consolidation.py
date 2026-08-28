"""Background memory consolidation.

Instead of logging every exchange verbatim, a single background asyncio task
pulls finished exchanges off a queue and runs the LLM to extract durable
atomic facts, deduplicate / supersede them against what's already stored,
maintain the Tier-1 core-memory blocks (condensing a block once it outgrows
its cap, rather than dropping its oldest — i.e. most foundational — lines),
periodically roll the single live summary forward, and draw higher-level
reflections. It also grows the minimal entity graph.

Three constraints shape the design (all inherited from long production use):

* The caller's turn is never blocked — extraction happens after the exchange
  is handed over, on the worker task.
* LLM turns are the scarce resource, so as few as possible are spent.
  Exchanges are BUFFERED and extracted in batches rather than one call per
  turn (a turn still inside the host's chat-history window is already visible
  to its model verbatim, so extracting it immediately is redundant); a
  CPU-only novelty gate drops near-repeat banter before it can cost a call;
  and the summary / reflection stages decide whether they have anything to do
  BEFORE spending a turn.
* Hosts that schedule LLM access (a GPU queue, a rate limiter) can wrap every
  consolidation completion by passing ``turn_gate`` — a callable returning an
  async context manager that is entered around each LLM call.

Because buffering holds work outside the queue, ``pending()`` counts buffered
exchanges too and ``drain()`` flushes them first — otherwise graceful shutdown
would report success while the tail of the session sat unsaved in RAM.

Concurrency note: everything here runs on one event loop. Buffer and counter
mutations happen with no ``await`` in between, so they need no locks — keep it
that way when editing (an ``await`` inserted into ``enqueue()`` or
``discard()`` would reopen the races the ancestor code needed locks for).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections import deque
from collections.abc import Callable

from .config import MemoryConfig
from .embedders import cosine
from .errors import LLMError
from .llm import LLMBackend
from .store import SQLiteMemoryStore, norm_subject

logger = logging.getLogger("memory_sdk")

_now = time.time

# Novelty gate: how many recent user utterances to remember per scope, and how
# many scopes to keep at all. Held in RAM only: losing the cache on restart
# costs at most a few redundant extractions, and persisting it would be a
# schema change for a pure optimization.
_GATE_CACHE_SIZE = 200
_GATE_MAX_SCOPES = 32


def _parse_json_array(text: str) -> list:
    """Tolerantly extract a JSON array from a (possibly chatty /
    trailing-comma'd) model response — small and quantized models are messy."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    raw = match.group(0)
    for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue
    return []


def _normalize_facts(data: list) -> list[dict]:
    out: list[dict] = []
    for element in data:
        if not isinstance(element, dict):
            continue
        text = str(element.get("text", "")).strip()
        if not text:
            continue
        try:
            importance = float(element.get("importance", 5))
        except (TypeError, ValueError):
            importance = 5.0
        importance = max(1.0, min(importance, 10.0))
        entities = element.get("entities", [])
        if isinstance(entities, list):
            entities = [str(e).strip() for e in entities if str(e).strip()]
        else:
            entities = []
        out.append(
            {
                "text": text,
                "subject": str(element.get("subject", "")).strip() or "user",
                "importance": importance,
                "scope": str(element.get("scope", "")).strip().lower(),
                "entities": entities,
            }
        )
    return out


def _crossed(count: int, delta: int, every: int) -> bool:
    """True when this bump moved the counter across an ``every`` boundary.

    Not ``count % every == 0``: a batch advances the counter by len(batch), so
    the modulo test steps straight over the boundary (with every=8, a batch of
    8 takes the counter 6 -> 14 and never lands on a multiple) and the stage
    silently never fires again.
    """
    if every <= 0 or delta <= 0:
        return False
    return (count - delta) // every != count // every


def _normalize_utterance(text: str) -> str:
    """Canonical form for the gate's exact-duplicate fast path."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _trim_core_block(text: str, cap: int) -> str:
    """Last-resort FIFO trim: drop oldest lines until under cap.

    This is the fallback, not the policy. It evicts the *most foundational*
    facts first — the oldest line in a real block was the user's name — which
    is exactly why LLM condensation runs ahead of it. Kept so the block can
    never grow unbounded when the LLM is unavailable.
    """
    while len(text) > cap and "\n" in text:
        text = text.split("\n", 1)[1]
    return text


class Consolidator:
    """The background worker. Normally owned and driven by
    :class:`memory_sdk.Memory`; usable directly for scripts and backfills via
    :meth:`consolidate`."""

    def __init__(
        self,
        config: MemoryConfig,
        store: SQLiteMemoryStore,
        llm: LLMBackend | None,
        turn_gate: Callable[[], object] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.llm = llm
        self._turn_gate = turn_gate or contextlib.nullcontext
        self._queue: asyncio.Queue[tuple[str, list[tuple[str, str]]]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        # Exchanges waiting to be batched, keyed by (scope, session_id). The
        # session in the key is what makes a session switch flush: two
        # conversations must not be extracted as one run of "consecutive"
        # exchanges, and the host may have no explicit switch event to hook.
        self._buffers: dict[tuple[str, str], list[tuple[str, str]]] = {}
        # Live activity surfaced via status() (and used by graceful shutdown).
        self._state = "idle"  # idle | extracting | summarizing | reflecting | condensing
        self._last_event = ""
        self._last_active_ts = 0.0
        self._facts_added = 0
        self._summaries_added = 0
        self._reflections_added = 0
        self._core_condensed = 0
        # Exchanges handed to the queue but not yet finished. Tracked
        # separately because queue.qsize() counts batches, and pending() has to
        # answer in exchanges for the shutdown path to be honest.
        self._queued = 0
        # Novelty gate: recent user utterances per scope (vectors + normalized
        # text for the free exact-match leg), plus what it did, for status().
        self._gate_vectors: dict[str, deque] = {}
        self._gate_seen: dict[str, deque] = {}
        self._gate_skipped = 0
        self._gate_would_skip = 0
        self._gate_unavailable = 0
        # (scope, block) -> earliest retry time, so a failing/uncooperative
        # model can't re-attempt core condensation on every single fact.
        self._core_condense_after: dict[tuple[str, str], float] = {}

    # -- status ----------------------------------------------------------------
    def _set_state(self, state: str, event: str | None = None) -> None:
        self._state = state
        if event:
            self._last_event = event
        if state != "idle":
            self._last_active_ts = _now()

    def status(self) -> dict[str, object]:
        """Snapshot of consolidation activity."""
        buffered = self._buffered()
        return {
            "enabled": bool(self.config.consolidation.enabled),
            "state": self._state,
            "busy": self._state != "idle",
            "pending": self._queued + buffered,
            "buffered": buffered,
            "facts_added": self._facts_added,
            "summaries_added": self._summaries_added,
            "reflections_added": self._reflections_added,
            "core_condensed": self._core_condensed,
            "gate_mode": str(self.config.consolidation.gate_mode),
            "gate_skipped": self._gate_skipped,
            "gate_would_skip": self._gate_would_skip,
            "gate_unavailable": self._gate_unavailable,
            "last_event": self._last_event,
            "last_active_ts": self._last_active_ts,
        }

    def _buffered(self) -> int:
        return sum(len(batch) for batch in self._buffers.values())

    def pending(self) -> int:
        """Exchanges not yet consolidated — buffered, queued, or in flight.

        Buffered ones count: they are real unsaved work that the queue cannot
        see, and ``drain()`` (i.e. shutdown) decides whether it is safe to stop
        the process on this number.
        """
        return self._queued + self._buffered()

    async def drain(self, timeout: float = 30.0) -> bool:
        """Flush the buffers, then wait until nothing is buffered, queued or
        mid-processing (or until timeout). Returns True if fully drained.

        Flushing first matters: a partially filled batch is invisible to the
        queue, so without it this would report success while the last few
        exchanges of the session waited for a batch-mate that never comes —
        and the host shuts down on that answer.
        """
        self.flush_buffers()
        if self.pending() > 0:
            self.start()
        deadline = _now() + max(0.0, timeout)
        while self.pending() > 0:
            if _now() >= deadline:
                return False
            await asyncio.sleep(0.05)
        return True

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        """Start the worker task (idempotent; needs a running event loop)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="memory-sdk-consolidation")

    async def aclose(self, drain_timeout: float = 30.0) -> bool:
        """Drain (bounded) and stop the worker. Returns whether the drain
        completed — False means up to ``drain_timeout`` seconds of work was
        abandoned (e.g. the LLM was down)."""
        drained = True
        if self.pending() > 0 or self._task is not None:
            drained = await self.drain(drain_timeout)
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        return drained

    async def consolidate(self, scope: str, user_text: str, assistant_text: str) -> None:
        """Run one exchange through the pipeline right now, bypassing the
        buffer and queue (backfills, scripts, tests)."""
        await self._process_batch(scope, [(user_text, assistant_text)])

    def enqueue(self, scope: str, user_text: str, assistant_text: str, session_id: str = "") -> None:
        """Buffer one finished exchange, submitting a batch when one is due.

        Called on the caller's turn, so it must never block on the LLM: a due
        batch is only handed to the queue, never processed here. (No awaits in
        this method — that is what makes the buffer mutations atomic.)
        """
        if not self.config.consolidation.enabled:
            return
        if not (user_text or assistant_text):
            return
        ready: list[tuple[str, list[tuple[str, str]]]] = []
        key = (scope, str(session_id or ""))
        # Same scope, different session = the user switched conversations.
        for other in [k for k in self._buffers if k[0] == scope and k != key]:
            ready.append((other[0], self._buffers.pop(other)))
        self._buffers.setdefault(key, []).append((user_text, assistant_text))
        if self._flush_due(key):
            ready.append((key[0], self._buffers.pop(key)))
        for batch_scope, exchanges in ready:
            self._submit(batch_scope, exchanges)

    def _flush_due(self, key: tuple[str, str]) -> bool:
        """Whether this buffer should go out now.

        Its own predicate so an idle-age trigger can be added here later
        without touching any call site. An oversized single exchange (a pasted
        document) trips the char cap on its own and goes out alone, which is
        the point — it must not drag seven more exchanges into an already-huge
        prompt.
        """
        buffer = self._buffers.get(key) or []
        if not buffer:
            return False
        cfg = self.config.consolidation
        if len(buffer) >= max(1, int(cfg.batch_size)):
            return True
        max_chars = int(cfg.max_batch_chars or 0)
        if max_chars > 0 and sum(len(u) + len(a) for u, a in buffer) >= max_chars:
            return True
        return False

    def discard(self, scope: str, session_id: str | None = None) -> int:
        """Drop buffered exchanges from a session (or a whole scope) the host
        just erased. Returns how many were dropped.

        Privacy companion to the store's hard deletes: without this, a deleted
        conversation's un-extracted tail would still be sitting in RAM and
        would be extracted into long-term memory minutes AFTER the user erased
        it. Only the buffer is touched — batches already handed to the queue
        carry exchanges the host's model has responded to, which is the
        erasure boundary.
        """
        if session_id is not None:
            return len(self._buffers.pop((scope, str(session_id)), []))
        dropped = 0
        for key in [k for k in self._buffers if k[0] == scope]:
            dropped += len(self._buffers.pop(key))
        return dropped

    def flush_buffers(self) -> None:
        """Hand every buffered exchange to the queue, however partial."""
        ready = [(scope, batch) for (scope, _session), batch in self._buffers.items() if batch]
        self._buffers.clear()
        for scope, exchanges in ready:
            self._submit(scope, exchanges)

    def _submit(self, scope: str, exchanges: list[tuple[str, str]]) -> None:
        if not exchanges:
            return
        self._queued += len(exchanges)
        self._queue.put_nowait((scope, exchanges))

    async def _loop(self) -> None:
        while True:
            scope, exchanges = await self._queue.get()
            try:
                await self._process_batch(scope, exchanges)
            except Exception:  # noqa: BLE001 - never let a bad batch kill the worker
                logger.exception("consolidation failed for scope %r", scope)
            finally:
                self._queued -= len(exchanges)
                self._queue.task_done()

    # -- LLM access --------------------------------------------------------------
    async def _complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        """One gated LLM completion. Raises LLMError on failure (including "no
        LLM configured") — callers treat that as "the stage produced nothing"."""
        if self.llm is None:
            raise LLMError("consolidation needs an LLM backend (Memory(llm=...))")
        async with self._turn_gate():  # type: ignore[attr-defined]
            return await self.llm.complete(messages, temperature=temperature)

    # -- pipeline ----------------------------------------------------------------
    async def _process_batch(self, scope: str, exchanges: list[tuple[str, str]]) -> None:
        if not self.config.consolidation.enabled:
            return
        if not exchanges:
            return
        turns = len(exchanges)
        cfg = self.config.consolidation
        try:
            self._set_state("extracting")
            # Gating is embedder-only — before the LLM, so a batch of pure
            # banter costs no completion at all.
            kept = await self._gate(scope, exchanges)
            if kept:
                facts = await self._extract(kept)
                for fact in facts:
                    if await self._store_fact(scope, fact):
                        self._facts_added += 1

            # Restart-persistent exchange counter (stored in the DB) so cadence
            # like "summarize every N turns" survives process restarts. Counts
            # every exchange, including gated-out ones: the turns happened, and
            # the cadence is meant to track conversation length.
            count = await self.store.bump_counter(scope, "consolidation_turns", turns)
            # Both stages work out whether they have anything to do BEFORE
            # spending an LLM turn. These reads are plain SQLite; paying a
            # completion (and its gate wait) only to discover the summary is
            # already current is pure waste.
            if _crossed(count, turns, int(cfg.summary_every_turns or 0)):
                inputs = await self._summary_inputs(scope)
                if inputs is not None:
                    self._set_state("summarizing")
                    if await self._summarize(scope, inputs):
                        self._summaries_added += 1
                        self._last_event = "summarized"
            if _crossed(count, turns, int(cfg.reflection_every_turns or 0)):
                facts_to_reflect = await self._reflect_inputs(scope)
                if facts_to_reflect:
                    self._set_state("reflecting")
                    added = await self._reflect(scope, facts_to_reflect)
                    if added:
                        self._reflections_added += added
                        self._last_event = "reflected"
        finally:
            self._set_state("idle")

    # -- novelty gate ------------------------------------------------------------
    def _semantic_embedder(self):
        """The store's embedder, but only when its vectors actually carry
        meaning.

        The store silently falls back to the hash embedder when the semantic
        one is missing or still loading (and a failed model latches into its
        failed state), so ``ready`` alone proves nothing. Hash-vector cosine is
        noise; gating on it would skip at random. Returns None instead, and the
        gate fails open.
        """
        embedder = getattr(self.store, "embedder", None)
        if embedder is None or not getattr(embedder, "ready", False):
            return None
        if not getattr(embedder, "semantic", False):
            return None
        if embedder is getattr(self.store, "fallback", None):
            return None
        return embedder

    async def _gate(self, scope: str, exchanges: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Drop exchanges whose user utterance repeats one already extracted
        from. Casual chat is mostly banter that yields nothing but dedup hits,
        yet each one costs a full LLM call.

        Compared utterance-to-utterance, never utterance-to-stored-fact:
        stored rows are extracted third-person prose ("user lives in Osaka")
        while the input is raw dialogue, so cosine between them never
        approaches a duplicate bar and such a gate would never fire at all.

        In "log" mode nothing is dropped and only the counters move — run that
        way first and read ``gate_would_skip`` against real traffic before
        trusting it.
        """
        mode = self.config.consolidation.gate_mode
        if mode not in ("log", "on"):
            return list(exchanges)
        embedder = self._semantic_embedder()
        if embedder is None:
            self._gate_unavailable += 1
            return list(exchanges)
        threshold = float(self.config.consolidation.gate_similarity_threshold)

        if scope not in self._gate_vectors and len(self._gate_vectors) >= _GATE_MAX_SCOPES:
            oldest = next(iter(self._gate_vectors))
            self._gate_vectors.pop(oldest, None)
            self._gate_seen.pop(oldest, None)
        vectors = self._gate_vectors.setdefault(scope, deque(maxlen=_GATE_CACHE_SIZE))
        seen = self._gate_seen.setdefault(scope, deque(maxlen=_GATE_CACHE_SIZE))

        kept: list[tuple[str, str]] = []
        skipped = 0
        for user_text, assistant_text in exchanges:
            normalized = _normalize_utterance(user_text)
            redundant = False
            if normalized and normalized in seen:
                redundant = True  # free exact-match leg; no embedding needed
            elif normalized:
                try:
                    vector = await embedder.encode(user_text)
                except Exception:  # noqa: BLE001 - a broken embedder must not eat memories
                    self._gate_unavailable += 1
                    return list(exchanges)
                redundant = any(cosine(vector, prior) >= threshold for prior in vectors)
                if not redundant:
                    # Only distinct utterances enter the cache, so the slots
                    # stay meaningful and a chain of near-neighbours can't
                    # drift the notion of "already seen" away from what was
                    # actually extracted.
                    vectors.append(vector)
                    seen.append(normalized)
            if redundant:
                skipped += 1
                if mode == "on":
                    continue
            kept.append((user_text, assistant_text))
        if skipped:
            if mode == "on":
                self._gate_skipped += skipped
                self._last_event = f"gate skipped {skipped}"
            else:
                self._gate_would_skip += skipped
        return kept

    # -- extraction --------------------------------------------------------------
    async def _extract(self, exchanges: list[tuple[str, str]]) -> list[dict]:
        if len(exchanges) == 1:
            user_text, assistant_text = exchanges[0]
            body = f"USER: {user_text}\nASSISTANT: {assistant_text}"
        else:
            # Numbered so the model can tell the exchanges apart and still
            # resolve a fact stated across several of them (it sees the arc,
            # which one call per turn structurally could not).
            body = "\n\n".join(
                f"[{index}] USER: {user}\n[{index}] ASSISTANT: {assistant}"
                for index, (user, assistant) in enumerate(exchanges, 1)
            )
        messages = [
            {"role": "system", "content": self.config.prompts.extract},
            {"role": "user", "content": f"{body}\n\nJSON:"},
        ]
        try:
            text = await self._complete(messages, temperature=0.1)
        except LLMError as exc:
            logger.warning("extraction failed: %s", exc)
            return []
        return _normalize_facts(_parse_json_array(text))

    async def _store_fact(self, scope: str, fact: dict, mem_type: str = "fact") -> bool:
        """Store one extracted fact (or insight). Returns True if a new row was
        written (insert or conflict-supersede), False if it only reinforced a
        duplicate.

        Dedup and supersede deliberately have different reach, because they
        differ in blast radius:

        * DEDUP is non-destructive — it bumps an existing row's importance and
          drops the incoming text. So it may cross subjects (the extractor
          labels one entity several ways) and may use the lexical fallback,
          each behind its own stricter bar.
        * SUPERSEDE is destructive — it sets ``valid_to`` and hides a fact
          from recall. So it stays scoped to the same (normalized) subject and
          to cosine only: a fact about a different entity is not a correction,
          and lexical overlap is too noisy to retire something true on.
        """
        text = fact["text"]
        subject = fact["subject"]
        importance = fact["importance"]
        subject_key = norm_subject(subject)
        dedup = self.config.dedup
        dup = float(dedup.dup_threshold)
        conflict = float(dedup.conflict_threshold)
        # Floored at `dup` so the stricter bars can never end up laxer than the
        # same-subject one — and so raising dup_threshold still disables dedup
        # wholesale.
        cross_dup = max(dup, float(dedup.cross_subject_dup_threshold))
        lex_dup = max(dup, float(dedup.lexical_dup_threshold))

        # One unscoped scan, partitioned below — cheaper than two queries, and
        # it lets the supersede pass see same-subject candidates the dedup pass
        # rejected. Rows are score-sorted, but the scores mix two scales
        # (cosine and lexical), so scan them all rather than stopping early.
        matches = await self.store.similar_facts(
            scope, text, subject=subject, limit=5, mem_types=(mem_type,), scope_subject=False
        )

        for match in matches:
            same_subject = norm_subject(match.item.subject) == subject_key
            if match.source == "lexical":
                bar = lex_dup
            else:
                bar = dup if same_subject else cross_dup
            if match.score >= bar:
                await self.store.bump_importance(match.item.id, 1.0)
                if not same_subject:
                    # Rare and worth being able to audit after the fact.
                    logger.info(
                        "dedup across subjects: %r ~ %r (%s %.2f)",
                        subject, match.item.subject, match.source, match.score,
                    )
                return False

        for match in matches:
            if match.score < conflict:
                continue
            if match.source != "cosine" or norm_subject(match.item.subject) != subject_key:
                continue
            await self.store.add(
                scope, text, mem_type=mem_type, importance=importance,
                subject=subject, supersedes=match.item.id,
            )
            await self.store.supersede(match.item.id)
            await self._link_entities(scope, subject, fact["entities"])
            await self._maybe_update_core(scope, fact)
            return True

        await self.store.add(
            scope, text, mem_type=mem_type, importance=importance, subject=subject
        )
        await self._link_entities(scope, subject, fact["entities"])
        await self._maybe_update_core(scope, fact)
        return True

    async def _link_entities(self, scope: str, subject: str, entities: list[str]) -> None:
        if not self.config.recall.graph_enabled:
            return
        if subject:
            await self.store.upsert_entity(scope, subject)
        for name in entities:
            await self.store.upsert_entity(scope, name)
            if subject and name and name.lower() != subject.lower():
                await self.store.add_cooccurrence(scope, subject, name)

    # -- core memory --------------------------------------------------------------
    async def _condense_core_block(self, scope: str, block: str, text: str) -> str | None:
        """LLM-rewrite an over-cap core block into tighter prose keeping every
        distinct durable fact. Returns None on any doubt, so the caller falls
        back to the FIFO trim and the block can never grow unbounded."""
        core = self.config.core
        if not core.condense_enabled:
            return None
        key = (scope, block)
        if _now() < self._core_condense_after.get(key, 0.0):
            return None

        messages = [
            {"role": "system", "content": self.config.prompts.core_condense},
            {
                "role": "user",
                "content": f"Compress to under {core.condense_target} characters:\n\n{text}",
            },
        ]
        previous_state = self._state
        self._set_state("condensing")
        try:
            cleaned = (await self._complete(messages, temperature=0.2)).strip()
        except LLMError:
            self._core_condense_after[key] = _now() + core.condense_cooldown_seconds
            return None
        finally:
            self._set_state(previous_state)

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?|```$", "", cleaned).strip()
        lines = [
            line if line.startswith("- ") else f"- {line}"
            for line in (raw.strip() for raw in cleaned.splitlines())
            if line
        ]
        cleaned = "\n".join(lines)

        # Reject anything suspicious rather than trusting the rewrite: too long
        # to have helped, or so short that facts were dropped instead of merged
        # (a model replying "ok" would otherwise wipe the block).
        if (
            not cleaned
            or len(cleaned) >= len(text)
            or len(cleaned) > core.block_cap
            or len(cleaned) < len(text) * core.condense_min_ratio
        ):
            self._core_condense_after[key] = _now() + core.condense_cooldown_seconds
            return None

        await self.store.snapshot_core_block(scope, block, "precondense")
        self._core_condensed += 1
        self._last_event = "condensed core memory"
        return cleaned

    async def _maybe_update_core(self, scope: str, fact: dict) -> None:
        core = self.config.core
        block = core.scope_map.get(fact.get("scope", ""))
        if not block or block not in core.blocks:
            return
        blocks = await self.store.get_core_blocks(scope)
        existing = blocks.get(block, "")
        if fact["text"].lower() in existing.lower():
            return
        line = f"- {fact['text']}"
        updated = (existing + "\n" + line).strip() if existing else line
        if len(updated) > core.block_cap:
            # Condense the EXISTING block only, then re-append the new line, so
            # the fact we just learned is never inside the condensation input
            # and can't be dropped by the model. If it turns out to be
            # redundant, a later pass merges it away.
            condensed = await self._condense_core_block(scope, block, existing)
            if condensed:
                updated = (condensed + "\n" + line).strip()
            if len(updated) > core.block_cap:
                updated = _trim_core_block(updated, core.block_cap)
        await self.store.set_core_blocks(scope, {block: updated})

    # -- rolling summary ------------------------------------------------------------
    async def _summary_inputs(self, scope: str):
        """``(previous, facts)`` for the summary stage, or None when there is
        nothing new to fold in. Pure SQLite, so the caller can decide whether
        the stage is worth an LLM turn before spending one."""
        previous = await self.store.recent(scope, limit=10, mem_types=("summary",))
        since = previous[-1].timestamp if previous else None
        facts = await self.store.recent(scope, limit=40, mem_types=("fact",), since=since)
        if not facts:
            # Nothing learned since the last summary — rewriting it would only
            # re-compress the same content and burn an LLM turn.
            return None
        if not previous and len(facts) < 3:
            return None
        return previous, facts

    async def _summarize(self, scope: str, inputs=None) -> bool:
        """Roll the single live summary forward.

        The naive design appends a fresh recap every N turns and never retires
        the old one: consecutive summaries overlap heavily and compete with
        real facts for recall slots forever — "rolling" in the design,
        append-only in practice. Here the previous summary is fed INTO the new
        one (so knowledge older than the fact window is carried forward rather
        than lost when the old row is retired) and is then superseded, leaving
        exactly one live summary. Old summaries are superseded, not deleted,
        so the chain stays auditable via ``include_superseded``.

        ``inputs`` is what ``_summary_inputs`` already read; omit it to have
        this look them up itself (direct callers, tests).
        """
        if inputs is None:
            inputs = await self._summary_inputs(scope)
        if inputs is None:
            return False
        previous, facts = inputs
        prev_text = "\n\n".join(item.content for item in previous)
        joined = "\n".join(f"- {item.content}" for item in facts)
        messages = [
            {"role": "system", "content": self.config.prompts.summary},
            {
                "role": "user",
                "content": (
                    f"PREVIOUS SUMMARY:\n{prev_text or '(none yet)'}\n\n"
                    f"FACTS SINCE:\n{joined}\n\nUpdated summary:"
                ),
            },
        ]
        try:
            text = await self._complete(messages, temperature=0.2)
        except LLMError:
            return False
        if not text.strip():
            return False
        # Write the replacement before retiring the originals (same order as
        # _store_fact). A crash between the two leaves an extra live summary,
        # which the next pass folds in and supersedes — self-healing, and there
        # is never a window with no live summary at all.
        await self.store.add(
            scope,
            text.strip()[: self.config.consolidation.summary_max_chars],
            mem_type="summary",
            importance=6.0,
            subject="session",
            # supersedes is a single column, so it records the newest
            # predecessor; every live summary is retired below regardless.
            supersedes=previous[-1].id if previous else None,
        )
        for old in previous:
            await self.store.supersede(old.id)
        return True

    # -- reflection -------------------------------------------------------------------
    async def _reflect_facts(self, scope: str):
        # mem_types=("fact",) keeps insights out of the input — otherwise the
        # reflection stage re-reads its own output every cycle and infers
        # insights from insights, compounding on itself.
        facts = await self.store.recent(scope, limit=40, mem_types=("fact",))
        return facts if len(facts) >= 5 else None

    async def _reflect_inputs(self, scope: str):
        """Facts for the reflection stage, or None to skip it entirely.

        Adds the freshness check that belongs to the *cadence* decision:
        without it, every N turns the stage re-reads the same 40 facts and
        re-derives the same insights however little is new. Same ``since``
        trick as ``_summary_inputs``. A direct ``_reflect()`` call deliberately
        bypasses this — an explicit "reflect now" should reflect.
        """
        prior = await self.store.recent(scope, limit=1, mem_types=("insight",))
        if prior and not await self.store.recent(
            scope, limit=1, mem_types=("fact",), since=prior[-1].timestamp
        ):
            return None
        return await self._reflect_facts(scope)

    async def _reflect(self, scope: str, facts=None) -> int:
        if facts is None:
            facts = await self._reflect_facts(scope)
        if not facts:
            return 0
        joined = "\n".join(f"- {item.content}" for item in facts)
        messages = [
            {"role": "system", "content": self.config.prompts.reflect},
            {"role": "user", "content": joined + "\n\nInsights (JSON array of strings):"},
        ]
        try:
            text = await self._complete(messages, temperature=0.3)
        except LLMError:
            return 0
        added = 0
        for insight in _parse_json_array(text):
            content = str(insight).strip()
            if not content:
                continue
            # Route through the same dedup/supersede path facts use, scoped to
            # mem_type="insight" so insights are compared against insights.
            # scope="" keeps them out of core memory: they are inferences, and
            # the pinned block is for observed durable facts.
            if await self._store_fact(
                scope,
                {"text": content, "subject": "insight", "importance": 8.0, "scope": "", "entities": []},
                mem_type="insight",
            ):
                added += 1
        return added
