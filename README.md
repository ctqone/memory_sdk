# MemorySDK

**Async-native layered memory for LLM agents.** Typed facts / rolling
summaries / insights with per-type recall slots, temporal supersession, an
entity graph with 1-hop recall expansion, and batched LLM consolidation behind
a novelty gate. One SQLite file. No vector database.

The internals were extracted from a long-running personal agent project where
every design decision below was earned against real traffic — including the
failure modes the architecture exists to prevent.

> **Status: alpha (0.1.x).** The API may change between minor versions until
> 1.0. The on-disk schema is versioned and migrated forward from day one — see
> [Schema stability](#schema-stability).

## Install

```bash
pip install git+https://github.com/ctqone/memory_sdk
# with local embeddings (sentence-transformers, CPU):
pip install "memory-sdk[local] @ git+https://github.com/ctqone/memory_sdk"
```

Python ≥ 3.10. Core dependencies: `pydantic`, `httpx`, `aiosqlite`.

## Quickstart

```python
import asyncio
from memory_sdk import Memory, MemoryConfig, OpenAICompatibleLLM

async def main():
    # Any /v1/chat/completions endpoint: OpenAI, llama.cpp server, vLLM,
    # Ollama, LM Studio, OpenRouter, ...
    llm = OpenAICompatibleLLM("http://127.0.0.1:8000/v1", model="my-model")

    async with Memory(MemoryConfig(db_path="agent_memory.sqlite3"), llm=llm) as mem:
        # After each finished exchange, hand it to background consolidation.
        # Returns immediately; extraction is batched and runs on a worker task.
        await mem.add("user-42", "I moved to Osaka last month!", "How exciting!",
                      session_id="chat-1")

        # Before the next model call, recall what matters and put it in the prompt.
        result = await mem.search("user-42", "where does the user live?")
        system_extra = mem.render(result)            # bulleted, prompt-ready
        pinned = await mem.render_core("user-42")    # tier-1 core memory block

    # exiting the context drains buffered exchanges, then closes cleanly

asyncio.run(main())
```

`scope` (`"user-42"` above) is an opaque string naming who a memory belongs
to. Compose it however your identity model works — per user, per
user×agent, per tenant. Two scopes never see each other's memories.

## What it does

**Extraction, not logging.** Raw turns are not memory. A background worker
batches finished exchanges and has the LLM extract *atomic third-person
facts*, each with a subject, an importance rating, and linked entities.
Batching is deliberate: a turn still inside your model's chat-history window
is already visible to it verbatim, so extracting it one call at a time is
redundant — and a batch lets the extractor see an arc that a single turn
cannot show.

**A novelty gate before the LLM.** Casual chat is mostly banter that yields
nothing but duplicate hits, and each one costs a completion. A CPU-only
embedding gate drops user utterances that repeat one already extracted from —
comparing utterance to utterance (never utterance to stored fact, which
structurally cannot match). It ships in `"log"` mode: run it against real
traffic and read `status()["gate_would_skip"]` before turning it on. It fails
open whenever a real semantic embedder isn't resident — hash-embedding cosine
is noise, and skipping on noise would eat memories at random.

**Dedup and supersession with different reach.** An incoming fact that
near-duplicates a stored one bumps that row's importance instead of inserting
(non-destructive, so it may match across subject labels and may use a lexical
fallback — each behind a stricter bar). A fact that *contradicts* a stored one
supersedes it: the old row gets `valid_to` and drops out of recall, but stays
auditable. Supersession is destructive, so it stays same-subject and
cosine-only. Facts change by supersession, never by mutation.

**Three memory layers, with reserved recall slots.** Facts, one rolling
summary (the previous summary is folded into the next and then retired —
exactly one is ever live), and reflection-derived insights. Recall reserves
slots per type instead of running one ranked contest, because a single pool
demonstrably starves synthesis: short specific facts out-score long general
summaries against short queries, 100% of the time once enough facts
accumulate.

**An entity graph, grown for free.** Extracted facts carry subjects and
mentioned entities; co-occurrence edges accumulate. At recall time a 1-hop
expansion seeded from the recalled facts' subjects surfaces
connected-but-not-similar context that vector search misses.

**Tier-1 core memory.** A tiny, always-in-context set of blocks (who the user
is, the state of the relationship) that consolidation maintains and your app
can render straight into the system prompt — never retrieved, never trimmed.
When a block outgrows its cap it is LLM-condensed (merge near-duplicates,
keep every distinct fact) with validation and a pre-rewrite snapshot; the
naive alternative — dropping the oldest lines — evicts the most foundational
facts first, starting with the user's name.

**Erasure that means it.** Deletes are hard `DELETE`s under SQLite
`secure_delete` (freed pages zeroed), and clearing a scope also drops its
un-extracted buffered exchanges — content a user erased must not surface as a
memory minutes later.

## Providing the LLM and embedder

Consolidation needs a completion backend. Use the built-in client, or
implement the one-method protocol:

```python
from memory_sdk import LLMBackend, LLMError

class MyBackend:                      # satisfies LLMBackend
    async def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        ...                           # raise LLMError on failure
```

Embeddings resolve in this order:

1. An `Embedder` instance passed to `Memory(embedder=...)` — including the
   built-in `OpenAICompatibleEmbedder` for any `/v1/embeddings` endpoint, or
   your own subclass.
2. `MemoryConfig.embedder`: a sentence-transformers model id, or `"hash"`.
3. Auto: sentence-transformers (`BAAI/bge-m3`, multilingual, CPU) if
   installed, else the zero-dependency hash embedder with a loud warning —
   recall degrades to lexical-ish and the novelty gate stays off.

Every stored vector is tagged with the backend that produced it; vectors from
different backends are never cosine-compared (they degrade to a bounded
lexical comparison instead). Switching embedders is therefore safe but makes
old rows lexically-matched until they are superseded by new ones.

If LLM access is contended in your app (a shared GPU, a rate limit), pass
`Memory(turn_gate=...)` — a factory for an async context manager that is
entered around every consolidation completion.

## Configuration

Everything lives on one validated `MemoryConfig` (pydantic v2), nested by
concern; every default is the value proven in the ancestor deployment.
No environment variables are read — sourcing config is your app's business.

```python
from memory_sdk import Memory, MemoryConfig

config = MemoryConfig(
    db_path="agent_memory.sqlite3",
    embedder=None,                        # auto; or a sentence-transformers id; or "hash"
    recall={"top_k": 6, "summary_slots": 1, "insight_slots": 1},
    dedup={"dup_threshold": 0.90, "conflict_threshold": 0.72},
    consolidation={"batch_size": 8, "gate_mode": "log",
                   "summary_every_turns": 24, "reflection_every_turns": 30},
    core={"block_cap": 2000},
)
```

See the docstrings in
[`config.py`](src/memory_sdk/config.py) — every knob carries the reasoning for
its default, including the measured failure the value guards against.

## API sketch

```python
async with Memory(config, llm=..., embedder=..., turn_gate=...) as mem:
    await mem.add(scope, user_text, assistant_text, session_id=...)   # -> consolidation
    result = await mem.search(scope, query)     # facts + summary + insights + graph
    text   = mem.render(result)                 # prompt-ready block
    await mem.remember(scope, content, mem_type=..., importance=..., subject=...)
    await mem.supersede(memory_id)
    await mem.list(scope) / mem.entities(scope) / mem.related(scope, subject)
    await mem.get_core(scope) / mem.set_core(scope, blocks) / mem.render_core(scope)
    await mem.delete(scope, ids) / mem.clear(scope)
    mem.discard_buffer(scope, session_id)       # host deleted a conversation
    await mem.flush()                           # drain consolidation now
    mem.status()                                # live worker/gate/embedder counters
```

Power users can drive `SQLiteMemoryStore` and `Consolidator` directly; the
facade is a thin composition of the two.

## Schema stability

The database carries `schema_version` in its `meta` table.

- This library **migrates older databases forward** automatically on open.
- It **refuses to open a newer database** (`StoreVersionError`) rather than
  write into a schema it does not understand.
- One-shot data migrations record themselves in `meta` so they can never
  re-fire and clobber later edits.

## Development

```bash
pip install -e ".[dev]"
pytest             # asyncio_mode=auto; no network, no models needed
ruff check .
pyright
```

The test suite is a characterization suite: most tests encode a real bug from
the ancestor codebase (recall-slot starvation, batch cadence stepping over its
boundary, dedup silently disabled by an embedder change, core memory evicting
the user's name, ...). Behaviour is the spec — adapt names, never semantics.

## Roadmap

- Sync facade (`SyncMemory`) over the async core
- Per-stage LLM routing (cheap model for extraction, stronger for reflection)
- First-party Anthropic backend (`memory-sdk[anthropic]`)
- PyPI release

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
