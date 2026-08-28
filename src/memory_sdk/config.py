"""Configuration.

One validated pydantic model, nested by concern. Every default here is the
value proven in long-running production use of the ancestor codebase — change
them from your own measurements, not on aesthetics.

Deliberately NO environment-variable reading: how configuration is sourced
(env, file, hardcoded) is the host application's business. This also
structurally prevents the classic drift bug where a default written in two
places (dataclass field + ``os.getenv`` fallback) silently disagrees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .prompts import (
    DEFAULT_CORE_CONDENSE_PROMPT,
    DEFAULT_EXTRACT_PROMPT,
    DEFAULT_REFLECT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)

#: Model used when ``MemoryConfig.embedder`` is ``None`` and
#: sentence-transformers is installed. bge-m3 is multilingual and CPU-friendly;
#: any sentence-transformers model id works if you prefer a lighter one.
DEFAULT_ST_MODEL = "BAAI/bge-m3"


class _Model(BaseModel):
    model_config = ConfigDict(validate_assignment=True)


class RecallConfig(_Model):
    #: Guaranteed recall slots for facts.
    top_k: int = Field(default=6, ge=0)
    #: Guaranteed slots for the two synthesis types. Recall reserves slots PER
    #: TYPE instead of running one ranked contest across facts, summaries and
    #: insights. Measured on live data, a single pool never delivered synthesis
    #: at all: facts took 94-100% of the slots, because a short specific fact
    #: always out-scores a long general one against a short query (best-ever
    #: cosine 0.70 for facts vs 0.53 summary / 0.46 insight — and the gap
    #: widens as facts accumulate). 0 drops that type out of recall entirely.
    summary_slots: int = Field(default=1, ge=0)
    insight_slots: int = Field(default=1, ge=0)
    #: Scoring = similarity_weight*cosine + recency_weight*recencyDecay
    #: + importance_weight*importance/10 + lexical_weight*lexical_overlap.
    #: Half-life sets how fast recency decays.
    recency_halflife_days: float = Field(default=14.0, gt=0)
    similarity_weight: float = 1.0
    recency_weight: float = 0.4
    importance_weight: float = 0.3
    lexical_weight: float = 0.3
    #: Entity graph: nodes + co-occurrence edges, with a seed->1-hop expansion
    #: at recall time, seeded from the recalled facts' subjects.
    graph_enabled: bool = True
    #: How many neighbor-entity facts the 1-hop expansion may return.
    graph_expand_limit: int = Field(default=4, ge=0)


class DedupConfig(_Model):
    #: Cosine >= this against an existing same-subject row -> the incoming text
    #: is a duplicate: bump the stored row's importance, write nothing.
    dup_threshold: float = 0.90
    #: Cosine >= this (same subject, below the dup bar) -> the incoming fact is
    #: a correction: store it and supersede the prior row.
    conflict_threshold: float = 0.72
    #: Dedup is also attempted ACROSS subjects, because extractors label one
    #: entity several ways (one live database carried the same person under two
    #: spellings that could never match). Dedup discards the incoming fact, so
    #: a false positive loses information: cross-subject needs a stricter bar
    #: than same-subject — two facts differing only in a proper noun can still
    #: score ~0.9 under a strong embedder. Floored at ``dup_threshold`` in
    #: code, so it can never be laxer than the same-subject bar.
    cross_subject_dup_threshold: float = 0.96
    #: Bar when the two rows' embeddings weren't comparable (different
    #: embed_model) and similarity fell back to bounded lexical overlap — the
    #: noisier signal, so near-verbatim only. Also floored at
    #: ``dup_threshold``. Lexical matches may dedup but never supersede.
    lexical_dup_threshold: float = 0.95


class ConsolidationConfig(_Model):
    #: Master switch for background extraction/summary/reflection.
    enabled: bool = True
    #: Exchanges are buffered and extracted in batches instead of one LLM call
    #: per turn: a turn still inside the host's chat-history window is already
    #: visible to its model verbatim, so extracting it immediately is
    #: redundant. Keep the batch under about half of the host's history window
    #: or memory lags what the model can still see.
    batch_size: int = Field(default=8, ge=1)
    #: Second flush trigger, so a batch of long turns (or one pasted document)
    #: goes out early rather than assembling an enormous extraction prompt.
    #: 0 disables the char trigger.
    max_batch_chars: int = Field(default=6000, ge=0)
    #: Novelty gate: skip extraction for user utterances that repeat one
    #: already extracted from (casual chat is mostly banter that yields only
    #: dedup hits, and each one costs a full LLM call). "log" counts what it
    #: WOULD skip without skipping anything — run that first and read
    #: ``gate_would_skip`` in ``Memory.status()`` against real traffic before
    #: switching to "on". "off" disables it.
    gate_mode: Literal["off", "log", "on"] = "log"
    gate_similarity_threshold: float = 0.93
    #: Roll the single live summary every N consolidated exchanges; draw
    #: higher-level insights every M. 0 disables a stage.
    summary_every_turns: int = Field(default=24, ge=0)
    reflection_every_turns: int = Field(default=30, ge=0)
    summary_max_chars: int = Field(default=1200, ge=100)


class CoreMemoryConfig(_Model):
    #: Ordered core blocks: name -> heading used when rendering the block into
    #: a system prompt. Keep this short — the whole point of core memory is a
    #: small pinned budget that is ALWAYS in context, never retrieved.
    blocks: dict[str, str] = Field(
        default_factory=lambda: {
            "user_profile": "What you durably know about the user",
            "relationship": "The state of your relationship with the user",
        }
    )
    #: Which extraction ``scope`` value feeds which core block. Facts with a
    #: scope not in this map (e.g. "world") stay in the fact store only.
    scope_map: dict[str, str] = Field(
        default_factory=lambda: {"user": "user_profile", "relationship": "relationship"}
    )
    #: Character cap per block. Over-cap blocks are LLM-condensed (merge
    #: near-duplicate lines), trimmed oldest-first only if condensation fails —
    #: the oldest line is the most foundational fact, so trimming is the
    #: fallback, not the policy.
    block_cap: int = Field(default=2000, ge=200)
    condense_enabled: bool = True
    #: Ask the condenser for headroom so the next few facts don't immediately
    #: re-trigger it.
    condense_target: int = Field(default=1400, ge=100)
    #: Output under this fraction of the input means facts were dropped, not
    #: merged (a model replying "ok" would otherwise wipe the block) — reject.
    condense_min_ratio: float = Field(default=0.4, gt=0, le=1.0)
    #: Seconds to wait after a failed condense attempt before retrying, so an
    #: uncooperative model can't be re-asked on every single fact.
    condense_cooldown_seconds: float = Field(default=300.0, ge=0)


class PromptsConfig(_Model):
    extract: str = DEFAULT_EXTRACT_PROMPT
    summary: str = DEFAULT_SUMMARY_PROMPT
    reflect: str = DEFAULT_REFLECT_PROMPT
    core_condense: str = DEFAULT_CORE_CONDENSE_PROMPT


class MemoryConfig(_Model):
    #: Path of the SQLite database (created on first open).
    db_path: Path = Path("memory.sqlite3")
    #: Embedding model for semantic recall. ``None`` = auto-resolve: use
    #: sentence-transformers with ``DEFAULT_ST_MODEL`` if installed, else fall
    #: back to the zero-dependency hash embedder (with a loud warning — hash
    #: cosine is only lexical-ish, and semantic features like the novelty gate
    #: disable themselves on it). A model id string selects that
    #: sentence-transformers model; ``"hash"`` forces the fallback. To use a
    #: remote embeddings endpoint (or your own), pass an ``Embedder`` instance
    #: to ``Memory(embedder=...)`` instead — an explicit instance always wins
    #: over this field.
    embedder: str | None = None
    recall: RecallConfig = Field(default_factory=RecallConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    core: CoreMemoryConfig = Field(default_factory=CoreMemoryConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
