"""Public data types.

A memory row is a *typed, temporally-aware* record, not a raw chat turn:

- ``mem_type``: ``'fact'`` (extracted atomic fact), ``'summary'`` (the rolling
  recap), or ``'insight'`` (a higher-level reflection). Recall only ever
  returns these three; any other value is stored but excluded from recall.
- ``importance``: 1..10, LLM-rated — the salience analog used in scoring.
- ``valid_from`` / ``valid_to``: temporal validity. ``valid_to`` is ``None``
  while a memory is current, and set to a timestamp when a newer one
  supersedes it. Recall hides superseded rows by default so a corrected fact
  wins — supersession, not mutation, is how a fact changes.
- ``supersedes``: id of the row this one replaced (audit / history view).
- ``subject``: the primary entity the memory is about; groups facts and links
  them into the entity graph.
- ``score``: transient, populated by ``search()``; never persisted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import NamedTuple


@dataclass
class MemoryItem:
    id: str
    scope: str
    content: str
    timestamp: float
    mem_type: str = "fact"
    importance: float = 5.0
    subject: str = ""
    valid_from: float = 0.0
    valid_to: float | None = None
    supersedes: str | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FactMatch(NamedTuple):
    """One dedup/conflict candidate from ``similar_facts``. ``source`` is
    ``"cosine"`` when the two embeddings were comparable and ``"lexical"`` when
    the store had to fall back — callers must treat lexical as the weaker
    signal and hold it to a stricter bar. A NamedTuple so ``item, score =
    match[:2]`` unpacking works alongside attribute access."""

    item: MemoryItem
    score: float
    source: str


@dataclass
class RecallResult:
    """What ``Memory.search()`` returns.

    ``memories`` is the slot-reserved recall set (facts + summaries +
    insights, best-first within each type). ``graph`` is the 1-hop
    entity-graph expansion seeded from the recalled facts' subjects —
    connected-but-not-similar context a flat vector search misses. The two are
    disjoint (expansion drops anything already recalled).
    """

    query: str
    memories: list[MemoryItem] = field(default_factory=list)
    graph: list[MemoryItem] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.memories or self.graph)
