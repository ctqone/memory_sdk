from __future__ import annotations

import pydantic
import pytest

from memory_sdk import MemoryConfig


def test_documented_defaults():
    """Spot-check the production-proven defaults the docs promise. If one of
    these fails, either the docs or the intent changed — update both."""
    cfg = MemoryConfig()
    assert cfg.recall.top_k == 6
    assert cfg.recall.summary_slots == 1
    assert cfg.recall.insight_slots == 1
    assert cfg.recall.recency_halflife_days == 14.0
    assert cfg.dedup.dup_threshold == 0.90
    assert cfg.dedup.conflict_threshold == 0.72
    assert cfg.dedup.cross_subject_dup_threshold == 0.96
    assert cfg.dedup.lexical_dup_threshold == 0.95
    assert cfg.consolidation.batch_size == 8
    assert cfg.consolidation.max_batch_chars == 6000
    assert cfg.consolidation.gate_mode == "log"
    assert cfg.consolidation.gate_similarity_threshold == 0.93
    assert cfg.consolidation.summary_every_turns == 24
    assert cfg.consolidation.reflection_every_turns == 30
    assert cfg.core.block_cap == 2000
    assert cfg.core.condense_target == 1400
    assert cfg.core.condense_min_ratio == 0.4
    assert cfg.core.condense_cooldown_seconds == 300.0
    assert cfg.embedder is None


def test_nested_overrides_via_dicts():
    cfg = MemoryConfig(consolidation={"batch_size": 4}, dedup={"dup_threshold": 0.8})
    assert cfg.consolidation.batch_size == 4
    assert cfg.dedup.dup_threshold == 0.8
    # Untouched sections keep their defaults.
    assert cfg.recall.top_k == 6


def test_validation_rejects_nonsense():
    with pytest.raises(pydantic.ValidationError):
        MemoryConfig(consolidation={"batch_size": 0})
    with pytest.raises(pydantic.ValidationError):
        MemoryConfig(consolidation={"gate_mode": "sometimes"})
    with pytest.raises(pydantic.ValidationError):
        MemoryConfig(core={"condense_min_ratio": 0.0})


def test_assignment_is_validated():
    cfg = MemoryConfig()
    cfg.consolidation.summary_every_turns = 4  # tests rely on this being settable
    assert cfg.consolidation.summary_every_turns == 4
    with pytest.raises(pydantic.ValidationError):
        cfg.consolidation.batch_size = -1
