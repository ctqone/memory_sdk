"""Characterization suite for the store, carried over from the ancestor
codebase — each test encodes a bug that actually happened there. Behaviour is
the spec; adapt names, never semantics."""

from __future__ import annotations

import asyncio

import pytest

from fakes import ConstEmbedder, MapEmbedder
from memory_sdk import MemoryConfig, SQLiteMemoryStore, StoreVersionError
from memory_sdk.store import SCHEMA_VERSION, lexical_similarity


class TestSearch:
    async def test_importance_breaks_ties(self, make_store):
        # Constant embedding + empty query => sim equal, no lexical signal, so
        # the higher-importance fact must rank first.
        store = await make_store(embedder=ConstEmbedder())
        await store.add("p", "fact A", importance=2.0)
        await store.add("p", "fact B", importance=9.0)
        results = await store.search("", "p", limit=2)
        assert [item.content for item in results] == ["fact B", "fact A"]

    async def test_lexical_recall_finds_relevant(self, make_store):
        store = await make_store()  # hash embedder -> lexical-ish
        await store.add("p", "the user has a dog named Max", subject="Max")
        await store.add("p", "the user enjoys black coffee", subject="user")
        top = await store.search("what is the dog's name", "p", limit=1)
        assert top[0].subject == "Max"

    async def test_supersede_hidden_by_default(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        old = await store.add("p", "user lives in Tokyo", subject="user")
        await store.supersede(old.id)
        current = [i.content for i in await store.search("", "p", limit=10)]
        assert "user lives in Tokyo" not in current
        history = [
            i.content for i in await store.search("", "p", limit=10, include_superseded=True)
        ]
        assert "user lives in Tokyo" in history

    async def test_nondefault_mem_type_excluded_from_recall(self, make_store):
        # Recall serves the three typed layers only; anything else stored is
        # invisible to search() and recent() by default.
        store = await make_store(embedder=ConstEmbedder())
        await store.add("p", "raw chat turn", mem_type="verbatim")
        await store.add("p", "a real fact")
        assert [i.content for i in await store.search("", "p", limit=10)] == ["a real fact"]
        assert [i.content for i in await store.recent("p")] == ["a real fact"]

    async def test_bump_importance_caps_at_ten(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        item = await store.add("p", "x", importance=9.5)
        await store.bump_importance(item.id, 1.0)
        cursor = await store.db.execute("SELECT importance FROM memories WHERE id = ?", (item.id,))
        row = await cursor.fetchone()
        assert row[0] == 10.0

    async def test_recent_since_filters_by_timestamp(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        first = await store.add("p", "older")
        await asyncio.sleep(0.02)
        await store.add("p", "newer")
        fresh = await store.recent("p", since=first.timestamp)
        assert [i.content for i in fresh] == ["newer"]

    async def test_delete_and_clear(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        a = await store.add("p", "one")
        await store.add("p", "two")
        assert await store.delete("p", [a.id]) == 1
        assert len(await store.recent("p")) == 1
        assert await store.clear("p") == 1
        assert len(await store.recent("p")) == 0

    async def test_scopes_are_isolated(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        await store.add("alice", "alice's fact")
        await store.add("bob", "bob's fact")
        assert [i.content for i in await store.search("", "alice", limit=10)] == ["alice's fact"]
        await store.clear("bob")
        assert len(await store.recent("alice")) == 1


class TestSimilarFacts:
    async def test_orders_by_cosine(self, make_store):
        embedder = MapEmbedder(
            {
                "user lives in Tokyo": [1.0, 0.0, 0.0],
                "user lives in Osaka": [0.8, 0.6, 0.0],
                "user likes tea": [0.0, 1.0, 0.0],
            }
        )
        store = await make_store(embedder=embedder)
        await store.add("p", "user lives in Tokyo", subject="user")
        await store.add("p", "user likes tea", subject="user")
        ranked = await store.similar_facts("p", "user lives in Osaka", subject="user", limit=2)
        assert ranked[0][0].content == "user lives in Tokyo"
        assert ranked[0][1] > ranked[1][1]

    async def test_matches_subject_case_insensitively(self, make_store):
        # Subjects used to be compared raw, so "Miyu" and "miyu " were two
        # different people and could never dedup against each other.
        store = await make_store(embedder=ConstEmbedder())
        await store.add("p", "Miyu dances samba", subject="Miyu")
        assert len(await store.similar_facts("p", "Miyu dances samba", subject="miyu ")) == 1

    async def test_unscoped_crosses_subjects(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        await store.add("p", "Miyu dances samba", subject="Kawabata Miyu")
        scoped = await store.similar_facts("p", "Miyu dances samba", subject="Miyu (かわばた)")
        unscoped = await store.similar_facts(
            "p", "Miyu dances samba", subject="Miyu (かわばた)", scope_subject=False
        )
        assert len(scoped) == 0
        assert len(unscoped) == 1

    async def test_lexical_fallback_when_models_differ(self, make_store):
        # A row embedded by one backend and a query embedded by another used to
        # score a flat 0.0, silently disabling dedup across an embedding change.
        store = await make_store(embedder=MapEmbedder({"user likes tea": [1.0, 0.0, 0.0]}))
        await store.add("p", "user likes tea", subject="user")
        store.embedder = ConstEmbedder()  # different backend name -> incomparable
        same = await store.similar_facts("p", "user likes tea", subject="user")
        assert same[0].source == "lexical"
        assert same[0].score == 1.0
        different = await store.similar_facts("p", "quantum chromodynamics", subject="user")
        assert different[0].source == "lexical"
        assert different[0].score < 0.5

    async def test_mem_types_filters(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        await store.add("p", "an observed fact", mem_type="fact", subject="user")
        await store.add("p", "an inferred insight", mem_type="insight", subject="insight")
        only = await store.similar_facts("p", "anything", mem_types=("insight",), scope_subject=False)
        assert [m.item.content for m in only] == ["an inferred insight"]


def test_lexical_similarity_bounded_and_symmetric():
    assert lexical_similarity("same text", "same text") == 1.0
    assert lexical_similarity("cats", "quantum") == 0.0
    assert lexical_similarity("a b c", "b c d") == lexical_similarity("b c d", "a b c")
    # The property the ranking leg lacks: a strict superset must NOT look
    # identical, or dedup would swallow the more specific fact.
    assert (
        lexical_similarity("user likes tea", "user likes tea in the morning after a long run")
        < 0.95
    )


class TestGraph:
    async def test_graph_expand_surfaces_connected_facts(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        await store.add("p", "the user adopted a dog named Max", subject="user")
        await store.add("p", "Max needs a Saturday walk", subject="Max")
        await store.add_cooccurrence("p", "user", "Max")
        expanded = await store.graph_expand("p", ["user"], limit=4)
        assert "Max needs a Saturday walk" in [i.content for i in expanded]

    async def test_entities_listed_by_mentions(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        await store.upsert_entity("p", "Max")
        await store.upsert_entity("p", "Max")
        await store.upsert_entity("p", "Kana")
        names = [e["name"] for e in await store.entities("p")]
        assert names[0] == "Max"  # most-mentioned first

    @pytest.mark.parametrize("label", [" Max ", "Ｍａｘ"])
    async def test_graph_expand_normalizes_subjects_like_the_rest_of_the_store(
        self, make_store, label
    ):
        """Expansion must key on subject_norm, not SQL LOWER(subject).

        Everything else in the store normalizes subjects in Python via
        norm_subject, precisely because SQLite's LOWER() is ASCII-only. The
        1-hop walk was once the lone holdout comparing LOWER(subject), so a
        fact stored under a padded or full-width label was reachable by dedup
        and by recall but silently invisible to graph expansion — the leg
        whose whole job is surfacing what vector search misses.
        """
        store = await make_store(embedder=ConstEmbedder())
        await store.add("p", "the user adopted a dog", subject="user")
        await store.add("p", f"{label.strip()} needs a walk", subject=label)
        # The entity graph already normalizes in Python, so the edge lands on
        # the same canonical name the fact was stored under.
        await store.add_cooccurrence("p", "user", label)
        expanded = await store.graph_expand("p", ["user"], limit=4)
        assert f"{label.strip()} needs a walk" in [item.content for item in expanded], (
            f"a fact stored under subject {label!r} fell out of graph expansion"
        )


class TestReservedSlots:
    """per_type_limits: synthesis rows reach recall even when every fact
    outranks them."""

    async def _stocked(self, make_store):
        # Facts embed onto the query vector, synthesis rows onto an orthogonal
        # one, so every fact strictly outscores every summary/insight — the
        # live situation that starved synthesis out of a single ranked pool.
        mapping = {"query": [1.0, 0.0, 0.0]}
        for i in range(10):
            mapping[f"fact {i}"] = [1.0, 0.0, 0.0]
        mapping["the rolling summary"] = [0.0, 1.0, 0.0]
        mapping["an inference"] = [0.0, 1.0, 0.0]
        store = await make_store(embedder=MapEmbedder(mapping))
        for i in range(10):
            await store.add("p", f"fact {i}", mem_type="fact", subject="user")
        await store.add("p", "the rolling summary", mem_type="summary", importance=6.0, subject="session")
        await store.add("p", "an inference", mem_type="insight", importance=8.0, subject="insight")
        return store

    async def test_single_pool_starves_synthesis(self, make_store):
        # Establishes the baseline the reservation exists to fix.
        store = await self._stocked(make_store)
        top = await store.search("query", "p", limit=6)
        assert {item.mem_type for item in top} == {"fact"}

    async def test_reserved_slots_deliver_one_of_each(self, make_store):
        store = await self._stocked(make_store)
        top = await store.search(
            "query", "p", limit=8, per_type_limits={"fact": 6, "summary": 1, "insight": 1}
        )
        counts: dict[str, int] = {}
        for item in top:
            counts[item.mem_type] = counts.get(item.mem_type, 0) + 1
        assert counts == {"fact": 6, "summary": 1, "insight": 1}

    async def test_zero_slots_drops_a_type(self, make_store):
        store = await self._stocked(make_store)
        top = await store.search(
            "query", "p", limit=8, per_type_limits={"fact": 6, "summary": 1, "insight": 0}
        )
        assert "insight" not in {item.mem_type for item in top}
        assert "summary" in {item.mem_type for item in top}

    async def test_caps_replace_limit_as_the_budget(self, make_store):
        store = await self._stocked(make_store)
        top = await store.search(
            "query", "p", limit=2,  # deliberately smaller than the caps
            per_type_limits={"fact": 6, "summary": 1, "insight": 1},
        )
        assert len(top) == 8

    async def test_a_type_with_nothing_stored_just_yields_fewer(self, make_store):
        store = await make_store(
            embedder=MapEmbedder({"query": [1.0, 0.0, 0.0], "only fact": [1.0, 0.0, 0.0]})
        )
        await store.add("p", "only fact", mem_type="fact", subject="user")
        top = await store.search(
            "query", "p", limit=8, per_type_limits={"fact": 6, "summary": 1, "insight": 1}
        )
        assert [item.mem_type for item in top] == ["fact"]


class TestSchemaVersion:
    async def test_refuses_a_newer_database(self, make_store, tmp_path):
        store = await make_store(name="versioned.sqlite3")
        await store.db.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 1),)
        )
        await store.db.commit()
        await store.close()
        newer = SQLiteMemoryStore(tmp_path / "versioned.sqlite3", MemoryConfig())
        with pytest.raises(StoreVersionError):
            await newer.open()

    async def test_fresh_database_is_stamped(self, make_store):
        store = await make_store()
        cursor = await store.db.execute("SELECT value FROM meta WHERE key = 'schema_version'")
        row = await cursor.fetchone()
        assert int(row[0]) == SCHEMA_VERSION


class TestCoreMemory:
    async def test_get_set_merge_and_render(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        assert await store.get_core_blocks("p") == {"user_profile": "", "relationship": ""}
        await store.set_core_blocks("p", {"user_profile": "- name is Aki"})
        # Merge semantics: an unmentioned block is left alone, unknown keys dropped.
        saved = await store.set_core_blocks("p", {"relationship": "- friendly", "bogus": "x"})
        assert saved == {"user_profile": "- name is Aki", "relationship": "- friendly"}
        rendered = store.render_core(saved)
        assert "- name is Aki" in rendered and "- friendly" in rendered
        assert store.render_core({"user_profile": "", "relationship": ""}) == ""

    async def test_snapshot_keeps_the_pre_rewrite_content(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        await store.set_core_blocks("p", {"user_profile": "original"})
        await store.snapshot_core_block("p", "user_profile", "precondense")
        await store.set_core_blocks("p", {"user_profile": "rewritten"})
        cursor = await store.db.execute(
            "SELECT content, tag FROM core_memory_snapshots WHERE scope = 'p'"
        )
        rows = await cursor.fetchall()
        assert ("original", "precondense") in rows

    async def test_clear_core_is_separate_from_clear(self, make_store):
        store = await make_store(embedder=ConstEmbedder())
        await store.add("p", "a fact")
        await store.set_core_blocks("p", {"user_profile": "- name is Aki"})
        await store.clear("p")
        assert (await store.get_core_blocks("p"))["user_profile"] == "- name is Aki"
        await store.clear_core("p")
        assert (await store.get_core_blocks("p"))["user_profile"] == ""
