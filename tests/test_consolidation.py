"""Characterization suite for consolidation, carried over from the ancestor
codebase. The scripted LLM fakes route on substrings of the system prompt
("summar" / "insight" select the summarize/reflect stages), which is why the
default core-condense prompt must contain neither — see prompts.py."""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest_asyncio

from fakes import ConstEmbedder, FailingLLM, MapEmbedder, RecordingLLM, ScriptedLLM, fact_json
from memory_sdk import Consolidator, HashEmbedder, MemoryConfig
from memory_sdk.consolidation import _crossed, _normalize_facts, _parse_json_array


def make_config(**sections) -> MemoryConfig:
    """MemoryConfig with nested overrides:
    ``make_config(consolidation={"batch_size": 4}, dedup={"dup_threshold": 2.0})``."""
    return MemoryConfig(**sections)


#: Dedup/conflict disabled — every extracted fact inserts a fresh row.
NO_DEDUP = {"dup_threshold": 2.0, "conflict_threshold": 2.0}


@pytest_asyncio.fixture
async def make_worker(make_store):
    workers: list[Consolidator] = []

    async def factory(llm=None, embedder=None, config=None, store=None) -> Consolidator:
        cfg = config or MemoryConfig()
        if store is None:
            store = await make_store(embedder=embedder or MapEmbedder({}), config=cfg)
        worker = Consolidator(cfg, store, llm or ScriptedLLM(["[]"]))
        workers.append(worker)
        return worker

    yield factory
    for worker in workers:
        if worker._task is not None:
            worker._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker._task


def submitted(worker: Consolidator):
    """Batches handed to the work queue so far, without running them."""
    out = []
    while True:
        try:
            out.append(worker._queue.get_nowait())
        except asyncio.QueueEmpty:
            return out


class TestParsing:
    def test_tolerates_prose_and_trailing_commas(self):
        raw = (
            'Sure! [{"text": "user likes tea", "subject": "user", "importance": 5, '
            '"scope": "user", "entities": [],},]'
        )
        facts = _normalize_facts(_parse_json_array(raw))
        assert len(facts) == 1
        assert facts[0]["text"] == "user likes tea"
        assert facts[0]["importance"] == 5.0

    def test_empty_array(self):
        assert _parse_json_array("nothing to remember []") == []
        assert _normalize_facts([]) == []

    def test_crossed_handles_batch_steps(self):
        # A batch advances the counter by len(batch); `count % every == 0`
        # steps straight over the boundary and the stage never fires again.
        assert _crossed(14, 8, 8)  # 6 -> 14 crosses 8
        assert not _crossed(6, 6, 8)
        assert not _crossed(14, 0, 8)
        assert not _crossed(14, 8, 0)


class TestConsolidation:
    async def test_extracts_fact_and_updates_core_block(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM([fact_json("user is a teacher", subject="user", scope="user")]),
            embedder=MapEmbedder({"user is a teacher": [1.0, 0.0, 0.0]}),
        )
        await worker.consolidate("p", "I teach high school", "Nice!")
        assert [i.content for i in await worker.store.recent("p")] == ["user is a teacher"]
        blocks = await worker.store.get_core_blocks("p")
        assert "user is a teacher" in blocks["user_profile"]

    async def test_duplicate_bumps_importance_no_new_row(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM([fact_json("user likes coffee", importance=5)]),
            embedder=MapEmbedder({"user likes coffee": [0.0, 1.0, 0.0]}),
        )
        await worker.consolidate("p", "I like coffee", "ok")
        await worker.consolidate("p", "I really like coffee", "ok")  # same fact -> dup
        rows = await worker.store.recent("p", include_superseded=True)
        assert len(rows) == 1
        cursor = await worker.store.db.execute(
            "SELECT importance FROM memories WHERE id = ?", (rows[0].id,)
        )
        assert (await cursor.fetchone())[0] > 5.0

    async def test_conflict_supersedes_prior_fact(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM(
                [
                    fact_json("user lives in Tokyo", subject="user"),
                    fact_json("user lives in Osaka", subject="user"),
                ]
            ),
            embedder=MapEmbedder(
                {
                    "user lives in Tokyo": [1.0, 0.0, 0.0],
                    "user lives in Osaka": [0.8, 0.6, 0.0],  # cosine 0.8 -> conflict band
                }
            ),
        )
        await worker.consolidate("p", "I live in Tokyo", "ok")
        await worker.consolidate("p", "I moved to Osaka", "ok")
        store = worker.store
        assert [i.content for i in await store.search("", "p", limit=10)] == ["user lives in Osaka"]
        history = [
            i.content for i in await store.search("", "p", limit=10, include_superseded=True)
        ]
        assert "user lives in Tokyo" in history
        # The new fact records what it superseded.
        cursor = await store.db.execute(
            "SELECT supersedes FROM memories WHERE content = ? AND valid_to IS NULL",
            ("user lives in Osaka",),
        )
        assert (await cursor.fetchone())[0] is not None

    async def test_builds_entity_graph(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM(
                [fact_json("the user adopted a dog named Max", subject="user", entities=["Max"])]
            ),
            embedder=MapEmbedder({"the user adopted a dog named Max": [1.0, 0.0, 0.0]}),
        )
        await worker.consolidate("p", "I got a dog called Max", "aww")
        names = [e["name"] for e in await worker.store.entities("p")]
        assert "Max" in names
        assert "user" in names

    async def test_summary_stored_with_summary_mem_type(self, make_worker):
        worker = await make_worker(llm=ScriptedLLM(["Aki teaches and lives in Osaka with a dog."]))
        for i in range(4):
            await worker.store.add("p", f"durable fact {i}", mem_type="fact", subject="user")
        await worker._summarize("p")
        summaries = await worker.store.recent("p", mem_types=("summary",))
        assert len(summaries) == 1
        assert summaries[0].mem_type == "summary"
        # and it must NOT have leaked into the fact pool
        assert len(await worker.store.recent("p", mem_types=("fact",))) == 4

    async def test_cadence_is_restart_persistent(self, make_worker):
        # summary_every=4. Worker A does 2 exchanges (no summary yet). A
        # brand-new worker B on the SAME store does 2 more: the persisted
        # counter reaches 4 and a summary fires. With an in-memory tally, B
        # would restart at 0 and never reach 4 here.
        class CountingLLM:
            def __init__(self):
                self.n = 0

            async def complete(self, messages, *, temperature=None, max_tokens=None):
                self.n += 1
                head = messages[0]["content"].lower()
                if "summar" in head or "insight" in head:
                    return "A concise summary of the user."
                return json.dumps(
                    [
                        {
                            "text": f"distinct fact {self.n}",
                            "subject": "user",
                            "importance": 5,
                            "scope": "world",
                            "entities": [],
                        }
                    ]
                )

        cfg = make_config(
            consolidation={"summary_every_turns": 4, "reflection_every_turns": 0},
            dedup=NO_DEDUP,
        )
        llm = CountingLLM()
        w1 = await make_worker(llm=llm, config=cfg)
        await w1.consolidate("p", "u1", "a1")
        await w1.consolidate("p", "u2", "a2")
        assert len(await w1.store.recent("p", mem_types=("summary",))) == 0

        w2 = await make_worker(llm=llm, config=cfg, store=w1.store)  # simulate restart
        await w2.consolidate("p", "u3", "a3")
        await w2.consolidate("p", "u4", "a4")
        assert len(await w2.store.recent("p", mem_types=("summary",))) == 1
        assert await w2.store.bump_counter("p", "consolidation_turns", 0) == 4

    # -- dedup/supersede reach ------------------------------------------------
    async def test_dedup_matches_across_subject_labels(self, make_worker):
        # The extractor labels one entity several ways; a live database once
        # carried the same person under two spellings that could never match.
        worker = await make_worker(
            llm=ScriptedLLM(
                [
                    fact_json("Miyu dances samba", subject="Miyu (かわばた／みゆ)", scope="world"),
                    fact_json("Miyu dances samba", subject="Kawabata Miyu", scope="world"),
                ]
            ),
            embedder=MapEmbedder({"Miyu dances samba": [1.0, 0.0, 0.0]}),
        )
        await worker.consolidate("p", "tell me about Miyu", "ok")
        await worker.consolidate("p", "again", "ok")
        assert len(await worker.store.recent("p", include_superseded=True)) == 1

    async def test_cross_subject_dedup_requires_higher_bar(self, make_worker):
        # 0.92 clears the same-subject bar (0.90) but not the cross-subject
        # one (0.96): two different entities, so keep both.
        worker = await make_worker(
            llm=ScriptedLLM(
                [
                    fact_json("Aki lives in Tokyo", subject="Aki", scope="world"),
                    fact_json("Miyu lives in Tokyo", subject="Miyu", scope="world"),
                ]
            ),
            embedder=MapEmbedder(
                {"Aki lives in Tokyo": [1.0, 0.0, 0.0], "Miyu lives in Tokyo": [0.92, 0.3919, 0.0]}
            ),
        )
        await worker.consolidate("p", "a", "ok")
        await worker.consolidate("p", "b", "ok")
        assert len(await worker.store.recent("p", include_superseded=True)) == 2

    async def test_supersede_never_crosses_subjects(self, make_worker):
        # Conflict band (0.8) but different entities: a fact about Miyu is not
        # a correction to a fact about Aki, so nothing may be retired.
        worker = await make_worker(
            llm=ScriptedLLM(
                [
                    fact_json("Aki lives in Tokyo", subject="Aki", scope="world"),
                    fact_json("Miyu lives in Osaka", subject="Miyu", scope="world"),
                ]
            ),
            embedder=MapEmbedder(
                {"Aki lives in Tokyo": [1.0, 0.0, 0.0], "Miyu lives in Osaka": [0.8, 0.6, 0.0]}
            ),
        )
        await worker.consolidate("p", "a", "ok")
        await worker.consolidate("p", "b", "ok")
        assert len(await worker.store.search("", "p", limit=10)) == 2

    async def test_supersede_matches_case_differing_subject(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM(
                [
                    fact_json("user lives in Tokyo", subject="User", scope="world"),
                    fact_json("user lives in Osaka", subject="user ", scope="world"),
                ]
            ),
            embedder=MapEmbedder(
                {"user lives in Tokyo": [1.0, 0.0, 0.0], "user lives in Osaka": [0.8, 0.6, 0.0]}
            ),
        )
        await worker.consolidate("p", "a", "ok")
        await worker.consolidate("p", "b", "ok")
        current = [i.content for i in await worker.store.search("", "p", limit=10)]
        assert current == ["user lives in Osaka"]

    async def test_lexical_match_dedups_but_never_supersedes(self, make_worker):
        # Embeddings incomparable -> lexical. Dice here is well below the
        # lexical dedup bar but above the conflict band; supersede must still
        # refuse, because lexical overlap is too noisy to retire a true fact.
        worker = await make_worker(
            llm=ScriptedLLM([fact_json("user lives in Osaka", subject="user", scope="world")]),
            embedder=MapEmbedder({"user lives in Tokyo": [1.0, 0.0, 0.0]}),
        )
        await worker.store.add("p", "user lives in Tokyo", subject="user")
        worker.store.embedder = ConstEmbedder()  # different backend -> incomparable
        await worker.consolidate("p", "b", "ok")
        assert len(await worker.store.search("", "p", limit=10)) == 2

    # -- insights ----------------------------------------------------------------
    async def test_reflection_stores_insight_type_and_dedups(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM(['["the user avoids conflict"]']),
            embedder=ConstEmbedder(),  # everything cosine 1.0
        )
        for i in range(6):
            await worker.store.add("p", f"durable fact {i}", mem_type="fact", subject="user")
        assert await worker._reflect("p") == 1
        assert await worker._reflect("p") == 0  # same insight -> deduped, not re-added
        insights = await worker.store.recent("p", mem_types=("insight",))
        assert [i.content for i in insights] == ["the user avoids conflict"]

    async def test_reflection_input_excludes_prior_insights(self, make_worker):
        llm = RecordingLLM(["[]"])
        worker = await make_worker(llm=llm)
        for i in range(6):
            await worker.store.add("p", f"durable fact {i}", mem_type="fact", subject="user")
        await worker.store.add("p", "a prior inference", mem_type="insight", subject="insight")
        await worker._reflect("p")
        assert "a prior inference" not in llm.captured[0]

    # -- rolling summary ----------------------------------------------------------
    async def test_summary_rolls_and_supersedes_previous(self, make_worker):
        llm = RecordingLLM(["summary revision 1", "summary revision 2"])
        worker = await make_worker(llm=llm)
        for i in range(4):
            await worker.store.add("p", f"durable fact {i}", mem_type="fact", subject="user")
        assert await worker._summarize("p")
        await worker.store.add("p", "a newly learned fact", mem_type="fact", subject="user")
        assert await worker._summarize("p")

        live = await worker.store.recent("p", mem_types=("summary",))
        assert [s.content for s in live] == ["summary revision 2"]
        history = await worker.store.recent("p", mem_types=("summary",), include_superseded=True)
        assert len(history) == 2  # the old one is retired, not deleted
        # The previous summary is fed forward so pre-window knowledge survives.
        assert "summary revision 1" in llm.captured[1]

    async def test_summary_collapses_preexisting_live_summaries(self, make_worker):
        # Two summaries left live by an append-forever implementation must
        # collapse to one in a single pass.
        worker = await make_worker(llm=ScriptedLLM(["merged summary"]))
        await worker.store.add("p", "old one", mem_type="summary", subject="session")
        await worker.store.add("p", "old two", mem_type="summary", subject="session")
        await worker.store.add("p", "something new", mem_type="fact", subject="user")
        assert await worker._summarize("p")
        assert [s.content for s in await worker.store.recent("p", mem_types=("summary",))] == [
            "merged summary"
        ]

    async def test_summary_skipped_when_no_new_facts(self, make_worker):
        llm = ScriptedLLM(["a summary"])
        worker = await make_worker(llm=llm)
        for i in range(4):
            await worker.store.add("p", f"durable fact {i}", mem_type="fact", subject="user")
        assert await worker._summarize("p")
        calls = llm.calls
        assert not await worker._summarize("p")  # nothing new -> no LLM turn burnt
        assert llm.calls == calls

    # -- core-block condensation ---------------------------------------------------
    SMALL_CAP = {"block_cap": 200}

    async def _seed_over_cap(self, worker) -> str:
        """Seed a core block already over the 200-char cap, oldest line first —
        the FIFO trim evicts that line, which in a real deployment was the
        user's name."""
        seeded = "\n".join(["- name is Aki"] + [f"- filler fact {i}" for i in range(12)])
        await worker.store.set_core_blocks("p", {"user_profile": seeded})
        return seeded

    async def test_core_block_condensed_by_llm_when_over_cap(self, make_worker):
        # Must stay above the over-compression floor (40% of the input) — a
        # drastically shorter rewrite is treated as dropped facts.
        condensed = "- name is Aki\n" + "\n".join(f"- merged fact {i}" for i in range(6))
        worker = await make_worker(
            llm=ScriptedLLM([condensed]), config=make_config(core=self.SMALL_CAP)
        )
        await self._seed_over_cap(worker)
        await worker._maybe_update_core("p", {"text": "user is a teacher", "scope": "user"})
        block = (await worker.store.get_core_blocks("p"))["user_profile"]
        assert len(block) <= worker.config.core.block_cap
        # The whole point: the oldest, most foundational fact SURVIVES, and the
        # just-learned fact is appended after condensation so it can't be lost.
        assert "- name is Aki" in block
        assert "user is a teacher" in block
        assert worker.status()["core_condensed"] == 1

    async def test_core_condense_falls_back_to_trim_when_llm_fails(self, make_worker):
        worker = await make_worker(llm=FailingLLM(), config=make_config(core=self.SMALL_CAP))
        await self._seed_over_cap(worker)
        await worker._maybe_update_core("p", {"text": "user is a teacher", "scope": "user"})
        block = (await worker.store.get_core_blocks("p"))["user_profile"]
        assert len(block) <= worker.config.core.block_cap  # never unbounded
        assert worker.status()["core_condensed"] == 0

    async def test_core_condense_rejects_over_compression(self, make_worker):
        # A model that replies "ok" would otherwise wipe the whole block.
        worker = await make_worker(
            llm=ScriptedLLM(["ok"]), config=make_config(core=self.SMALL_CAP)
        )
        await self._seed_over_cap(worker)
        await worker._maybe_update_core("p", {"text": "user is a teacher", "scope": "user"})
        block = (await worker.store.get_core_blocks("p"))["user_profile"]
        assert worker.status()["core_condensed"] == 0
        assert len(block) <= worker.config.core.block_cap
        assert block.strip() != "- ok"

    async def test_core_condense_disabled_by_config(self, make_worker):
        llm = ScriptedLLM(["- name is Aki\n- merged"])
        worker = await make_worker(
            llm=llm, config=make_config(core={"block_cap": 200, "condense_enabled": False})
        )
        await self._seed_over_cap(worker)
        await worker._maybe_update_core("p", {"text": "user is a teacher", "scope": "user"})
        assert llm.calls == 0  # no LLM turn taken at all
        block = (await worker.store.get_core_blocks("p"))["user_profile"]
        assert len(block) <= worker.config.core.block_cap

    async def test_core_condense_not_invoked_under_cap(self, make_worker):
        llm = ScriptedLLM([fact_json("user is a teacher", subject="user", scope="user")])
        worker = await make_worker(
            llm=llm, embedder=MapEmbedder({"user is a teacher": [1.0, 0.0, 0.0]})
        )
        await worker.consolidate("p", "I teach", "ok")
        assert llm.calls == 1  # extraction only; block is far under cap

    # -- status / lifecycle ---------------------------------------------------------
    async def test_status_tracks_counts_and_drain(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM([fact_json("user likes hiking", subject="user", scope="world")]),
            config=make_config(
                consolidation={"summary_every_turns": 0, "reflection_every_turns": 0},
                dedup=NO_DEDUP,
            ),
        )
        before = worker.status()
        assert before["enabled"]
        assert before["state"] == "idle"

        await worker.consolidate("p", "I love hiking", "nice")
        after = worker.status()
        assert after["facts_added"] == 1
        assert after["state"] == "idle"
        assert not after["busy"]
        assert worker.pending() == 0
        assert await worker.drain(timeout=1.0)

    async def test_disabled_extraction_is_noop(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM([fact_json("x")]),
            config=make_config(consolidation={"enabled": False}),
        )
        await worker.consolidate("p", "hello", "hi")
        assert len(await worker.store.recent("p")) == 0


class TestBatching:
    async def test_flushes_at_batch_size(self, make_worker):
        worker = await make_worker(config=make_config(consolidation={"batch_size": 8}))
        for i in range(7):
            worker.enqueue("p", f"u{i}", "a", session_id="c1")
        assert submitted(worker) == []  # still buffered, no call yet
        worker.enqueue("p", "u7", "a", session_id="c1")
        batches = submitted(worker)
        assert len(batches) == 1
        assert len(batches[0][1]) == 8

    async def test_flushes_at_char_cap(self, make_worker):
        worker = await make_worker(
            config=make_config(consolidation={"batch_size": 100, "max_batch_chars": 50})
        )
        worker.enqueue("p", "x" * 20, "y" * 20, session_id="c1")  # 40 chars, under the cap
        assert submitted(worker) == []
        worker.enqueue("p", "z" * 20, "", session_id="c1")  # 60 total, over
        assert len(submitted(worker)[0][1]) == 2
        # One oversized exchange goes out alone rather than dragging the next
        # seven into an enormous prompt.
        worker.enqueue("p", "q" * 500, "", session_id="c1")
        assert len(submitted(worker)[0][1]) == 1

    async def test_session_switch_flushes_the_previous_session(self, make_worker):
        worker = await make_worker(config=make_config(consolidation={"batch_size": 8}))
        worker.enqueue("p", "u1", "a", session_id="c1")
        worker.enqueue("p", "u2", "a", session_id="c1")
        assert submitted(worker) == []
        worker.enqueue("p", "u3", "a", session_id="c2")
        batches = submitted(worker)
        assert len(batches) == 1
        assert [user for user, _ in batches[0][1]] == ["u1", "u2"]
        assert worker._buffered() == 1  # the new session's turn is still held

    async def test_flush_buffers_submits_every_partial_buffer(self, make_worker):
        worker = await make_worker(config=make_config(consolidation={"batch_size": 8}))
        worker.enqueue("p", "u1", "a", session_id="c1")
        worker.enqueue("q", "u2", "a", session_id="c1")
        assert submitted(worker) == []
        worker.flush_buffers()
        assert sorted(scope for scope, _ in submitted(worker)) == ["p", "q"]
        assert worker._buffered() == 0

    async def test_pending_counts_buffered_exchanges(self, make_worker):
        # Without this, drain() reports success while the tail of the session
        # sits in RAM — and the host shuts down on that answer.
        worker = await make_worker(config=make_config(consolidation={"batch_size": 8}))
        for i in range(3):
            worker.enqueue("p", f"u{i}", "a", session_id="c1")
        assert worker.pending() == 3
        assert worker.status()["pending"] == 3

    async def test_drain_flushes_a_partial_batch_through_the_worker(self, make_worker):
        worker = await make_worker(
            llm=ScriptedLLM([fact_json("user likes hiking", subject="user", scope="world")]),
            config=make_config(
                consolidation={
                    "batch_size": 8,
                    "summary_every_turns": 0,
                    "reflection_every_turns": 0,
                }
            ),
        )
        worker.start()
        worker.enqueue("p", "I love hiking", "nice", session_id="c1")
        assert await worker.drain(timeout=5.0)
        assert worker.status()["facts_added"] == 1
        assert worker.pending() == 0

    async def test_batch_crossing_still_fires_the_summary(self, make_worker):
        # 6 then 8 exchanges takes the counter 6 -> 14 with summary_every=8,
        # which `count % every == 0` steps straight over: the stage would never
        # fire again.
        class CountingLLM:
            def __init__(self):
                self.n = 0
                self.summaries = 0

            async def complete(self, messages, *, temperature=None, max_tokens=None):
                self.n += 1
                if "summar" in messages[0]["content"].lower():
                    self.summaries += 1
                    return "A concise summary of the user."
                return json.dumps(
                    [
                        {
                            "text": f"distinct fact {self.n}.{k}",
                            "subject": "user",
                            "importance": 5,
                            "scope": "world",
                            "entities": [],
                        }
                        for k in range(3)
                    ]
                )

        llm = CountingLLM()
        worker = await make_worker(
            llm=llm,
            config=make_config(
                consolidation={"summary_every_turns": 8, "reflection_every_turns": 0},
                dedup=NO_DEDUP,
            ),
        )
        await worker._process_batch("p", [(f"a{i}", "x") for i in range(6)])
        assert llm.summaries == 0
        await worker._process_batch("p", [(f"b{i}", "x") for i in range(8)])
        assert llm.summaries == 1
        assert await worker.store.bump_counter("p", "consolidation_turns", 0) == 14


class TestStagePredicates:
    async def test_predicates_decide_without_an_llm_call(self, make_worker):
        llm = ScriptedLLM(["must not be called"])
        worker = await make_worker(llm=llm)
        assert await worker._summary_inputs("p") is None
        assert await worker._reflect_inputs("p") is None
        assert llm.calls == 0

    async def test_reflect_inputs_skips_when_nothing_is_newer_than_the_last_insight(
        self, make_worker
    ):
        worker = await make_worker()
        store = worker.store
        for i in range(6):
            await store.add("p", f"durable fact {i}", mem_type="fact", subject="user")
        assert await worker._reflect_inputs("p") is not None
        await store.add("p", "an inference", mem_type="insight", subject="insight")
        assert await worker._reflect_inputs("p") is None
        await store.add("p", "something new", mem_type="fact", subject="user")
        assert await worker._reflect_inputs("p") is not None


class TestNoveltyGate:
    async def test_exact_repeat_is_skipped(self, make_worker):
        worker = await make_worker(
            embedder=MapEmbedder({"hello": [1.0, 0.0, 0.0]}),
            config=make_config(consolidation={"gate_mode": "on"}),
        )
        assert len(await worker._gate("p", [("hello", "hi")])) == 1
        assert await worker._gate("p", [(" Hello ", "hi again")]) == []
        assert worker.status()["gate_skipped"] == 1

    async def test_novel_utterance_passes(self, make_worker):
        worker = await make_worker(
            embedder=MapEmbedder(
                {"hello": [1.0, 0.0, 0.0], "my sister is Rin": [0.0, 1.0, 0.0]}
            ),
            config=make_config(consolidation={"gate_mode": "on"}),
        )
        await worker._gate("p", [("hello", "hi")])
        assert len(await worker._gate("p", [("my sister is Rin", "oh!")])) == 1
        assert worker.status()["gate_skipped"] == 0

    async def test_near_duplicate_is_skipped_by_cosine(self, make_worker):
        worker = await make_worker(
            embedder=MapEmbedder(
                {"good morning": [1.0, 0.0, 0.0], "morning!": [0.99, 0.141, 0.0]}
            ),
            config=make_config(consolidation={"gate_mode": "on"}),
        )
        await worker._gate("p", [("good morning", "hi")])
        assert await worker._gate("p", [("morning!", "hi")]) == []

    async def test_log_mode_counts_but_never_skips(self, make_worker):
        worker = await make_worker(
            embedder=MapEmbedder({"hello": [1.0, 0.0, 0.0]}),
            config=make_config(consolidation={"gate_mode": "log"}),
        )
        await worker._gate("p", [("hello", "hi")])
        assert len(await worker._gate("p", [("hello", "hi")])) == 1
        assert worker.status()["gate_would_skip"] == 1
        assert worker.status()["gate_skipped"] == 0

    async def test_off_mode_disables_the_gate(self, make_worker):
        worker = await make_worker(
            embedder=MapEmbedder({"hello": [1.0, 0.0, 0.0]}),
            config=make_config(consolidation={"gate_mode": "off"}),
        )
        await worker._gate("p", [("hello", "hi")])
        assert len(await worker._gate("p", [("hello", "hi")])) == 1
        assert worker.status()["gate_would_skip"] == 0
        assert worker.status()["gate_unavailable"] == 0

    async def test_fails_open_when_the_embedder_is_not_semantic(self, make_worker):
        # The hash embedder is what the store falls back to when no semantic
        # model is available. Its cosine carries no meaning, so the gate must
        # extract rather than skip — even for an utterance the exact-match leg
        # would otherwise have caught.
        worker = await make_worker(
            embedder=HashEmbedder(),
            config=make_config(consolidation={"gate_mode": "on"}),
        )
        assert len(await worker._gate("p", [("hello", "hi"), ("hello", "hi")])) == 2
        assert worker.status()["gate_unavailable"] == 1

    async def test_fully_gated_batch_costs_no_llm_call(self, make_worker):
        llm = ScriptedLLM(["[]"])
        worker = await make_worker(
            llm=llm,
            embedder=MapEmbedder({"hello": [1.0, 0.0, 0.0]}),
            config=make_config(
                consolidation={
                    "gate_mode": "on",
                    "summary_every_turns": 0,
                    "reflection_every_turns": 0,
                }
            ),
        )
        await worker._process_batch("p", [("hello", "hi")])
        assert llm.calls == 1
        await worker._process_batch("p", [("hello", "hi")])
        assert llm.calls == 1  # gated out entirely — no extraction call
        # The turns still happened, so cadence keeps counting them.
        assert await worker.store.bump_counter("p", "consolidation_turns", 0) == 2


class TestDiscardBufferedExchanges:
    """Erasing a conversation must drop its un-extracted buffer tail with it —
    content the user just deleted must not be extracted into memory minutes
    later."""

    def _worker(self) -> Consolidator:
        cfg = make_config(consolidation={"batch_size": 50})  # nothing flushes mid-test
        return Consolidator(cfg, store=None, llm=None)  # type: ignore[arg-type]

    def test_discarding_a_session_drops_its_buffered_tail(self):
        # The realistic shape: the user is IN a conversation (its exchanges
        # buffered) and deletes it. Only one session per scope can be buffered
        # at a time — enqueue() flushes the previous session's buffer on a
        # switch — so the deleted conversation's tail is exactly what discard
        # removes.
        worker = self._worker()
        worker.enqueue("p1", "erase this", "reply", session_id="doomed")
        worker.enqueue("p1", "and this", "reply", session_id="doomed")
        assert worker.pending() == 2
        assert worker.discard("p1", "doomed") == 2
        assert worker.pending() == 0, "the erased conversation's tail was kept"
        # A different session's buffer is untouched by a later targeted discard.
        worker.enqueue("p1", "other room", "reply", session_id="safe")
        assert worker.discard("p1", "doomed") == 0
        assert worker.pending() == 1

    def test_discarding_a_scope_leaves_other_scopes_alone(self):
        worker = self._worker()
        worker.enqueue("p1", "a", "r", session_id="one")
        worker.enqueue("p2", "c", "r", session_id="one")
        assert worker.discard("p1") == 1
        assert worker.pending() == 1, "another scope's buffer must survive"
