"""End-to-end tests through the public ``Memory`` client."""

from __future__ import annotations

import pytest

from fakes import MapEmbedder, ScriptedLLM, fact_json
from memory_sdk import ConfigurationError, Memory, MemoryConfig


def _config(tmp_path, **sections) -> MemoryConfig:
    return MemoryConfig(db_path=tmp_path / "mem.sqlite3", **sections)


async def test_add_requires_an_llm(tmp_path):
    async with Memory(_config(tmp_path), embedder=MapEmbedder({})) as mem:
        with pytest.raises(ConfigurationError):
            await mem.add("p", "hello", "hi")


async def test_add_flush_search_render_round_trip(tmp_path):
    llm = ScriptedLLM([fact_json("user is a teacher", subject="user", scope="user")])
    embedder = MapEmbedder({"user is a teacher": [1.0, 0.0, 0.0], "teacher?": [1.0, 0.0, 0.0]})
    async with Memory(_config(tmp_path), llm=llm, embedder=embedder) as mem:
        await mem.add("p", "I teach high school", "Nice!")
        assert await mem.flush(timeout=5.0)
        result = await mem.search("p", "teacher?")
        assert "user is a teacher" in [m.content for m in result.memories]
        assert "user is a teacher" in mem.render(result)
        # Core memory picked the fact up too (scope "user").
        assert "user is a teacher" in (await mem.get_core("p"))["user_profile"]
        assert "user is a teacher" in await mem.render_core("p")


async def test_aclose_drains_buffered_exchanges(tmp_path):
    # One exchange, batch_size 8: nothing has flushed when the context exits —
    # a clean shutdown must consolidate it anyway, not drop it.
    llm = ScriptedLLM([fact_json("user likes hiking", subject="user", scope="world")])
    mem = Memory(_config(tmp_path), llm=llm, embedder=MapEmbedder({}))
    async with mem:
        await mem.add("p", "I love hiking", "nice")
    assert llm.calls == 1
    # Reopen the same database: the fact persisted.
    async with Memory(_config(tmp_path), embedder=MapEmbedder({})) as reopened:
        assert [m.content for m in await reopened.list("p")] == ["user likes hiking"]


async def test_search_includes_graph_expansion(tmp_path):
    mapping = {
        "the user adopted a dog named Max": [1.0, 0.0, 0.0],
        "Max needs a Saturday walk": [0.0, 1.0, 0.0],
        "dog?": [1.0, 0.0, 0.0],
    }
    async with Memory(_config(tmp_path), embedder=MapEmbedder(mapping)) as mem:
        await mem.remember("p", "the user adopted a dog named Max", subject="user")
        await mem.remember("p", "Max needs a Saturday walk", subject="Max")
        await mem.store.add_cooccurrence("p", "user", "Max")
        result = await mem.search("p", "dog?")
        recalled = [m.content for m in result.memories]
        assert "the user adopted a dog named Max" in recalled
        # The graph leg carries the connected fact without duplicating recall.
        graph = [m.content for m in result.graph]
        assert "Max needs a Saturday walk" in graph or "Max needs a Saturday walk" in recalled
        assert not (set(graph) & set(recalled))
        assert "Related context:" in mem.render(result) or not graph


async def test_remember_and_supersede(tmp_path):
    async with Memory(_config(tmp_path), embedder=MapEmbedder({})) as mem:
        old = await mem.remember("p", "user lives in Tokyo", subject="user")
        new = await mem.remember("p", "user lives in Osaka", subject="user")
        await mem.supersede(old.id)
        current = [m.content for m in await mem.list("p")]
        assert current == ["user lives in Osaka"]
        assert new.id != old.id


async def test_clear_discards_buffered_exchanges_too(tmp_path):
    llm = ScriptedLLM(["[]"])
    async with Memory(_config(tmp_path), llm=llm, embedder=MapEmbedder({})) as mem:
        await mem.remember("p", "a stored fact")
        await mem.add("p", "erase this before it is extracted", "ok", session_id="doomed")
        assert mem.consolidator.pending() == 1
        await mem.clear("p")
        assert mem.consolidator.pending() == 0
        assert await mem.list("p") == []
    assert llm.calls == 0  # the buffered exchange never reached the LLM


async def test_status_reports_embedder(tmp_path):
    async with Memory(_config(tmp_path), embedder=MapEmbedder({})) as mem:
        status = mem.status()
        assert status["embedder"] == "map"
        assert status["embedder_semantic"] is True
        assert status["state"] == "idle"


async def test_set_core_merges_and_clear_core(tmp_path):
    async with Memory(_config(tmp_path), embedder=MapEmbedder({})) as mem:
        await mem.set_core("p", {"user_profile": "- name is Aki"})
        saved = await mem.set_core("p", {"relationship": "- friendly"})
        assert saved["user_profile"] == "- name is Aki"
        await mem.clear_core("p")
        assert await mem.render_core("p") == ""
