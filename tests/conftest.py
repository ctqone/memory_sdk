from __future__ import annotations

import itertools

import pytest_asyncio

from memory_sdk import MemoryConfig, SQLiteMemoryStore


@pytest_asyncio.fixture
async def make_store(tmp_path):
    """Factory for opened stores in the test's tmp dir, auto-closed.

    ``name`` selects the database file, so a test can reopen the same file
    with a second store, or create fully independent ones.
    """
    stores: list[SQLiteMemoryStore] = []
    counter = itertools.count()

    async def factory(embedder=None, config=None, name=None):
        cfg = config or MemoryConfig()
        filename = name or f"m{next(counter)}.sqlite3"
        store = SQLiteMemoryStore(tmp_path / filename, cfg, embedder)
        await store.open()
        stores.append(store)
        return store

    yield factory
    for store in stores:
        await store.close()
