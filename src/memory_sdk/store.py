"""The SQLite store: typed, temporal, layered memory plus the entity graph.

Single-file SQLite (via ``aiosqlite``), no vector database. Design points that
are all load-bearing, each learned the hard way in the ancestor codebase:

- **Typed rows** (``fact`` / ``summary`` / ``insight``) with recall slots
  reserved per type — see :meth:`SQLiteMemoryStore.search`.
- **Temporal supersession**: a corrected fact supersedes its predecessor
  (``valid_to`` set) rather than overwriting it; recall hides superseded rows
  by default and the history stays auditable.
- **Model-tagged embeddings**: cosine is only computed between vectors from
  the same embedding backend. Anything else compares incompatible vector
  spaces; incomparable pairs degrade to a bounded lexical score instead of a
  silent flat 0.0 (which once disabled dedup entirely across an embedder
  change).
- **Python-side subject normalization** (:func:`norm_subject`), never SQL
  ``LOWER()`` — SQLite's ``LOWER`` is ASCII-only, so non-ASCII subjects would
  permanently disagree between writers and readers. Every subject comparison
  in this module goes through the ``subject_norm`` column.
- **Durability pragmas**: WAL + ``synchronous=FULL`` (memories are not
  regenerable), and ``secure_delete=ON`` so a user delete is byte-level
  erasure, not just a free-list entry.

Schema versioning: the ``meta`` table records ``schema_version``. Opening a
database written by a NEWER version raises :class:`StoreVersionError` — this
library migrates old databases forward but never writes into a schema it does
not understand. One-shot data migrations (as opposed to additive ``ALTER
TABLE``) also record themselves in ``meta`` so they cannot re-fire and clobber
later edits.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from hashlib import sha1
from pathlib import Path

import aiosqlite

from .config import MemoryConfig
from .embedders import Embedder, HashEmbedder, cosine, tokens
from .errors import StoreVersionError
from .models import FactMatch, MemoryItem

SCHEMA_VERSION = 1

_ITEM_COLS = (
    "id, scope, content, timestamp, mem_type, importance, subject, "
    "valid_from, valid_to, supersedes"
)


def norm_subject(name: str) -> str:
    """Canonical form of a subject / entity label.

    The entity graph is always keyed on this, and facts store it alongside the
    raw label (``subject_norm``). Import this rather than re-implementing it —
    two normalizers is how "User" and "user " become different people.
    """
    return (name or "").strip().lower()


def _lexical_recall_score(
    query_clean: str, query_tokens: set, content_clean: str, content_tokens: set
) -> float:
    """The lexical leg used by ``search()`` for RANKING: 1.0 for a verbatim
    substring hit, plus the fraction of QUERY tokens present in the content.

    Deliberately asymmetric (range 0..2): a short query matched against long
    content should not be penalised for the content's extra words. That is
    right for ranking and wrong for dedup — see :func:`lexical_similarity`.
    """
    score = 0.0
    if query_clean and query_clean in content_clean:
        score += 1.0
    if query_tokens and content_tokens:
        score += len(query_tokens & content_tokens) / len(query_tokens)
    return score


def lexical_similarity(left: str, right: str) -> float:
    """Bounded 0..1 lexical similarity, used for DEDUP when two rows'
    embeddings are not comparable (different embed_model or dimensionality).

    Sørensen-Dice over token sets. Symmetric, and — unlike the ranking leg
    above — NOT 1.0 when one text is a strict superset of the other, which is
    precisely the property dedup needs: "user likes tea" must not swallow
    "user likes tea in the morning after a run". Roughly comparable in scale
    to cosine so the same threshold family applies, though callers hold it to
    a stricter bar because it is the noisier signal.
    """
    a, b = left.strip().lower(), right.strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    left_tokens, right_tokens = set(tokens(a)), set(tokens(b))
    if not left_tokens or not right_tokens:
        return 0.0
    return 2.0 * len(left_tokens & right_tokens) / (len(left_tokens) + len(right_tokens))


def _row_to_item(row) -> MemoryItem:
    return MemoryItem(
        id=row[0],
        scope=row[1],
        content=row[2],
        timestamp=row[3],
        mem_type=row[4] or "fact",
        importance=row[5] if row[5] is not None else 5.0,
        subject=row[6] or "",
        valid_from=row[7] or 0.0,
        valid_to=row[8],
        supersedes=row[9],
    )


def _parse_embedding(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(value) for value in raw.split(",") if value]


class SQLiteMemoryStore:
    """The store. Construct, then ``await open()`` (idempotent) before use —
    or use it through :class:`memory_sdk.Memory`, which does both."""

    def __init__(
        self,
        path: Path | str,
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.path = Path(path)
        self.config = config or MemoryConfig()
        self.embedder = embedder if embedder is not None else HashEmbedder()
        self.fallback = HashEmbedder()
        self.db: aiosqlite.Connection | None = None
        # aiosqlite serializes statements on one worker thread, but a
        # read-modify-write spanning several statements can still interleave
        # between coroutines — this lock keeps those sections atomic.
        self._write_lock = asyncio.Lock()
        self._open_lock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------
    async def open(self) -> None:
        async with self._open_lock:
            if self.db is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = await aiosqlite.connect(self.path)
            try:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=FULL")
                await db.execute("PRAGMA secure_delete=ON")
                await self._init_schema(db)
            except BaseException:
                await db.close()
                raise
            self.db = db

    async def close(self) -> None:
        if self.db is not None:
            await self.db.close()
            self.db = None

    def _conn(self) -> aiosqlite.Connection:
        if self.db is None:
            raise RuntimeError("store is not open — await open() first")
        return self.db

    # -- schema --------------------------------------------------------------
    async def _init_schema(self, db: aiosqlite.Connection) -> None:
        # meta first: the version gate must run before this library writes
        # anything into a database a newer version may own.
        await db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        cursor = await db.execute("SELECT value FROM meta WHERE key = 'schema_version'")
        row = await cursor.fetchone()
        if row is not None and int(row[0]) > SCHEMA_VERSION:
            raise StoreVersionError(
                f"database at {self.path} has schema version {row[0]}, "
                f"newer than this library's {SCHEMA_VERSION} — upgrade memory-sdk"
            )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
              id TEXT PRIMARY KEY,
              scope TEXT NOT NULL,
              content TEXT NOT NULL,
              timestamp REAL NOT NULL,
              embedding TEXT NOT NULL,
              embed_model TEXT NOT NULL DEFAULT '',
              mem_type TEXT NOT NULL DEFAULT 'fact',
              importance REAL NOT NULL DEFAULT 5.0,
              subject TEXT DEFAULT '',
              subject_norm TEXT DEFAULT '',
              valid_from REAL,
              valid_to REAL,
              supersedes TEXT,
              last_access REAL,
              access_count INTEGER DEFAULT 0
            )
            """
        )
        # Minimal temporal knowledge graph. Entities are deduplicated by
        # (scope, norm_name); edges carry the same valid_from/valid_to
        # temporal validity as facts.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
              id TEXT PRIMARY KEY,
              scope TEXT NOT NULL,
              name TEXT NOT NULL,
              norm_name TEXT NOT NULL,
              type TEXT DEFAULT '',
              summary TEXT DEFAULT '',
              timestamp REAL,
              mention_count INTEGER DEFAULT 1
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
              id TEXT PRIMARY KEY,
              scope TEXT NOT NULL,
              src TEXT NOT NULL,
              dst TEXT NOT NULL,
              kind TEXT NOT NULL DEFAULT 'co_occurrence',
              weight REAL DEFAULT 1.0,
              timestamp REAL,
              valid_from REAL,
              valid_to REAL
            )
            """
        )
        # Persistent per-scope counters (e.g. consolidated-exchange count) so
        # cadence like "summarize every N turns" survives a process restart
        # instead of resetting an in-memory tally.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS counters (
              scope TEXT NOT NULL,
              name TEXT NOT NULL,
              value INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY (scope, name)
            )
            """
        )
        # Tier-1 core memory: tiny always-in-context blocks, maintained by
        # consolidation and editable through the API. Snapshots hold the
        # pre-rewrite content whenever the LLM condenses a block, because
        # condensation is the one place this library rewrites (possibly
        # user-edited) text wholesale.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS core_memory (
              scope TEXT NOT NULL,
              block TEXT NOT NULL,
              content TEXT NOT NULL DEFAULT '',
              updated_at REAL,
              PRIMARY KEY (scope, block)
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS core_memory_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              scope TEXT NOT NULL,
              block TEXT NOT NULL,
              content TEXT NOT NULL,
              tag TEXT DEFAULT '',
              created_at REAL
            )
            """
        )
        await db.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope, mem_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_mem_subject ON memories(scope, subject_norm)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_ent_scope ON entities(scope, norm_name)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_edge_scope ON edges(scope, src)")
        if row is None:
            await db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        await db.commit()

    # -- one-shot data-migration bookkeeping ----------------------------------
    async def _migration_done(self, key: str) -> bool:
        cursor = await self._conn().execute("SELECT value FROM meta WHERE key = ?", (key,))
        return await cursor.fetchone() is not None

    async def _mark_migration(self, key: str) -> None:
        await self._conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(time.time()))
        )
        await self._conn().commit()

    # -- embedding helpers -----------------------------------------------------
    async def _embed_store(self, text: str) -> tuple[list[float], str]:
        """Embed for storage — may block to load a heavy model (consolidation
        runs off the interactive path, so it can afford to wait)."""
        if await self.embedder.load():
            try:
                return await self.embedder.encode(text), self.embedder.name
            except Exception:  # noqa: BLE001 - any embedder failure -> hash fallback
                pass
        return await self.fallback.encode(text), self.fallback.name

    async def _embed_query(self, text: str) -> tuple[list[float], str]:
        """Embed for recall — never blocks on a model load. If the model is not
        resident yet, use the hash fallback for this call (semantic recall
        degrades to lexical-only rather than stalling the caller's turn)."""
        if self.embedder.ready:
            try:
                return await self.embedder.encode(text), self.embedder.name
            except Exception:  # noqa: BLE001
                pass
        return await self.fallback.encode(text), self.fallback.name

    # -- writes ----------------------------------------------------------------
    async def add(
        self,
        scope: str,
        content: str,
        *,
        mem_type: str = "fact",
        importance: float = 5.0,
        subject: str = "",
        valid_from: float | None = None,
        valid_to: float | None = None,
        supersedes: str | None = None,
    ) -> MemoryItem:
        content = content.strip()
        now = time.time()
        vfrom = valid_from if valid_from is not None else now
        memory_id = uuid.uuid4().hex
        vector, embed_model = await self._embed_store(content)
        embedding = ",".join(f"{value:.6f}" for value in vector)
        async with self._write_lock:
            await self._conn().execute(
                """
                INSERT INTO memories
                  (id, scope, content, timestamp, embedding, embed_model, mem_type,
                   importance, subject, subject_norm, valid_from, valid_to,
                   supersedes, last_access, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    memory_id, scope, content, now, embedding, embed_model, mem_type,
                    importance, subject, norm_subject(subject), vfrom, valid_to,
                    supersedes, now,
                ),
            )
            await self._conn().commit()
        return MemoryItem(
            id=memory_id, scope=scope, content=content, timestamp=now,
            mem_type=mem_type, importance=importance, subject=subject,
            valid_from=vfrom, valid_to=valid_to, supersedes=supersedes,
        )

    async def bump_importance(self, memory_id: str, delta: float = 1.0, cap: float = 10.0) -> None:
        now = time.time()
        async with self._write_lock:
            await self._conn().execute(
                "UPDATE memories SET importance = MIN(?, importance + ?), "
                "last_access = ?, access_count = access_count + 1 WHERE id = ?",
                (cap, delta, now, memory_id),
            )
            await self._conn().commit()

    async def supersede(self, old_id: str, when: float | None = None) -> None:
        when = when if when is not None else time.time()
        async with self._write_lock:
            await self._conn().execute(
                "UPDATE memories SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
                (when, old_id),
            )
            await self._conn().commit()

    async def bump_counter(self, scope: str, name: str = "turns", delta: int = 1) -> int:
        """Atomically increment a persistent per-scope counter and return its
        new value. Drives restart-persistent consolidation cadence."""
        async with self._write_lock:
            await self._conn().execute(
                "INSERT INTO counters (scope, name, value) VALUES (?, ?, ?) "
                "ON CONFLICT(scope, name) DO UPDATE SET value = value + ?",
                (scope, name, delta, delta),
            )
            await self._conn().commit()
            cursor = await self._conn().execute(
                "SELECT value FROM counters WHERE scope = ? AND name = ?", (scope, name)
            )
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def _touch(self, ids: list[str]) -> None:
        if not ids:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in ids)
        async with self._write_lock:
            await self._conn().execute(
                f"UPDATE memories SET last_access = ?, access_count = access_count + 1 "
                f"WHERE id IN ({placeholders})",
                (now, *ids),
            )
            await self._conn().commit()

    # -- reads -----------------------------------------------------------------
    async def search(
        self,
        query: str,
        scope: str,
        limit: int = 6,
        include_superseded: bool = False,
        mem_types: tuple[str, ...] = ("fact", "summary", "insight"),
        per_type_limits: dict[str, int] | None = None,
    ) -> list[MemoryItem]:
        """Best-first recall.

        ``per_type_limits`` reserves slots per mem_type instead of running one
        ranked contest across all of them, and when given it REPLACES ``limit``
        as the budget (the caps are the budget). One ranked pool structurally
        starves the synthesis types: a short specific fact out-scores a long
        general summary against a short query, so facts take all the slots and
        insights none. Scoring still happens once over all rows — the caps are
        applied to the already-sorted list, costing no extra query embedding.
        """
        now = time.time()
        recall = self.config.recall
        halflife = max(1.0, float(recall.recency_halflife_days)) * 86400.0
        w_sim = float(recall.similarity_weight)
        w_rec = float(recall.recency_weight)
        w_imp = float(recall.importance_weight)
        w_lex = float(recall.lexical_weight)

        q_vector, q_model = await self._embed_query(query)
        query_clean = query.strip().lower()
        query_tokens = set(tokens(query_clean))

        type_ph = ",".join("?" for _ in mem_types) or "''"
        sql = (
            f"SELECT {_ITEM_COLS}, embed_model, embedding FROM memories "
            f"WHERE scope = ? AND mem_type IN ({type_ph})"
        )
        params: list[object] = [scope, *mem_types]
        if not include_superseded:
            sql += " AND (valid_to IS NULL OR valid_to > ?)"
            params.append(now)
        cursor = await self._conn().execute(sql, params)
        rows = await cursor.fetchall()

        scored: list[tuple[float, MemoryItem]] = []
        for row in rows:
            item = _row_to_item(row)
            embed_model = row[10]
            embedding = _parse_embedding(row[11])
            sim = 0.0
            if embed_model == q_model and len(embedding) == len(q_vector):
                sim = cosine(q_vector, embedding)
            recency = math.exp(-max(0.0, now - item.timestamp) / halflife)
            imp = max(0.0, min(item.importance, 10.0)) / 10.0
            content_clean = item.content.lower()
            lex = _lexical_recall_score(
                query_clean, query_tokens, content_clean, set(tokens(content_clean))
            )
            score = w_sim * sim + w_rec * recency + w_imp * imp + w_lex * lex
            if score > 0:
                item.score = score
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if per_type_limits:
            taken: dict[str, int] = {}
            top = []
            for _score, item in scored:
                if taken.get(item.mem_type, 0) >= per_type_limits.get(item.mem_type, 0):
                    continue
                top.append(item)
                taken[item.mem_type] = taken.get(item.mem_type, 0) + 1
        else:
            top = [item for _, item in scored[:limit]]
        await self._touch([item.id for item in top])
        return top

    async def similar_facts(
        self,
        scope: str,
        text: str,
        subject: str | None = None,
        limit: int = 5,
        mem_types: tuple[str, ...] = ("fact",),
        scope_subject: bool = True,
    ) -> list[FactMatch]:
        """Current (non-superseded) rows most similar to ``text``, best first.
        Used by consolidation for dedup / conflict detection.

        ``scope_subject=False`` drops the subject predicate entirely
        (``subject`` is then ignored) — near-identical TEXT means a duplicate
        whatever label the extractor happened to attach, and in practice it
        attaches several for one entity. Scoped matching uses ``subject_norm``,
        so "User" matches "user ".

        Each result carries how it was scored. When the stored row's embedding
        is not comparable to ours (different model, or different
        dimensionality) this yields a bounded lexical score flagged as such, so
        the caller can hold the noisier signal to a stricter bar — a flat 0.0
        here once silently disabled dedup across an embedding change,
        permanently, because the failed model latched into its failed state.
        """
        vector, model = await self._embed_store(text)
        type_ph = ",".join("?" for _ in mem_types) or "''"
        sql = (
            f"SELECT {_ITEM_COLS}, embed_model, embedding FROM memories "
            f"WHERE scope = ? AND mem_type IN ({type_ph}) AND valid_to IS NULL"
        )
        params: list[object] = [scope, *mem_types]
        if scope_subject and subject:
            sql += " AND subject_norm = ?"
            params.append(norm_subject(subject))
        cursor = await self._conn().execute(sql, params)
        rows = await cursor.fetchall()
        out: list[FactMatch] = []
        for row in rows:
            item = _row_to_item(row)
            embed_model = row[10]
            embedding = _parse_embedding(row[11])
            if embed_model == model and len(embedding) == len(vector):
                out.append(FactMatch(item, cosine(vector, embedding), "cosine"))
            else:
                out.append(FactMatch(item, lexical_similarity(text, item.content), "lexical"))
        out.sort(key=lambda match: match.score, reverse=True)
        return out[:limit]

    async def recent(
        self,
        scope: str,
        limit: int = 40,
        include_superseded: bool = False,
        mem_types: tuple[str, ...] = ("fact", "summary", "insight"),
        since: float | None = None,
    ) -> list[MemoryItem]:
        """Most recent rows, oldest-first. ``since`` restricts to rows written
        after that timestamp — the rolling summary uses it to fetch only what
        has been learned since the previous summary was written."""
        type_ph = ",".join("?" for _ in mem_types) or "''"
        sql = f"SELECT {_ITEM_COLS} FROM memories WHERE scope = ? AND mem_type IN ({type_ph})"
        params: list[object] = [scope, *mem_types]
        if not include_superseded:
            sql += " AND (valid_to IS NULL OR valid_to > ?)"
            params.append(time.time())
        if since is not None:
            sql += " AND timestamp > ?"
            params.append(since)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor = await self._conn().execute(sql, params)
        rows = list(await cursor.fetchall())
        return [_row_to_item(row) for row in reversed(rows)]

    async def clear(self, scope: str) -> int:
        """Hard-delete every memory, entity, edge and counter in ``scope``.
        With ``secure_delete=ON`` this is byte-level erasure. Core-memory
        blocks are separate on purpose — clear them via ``clear_core``."""
        async with self._write_lock:
            cursor = await self._conn().execute("DELETE FROM memories WHERE scope = ?", (scope,))
            await self._conn().execute("DELETE FROM entities WHERE scope = ?", (scope,))
            await self._conn().execute("DELETE FROM edges WHERE scope = ?", (scope,))
            await self._conn().execute("DELETE FROM counters WHERE scope = ?", (scope,))
            await self._conn().commit()
        return cursor.rowcount

    async def delete(self, scope: str, ids) -> int:
        id_list = [str(i) for i in ids if i]
        if not id_list:
            return 0
        placeholders = ",".join("?" for _ in id_list)
        async with self._write_lock:
            cursor = await self._conn().execute(
                f"DELETE FROM memories WHERE scope = ? AND id IN ({placeholders})",
                (scope, *id_list),
            )
            await self._conn().commit()
        return cursor.rowcount

    # -- entity graph ----------------------------------------------------------
    async def upsert_entity(self, scope: str, name: str, type_: str = "", summary: str = "") -> str:
        norm = norm_subject(name)
        if not norm:
            return ""
        now = time.time()
        # Deterministic id: the (scope, normalized name) pair IS the identity.
        entity_id = sha1(f"{scope}:{norm}".encode()).hexdigest()
        async with self._write_lock:
            cursor = await self._conn().execute(
                "SELECT id FROM entities WHERE scope = ? AND norm_name = ?", (scope, norm)
            )
            existing = await cursor.fetchone()
            if existing:
                entity_id = existing[0]
                await self._conn().execute(
                    "UPDATE entities SET mention_count = mention_count + 1, timestamp = ?"
                    + (", summary = ?" if summary else "")
                    + (", type = ?" if type_ else "")
                    + " WHERE id = ?",
                    tuple(
                        [now]
                        + ([summary] if summary else [])
                        + ([type_] if type_ else [])
                        + [entity_id]
                    ),
                )
            else:
                await self._conn().execute(
                    "INSERT INTO entities (id, scope, name, norm_name, type, summary, timestamp, mention_count)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                    (entity_id, scope, name.strip(), norm, type_, summary, now),
                )
            await self._conn().commit()
        return entity_id

    async def add_cooccurrence(
        self, scope: str, name_a: str, name_b: str, kind: str = "co_occurrence"
    ) -> None:
        if norm_subject(name_a) == norm_subject(name_b):
            return
        src = await self.upsert_entity(scope, name_a)
        dst = await self.upsert_entity(scope, name_b)
        if not src or not dst:
            return
        # Undirected: one edge keyed on the sorted pair, so repeats bump weight.
        lo, hi = sorted([src, dst])
        edge_id = sha1(f"{scope}:{kind}:{lo}:{hi}".encode()).hexdigest()
        now = time.time()
        async with self._write_lock:
            cursor = await self._conn().execute("SELECT id FROM edges WHERE id = ?", (edge_id,))
            existing = await cursor.fetchone()
            if existing:
                await self._conn().execute(
                    "UPDATE edges SET weight = weight + 1.0, timestamp = ? WHERE id = ?",
                    (now, edge_id),
                )
            else:
                await self._conn().execute(
                    "INSERT INTO edges (id, scope, src, dst, kind, weight, timestamp, valid_from, valid_to)"
                    " VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, NULL)",
                    (edge_id, scope, lo, hi, kind, now, now),
                )
            await self._conn().commit()

    async def _entity_ids_for_subjects(self, scope: str, subjects: list[str]) -> list[str]:
        norms = [norm_subject(s) for s in subjects if s and s.strip()]
        if not norms:
            return []
        placeholders = ",".join("?" for _ in norms)
        cursor = await self._conn().execute(
            f"SELECT id FROM entities WHERE scope = ? AND norm_name IN ({placeholders})",
            (scope, *norms),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def graph_expand(self, scope: str, subjects: list[str], limit: int = 4) -> list[MemoryItem]:
        """Seed on ``subjects`` (entity names), walk one hop along
        non-superseded edges, and return current facts about the neighbor
        entities — connected-but-not-similar context a flat vector search
        misses."""
        seed_ids = await self._entity_ids_for_subjects(scope, subjects)
        if not seed_ids:
            return []
        seed_set = set(seed_ids)
        placeholders = ",".join("?" for _ in seed_ids)
        cursor = await self._conn().execute(
            f"SELECT src, dst FROM edges WHERE scope = ? AND valid_to IS NULL "
            f"AND (src IN ({placeholders}) OR dst IN ({placeholders}))",
            (scope, *seed_ids, *seed_ids),
        )
        rows = await cursor.fetchall()
        neighbor_ids: set[str] = set()
        for src, dst in rows:
            for node in (src, dst):
                if node not in seed_set:
                    neighbor_ids.add(node)
        if not neighbor_ids:
            return []
        nid_ph = ",".join("?" for _ in neighbor_ids)
        cursor = await self._conn().execute(
            f"SELECT norm_name FROM entities WHERE id IN ({nid_ph})", tuple(neighbor_ids)
        )
        name_rows = await cursor.fetchall()
        neighbor_norms = [row[0] for row in name_rows]
        if not neighbor_norms:
            return []
        norm_ph = ",".join("?" for _ in neighbor_norms)
        # subject_norm, NOT LOWER(subject): SQLite's LOWER is ASCII-only, so a
        # padded or full-width subject would never match the Python-normalized
        # entity name and its facts would silently fall out of expansion — the
        # one leg whose whole job is surfacing what vector search misses.
        cursor = await self._conn().execute(
            f"SELECT {_ITEM_COLS} FROM memories WHERE scope = ? AND mem_type = 'fact' "
            f"AND valid_to IS NULL AND subject_norm IN ({norm_ph}) "
            f"ORDER BY importance DESC, timestamp DESC LIMIT ?",
            (scope, *neighbor_norms, limit),
        )
        fact_rows = await cursor.fetchall()
        return [_row_to_item(row) for row in fact_rows]

    async def entities(self, scope: str, limit: int = 100) -> list[dict[str, object]]:
        cursor = await self._conn().execute(
            "SELECT name, type, summary, mention_count FROM entities WHERE scope = ? "
            "ORDER BY mention_count DESC, timestamp DESC LIMIT ?",
            (scope, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"name": row[0], "type": row[1] or "", "summary": row[2] or "", "mentions": int(row[3] or 0)}
            for row in rows
        ]

    # -- core memory (Tier 1) --------------------------------------------------
    async def get_core_blocks(self, scope: str) -> dict[str, str]:
        """The configured blocks for ``scope``, empty string when unset."""
        known = list(self.config.core.blocks)
        cursor = await self._conn().execute(
            "SELECT block, content FROM core_memory WHERE scope = ?", (scope,)
        )
        rows = await cursor.fetchall()
        stored = {row[0]: row[1] for row in rows}
        return {block: str(stored.get(block, "")).strip() for block in known}

    async def set_core_blocks(self, scope: str, blocks: dict[str, object]) -> dict[str, str]:
        """Merge the given blocks over what's stored (only configured block
        names) and persist. Returns the full saved set."""
        current = await self.get_core_blocks(scope)
        now = time.time()
        async with self._write_lock:
            for block in self.config.core.blocks:
                if block in blocks and blocks[block] is not None:
                    current[block] = str(blocks[block]).strip()
                    await self._conn().execute(
                        "INSERT INTO core_memory (scope, block, content, updated_at) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(scope, block) DO UPDATE SET content = excluded.content, "
                        "updated_at = excluded.updated_at",
                        (scope, block, current[block], now),
                    )
            await self._conn().commit()
        return current

    async def snapshot_core_block(self, scope: str, block: str, tag: str) -> None:
        """Snapshot a block's current content before something rewrites it.

        Core memory is editable through the API and always in context, and LLM
        condensation is the one place this library rewrites possibly
        user-authored text wholesale — preserve what is about to be replaced
        rather than trusting the replacement. Best-effort: it must never block
        the write it precedes. Snapshots are pruned to the last 10 per
        (scope, block).
        """
        try:
            cursor = await self._conn().execute(
                "SELECT content FROM core_memory WHERE scope = ? AND block = ?", (scope, block)
            )
            row = await cursor.fetchone()
            if row is None or not row[0]:
                return
            async with self._write_lock:
                await self._conn().execute(
                    "INSERT INTO core_memory_snapshots (scope, block, content, tag, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (scope, block, row[0], tag, time.time()),
                )
                await self._conn().execute(
                    "DELETE FROM core_memory_snapshots WHERE scope = ? AND block = ? AND id NOT IN ("
                    "SELECT id FROM core_memory_snapshots WHERE scope = ? AND block = ? "
                    "ORDER BY id DESC LIMIT 10)",
                    (scope, block, scope, block),
                )
                await self._conn().commit()
        except Exception:  # noqa: BLE001 - best-effort by contract
            pass

    async def clear_core(self, scope: str) -> None:
        async with self._write_lock:
            await self._conn().execute("DELETE FROM core_memory WHERE scope = ?", (scope,))
            await self._conn().execute(
                "DELETE FROM core_memory_snapshots WHERE scope = ?", (scope,)
            )
            await self._conn().commit()

    def render_core(self, blocks: dict[str, str]) -> str:
        """Render non-empty blocks for injection into a system prompt. Empty
        string when nothing is known yet (so it contributes no prompt noise)."""
        headings = self.config.core.blocks
        parts = [f"{headings[name]}:\n{blocks[name]}" for name in headings if blocks.get(name)]
        if not parts:
            return ""
        return "Core memory (durable — always keep these in mind):\n\n" + "\n\n".join(parts)
