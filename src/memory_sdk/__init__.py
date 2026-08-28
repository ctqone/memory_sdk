"""MemorySDK — async-native layered memory for LLM agents.

Typed facts / rolling summaries / insights with per-type recall slots,
temporal supersession, an entity graph with 1-hop recall expansion, and
batched LLM consolidation behind a novelty gate. One SQLite file, no vector
database. See :class:`Memory` for the front door.
"""

from .config import (
    DEFAULT_ST_MODEL,
    ConsolidationConfig,
    CoreMemoryConfig,
    DedupConfig,
    MemoryConfig,
    PromptsConfig,
    RecallConfig,
)
from .consolidation import Consolidator
from .embedders import (
    Embedder,
    HashEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    resolve_embedder,
)
from .errors import ConfigurationError, LLMError, MemorySDKError, StoreVersionError
from .facade import Memory
from .llm import LLMBackend, OpenAICompatibleLLM
from .models import FactMatch, MemoryItem, RecallResult
from .render import render_memories
from .store import SQLiteMemoryStore, norm_subject

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "MemoryConfig",
    "RecallConfig",
    "DedupConfig",
    "ConsolidationConfig",
    "CoreMemoryConfig",
    "PromptsConfig",
    "DEFAULT_ST_MODEL",
    "MemoryItem",
    "FactMatch",
    "RecallResult",
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAICompatibleEmbedder",
    "resolve_embedder",
    "LLMBackend",
    "OpenAICompatibleLLM",
    "SQLiteMemoryStore",
    "Consolidator",
    "norm_subject",
    "render_memories",
    "MemorySDKError",
    "ConfigurationError",
    "LLMError",
    "StoreVersionError",
    "__version__",
]
