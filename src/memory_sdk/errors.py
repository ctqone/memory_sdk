"""Exception hierarchy.

Kept deliberately small: callers should be able to catch ``MemorySDKError`` and
know they have seen everything this library raises on purpose. Programming
errors (bad arguments, misuse) raise the ordinary builtins instead.
"""

from __future__ import annotations


class MemorySDKError(Exception):
    """Base class for every error MemorySDK raises deliberately."""


class ConfigurationError(MemorySDKError):
    """The client was constructed or called in a way its configuration cannot
    support (e.g. ``add()`` without an LLM backend to consolidate with)."""


class LLMError(MemorySDKError):
    """An LLM completion failed after retries. Consolidation stages treat this
    as "the stage produced nothing" rather than crashing the worker."""


class StoreVersionError(MemorySDKError):
    """The database on disk was written by a NEWER schema than this library
    understands. Refusing to open is deliberate: an older library writing into
    a newer schema is how data quietly rots. Upgrade the library instead."""
