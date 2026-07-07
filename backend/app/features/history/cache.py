"""
In-memory caching layer for History Engine.

Provides a lightweight, configurable cache to reduce repeated computations
and improve performance when querying historical match data.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import pandas as pd

from ...data.logger import PipelineLogger
from .exceptions import CacheError


MODULE_NAME = "HistoryCache"


class HistoryCache:
    """
    Lightweight in-memory cache for history queries.

    Caches frequently accessed history subsets to avoid repeated
    DataFrame filtering operations. Supports TTL-free simple eviction
    for memory efficiency.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize the cache.

        Args:
            max_size (int): Maximum number of cached entries. Defaults to 1000.

        Raises:
            ValueError: If max_size is less than 1.
        """
        if max_size < 1:
            raise ValueError("Cache max_size must be at least 1")

        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.logger = PipelineLogger.get_logger(MODULE_NAME)
        self.logger.info(f"Initialized cache with max_size={max_size}")

    def make_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Generate a cache key from arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            str: A unique cache key.
        """
        key_parts = [self._normalize_part(arg) for arg in args]
        key_parts.extend(
            f"{key}={self._normalize_part(value)}"
            for key, value in sorted(kwargs.items())
        )
        return "|".join(key_parts)

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Backward-compatible alias for older callers."""
        return self.make_key(*args, **kwargs)

    def _normalize_part(self, value: Any) -> str:
        """Normalize key parts into stable, hashable string fragments."""
        if isinstance(value, dict):
            items = sorted((key, self._normalize_part(item)) for key, item in value.items())
            return str(tuple(items))
        if isinstance(value, (list, tuple, set)):
            return str(tuple(self._normalize_part(item) for item in value))
        return str(value)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key (str): Cache key.

        Returns:
            Optional[Any]: Cached value or None if not found.
        """
        value = self._cache.get(key)
        if value is None:
            return None

        self._cache.move_to_end(key)
        if isinstance(value, pd.DataFrame):
            return value.copy()
        return value

    def put(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Evicts oldest entry if cache is full (simple FIFO eviction).

        Args:
            key (str): Cache key.
            value (Any): Value to cache.

        Raises:
            CacheError: If value cannot be cached.
        """
        if not isinstance(key, str) or not key:
            raise CacheError("Cache key must be a non-empty string")

        cache_value = value.copy() if isinstance(value, pd.DataFrame) else value

        if key in self._cache:
            self._cache[key] = cache_value
            self._cache.move_to_end(key)
            return

        if len(self._cache) >= self.max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self.logger.debug(f"Evicted cache entry: {oldest_key}")

        self._cache[key] = cache_value

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self.logger.info("Cache cleared")

    def size(self) -> int:
        """Return current number of cached entries."""
        return len(self._cache)

    def describe(self) -> str:
        """Return cache statistics as a string."""
        return f"HistoryCache(size={self.size()}/{self.max_size})"
