"""
Response caching - saves API responses to avoid repeat calls
Supports memory, sqlite, and redis backends
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class CacheBackend(ABC):
    """Base cache interface"""

    @abstractmethod
    def get(self, key: str) -> str | None:
        ...

    @abstractmethod
    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        ...


class MemoryCache(CacheBackend):
    """Simple in-memory cache - fast but resets on restart"""

    def __init__(self, max_size: int = 10000):
        self._data: dict[str, tuple[str, float | None]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        if key in self._data:
            val, exp = self._data[key]
            if exp is None or time.time() < exp:
                self.hits += 1
                return val
            del self._data[key]
        self.misses += 1
        return None

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        # simple eviction - just remove oldest
        if len(self._data) >= self.max_size:
            oldest = next(iter(self._data))
            del self._data[oldest]
        exp = time.time() + ttl if ttl else None
        self._data[key] = (value, exp)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()
        self.hits = self.misses = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": "memory",
            "size": len(self._data),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0,
        }


class SQLiteCache(CacheBackend):
    """SQLite cache - persists to disk"""

    def __init__(self, db_path: str = ".cache/responses.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._setup_db()
        self.hits = 0
        self.misses = 0

    def _setup_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL
            )
        """)
        self._conn.commit()

    def get(self, key: str) -> str | None:
        cur = self._conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        if row:
            val, exp = row
            if exp is None or time.time() < exp:
                self.hits += 1
                return val
            self.delete(key)
        self.misses += 1
        return None

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        exp = time.time() + ttl if ttl else None
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (key, value, time.time(), exp)
        )
        self._conn.commit()

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()
        self.hits = self.misses = 0

    def stats(self) -> dict[str, Any]:
        cur = self._conn.execute("SELECT COUNT(*) FROM cache")
        size = cur.fetchone()[0]
        total = self.hits + self.misses
        return {
            "backend": "sqlite",
            "db_path": str(self.db_path),
            "size": size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0,
        }


class RedisCache(CacheBackend):
    """Redis cache - for production use"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, prefix: str = "lb:"):
        try:
            import redis
        except ImportError:
            raise ImportError("redis not installed. pip install redis")

        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.prefix = prefix
        self.hits = 0
        self.misses = 0

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> str | None:
        val = self._client.get(self._k(key))
        if val:
            self.hits += 1
            return val
        self.misses += 1
        return None

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        self._client.set(self._k(key), value, ex=ttl)

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(self._k(key)))

    def delete(self, key: str) -> None:
        self._client.delete(self._k(key))

    def clear(self) -> None:
        keys = self._client.keys(f"{self.prefix}*")
        if keys:
            self._client.delete(*keys)
        self.hits = self.misses = 0

    def stats(self) -> dict[str, Any]:
        keys = self._client.keys(f"{self.prefix}*")
        total = self.hits + self.misses
        return {
            "backend": "redis",
            "size": len(keys),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0,
        }


class ResponseCache:
    """
    Main cache class for caching LLM responses

    Example:
        cache = ResponseCache(backend="sqlite")
        resp = cache.get(model="llama", prompt="hello")
        if not resp:
            resp = call_api()
            cache.set(model="llama", prompt="hello", response=resp)
    """

    def __init__(self, backend: str = "sqlite", ttl: int | None = 604800, **kwargs):
        self.ttl = ttl  # default 7 days

        if backend == "memory":
            self._backend = MemoryCache(**kwargs)
        elif backend == "sqlite":
            self._backend = SQLiteCache(**kwargs)
        elif backend == "redis":
            self._backend = RedisCache(**kwargs)
        else:
            raise ValueError(f"unknown backend: {backend}")

    def _make_key(self, model: str, prompt: str, **kw) -> str:
        # hash the request to make a cache key
        data = {"model": model, "prompt": prompt, **kw}
        s = json.dumps(data, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    def get(self, model: str, prompt: str, **kw) -> str | None:
        key = self._make_key(model, prompt, **kw)
        return self._backend.get(key)

    def set(self, model: str, prompt: str, response: str, ttl: int | None = None, **kw) -> None:
        key = self._make_key(model, prompt, **kw)
        self._backend.set(key, response, ttl=ttl or self.ttl)

    def get_or_call(self, model: str, prompt: str, fn, ttl: int | None = None, **kw):
        """Get from cache or call fn() and cache result"""
        cached = self.get(model, prompt, **kw)
        if cached is not None:
            return cached, True
        result = fn()
        self.set(model, prompt, result, ttl=ttl, **kw)
        return result, False

    def stats(self) -> dict:
        return self._backend.stats()

    def clear(self) -> None:
        self._backend.clear()

    def print_stats(self) -> None:
        s = self.stats()
        print(f"\nCache Stats ({s['backend']}):")
        print(f"  Size: {s['size']}")
        print(f"  Hits: {s['hits']}, Misses: {s['misses']}")
        print(f"  Hit rate: {s['hit_rate']:.1%}")
