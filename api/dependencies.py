"""Shared dependencies: data loading, cache, thread pool."""

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any

import pandas as pd
from cachetools import TTLCache

from data.synthetic_generator import generate_synthetic_data

# ---------------------------------------------------------------------------
# Global data cache (thread-safe singleton)
# ---------------------------------------------------------------------------
_data_cache: dict[str, Any] = {}
_data_lock = threading.Lock()


def get_data() -> pd.DataFrame:
    if "df" not in _data_cache:
        with _data_lock:
            if "df" not in _data_cache:
                _data_cache["df"] = generate_synthetic_data()
    return _data_cache["df"]


# ---------------------------------------------------------------------------
# Cluster info cache (set by similarity, read by forecasting)
# ---------------------------------------------------------------------------
def set_cluster_info(info: dict) -> None:
    _data_cache["cluster_info"] = info


def get_cluster_info() -> dict | None:
    return _data_cache.get("cluster_info")


# ---------------------------------------------------------------------------
# TTL endpoint cache
# ---------------------------------------------------------------------------
_endpoint_cache = TTLCache(maxsize=100, ttl=600)
_cache_lock_ep = threading.Lock()


def cached_endpoint(ttl: int = 600):
    """Decorator that caches the return value of an endpoint function."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__module__}.{func.__name__}"
            with _cache_lock_ep:
                if key in _endpoint_cache:
                    return _endpoint_cache[key]
            result = await func(*args, **kwargs) if _is_coroutine(func) else func(*args, **kwargs)
            with _cache_lock_ep:
                _endpoint_cache[key] = result
            return result
        return wrapper
    return decorator


def clear_cache() -> None:
    _endpoint_cache.clear()


def _is_coroutine(func) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(func)


# ---------------------------------------------------------------------------
# Thread pool for CPU-bound work
# ---------------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=4)
