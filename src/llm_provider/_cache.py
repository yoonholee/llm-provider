"""Shared caching infrastructure."""

import hashlib
import json
import logging
import os
from pathlib import Path

for name in ("openai", "httpx", "LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(name).setLevel(logging.WARNING)

_cache_dir = os.environ.get("LLM_CACHE_DIR", None)
if _cache_dir is None:
    candidates = [
        Path("/scr/yoonho/llm-cache"),
        Path("/iris/u/yoonho/.cache/llm_cache"),
        Path("/tmp/llm_cache"),
    ]
    _cache_dir = str(next((p for p in candidates if p.parent.exists()), candidates[-1]))


class _LazyCache:
    """Lazy-initialized FanoutCache (avoids 3s+ NFS penalty at import time)."""

    def __init__(self):
        self._cache = None

    def _ensure(self):
        if self._cache is None:
            from diskcache import FanoutCache

            self._cache = FanoutCache(str(Path(_cache_dir) / "direct"), shards=8)

    def get(self, key):
        self._ensure()
        return self._cache.get(key)

    def set(self, key, value):
        self._ensure()
        return self._cache.set(key, value)


direct_cache = _LazyCache()

_litellm_cache_initialized = False


def _ensure_litellm_cache():
    global _litellm_cache_initialized
    if _litellm_cache_initialized:
        return
    import litellm
    from litellm.caching.caching import Cache

    litellm.suppress_debug_info = True
    litellm.cache = Cache(type="disk", disk_cache_dir=_cache_dir)
    _litellm_cache_initialized = True


def cache_key(model: str, prompt: str, system: str | None, config: dict) -> str:
    blob = json.dumps(
        {"model": model, "prompt": prompt, "system": system, "config": config},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()
