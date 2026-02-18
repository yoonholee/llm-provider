"""Shared caching infrastructure and logger silencing."""

import hashlib
import json
import logging
import os
from pathlib import Path

import litellm
from diskcache import FanoutCache
from litellm.caching.caching import Cache

for name in ("openai", "httpx", "LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(name).setLevel(logging.WARNING)
litellm.suppress_debug_info = True

_cache_dir = os.environ.get("LLM_CACHE_DIR", None)
if _cache_dir is None:
    candidates = [Path("/iris/u/yoonho/.cache/llm_cache"), Path("/tmp/llm_cache")]
    _cache_dir = str(next((p for p in candidates if p.parent.exists()), candidates[-1]))

# litellm's own disk cache (used by litellm fallback path)
litellm.cache = Cache(type="disk", disk_cache_dir=_cache_dir)

# Shared disk cache for all direct SDK paths (Gemini, OpenAI, Together)
direct_cache = FanoutCache(str(Path(_cache_dir) / "direct"), shards=8)


def cache_key(model: str, prompt: str, system: str | None, config: dict) -> str:
    """Deterministic cache key for a direct SDK request."""
    blob = json.dumps(
        {"model": model, "prompt": prompt, "system": system, "config": config},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()
