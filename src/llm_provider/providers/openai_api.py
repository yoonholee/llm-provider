"""OpenAI provider via direct SDK with HTTP/2.

Also used by Together and local (OpenAI-compatible API).
Client creation is handled by _registry.py; this module provides
call(), call_messages_sync(), and shared response helpers.
"""

import re

from llm_provider._cache import cache_key, cache_key_messages, direct_cache
from llm_provider.pricing import cost as _pricing_cost

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (and unclosed <think>) from model output."""
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def model_id(model: str) -> str:
    """'openai/gpt-4.1-nano' -> 'gpt-4.1-nano'"""
    return model.removeprefix("openai/")


# --- Shared helpers for call() and call_messages_sync() ---

_REASONING_PREFIXES = ("o1", "o3", "o4")
_MAX_COMPLETION_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _prepare_kwargs(mid: str, kwargs: dict) -> tuple[dict, bool]:
    """Normalize kwargs for reasoning model compatibility. Returns (kwargs, use_cache)."""
    kwargs = dict(kwargs)
    use_cache = kwargs.pop("cache", True)

    # Newer OpenAI models require max_completion_tokens instead of max_tokens.
    # Reasoning models need much higher limits since reasoning tokens count.
    if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
        if any(mid.startswith(p) for p in _MAX_COMPLETION_PREFIXES):
            requested = kwargs.pop("max_tokens")
            kwargs["max_completion_tokens"] = max(requested, 32768)

    # Reasoning models reject temperature
    if any(mid.startswith(p) for p in _REASONING_PREFIXES):
        kwargs.pop("temperature", None)

    return kwargs, use_cache


def _parse_response(mid: str, response) -> tuple[list[str], dict]:
    """Extract texts and usage from OpenAI API response."""
    texts = [c.message.content or "" for c in response.choices]
    if not texts:
        texts = [""]

    # Strip <think> blocks when server hasn't already parsed them into
    # reasoning_content (i.e. vLLM without --enable-reasoning).
    has_reasoning_field = any(
        getattr(c.message, "reasoning_content", None) for c in response.choices
    )
    if not has_reasoning_field:
        texts = [strip_thinking(t) if "<think>" in t else t for t in texts]
    else:
        texts = [t.strip() for t in texts]

    usage: dict = {}
    if response.usage:
        usage["input_tokens"] = response.usage.prompt_tokens or 0
        usage["output_tokens"] = response.usage.completion_tokens or 0
        details = getattr(response.usage, "prompt_tokens_details", None)
        if details:
            usage["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

    c = _pricing_cost(
        mid,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    if c is not None:
        usage["cost"] = c

    return texts, usage


def _cache_lookup(key: str, use_cache: bool):
    """Check cache. Returns (texts, {}) on hit, None on miss."""
    if not use_cache:
        return None
    cached = direct_cache.get(key)
    if cached is None:
        return None
    if isinstance(cached, list):
        return cached, {}
    return [cached], {}


def _cache_store(key: str, texts: list[str], use_cache: bool):
    """Store response in cache if enabled."""
    if use_cache and texts[0]:
        if len(texts) == 1:
            direct_cache.set(key, texts[0])
        else:
            direct_cache.set(key, texts)


# --- Public API ---


async def call(
    client,
    model_id: str,
    prompt: str,
    system_prompt: str = "",
    **kwargs,
):
    """Returns (texts: list[str], usage: dict)."""
    kwargs, use_cache = _prepare_kwargs(model_id, kwargs)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    config_dict = {k: v for k, v in sorted(kwargs.items()) if v is not None}
    key = cache_key(model_id, prompt, system_prompt or None, config_dict)
    hit = _cache_lookup(key, use_cache)
    if hit is not None:
        return hit

    response = await client.chat.completions.create(
        model=model_id, messages=messages, **kwargs
    )

    texts, usage = _parse_response(model_id, response)
    _cache_store(key, texts, use_cache)
    return texts, usage


def call_messages_sync(client, model_id: str, messages: list, **kwargs):
    """Sync chat completion with pre-built messages. Returns (texts, usage)."""
    kwargs, use_cache = _prepare_kwargs(model_id, kwargs)

    config_dict = {k: v for k, v in sorted(kwargs.items()) if v is not None}
    key = cache_key_messages(model_id, messages, config_dict)
    hit = _cache_lookup(key, use_cache)
    if hit is not None:
        return hit

    response = client.chat.completions.create(
        model=model_id, messages=messages, **kwargs
    )

    texts, usage = _parse_response(model_id, response)
    _cache_store(key, texts, use_cache)
    return texts, usage
