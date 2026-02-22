"""OpenAI provider via direct SDK with HTTP/2.

Also used by Together and local (OpenAI-compatible API).
"""

import os
import re
import time

import httpx

from llm_provider._cache import cache_key, cache_key_messages, direct_cache
from llm_provider.pricing import cost as _pricing_cost

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (and unclosed <think>) from model output."""
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def create_client(max_retries: int = 2, on_headers=None):
    from openai import AsyncOpenAI

    from llm_provider.providers._pool import ClientPool

    event_hooks = {}
    if on_headers:
        from llm_provider.providers._headers import parse_openai_headers

        async def _hook(response: httpx.Response):
            info = parse_openai_headers(response.headers)
            if info:
                on_headers(info["remaining"], info["limit"])

        event_hooks["response"] = [_hook]

    # OPENAI_KEYS: comma-separated list for rotation; OPENAI_API_KEY: single key
    keys_str = os.environ.get("OPENAI_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        clients = [
            AsyncOpenAI(
                api_key=k,
                http_client=httpx.AsyncClient(http2=True, event_hooks=event_hooks),
                max_retries=max_retries,
            )
            for k in keys
        ]
        return ClientPool(clients)
    return AsyncOpenAI(
        http_client=httpx.AsyncClient(http2=True, event_hooks=event_hooks),
        max_retries=max_retries,
    )


def create_sync_client(max_retries: int = 2, on_headers=None):
    from openai import OpenAI

    from llm_provider.providers._pool import ClientPool

    event_hooks = {}
    if on_headers:
        from llm_provider.providers._headers import parse_openai_headers

        def _hook(response: httpx.Response):
            info = parse_openai_headers(response.headers)
            if info:
                on_headers(info["remaining"], info["limit"])

        event_hooks["response"] = [_hook]

    keys_str = os.environ.get("OPENAI_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        clients = [
            OpenAI(
                api_key=k,
                http_client=httpx.Client(http2=True, event_hooks=event_hooks),
                max_retries=max_retries,
            )
            for k in keys
        ]
        return ClientPool(clients)
    return OpenAI(
        http_client=httpx.Client(http2=True, event_hooks=event_hooks),
        max_retries=max_retries,
    )


def model_id(model: str) -> str:
    """'openai/gpt-4.1-nano' -> 'gpt-4.1-nano'"""
    return model.removeprefix("openai/")


async def call(
    client,
    model_id: str,
    prompt: str,
    system_prompt: str = "",
    **kwargs,
):
    """Returns (texts: list[str], usage: dict)."""
    kwargs = dict(kwargs)
    use_cache = kwargs.pop("cache", True)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Reasoning models reject temperature
    if kwargs.get("max_completion_tokens") and "temperature" in kwargs:
        kwargs.pop("temperature")

    # Cache lookup -- include all kwargs that affect output
    config_dict = {k: v for k, v in sorted(kwargs.items()) if v is not None}
    key = cache_key(model_id, prompt, system_prompt or None, config_dict)
    if use_cache:
        cached = direct_cache.get(key)
        if cached is not None:
            if isinstance(cached, list):
                return cached, {}
            return [cached], {}

    response = await client.chat.completions.create(
        model=model_id, messages=messages, **kwargs
    )

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

    usage = {}
    if response.usage:
        usage["input_tokens"] = response.usage.prompt_tokens or 0
        usage["output_tokens"] = response.usage.completion_tokens or 0
        details = getattr(response.usage, "prompt_tokens_details", None)
        if details:
            usage["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

    c = _pricing_cost(
        model_id,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    if c is not None:
        usage["cost"] = c

    if use_cache and texts[0]:
        if len(texts) == 1:
            direct_cache.set(key, texts[0])
        else:
            direct_cache.set(key, texts)
    return texts, usage


def call_messages_sync(client, model_id: str, messages: list, **kwargs):
    """Sync chat completion with pre-built messages. Returns (texts, usage)."""
    kwargs = dict(kwargs)
    use_cache = kwargs.pop("cache", True)

    if kwargs.get("max_completion_tokens") and "temperature" in kwargs:
        kwargs.pop("temperature")

    config_dict = {k: v for k, v in sorted(kwargs.items()) if v is not None}
    key = cache_key_messages(model_id, messages, config_dict)
    if use_cache:
        cached = direct_cache.get(key)
        if cached is not None:
            if isinstance(cached, list):
                return cached, {}
            return [cached], {}

    response = client.chat.completions.create(
        model=model_id, messages=messages, **kwargs
    )

    texts = [c.message.content or "" for c in response.choices]
    if not texts:
        texts = [""]

    has_reasoning_field = any(
        getattr(c.message, "reasoning_content", None) for c in response.choices
    )
    if not has_reasoning_field:
        texts = [strip_thinking(t) if "<think>" in t else t for t in texts]
    else:
        texts = [t.strip() for t in texts]

    usage = {}
    if response.usage:
        usage["input_tokens"] = response.usage.prompt_tokens or 0
        usage["output_tokens"] = response.usage.completion_tokens or 0
        details = getattr(response.usage, "prompt_tokens_details", None)
        if details:
            usage["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

    c = _pricing_cost(
        model_id,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    if c is not None:
        usage["cost"] = c

    if use_cache and texts[0]:
        if len(texts) == 1:
            direct_cache.set(key, texts[0])
        else:
            direct_cache.set(key, texts)
    return texts, usage


async def bench_stream(client, model_id: str, messages: list, **kwargs):
    """Streaming benchmark -> (ttft, total_time, output_tokens)."""
    kwargs = dict(kwargs)

    t0 = time.monotonic()
    ttft = None
    chunks: list[str] = []

    stream = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )
    last_chunk = None
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if ttft is None:
                ttft = time.monotonic() - t0
            chunks.append(delta)
        last_chunk = chunk

    total = time.monotonic() - t0
    output_tokens = len("".join(chunks)) // 4
    if last_chunk and hasattr(last_chunk, "usage") and last_chunk.usage:
        output_tokens = last_chunk.usage.completion_tokens or output_tokens

    return ttft or total, total, output_tokens
