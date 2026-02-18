"""Gemini provider via native google.genai SDK with streaming."""

import os
import threading
import time

from llm_provider._cache import cache_key, direct_cache


class _ClientPool:
    """Round-robin pool of Gemini clients for API key rotation.

    Accepts GEMINI_API_KEY as a single key or comma-separated list.
    On 429, call rotate() to advance to the next key.
    """

    def __init__(self, keys: list[str]):
        import google.genai as genai

        self._clients = [genai.Client(api_key=k.strip()) for k in keys if k.strip()]
        if not self._clients:
            raise ValueError("GEMINI_API_KEY required for Gemini models")
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def current(self):
        return self._clients[self._idx % len(self._clients)]

    def rotate(self):
        """Advance to the next client. Returns True if there are multiple keys."""
        if len(self._clients) <= 1:
            return False
        with self._lock:
            self._idx = (self._idx + 1) % len(self._clients)
        return True

    def __len__(self):
        return len(self._clients)


def create_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required for Gemini models")
    keys = api_key.split(",")
    return _ClientPool(keys)


def model_id(model: str) -> str:
    """'gemini/gemini-3-flash-preview' -> 'gemini-3-flash-preview'"""
    return model.removeprefix("gemini/")


async def call(pool, model: str, prompt: str, system_prompt: str = "", **kwargs):
    """Returns (texts: list[str], usage: dict).

    pool: _ClientPool (round-robin across API keys).
    """
    import google.genai.types as types

    kwargs = dict(kwargs)
    use_cache = kwargs.pop("cache", True)
    n = kwargs.pop("n", 1)
    mid = model_id(model)

    # thinking_budget=0 disables thinking (fastest config)
    thinking_config = kwargs.pop("thinking_config", None)
    if thinking_config is None:
        thinking_config = types.ThinkingConfig(thinking_budget=0)

    max_tokens = kwargs.pop("max_tokens", None)
    if max_tokens is None:
        max_tokens = kwargs.pop("max_output_tokens", None)

    # Forward remaining kwargs (temperature, top_p, etc.) to config
    temperature = kwargs.pop("temperature", None)
    top_p = kwargs.pop("top_p", None)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        thinking_config=thinking_config,
        **({"max_output_tokens": max_tokens} if max_tokens is not None else {}),
        **({"temperature": temperature} if temperature is not None else {}),
        **({"top_p": top_p} if top_p is not None else {}),
    )

    # Cache lookup
    tc_dict = {}
    if getattr(thinking_config, "thinking_budget", None) is not None:
        tc_dict["thinking_budget"] = thinking_config.thinking_budget
    if getattr(thinking_config, "thinking_level", None) is not None:
        tc_dict["thinking_level"] = str(thinking_config.thinking_level)
    config_dict = {
        "thinking": tc_dict,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "system": system_prompt or None,
    }
    if n > 1:
        config_dict["n"] = n
    key = cache_key(mid, prompt, system_prompt or None, config_dict)
    if use_cache:
        cached = direct_cache.get(key)
        if cached is not None:
            if isinstance(cached, list):
                return cached, {}
            return [cached], {}

    async def _single_call():
        chunks: list[str] = []
        last_chunk = None
        async for chunk in await pool.current.aio.models.generate_content_stream(
            model=mid, contents=prompt, config=config
        ):
            if chunk.text:
                chunks.append(chunk.text)
            last_chunk = chunk
        return "".join(chunks), last_chunk

    if n > 1:
        import asyncio

        results = await asyncio.gather(*[_single_call() for _ in range(n)])
        texts = [r[0] for r in results]
        usage = {}
        last = results[-1][1]
        if last and last.usage_metadata:
            meta = last.usage_metadata
            usage["input_tokens"] = (meta.prompt_token_count or 0) * n
            usage["output_tokens"] = sum(
                (r[1].usage_metadata.candidates_token_count or 0)
                for r in results
                if r[1] and r[1].usage_metadata
            )
    else:
        text, last_chunk = await _single_call()
        texts = [text]
        usage = {}
        if last_chunk and last_chunk.usage_metadata:
            meta = last_chunk.usage_metadata
            usage["input_tokens"] = meta.prompt_token_count or 0
            usage["output_tokens"] = meta.candidates_token_count or 0

    if use_cache and texts[0]:
        if len(texts) == 1:
            direct_cache.set(key, texts[0])
        else:
            direct_cache.set(key, texts)
    return texts, usage


async def bench_stream(pool, model: str, messages: list, **kwargs):
    """Streaming benchmark -> (ttft, total_time, output_tokens)."""
    import google.genai.types as types

    kwargs = dict(kwargs)
    mid = model_id(model)

    thinking_config = kwargs.pop(
        "thinking_config", types.ThinkingConfig(thinking_budget=0)
    )
    max_tokens = kwargs.pop("max_tokens", None)
    if max_tokens is None:
        max_tokens = kwargs.pop("max_output_tokens", None)
    kwargs.pop("temperature", None)

    system_instruction = None
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            prompt = msg["content"]

    config = types.GenerateContentConfig(
        thinking_config=thinking_config,
        system_instruction=system_instruction,
        **({"max_output_tokens": max_tokens} if max_tokens is not None else {}),
    )

    t0 = time.monotonic()
    ttft = None
    chunks: list[str] = []
    last_chunk = None

    async for chunk in await pool.current.aio.models.generate_content_stream(
        model=mid, contents=prompt, config=config
    ):
        if chunk.text:
            if ttft is None:
                ttft = time.monotonic() - t0
            chunks.append(chunk.text)
        last_chunk = chunk

    total = time.monotonic() - t0
    output_tokens = len("".join(chunks)) // 4
    if last_chunk and last_chunk.usage_metadata:
        output_tokens = (
            last_chunk.usage_metadata.candidates_token_count or output_tokens
        )

    return ttft or total, total, output_tokens
