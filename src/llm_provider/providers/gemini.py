"""Gemini provider via native google.genai SDK with streaming."""

import os

from llm_provider._cache import cache_key, cache_key_messages, direct_cache
from llm_provider.providers._pool import ClientPool


def create_client():
    import google.genai as genai

    # GEMINI_KEYS: comma-separated list for rotation; GEMINI_API_KEY: single key
    keys_str = os.environ.get("GEMINI_KEYS") or os.environ.get("GEMINI_API_KEY")
    if not keys_str:
        raise ValueError("GEMINI_API_KEY or GEMINI_KEYS required for Gemini models")
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    clients = [genai.Client(api_key=k) for k in keys]
    return ClientPool(clients)


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


def call_messages_sync(pool, model: str, messages: list, **kwargs):
    """Sync Gemini call with messages. Returns (texts, usage).

    Converts OpenAI message format to Gemini format:
      - system messages -> system_instruction
      - assistant -> role="model"
    Uses sync non-streaming API.
    """
    import google.genai.types as types

    kwargs = dict(kwargs)
    use_cache = kwargs.pop("cache", True)
    mid = model_id(model)

    thinking_config = kwargs.pop("thinking_config", None)
    if thinking_config is None:
        thinking_config = types.ThinkingConfig(thinking_budget=0)

    max_tokens = kwargs.pop("max_tokens", None)
    if max_tokens is None:
        max_tokens = kwargs.pop("max_output_tokens", None)
    temperature = kwargs.pop("temperature", None)
    top_p = kwargs.pop("top_p", None)

    # Convert messages to Gemini format
    system_instruction = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=msg["content"])])
            )
        elif msg["role"] == "assistant":
            contents.append(
                types.Content(role="model", parts=[types.Part(text=msg["content"])])
            )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
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
    }
    key = cache_key_messages(mid, messages, config_dict)
    if use_cache:
        cached = direct_cache.get(key)
        if cached is not None:
            if isinstance(cached, list):
                return cached, {}
            return [cached], {}

    # Sync non-streaming call
    response = pool.current.models.generate_content(
        model=mid, contents=contents, config=config
    )

    text = response.text or ""
    texts = [text]
    usage = {}
    if response.usage_metadata:
        meta = response.usage_metadata
        usage["input_tokens"] = meta.prompt_token_count or 0
        usage["output_tokens"] = meta.candidates_token_count or 0

    if use_cache and texts[0]:
        direct_cache.set(key, texts[0])
    return texts, usage
