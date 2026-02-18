"""litellm fallback provider (Anthropic and everything else)."""

import time


def _litellm():
    """Lazy import of litellm (saves ~1.5s when using direct SDK paths)."""
    import litellm

    from llm_provider._cache import _ensure_litellm_cache

    _ensure_litellm_cache()
    return litellm


async def call(
    model: str, prompt: str, system_prompt: str = "", max_retries: int = 2, **kwargs
):
    """Returns (texts: list[str], usage: dict)."""
    kwargs = dict(kwargs)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Reasoning models reject temperature
    if kwargs.get("max_completion_tokens") and "temperature" in kwargs:
        kwargs.pop("temperature")

    response = await _litellm().acompletion(
        model=model,
        messages=messages,
        num_retries=max_retries,
        caching=True,
        **kwargs,
    )

    texts = [c.message.content or "" for c in response.choices]
    if not texts:
        texts = [""]

    usage = {}
    if response.usage:
        usage["input_tokens"] = response.usage.prompt_tokens or 0
        usage["output_tokens"] = response.usage.completion_tokens or 0
        details = getattr(response.usage, "prompt_tokens_details", None)
        if details:
            usage["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0
    try:
        usage["cost"] = _litellm().completion_cost(completion_response=response)
    except Exception:
        pass

    return texts, usage


async def bench_stream(model: str, messages: list, **kwargs):
    """Streaming benchmark -> (ttft, total_time, output_tokens)."""
    t0 = time.monotonic()
    ttft = None
    chunks: list[str] = []
    last_chunk = None

    response = await _litellm().acompletion(
        model=model, messages=messages, stream=True, num_retries=1, **kwargs
    )
    async for chunk in response:
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
