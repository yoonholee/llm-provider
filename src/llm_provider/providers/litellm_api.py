"""litellm fallback provider (Anthropic and everything else)."""


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
    use_cache = kwargs.pop("cache", True)
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
        caching=use_cache,
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


def call_messages_sync(model: str, messages: list, max_retries: int = 2, **kwargs):
    """Sync chat completion with pre-built messages. Returns (texts, usage)."""
    kwargs = dict(kwargs)
    use_cache = kwargs.pop("cache", True)

    if kwargs.get("max_completion_tokens") and "temperature" in kwargs:
        kwargs.pop("temperature")

    response = _litellm().completion(
        model=model,
        messages=messages,
        num_retries=max_retries,
        caching=use_cache,
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
