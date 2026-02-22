"""OpenRouter provider via OpenAI-compatible API.

Usage: LLM("openrouter/anthropic/claude-sonnet-4")
Set OPENROUTER_API_KEY or OPENROUTER_KEYS (comma-separated for rotation).

Defaults to sorting by price (cheapest provider first).
Override with extra_body={"provider": {...}} in kwargs.
"""

import os

_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_PROVIDER = {"sort": "price"}


def create_client(max_retries: int = 2):
    from openai import AsyncOpenAI

    from llm_provider.providers._pool import ClientPool

    keys_str = os.environ.get("OPENROUTER_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        clients = [
            AsyncOpenAI(api_key=k, base_url=_BASE_URL, max_retries=max_retries)
            for k in keys
        ]
        return ClientPool(clients)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY or OPENROUTER_KEYS required for OpenRouter models"
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=_BASE_URL,
        max_retries=max_retries,
    )


def create_sync_client(max_retries: int = 2):
    from openai import OpenAI

    from llm_provider.providers._pool import ClientPool

    keys_str = os.environ.get("OPENROUTER_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        clients = [
            OpenAI(api_key=k, base_url=_BASE_URL, max_retries=max_retries) for k in keys
        ]
        return ClientPool(clients)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY or OPENROUTER_KEYS required for OpenRouter models"
        )
    return OpenAI(
        api_key=api_key,
        base_url=_BASE_URL,
        max_retries=max_retries,
    )


def model_id(model: str) -> str:
    """'openrouter/anthropic/claude-sonnet-4' -> 'anthropic/claude-sonnet-4'"""
    return model.removeprefix("openrouter/")


def inject_provider_kwargs(kwargs: dict) -> dict:
    """Merge default provider config into extra_body if not already set."""
    kwargs = dict(kwargs)
    extra = kwargs.get("extra_body") or {}
    if "provider" not in extra:
        extra = {**extra, "provider": _DEFAULT_PROVIDER}
        kwargs["extra_body"] = extra
    return kwargs


# call() delegates to openai_api (OpenAI-compatible)
