"""SambaNova provider via OpenAI-compatible API.

Usage: LLM("sambanova/Meta-Llama-3.3-70B-Instruct")
Set SAMBANOVA_API_KEY or SAMBANOVA_KEYS (comma-separated for rotation).
"""

import os

_BASE_URL = "https://api.sambanova.ai/v1"


def create_client(max_retries: int = 2):
    from openai import AsyncOpenAI

    from llm_provider.providers._pool import ClientPool

    keys_str = os.environ.get("SAMBANOVA_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        clients = [
            AsyncOpenAI(api_key=k, base_url=_BASE_URL, max_retries=max_retries)
            for k in keys
        ]
        return ClientPool(clients)
    api_key = os.environ.get("SAMBANOVA_API_KEY")
    if not api_key:
        raise ValueError(
            "SAMBANOVA_API_KEY or SAMBANOVA_KEYS required for SambaNova models"
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=_BASE_URL,
        max_retries=max_retries,
    )


def create_sync_client(max_retries: int = 2):
    from openai import OpenAI

    from llm_provider.providers._pool import ClientPool

    keys_str = os.environ.get("SAMBANOVA_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        clients = [
            OpenAI(api_key=k, base_url=_BASE_URL, max_retries=max_retries) for k in keys
        ]
        return ClientPool(clients)
    api_key = os.environ.get("SAMBANOVA_API_KEY")
    if not api_key:
        raise ValueError(
            "SAMBANOVA_API_KEY or SAMBANOVA_KEYS required for SambaNova models"
        )
    return OpenAI(
        api_key=api_key,
        base_url=_BASE_URL,
        max_retries=max_retries,
    )


def model_id(model: str) -> str:
    """'sambanova/Meta-Llama-3.3-70B-Instruct' -> 'Meta-Llama-3.3-70B-Instruct'"""
    return model.removeprefix("sambanova/")


# call() delegates to openai_api (OpenAI-compatible)
