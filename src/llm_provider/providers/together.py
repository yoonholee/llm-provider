"""Together provider via OpenAI-compatible API (direct SDK, no HTTP/2)."""

import os


def create_client(max_retries: int = 2):
    from openai import AsyncOpenAI

    from llm_provider.providers._pool import ClientPool

    # TOGETHER_KEYS: comma-separated list for rotation; TOGETHER_API_KEY: single key
    keys_str = os.environ.get("TOGETHER_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        clients = [
            AsyncOpenAI(
                api_key=k,
                base_url="https://api.together.xyz/v1",
                max_retries=max_retries,
            )
            for k in keys
        ]
        return ClientPool(clients)
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHERAI_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY or TOGETHER_KEYS required for Together models"
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
        max_retries=max_retries,
    )


def create_sync_client(max_retries: int = 2):
    from openai import OpenAI

    from llm_provider.providers._pool import ClientPool

    keys_str = os.environ.get("TOGETHER_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        clients = [
            OpenAI(
                api_key=k,
                base_url="https://api.together.xyz/v1",
                max_retries=max_retries,
            )
            for k in keys
        ]
        return ClientPool(clients)
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHERAI_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY or TOGETHER_KEYS required for Together models"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
        max_retries=max_retries,
    )


def model_id(model: str) -> str:
    """'together_ai/meta-llama/...' -> 'meta-llama/...'"""
    return model.removeprefix("together_ai/")


# call() and bench_stream() delegate to openai_api (OpenAI-compatible)
