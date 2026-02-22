"""Local provider via OpenAI-compatible API (e.g. vLLM, SGLang, Ollama).

Usage: LLM("local/Qwen/Qwen3-4B")
Set LOCAL_BASE_URL to override the default http://localhost:30000/v1.
"""

import os


def create_client(max_retries: int = 2):
    from openai import AsyncOpenAI

    base_url = os.environ.get("LOCAL_BASE_URL", "http://localhost:30000/v1")
    return AsyncOpenAI(
        api_key="unused",
        base_url=base_url,
        max_retries=max_retries,
    )


def create_sync_client(max_retries: int = 2):
    from openai import OpenAI

    base_url = os.environ.get("LOCAL_BASE_URL", "http://localhost:30000/v1")
    return OpenAI(
        api_key="unused",
        base_url=base_url,
        max_retries=max_retries,
    )


def model_id(model: str) -> str:
    """'local/Qwen/Qwen3-4B' -> 'Qwen/Qwen3-4B'"""
    return model.removeprefix("local/")


# call() delegates to openai_api (OpenAI-compatible)
