"""Together provider via OpenAI-compatible API (direct SDK, no HTTP/2)."""

import os


def create_client(max_retries: int = 2):
    from openai import AsyncOpenAI

    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHERAI_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY required for Together models (direct SDK)")
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
        max_retries=max_retries,
    )


def model_id(model: str) -> str:
    """'together_ai/meta-llama/...' -> 'meta-llama/...'"""
    return model.removeprefix("together_ai/")


# call() and bench_stream() delegate to openai_api (OpenAI-compatible)
