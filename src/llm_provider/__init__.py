"""Thin LLM wrapper with disk caching and async batching.

Per-provider routing for optimal throughput:
  - Gemini:   native google.genai SDK (streaming, thinking_budget=0)
  - OpenAI:   direct SDK with HTTP/2 (~1.5-2x throughput over litellm)
  - Together:  direct SDK via OpenAI-compatible API (~1.6x over litellm)
  - All others: litellm fallback (Anthropic, etc.)

Usage:
    from llm_provider import LLM
    llm = LLM("gpt-4.1-nano")
    results = llm.generate(["What is 2+2?", "Name a color."])
    # [["4"], ["Blue"]]

    llm = LLM("gemini/gemini-3-flash-preview")
    results = llm.generate(["What is 2+2?"])
    # [["4"]]

Install:
    uv add git+https://github.com/yoonho/llm-provider

Notes:
  - Default max_concurrent=32 (optimal for OpenAI; c=64 triggers rate limiting)
  - Prefix caching: automatic for gpt-4.1-mini+ with identical system prompts
    (requires >=1024 token prefix; gpt-4.1-nano doesn't support it)
  - Tracks cached input tokens from OpenAI/Anthropic prefix caching
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
import litellm
from diskcache import FanoutCache
from litellm.caching.caching import Cache

# Silence noisy loggers
for name in ("openai", "httpx", "LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(name).setLevel(logging.WARNING)
litellm.suppress_debug_info = True

# Enable disk caching (litellm models only; direct SDK paths use _direct_cache)
_cache_dir = os.environ.get("LLM_CACHE_DIR", None)
if _cache_dir is None:
    candidates = [Path("/iris/u/yoonho/.cache/llm_cache"), Path("/tmp/llm_cache")]
    _cache_dir = str(next((p for p in candidates if p.parent.exists()), candidates[-1]))
litellm.cache = Cache(type="disk", disk_cache_dir=_cache_dir)

# Disk cache for all direct SDK paths (Gemini, OpenAI, Together)
_direct_cache = FanoutCache(str(Path(_cache_dir) / "direct"), shards=8)


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


# --- Provider detection ---

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def _is_gemini(model: str) -> bool:
    return model.startswith("gemini/")


def _is_openai(model: str) -> bool:
    m = model.removeprefix("openai/")
    return any(m.startswith(p) for p in _OPENAI_PREFIXES)


def _is_together(model: str) -> bool:
    return model.startswith("together_ai/")


def _gemini_model_id(model: str) -> str:
    """'gemini/gemini-3-flash-preview' -> 'gemini-3-flash-preview'"""
    return model.removeprefix("gemini/")


def _openai_model_id(model: str) -> str:
    """'openai/gpt-4.1-nano' -> 'gpt-4.1-nano', 'gpt-4.1-nano' -> 'gpt-4.1-nano'"""
    return model.removeprefix("openai/")


def _together_model_id(model: str) -> str:
    """'together_ai/meta-llama/...' -> 'meta-llama/...'"""
    return model.removeprefix("together_ai/")


def _cache_key(model: str, prompt: str, system: str | None, config: dict) -> str:
    """Deterministic cache key for a direct SDK request."""
    blob = json.dumps(
        {"model": model, "prompt": prompt, "system": system, "config": config},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class LLM:
    """Thin wrapper with async batching and usage tracking.

    Routes to the optimal SDK per provider:
      - Gemini (gemini/*): native google.genai SDK with streaming
      - OpenAI (gpt-*, o1*, etc.): direct SDK with HTTP/2
      - Together (together_ai/*): direct SDK via OpenAI-compatible API
      - All others: litellm fallback

    Token counts are cumulative across generate() calls.
    """

    def __init__(self, model: str, max_concurrent: int = 32, max_retries: int = 2):
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost = 0.0

        if _is_gemini(model):
            import google.genai as genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY required for Gemini models")
            self._genai_client = genai.Client(api_key=api_key)

        elif _is_openai(model):
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(
                http_client=httpx.AsyncClient(http2=True),
                max_retries=max_retries,
            )

        elif _is_together(model):
            from openai import AsyncOpenAI

            api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get(
                "TOGETHERAI_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "TOGETHER_API_KEY required for Together models (direct SDK)"
                )
            self._together_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
                max_retries=max_retries,
            )

    def generate(
        self,
        prompts: str | list[str],
        system_prompt: str | None = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Generate completions. Returns list of response lists (one per prompt)."""
        prompt_list = [prompts] if isinstance(prompts, str) else prompts

        t0 = time.monotonic()
        results = _run_async(
            self._batch(prompt_list, system_prompt or "", silent, **kwargs)
        )
        elapsed = time.monotonic() - t0

        if not silent and self.total_output_tokens > 0:
            tps = self.total_output_tokens / elapsed if elapsed > 0 else 0
            cost_str = f" ${self.total_cost:.4f}" if self.total_cost > 0 else ""
            cache_str = (
                f" ({self.total_cached_tokens} cached)"
                if self.total_cached_tokens > 0
                else ""
            )
            print(
                f"  {self.total_input_tokens} in{cache_str} | {self.total_output_tokens} out |{cost_str} {tps:.0f} tok/s"
            )

        return results

    async def _batch(
        self, prompts: list[str], system_prompt: str, silent: bool, **kwargs
    ) -> list[list[str]]:
        sem = asyncio.Semaphore(self.max_concurrent)

        async def run_one(prompt: str) -> list[str]:
            async with sem:
                if _is_gemini(self.model):
                    return await self._call_gemini(prompt, system_prompt, **kwargs)
                if _is_openai(self.model):
                    return await self._call_openai(prompt, system_prompt, **kwargs)
                if _is_together(self.model):
                    return await self._call_together(prompt, system_prompt, **kwargs)
                return await self._call_litellm(prompt, system_prompt, **kwargs)

        tasks = [run_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    # --- Gemini native SDK path (streaming) ---

    async def _call_gemini(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> list[str]:
        import google.genai.types as types

        kwargs = dict(kwargs)  # don't mutate shared dict across concurrent calls
        model_id = _gemini_model_id(self.model)

        # Gemini 3: thinking_budget=0 disables thinking (faster than thinking_level=MINIMAL).
        # Don't set temperature for Gemini 3 -- default 1.0 is optimal, lower causes loops.
        thinking_config = kwargs.pop("thinking_config", None)
        if thinking_config is None:
            thinking_config = types.ThinkingConfig(thinking_budget=0)

        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_output_tokens", None)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            thinking_config=thinking_config,
            **({"max_output_tokens": max_tokens} if max_tokens is not None else {}),
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
            "system": system_prompt or None,
        }
        cache_key = _cache_key(model_id, prompt, system_prompt or None, config_dict)
        cached = _direct_cache.get(cache_key)
        if cached is not None:
            return [cached]

        # Stream response for lower TTFT
        chunks: list[str] = []
        last_chunk = None
        async for chunk in await self._genai_client.aio.models.generate_content_stream(
            model=model_id,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                chunks.append(chunk.text)
            last_chunk = chunk

        text = "".join(chunks)

        # Usage tracking from the last chunk
        if last_chunk and last_chunk.usage_metadata:
            meta = last_chunk.usage_metadata
            self.total_input_tokens += meta.prompt_token_count or 0
            self.total_output_tokens += meta.candidates_token_count or 0

        if text:
            _direct_cache.set(cache_key, text)
        return [text]

    # --- OpenAI-compatible direct SDK path (OpenAI, Together) ---

    async def _call_openai_compat(
        self,
        client,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        **kwargs,
    ) -> list[str]:
        """Shared implementation for OpenAI-compatible APIs (OpenAI, Together)."""
        kwargs = dict(kwargs)  # don't mutate shared dict across concurrent calls
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Reasoning models (e.g. gpt-5-mini) reject temperature
        if kwargs.get("max_completion_tokens") and "temperature" in kwargs:
            kwargs.pop("temperature")

        # Cache lookup
        config_dict = {}
        for k in ("temperature", "max_tokens", "max_completion_tokens", "top_p"):
            if k in kwargs and kwargs[k] is not None:
                config_dict[k] = kwargs[k]
        key = _cache_key(model_id, prompt, system_prompt or None, config_dict)
        cached = _direct_cache.get(key)
        if cached is not None:
            return [cached]

        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            **kwargs,
        )

        texts = [c.message.content or "" for c in response.choices]
        if not texts:
            texts = [""]

        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0
            # Track prefix caching (OpenAI returns cached_tokens in prompt_tokens_details)
            details = getattr(response.usage, "prompt_tokens_details", None)
            if details:
                self.total_cached_tokens += getattr(details, "cached_tokens", 0) or 0

        # Cost via litellm's pricing database
        try:
            self.total_cost += litellm.completion_cost(
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=(
                    response.usage.completion_tokens if response.usage else 0
                ),
            )
        except Exception:
            pass

        if texts[0]:
            _direct_cache.set(key, texts[0])
        return texts

    async def _call_openai(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> list[str]:
        return await self._call_openai_compat(
            self._openai_client,
            _openai_model_id(self.model),
            prompt,
            system_prompt,
            **kwargs,
        )

    async def _call_together(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> list[str]:
        return await self._call_openai_compat(
            self._together_client,
            _together_model_id(self.model),
            prompt,
            system_prompt,
            **kwargs,
        )

    # --- litellm fallback path (Anthropic and everything else) ---

    async def _call_litellm(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> list[str]:
        kwargs = dict(kwargs)  # don't mutate shared dict across concurrent calls
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Reasoning models (e.g. gpt-5-mini) reject temperature
        if kwargs.get("max_completion_tokens") and "temperature" in kwargs:
            kwargs.pop("temperature")

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            num_retries=self.max_retries,
            caching=True,
            **kwargs,
        )

        texts = [c.message.content or "" for c in response.choices]
        if not texts:
            texts = [""]

        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0
            # Track prefix caching (litellm passes through provider's cached token info)
            details = getattr(response.usage, "prompt_tokens_details", None)
            if details:
                self.total_cached_tokens += getattr(details, "cached_tokens", 0) or 0
        try:
            self.total_cost += litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        return texts

