"""LLM class: per-provider routing with disk caching and async batching."""

import asyncio
import time
from typing import Any

from llm_provider.providers import gemini, litellm_api, local, openai_api, together

# --- Provider detection ---

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def _is_gemini(model: str) -> bool:
    return model.startswith("gemini/")


def _is_openai(model: str) -> bool:
    m = model.removeprefix("openai/")
    return any(m.startswith(p) for p in _OPENAI_PREFIXES)


def _is_together(model: str) -> bool:
    return model.startswith("together_ai/")


def _is_local(model: str) -> bool:
    return model.startswith("local/")


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


class LLM:
    """Thin wrapper with async batching and usage tracking.

    Routes to the optimal SDK per provider:
      - Gemini (gemini/*): native google.genai SDK with streaming
      - OpenAI (gpt-*, o1*, etc.): direct SDK with HTTP/2
      - Together (together_ai/*): direct SDK via OpenAI-compatible API
      - Local (local/*): OpenAI-compatible API on localhost (vLLM, SGLang, etc.)
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
            self._client = gemini.create_client()
        elif _is_openai(model):
            self._client = openai_api.create_client(max_retries=max_retries)
        elif _is_together(model):
            self._client = together.create_client(max_retries=max_retries)
        elif _is_local(model):
            self._client = local.create_client(max_retries=max_retries)

    def generate(
        self,
        prompts: str | list[str],
        system_prompt: str | None = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Generate completions. Returns list of response lists (one per prompt)."""
        prompt_list = [prompts] if isinstance(prompts, str) else prompts

        out_before = self.total_output_tokens
        in_before = self.total_input_tokens
        cached_before = self.total_cached_tokens
        cost_before = self.total_cost

        t0 = time.monotonic()
        results = _run_async(
            self._batch(prompt_list, system_prompt or "", silent, **kwargs)
        )
        elapsed = time.monotonic() - t0

        call_out = self.total_output_tokens - out_before
        call_in = self.total_input_tokens - in_before
        call_cached = self.total_cached_tokens - cached_before
        call_cost = self.total_cost - cost_before

        if not silent and call_out > 0:
            tps = call_out / elapsed if elapsed > 0 else 0
            cost_str = f" ${call_cost:.4f}" if call_cost > 0 else ""
            cache_str = f" ({call_cached} cached)" if call_cached > 0 else ""
            print(
                f"  {call_in} in{cache_str} | {call_out} out |{cost_str} {tps:.0f} tok/s"
            )

        return results

    async def _batch(
        self, prompts: list[str], system_prompt: str, silent: bool, **kwargs
    ) -> list[list[str]]:
        sem = asyncio.Semaphore(self.max_concurrent)

        async def run_one(prompt: str) -> list[str]:
            async with sem:
                return await self._call(prompt, system_prompt, **kwargs)

        tasks = [run_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _call(self, prompt: str, system_prompt: str, **kwargs) -> list[str]:
        if _is_gemini(self.model):
            texts, usage = await gemini.call(
                self._client, self.model, prompt, system_prompt, **kwargs
            )
        elif _is_openai(self.model):
            texts, usage = await openai_api.call(
                self._client,
                self.model,
                openai_api.model_id(self.model),
                prompt,
                system_prompt,
                **kwargs,
            )
        elif _is_together(self.model):
            texts, usage = await openai_api.call(
                self._client,
                self.model,
                together.model_id(self.model),
                prompt,
                system_prompt,
                **kwargs,
            )
        elif _is_local(self.model):
            texts, usage = await openai_api.call(
                self._client,
                self.model,
                local.model_id(self.model),
                prompt,
                system_prompt,
                **kwargs,
            )
        else:
            texts, usage = await litellm_api.call(
                self.model, prompt, system_prompt, self.max_retries, **kwargs
            )

        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)
        self.total_cached_tokens += usage.get("cached_tokens", 0)
        self.total_cost += usage.get("cost", 0.0)
        return texts


# --- Benchmark helpers ---


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


async def _bench_one_stream(model: str, messages: list, *, clients=None, **kwargs):
    """Single streaming request -> (ttft, total_time, output_tokens)."""
    clients = clients or {}
    if _is_gemini(model) and "gemini" in clients:
        return await gemini.bench_stream(clients["gemini"], model, messages, **kwargs)
    if _is_openai(model) and "openai" in clients:
        return await openai_api.bench_stream(
            clients["openai"], openai_api.model_id(model), messages, **kwargs
        )
    if _is_together(model) and "together" in clients:
        return await openai_api.bench_stream(
            clients["together"], together.model_id(model), messages, **kwargs
        )
    if _is_local(model) and "local" in clients:
        return await openai_api.bench_stream(
            clients["local"], local.model_id(model), messages, **kwargs
        )
    return await litellm_api.bench_stream(model, messages, **kwargs)
