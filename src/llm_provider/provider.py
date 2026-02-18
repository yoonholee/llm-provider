"""LLM class: per-provider routing with disk caching and async batching."""

import asyncio
import fcntl
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from llm_provider.providers import gemini, litellm_api, local, openai_api, together

log = logging.getLogger(__name__)

# --- Adaptive concurrency (AIMD) ---

_MIN_CONCURRENCY = 4
_MAX_RETRIES = 5
_BASE_DELAY = 2.0  # seconds


class _AIMDSemaphore:
    """Semaphore with additive-increase / multiplicative-decrease.

    On success: window += 1 (up to max).
    On 429: window //= 2 (down to _MIN_CONCURRENCY).
    """

    def __init__(self, initial: int):
        self._max = initial
        self._window = initial
        self._sem = asyncio.Semaphore(initial)

    async def acquire(self):
        await self._sem.acquire()

    def release(self):
        self._sem.release()

    def on_success(self):
        if self._window < self._max:
            self._window = min(self._window + 1, self._max)
            # Grow the semaphore by releasing one extra permit
            self._sem.release()

    def on_rate_limit(self):
        new = max(self._window // 2, _MIN_CONCURRENCY)
        if new < self._window:
            shrink = self._window - new
            for _ in range(shrink):
                if self._sem._value > 0:  # noqa: SLF001
                    self._sem._value -= 1  # noqa: SLF001
            self._window = new

    @property
    def window(self):
        return self._window


# --- Cross-process file-lock semaphore ---


class _FileSlotSemaphore:
    """Cross-process semaphore using file locks.

    Creates N lock files in lock_dir. Each acquire() tries to flock one.
    Locks auto-release on process crash (OS guarantee).
    """

    def __init__(self, max_slots: int, lock_dir: str = "/tmp"):
        self._slots = max_slots
        self._dir = Path(lock_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._paths = [
            self._dir / f"llm_provider_slot_{i}.lock" for i in range(max_slots)
        ]
        self._held: dict = {}  # task -> fd

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        self.release()

    async def acquire(self):
        """Block until a slot is available."""
        task = asyncio.current_task()
        while True:
            for path in self._paths:
                fd = open(path, "w")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._held[task] = fd
                    return
                except (BlockingIOError, OSError):
                    fd.close()
            await asyncio.sleep(0.05 + random.random() * 0.05)

    def release(self):
        task = asyncio.current_task()
        fd = self._held.pop(task, None)
        if fd:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()


def _get_global_semaphore() -> _FileSlotSemaphore | None:
    """Return a global file-lock semaphore if LLM_GLOBAL_CONCURRENCY is set."""
    val = os.environ.get("LLM_GLOBAL_CONCURRENCY")
    if val:
        return _FileSlotSemaphore(int(val))
    return None


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


def _is_rate_limit(exc: Exception) -> bool:
    """Check if an exception is a 429 rate limit error."""
    # OpenAI SDK
    if type(exc).__name__ == "RateLimitError":
        return True
    # HTTP status code — check .status_code, .status, and .code (google.genai uses .code)
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if val == 429:
            return True
    return False


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
        n: int = 1,
        cache: bool = True,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Generate completions. Returns list of response lists (one per prompt).

        Args:
            n: Number of completions per prompt (default 1).
            cache: Whether to use disk cache (default True).
        """
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        if n > 1:
            kwargs["n"] = n
        if not cache:
            kwargs["cache"] = False

        out_before = self.total_output_tokens
        in_before = self.total_input_tokens
        cached_before = self.total_cached_tokens
        cost_before = self.total_cost

        t0 = time.monotonic()
        results = _run_async(self._batch(prompt_list, system_prompt or "", **kwargs))
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
        self, prompts: list[str], system_prompt: str, **kwargs
    ) -> list[list[str]]:
        sem = _AIMDSemaphore(self.max_concurrent)
        global_sem = _get_global_semaphore()

        async def run_one(prompt: str) -> list[str]:
            for attempt in range(_MAX_RETRIES + 1):
                if global_sem:
                    await global_sem.acquire()
                await sem.acquire()
                try:
                    result = await self._call(prompt, system_prompt, **kwargs)
                    sem.on_success()
                    return result
                except Exception as e:
                    if _is_rate_limit(e):
                        sem.on_rate_limit()
                        if attempt < _MAX_RETRIES:
                            delay = _BASE_DELAY * (2**attempt) + random.random()
                            log.warning(
                                "429 rate limit (attempt %d/%d), retrying in %.1fs",
                                attempt + 1,
                                _MAX_RETRIES,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                    raise
                finally:
                    sem.release()
                    if global_sem:
                        global_sem.release()

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
                openai_api.model_id(self.model),
                prompt,
                system_prompt,
                **kwargs,
            )
        elif _is_together(self.model):
            texts, usage = await openai_api.call(
                self._client,
                together.model_id(self.model),
                prompt,
                system_prompt,
                **kwargs,
            )
        elif _is_local(self.model):
            texts, usage = await openai_api.call(
                self._client,
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
