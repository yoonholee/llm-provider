# Plan

Status: complete

## Last completed (2026-02-22)

### Adaptive Concurrency + Batch API

**Adaptive concurrency:**
- Replaced `_AIMDSemaphore` with `_AdaptiveSemaphore` in `provider.py`
- Header-based proportional control: `on_headers(remaining, limit)` with EMA smoothing (alpha=0.3)
- AIMD fallback when no headers (Gemini, Together, etc.)
- 429 always halves window (overrides headers)
- Semaphore is now persistent on `LLM` instance (`self._sem`) instead of per-`_batch()` call
- httpx event hooks intercept `x-ratelimit-*` headers transparently for OpenAI
- New file: `src/llm_provider/providers/_headers.py`

**Batch API:**
- New file: `src/llm_provider/providers/_batch.py` (OpenAI, Anthropic, Gemini)
- `LLM.batch(prompts, system_prompt=..., poll_interval=60)` -- all-in-one
- `LLM.batch_submit()` / `batch_status()` / `batch_retrieve()` -- split workflow
- Batch IDs encoded as `"{provider}:{n_prompts}:{raw_id}"` (stateless)
- 50% cost savings vs `generate()` for supported providers
- Supported: `gpt-*`, `o3-*`, `o4-*` (OpenAI), `anthropic/*` (Anthropic), `gemini/*` (Gemini)
- Others raise `NotImplementedError`

**Tests:** 113 total (86 test_provider.py + 27 test_batch.py), all passing.

### Previous (2026-02-22)

- OpenRouter provider (`openrouter/*`), sorts by price by default
- `multi_generate()` / `multi_chat()` for parallel multi-model queries
- Fixed `_batch()` hasattr bug for litellm models
- Removed dead `_median()`, code review pass
- Benchmarked HTTP/2 on OpenRouter (hurts, left off)
- Updated README with all providers, chat, multi-model docs
