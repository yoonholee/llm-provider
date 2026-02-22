# llm-provider

Thin LLM wrapper that bypasses litellm's overhead for major providers. Uses direct SDKs with provider-specific optimizations, falling back to litellm for everything else.

## Why not just litellm?

At low concurrency the gap is small, but at batch scale (c=32+) HTTP/2 multiplexing dominates:

| Provider | c | Throughput | TTFT p50 | TTFT p95 | Notes |
|---|---|---|---|---|---|
| OpenAI | 32 | **1.5x** (2456 vs 1674) | **2.4x** (0.21 vs 0.51s) | **2.1x** (0.75 vs 1.59s) | HTTP/2 via `httpx[http2]` |
| OpenAI | 64 | **2.3x** (4096 vs 1783) | **2.6x** (0.29 vs 0.77s) | **5.2x** (0.56 vs 2.92s) | HTTP/2 multiplexing vs HTTP/1.1 connection limits |
| Together | 32 | **1.3x** (2010 vs 1597) | 1.2x (0.39 vs 0.48s) | 1.2x | Direct SDK avoids litellm overhead |
| Gemini | 64 | **1.6x** (4209 vs 2692) | ~same (0.89 vs 0.92s) | **2.1x** (1.10 vs 2.28s) | Native SDK; litellm returns `content=None` with thinking |
| Anthropic | -- | -- | -- | -- | Kept on litellm (direct SDK benchmarked slower) |

*Measured with streaming, n=64, litellm 1.81.13 (Feb 2026).*

Other features:
- **Lazy imports** -- 0.1s import vs 4s with litellm eager loading
- **Disk caching** -- deterministic SHA-256 keyed; works across all providers including direct SDK paths (litellm only caches its own calls)
- **Adaptive concurrency** -- header-aware for OpenAI (preemptive backoff from `x-ratelimit-*`), AIMD fallback for others
- **Batch API** -- 50% cost savings via OpenAI/Anthropic/Gemini batch endpoints
- **Cumulative usage tracking** -- input/output/cached tokens + cost per instance (litellm tracks globally, not per-instance)
- **Prefix cache tracking** for OpenAI/Anthropic

## Install / upgrade

```bash
uv add "llm-provider @ git+https://github.com/yoonho/llm-provider" --upgrade
```

## Usage

```python
from llm_provider import LLM

llm = LLM("gpt-4.1-nano")
results = llm.generate(["What is 2+2?", "Name a color."])
# [["4"], ["Blue"]]

# With system prompt
llm = LLM("gemini/gemini-3-flash-preview")
results = llm.generate("Solve: 2+2", system_prompt="Be concise.")

# Batching happens automatically -- all prompts run concurrently
llm = LLM("gpt-4.1-mini")
results = llm.generate([f"Question {i}" for i in range(100)])

# Check usage
print(llm.total_input_tokens, llm.total_output_tokens, llm.total_cost)
```

### Multi-turn chat

```python
result = llm.chat([
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
])  # -> "6"
```

### Multi-model parallel

Query multiple models on the same prompt(s), in parallel:

```python
from llm_provider import multi_generate, multi_chat

# Batch generation across models
results = multi_generate(
    ["gpt-4.1-mini", "gemini/gemini-3-flash-preview", "anthropic/claude-sonnet-4-6"],
    ["What is 2+2?", "Name a color."],
)
# {"gpt-4.1-mini": [["4"], ["Blue"]], ...}

# Chat across models
results = multi_chat(
    ["gpt-4.1-mini", "openrouter/anthropic/claude-sonnet-4"],
    [{"role": "user", "content": "Hello"}],
)
# {"gpt-4.1-mini": "Hi!", "openrouter/anthropic/claude-sonnet-4": "Hello!"}
```

### Batch API (50% cost savings)

For offline workloads, the batch API submits to provider batch endpoints at half the cost of real-time requests. Same return format as `generate()`.

```python
llm = LLM("gpt-4.1-mini")

# All-in-one: submit, poll, retrieve
results = llm.batch(["What is 2+2?", "Name a color."], system_prompt="Be concise.")
# [["4"], ["Blue"]]  -- same format as generate()

# Split workflow for long-running jobs
batch_id = llm.batch_submit(["What is 2+2?", "Name a color."])
# "openai:2:batch_abc123"

status = llm.batch_status(batch_id)
# {"status": "in_progress", "counts": {"completed": 1, "failed": 0, "total": 2}}

results = llm.batch_retrieve(batch_id)  # None if not done
# [["4"], ["Blue"]]
```

`batch()` polls every 60s by default (configurable via `poll_interval`). Batch IDs are self-contained strings -- you can save them and retrieve results in a different session.

Supported models:
- `gpt-*`, `o3-*`, `o4-*` (OpenAI)
- `anthropic/*` (Anthropic)
- `gemini/*` (Gemini)
- Others raise `NotImplementedError`

## Model naming

Uses litellm conventions:
- `gpt-4.1-nano`, `gpt-4.1-mini`, `o3-mini` -- OpenAI (direct SDK + HTTP/2)
- `gemini/gemini-3-flash-preview` -- Gemini (native SDK)
- `together_ai/meta-llama/...` -- Together (direct SDK)
- `sambanova/Meta-Llama-3.3-70B-Instruct` -- SambaNova (direct SDK)
- `openrouter/anthropic/claude-sonnet-4` -- OpenRouter (sorts by price)
- `local/model-name` -- Local OpenAI-compatible server (vLLM, SGLang, Ollama)
- `anthropic/claude-sonnet-4-6` -- Anthropic (litellm fallback)
- Everything else -- litellm fallback

## Environment variables

API keys (single key or comma-separated for rotation):
- `OPENAI_API_KEY` / `OPENAI_KEYS`
- `GEMINI_API_KEY` / `GEMINI_KEYS`
- `TOGETHER_API_KEY` / `TOGETHER_KEYS`
- `SAMBANOVA_API_KEY` / `SAMBANOVA_KEYS`
- `OPENROUTER_API_KEY` / `OPENROUTER_KEYS`

Config:
- `LLM_CACHE_DIR` -- override cache location (default: `/scr/yoonho/llm-cache`, fallback to `/tmp/llm_cache`)
- `LLM_GLOBAL_CONCURRENCY` -- cross-process concurrency limit via file locks (e.g., `64` for 16 processes sharing a rate limit)
- `LOCAL_BASE_URL` -- override local server URL (default: `http://localhost:30000/v1`)
