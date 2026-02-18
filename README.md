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
- **Adaptive concurrency** -- AIMD: ramps up on success, backs off on 429 (litellm uses fixed concurrency)
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

## Model naming

Uses litellm conventions:
- `gpt-4.1-nano`, `gpt-4.1-mini`, `o3-mini` -- OpenAI (direct SDK)
- `gemini/gemini-3-flash-preview` -- Gemini (native SDK)
- `together_ai/meta-llama/...` -- Together (direct SDK)
- `anthropic/claude-sonnet-4-6` -- Anthropic (litellm fallback)
- Everything else -- litellm fallback

## Environment variables

- `OPENAI_API_KEY` -- for OpenAI models
- `GEMINI_API_KEY` -- for Gemini models
- `TOGETHER_API_KEY` -- for Together models
- `LLM_CACHE_DIR` -- override cache location (default: `/scr/yoonho/llm-cache`, fallback to `/tmp/llm_cache`)
- `LLM_GLOBAL_CONCURRENCY` -- cross-process concurrency limit via file locks (e.g., `64` for 16 processes sharing a rate limit)
