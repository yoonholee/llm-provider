# llm-provider

Thin LLM wrapper that bypasses litellm's overhead for major providers. Uses direct SDKs with provider-specific optimizations, falling back to litellm for everything else.

## Why not just litellm?

litellm is convenient but adds measurable overhead. Benchmarked improvements from direct SDK routing:

| Provider | Improvement | Key optimization |
|---|---|---|
| OpenAI | 1.5-2x throughput, 2x TTFT | HTTP/2 via `httpx[http2]` |
| Together | 1.6x throughput, 2.5x TTFT | Skip litellm dispatch |
| Gemini | 1.4x TTFT + correctness | Native `google.genai` SDK (litellm returns `content=None` with thinking enabled) |
| Anthropic | -- | Kept on litellm (direct SDK is actually slower) |

Other features litellm doesn't give you out of the box:
- **Disk caching** across all providers (deterministic, SHA-256 keyed)
- **Adaptive concurrency** (AIMD: ramps up on success, backs off on 429)
- **Lazy imports** (0.1s import vs 4s with litellm eager loading)
- **Cumulative usage tracking** (input/output/cached tokens, cost)
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
