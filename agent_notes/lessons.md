# Lessons Learned

## 2026-02-17: llm_provider.py optimization

### Architecture: per-provider routing
Provider detection: Gemini (`gemini/*`), OpenAI (`gpt-*`, `o1*`, `o3*`, `o4*`, `chatgpt-*`), Together (`together_ai/*`), everything else → litellm.
- OpenAI: `AsyncOpenAI(http_client=httpx.AsyncClient(http2=True))` -- HTTP/2 is the key
- Together: `AsyncOpenAI(api_key=..., base_url="https://api.together.xyz/v1")` -- default httpx
- Gemini: `genai.Client(api_key=...)` -- native SDK with streaming
- Anthropic/others: `litellm.acompletion()` -- litellm is faster than direct SDK for Anthropic
- All direct paths share `_direct_cache` (FanoutCache, shards=8) for disk caching
- Default `max_concurrent=32` (benchmarked optimal for OpenAI; c=64 triggers rate limiting)

### What worked
| Optimization | Impact | Notes |
|---|---|---|
| Direct OpenAI SDK + HTTP/2 | 1.5-2x throughput, 2x TTFT | `http2=True` is the only httpx knob that matters |
| Direct Together SDK | 1.6x throughput, 2.5x TTFT | Avoids litellm overhead |
| Native Gemini SDK | 1.4x TTFT + correctness | Fixes litellm content=None bug |
| Concurrency: 256 → 32 | Prevents rate limiting | c=32 = 3691 tok/s, c=64 = 2308 tok/s (OpenAI) |
| thinking_budget=0 | Fastest Gemini config | 0.67s vs 0.84s for thinking_level=MINIMAL |
| Client reuse | Critical | Per-request creation dropped Gemini 887 → 245 tok/s |

### What didn't work
| Idea | Result |
|---|---|
| httpx keepalive_expiry, pool_timeout, big pool | No improvement on top of http2 |
| TCP_NODELAY (socket_options) | Not supported by httpx.AsyncClient |
| HTTP/2 for Together | Hurt throughput |
| Direct Anthropic SDK | Slower than litellm (636 vs 1051 tok/s) |
| Scheduling order (longest-first) | Minimal effect with gather+semaphore |
| litellm 1.81.x | Worse bugs than 1.80.9 (pass-through broken, event loop spam) |

### Provider-specific notes

**OpenAI:**
- c=32 optimal, c=64 hits rate limit wall
- Prefix caching: automatic for gpt-4.1-mini+ (49% cached on 2nd batch, needs >=1024 token prefix). gpt-4.1-nano doesn't support it. Mainly cost savings (50% off), minimal TTFT impact.
- Long system prompts don't hurt TTFT or throughput

**Gemini 3 Flash:**
- `thinking_budget=0` disables thinking (fastest). `thinking_level` and `thinking_budget` can't be mixed (400 error). `thinking_budget=0` works on both Gemini 2.5 and 3.
- Default temperature 1.0 is optimal. Lower causes thinking loops.
- Scales linearly to c=64 without rate limiting (unlike OpenAI)
- TTFT is rock-steady (~0.87s) regardless of system prompt length
- Long system prompts don't hurt throughput at all (1986 tok/s with long sys vs 1796 without)
- Must reuse `genai.Client` across requests
- Cache key must reflect actual thinking_config
- `chunk` after async for is undefined if stream is empty -- always track `last_chunk = None`

**Together:**
- Default httpx settings are optimal. HTTP/2 hurts. Big pool doesn't help.
- Suffers badly with long system prompts (4.14s TTFT vs 0.23s for OpenAI)

**Anthropic:**
- Keep on litellm (direct SDK is slower)
- litellm 1.80.9–1.80.11 recommended (avoid 1.81.x)

**litellm:**
- Returns `content=None` for Gemini 3 with thinking enabled (except `thinking_budget=0`)
- LoggingWorker spams "bound to a different event loop" errors. Harmless noise.

### Future optimization ideas (not yet implemented)
- **Adaptive concurrency**: Ramp up on success, halve on 429.
- **Key rotation**: Round-robin across API keys for Nx rate limit multiplier.
- **Batch APIs**: OpenAI/Anthropic batch endpoints for offline work (50% cost reduction).
- **Structured output / JSON mode**: Constrained decoding eliminates retry-on-parse-failure.
- After HTTP/2 + direct SDK, the bottleneck is API servers. Further gains come from spending less, not going faster.
