# Lessons Learned

## 2026-02-17: llm_provider.py optimization

### Architecture: per-provider routing
Provider detection: Gemini (`gemini/*`), OpenAI (`gpt-*`, `o1*`, `o3*`, `o4*`, `chatgpt-*`), Together (`together_ai/*`), everything else → litellm.
- OpenAI: `AsyncOpenAI(http_client=httpx.AsyncClient(http2=True))` -- HTTP/2 is the key
- Together: `AsyncOpenAI(api_key=..., base_url="https://api.together.xyz/v1")` -- default httpx
- Gemini: `genai.Client(api_key=...)` -- native SDK with streaming
- Anthropic/others: `litellm.acompletion()` -- litellm is faster than direct SDK for Anthropic
- All direct paths share lazy FanoutCache (shards=8, on /scr NVMe) for disk caching
- Default `max_concurrent=32` (benchmarked optimal for OpenAI; c=64 triggers rate limiting)
- Adaptive concurrency via AIMD semaphore: +1 on success, halve on 429

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
| Direct Anthropic SDK | Slower than litellm at c=32 (1543 vs 1882 tok/s), noisy at c=64. HTTP/2 doesn't help. |
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
- Keep on litellm (direct SDK is slower). Re-tested 2026-02-17 with anthropic 0.81.0:
  - c=32: direct+h2 = 0.80-0.83x of litellm (consistently slower, 3 runs)
  - c=64: noisy (0.58x to 1.13x), no reliable advantage either way
  - TTFT p50 slightly better with direct+h2 at c=64 (~0.49s vs ~0.74s) but throughput isn't
  - HTTP/2 doesn't help Anthropic the way it helps OpenAI (non-streaming: h2 was 1275 vs default 1376 tok/s at c=32)
  - Hypothesis: Anthropic API may use connection-level rate limiting that negates HTTP/2 multiplexing benefits
- litellm 1.80.9–1.80.11 recommended (avoid 1.81.x)

**litellm:**
- Returns `content=None` for Gemini 3 with thinking enabled (except `thinking_budget=0`)
- LoggingWorker spams "bound to a different event loop" errors. Harmless noise.

### litellm dependency reduction (2026-02-17)
- Replaced `litellm.completion_cost()` with own `pricing.cost()` (from `prices.csv`) for all direct SDK paths (OpenAI, Together, Local). litellm is no longer imported by `openai_api.py`.
- litellm now only used for: (1) Anthropic API calls via `litellm_api.py`, (2) disk cache setup for the litellm fallback path in `_cache.py`.
- Removed `litellm_model` parameter from `openai_api.call()` -- was only needed for `litellm.completion_cost()`.
- To fully drop litellm: write a direct Anthropic provider. But direct Anthropic SDK benchmarked slower (636 vs 1051 tok/s), so litellm stays for now.

### Caching design notes
- `cache_key()`: SHA-256 of `json.dumps(sort_keys=True)`. Correct approach -- deterministic serialization + fixed-size key for compact SQLite index. blake2b would be faster but irrelevant next to API latency.
- FanoutCache shards=8: docs recommend one shard per concurrent writer. With max_concurrent=32 you could go higher, but cache writes are fast and rarely contend -- 8 is fine. More shards = more file handles + SQLite overhead for negligible benefit.
- diskcache handles arbitrary-length string keys, so the SHA-256 step isn't strictly necessary, but it keeps index size predictable for long prompts.

### Lazy imports + cache init (2026-02-17)

Import time: 4.2s → 0.12s (35x improvement). Test runtime: 7.2s → 1.1s.

**Root causes of slow import:**
- `FanoutCache(shards=8)` on NFS: **4.7s** (each shard opens a SQLite DB on NFS)
- `import litellm`: **1.5s** (pulls in pydantic, httpx, tons of submodules)
- `import google.genai`: **0.6s**
- `import openai`: **0.3s**

**Fixes:**
1. Lazy FanoutCache via `_LazyCache` proxy: init deferred to first `.get()`/`.set()` call
2. Lazy litellm import: only loaded when litellm fallback path is actually used
3. google.genai and openai were already lazy (imported inside `create_client()`)

**FanoutCache shards vs NFS init time:** (measured)
- 8 shards: 4.7s, 4 shards: 1.8s, 2 shards: 0.9s, 1 shard: 0.5s, no shards (Cache): 0.4s
- On /tmp (local disk): 8 shards = 0.18s. The NFS penalty is ~25x per shard.
- Switched default cache dir to `/scr/yoonho/llm-cache` (node-local NVMe, ~0.18s init)
- Kept shards=8 since /scr is fast. Falls back to NFS or /tmp if /scr unavailable.

### Adaptive concurrency / AIMD (2026-02-17)

Replaced fixed `asyncio.Semaphore(max_concurrent)` with AIMD controller.
- On success: window += 1 (up to max)
- On 429 RateLimitError: window //= 2 (floor at 4)
- Throughput: +9% OpenAI, +4% Gemini in benchmarks (within noise, but lower variance)
- Real value: robustness. Adapts to rate limit changes without manual tuning.
- Detection: `type(exc).__name__ == "RateLimitError"` or `status_code == 429`

### uvloop (2026-02-17) -- no improvement

Tested uvloop as event loop replacement. Results within noise on Python 3.13:
- OpenAI: 846 tok/s (baseline 965) -- actually worse
- Gemini: 775 tok/s (baseline 742) -- marginally better
Not worth the dependency. The asyncio bottleneck is API latency, not event loop scheduling.

### Concurrency sweep (2026-02-17)

n=64 prompts, streaming, direct SDK vs litellm at each concurrency level.

**OpenAI (gpt-4.1-nano):**
| c | Path | tok/s | TTFT p50 | TTFT p95 | wall |
|---|---|---|---|---|---|
| 32 | direct | 2456 | 0.213s | 0.753s | 3.50s |
| 32 | litellm | 1674 | 0.507s | 1.585s | 5.26s |
| 64 | direct | 4096 | 0.293s | 0.563s | 2.09s |
| 64 | litellm | 1783 | 0.766s | 2.924s | 4.70s |

Key: HTTP/2 multiplexing dominates at c=32+. litellm's HTTP/1.1 can't multiplex, so each concurrent request needs its own TCP connection, hitting limits.

**Gemini (gemini-3-flash-preview):**
| c | Path | tok/s | TTFT p50 | TTFT p95 | wall |
|---|---|---|---|---|---|
| 8 | direct | 682 | 0.836s | 1.025s | 12.57s |
| 8 | litellm | 640 | 0.826s | 1.039s | 13.15s |
| 16 | direct | 1369 | 0.800s | 1.063s | 6.28s |
| 16 | litellm | 1303 | 0.818s | 1.033s | 6.44s |
| 32 | direct | 2420 | 0.872s | 1.076s | 3.52s |
| 32 | litellm | 2329 | 0.814s | 1.057s | 3.75s |
| 64 | direct | 4209 | 0.889s | 1.096s | 2.00s |
| 64 | litellm | 2692 | 0.915s | 2.275s | 3.17s |

Key: Nearly identical until c=64, where direct SDK pulls ahead 1.6x. Gemini native SDK uses HTTP/2 by default. litellm's p95 TTFT degrades 2.1x at c=64.

**Together (Llama-3.3-70B-Instruct-Turbo):**
| c | Path | tok/s | TTFT p50 | TTFT p95 |
|---|---|---|---|---|
| 32 | direct | 2010 | 0.394s | 0.984s |
| 32 | litellm | 1597 | 0.475s | 1.145s |
| 64 | direct | 1145 | 0.850s | 2.625s |
| 64 | litellm | 1720 | 0.806s | 5.112s |

Key: Direct wins at c=32 (1.3x), but at c=64 Together's rate limits kick in and both degrade. HTTP/2 actually hurts Together (tested separately).

**Anthropic (claude-haiku-4.5, anthropic 0.81.0):**
| c | Path | tok/s | TTFT p50 | TTFT p95 |
|---|---|---|---|---|
| 8 | direct+h2 | 462 | 0.570s | 1.214s |
| 8 | litellm | 658 | 0.445s | 0.686s |
| 16 | direct+h2 | 940 | 0.446s | 0.911s |
| 16 | litellm | 1062 | 0.462s | 0.932s |
| 32 | direct+h2 | 1543 (median of 3) | 0.486s | 0.803s |
| 32 | litellm | 1861 (median of 3) | 0.544s | 0.776s |
| 64 | direct+h2 | 1587 (high variance) | 0.491s | 1.003s |
| 64 | litellm | 1972 (high variance) | 0.793s | 1.796s |

Key: litellm is consistently faster at c=8-32. At c=64, throughput is noisy (ratio 0.58-1.13x across runs). Direct+h2 has better TTFT p50 at c=64 but unreliable throughput advantage. HTTP/2 doesn't help Anthropic — may use per-connection rate limiting. Decision: keep on litellm.

### Future optimization ideas (not yet implemented)
- **Key rotation**: Round-robin across API keys for Nx rate limit multiplier.
- **Batch APIs**: OpenAI/Anthropic batch endpoints for offline work (50% cost reduction).
- **Structured output / JSON mode**: Constrained decoding eliminates retry-on-parse-failure.
- After HTTP/2 + direct SDK, the bottleneck is API servers. Further gains come from spending less, not going faster.
