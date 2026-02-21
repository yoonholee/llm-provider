# Plan: Multi-turn Chat, Sync API, SambaNova Provider

Status: complete

## Summary

Three features, two of which combine naturally:

1. **`LLM.chat(messages)`** — multi-turn chat completion, returns `str`
2. **Sync API** — `chat()` uses sync SDKs internally (no asyncio/nest_asyncio)
3. **SambaNova provider** — OpenAI-compatible, follows Together pattern

Features 1+2 ship together: `chat()` is fully synchronous and accepts a messages list.

## API Design

```python
# New method
result = llm.chat([
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
])  # -> "6"

# Signature
def chat(
    self,
    messages: list[dict[str, str]],
    cache: bool = True,
    silent: bool = False,
    **kwargs: Any,
) -> str

# SambaNova
llm = LLM("sambanova/Meta-Llama-3.3-70B-Instruct")
```

## Architecture

### chat() is sync — no asyncio

`generate()` keeps async batching (needs concurrency). `chat()` uses sync SDK clients:

| Provider | Async (generate) | Sync (chat) |
|---|---|---|
| OpenAI | `AsyncOpenAI` | `OpenAI` (with HTTP/2) |
| Together | `AsyncOpenAI` + base_url | `OpenAI` + base_url |
| Local | `AsyncOpenAI` + base_url | `OpenAI` + base_url |
| SambaNova | `AsyncOpenAI` + base_url | `OpenAI` + base_url |
| Gemini | `pool.current.aio.models` | `pool.current.models` (same client) |
| litellm | `litellm.acompletion()` | `litellm.completion()` |

Sync client is lazy-created on first `chat()` call via `_get_sync_client()`.

### Retry logic

`chat()` has its own retry loop (sync `time.sleep`), same backoff policy as `_batch()`:
- 429 → exponential backoff + jitter + key rotation
- Non-429 → raise immediately
- No AIMD semaphore (single call, no concurrency to manage)

### Cache key for messages

New `cache_key_messages(model, messages, config)` in `_cache.py`. Separate namespace from prompt-based keys (different JSON structure ensures no collision).

## File Changes

### `_cache.py`
- Add `cache_key_messages(model: str, messages: list, config: dict) -> str`

### `provider.py`
- Add `_is_sambanova(model)` detection
- Add `LLM._sync_client = None` in `__init__`
- Add `LLM._get_sync_client()` — lazy sync client creation
- Add `LLM._call_messages_sync(messages, **kwargs)` — dispatch to providers
- Add `LLM.chat(messages, cache, silent, **kwargs) -> str` — public API
- Update `__init__` and `_call` for SambaNova routing
- Import sambanova provider

### `providers/openai_api.py`
- Add `create_sync_client(max_retries)` — `OpenAI` with HTTP/2, key rotation via ClientPool
- Add `call_messages_sync(client, model_id, messages, **kwargs)` — sync call with messages

### `providers/gemini.py`
- Add `call_messages_sync(pool, model, messages, **kwargs)`
  - Convert messages: system→system_instruction, assistant→role="model"
  - Use sync non-streaming: `pool.current.models.generate_content()`

### `providers/litellm_api.py`
- Add `call_messages_sync(model, messages, max_retries, **kwargs)`
  - Use `litellm.completion()` (sync) instead of `litellm.acompletion()`

### `providers/together.py`
- Add `create_sync_client(max_retries)` — `OpenAI` with Together base_url

### `providers/local.py`
- Add `create_sync_client(max_retries)` — `OpenAI` with local base_url

### `providers/sambanova.py` (new)
- `create_client(max_retries)` — `AsyncOpenAI` with SambaNova base_url
- `create_sync_client(max_retries)` — `OpenAI` with SambaNova base_url
- `model_id(model)` — strip `sambanova/` prefix

### `prices.csv`
- Add SambaNova model pricing (look up during implementation)

### `tests/test_provider.py`
- `TestSambaNovaProvider` — detection, model_id, client creation
- `TestChatCacheKey` — determinism, namespace separation, ordering
- `TestLLMChat` — returns str, multi-turn, passthrough, caching, cost tracking, 429 retry

## Implementation Order

1. `_cache.py`: `cache_key_messages`
2. `providers/sambanova.py`: new provider
3. `providers/openai_api.py`: `create_sync_client`, `call_messages_sync`
4. `providers/together.py`: `create_sync_client`
5. `providers/local.py`: `create_sync_client`
6. `providers/gemini.py`: `call_messages_sync`
7. `providers/litellm_api.py`: `call_messages_sync`
8. `provider.py`: `chat()`, `_is_sambanova`, sync client dispatch
9. `prices.csv`: SambaNova pricing
10. Run tests
