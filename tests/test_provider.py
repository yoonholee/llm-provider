"""Tests for llm_provider (no API keys needed)."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_provider._cache import cache_key
from llm_provider.provider import (
    LLM,
    _AdaptiveSemaphore,
    _FileSlotSemaphore,
    _MIN_CONCURRENCY,
    _is_gemini,
    _is_local,
    _is_openai,
    _is_openrouter,
    _is_together,
    _is_rate_limit,
    multi_chat,
    multi_generate,
)
from llm_provider.providers import gemini, local, openai_api, openrouter, together
from llm_provider.providers._headers import parse_openai_headers


# --- Header parsing ---


class TestHeaderParsing:
    def test_openai_headers_present(self):
        headers = {
            "x-ratelimit-remaining-requests": "450",
            "x-ratelimit-limit-requests": "500",
        }
        result = parse_openai_headers(headers)
        assert result == {"remaining": 450, "limit": 500}

    def test_openai_headers_missing(self):
        assert parse_openai_headers({}) is None

    def test_openai_headers_partial(self):
        headers = {"x-ratelimit-remaining-requests": "450"}
        assert parse_openai_headers(headers) is None

    def test_openai_headers_zero_limit(self):
        headers = {
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-limit-requests": "0",
        }
        assert parse_openai_headers(headers) is None

    def test_openai_headers_non_numeric(self):
        headers = {
            "x-ratelimit-remaining-requests": "abc",
            "x-ratelimit-limit-requests": "500",
        }
        assert parse_openai_headers(headers) is None


# --- Adaptive semaphore ---


class TestAdaptiveSemaphore:
    def test_aimd_additive_increase(self):
        """Without headers, on_success should +1 (AIMD)."""
        sem = _AdaptiveSemaphore(10)
        # Start at max, on_success should be no-op
        assert sem.window == 10
        sem.on_success()
        assert sem.window == 10

    def test_aimd_increase_from_below_max(self):
        """AIMD should grow window by 1 when below max."""
        sem = _AdaptiveSemaphore(10)
        # Force window down
        sem.on_rate_limit()  # 10 -> 5
        assert sem.window == 5
        sem.on_success()
        assert sem.window == 6

    def test_rate_limit_halves(self):
        """on_rate_limit should halve the window."""
        sem = _AdaptiveSemaphore(32)
        sem.on_rate_limit()
        assert sem.window == 16
        sem.on_rate_limit()
        assert sem.window == 8
        sem.on_rate_limit()
        assert sem.window == _MIN_CONCURRENCY

    def test_rate_limit_floor(self):
        """Window should never go below _MIN_CONCURRENCY."""
        sem = _AdaptiveSemaphore(8)
        for _ in range(10):
            sem.on_rate_limit()
        assert sem.window == _MIN_CONCURRENCY

    def test_header_above_threshold_grows(self):
        """With plenty of headroom (>10%), headers should grow like AIMD."""
        sem = _AdaptiveSemaphore(32)
        sem.on_rate_limit()  # Drop to 16
        assert sem.window == 16
        # 50% remaining -> above threshold -> should grow +1
        sem.on_headers(250, 500)
        assert sem.window == 17

    def test_header_above_threshold_grows_to_max(self):
        """Repeated above-threshold headers should grow to max."""
        sem = _AdaptiveSemaphore(10)
        sem.on_rate_limit()  # Drop to 5
        for _ in range(10):
            sem.on_headers(400, 500)  # 80% remaining
        assert sem.window == 10

    def test_header_below_threshold_backs_off(self):
        """When remaining < 10% of limit, window should drop proportionally."""
        sem = _AdaptiveSemaphore(32)
        # 5% remaining -> scale = 0.5 -> target = 4 + 0.5*28 = 18
        sem.on_headers(25, 500)
        assert sem.window == 18

    def test_header_near_zero_hits_floor(self):
        """Near-zero remaining should drop to MIN_CONCURRENCY."""
        sem = _AdaptiveSemaphore(32)
        sem.on_headers(1, 500)  # 0.2% remaining
        assert sem.window == _MIN_CONCURRENCY

    def test_header_suppresses_plain_aimd(self):
        """Once headers arrive, on_success should be a no-op."""
        sem = _AdaptiveSemaphore(10)
        sem.on_rate_limit()  # Drop to 5
        assert sem.window == 5
        sem.on_headers(450, 500)  # Header signal (above threshold, grows +1)
        w_after_header = sem.window
        assert w_after_header == 6
        sem.on_success()
        assert sem.window == w_after_header  # No additional AIMD change

    def test_429_overrides_headers(self):
        """on_rate_limit should halve regardless of header state."""
        sem = _AdaptiveSemaphore(32)
        # Feed above-threshold headers to grow to max
        for _ in range(5):
            sem.on_headers(480, 500)
        assert sem.window == 32  # Still at max
        sem.on_rate_limit()
        assert sem.window == 16

    def test_recovery_after_backoff(self):
        """After header-triggered backoff, above-threshold headers should recover fast."""
        sem = _AdaptiveSemaphore(32)
        # Drop via low remaining
        sem.on_headers(5, 500)  # 1% -> target = 4 + 0.1*28 = 6
        low = sem.window
        assert low < 10
        # Recover: 10 above-threshold signals should grow by 10
        for _ in range(10):
            sem.on_headers(400, 500)  # 80% remaining
        assert sem.window == low + 10


# --- Rate limit detection ---


class TestIsRateLimit:
    def test_status_code_429(self):
        e = Exception()
        e.status_code = 429
        assert _is_rate_limit(e)

    def test_code_429_gemini(self):
        """google.genai.errors.ClientError uses .code for HTTP status."""
        e = Exception()
        e.code = 429
        assert _is_rate_limit(e)

    def test_rate_limit_error_by_name(self):
        class RateLimitError(Exception):
            pass

        assert _is_rate_limit(RateLimitError())

    def test_non_429(self):
        e = Exception()
        e.status_code = 500
        assert not _is_rate_limit(e)

    def test_real_gemini_client_error_429(self):
        """Regression: real google.genai ClientError with code=429 must be detected."""
        from google.genai.errors import ClientError as GeminiClientError

        e = GeminiClientError(429, "Resource exhausted")
        assert _is_rate_limit(e)

    def test_real_gemini_client_error_non_429(self):
        from google.genai.errors import ClientError as GeminiClientError

        e = GeminiClientError(400, "Bad request")
        assert not _is_rate_limit(e)


# --- Provider detection ---


class TestProviderDetection:
    def test_gemini(self):
        assert _is_gemini("gemini/gemini-3-flash-preview")
        assert _is_gemini("gemini/gemini-2.5-pro")
        assert not _is_gemini("gpt-4.1-nano")
        assert not _is_gemini("together_ai/meta-llama/foo")

    def test_openai(self):
        assert _is_openai("gpt-4.1-nano")
        assert _is_openai("gpt-4.1-mini")
        assert _is_openai("o3-mini")
        assert _is_openai("o4-mini")
        assert _is_openai("chatgpt-4o-latest")
        assert _is_openai("openai/gpt-4.1-nano")
        assert not _is_openai("gemini/gemini-3-flash-preview")
        assert not _is_openai("anthropic/claude-sonnet-4-6")

    def test_together(self):
        assert _is_together("together_ai/meta-llama/Llama-3-8b")
        assert not _is_together("gpt-4.1-nano")
        assert not _is_together("gemini/gemini-3-flash-preview")

    def test_local(self):
        assert _is_local("local/Qwen/Qwen3-4B")
        assert _is_local("local/meta-llama/Llama-3-8b")
        assert not _is_local("gpt-4.1-nano")
        assert not _is_local("together_ai/meta-llama/Llama-3-8b")

    def test_openrouter(self):
        assert _is_openrouter("openrouter/anthropic/claude-sonnet-4")
        assert _is_openrouter("openrouter/openai/gpt-4.1-nano")
        assert not _is_openrouter("gpt-4.1-nano")
        assert not _is_openrouter("together_ai/meta-llama/Llama-3-8b")


# --- Model ID extraction ---


class TestModelId:
    def test_gemini_model_id(self):
        assert (
            gemini.model_id("gemini/gemini-3-flash-preview") == "gemini-3-flash-preview"
        )

    def test_openai_model_id(self):
        assert openai_api.model_id("openai/gpt-4.1-nano") == "gpt-4.1-nano"
        assert openai_api.model_id("gpt-4.1-nano") == "gpt-4.1-nano"

    def test_together_model_id(self):
        assert (
            together.model_id("together_ai/meta-llama/Llama-3-8b")
            == "meta-llama/Llama-3-8b"
        )

    def test_local_model_id(self):
        assert local.model_id("local/Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"
        assert local.model_id("local/meta-llama/Llama-3-8b") == "meta-llama/Llama-3-8b"

    def test_openrouter_model_id(self):
        assert (
            openrouter.model_id("openrouter/anthropic/claude-sonnet-4")
            == "anthropic/claude-sonnet-4"
        )
        assert (
            openrouter.model_id("openrouter/openai/gpt-4.1-nano")
            == "openai/gpt-4.1-nano"
        )


# --- Cache key ---


class TestCacheKey:
    def test_deterministic(self):
        k1 = cache_key("model", "prompt", "sys", {"t": 0.7})
        k2 = cache_key("model", "prompt", "sys", {"t": 0.7})
        assert k1 == k2

    def test_different_inputs(self):
        k1 = cache_key("model", "prompt1", "sys", {})
        k2 = cache_key("model", "prompt2", "sys", {})
        assert k1 != k2

    def test_is_sha256(self):
        k = cache_key("m", "p", None, {})
        assert len(k) == 64  # SHA-256 hex digest


# --- Thinking tag stripping ---


class TestStripThinking:
    """Test strip_thinking with common reasoning model output formats."""

    def test_qwen3_closed(self):
        """Qwen3 style: <think>reasoning</think>answer"""
        raw = "<think>\nThe capital of France is Paris.\n</think>\n\nParis"
        assert openai_api.strip_thinking(raw) == "Paris"

    def test_deepseek_r1_closed(self):
        """DeepSeek-R1 also uses <think> tags."""
        raw = "<think>\nLet me reason step by step.\n1. First...\n2. Then...\n</think>\nThe answer is 42."
        assert openai_api.strip_thinking(raw) == "The answer is 42."

    def test_unclosed_think_truncated(self):
        """Model hit token limit during thinking — no closing tag."""
        raw = "<think>\nI need to think about this carefully. Let me consider"
        assert openai_api.strip_thinking(raw) == ""

    def test_no_think_tags_passthrough(self):
        """Non-thinking model output passes through unchanged."""
        raw = "The capital of France is Paris."
        assert openai_api.strip_thinking(raw) == raw

    def test_empty_think_block(self):
        """Edge case: empty thinking block."""
        raw = "<think></think>answer"
        assert openai_api.strip_thinking(raw) == "answer"

    def test_multiline_answer(self):
        """Answer after thinking can be multi-line."""
        raw = "<think>\nreasoning\n</think>\n\nLine 1\nLine 2\nLine 3"
        assert openai_api.strip_thinking(raw) == "Line 1\nLine 2\nLine 3"

    def test_whitespace_only_after_strip(self):
        """If thinking consumed everything and only whitespace remains."""
        raw = "<think>\nall thinking no answer\n</think>\n   \n"
        assert openai_api.strip_thinking(raw) == ""


class TestCallThinkingIntegration:
    """Test that openai_api.call() correctly handles thinking in responses."""

    @staticmethod
    def _make_response(content, reasoning_content=None):
        """Build a mock OpenAI response."""
        msg = MagicMock()
        msg.content = content
        msg.reasoning_content = reasoning_content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.prompt_tokens_details = None
        return resp

    @staticmethod
    def _run(coro):
        import asyncio

        return asyncio.run(coro)

    def _call_with_mock_cache(self, client, model_id, prompt):
        """Run openai_api.call() with cache bypassed."""
        import llm_provider.providers.openai_api as oai_mod

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        original = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache
        try:
            return self._run(openai_api.call(client, model_id, prompt))
        finally:
            oai_mod.direct_cache = original

    def test_server_parsed_reasoning(self):
        """When server sets reasoning_content, content is stripped of whitespace only."""
        client = AsyncMock()
        resp = self._make_response(
            content="\n\nParis",
            reasoning_content="The capital of France is Paris.",
        )
        client.chat.completions.create = AsyncMock(return_value=resp)

        texts, usage = self._call_with_mock_cache(client, "Qwen/Qwen3-4B", "test")
        assert texts == ["Paris"]

    def test_client_side_strip(self):
        """When server doesn't parse reasoning, strip <think> tags client-side."""
        client = AsyncMock()
        resp = self._make_response(
            content="<think>\nreasoning here\n</think>\n\nParis",
            reasoning_content=None,
        )
        client.chat.completions.create = AsyncMock(return_value=resp)

        texts, usage = self._call_with_mock_cache(client, "Qwen/Qwen3-4B", "test")
        assert texts == ["Paris"]

    def test_no_thinking_passthrough(self):
        """Non-thinking model output passes through unchanged."""
        client = AsyncMock()
        resp = self._make_response(content="Hello world")
        client.chat.completions.create = AsyncMock(return_value=resp)

        texts, usage = self._call_with_mock_cache(client, "gpt-4.1-nano", "test")
        assert texts == ["Hello world"]


# --- LLM class (mocked) ---


class TestLLMInit:
    def test_litellm_model_no_client(self):
        """Non-provider-specific models should init without errors or clients."""
        llm = LLM("anthropic/claude-sonnet-4-6")
        assert llm.model == "anthropic/claude-sonnet-4-6"
        assert llm.total_input_tokens == 0
        assert not hasattr(llm, "_client")

    def test_openai_creates_client(self):
        llm = LLM("gpt-4.1-nano")
        assert hasattr(llm, "_client")

    def test_gemini_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                LLM("gemini/gemini-3-flash-preview")

    def test_together_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
                LLM("together_ai/meta-llama/Llama-3-8b")


class TestLLMGenerate:
    def test_litellm_generate(self):
        """Test generate() with mocked litellm.acompletion."""
        llm = LLM("anthropic/claude-sonnet-4-6")

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.prompt_tokens_details = None

        with patch("llm_provider.providers.litellm_api._litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                return_value=mock_response
            )
            mock_litellm.return_value.completion_cost.return_value = 0.001

            results = llm.generate("Hi", silent=True)

        assert results == [["Hello!"]]
        assert llm.total_input_tokens == 10
        assert llm.total_output_tokens == 5

    def test_string_input_wraps_to_list(self):
        """Single string prompt should be wrapped in a list."""
        llm = LLM("anthropic/claude-sonnet-4-6")

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.prompt_tokens_details = None

        with patch("llm_provider.providers.litellm_api._litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                return_value=mock_response
            )
            mock_litellm.return_value.completion_cost.return_value = 0.0

            results = llm.generate("single prompt", silent=True)

        assert len(results) == 1

    def test_openai_cache_hit(self):
        """OpenAI path should return cached response without API call."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")

        mock_cache = MagicMock()
        mock_cache.get.return_value = "cached!"
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache
        try:
            results = llm.generate("test", silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert results == [["cached!"]]
        assert llm.total_input_tokens == 0

    def test_cache_false_skips_cache(self):
        """cache=False should call API even when cache has value."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")

        mock_cache = MagicMock()
        mock_cache.get.return_value = "cached!"
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache

        # Mock API response
        msg = MagicMock()
        msg.content = "fresh!"
        msg.reasoning_content = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.prompt_tokens_details = None
        llm._client.chat.completions.create = AsyncMock(return_value=resp)

        try:
            results = llm.generate("test", cache=False, silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert results == [["fresh!"]]
        # Cache.get should not have been called
        mock_cache.get.assert_not_called()
        # Cache.set should not have been called
        mock_cache.set.assert_not_called()

    def test_cache_hit_with_list(self):
        """Cache storing a list (from n>1) should return correctly."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")

        mock_cache = MagicMock()
        mock_cache.get.return_value = ["resp1", "resp2", "resp3"]
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache
        try:
            results = llm.generate("test", silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert results == [["resp1", "resp2", "resp3"]]

    def test_n_returns_multiple(self):
        """n=3 should return 3 responses per prompt."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")

        # Mock cache miss
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache

        # Mock API returning 3 choices
        choices = []
        for text in ["resp1", "resp2", "resp3"]:
            msg = MagicMock()
            msg.content = text
            msg.reasoning_content = None
            choice = MagicMock()
            choice.message = msg
            choices.append(choice)
        resp = MagicMock()
        resp.choices = choices
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 15
        resp.usage.prompt_tokens_details = None
        llm._client.chat.completions.create = AsyncMock(return_value=resp)

        try:
            results = llm.generate("test", n=3, silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert results == [["resp1", "resp2", "resp3"]]
        # Should have cached the full list
        mock_cache.set.assert_called_once()
        cached_value = mock_cache.set.call_args[0][1]
        assert cached_value == ["resp1", "resp2", "resp3"]

    def test_n_changes_cache_key(self):
        """n>1 should produce a different cache key than n=1."""
        k1 = cache_key("model", "prompt", None, {})
        k2 = cache_key("model", "prompt", None, {"n": 3})
        assert k1 != k2


# --- Retry on 429 ---


class TestRetryOn429:
    def test_retries_on_rate_limit(self):
        """429 errors should be retried with backoff, not raised immediately."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache

        # First call raises 429, second succeeds
        rate_err = Exception("rate limit")
        rate_err.status_code = 429

        msg = MagicMock()
        msg.content = "ok"
        msg.reasoning_content = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.prompt_tokens_details = None

        # Replace _client entirely so key rotation can't escape to unmocked clients
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[rate_err, resp])
        llm._client = mock_client

        try:
            results = llm.generate("test", silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert results == [["ok"]]
        assert mock_client.chat.completions.create.call_count == 2

    def test_non_429_errors_raise_immediately(self):
        """Non-rate-limit errors should not be retried."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache

        llm._client.chat.completions.create = AsyncMock(
            side_effect=ValueError("bad input")
        )

        try:
            with pytest.raises(ValueError, match="bad input"):
                llm.generate("test", silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert llm._client.chat.completions.create.call_count == 1


# --- File-based global semaphore ---


class TestFileSlotSemaphore:
    def test_limits_concurrency(self):
        """FileSlotSemaphore should limit concurrent access to max_slots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sem = _FileSlotSemaphore(max_slots=2, lock_dir=tmpdir)
            active = []
            max_active = [0]

            async def worker(i):
                async with sem:
                    active.append(i)
                    max_active[0] = max(max_active[0], len(active))
                    await asyncio.sleep(0.05)
                    active.remove(i)

            async def run_all():
                await asyncio.gather(*[worker(i) for i in range(6)])

            asyncio.run(run_all())
            assert max_active[0] <= 2

    def test_releases_on_exception(self):
        """Slots should be released even if the task raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sem = _FileSlotSemaphore(max_slots=1, lock_dir=tmpdir)

            async def failing():
                async with sem:
                    raise RuntimeError("boom")

            async def succeeding():
                async with sem:
                    return "ok"

            async def run():
                with pytest.raises(RuntimeError):
                    await failing()
                return await succeeding()

            result = asyncio.run(run())
            assert result == "ok"

    def test_namespace_isolation(self):
        """Different namespaces should use different lock files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sem_a = _FileSlotSemaphore(max_slots=1, lock_dir=tmpdir, namespace="userA")
            sem_b = _FileSlotSemaphore(max_slots=1, lock_dir=tmpdir, namespace="userB")

            async def run():
                # Both should acquire without blocking each other
                await sem_a.acquire()
                await sem_b.acquire()
                sem_a.release()
                sem_b.release()

            asyncio.run(run())

    def test_double_acquire_releases_old_fd(self):
        """Double acquire on same task should release the old fd, not leak it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sem = _FileSlotSemaphore(max_slots=2, lock_dir=tmpdir)

            async def run():
                await sem.acquire()
                # Acquire again without releasing — old fd should be cleaned up
                await sem.acquire()
                sem.release()

            asyncio.run(run())


# --- SambaNova provider ---


class TestSambaNovaProvider:
    def test_detection(self):
        from llm_provider.provider import _is_sambanova

        assert _is_sambanova("sambanova/Meta-Llama-3.3-70B-Instruct")
        assert _is_sambanova("sambanova/DeepSeek-R1")
        assert not _is_sambanova("together_ai/meta-llama/Llama-3-8b")
        assert not _is_sambanova("gpt-4.1-nano")
        assert not _is_sambanova("gemini/gemini-3-flash-preview")

    def test_model_id(self):
        from llm_provider.providers import sambanova

        assert (
            sambanova.model_id("sambanova/Meta-Llama-3.3-70B-Instruct")
            == "Meta-Llama-3.3-70B-Instruct"
        )

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="SAMBANOVA_API_KEY"):
                LLM("sambanova/Meta-Llama-3.3-70B-Instruct")

    def test_creates_client(self):
        with patch.dict("os.environ", {"SAMBANOVA_API_KEY": "test-key"}):
            llm = LLM("sambanova/Meta-Llama-3.3-70B-Instruct")
            assert hasattr(llm, "_client")


# --- OpenRouter provider ---


class TestOpenRouterProvider:
    def test_detection(self):
        assert _is_openrouter("openrouter/anthropic/claude-sonnet-4")
        assert _is_openrouter("openrouter/openai/gpt-4.1-nano")
        assert not _is_openrouter("together_ai/meta-llama/Llama-3-8b")
        assert not _is_openrouter("gpt-4.1-nano")

    def test_model_id(self):
        assert (
            openrouter.model_id("openrouter/anthropic/claude-sonnet-4")
            == "anthropic/claude-sonnet-4"
        )

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                LLM("openrouter/anthropic/claude-sonnet-4")

    def test_creates_client(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            llm = LLM("openrouter/anthropic/claude-sonnet-4")
            assert hasattr(llm, "_client")

    def test_inject_default_provider(self):
        """inject_provider_kwargs should add sort=price by default."""
        result = openrouter.inject_provider_kwargs({})
        assert result["extra_body"]["provider"]["sort"] == "price"

    def test_inject_preserves_existing_provider(self):
        """User-supplied provider config should not be overwritten."""
        kwargs = {"extra_body": {"provider": {"sort": "throughput"}}}
        result = openrouter.inject_provider_kwargs(kwargs)
        assert result["extra_body"]["provider"]["sort"] == "throughput"

    def test_inject_preserves_other_extra_body(self):
        """Other extra_body keys should be preserved."""
        kwargs = {"extra_body": {"transforms": ["middle-out"]}}
        result = openrouter.inject_provider_kwargs(kwargs)
        assert result["extra_body"]["transforms"] == ["middle-out"]
        assert result["extra_body"]["provider"]["sort"] == "price"

    def test_inject_does_not_mutate_input(self):
        """inject_provider_kwargs should not mutate the original dict."""
        original = {"temperature": 0.7}
        result = openrouter.inject_provider_kwargs(original)
        assert "extra_body" not in original
        assert "extra_body" in result


# --- Chat cache key ---


class TestChatCacheKey:
    def test_deterministic(self):
        from llm_provider._cache import cache_key_messages

        msgs = [{"role": "user", "content": "hello"}]
        k1 = cache_key_messages("model", msgs, {})
        k2 = cache_key_messages("model", msgs, {})
        assert k1 == k2

    def test_differs_from_prompt_key(self):
        """Messages cache key must differ from single-prompt cache key."""
        from llm_provider._cache import cache_key_messages

        msg_key = cache_key_messages(
            "model", [{"role": "user", "content": "hello"}], {}
        )
        prompt_key = cache_key("model", "hello", None, {})
        assert msg_key != prompt_key

    def test_different_messages(self):
        from llm_provider._cache import cache_key_messages

        k1 = cache_key_messages("model", [{"role": "user", "content": "a"}], {})
        k2 = cache_key_messages("model", [{"role": "user", "content": "b"}], {})
        assert k1 != k2

    def test_message_order_matters(self):
        from llm_provider._cache import cache_key_messages

        msgs1 = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        msgs2 = [
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "a"},
        ]
        k1 = cache_key_messages("model", msgs1, {})
        k2 = cache_key_messages("model", msgs2, {})
        assert k1 != k2

    def test_is_sha256(self):
        from llm_provider._cache import cache_key_messages

        k = cache_key_messages("m", [{"role": "user", "content": "p"}], {})
        assert len(k) == 64


# --- LLM.chat() ---


class TestLLMChat:
    def test_returns_string(self):
        """chat() should return a string, not list of lists."""
        llm = LLM("anthropic/claude-sonnet-4-6")

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.prompt_tokens_details = None

        with patch("llm_provider.providers.litellm_api._litellm") as mock_litellm:
            mock_litellm.return_value.completion = MagicMock(return_value=mock_response)
            mock_litellm.return_value.completion_cost.return_value = 0.001

            result = llm.chat([{"role": "user", "content": "Hi"}], silent=True)

        assert isinstance(result, str)
        assert result == "Hello!"

    def test_multi_turn(self):
        """chat() with multi-turn conversation should work."""
        llm = LLM("anthropic/claude-sonnet-4-6")

        mock_choice = MagicMock()
        mock_choice.message.content = "6"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 1
        mock_response.usage.prompt_tokens_details = None

        with patch("llm_provider.providers.litellm_api._litellm") as mock_litellm:
            mock_litellm.return_value.completion = MagicMock(return_value=mock_response)
            mock_litellm.return_value.completion_cost.return_value = 0.0

            result = llm.chat(
                [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "And 3+3?"},
                ],
                silent=True,
            )

        assert result == "6"

    def test_passes_messages_directly(self):
        """chat() should pass messages to the provider without modification."""
        llm = LLM("anthropic/claude-sonnet-4-6")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]

        mock_choice = MagicMock()
        mock_choice.message.content = "I'm good!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 3
        mock_response.usage.prompt_tokens_details = None

        with patch("llm_provider.providers.litellm_api._litellm") as mock_litellm:
            mock_litellm.return_value.completion = MagicMock(return_value=mock_response)
            mock_litellm.return_value.completion_cost.return_value = 0.0

            llm.chat(messages, silent=True)

            call_kwargs = mock_litellm.return_value.completion.call_args
            assert call_kwargs.kwargs["messages"] == messages

    def test_cost_tracking(self):
        """chat() should accumulate token counts and cost."""
        llm = LLM("anthropic/claude-sonnet-4-6")

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens_details = None

        with patch("llm_provider.providers.litellm_api._litellm") as mock_litellm:
            mock_litellm.return_value.completion = MagicMock(return_value=mock_response)
            mock_litellm.return_value.completion_cost.return_value = 0.005

            llm.chat([{"role": "user", "content": "test"}], silent=True)

        assert llm.total_input_tokens == 20
        assert llm.total_output_tokens == 10
        assert llm.total_cost == 0.005

    def test_openai_cache_hit(self):
        """OpenAI chat() should return cached response without API call."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")
        llm._sync_client = MagicMock()  # Pre-set to avoid lazy creation

        mock_cache = MagicMock()
        mock_cache.get.return_value = "cached response"
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache
        try:
            result = llm.chat([{"role": "user", "content": "test"}], silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert result == "cached response"
        # API should not have been called
        llm._sync_client.chat.completions.create.assert_not_called()

    def test_openai_cache_false(self):
        """cache=False should bypass cache for chat()."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")

        mock_cache = MagicMock()
        mock_cache.get.return_value = "cached"
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache

        msg = MagicMock()
        msg.content = "fresh"
        msg.reasoning_content = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.prompt_tokens_details = None

        sync_client = MagicMock()
        sync_client.chat.completions.create.return_value = resp
        llm._sync_client = sync_client

        try:
            result = llm.chat(
                [{"role": "user", "content": "test"}],
                cache=False,
                silent=True,
            )
        finally:
            oai_mod.direct_cache = original_cache

        assert result == "fresh"
        mock_cache.get.assert_not_called()

    def test_openai_multi_turn(self):
        """chat() passes multi-turn messages through to OpenAI."""
        import llm_provider.providers.openai_api as oai_mod

        llm = LLM("gpt-4.1-nano")

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        original_cache = oai_mod.direct_cache
        oai_mod.direct_cache = mock_cache

        msg = MagicMock()
        msg.content = "6"
        msg.reasoning_content = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.prompt_tokens = 50
        resp.usage.completion_tokens = 1
        resp.usage.prompt_tokens_details = None

        sync_client = MagicMock()
        sync_client.chat.completions.create.return_value = resp
        llm._sync_client = sync_client

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        try:
            result = llm.chat(messages, silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert result == "6"
        # All 4 messages should be passed through
        call_kwargs = sync_client.chat.completions.create.call_args
        assert len(call_kwargs.kwargs["messages"]) == 4

    def test_retry_on_429(self):
        """chat() should retry on 429 errors with backoff."""
        llm = LLM("anthropic/claude-sonnet-4-6")

        rate_err = Exception("rate limit")
        rate_err.status_code = 429

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.prompt_tokens_details = None

        with (
            patch("llm_provider.providers.litellm_api._litellm") as mock_litellm,
            patch("time.sleep"),
        ):
            mock_litellm.return_value.completion = MagicMock(
                side_effect=[rate_err, mock_response]
            )
            mock_litellm.return_value.completion_cost.return_value = 0.0

            result = llm.chat([{"role": "user", "content": "test"}], silent=True)

        assert result == "ok"
        assert mock_litellm.return_value.completion.call_count == 2


# --- Multi-model parallel ---


class TestMultiGenerate:
    def test_returns_dict_keyed_by_model(self):
        """multi_generate should return results keyed by model name."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.prompt_tokens_details = None

        with patch("llm_provider.providers.litellm_api._litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                return_value=mock_response
            )
            mock_litellm.return_value.completion_cost.return_value = 0.0

            results = multi_generate(
                ["anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5"],
                "Hi",
                silent=True,
            )

        assert set(results.keys()) == {
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
        }
        assert results["anthropic/claude-sonnet-4-6"] == [["Hello!"]]
        assert results["anthropic/claude-haiku-4-5"] == [["Hello!"]]


class TestMultiChat:
    def test_returns_dict_of_strings(self):
        """multi_chat should return {model: response_string}."""
        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.prompt_tokens_details = None

        with patch("llm_provider.providers.litellm_api._litellm") as mock_litellm:
            mock_litellm.return_value.completion = MagicMock(return_value=mock_response)
            mock_litellm.return_value.completion_cost.return_value = 0.0

            results = multi_chat(
                ["anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5"],
                [{"role": "user", "content": "Hello"}],
                silent=True,
            )

        assert set(results.keys()) == {
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
        }
        for v in results.values():
            assert isinstance(v, str)
            assert v == "response"
