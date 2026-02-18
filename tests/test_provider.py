"""Tests for llm_provider (no API keys needed)."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_provider._cache import cache_key
from llm_provider.provider import (
    LLM,
    _FileSlotSemaphore,
    _is_gemini,
    _is_local,
    _is_openai,
    _is_together,
    _is_rate_limit,
    _median,
)
from llm_provider.providers import gemini, local, openai_api, together


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


# --- Median ---


class TestMedian:
    def test_odd(self):
        assert _median([3, 1, 2]) == 2

    def test_even(self):
        assert _median([4, 1, 3, 2]) == 2.5

    def test_single(self):
        assert _median([42]) == 42


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

        llm._client.chat.completions.create = AsyncMock(side_effect=[rate_err, resp])

        try:
            results = llm.generate("test", silent=True)
        finally:
            oai_mod.direct_cache = original_cache

        assert results == [["ok"]]
        assert llm._client.chat.completions.create.call_count == 2

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
