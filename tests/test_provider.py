"""Tests for llm_provider.provider (no API keys needed)."""

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_provider.provider import (
    LLM,
    _cache_key,
    _is_gemini,
    _is_openai,
    _is_together,
    _gemini_model_id,
    _median,
    _openai_model_id,
    _together_model_id,
)


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


# --- Model ID extraction ---


class TestModelId:
    def test_gemini_model_id(self):
        assert (
            _gemini_model_id("gemini/gemini-3-flash-preview")
            == "gemini-3-flash-preview"
        )

    def test_openai_model_id(self):
        assert _openai_model_id("openai/gpt-4.1-nano") == "gpt-4.1-nano"
        assert _openai_model_id("gpt-4.1-nano") == "gpt-4.1-nano"

    def test_together_model_id(self):
        assert (
            _together_model_id("together_ai/meta-llama/Llama-3-8b")
            == "meta-llama/Llama-3-8b"
        )


# --- Cache key ---


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("model", "prompt", "sys", {"t": 0.7})
        k2 = _cache_key("model", "prompt", "sys", {"t": 0.7})
        assert k1 == k2

    def test_different_inputs(self):
        k1 = _cache_key("model", "prompt1", "sys", {})
        k2 = _cache_key("model", "prompt2", "sys", {})
        assert k1 != k2

    def test_is_sha256(self):
        k = _cache_key("m", "p", None, {})
        assert len(k) == 64  # SHA-256 hex digest


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
        assert not hasattr(llm, "_openai_client")
        assert not hasattr(llm, "_genai_client")

    def test_openai_creates_client(self):
        llm = LLM("gpt-4.1-nano")
        assert hasattr(llm, "_openai_client")

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

        with patch("llm_provider.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.completion_cost.return_value = 0.001

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

        with patch("llm_provider.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.completion_cost.return_value = 0.0

            results = llm.generate("single prompt", silent=True)

        assert len(results) == 1

    def test_openai_cache_hit(self):
        """OpenAI path should return cached response without API call."""
        import llm_provider.provider as mod

        llm = LLM("gpt-4.1-nano")

        mock_cache = MagicMock()
        mock_cache.get.return_value = "cached!"
        original_cache = mod._direct_cache
        mod._direct_cache = mock_cache
        try:
            results = llm.generate("test", silent=True)
        finally:
            mod._direct_cache = original_cache

        assert results == [["cached!"]]
        assert llm.total_input_tokens == 0
