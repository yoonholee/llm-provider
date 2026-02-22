"""Thin LLM wrapper with disk caching and async batching.

Usage:
    from llm_provider import LLM
    llm = LLM("gpt-4.1-nano")
    results = llm.generate(["What is 2+2?", "Name a color."])
"""

from llm_provider.pricing import PRICES, cost
from llm_provider.provider import LLM, multi_chat, multi_generate

__all__ = ["LLM", "PRICES", "cost", "multi_generate", "multi_chat"]
