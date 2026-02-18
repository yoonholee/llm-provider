"""Throughput benchmark for llm_provider optimizations.

Usage:
  uv run python bench.py [--provider openai|gemini|together|local|all] [--runs N]
  uv run python bench.py --no-cache   # bypass disk cache (always hit API)
  uv run python bench.py --provider local --local-url http://10.0.0.1:8080/v1 --local-model Qwen/Qwen3-8B

Measures:
- Throughput (output tok/s) for batch of 20 short prompts
- Latency (wall clock time)
- Reports median across runs
"""

import argparse
import os
import statistics
import time

# Suppress litellm noise before any imports
os.environ.setdefault("LITELLM_LOG", "ERROR")

from llm_provider import LLM


MODELS = {
    "openai": "gpt-4.1-nano",
    "gemini": "gemini/gemini-3-flash-preview",
    "together": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

# Mixed complexity: short answers, medium explanations, longer analysis
PROMPT_TEMPLATES = [
    # Short (1-2 sentences)
    "What is the capital of {topic}? One sentence.",
    "Define {topic} in one sentence.",
    # Medium (2-4 sentences)
    "Explain {topic} in exactly 3 sentences.",
    "What are two key facts about {topic}? Be concise.",
    "Compare {topic_a} and {topic_b} in 2 sentences.",
    # Longer (paragraph)
    "Write a paragraph explaining how {topic} works and why it matters.",
    "Describe {topic} as if explaining to a college student. One paragraph.",
]

TOPICS = [
    "quantum computing",
    "photosynthesis",
    "black holes",
    "neural networks",
    "plate tectonics",
    "DNA replication",
    "game theory",
    "ocean currents",
    "machine learning",
    "nuclear fusion",
    "evolution",
    "cryptography",
    "climate change",
    "antibiotics",
    "superconductors",
    "dark matter",
    "genetics",
    "robotics",
    "volcanoes",
    "relativity",
]


def make_prompts(run_idx: int, n: int = 20) -> list[str]:
    """Generate unique prompts per run with varied complexity."""
    prompts = []
    for i in range(n):
        tmpl = PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)]
        topic = TOPICS[(i + run_idx) % len(TOPICS)]
        topic_b = TOPICS[(i + run_idx + 7) % len(TOPICS)]
        p = tmpl.format(topic=topic, topic_a=topic, topic_b=topic_b)
        prompts.append(f"(id={run_idx}.{i}) {p}")
    return prompts


def disable_cache():
    """Replace disk caches with no-op so every call hits the API."""
    import llm_provider._cache as cache_mod
    import llm_provider.providers.openai_api as oai_mod
    import llm_provider.providers.gemini as gem_mod

    class NoOpCache:
        def get(self, key):
            return None

        def set(self, key, value):
            pass

    noop = NoOpCache()
    cache_mod.direct_cache = noop
    oai_mod.direct_cache = noop
    gem_mod.direct_cache = noop

    # Also disable litellm caching
    import litellm

    litellm.cache = None


def bench_one(model: str, run_idx: int) -> dict:
    """Single benchmark run. Returns {elapsed, output_tokens, throughput}."""
    prompts = make_prompts(run_idx)
    llm = LLM(model, max_concurrent=32)

    t0 = time.monotonic()
    llm.generate(prompts, silent=True)
    elapsed = time.monotonic() - t0

    out_tok = llm.total_output_tokens
    tps = out_tok / elapsed if elapsed > 0 else 0

    return {
        "run": run_idx,
        "elapsed": elapsed,
        "output_tokens": out_tok,
        "throughput": tps,
    }


def bench_provider(provider: str, model: str, n_runs: int):
    print(f"\n{'=' * 60}")
    print(f"Provider: {provider} | Model: {model} | Runs: {n_runs}")
    print(f"{'=' * 60}")

    results = []
    for i in range(n_runs):
        r = bench_one(model, i + 1000)
        print(
            f"  Run {i + 1}: {r['elapsed']:.2f}s | {r['output_tokens']} tok | {r['throughput']:.0f} tok/s"
        )
        results.append(r)

    throughputs = [r["throughput"] for r in results]
    elapsed_vals = [r["elapsed"] for r in results]

    print(f"\n  Median throughput: {statistics.median(throughputs):.0f} tok/s")
    print(f"  Median latency:   {statistics.median(elapsed_vals):.2f}s")
    if len(throughputs) > 1:
        print(f"  Stdev throughput: {statistics.stdev(throughputs):.0f} tok/s")

    return {
        "provider": provider,
        "model": model,
        "median_tps": statistics.median(throughputs),
        "median_latency": statistics.median(elapsed_vals),
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gemini", "together", "local", "all"],
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk cache (always hit API)",
    )
    parser.add_argument(
        "--local-url",
        default="http://localhost:30000/v1",
        help="Base URL for local provider (default: http://localhost:30000/v1)",
    )
    parser.add_argument(
        "--local-model",
        default="Qwen/Qwen3-4B",
        help="Model name for local provider (default: Qwen/Qwen3-4B)",
    )
    args = parser.parse_args()

    if args.no_cache:
        disable_cache()

    # Set local URL via env var so the provider picks it up
    os.environ["LOCAL_BASE_URL"] = args.local_url

    models = dict(MODELS)
    models["local"] = f"local/{args.local_model}"

    if args.provider == "all":
        providers = list(MODELS.keys())  # exclude local from "all"
    else:
        providers = [args.provider]

    all_results = {}
    for p in providers:
        all_results[p] = bench_provider(p, models[p], args.runs)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for p, r in all_results.items():
        print(f"  {p:12s}: {r['median_tps']:.0f} tok/s | {r['median_latency']:.2f}s")


if __name__ == "__main__":
    main()
