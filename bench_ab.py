"""A/B benchmark: direct SDK vs litellm for the same prompts.

Measures throughput (tok/s) and TTFT for each provider path.
"""

import asyncio
import os
import statistics
import time

os.environ.setdefault("LITELLM_LOG", "ERROR")

TOPICS = [
    "quantum computing", "photosynthesis", "black holes", "neural networks",
    "plate tectonics", "DNA replication", "game theory", "ocean currents",
    "machine learning", "nuclear fusion", "evolution", "cryptography",
    "climate change", "antibiotics", "superconductors", "dark matter",
    "genetics", "robotics", "volcanoes", "relativity",
]

TEMPLATES = [
    "What is the capital of {topic}? One sentence.",
    "Define {topic} in one sentence.",
    "Explain {topic} in exactly 3 sentences.",
    "What are two key facts about {topic}? Be concise.",
    "Write a paragraph explaining how {topic} works and why it matters.",
    "Describe {topic} as if explaining to a college student. One paragraph.",
]


def make_prompts(run_idx: int, n: int = 20) -> list[str]:
    prompts = []
    for i in range(n):
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        topic = TOPICS[(i + run_idx) % len(TOPICS)]
        prompts.append(f"(id={run_idx}.{i}) {tmpl.format(topic=topic)}")
    return prompts


def bench_direct(model: str, run_idx: int) -> dict:
    """Benchmark using direct SDK (llm_provider)."""
    from llm_provider import LLM

    # Disable cache
    import llm_provider._cache as cm
    class NoOp:
        def get(self, k): return None
        def set(self, k, v): pass
    cm.direct_cache = NoOp()

    prompts = make_prompts(run_idx)
    llm = LLM(model, max_concurrent=32)
    t0 = time.monotonic()
    llm.generate(prompts, silent=True)
    elapsed = time.monotonic() - t0
    tps = llm.total_output_tokens / elapsed if elapsed > 0 else 0
    return {"elapsed": elapsed, "output_tokens": llm.total_output_tokens, "throughput": tps}


def bench_litellm(model: str, run_idx: int) -> dict:
    """Benchmark using litellm.acompletion directly."""
    import litellm
    litellm.suppress_debug_info = True

    prompts = make_prompts(run_idx)
    sem = asyncio.Semaphore(32)

    async def call_one(prompt):
        messages = [{"role": "user", "content": prompt}]
        async with sem:
            resp = await litellm.acompletion(model=model, messages=messages, num_retries=2)
        return resp

    async def run_all():
        tasks = [call_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    t0 = time.monotonic()
    results = asyncio.run(run_all())
    elapsed = time.monotonic() - t0

    total_out = sum(r.usage.completion_tokens or 0 for r in results if r.usage)
    tps = total_out / elapsed if elapsed > 0 else 0
    return {"elapsed": elapsed, "output_tokens": total_out, "throughput": tps}


def run_ab(label: str, direct_model: str, litellm_model: str, n_runs: int = 5):
    print(f"\n{'='*60}")
    print(f"{label}: direct={direct_model} vs litellm={litellm_model}")
    print(f"{'='*60}")

    direct_results = []
    litellm_results = []

    for i in range(n_runs):
        # Alternate to reduce time-of-day bias
        d = bench_direct(direct_model, 2000 + i * 2)
        l = bench_litellm(litellm_model, 2000 + i * 2 + 1)
        print(f"  Run {i+1}: direct={d['throughput']:.0f} tok/s  litellm={l['throughput']:.0f} tok/s")
        direct_results.append(d)
        litellm_results.append(l)

    d_med = statistics.median([r["throughput"] for r in direct_results])
    l_med = statistics.median([r["throughput"] for r in litellm_results])
    ratio = d_med / l_med if l_med > 0 else float("inf")

    print(f"\n  Direct median:  {d_med:.0f} tok/s")
    print(f"  Litellm median: {l_med:.0f} tok/s")
    print(f"  Speedup:        {ratio:.2f}x")
    return {"direct": d_med, "litellm": l_med, "ratio": ratio}


if __name__ == "__main__":
    results = {}
    results["OpenAI"] = run_ab("OpenAI", "gpt-4.1-nano", "gpt-4.1-nano")
    results["Gemini"] = run_ab("Gemini", "gemini/gemini-3-flash-preview", "gemini/gemini-3-flash-preview")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:12s}: direct={r['direct']:.0f}  litellm={r['litellm']:.0f}  ({r['ratio']:.2f}x)")
