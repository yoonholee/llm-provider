"""Batch API implementations for OpenAI, Anthropic, and Gemini.

Each provider exposes submit(), status(), retrieve() with a common interface.
"""

import io
import json
import logging
import time

log = logging.getLogger(__name__)


# --- OpenAI ---


def openai_submit(
    client, model_id: str, prompts: list[str], system_prompt: str = "", **kwargs
) -> str:
    """Submit batch to OpenAI. Returns batch ID."""
    lines = []
    for i, prompt in enumerate(prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {"model": model_id, "messages": messages, **kwargs}
        lines.append(
            json.dumps(
                {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        )

    jsonl_bytes = ("\n".join(lines) + "\n").encode()
    file_obj = client.files.create(
        file=("batch.jsonl", io.BytesIO(jsonl_bytes)),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def openai_status(client, batch_id: str) -> dict:
    """Check OpenAI batch status."""
    batch = client.batches.retrieve(batch_id)
    counts = {}
    if batch.request_counts:
        counts = {
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed,
            "total": batch.request_counts.total,
        }
    return {"status": batch.status, "counts": counts}


def openai_retrieve(client, batch_id: str, n_prompts: int) -> list[list[str]] | None:
    """Retrieve OpenAI batch results. Returns None if not done."""
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"OpenAI batch {batch_id} {batch.status}")
        return None

    content = client.files.content(batch.output_file_id)
    results: dict[int, list[str]] = {}
    for line in content.text.strip().split("\n"):
        entry = json.loads(line)
        idx = int(entry["custom_id"].removeprefix("req-"))
        body = entry["response"]["body"]
        texts = [c["message"]["content"] or "" for c in body["choices"]]
        results[idx] = texts

    return [results.get(i, [""]) for i in range(n_prompts)]


# --- Anthropic ---


def anthropic_submit(
    client, model_id: str, prompts: list[str], system_prompt: str = "", **kwargs
) -> str:
    """Submit batch to Anthropic. Returns batch ID."""
    max_tokens = kwargs.pop("max_tokens", 4096)
    requests = []
    for i, prompt in enumerate(prompts):
        params = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            params["system"] = system_prompt
        params.update(kwargs)
        requests.append({"custom_id": f"req-{i}", "params": params})

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def anthropic_status(client, batch_id: str) -> dict:
    """Check Anthropic batch status."""
    batch = client.messages.batches.retrieve(batch_id)
    counts = {}
    if batch.request_counts:
        counts = {
            "processing": batch.request_counts.processing,
            "succeeded": batch.request_counts.succeeded,
            "errored": batch.request_counts.errored,
            "canceled": batch.request_counts.canceled,
            "expired": batch.request_counts.expired,
        }
    # Map Anthropic status to a common format
    if batch.processing_status == "ended":
        status = "completed"
    elif batch.processing_status == "canceling":
        status = "cancelling"
    else:
        status = batch.processing_status  # "in_progress"
    return {"status": status, "counts": counts}


def anthropic_retrieve(client, batch_id: str, n_prompts: int) -> list[list[str]] | None:
    """Retrieve Anthropic batch results. Returns None if not done."""
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        return None

    results: dict[int, list[str]] = {}
    for entry in client.messages.batches.results(batch_id):
        idx = int(entry.custom_id.removeprefix("req-"))
        if entry.result.type == "succeeded":
            # Extract text from content blocks
            texts = []
            for block in entry.result.message.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            results[idx] = texts if texts else [""]
        else:
            results[idx] = [""]

    return [results.get(i, [""]) for i in range(n_prompts)]


# --- Gemini ---


def gemini_submit(
    client, model_id: str, prompts: list[str], system_prompt: str = "", **kwargs
) -> str:
    """Submit batch to Gemini. Returns batch job name."""
    import google.genai.types as types

    thinking_config = kwargs.pop("thinking_config", None)
    if thinking_config is None:
        thinking_config = types.ThinkingConfig(thinking_budget=0)
    max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop("max_output_tokens", None)
    temperature = kwargs.pop("temperature", None)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        thinking_config=thinking_config,
        **({"max_output_tokens": max_tokens} if max_tokens is not None else {}),
        **({"temperature": temperature} if temperature is not None else {}),
    )

    requests = []
    for i, prompt in enumerate(prompts):
        requests.append(
            types.InlinedRequest(
                contents=prompt,
                config=config,
                metadata={"id": f"req-{i}"},
            )
        )

    batch_job = client.batches.create(model=model_id, src=requests)
    return batch_job.name


def gemini_status(client, batch_name: str) -> dict:
    """Check Gemini batch status."""
    batch_job = client.batches.get(name=batch_name)
    state = str(batch_job.state) if batch_job.state else "unknown"
    counts = {}
    if batch_job.completion_stats:
        counts = {
            "successful": batch_job.completion_stats.successful_count or 0,
            "failed": batch_job.completion_stats.failed_count or 0,
            "incomplete": batch_job.completion_stats.incomplete_count or 0,
        }
    # Map Gemini state to common format
    status_map = {
        "JobState.JOB_STATE_SUCCEEDED": "completed",
        "JobState.JOB_STATE_FAILED": "failed",
        "JobState.JOB_STATE_CANCELLED": "cancelled",
        "JobState.JOB_STATE_EXPIRED": "expired",
    }
    status = status_map.get(state, "in_progress")
    return {"status": status, "counts": counts}


def gemini_retrieve(client, batch_name: str, n_prompts: int) -> list[list[str]] | None:
    """Retrieve Gemini batch results. Returns None if not done."""
    batch_job = client.batches.get(name=batch_name)
    state = str(batch_job.state) if batch_job.state else ""
    if "SUCCEEDED" not in state and "PARTIALLY_SUCCEEDED" not in state:
        if "FAILED" in state or "CANCELLED" in state or "EXPIRED" in state:
            raise RuntimeError(f"Gemini batch {batch_name} {state}")
        return None

    results: dict[int, list[str]] = {}
    if batch_job.dest and batch_job.dest.inlined_responses:
        for resp in batch_job.dest.inlined_responses:
            idx = -1
            if resp.metadata and "id" in resp.metadata:
                idx = int(resp.metadata["id"].removeprefix("req-"))
            if resp.response and resp.response.text:
                results[idx] = [resp.response.text]
            else:
                results[idx] = [""]

    return [results.get(i, [""]) for i in range(n_prompts)]


# --- Polling helper ---


def poll_until_done(status_fn, poll_interval: float = 60, timeout: float = 86400):
    """Poll status_fn() until terminal state. Returns final status dict."""
    t0 = time.monotonic()
    while True:
        status = status_fn()
        s = status["status"]
        if s in ("completed", "failed", "cancelled", "expired"):
            return status
        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            raise TimeoutError(f"Batch not done after {timeout}s, last status: {s}")
        counts_str = str(status.get("counts", ""))
        log.info("Batch %s (%.0fs elapsed) %s", s, elapsed, counts_str)
        time.sleep(poll_interval)
