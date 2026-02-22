"""Tests for batch API (no API keys needed)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_provider.provider import LLM
from llm_provider.providers import _batch


# --- Batch ID encode/decode ---


class TestBatchId:
    def test_roundtrip(self):
        encoded = LLM._encode_batch_id("openai", 5, "batch_abc123")
        provider, n, raw_id = LLM._decode_batch_id(encoded)
        assert provider == "openai"
        assert n == 5
        assert raw_id == "batch_abc123"

    def test_roundtrip_with_colons_in_id(self):
        """Gemini batch names can contain colons."""
        encoded = LLM._encode_batch_id("gemini", 3, "batches/123:456")
        provider, n, raw_id = LLM._decode_batch_id(encoded)
        assert provider == "gemini"
        assert n == 3
        assert raw_id == "batches/123:456"

    def test_single_prompt(self):
        encoded = LLM._encode_batch_id("anthropic", 1, "msgbatch_001")
        provider, n, raw_id = LLM._decode_batch_id(encoded)
        assert n == 1


# --- OpenAI batch ---


class TestOpenAIBatch:
    def test_submit_jsonl_format(self):
        """openai_submit should generate correct JSONL and call files.create + batches.create."""
        client = MagicMock()
        file_obj = MagicMock()
        file_obj.id = "file-abc"
        client.files.create.return_value = file_obj
        batch_obj = MagicMock()
        batch_obj.id = "batch-123"
        client.batches.create.return_value = batch_obj

        result = _batch.openai_submit(
            client, "gpt-4.1-mini", ["Hello", "World"], "Be helpful."
        )
        assert result == "batch-123"

        # Verify JSONL content
        file_call = client.files.create.call_args
        name, buf = file_call.kwargs["file"]
        assert name == "batch.jsonl"
        content = buf.read().decode()
        lines = content.strip().split("\n")
        assert len(lines) == 2

        line0 = json.loads(lines[0])
        assert line0["custom_id"] == "req-0"
        assert line0["method"] == "POST"
        assert line0["url"] == "/v1/chat/completions"
        assert line0["body"]["model"] == "gpt-4.1-mini"
        assert line0["body"]["messages"][0]["role"] == "system"
        assert line0["body"]["messages"][0]["content"] == "Be helpful."
        assert line0["body"]["messages"][1]["role"] == "user"
        assert line0["body"]["messages"][1]["content"] == "Hello"

        line1 = json.loads(lines[1])
        assert line1["custom_id"] == "req-1"
        assert line1["body"]["messages"][-1]["content"] == "World"

    def test_submit_no_system_prompt(self):
        """Without system_prompt, JSONL should have only user message."""
        client = MagicMock()
        client.files.create.return_value = MagicMock(id="file-x")
        client.batches.create.return_value = MagicMock(id="batch-x")

        _batch.openai_submit(client, "gpt-4.1-nano", ["Hi"])
        file_call = client.files.create.call_args
        _, buf = file_call.kwargs["file"]
        line = json.loads(buf.read().decode().strip())
        assert len(line["body"]["messages"]) == 1
        assert line["body"]["messages"][0]["role"] == "user"

    def test_status(self):
        client = MagicMock()
        batch = MagicMock()
        batch.status = "in_progress"
        batch.request_counts.completed = 3
        batch.request_counts.failed = 0
        batch.request_counts.total = 10
        client.batches.retrieve.return_value = batch

        result = _batch.openai_status(client, "batch-123")
        assert result["status"] == "in_progress"
        assert result["counts"]["completed"] == 3
        assert result["counts"]["total"] == 10

    def test_retrieve_reorders(self):
        """Results may come back in arbitrary order; retrieve must reorder by index."""
        client = MagicMock()
        batch = MagicMock()
        batch.status = "completed"
        batch.output_file_id = "file-out"
        client.batches.retrieve.return_value = batch

        # Results in reverse order
        output_lines = [
            json.dumps(
                {
                    "id": "resp-2",
                    "custom_id": "req-2",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "third"}}]},
                    },
                }
            ),
            json.dumps(
                {
                    "id": "resp-0",
                    "custom_id": "req-0",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "first"}}]},
                    },
                }
            ),
            json.dumps(
                {
                    "id": "resp-1",
                    "custom_id": "req-1",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "second"}}]},
                    },
                }
            ),
        ]
        content_obj = MagicMock()
        content_obj.text = "\n".join(output_lines)
        client.files.content.return_value = content_obj

        results = _batch.openai_retrieve(client, "batch-123", 3)
        assert results == [["first"], ["second"], ["third"]]

    def test_retrieve_not_done(self):
        client = MagicMock()
        batch = MagicMock()
        batch.status = "in_progress"
        client.batches.retrieve.return_value = batch

        result = _batch.openai_retrieve(client, "batch-123", 2)
        assert result is None

    def test_retrieve_failed_raises(self):
        client = MagicMock()
        batch = MagicMock()
        batch.status = "failed"
        client.batches.retrieve.return_value = batch

        with pytest.raises(RuntimeError, match="failed"):
            _batch.openai_retrieve(client, "batch-123", 2)


# --- Anthropic batch ---


class TestAnthropicBatch:
    def test_submit(self):
        client = MagicMock()
        batch = MagicMock()
        batch.id = "msgbatch_001"
        client.messages.batches.create.return_value = batch

        result = _batch.anthropic_submit(
            client, "claude-sonnet-4-20250514", ["Hi", "Bye"], "Be brief."
        )
        assert result == "msgbatch_001"

        call_kwargs = client.messages.batches.create.call_args.kwargs
        reqs = call_kwargs["requests"]
        assert len(reqs) == 2
        assert reqs[0]["custom_id"] == "req-0"
        assert reqs[0]["params"]["model"] == "claude-sonnet-4-20250514"
        assert reqs[0]["params"]["system"] == "Be brief."
        assert reqs[1]["custom_id"] == "req-1"

    def test_status_in_progress(self):
        client = MagicMock()
        batch = MagicMock()
        batch.processing_status = "in_progress"
        batch.request_counts.processing = 5
        batch.request_counts.succeeded = 3
        batch.request_counts.errored = 0
        batch.request_counts.canceled = 0
        batch.request_counts.expired = 0
        client.messages.batches.retrieve.return_value = batch

        result = _batch.anthropic_status(client, "msgbatch_001")
        assert result["status"] == "in_progress"

    def test_status_ended(self):
        client = MagicMock()
        batch = MagicMock()
        batch.processing_status = "ended"
        batch.request_counts.processing = 0
        batch.request_counts.succeeded = 5
        batch.request_counts.errored = 0
        batch.request_counts.canceled = 0
        batch.request_counts.expired = 0
        client.messages.batches.retrieve.return_value = batch

        result = _batch.anthropic_status(client, "msgbatch_001")
        assert result["status"] == "completed"

    def test_retrieve_reorders(self):
        client = MagicMock()
        batch = MagicMock()
        batch.processing_status = "ended"
        client.messages.batches.retrieve.return_value = batch

        # Mock results iterator (out of order)
        result1 = MagicMock()
        result1.custom_id = "req-1"
        result1.result.type = "succeeded"
        block1 = MagicMock()
        block1.text = "second"
        result1.result.message.content = [block1]

        result0 = MagicMock()
        result0.custom_id = "req-0"
        result0.result.type = "succeeded"
        block0 = MagicMock()
        block0.text = "first"
        result0.result.message.content = [block0]

        client.messages.batches.results.return_value = [result1, result0]

        results = _batch.anthropic_retrieve(client, "msgbatch_001", 2)
        assert results == [["first"], ["second"]]

    def test_retrieve_not_done(self):
        client = MagicMock()
        batch = MagicMock()
        batch.processing_status = "in_progress"
        client.messages.batches.retrieve.return_value = batch

        assert _batch.anthropic_retrieve(client, "msgbatch_001", 2) is None


# --- Gemini batch ---


class TestGeminiBatch:
    def test_submit(self):
        client = MagicMock()
        batch_job = MagicMock()
        batch_job.name = "batches/123"
        client.batches.create.return_value = batch_job

        result = _batch.gemini_submit(client, "gemini-2.0-flash", ["Hello", "World"])
        assert result == "batches/123"

        call_kwargs = client.batches.create.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert len(call_kwargs["src"]) == 2

    def test_status_succeeded(self):
        client = MagicMock()
        batch_job = MagicMock()
        batch_job.state = "JobState.JOB_STATE_SUCCEEDED"
        batch_job.completion_stats.successful_count = 5
        batch_job.completion_stats.failed_count = 0
        batch_job.completion_stats.incomplete_count = 0
        client.batches.get.return_value = batch_job

        result = _batch.gemini_status(client, "batches/123")
        assert result["status"] == "completed"

    def test_status_running(self):
        client = MagicMock()
        batch_job = MagicMock()
        batch_job.state = "JobState.JOB_STATE_RUNNING"
        batch_job.completion_stats = None
        client.batches.get.return_value = batch_job

        result = _batch.gemini_status(client, "batches/123")
        assert result["status"] == "in_progress"

    def test_retrieve_reorders(self):
        client = MagicMock()
        batch_job = MagicMock()
        batch_job.state = "JobState.JOB_STATE_SUCCEEDED"

        # Out-of-order inlined responses
        resp1 = MagicMock()
        resp1.metadata = {"id": "req-1"}
        resp1.response.text = "second"
        resp1.error = None

        resp0 = MagicMock()
        resp0.metadata = {"id": "req-0"}
        resp0.response.text = "first"
        resp0.error = None

        batch_job.dest.inlined_responses = [resp1, resp0]
        client.batches.get.return_value = batch_job

        results = _batch.gemini_retrieve(client, "batches/123", 2)
        assert results == [["first"], ["second"]]

    def test_retrieve_not_done(self):
        client = MagicMock()
        batch_job = MagicMock()
        batch_job.state = "JobState.JOB_STATE_RUNNING"
        client.batches.get.return_value = batch_job

        assert _batch.gemini_retrieve(client, "batches/123", 2) is None

    def test_retrieve_failed_raises(self):
        client = MagicMock()
        batch_job = MagicMock()
        batch_job.state = "JobState.JOB_STATE_FAILED"
        client.batches.get.return_value = batch_job

        with pytest.raises(RuntimeError, match="FAILED"):
            _batch.gemini_retrieve(client, "batches/123", 2)


# --- Polling ---


class TestPollUntilDone:
    def test_immediate_completion(self):
        fn = MagicMock(return_value={"status": "completed", "counts": {}})
        result = _batch.poll_until_done(fn, poll_interval=0.01)
        assert result["status"] == "completed"
        fn.assert_called_once()

    def test_polls_then_completes(self):
        fn = MagicMock(
            side_effect=[
                {"status": "in_progress", "counts": {}},
                {"status": "in_progress", "counts": {}},
                {"status": "completed", "counts": {}},
            ]
        )
        with patch("time.sleep"):
            result = _batch.poll_until_done(fn, poll_interval=0.01)
        assert result["status"] == "completed"
        assert fn.call_count == 3

    def test_timeout(self):
        fn = MagicMock(return_value={"status": "in_progress", "counts": {}})
        with patch("time.sleep"):
            with pytest.raises(TimeoutError):
                _batch.poll_until_done(fn, poll_interval=0.01, timeout=0)


# --- LLM.batch* integration ---


class TestLLMBatchMethods:
    def test_unsupported_model(self):
        llm = LLM("together_ai/meta-llama/Llama-3-8b")
        with pytest.raises(NotImplementedError, match="not supported"):
            llm.batch_submit("test")

    def test_openai_batch_submit(self):
        llm = LLM("gpt-4.1-nano")

        # Mock the sync client
        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-abc")
        mock_client.batches.create.return_value = MagicMock(id="batch-123")
        llm._sync_client = mock_client

        batch_id = llm.batch_submit(["Hello", "World"], system_prompt="Be helpful.")
        provider, n, raw_id = LLM._decode_batch_id(batch_id)
        assert provider == "openai"
        assert n == 2
        assert raw_id == "batch-123"

    def test_anthropic_batch_submit(self):
        llm = LLM("anthropic/claude-sonnet-4-20250514")

        mock_client = MagicMock()
        mock_client.messages.batches.create.return_value = MagicMock(id="msgbatch_001")

        with patch.object(llm, "_get_anthropic_client", return_value=mock_client):
            batch_id = llm.batch_submit(["Hello"])

        provider, n, raw_id = LLM._decode_batch_id(batch_id)
        assert provider == "anthropic"
        assert n == 1
        assert raw_id == "msgbatch_001"

    def test_string_prompt_wraps(self):
        """Single string prompt should work."""
        llm = LLM("gpt-4.1-nano")
        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-abc")
        mock_client.batches.create.return_value = MagicMock(id="batch-123")
        llm._sync_client = mock_client

        batch_id = llm.batch_submit("single prompt")
        _, n, _ = LLM._decode_batch_id(batch_id)
        assert n == 1
