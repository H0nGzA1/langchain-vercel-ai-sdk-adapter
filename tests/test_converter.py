"""Tests for AI SDK 5.0 SSE format output."""
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_ai_sdk_adapter.converter import to_ui_message_stream, create_ui_message_stream_response


async def chunks_to_async(chunks):
    for chunk in chunks:
        yield chunk


def parse_sse_events(lines):
    """Parse SSE lines into list of data dicts."""
    events = []
    for line in lines:
        if line.startswith("data: "):
            import json
            data = json.loads(line[6:])
            events.append(data)
    return events


class TestToUiMessageStreamSingleChunk:
    """Test that a single text chunk produces correct SSE event sequence."""

    @pytest.mark.asyncio
    async def test_single_chunk_emits_five_events(self):
        """One AIMessageChunk with text 'Hello' should produce: start -> text-start -> text-delta -> text-end -> finish."""
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hello"}])
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        assert len(result) == 5, f"Expected 5 events, got {len(result)}: {result}"
        assert result[0].startswith("data: ")
        assert result[4].startswith("data: ")

    @pytest.mark.asyncio
    async def test_event_sequence_is_start_textstart_textdelta_textend_finish(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hi"}])
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        events = parse_sse_events(result)
        types = [e.get("type") for e in events]
        assert types == ["start", "text-start", "text-delta", "text-end", "finish"]

    @pytest.mark.asyncio
    async def test_text_delta_contains_correct_text(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hello"}])
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        events = parse_sse_events(result)
        text_delta = next(e for e in events if e.get("type") == "text-delta")
        assert text_delta.get("delta") == "Hello"

    @pytest.mark.asyncio
    async def test_text_start_has_id(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hi"}])
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        events = parse_sse_events(result)
        text_start = next(e for e in events if e.get("type") == "text-start")
        assert "id" in text_start
        assert text_start["id"].startswith("text_")

    @pytest.mark.asyncio
    async def test_start_has_message_id(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hi"}])
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        events = parse_sse_events(result)
        start = next(e for e in events if e.get("type") == "start")
        assert "messageId" in start
        assert start["messageId"].startswith("msg_")

    @pytest.mark.asyncio
    async def test_finish_has_finish_reason_and_usage(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hi"}])
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        events = parse_sse_events(result)
        finish = next(e for e in events if e.get("type") == "finish")
        assert "finishReason" in finish
        assert finish["finishReason"] == "stop"
        assert "usage" in finish

    @pytest.mark.asyncio
    async def test_no_done_sentinel(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Done"}])
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        done_lines = [l for l in result if "DONE" in l]
        assert len(done_lines) == 0, f"Found [DONE] in output: {done_lines}"


class TestToUiMessageStreamMultipleChunks:
    """Test multiple AIMessageChunk objects accumulate into multiple text-delta events."""

    @pytest.mark.asyncio
    async def test_multiple_chunks_produce_multiple_text_deltas(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hello"}]),
            AIMessageChunk(content=[{"type": "text", "text": " world"}]),
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        events = parse_sse_events(result)
        text_deltas = [e for e in events if e.get("type") == "text-delta"]
        assert len(text_deltas) == 2
        assert text_deltas[0].get("delta") == "Hello"
        assert text_deltas[1].get("delta") == " world"
        assert text_deltas[0]["id"] == text_deltas[1]["id"], "text-delta id should be stable across chunks"

    @pytest.mark.asyncio
    async def test_event_count_with_multiple_chunks(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "A"}]),
            AIMessageChunk(content=[{"type": "text", "text": "B"}]),
            AIMessageChunk(content=[{"type": "text", "text": "C"}]),
        ]
        result = []
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

        events = parse_sse_events(result)
        types = [e.get("type") for e in events]
        assert types == ["start", "text-start", "text-delta", "text-delta", "text-delta", "text-end", "finish"]


class TestCreateUiMessageStreamResponse:
    """Test create_ui_message_stream_response()."""

    def test_response_has_v1_header(self):
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hi"}])
        ]
        response = create_ui_message_stream_response(
            to_ui_message_stream(chunks_to_async(chunks))
        )
        assert response.media_type == "text/event-stream"
        assert response.headers.get("x-vercel-ai-ui-message-stream") == "v1"

    def test_returns_streaming_response(self):
        from starlette.responses import StreamingResponse
        chunks = [
            AIMessageChunk(content=[{"type": "text", "text": "Hi"}])
        ]
        response = create_ui_message_stream_response(
            to_ui_message_stream(chunks_to_async(chunks))
        )
        assert isinstance(response, StreamingResponse)
