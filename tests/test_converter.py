import pytest
import uuid
from langchain_core.messages import AIMessageChunk
from langchain_ai_sdk_adapter.converter import to_ui_message_stream


async def chunks_to_async(chunks):
    """Convert a list to an async generator."""
    for chunk in chunks:
        yield chunk


def test_to_ui_message_stream_text_chunk():
    """Test that a single text chunk produces correct SSE line."""
    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Hello"}])
    ]

    # Consume the generator
    result = []
    async def consume():
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

    import asyncio
    asyncio.get_event_loop().run_until_complete(consume())

    assert len(result) == 2  # 1 content + 1 [DONE]
    assert result[0].startswith("data: ")
    assert '"role": "assistant"' in result[0]
    assert '"content": "Hello"' in result[0]
    assert result[0].endswith("\n\n")
    assert result[1] == "data: [DONE]\n\n"


def test_to_ui_message_stream_multiple_chunks():
    """Test multiple text chunks accumulate correctly."""
    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Hello"}]),
        AIMessageChunk(content=[{"type": "text", "text": " world"}]),
    ]

    result = []
    async def consume():
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

    import asyncio
    asyncio.get_event_loop().run_until_complete(consume())

    # Each chunk emits its own SSE line + 1 [DONE]
    assert len(result) == 3
    assert '"content": "Hello"' in result[0]
    assert '"content": " world"' in result[1]


def test_to_ui_message_stream_done():
    """Test that [DONE] is emitted at the end."""
    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Done"}])
    ]

    result = []
    async def consume():
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

    import asyncio
    asyncio.get_event_loop().run_until_complete(consume())

    assert result[-1] == "data: [DONE]\n\n"


def test_create_ui_message_stream_response_returns_streaming_response():
    """Test that create_ui_message_stream_response returns a StreamingResponse."""
    from starlette.responses import StreamingResponse
    from langchain_ai_sdk_adapter.converter import create_ui_message_stream_response

    chunks = [
        AIMessageChunk(content=[{"type": "text", "text": "Hello"}])
    ]

    response = create_ui_message_stream_response(
        to_ui_message_stream(chunks_to_async(chunks))
    )

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
