"""Integration tests using actual LangChain components."""
import pytest
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_ai_sdk_adapter import to_base_messages, to_ui_message_stream


async def chunks_to_async(chunks):
    """Convert a list to an async generator."""
    for chunk in chunks:
        yield chunk


def test_to_base_messages_with_real_langchain_types():
    """Test to_base_messages produces LangChain message types."""
    ui_messages = [
        {"id": "1", "role": "system", "content": "You are helpful"},
        {"id": "2", "role": "user", "content": "Hello"},
        {"id": "3", "role": "assistant", "content": "Hi there!"},
    ]

    result = to_base_messages(ui_messages)

    assert len(result) == 3
    assert result[0].content == "You are helpful"
    assert result[1].content == "Hello"
    assert result[2].content == "Hi there!"


def test_to_ui_message_stream_integration():
    """Test SSE conversion with AIMessageChunk stream."""
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

    # 2 content lines + 1 [DONE]
    assert len(result) == 3
    assert '"content": "Hello"' in result[0]
    assert '"content": " world"' in result[1]
    assert result[2] == "data: [DONE]\n\n"


def test_to_ui_message_stream_skips_thinking_blocks():
    """Test that thinking blocks are skipped (only text content is emitted)."""
    chunks = [
        AIMessageChunk(content=[{"type": "thinking", "thinking": "Let me think..."}]),
        AIMessageChunk(content=[{"type": "text", "text": "Hello"}]),
    ]

    result = []
    async def consume():
        async for line in to_ui_message_stream(chunks_to_async(chunks)):
            result.append(line)

    import asyncio
    asyncio.get_event_loop().run_until_complete(consume())

    # Only 1 content line (text) + 1 [DONE]
    assert len(result) == 2
    assert '"content": "Hello"' in result[0]
    assert "thinking" not in result[0]
