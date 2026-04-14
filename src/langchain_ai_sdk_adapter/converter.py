"""Convert LangChain streams to AI SDK SSE format."""
import json
import uuid
from typing import AsyncGenerator, TYPE_CHECKING

from langchain_core.messages import AIMessageChunk

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse


def _generate_message_id() -> str:
    """Generate a unique message ID matching AI SDK format."""
    return f"msg_{uuid.uuid4().hex[:24]}"


def _chunk_to_content_text(chunk: AIMessageChunk) -> str:
    """Extract text content from an AIMessageChunk.

    Handles both list content blocks and string content.
    """
    content = getattr(chunk, "content", None)

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Extract text from all text-type blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    return ""


async def to_ui_message_stream(
    chunks,
) -> AsyncGenerator[str, None]:
    """Convert LangChain AIMessageChunk stream to AI SDK SSE format.

    Args:
        chunks: An async iterable of AIMessageChunk objects (from
            model.stream() or graph.astream_events() with on_chat_model_stream events).

    Yields:
        SSE-formatted strings in AI SDK useChat format:
        - data: {"id":"msg_xxx","role":"assistant","content":"..."}\n\n
        - data: [DONE]\n\n

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic
        from langchain_ai_sdk_adapter import to_ui_message_stream

        model = ChatAnthropic(model="claude-3-5-sonnet")
        stream = await model.astream([HumanMessage(content="Hi")])

        async for line in to_ui_message_stream(stream):
            print(line, end="")
        ```
    """
    msg_id = _generate_message_id()
    role = "assistant"

    async for chunk in chunks:
        if not isinstance(chunk, AIMessageChunk):
            continue

        text = _chunk_to_content_text(chunk)
        if not text:
            continue

        data = {
            "id": msg_id,
            "role": role,
            "content": text,
        }
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


def create_ui_message_stream_response(stream) -> "StreamingResponse":
    """Create a Starlette StreamingResponse from a to_ui_message_stream generator.

    Requires starlette or fastapi to be installed.

    Args:
        stream: Async generator from to_ui_message_stream()

    Returns:
        StreamingResponse with text/event-stream media type

    Raises:
        ImportError: If starlette is not installed
    """
    try:
        from starlette.responses import StreamingResponse
    except ImportError:
        raise ImportError(
            "starlette is required for create_ui_message_stream_response. "
            "Install with: pip install langchain-ai-sdk-adapter[fastapi]"
        )

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
